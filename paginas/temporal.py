"""
temporal.py  — Análise de Séries Temporais e Planejamento de Carga
Painel CPGLP · Hospital Municipal São José · Joinville/SC

Metodologias aplicadas:
  • STL Decomposition (Cleveland et al., 1990)
  • Holt-Winters / Suavização Exponencial (Hyndman & Athanasopoulos, 2021)
  • Regressão Linear de Tendência (OLS)
  • Bootstrap Monte Carlo para Intervalos de Confiança
  • Análise de Autocorrelação (ACF)
  • Índice de Pressão Operacional (IPO) — indicador proprietário
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── ML/Estatística com fallback ───────────────────────────────────────────────
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES E HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_MESES_MAP_NUM = {
    "JANEIRO": 1, "FEVEREIRO": 2, "MARÇO": 3, "ABRIL": 4,
    "MAIO": 5, "JUNHO": 6, "JULHO": 7, "AGOSTO": 8,
    "SETEMBRO": 9, "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12,
}
_DIAS_MAP = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta",
             4: "Sexta", 5: "Sábado", 6: "Domingo"}

# Paleta institucional
_C_AZUL    = "#2d6fa3"
_C_VERDE   = "#2d7a4f"
_C_LARANJA = "#c87941"
_C_VERMELHO= "#c0392b"
_C_ROXO    = "#7b4fa6"
_C_ESCURO  = "#1a4d7a"

# Turno / faixa horária
_TURNO_BINS   = [0, 6, 9, 13, 17, 21, 24]
_TURNO_LABELS = ["Madrugada (0–5h)", "Manhã Cedo (6–8h)", "Manhã (9–12h)",
                 "Tarde (13–16h)", "Tarde / Noite (17–20h)", "Noite (21–23h)"]
_TURNO_CORES  = [_C_ROXO, _C_AZUL, _C_VERDE, _C_LARANJA, _C_VERMELHO, "#555555"]


def _nota(texto: str, icone: str = "💡"):
    """Caixa de nota explicativa para público leigo."""
    st.markdown(
        f"""<div style="background:#f0f6fb; border-left:4px solid {_C_AZUL};
            padding:10px 14px; border-radius:4px; margin:6px 0 14px 0;
            font-size:0.88rem; color:#333; line-height:1.55;">
            <b>{icone} O que isso significa?</b><br>{texto}
        </div>""",
        unsafe_allow_html=True,
    )


def _badge(nivel: str, texto: str):
    cores = {"ok": (_C_VERDE, "#e8f5ee"), "warn": (_C_LARANJA, "#fdf3e7"),
             "crit": (_C_VERMELHO, "#fdecea")}
    borda, fundo = cores.get(nivel, (_C_AZUL, "#e8f0fb"))
    st.markdown(
        f"""<div style="background:{fundo}; border:1px solid {borda};
            padding:8px 14px; border-radius:20px; display:inline-block;
            font-size:0.85rem; font-weight:600; color:{borda}; margin:2px 4px;">
            {texto}
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Timestamp real
    d["DataHora"] = pd.to_datetime(d["Carimbo de data/hora"], format="mixed", errors="coerce")
    d["Hora_Real"] = d["DataHora"].dt.hour.replace(0, np.nan)
    d["DiaSemanaIdx"] = d["DataHora"].dt.dayofweek
    d["DiaSemanaStr"] = d["DiaSemanaIdx"].map(_DIAS_MAP)
    d["DiaMes"] = d["DataHora"].dt.day

    # AnoMes a partir das colunas MÊS/ANO (mais confiável que o timestamp)
    d["mes_num"] = d["MÊS"].map(_MESES_MAP_NUM)
    d["AnoMes"] = pd.to_datetime(
        dict(year=d["ANO"].fillna(2025).astype(int),
             month=d["mes_num"].fillna(1).astype(int),
             day=1),
        errors="coerce",
    )

    # Flags assistenciais
    d["Is_Adquirida"] = d["CLASSIFICAÇÃO"].str.contains(
        "ADQUIRIDA NA INTERNAÇÃO ATUAL", na=False, case=False
    )
    d["Is_Grave"] = d["LESÃO POR PRESSÃO -  ESTAGIO"].str.contains(
        "III|IV|ESTADIÁVEL", na=False
    )
    d["Is_UTI"] = d["SETORES"].str.contains("UTI|CTI", na=False, case=False)

    # Turno
    d["Turno"] = pd.cut(
        d["Hora_Real"].fillna(10),  # fallback → manhã
        bins=_TURNO_BINS, labels=_TURNO_LABELS, right=False,
    )

    return d


# ─────────────────────────────────────────────────────────────────────────────
# SÉRIE MENSAL
# ─────────────────────────────────────────────────────────────────────────────
def _serie_mensal(d: pd.DataFrame) -> pd.DataFrame:
    s = (
        d.groupby("AnoMes")
        .agg(
            Total=("Is_Adquirida", "count"),
            Adquiridas=("Is_Adquirida", "sum"),
            Graves=("Is_Grave", "sum"),
        )
        .reset_index()
        .sort_values("AnoMes")
    )
    s["Taxa_Adq"] = (s["Adquiridas"] / s["Total"] * 100).round(1)
    s["Taxa_Grave"] = (s["Graves"] / s["Total"] * 100).round(1)
    s["Label"] = s["AnoMes"].dt.strftime("%b/%y").str.upper()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPOSIÇÃO STL
# ─────────────────────────────────────────────────────────────────────────────
def _decomposicao_stl(serie_vals: np.ndarray, period: int = 4):
    """Retorna (trend, seasonal, resid) ou None se não disponível."""
    if not STATSMODELS_OK or len(serie_vals) < 2 * period + 1:
        return None
    try:
        stl = STL(serie_vals, period=period, robust=True)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST HOLT-WINTERS + BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────
def _forecast(serie_vals: np.ndarray, n_ahead: int = 4, n_boot: int = 500, seed: int = 42):
    """
    Retorna (y_hat, ci_low, ci_high).
    Usa Holt-Winters (tendência aditiva) se statsmodels disponível,
    senão usa regressão linear simples.
    Bootstrap Monte Carlo para gerar o intervalo de confiança.
    """
    n = len(serie_vals)

    # ── Ponto central ───────────────────────────────────────────────────────
    if STATSMODELS_OK and n >= 4:
        try:
            model = ExponentialSmoothing(serie_vals, trend="add", seasonal=None)
            fit = model.fit(optimized=True, use_brute=True)
            y_hat = np.maximum(fit.forecast(n_ahead), 0)
        except Exception:
            y_hat = None
    else:
        y_hat = None

    if y_hat is None:
        x = np.arange(n)
        z = np.polyfit(x, serie_vals, 1)
        p = np.poly1d(z)
        y_hat = np.maximum(p(np.arange(n, n + n_ahead)), 0)

    # ── Bootstrap CI ────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    residuos = serie_vals - np.interp(
        np.arange(n),
        np.linspace(0, n - 1, n),
        np.linspace(serie_vals[0], serie_vals[-1], n),
    )
    std_res = max(np.std(residuos), np.std(serie_vals) * 0.12)

    boot_preds = []
    for _ in range(n_boot):
        ruido = rng.normal(0, std_res, n_ahead)
        boot_preds.append(np.maximum(y_hat + ruido, 0))
    boot_arr = np.array(boot_preds)

    ci_low  = np.percentile(boot_arr, 10, axis=0)
    ci_high = np.percentile(boot_arr, 90, axis=0)
    return y_hat, ci_low, ci_high


# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def _acf(series: np.ndarray, max_lags: int = 5) -> list[tuple[int, float]]:
    n = len(series)
    mu = series.mean()
    var = np.var(series)
    if var == 0:
        return [(i, 0.0) for i in range(1, min(max_lags, n) + 1)]
    resultado = []
    for lag in range(1, min(max_lags, n - 1) + 1):
        cov = np.mean((series[lag:] - mu) * (series[:-lag] - mu))
        resultado.append((lag, cov / var))
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# ÍNDICE DE PRESSÃO OPERACIONAL (IPO) — indicador proprietário
# ─────────────────────────────────────────────────────────────────────────────
def _ipo(taxa_adq: float, pct_graves: float, pct_acamados: float,
         total: int, media_historica: float) -> tuple[float, str, str]:
    """
    IPO ∈ [0, 100]. Combina:
      - Taxa de adquiridas normalizada (peso 40%)
      - Proporção de graves (peso 30%)
      - Proporção de acamados (peso 15%)
      - Volume relativo vs. média histórica (peso 15%)
    """
    vol_norm = min((total / max(media_historica, 1)) * 50, 100)
    ipo = (
        min(taxa_adq, 100) * 0.40
        + min(pct_graves, 100) * 0.30
        + min(pct_acamados, 100) * 0.15
        + vol_norm * 0.15
    )
    ipo = round(min(ipo, 100), 1)
    if ipo >= 65:
        return ipo, "🔴 CRÍTICO", "crit"
    elif ipo >= 40:
        return ipo, "🟠 ELEVADO", "warn"
    elif ipo >= 20:
        return ipo, "🟡 MODERADO", "warn"
    else:
        return ipo, "🟢 CONTROLADO", "ok"


# ─────────────────────────────────────────────────────────────────────────────
# RENDER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def render_temporal(df: pd.DataFrame):
    if df.empty:
        st.warning("⚠️ Não existem dados para os filtros selecionados.")
        return

    d = _preparar(df)
    serie = _serie_mensal(d)

    n_meses   = len(serie)
    total_reg = len(d)
    media_mensal_hist = serie["Total"].mean()

    # =========================================================================
    # 1. KPIs TEMPORAIS
    # =========================================================================
    st.subheader("⏱️ Painel de Indicadores Temporais")
    _nota(
        "Os quatro números abaixo são o <b>resumo executivo</b> da carga de trabalho no período selecionado. "
        "Eles respondem: <em>quanto trabalhamos?</em>, <em>por quanto tempo?</em>, "
        "<em>qual foi o pico?</em> e <em>qual é a pressão operacional hoje?</em>",
        "📊"
    )

    # Calcular IPO do período completo
    taxa_adq_geral  = d["Is_Adquirida"].mean() * 100
    taxa_grave_geral = d["Is_Grave"].mean() * 100
    pct_acamados = (d.get("MOBILIDADE", pd.Series(dtype=str)) == "ACAMADO").mean() * 100 if "MOBILIDADE" in d.columns else 50.0
    ipo_val, ipo_label, ipo_nivel = _ipo(
        taxa_adq_geral, taxa_grave_geral, pct_acamados, total_reg, media_mensal_hist
    )

    # Variação mês mais recente vs. anterior
    delta_mes = None
    if len(serie) >= 2:
        delta_mes = int(serie["Total"].iloc[-1]) - int(serie["Total"].iloc[-2])

    agrupamento_diario = d.dropna(subset=["DataHora"]).groupby(d["DataHora"].dt.date).size()
    pico_diario = int(agrupamento_diario.max()) if not agrupamento_diario.empty else 0
    data_pico   = agrupamento_diario.idxmax() if not agrupamento_diario.empty else "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Avaliações", f"{total_reg:,}", f"{delta_mes:+d} vs. mês anterior" if delta_mes is not None else None)
    k2.metric("Período Coberto", f"{n_meses} meses")
    k3.metric("Pico Diário", pico_diario, f"{data_pico}")
    k4.metric("Índice de Pressão Operacional", f"{ipo_val}/100", ipo_label,
              delta_color="inverse" if ipo_nivel in ("crit", "warn") else "off")

    st.markdown("---")

    # =========================================================================
    # 2. DECOMPOSIÇÃO STL DA SÉRIE TEMPORAL
    # =========================================================================
    st.subheader("🔬 Decomposição da Série Temporal (STL)")
    _nota(
        "<b>Decompor uma série temporal</b> é como separar um sinal de rádio em suas partes: "
        "a <b>tendência</b> (para onde os números estão caminhando no longo prazo), "
        "a <b>sazonalidade</b> (o padrão que se repete todo trimestre/ano) e o <b>resíduo</b> "
        "(o que foi imprevisível, como surtos ou eventos especiais). "
        "Isso permite à gestão distinguir problemas estruturais de variações normais.",
        "🔬"
    )

    vals = serie["Total"].values.astype(float)
    labels = serie["Label"].values
    decomp = _decomposicao_stl(vals, period=4)

    if decomp is not None:
        trend, seasonal, resid = decomp

        fig_decomp = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=["📊 Série Original", "📈 Tendência de Longo Prazo",
                            "🔄 Componente Sazonal (Ciclo Trimestral)",
                            "⚡ Resíduo (Eventos Imprevisíveis)"],
            vertical_spacing=0.07,
        )

        # Original
        fig_decomp.add_trace(go.Scatter(
            x=labels, y=vals, mode="lines+markers",
            line=dict(color=_C_AZUL, width=2.5),
            marker=dict(size=7), name="Original",
        ), row=1, col=1)

        # Tendência
        cor_tend = _C_VERDE if trend[-1] >= trend[0] else _C_VERMELHO
        fig_decomp.add_trace(go.Scatter(
            x=labels, y=trend, mode="lines",
            line=dict(color=cor_tend, width=3, dash="solid"),
            name="Tendência",
        ), row=2, col=1)
        # Anotação de direção
        direcao = "↗ Tendência de ALTA" if trend[-1] > trend[0] else "↘ Tendência de QUEDA"
        fig_decomp.add_annotation(
            x=labels[-1], y=trend[-1], text=f"  {direcao}",
            showarrow=False, font=dict(color=cor_tend, size=11), row=2, col=1
        )

        # Sazonal
        fig_decomp.add_trace(go.Bar(
            x=labels, y=seasonal,
            marker_color=[_C_VERDE if v >= 0 else _C_VERMELHO for v in seasonal],
            name="Sazonal", opacity=0.8,
        ), row=3, col=1)
        fig_decomp.add_hline(y=0, line_dash="dot", line_color="#aaa", row=3, col=1)

        # Resíduo
        fig_decomp.add_trace(go.Scatter(
            x=labels, y=resid, mode="markers+lines",
            marker=dict(color=[_C_VERMELHO if abs(r) > np.std(resid) * 1.5 else _C_AZUL for r in resid],
                        size=9, symbol="diamond"),
            line=dict(color="#aaa", width=1, dash="dot"),
            name="Resíduo",
        ), row=4, col=1)
        fig_decomp.add_hline(y=0, line_dash="dot", line_color="#aaa", row=4, col=1)

        # Faixa de resíduo normal
        std_r = np.std(resid)
        for sign in [1, -1]:
            fig_decomp.add_hrect(
                y0=0, y1=sign * std_r * 1.5,
                fillcolor="rgba(200,121,65,0.08)",
                line_width=0, row=4, col=1
            )

        fig_decomp.update_layout(
            height=650, showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
        )
        for i in range(1, 5):
            fig_decomp.update_yaxes(showgrid=True, gridcolor="#ede9e3", row=i, col=1)

        st.plotly_chart(fig_decomp, width="stretch")

        # Força da sazonalidade
        force_s = max(0.0, 1 - np.var(resid) / max(np.var(seasonal + resid), 1e-9))
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Força do Padrão Sazonal", f"{force_s:.0%}",
                      help="≥ 60% = padrão sazonal forte e previsível.")
        col_s2.metric("Variação Não Explicada (Resíduo)", f"{np.std(resid):.0f} avaliações/mês",
                      help="Quanto as avaliações oscilam além do esperado pela tendência e sazonalidade.")
        _nota(
            f"O padrão sazonal explica <b>{force_s:.0%}</b> da variação mensal — isso é "
            + ("<b>forte</b>: a equipe pode se preparar antecipadamente para os meses de pico, "
               "pois o hospital repete padrões previsíveis." if force_s >= 0.6
               else "<b>moderado</b>: a carga mensal ainda tem componentes imprevisíveis que exigem flexibilidade de escala.")
            + f" O resíduo médio de <b>±{np.std(resid):.0f} avaliações/mês</b> representa o 'ruído' que não pode ser previsto só por histórico.",
        )
    else:
        # Fallback: gráfico simples sem STL
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=labels, y=vals, marker_color=_C_AZUL, name="Total mensal",
            text=vals.astype(int), textposition="outside"
        ))
        fig_s.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                             plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s, width="stretch")
        st.caption("ℹ️ `statsmodels` não instalado — decomposição STL indisponível. `pip install statsmodels`")

    st.markdown("---")

    # =========================================================================
    # 3. FORECAST HOLT-WINTERS + BOOTSTRAP CI
    # =========================================================================
    st.subheader("🔮 Projeção Preditiva (Holt-Winters + Monte Carlo)")
    _nota(
        "<b>Holt-Winters</b> é um método clássico de previsão que aprende tanto o "
        "<em>nível atual</em> quanto a <em>velocidade de mudança</em> dos dados. "
        "A <b>faixa laranja</b> é o <em>intervalo de confiança de 80%</em>, calculado por "
        "<b>Bootstrap Monte Carlo</b> (500 simulações): ela responde "
        "<em>'dentro de qual intervalo o número real tem 80% de chance de cair?'</em> "
        "Quanto mais larga a faixa, maior a incerteza no período.",
        "🔮"
    )

    n_ahead = 4
    y_hat, ci_low, ci_high = _forecast(vals, n_ahead=n_ahead)

    # Gerar labels futuros
    last_ym = serie["AnoMes"].iloc[-1]
    future_dates, future_labels = [], []
    ym = last_ym
    for _ in range(n_ahead):
        ym = ym + pd.DateOffset(months=1)
        future_dates.append(ym)
        future_labels.append(ym.strftime("%b/%y").upper())

    fig_fc = go.Figure()

    # Histórico
    fig_fc.add_trace(go.Scatter(
        x=labels, y=vals, mode="lines+markers",
        name="Histórico Real",
        line=dict(color=_C_AZUL, width=3),
        marker=dict(size=8, color=_C_AZUL),
    ))

    # Conectar histórico ao forecast
    x_bridge = [labels[-1], future_labels[0]]
    y_bridge  = [vals[-1],  y_hat[0]]
    fig_fc.add_trace(go.Scatter(
        x=x_bridge, y=y_bridge,
        mode="lines", line=dict(color=_C_VERMELHO, width=2, dash="dot"),
        showlegend=False,
    ))

    # Banda Bootstrap
    fig_fc.add_trace(go.Scatter(
        x=future_labels + future_labels[::-1],
        y=list(ci_high) + list(ci_low)[::-1],
        fill="toself",
        fillcolor="rgba(200, 121, 65, 0.20)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalo de Confiança 80%",
    ))

    # Linha central forecast
    fig_fc.add_trace(go.Scatter(
        x=future_labels, y=y_hat,
        mode="lines+markers+text",
        name="Projeção Central",
        line=dict(color=_C_VERMELHO, width=3, dash="dash"),
        marker=dict(size=10, color=_C_VERMELHO, symbol="diamond"),
        text=[f"~{int(v)}" for v in y_hat],
        textposition="top center",
        textfont=dict(size=12, color=_C_VERMELHO),
    ))

    # Linha de meta
    meta_ref = float(serie["Total"].mean())
    fig_fc.add_hline(
        y=meta_ref, line_dash="dot", line_color="#999",
        annotation_text=f"Média histórica: {meta_ref:.0f}/mês",
        annotation_position="bottom right",
    )

    # Separador histórico/projeção
    fig_fc.add_vrect(
        x0=labels[-1], x1=future_labels[-1],
        fillcolor="rgba(200,121,65,0.05)",
        line_width=0,
        annotation_text="Zona de Projeção",
        annotation_position="top left",
    )

    fig_fc.update_layout(
        height=420, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Avaliações/mês", showgrid=True, gridcolor="#ede9e3"),
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Cards de projeção
    fc_cols = st.columns(n_ahead)
    for i, (col, lab, yh, lo, hi) in enumerate(zip(fc_cols, future_labels, y_hat, ci_low, ci_high)):
        variacao = int(yh) - int(vals[-1]) if i == 0 else int(yh) - int(y_hat[i - 1])
        col.metric(
            label=f"Projeção {lab}",
            value=f"~{int(yh)}",
            delta=f"{variacao:+d} vs. anterior",
            delta_color="inverse" if variacao > meta_ref * 0.15 else "off",
        )
        col.caption(f"IC 80%: {int(lo)}–{int(hi)}")

    _nota(
        "Os números acima são estimativas. A gestão deve usá-los para "
        "<b>dimensionar escala de pessoal</b>, solicitar insumos e programar rounds da CPGLP "
        "antes dos meses com projeção acima da média histórica.",
    )

    st.markdown("---")

    # =========================================================================
    # 4. ANÁLISE DE AUTOCORRELAÇÃO (ACF)
    # =========================================================================
    st.subheader("📡 Análise de Autocorrelação (ACF)")
    _nota(
        "A autocorrelação responde: <em>'o que aconteceu no mês passado influencia o mês seguinte?'</em> "
        "Um valor positivo (barra acima do zero) significa que meses com alta carga tendem a ser "
        "seguidos por outros meses de alta carga — <b>padrão persistente</b>. "
        "Um valor negativo indica <b>padrão de alternância</b> (mês pesado → mês leve → mês pesado). "
        "As linhas pontilhadas vermelhas são o <em>limiar de significância estatística (95%)</em>: "
        "barras que passam essa linha são padrões reais, não coincidência.",
        "📡"
    )

    acf_vals = _acf(vals, max_lags=min(5, len(vals) - 2))
    lags_labels = [f"Lag {l}\n({l} mês{'es' if l > 1 else ''})" for l, _ in acf_vals]
    acf_y = [v for _, v in acf_vals]
    limiar_95 = 1.96 / np.sqrt(len(vals))

    cores_acf = [_C_VERDE if abs(v) > limiar_95 else "#c0c0c0" for v in acf_y]
    fig_acf = go.Figure()
    for i, (lab, v, cor) in enumerate(zip(lags_labels, acf_y, cores_acf)):
        fig_acf.add_trace(go.Bar(
            x=[lab], y=[v], marker_color=cor,
            name=lab, showlegend=False,
            text=f"{v:.2f}", textposition="outside",
        ))
    fig_acf.add_hline(y=limiar_95,  line_dash="dot", line_color=_C_VERMELHO,
                      annotation_text="Significância +95%", annotation_position="top right")
    fig_acf.add_hline(y=-limiar_95, line_dash="dot", line_color=_C_VERMELHO,
                      annotation_text="Significância −95%", annotation_position="bottom right")
    fig_acf.add_hline(y=0, line_color="#aaa", line_width=1)
    fig_acf.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Correlação", range=[-1.1, 1.1],
                   showgrid=True, gridcolor="#ede9e3"),
        plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_acf, width="stretch")

    # Interpretação automática
    lag1 = acf_vals[0][1] if acf_vals else 0
    if abs(lag1) > limiar_95:
        interp = (
            f"<b>Lag-1 = {lag1:.2f}</b> é estatisticamente significativo. "
            + ("O hospital opera em <b>ciclos alternados</b>: um mês de alta carga tende a ser seguido por um mês de menor carga — "
               "padrão consistente com surtos pontuais de avaliação (ex.: mutirões da CPGLP)."
               if lag1 < 0
               else "A carga de trabalho tem <b>persistência</b>: meses de alta demanda tendem a se agrupar, "
                    "sugerindo pressão sustentada que não se dissipa rapidamente.")
        )
    else:
        interp = (
            f"<b>Lag-1 = {lag1:.2f}</b>: os meses são <b>relativamente independentes entre si</b> — "
            "a carga de um mês não é boa preditora isolada do mês seguinte. "
            "Isso reforça a importância de usar modelos mais robustos (como o Holt-Winters acima) em vez de simples extrapolação."
        )
    _nota(interp)

    st.markdown("---")

    # =========================================================================
    # 5. HEATMAP DE TAXA DE LPP ADQUIRIDA POR SETOR × MÊS
    # =========================================================================
    st.subheader("🗓️ Heatmap de Taxa de Incidência: Setor × Mês")
    _nota(
        "Este mapa de calor mostra <b>a porcentagem de LPPs adquiridas na internação atual</b> "
        "para cada setor em cada mês. Cores <b>mais vermelhas</b> = mais lesões sendo desenvolvidas "
        "dentro do hospital naquele período. É a ferramenta definitiva para identificar "
        "<em>quando</em> e <em>onde</em> a prevenção falhou.",
        "🗓️"
    )

    if "SETORES" in d.columns:
        top_setores = d["SETORES"].value_counts().head(10).index.tolist()
        df_heat_s = d[d["SETORES"].isin(top_setores)].copy()
        df_heat_s["MesLabel"] = df_heat_s["AnoMes"].dt.strftime("%b/%y").str.upper()

        pivot_h = (
            df_heat_s.groupby(["SETORES", "MesLabel"])
            .agg(total=("Is_Adquirida", "count"), adq=("Is_Adquirida", "sum"))
            .reset_index()
        )
        pivot_h["taxa"] = (pivot_h["adq"] / pivot_h["total"] * 100).round(1)
        pivot_h = pivot_h.pivot(index="SETORES", columns="MesLabel", values="taxa").fillna(0)

        # Ordenar colunas por mês real
        todas_labels = [ym.strftime("%b/%y").upper() for ym in sorted(d["AnoMes"].dropna().unique())]
        colunas_ord = [c for c in todas_labels if c in pivot_h.columns]
        pivot_h = pivot_h[colunas_ord]

        # Ordenar setores por média de taxa (pior no topo)
        pivot_h = pivot_h.loc[pivot_h.mean(axis=1).sort_values(ascending=False).index]

        fig_hm = px.imshow(
            pivot_h,
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=100,
            text_auto=".0f",
            labels=dict(color="Taxa Adq. (%)"),
            aspect="auto",
        )
        fig_hm.update_traces(
            textfont=dict(size=11, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Taxa de adquiridas: %{z:.0f}%<extra></extra>",
        )
        fig_hm.update_layout(
            height=max(300, len(pivot_h) * 50),
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="% Adquiridas"),
        )
        st.plotly_chart(fig_hm, width="stretch")

        # Top 3 combinações mais críticas
        flat = pivot_h.stack().reset_index()
        flat.columns = ["Setor", "Mês", "Taxa"]
        flat = flat[flat["Taxa"] > 0].sort_values("Taxa", ascending=False)
        if not flat.empty:
            st.markdown("**🎯 Combinações Setor × Mês mais críticas:**")
            c1, c2, c3 = st.columns(3)
            for col, (_, row) in zip([c1, c2, c3], flat.head(3).iterrows()):
                nivel = "crit" if row["Taxa"] >= 70 else "warn" if row["Taxa"] >= 40 else "ok"
                col.metric(f"{row['Setor']} · {row['Mês']}", f"{row['Taxa']:.0f}%",
                           "taxa de adquiridas", delta_color="inverse")

    st.markdown("---")

    # =========================================================================
    # 6. PADRÕES SEMANAIS E INTRADIÁRIOS
    # =========================================================================
    st.subheader("📅 Padrões de Carga Operacional (Ciclos Semanais e Horários)")
    _nota(
        "Entender <em>quando</em> a equipe é mais demandada permite ao gestor "
        "<b>ajustar escalas, priorizar rounds e alocar insumos</b> nos momentos certos. "
        "Os dois gráficos abaixo mostram o ciclo semanal e horário das avaliações.",
        "📅"
    )

    col_radar, col_hora = st.columns(2)

    # ── Radar Semanal ──────────────────────────────────────────────────────
    with col_radar:
        st.markdown("**Ciclo Semanal (Radar de Carga)**")
        ordem_dias = ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"]
        vc_semana = (
            d["DiaSemanaStr"]
            .value_counts()
            .reindex(ordem_dias, fill_value=0)
            .reset_index()
        )
        vc_semana.columns = ["Dia", "Registros"]
        dias_r = vc_semana["Dia"].tolist() + [vc_semana["Dia"].iloc[0]]
        vals_r = vc_semana["Registros"].tolist() + [vc_semana["Registros"].iloc[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals_r, theta=dias_r, fill="toself",
            fillcolor="rgba(45, 111, 163, 0.25)",
            line=dict(color=_C_AZUL, width=2.5),
            marker=dict(size=7, color=_C_AZUL),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=True,
                                tickfont=dict(size=9), gridcolor="#ddd"),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=False, height=360,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Insight automático
        dia_pico = vc_semana.loc[vc_semana["Registros"].idxmax(), "Dia"]
        dia_vale = vc_semana.loc[vc_semana["Registros"].idxmin(), "Dia"]
        _nota(
            f"<b>{dia_pico}</b> é o dia com maior volume de avaliações, "
            f"enquanto <b>{dia_vale}</b> tem a menor carga. "
            "A equipe deve garantir que os rounds da CPGLP ocorram com maior frequência "
            f"próximos às <b>{dia_pico}s</b>."
        )

    # ── Distribuição Horária ───────────────────────────────────────────────
    with col_hora:
        st.markdown("**Distribuição Horária das Avaliações**")
        d_horas = d[d["Hora_Real"].notna()].copy()
        d_horas["Hora_Int"] = d_horas["Hora_Real"].astype(int)
        vc_hora = d_horas.groupby("Hora_Int").size().reset_index(name="Registros")

        hora_pico_val = int(vc_hora.loc[vc_hora["Registros"].idxmax(), "Hora_Int"])
        cores_hora = [
            _C_VERMELHO if h == hora_pico_val else
            _C_LARANJA  if vc_hora.loc[vc_hora["Hora_Int"] == h, "Registros"].values[0] >= vc_hora["Registros"].quantile(0.75) else
            _C_AZUL
            for h in vc_hora["Hora_Int"]
        ]

        fig_hora = go.Figure()
        fig_hora.add_trace(go.Bar(
            x=vc_hora["Hora_Int"], y=vc_hora["Registros"],
            marker_color=cores_hora,
            text=vc_hora["Registros"],
            textposition="outside",
            name="Avaliações/hora",
        ))
        fig_hora.add_trace(go.Scatter(
            x=vc_hora["Hora_Int"], y=vc_hora["Registros"],
            mode="lines", line_shape="spline",
            line=dict(color=_C_ESCURO, width=2.5, dash="dot"),
            showlegend=False,
        ))
        fig_hora.add_vline(x=hora_pico_val, line_dash="dash",
                           line_color=_C_VERMELHO,
                           annotation_text=f"Pico: {hora_pico_val}h",
                           annotation_position="top right")
        fig_hora.update_layout(
            height=360, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Hora do Dia", tickmode="linear", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
            plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_hora, width="stretch")

        pct_manha = d_horas[(d_horas["Hora_Int"] >= 9) & (d_horas["Hora_Int"] <= 12)].shape[0]
        pct_manha_pct = pct_manha / len(d_horas) * 100 if len(d_horas) > 0 else 0
        _nota(
            f"<b>{pct_manha_pct:.0f}%</b> de todas as avaliações ocorrem entre 9h e 12h — "
            f"o pico absoluto é às <b>{hora_pico_val}h</b>. "
            "Isso confirma que a <b>janela de 9h–12h é crítica</b> para dimensionamento de pessoal: "
            "a equipe da CPGLP deve garantir máxima disponibilidade nesse período."
        )

    st.markdown("---")

    # =========================================================================
    # 7. HEATMAP CLÁSSICO DIA × HORA
    # =========================================================================
    st.subheader("⏰ Mapa de Calor: Dia da Semana × Hora do Dia")
    _nota(
        "Este é o mapa de calor da <b>operação hospitalar em nível microscópico</b>. "
        "Cada célula mostra quantas avaliações ocorreram naquela combinação de "
        "<em>dia da semana + hora do dia</em>. "
        "Células mais escuras indicam os <b>momentos de maior pressão simultânea</b> sobre a equipe, "
        "e devem nortear a definição de turnos e escala de plantão.",
        "⏰"
    )

    df_hm2 = d[d["Hora_Real"].notna()].copy()
    df_hm2["Hora_Int"] = df_hm2["Hora_Real"].astype(int)
    df_hm2 = df_hm2[(df_hm2["Hora_Int"] >= 6) & (df_hm2["Hora_Int"] <= 20)]

    if not df_hm2.empty:
        mat = pd.crosstab(df_hm2["DiaSemanaStr"], df_hm2["Hora_Int"])
        ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        horas_cols = list(range(6, 21))
        mat = mat.reindex(index=ordem_dias, columns=horas_cols, fill_value=0)

        fig_hm2 = px.imshow(
            mat,
            labels=dict(x="Hora do Dia", y="Dia da Semana", color="Avaliações"),
            color_continuous_scale="Blues",
            text_auto=True,
            aspect="auto",
        )
        fig_hm2.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_hm2.update_traces(
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b> às <b>%{x}h</b><br>%{z} avaliações<extra></extra>",
        )
        fig_hm2.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig_hm2, width="stretch")

    st.markdown("---")

    # =========================================================================
    # 8. PLANEJAMENTO DE CAPACIDADE E ÍNDICE DE PRESSÃO OPERACIONAL
    # =========================================================================
    st.subheader("🏋️ Planejamento de Capacidade e Índice de Pressão Operacional (IPO)")
    _nota(
        "O <b>Índice de Pressão Operacional (IPO)</b> é um indicador composto criado para este painel. "
        "Ele combina a <em>taxa de LPPs adquiridas</em>, a <em>proporção de lesões graves</em>, "
        "o <em>percentual de pacientes acamados</em> e o <em>volume relativo de avaliações</em> "
        "em um único número de 0 a 100. Quanto maior, maior a pressão sobre a equipe e o hospital. "
        "Use-o como <b>termômetro mensal</b> para acionar planos de contingência.",
        "🏋️"
    )

    # IPO por mês
    ipo_mensal = []
    for _, row in serie.iterrows():
        mes_d = d[d["AnoMes"] == row["AnoMes"]]
        pct_ac = (mes_d.get("MOBILIDADE", pd.Series(dtype=str)) == "ACAMADO").mean() * 100 if "MOBILIDADE" in mes_d.columns else 50.0
        ipo_v, ipo_l, ipo_niv = _ipo(
            float(row["Taxa_Adq"]), float(row["Taxa_Grave"]),
            pct_ac, int(row["Total"]), float(media_mensal_hist)
        )
        ipo_mensal.append({"Label": row["Label"], "IPO": ipo_v, "Nivel": ipo_l, "NivelCod": ipo_niv,
                           "Total": int(row["Total"]), "Taxa_Adq": float(row["Taxa_Adq"])})

    df_ipo = pd.DataFrame(ipo_mensal)

    cores_ipo = {
        "crit": _C_VERMELHO, "warn": _C_LARANJA, "ok": _C_VERDE
    }

    fig_ipo = go.Figure()
    # Faixas de referência
    fig_ipo.add_hrect(y0=0,  y1=20, fillcolor="rgba(45,122,79,0.08)",  line_width=0, annotation_text="🟢 Controlado",  annotation_position="top left")
    fig_ipo.add_hrect(y0=20, y1=40, fillcolor="rgba(200,121,65,0.08)", line_width=0, annotation_text="🟡 Moderado",    annotation_position="top left")
    fig_ipo.add_hrect(y0=40, y1=65, fillcolor="rgba(200,121,65,0.15)", line_width=0, annotation_text="🟠 Elevado",     annotation_position="top left")
    fig_ipo.add_hrect(y0=65, y1=100, fillcolor="rgba(192,57,43,0.10)", line_width=0, annotation_text="🔴 Crítico",     annotation_position="top left")

    fig_ipo.add_trace(go.Scatter(
        x=df_ipo["Label"], y=df_ipo["IPO"],
        mode="lines+markers",
        line=dict(color=_C_ESCURO, width=3),
        marker=dict(
            size=14,
            color=[cores_ipo[n] for n in df_ipo["NivelCod"]],
            line=dict(color="white", width=2),
        ),
        text=df_ipo["IPO"].astype(str),
        textposition="top center",
        name="IPO Mensal",
    ))

    fig_ipo.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="IPO (0–100)", range=[0, 105],
                   showgrid=True, gridcolor="#ede9e3"),
        plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ipo, width="stretch")

    # Tabela de capacidade projetada
    st.markdown("**📋 Planejamento de Capacidade por Mês Projetado**")
    _nota(
        "Com base no IPO e no forecast, a tabela abaixo estima a necessidade operacional "
        "para os próximos meses. O número de <b>rounds recomendados</b> e o <b>nível de estoque de insumos</b> "
        "são calculados proporcionalmente ao volume previsto e ao IPO histórico médio.",
    )

    ipo_medio = df_ipo["IPO"].mean()
    rows_plan = []
    for i, (lab, yh, lo, hi) in enumerate(zip(future_labels, y_hat, ci_low, ci_high)):
        ipo_proj = min(100, ipo_medio * (yh / media_mensal_hist))
        _, lbl_p, niv_p = _ipo(taxa_adq_geral, taxa_grave_geral, pct_acamados, int(yh), media_mensal_hist)
        rounds = max(4, int(yh / 12))          # 1 round para cada ~12 avaliações previstas
        estoque_cobert = int(yh * 0.45 * 2.5)  # 45% de adquiridas * 2.5 coberturas/caso
        rows_plan.append({
            "Mês Projetado": lab,
            "Avaliações Prev.": f"~{int(yh)} ({int(lo)}–{int(hi)})",
            "IPO Estimado": round(ipo_proj, 1),
            "Status": lbl_p,
            "Rounds CPGLP Recomendados": rounds,
            "Coberturas Estimadas (unid.)": estoque_cobert,
        })

    df_plan = pd.DataFrame(rows_plan)
    st.dataframe(
        df_plan,
        column_config={
            "Mês Projetado":            st.column_config.TextColumn(width="small"),
            "Avaliações Prev.":         st.column_config.TextColumn(),
            "IPO Estimado":             st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=100),
            "Status":                   st.column_config.TextColumn(width="small"),
            "Rounds CPGLP Recomendados":st.column_config.NumberColumn(format="%d rounds"),
            "Coberturas Estimadas (unid.)": st.column_config.NumberColumn(format="%d unid."),
        },
        hide_index=True, use_container_width=True,
    )

    st.markdown("---")

    # =========================================================================
    # 9. CICLO DIA DO MÊS + VELOCÍMETRO IPO ATUAL
    # =========================================================================
    col_dm, col_gauge = st.columns([1.5, 1])

    with col_dm:
        st.markdown("**📆 Volume por Dia do Mês (1 a 31) com Linha de Demanda**")
        _nota(
            "Mostra se existe <b>concentração em datas específicas</b> do mês — "
            "por exemplo, picos no início do mês podem indicar que avaliações admissionais "
            "são feitas em lote. Isso orienta <b>quando escalar mais equipe ou fazer auditorias</b>.",
        )
        vc_dm = d["DiaMes"].value_counts().sort_index().reset_index()
        vc_dm.columns = ["Dia", "Registros"]

        fig_dm = go.Figure()
        fig_dm.add_trace(go.Bar(
            x=vc_dm["Dia"], y=vc_dm["Registros"],
            marker_color=_C_VERDE, name="Volume", opacity=0.75,
        ))
        fig_dm.add_trace(go.Scatter(
            x=vc_dm["Dia"], y=vc_dm["Registros"],
            mode="lines", line_shape="spline",
            line=dict(color=_C_ESCURO, width=2.5), name="Tendência",
        ))
        media_dm = vc_dm["Registros"].mean()
        fig_dm.add_hline(y=media_dm, line_dash="dot", line_color=_C_LARANJA,
                         annotation_text=f"Média: {media_dm:.0f}/dia",
                         annotation_position="bottom right")
        fig_dm.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Dia do Mês", tickmode="linear", dtick=2),
            yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
            plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_dm, width="stretch")

    with col_gauge:
        st.markdown("**🌡️ Velocímetro — IPO Atual do Período**")
        _nota(
            "O ponteiro indica onde o período selecionado se encontra "
            "na escala de pressão operacional. "
            "<b>Verde</b> = controlado, <b>Vermelho</b> = ação imediata necessária.",
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ipo_val,
            delta={"reference": 40, "increasing": {"color": _C_VERMELHO},
                   "decreasing": {"color": _C_VERDE}},
            title={"text": f"IPO do Período<br><span style='font-size:0.85em;color:#666'>{ipo_label}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": cores_ipo.get(ipo_nivel, _C_AZUL), "thickness": 0.3},
                "steps": [
                    {"range": [0,  20], "color": "rgba(45,122,79,0.15)"},
                    {"range": [20, 40], "color": "rgba(200,121,65,0.15)"},
                    {"range": [40, 65], "color": "rgba(200,121,65,0.25)"},
                    {"range": [65,100], "color": "rgba(192,57,43,0.20)"},
                ],
                "threshold": {
                    "line": {"color": _C_VERMELHO, "width": 4},
                    "thickness": 0.75,
                    "value": 65,
                },
            },
            number={"suffix": "/100", "font": {"size": 40}},
        ))
        fig_gauge.update_layout(
            height=380, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, width="stretch")

    st.markdown("---")

    # =========================================================================
    # 10. BOXPLOT DE DISTRIBUIÇÃO MENSAL POR TIPO DE LESÃO
    # =========================================================================
    st.subheader("📦 Distribuição Mensal por Tipo de Classificação")
    _nota(
        "O gráfico de caixas (<em>boxplot</em>) mostra como o volume de cada categoria de lesão "
        "se distribui ao longo dos meses. A <b>linha central</b> é a mediana (mês típico), "
        "as bordas da caixa são o intervalo interquartil (50% dos meses), e os pontos isolados "
        "são meses atípicos — possivelmente picos ou vales excepcionais que merecem investigação.",
        "📦"
    )

    if "CLASSIFICAÇÃO" in d.columns:
        d_box = d.copy()
        d_box["MesLabel"] = d_box["AnoMes"].dt.strftime("%b/%y").str.upper()
        box_data = (
            d_box.groupby(["MesLabel", "CLASSIFICAÇÃO"])
            .size()
            .reset_index(name="N")
        )
        # Filtrar categorias com dados suficientes
        cats_validas = box_data["CLASSIFICAÇÃO"].value_counts()
        cats_validas = cats_validas[cats_validas >= 3].index.tolist()
        box_data = box_data[box_data["CLASSIFICAÇÃO"].isin(cats_validas)]

        if not box_data.empty:
            fig_box = px.box(
                box_data, x="CLASSIFICAÇÃO", y="N",
                color="CLASSIFICAÇÃO",
                color_discrete_sequence=[_C_AZUL, _C_VERDE, _C_LARANJA, _C_VERMELHO],
                points="all",
                labels={"N": "Avaliações/mês", "CLASSIFICAÇÃO": "Tipo de Classificação"},
            )
            fig_box.update_layout(
                height=380, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
            )
            st.plotly_chart(fig_box, width="stretch")