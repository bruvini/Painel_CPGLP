import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Imports de ML (com fallback gracioso se não instalado) ───────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
    ML_DISPONIVEL = True
except ImportError:
    ML_DISPONIVEL = False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

_MESES_ORDEM = [
    "JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO",
    "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"
]

_BRADEN_MAP = {
    "SEM RISCO ( 19-23 PONTOS)":       21,
    "BAIXO RISCO ( 15 -18 PONTOS)":    16,
    "RISCO MODERADO (13 -14 PONTOS)":  13,
    "RISCO ALTO ( 10 -12 PONTOS)":     11,
    "RISCO MUITO ALTO 6 -9 PONTOS)":    7,
}
_BWAT_MAP = {
    "GRAVIDADE MINIMA ( 13 A 20 PONTOS)":    16,
    "GRAVIDADE LEVE ( 21 A 30 PONTOS)":      25,
    "GRAVIDADE MODERADA ( 31 A 40 PONTOS )": 35,
    "GRAVIDADE CRITICA ( 41 A 65 PONTOS)":   50,
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (compartilhado entre custeio e ML)
# ─────────────────────────────────────────────────────────────────────────────
def _engenharia(df: pd.DataFrame, dias_extra: int, diaria_uti: float, diaria_enf: float,
                custo_insumo_leve: float, custo_insumo_grave: float,
                custo_hora_enf: float, horas_curativo: float,
                val_grau_1: float, val_grau_2_plus: float, val_desbrid: float) -> pd.DataFrame:
    d = df.copy()

    # ── Classificação ────────────────────────────────────────────────────────
    d["Is_Adquirida"] = d.get("CLASSIFICAÇÃO", pd.Series(dtype=str)).str.contains(
        "ADQUIRIDA NA INTERNAÇÃO ATUAL", na=False, case=False
    )
    d["Is_Grave"] = d.get("LESÃO POR PRESSÃO -  ESTAGIO", pd.Series(dtype=str)).str.contains(
        "III|IV|ESTADIÁVEL", na=False
    )
    d["Tipo_Leito"] = "Enfermaria"
    if "SETORES" in d.columns:
        d.loc[d["SETORES"].str.contains("UTI|U.T.I|CTI", na=False, case=False), "Tipo_Leito"] = "UTI"

    # ── Grau de LPP ─────────────────────────────────────────────────────────
    d["LPP_Grau"] = "Sem LPP"
    if "LESÃO POR PRESSÃO -  ESTAGIO" in d.columns:
        m1 = d["LESÃO POR PRESSÃO -  ESTAGIO"].str.fullmatch("ESTAGIO I", na=False)
        m2 = d["LESÃO POR PRESSÃO -  ESTAGIO"].str.fullmatch("ESTAGIO II", na=False)
        mg = d["LESÃO POR PRESSÃO -  ESTAGIO"].str.contains("III|IV|ESTADIÁVEL", na=False)
        d.loc[m1, "LPP_Grau"] = "Grau I"
        d.loc[m2, "LPP_Grau"] = "Grau II"
        d.loc[mg, "LPP_Grau"] = "Grau III+"

    d["Gravidade_Insumo"] = "Nulo"
    d.loc[d["LPP_Grau"].isin(["Grau I", "Grau II"]), "Gravidade_Insumo"] = "Leve/Moderado"
    d.loc[d["LPP_Grau"] == "Grau III+",              "Gravidade_Insumo"] = "Grave/Complexo"

    # ── Desbridamento ────────────────────────────────────────────────────────
    d["Tem_Desbridamento"] = False
    if "DESBRIDAMENTO " in d.columns:
        d["Tem_Desbridamento"] = d["DESBRIDAMENTO "].isin(
            ["ENZIMÁTICO", "CIRURGICO", "INSTRUMENTAL", "MECANICO"]
        )

    # ── Faturamento SUS ──────────────────────────────────────────────────────
    d["Faturamento_SUS"] = 0.0
    d.loc[d["LPP_Grau"] == "Grau I",                       "Faturamento_SUS"] += val_grau_1
    d.loc[d["LPP_Grau"].isin(["Grau II", "Grau III+"]),    "Faturamento_SUS"] += val_grau_2_plus
    d.loc[d["Tem_Desbridamento"],                           "Faturamento_SUS"] += val_desbrid

    # ── Custos de tratamento (apenas adquiridas ATUAIS) ──────────────────────
    mask = d["Is_Adquirida"]
    d["Custo_Leito"]  = 0.0
    d.loc[mask & (d["Tipo_Leito"] == "UTI"),        "Custo_Leito"]  = dias_extra * diaria_uti
    d.loc[mask & (d["Tipo_Leito"] == "Enfermaria"), "Custo_Leito"]  = dias_extra * diaria_enf

    d["Custo_Insumo"] = 0.0
    d.loc[mask & (d["Gravidade_Insumo"] == "Leve/Moderado"), "Custo_Insumo"] = dias_extra * custo_insumo_leve
    d.loc[mask & (d["Gravidade_Insumo"] == "Grave/Complexo"),"Custo_Insumo"] = dias_extra * custo_insumo_grave

    d["Custo_RH"] = 0.0
    d.loc[mask, "Custo_RH"] = dias_extra * (horas_curativo * custo_hora_enf)

    d["Custo_Total_Oculto"] = d["Custo_Leito"] + d["Custo_Insumo"] + d["Custo_RH"]

    # ── Features de ML ───────────────────────────────────────────────────────
    d["braden_num"] = d.get("ESCALA BRADEN", pd.Series(dtype=str)).map(_BRADEN_MAP).fillna(14)
    d["bwat_num"]   = d.get("ESCALA BWAT",   pd.Series(dtype=str)).map(_BWAT_MAP).fillna(0)
    d["is_uti"]          = (d["Tipo_Leito"] == "UTI").astype(int)
    d["is_acamado"]      = (d.get("MOBILIDADE", "") == "ACAMADO").astype(int)
    d["is_antibiotico"]  = (d.get("ANTIBIÓTICO TERAPIA SISTÊMICO", "") == "SIM").astype(int)
    d["is_precaucao"]    = (d.get("PRECAUÇÃO", "") == "SIM").astype(int)
    d["nutricao_risco"]  = d.get("NUTRIÇÃO", pd.Series(dtype=str)).isin(["SNE", "NPT", "DIETA ZERO"]).astype(int)
    d["tem_desbrid_ml"]  = d["Tem_Desbridamento"].astype(int)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PREDITIVO DE RISCO
# ─────────────────────────────────────────────────────────────────────────────
_ML_FEATURES = ["braden_num", "bwat_num", "is_uti", "is_acamado",
                "is_antibiotico", "is_precaucao", "nutricao_risco", "tem_desbrid_ml"]

_FEATURE_LABELS = {
    "braden_num":     "Escore Braden",
    "bwat_num":       "Gravidade BWAT",
    "is_uti":         "Internação em UTI",
    "is_acamado":     "Paciente Acamado",
    "is_antibiotico": "Uso de Antibiótico",
    "is_precaucao":   "Precaução de Contato",
    "nutricao_risco": "Nutrição de Risco (SNE/NPT)",
    "tem_desbrid_ml": "Necessidade de Desbridamento",
}

@st.cache_resource(show_spinner=False)
def _treinar_modelo(X_hash: str, X_arr, y_arr):
    """Treina RF cacheado — re-treina só se os dados mudarem."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_arr, y_arr)
    aucs = cross_val_score(rf, X_arr, y_arr, cv=5, scoring="roc_auc")
    return rf, aucs.mean(), aucs.std()


def _modelo_risco(df_eng: pd.DataFrame):
    """Retorna (modelo_treinado, auc_media, auc_std, df_com_prob)."""
    X = df_eng[_ML_FEATURES].values
    y = df_eng["Is_Adquirida"].astype(int).values

    # hash simples para cache
    import hashlib
    h = hashlib.md5(X.tobytes()).hexdigest()
    rf, auc_m, auc_s = _treinar_modelo(h, X, y)

    df_eng = df_eng.copy()
    df_eng["prob_lpp"] = rf.predict_proba(X)[:, 1]
    df_eng["risco_class"] = pd.cut(
        df_eng["prob_lpp"],
        bins=[0, 0.25, 0.50, 0.70, 1.0],
        labels=["🟢 Baixo", "🟡 Moderado", "🟠 Alto", "🔴 Crítico"]
    )
    return rf, auc_m, auc_s, df_eng


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def render_financeiro(df: pd.DataFrame):
    if df.empty:
        st.warning("⚠️ Não existem dados para os filtros selecionados.")
        return

    # =========================================================================
    # 1. MOTOR DE SIMULAÇÃO DE CUSTOS (PARÂMETROS AJUSTÁVEIS)
    # =========================================================================
    with st.expander("⚙️ Motor de Simulação de Custos Hospitalares e Tabela SUS", expanded=False):
        st.markdown("Ajuste os parâmetros institucionais para modelar cenários preditivos e de custeio ABC (*Activity-Based Costing*).")
        tab_sus, tab_leito, tab_insumo, tab_rh = st.tabs(["Faturamento SUS", "Custos de Leito", "Insumos & Prevenção", "Recursos Humanos"])
        with tab_sus:
            c1, c2 = st.columns(2)
            val_grau_1       = c1.number_input("Tabela SUS – LPP Grau I (R$)",   value=10.00, step=1.0)
            val_grau_2_plus  = c2.number_input("Tabela SUS – LPP Grau II+ (R$)", value=27.22, step=1.0)
            val_desbrid      = c1.number_input("Tabela SUS – Desbridamento (R$)", value=123.55, step=10.0)
        with tab_leito:
            c3, c4 = st.columns(2)
            diaria_enf = c3.number_input("Custo Diária Enfermaria (R$)", value=240.00, step=10.0)
            diaria_uti = c4.number_input("Custo Diária UTI (R$)",        value=1200.00, step=50.0)
            dias_extra = st.slider("Média de Dias Extras p/ LPP Adq.", 1, 30, 7)
        with tab_insumo:
            st.caption("Custos estimados de materiais de curativo e protocolo de prevenção.")
            c5, c6, c7 = st.columns(3)
            custo_insumo_leve  = c5.number_input("Insumo LPP I/II (R$/dia)",        value=15.50)
            custo_insumo_grave = c6.number_input("Insumo LPP III/IV (R$/dia)",       value=85.00)
            custo_prevencao    = c7.number_input("Custo Kit Prevenção/Pac. (R$)",    value=45.00,
                                                 help="Coxins, AGE, filme transparente preventivo por paciente")
        with tab_rh:
            c8, c9 = st.columns(2)
            custo_hora_enf  = c8.number_input("Custo Hora/Enfermagem (R$)",             value=45.00)
            tempo_curativo  = c9.number_input("Tempo médio por curativo complexo (min)", value=40)
        horas_curativo = tempo_curativo / 60

    # =========================================================================
    # 2. ENGENHARIA DE FEATURES + CUSTEIO
    # =========================================================================
    df_fin = _engenharia(
        df, dias_extra, diaria_uti, diaria_enf,
        custo_insumo_leve, custo_insumo_grave,
        custo_hora_enf, horas_curativo,
        val_grau_1, val_grau_2_plus, val_desbrid
    )

    total_sus       = df_fin["Faturamento_SUS"].sum()
    total_leito     = df_fin["Custo_Leito"].sum()
    total_insumo    = df_fin["Custo_Insumo"].sum()
    total_rh        = df_fin["Custo_RH"].sum()
    total_perda     = df_fin["Custo_Total_Oculto"].sum()
    saldo_final     = total_sus - total_perda
    n_adq           = int(df_fin["Is_Adquirida"].sum())
    total_pac       = len(df_fin)
    invest_prev     = total_pac * custo_prevencao
    roi_mult        = (total_perda / invest_prev) if invest_prev > 0 else 0.0
    custo_medio_adq = total_perda / max(n_adq, 1)

    # =========================================================================
    # 3. MODELO PREDITIVO DE RISCO (RF)
    # =========================================================================
    rf_model, auc_m, auc_s, df_fin = _modelo_risco(df_fin) if ML_DISPONIVEL else (None, 0, 0, df_fin)

    # =========================================================================
    # 4. KPIs GERENCIAIS
    # =========================================================================
    st.subheader("💰 Auditoria e Balanço Financeiro Geral")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Receita Máx. Estimada",   _brl(total_sus))
    k2.metric("Custo Oculto – LPP Adq.", _brl(total_perda),
              delta="↑ Desperdício", delta_color="inverse")
    k3.metric("Balanço Operacional",     _brl(saldo_final),
              delta_color="normal" if saldo_final >= 0 else "inverse")
    k4.metric("Custo Médio / LPP Adq.",  _brl(custo_medio_adq))
    k5.metric("ROI Prevenção",           f"1 : {roi_mult:.1f}x",
              delta="Retorno Econômico", delta_color="normal")

    st.markdown("---")

    # =========================================================================
    # 5. CARD ROI
    # =========================================================================
    if total_perda > 0:
        st.markdown(f"""
        <div style="background:#fdfdfd; padding:1.5rem; border-left:6px solid #1a4d7a;
                    border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:1.5rem;">
            <h3 style="margin-top:0; color:#1a4d7a;">📈 Impacto Econômico da Prevenção (ROI)</h3>
            <p style="font-size:1.05rem; color:#333; line-height:1.6;">
                Um investimento de <b>{_brl(custo_prevencao)}</b>/paciente totalizaria <b>{_brl(invest_prev)}</b>
                para os <b>{total_pac}</b> pacientes analisados — frente a <b>{_brl(total_perda)}</b>
                em custos ocultos gerados pelas <b>{n_adq}</b> LPPs adquiridas na internação atual.
            </p>
            <div style="background:#e8f4fd; padding:14px; border-radius:5px; text-align:center; margin:12px 0;">
                <h4 style="color:#1a4d7a; margin:0; font-size:1.25rem;">
                    Para cada <b>R$ 1,00</b> investido em Prevenção Sistemática,<br>
                    o Hospital São José economiza <b>R$ {roi_mult:.2f}</b> em Custos de Tratamento Oculto.
                </h4>
            </div>
            <p style="font-size:0.88rem; color:#666; margin:0;">
                <em>*ROI = Custo total de tratamento das LPPs adquiridas ÷ Custo profilático universal estimado.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # 6. FLUXO DE CAIXA (WATERFALL) + DECOMPOSIÇÃO DE CUSTO
    # =========================================================================
    st.subheader("📊 Análise de Fluxo de Caixa e Composição do Custo Oculto")
    col_w, col_p = st.columns([1.5, 1])

    with col_w:
        st.caption("Waterfall – Fluxo de Caixa Assistencial")
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Faturamento SUS", "Insumos/Coberturas", "Mão de Obra (RH)", "Leito Extra", "Saldo Retido"],
            textposition="outside",
            text=[_brl(total_sus), _brl(-total_insumo), _brl(-total_rh), _brl(-total_leito), _brl(saldo_final)],
            y=[total_sus, -total_insumo, -total_rh, -total_leito, saldo_final],
            connector={"line": {"color": "#1a1714"}},
            decreasing={"marker": {"color": "#c0392b"}},
            increasing={"marker": {"color": "#2d7a4f"}},
            totals={"marker": {"color": "#1a4d7a"}},
        ))
        fig_w.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                            yaxis=dict(showticklabels=False),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_w, width="stretch")

    with col_p:
        st.caption("Pizza – Decomposição do Custo Oculto Total")
        if total_perda > 0:
            df_break = pd.DataFrame({
                "Categoria": ["Leito Bloqueado", "Insumos (Coberturas)", "Horas Enfermagem"],
                "Valor": [total_leito, total_insumo, total_rh]
            })
            fig_p = px.pie(df_break, values="Valor", names="Categoria", hole=0.5,
                           color_discrete_sequence=["#c87941", "#7b4fa6", "#2d6fa3"])
            fig_p.update_traces(textinfo="percent+label", textposition="inside")
            fig_p.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_p, width="stretch")
        else:
            st.info("Não há custos de desperdício para a seleção atual.")

    st.markdown("---")

    # =========================================================================
    # 7. SIMULADOR DE CENÁRIOS "E SE?" (WHAT-IF)
    # =========================================================================
    st.subheader("🔮 Simulador de Cenários Preventivos (What-If)")
    st.caption(
        "Arraste o slider para simular quanto o hospital economizaria se a CPGLP conseguisse reduzir "
        "a taxa de LPPs adquiridas em diferentes percentuais. As projeções usam os custos médios "
        "calibrados pelo Motor de Simulação acima."
    )

    col_sl, col_res = st.columns([1.5, 1])
    with col_sl:
        reducao_pct = st.slider(
            "Redução hipotética da incidência de LPP Adquirida (%)",
            min_value=10, max_value=100, value=40, step=5,
            help="Estudos do MS/ANVISA mostram que bundles de prevenção reduzem a incidência em 40-60%."
        )

    lpp_evitadas      = int(n_adq * (reducao_pct / 100))
    economia_bruta    = lpp_evitadas * custo_medio_adq
    custo_add_prev    = lpp_evitadas * custo_prevencao   # custo extra para prevenir os casos
    economia_liquida  = economia_bruta - custo_add_prev
    vidas_risco_red   = int(lpp_evitadas * 0.08)         # ~8% das LPPs graves levam a sepse com risco de vida

    with col_res:
        st.markdown(f"""
        <div style="background:#fff; border:1px solid #c0c0c0; border-radius:8px; padding:16px; font-size:0.95rem;">
            <b>Cenário: {reducao_pct}% de redução</b><br><br>
            🏥 LPPs evitadas: <b style="color:#2d7a4f;">{lpp_evitadas}</b> de {n_adq}<br>
            💰 Economia bruta estimada: <b style="color:#2d7a4f;">{_brl(economia_bruta)}</b><br>
            🛡️ Investimento adicional em prev.: <b style="color:#c87941;">{_brl(custo_add_prev)}</b><br>
            ✅ Economia líquida: <b style="color:#1a4d7a; font-size:1.1rem;">{_brl(economia_liquida)}</b><br>
            ❤️ Eventos adversos evitados: ~<b>{vidas_risco_red}</b> situações de risco de vida
        </div>
        """, unsafe_allow_html=True)

    # Gráfico de curva de impacto (todos os percentuais)
    cenarios = pd.DataFrame({
        "Redução (%)": range(0, 101, 5),
    })
    cenarios["LPPs Evitadas"]     = (cenarios["Redução (%)"] / 100 * n_adq).astype(int)
    cenarios["Economia Bruta"]    = cenarios["LPPs Evitadas"] * custo_medio_adq
    cenarios["Custo Prevenção"]   = cenarios["LPPs Evitadas"] * custo_prevencao
    cenarios["Economia Líquida"]  = cenarios["Economia Bruta"] - cenarios["Custo Prevenção"]

    fig_cenario = go.Figure()
    fig_cenario.add_trace(go.Scatter(
        x=cenarios["Redução (%)"], y=cenarios["Economia Bruta"],
        name="Economia Bruta", mode="lines", line=dict(color="#2d7a4f", width=2.5)
    ))
    fig_cenario.add_trace(go.Scatter(
        x=cenarios["Redução (%)"], y=cenarios["Custo Prevenção"],
        name="Investimento em Prevenção", mode="lines",
        line=dict(color="#c87941", width=2, dash="dot")
    ))
    fig_cenario.add_trace(go.Scatter(
        x=cenarios["Redução (%)"], y=cenarios["Economia Líquida"],
        name="Economia Líquida (ROI Real)", mode="lines+markers",
        line=dict(color="#1a4d7a", width=3),
        marker=dict(size=5)
    ))
    fig_cenario.add_vline(
        x=reducao_pct, line_dash="dash", line_color="#c0392b",
        annotation_text=f"Cenário atual: {reducao_pct}%",
        annotation_position="top right"
    )
    fig_cenario.update_layout(
        height=320, margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Redução da Incidência de LPP Adquirida (%)",
        yaxis_title="Impacto Financeiro (R$)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#ede9e3"),
        yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
    )
    st.plotly_chart(fig_cenario, width="stretch")

    st.markdown("---")

    # =========================================================================
    # 8. ML: MODELO PREDITIVO DE RISCO DE LPP ADQUIRIDA
    # =========================================================================
    if ML_DISPONIVEL and rf_model is not None:
        st.subheader("🤖 Modelo Preditivo de Risco – Random Forest (ML)")
        st.caption(
            f"Modelo treinado com **{total_pac} registros reais** do dataset. "
            f"AUC validação cruzada 5-fold: **{auc_m:.3f} ± {auc_s:.3f}**. "
            "O modelo aprende o perfil dos pacientes que desenvolvem LPP adquirida "
            "e gera uma probabilidade individual de risco."
        )

        # ── Feature Importance ───────────────────────────────────────────────
        col_imp, col_risco = st.columns([1.2, 1])
        with col_imp:
            st.markdown("**Importância das Variáveis Clínicas**")
            imp_df = pd.DataFrame({
                "Variável": [_FEATURE_LABELS[f] for f in _ML_FEATURES],
                "Importância": rf_model.feature_importances_,
            }).sort_values("Importância", ascending=True)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importância"],
                y=imp_df["Variável"],
                orientation="h",
                marker=dict(
                    color=imp_df["Importância"],
                    colorscale="Blues",
                    showscale=False,
                ),
                text=[f"{v:.1%}" for v in imp_df["Importância"]],
                textposition="outside",
            ))
            fig_imp.update_layout(
                height=340, margin=dict(l=0, r=60, t=10, b=0),
                xaxis=dict(showticklabels=False, showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_imp, width="stretch")

        with col_risco:
            st.markdown("**Distribuição de Risco Previsto**")
            dist = df_fin["risco_class"].value_counts().reindex(
                ["🟢 Baixo", "🟡 Moderado", "🟠 Alto", "🔴 Crítico"], fill_value=0
            ).reset_index()
            dist.columns = ["Nível de Risco", "Pacientes"]
            dist["Pct"] = (dist["Pacientes"] / dist["Pacientes"].sum() * 100).round(1)

            fig_dist = go.Figure(go.Bar(
                x=dist["Nível de Risco"],
                y=dist["Pacientes"],
                text=[f"{p}%" for p in dist["Pct"]],
                textposition="outside",
                marker_color=["#2d7a4f", "#c87941", "#e67e22", "#c0392b"],
            ))
            fig_dist.update_layout(
                height=340, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_dist, width="stretch")

        # ── Custo Projetado pelo Modelo ───────────────────────────────────────
        st.markdown("**💸 Projeção Financeira por Nível de Risco Preditivo**")
        st.caption(
            "Cruzamento entre a probabilidade de LPP prevista pelo modelo e o custo médio de tratamento — "
            "gerando um score de 'custo esperado não prevenido' por grupo de risco."
        )
        risk_finance = df_fin.groupby("risco_class", observed=True).agg(
            Pacientes=("prob_lpp", "count"),
            Prob_Media=("prob_lpp", "mean"),
            Custo_Medio_Real=("Custo_Total_Oculto", "mean"),
            Custo_Total_Real=("Custo_Total_Oculto", "sum"),
        ).reset_index()
        risk_finance["Custo_Esperado_Nao_Prev"] = (
            risk_finance["Prob_Media"] * risk_finance["Pacientes"] * custo_medio_adq
        )
        risk_finance.columns = [
            "Nível de Risco", "Pacientes", "Prob. Média (%)",
            "Custo Médio Real (R$)", "Custo Total Real (R$)", "Custo Esperado Não Prevenido (R$)"
        ]
        risk_finance["Prob. Média (%)"] = (risk_finance["Prob. Média (%)"] * 100).round(1)

        st.dataframe(
            risk_finance,
            column_config={
                "Nível de Risco":                  st.column_config.TextColumn(width="small"),
                "Pacientes":                       st.column_config.NumberColumn(format="%d"),
                "Prob. Média (%)":                 st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100),
                "Custo Médio Real (R$)":           st.column_config.NumberColumn(format="R$ %.2f"),
                "Custo Total Real (R$)":           st.column_config.NumberColumn(format="R$ %.2f"),
                "Custo Esperado Não Prevenido (R$)": st.column_config.ProgressColumn(
                    format="R$ %.0f",
                    min_value=0,
                    max_value=float(risk_finance["Custo Esperado Não Prevenido (R$)"].max() + 1)
                ),
            },
            hide_index=True, use_container_width=True
        )

        # ── Scatter Risco x Custo por Setor ──────────────────────────────────
        if "SETORES" in df_fin.columns:
            st.markdown("**🎯 Mapa de Risco Financeiro por Setor (ML + Custeio)**")
            st.caption(
                "Cada bolha é um setor: Eixo X = probabilidade média de LPP prevista pelo modelo, "
                "Eixo Y = custo oculto total gerado, tamanho = número de pacientes em risco alto/crítico."
            )
            setor_ml = df_fin.groupby("SETORES").agg(
                Prob_Media=("prob_lpp", "mean"),
                Custo_Total=("Custo_Total_Oculto", "sum"),
                N_Alto_Risco=("prob_lpp", lambda x: (x >= 0.50).sum()),
                Total_Pac=("prob_lpp", "count"),
            ).reset_index()
            setor_ml = setor_ml[setor_ml["Total_Pac"] >= 3]
            setor_ml["Prob_Media_Pct"] = (setor_ml["Prob_Media"] * 100).round(1)

            fig_scatter_ml = px.scatter(
                setor_ml,
                x="Prob_Media_Pct", y="Custo_Total",
                size=setor_ml["N_Alto_Risco"].clip(lower=1),
                color="Custo_Total",
                text="SETORES",
                color_continuous_scale="Reds",
                size_max=50,
                hover_data={"Total_Pac": True, "N_Alto_Risco": True},
                labels={
                    "Prob_Media_Pct": "Probabilidade Média de LPP (%)",
                    "Custo_Total": "Custo Oculto Total (R$)",
                },
            )
            fig_scatter_ml.update_traces(
                textposition="top center",
                textfont=dict(size=11, color="#1a1714")
            )
            fig_scatter_ml.add_vline(
                x=setor_ml["Prob_Media_Pct"].mean(),
                line_dash="dash", line_color="#a09890"
            )
            fig_scatter_ml.add_hline(
                y=setor_ml["Custo_Total"].mean(),
                line_dash="dash", line_color="#a09890"
            )
            fig_scatter_ml.update_layout(
                height=520, margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#ede9e3"),
                yaxis=dict(showgrid=True, gridcolor="#ede9e3"),
            )
            st.plotly_chart(fig_scatter_ml, width="stretch")

        st.markdown("---")

    # =========================================================================
    # 9. SÉRIE TEMPORAL DE CUSTOS POR MÊS
    # =========================================================================
    if "MÊS" in df_fin.columns:
        st.subheader("📅 Evolução Mensal do Custo Oculto e da Taxa de Incidência")
        st.caption("Série temporal combinando custo financeiro acumulado por mês com a taxa de LPPs adquiridas.")

        df_fin["mes_cat"] = pd.Categorical(
            df_fin["MÊS"], categories=_MESES_ORDEM, ordered=True
        )
        serie = df_fin.groupby("mes_cat", observed=True).agg(
            Total=("Is_Adquirida", "count"),
            Adq=("Is_Adquirida", "sum"),
            Custo=("Custo_Total_Oculto", "sum"),
            Prob_Media=("prob_lpp", "mean") if ML_DISPONIVEL else ("Custo_Total_Oculto", "mean"),
        ).reset_index()
        serie.columns = ["Mês", "Total", "Adquiridas", "Custo Oculto", "Prob Média ML"]
        serie = serie[serie["Total"] > 0]
        serie["Taxa (%)"] = (serie["Adquiridas"] / serie["Total"] * 100).round(1)

        fig_serie = go.Figure()
        fig_serie.add_trace(go.Bar(
            x=serie["Mês"], y=serie["Custo Oculto"],
            name="Custo Oculto (R$)",
            marker_color="#c0392b", opacity=0.75, yaxis="y"
        ))
        fig_serie.add_trace(go.Scatter(
            x=serie["Mês"], y=serie["Taxa (%)"],
            name="Taxa Incidência (%)",
            mode="lines+markers",
            line=dict(color="#1a4d7a", width=3),
            marker=dict(size=8), yaxis="y2"
        ))
        fig_serie.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Custo Oculto (R$)", showgrid=True, gridcolor="#ede9e3"),
            yaxis2=dict(title="Taxa de Incidência (%)", overlaying="y", side="right",
                        range=[0, serie["Taxa (%)"].max() * 1.3 + 1], showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#fdfdfd", paper_bgcolor="rgba(0,0,0,0)",
            barmode="group"
        )
        st.plotly_chart(fig_serie, width="stretch")
        st.markdown("---")

    # =========================================================================
    # 10. MATRIZ FINANCEIRA POR SETOR (TREEMAP + SUNBURST)
    # =========================================================================
    st.subheader("🏥 Matriz de Risco Financeiro por Setor")
    col_tree, col_sun = st.columns(2)

    with col_tree:
        st.caption("Treemap – Complexidade da Área vs. Perda Financeira")
        df_tree = df_fin[df_fin["Custo_Total_Oculto"] > 0]
        if not df_tree.empty and "SETORES" in df_tree.columns:
            td = df_tree.groupby(["Tipo_Leito", "SETORES"]).agg(
                Custo=("Custo_Total_Oculto", "sum"),
                N=("Is_Adquirida", "count")
            ).reset_index()
            fig_tree = px.treemap(
                td, path=[px.Constant("Visão Geral"), "Tipo_Leito", "SETORES"],
                values="Custo", color="Custo", color_continuous_scale="Reds"
            )
            fig_tree.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, width="stretch")
        else:
            st.info("Sem dados suficientes.")

    with col_sun:
        st.caption("Sunburst – Gravidade e Custo de Coberturas por Tipo de Leito")
        df_sun = df_fin[df_fin["Custo_Insumo"] > 0]
        if not df_sun.empty:
            sd = df_sun.groupby(["Tipo_Leito", "LPP_Grau", "Gravidade_Insumo"]).agg(
                Custo=("Custo_Insumo", "sum")
            ).reset_index()
            fig_sun = px.sunburst(
                sd, path=["Tipo_Leito", "LPP_Grau", "Gravidade_Insumo"],
                values="Custo", color="Custo", color_continuous_scale="Purples"
            )
            fig_sun.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_sun, width="stretch")
        else:
            st.info("Sem dados suficientes.")

    st.markdown("---")

    # =========================================================================
    # 11. TABELA SIGTAP E REFERÊNCIAS
    # =========================================================================
    st.subheader("📋 Tabela de Faturamento SUS (SIGTAP) e Bases Normativas")
    st.caption("Os cálculos utilizam como base os seguintes códigos SIGTAP e referências literárias.")

    col_tab, col_ref = st.columns([1.2, 1])
    with col_tab:
        st.markdown("""
        <div style="background:#fff; border:1px solid #e2ddd6; border-radius:5px; padding:10px;">
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
            <thead>
                <tr style="border-bottom:2px solid #c0392b;">
                    <th style="padding:8px;">Código SUS</th>
                    <th style="padding:8px;">Descrição</th>
                    <th style="padding:8px;">Valor (R$)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;">03.01.01.004-8</td>
                    <td style="padding:8px;">Consulta Profissional Nível Superior – Atenção Especializada</td>
                    <td style="padding:8px;">R$ 10,00</td>
                </tr>
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;">04.14.02.009-8</td>
                    <td style="padding:8px;">Curativo Grau II c/ ou s/ Desbridamento</td>
                    <td style="padding:8px;">R$ 14,70</td>
                </tr>
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;">04.14.02.010-1</td>
                    <td style="padding:8px;">Curativo Grau III ou IV c/ ou s/ Desbridamento</td>
                    <td style="padding:8px;">R$ 27,22</td>
                </tr>
                <tr>
                    <td style="padding:8px;">04.01.01.002-3</td>
                    <td style="padding:8px;">Desbridamento Cirúrgico / Exérese de Tecido Necrótico</td>
                    <td style="padding:8px;">R$ 123,55</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col_ref:
        st.markdown("""
        **Referências e Fontes:**
        - **SIGTAP:** Tabela de Procedimentos SUS — *Competência base: Out/2025.*
        - **Permanência:** Dealey C, Posnett J, Walker A. *The cost of pressure ulcers in the UK*. J Wound Care. 2012. *(Base para 7 dias extras)*
        - **Bundle de Prevenção:** MS/ANVISA — *Protocolo para Prevenção de Úlcera por Pressão*, 2013. *(Redução de até 60%)*
        - **NPIAP/EPUAP 2019:** *Prevention and Treatment of Pressure Ulcers/Injuries: Clinical Practice Guideline.*
        - **Modelo Preditivo:** Random Forest treinado nos dados reais do dataset — AUC validada por 5-fold cross-validation.
        """)

    st.markdown("---")

    # =========================================================================
    # 12. RELATÓRIO ANALÍTICO POR SETOR (TABELA EXPORTÁVEL)
    # =========================================================================
    st.subheader("📋 Relatório Analítico de Custos por Setor")
    st.caption("Tabela com dados tratados pelo motor financeiro, pronta para exportação em auditorias de qualidade.")

    if "SETORES" in df_fin.columns:
        agg_cols = {
            "Volume_Registros":       ("SETORES",               "count"),
            "LPPs_Adquiridas":        ("Is_Adquirida",          "sum"),
            "Faturamento_Gerado":     ("Faturamento_SUS",       "sum"),
            "Custo_Insumos":          ("Custo_Insumo",          "sum"),
            "Custo_Leito_Extra":      ("Custo_Leito",           "sum"),
            "Custo_RH":               ("Custo_RH",              "sum"),
            "Custo_Total_Desperdicio":("Custo_Total_Oculto",    "sum"),
        }
        if ML_DISPONIVEL:
            agg_cols["Prob_Media_LPP"] = ("prob_lpp", "mean")

        df_tabela = (
            df_fin.groupby("SETORES")
            .agg(**agg_cols)
            .reset_index()
            .sort_values("Custo_Total_Desperdicio", ascending=False)
        )
        if ML_DISPONIVEL:
            df_tabela["Prob_Media_LPP"] = (df_tabela["Prob_Media_LPP"] * 100).round(1)

        col_cfg = {
            "SETORES":                   "Setor",
            "Volume_Registros":          st.column_config.NumberColumn("Total Aval.", format="%d"),
            "LPPs_Adquiridas":           st.column_config.NumberColumn("LPPs Adq."),
            "Faturamento_Gerado":        st.column_config.NumberColumn("Receita Estimada", format="R$ %.2f"),
            "Custo_Insumos":             st.column_config.NumberColumn("Custo Coberturas", format="R$ %.2f"),
            "Custo_Leito_Extra":         st.column_config.NumberColumn("Leito Bloqueado",  format="R$ %.2f"),
            "Custo_RH":                  st.column_config.NumberColumn("Custo RH",         format="R$ %.2f"),
            "Custo_Total_Desperdicio":   st.column_config.ProgressColumn(
                "Custo Oculto Total", format="R$ %.2f",
                min_value=0, max_value=float(df_tabela["Custo_Total_Desperdicio"].max() + 1)
            ),
        }
        if ML_DISPONIVEL:
            col_cfg["Prob_Media_LPP"] = st.column_config.ProgressColumn(
                "Risco Previsto (%)", format="%.1f%%", min_value=0, max_value=100
            )

        st.dataframe(df_tabela, column_config=col_cfg,
                     hide_index=True, use_container_width=True)