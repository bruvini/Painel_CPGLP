import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_estrategica(df):
    if df.empty:
        st.warning("⚠️ Não existem dados para os filtros selecionados.")
        return

    df_est = df.copy()

    # ==========================================
    # 1. ENGENHARIA DE FEATURES BÁSICAS
    # ==========================================
    df_est['Is_Adquirida'] = False
    if 'CLASSIFICAÇÃO' in df_est.columns:
        df_est['Is_Adquirida'] = df_est['CLASSIFICAÇÃO'].str.contains(
            'ADQUIRIDA NA INTERNAÇÃO ATUAL', na=False, case=False
        )

    df_est['Is_Grave'] = False
    if 'LESÃO POR PRESSÃO -  ESTAGIO' in df_est.columns:
        df_est['Is_Grave'] = df_est['LESÃO POR PRESSÃO -  ESTAGIO'].str.contains(
            'III|IV|ESTADIÁVEL', na=False
        )

    total_avaliacoes = len(df_est)
    total_adquiridas = int(df_est['Is_Adquirida'].sum())
    taxa_adquirida = (total_adquiridas / total_avaliacoes) * 100 if total_avaliacoes > 0 else 0.0
    META_QUALIDADE = 10.0

    # ==========================================
    # 2. MOTOR DE ALERTAS INTELIGENTES
    # ==========================================
    st.subheader("🚨 Central de Alertas e Recomendações Clínicas")
    st.caption("Auditoria automatizada baseada nos limites de segurança assistencial e protocolos de prevenção.")

    if taxa_adquirida > META_QUALIDADE:
        st.error(
            f"**ALERTA CRÍTICO: Taxa de Incidência Elevada ({taxa_adquirida:.1f}%)**\n\n"
            f"O volume de LPPs adquiridas na internação atual ultrapassa a meta de segurança institucional de {META_QUALIDADE}%. "
            f"Recomenda-se revisão imediata das rotinas de mudança de decúbito e uso de coxins profiláticos."
        )
    elif taxa_adquirida > (META_QUALIDADE / 2):
        st.warning(
            f"**ATENÇÃO: Taxa de Incidência em Limite de Atenção ({taxa_adquirida:.1f}%)**\n\n"
            f"A métrica de LPPs adquiridas está crescendo em direção à margem de risco (Meta: {META_QUALIDADE}%). "
            f"Reforçar rounds preventivos da equipe de enfermagem."
        )
    else:
        st.success(
            f"**MÉTRICA SAUDÁVEL: Controle Profilático Eficiente ({taxa_adquirida:.1f}%)**\n\n"
            f"A taxa de LPPs adquiridas está contida dentro das metas de segurança do paciente. "
            f"Manter os protocolos atuais."
        )

    if 'SETORES' in df_est.columns and total_adquiridas > 0:
        setores_adq = df_est[df_est['Is_Adquirida']]['SETORES'].value_counts()
        if not setores_adq.empty:
            pior_setor = setores_adq.index[0]
            qtd_pior = setores_adq.iloc[0]
            st.info(
                f"**🎯 ALVO PRIORITÁRIO:** O setor **{pior_setor}** é responsável pelo maior foco individual "
                f"de lesões adquiridas ({qtd_pior} registros na atual seleção). "
                f"Sugerimos priorizar treinamentos in loco e auditoria de leitos para esta unidade específica nesta semana."
            )

    st.markdown("---")

    # ==========================================
    # 3. ANÁLISE DE CAUSA RAIZ (PARETO 80/20)
    # ==========================================
    st.subheader("🔍 Análise de Causa Raiz: Princípio de Pareto (80/20)")
    st.caption("Identifique visualmente quais setores ou localizações anatômicas concentram a maior parte das ocorrências.")

    col_pareto1, col_pareto2 = st.columns(2)

    def plot_pareto(df_data, col_name, title):
        if df_data.empty or col_name not in df_data.columns:
            return None
        df_p = df_data[col_name].value_counts().reset_index()
        df_p.columns = [col_name, 'Frequência']
        df_p = df_p[df_p['Frequência'] > 0]
        if df_p.empty:
            return None
        df_p['% Acumulado'] = (df_p['Frequência'].cumsum() / df_p['Frequência'].sum()) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_p[col_name], y=df_p['Frequência'],
            name='Registros', marker_color='#2d6fa3', yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=df_p[col_name], y=df_p['% Acumulado'],
            name='% Acumulado', mode='lines+markers',
            line=dict(color='#c0392b', width=3),
            marker=dict(size=8), yaxis='y2'
        ))
        fig.update_layout(
            title=title,
            height=400, margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(title='Frequência', side='left', showgrid=False),
            yaxis2=dict(title='% Acumulado', side='right', overlaying='y', range=[0, 105], showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_hline(y=80, yref='y2', line_dash="dot", line_color="#4a4540", annotation_text="Corte 80%")
        return fig

    with col_pareto1:
        if 'SETORES' in df_est.columns:
            df_adq_pareto = df_est[df_est['Is_Adquirida']]
            fig_pareto_setor = plot_pareto(df_adq_pareto, 'SETORES', "Concentração de LPP Adq. por Setor")
            if fig_pareto_setor:
                st.plotly_chart(fig_pareto_setor, width="stretch")
            else:
                st.info("Nenhuma LPP adquirida no período para este filtro.")
        else:
            st.info("Dado de setores não disponível.")

    with col_pareto2:
        if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns:
            fig_pareto_loc = plot_pareto(df_est, 'LOCALIZAÇÃO DA LESÃO', "Localizações Anatômicas mais Afetadas")
            if fig_pareto_loc:
                st.plotly_chart(fig_pareto_loc, width="stretch")
            else:
                st.info("Dado de localização não disponível.")
        else:
            st.info("Dado de localização não disponível.")

    st.markdown("---")

    # ==========================================
    # 4. MATRIZ DE RISCO OPERACIONAL (QUADRANTES)
    # ==========================================
    st.subheader("🎯 Matriz de Risco Clínico-Operacional (Setores)")
    st.caption(
        "Cruzamento estratégico: Eixo X (Volume Total de Avaliações) vs Eixo Y (% de Lesões Graves Nível III/IV). "
        "Tamanho da bolha indica o número de LPPs Adquiridas."
    )

    # Inicializa df_matriz vazio para garantir que sempre existe
    df_matriz = pd.DataFrame()

    if 'SETORES' in df_est.columns:
        df_matriz = df_est.groupby('SETORES').agg(
            Volume_Total=('SETORES', 'count'),
            Graves=('Is_Grave', 'sum'),
            Adquiridas=('Is_Adquirida', 'sum')
        ).reset_index()

        df_matriz['Taxa_Gravidade'] = (df_matriz['Graves'] / df_matriz['Volume_Total']) * 100
        df_matriz = df_matriz[df_matriz['Volume_Total'] >= 3]

        if not df_matriz.empty:
            media_vol  = df_matriz['Volume_Total'].mean()
            media_grav = df_matriz['Taxa_Gravidade'].mean()

            # Garante que 'Adquiridas' tem pelo menos valor 1 para size (bolhas visíveis)
            size_col = df_matriz['Adquiridas'].clip(lower=1)

            fig_matriz = px.scatter(
                df_matriz, x='Volume_Total', y='Taxa_Gravidade',
                size=size_col, color='Taxa_Gravidade', text='SETORES',
                color_continuous_scale='Reds',
                size_max=40,
                hover_data=['Graves', 'Adquiridas']
            )
            fig_matriz.update_traces(textposition='top center', textfont=dict(size=11, color='#1a1714'))
            fig_matriz.add_vline(x=media_vol, line_width=1, line_dash="dash", line_color="#a09890")
            fig_matriz.add_hline(y=media_grav, line_width=1, line_dash="dash", line_color="#a09890")
            fig_matriz.add_annotation(
                x=df_matriz['Volume_Total'].max() * 0.9,
                y=df_matriz['Taxa_Gravidade'].max() * 0.95,
                text="CRÍTICO<br>(Alto Volume, Alta Gravidade)",
                showarrow=False, font=dict(color="#c0392b", size=10), opacity=0.6
            )
            fig_matriz.add_annotation(
                x=df_matriz['Volume_Total'].min() * 1.05,
                y=df_matriz['Taxa_Gravidade'].min() * 1.05,
                text="EFICIENTE<br>(Baixo Volume, Baixa Gravidade)",
                showarrow=False, font=dict(color="#2d7a4f", size=10), opacity=0.6
            )
            fig_matriz.update_layout(
                height=550, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Volume Total de Avaliações (Alta Demanda)",
                yaxis_title="Taxa de Gravidade % (LPPs Estágio III, IV e NE)",
                coloraxis_showscale=False,
                plot_bgcolor='#fdfdfd',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#ede9e3'),
                yaxis=dict(showgrid=True, gridcolor='#ede9e3')
            )
            st.plotly_chart(fig_matriz, width="stretch")

            criticos = df_matriz[
                (df_matriz['Volume_Total'] >= media_vol) &
                (df_matriz['Taxa_Gravidade'] >= media_grav)
            ]['SETORES'].tolist()
            if criticos:
                st.warning(
                    f"**Ação Recomendada:** Os setores **{', '.join(criticos)}** estão no quadrante CRÍTICO da matriz. "
                    f"Concentram grande volume de trabalho e alta incidência de casos graves/complexos. "
                    f"Recomendada intervenção multidisciplinar imediata."
                )
        else:
            st.info("Volume de dados insuficiente para montar a matriz de dispersão por setores.")

    # ==========================================
    # 5. PLANO DE AÇÃO ESTRATÉGICO DINÂMICO
    # ==========================================
    _render_plano_acao(df_est, df_matriz, taxa_adquirida, META_QUALIDADE)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCO INTERNO — não exportado diretamente; chamado por render_estrategica
# ─────────────────────────────────────────────────────────────────────────────
def _render_plano_acao(
    df_est: pd.DataFrame,
    df_matriz: pd.DataFrame,
    taxa_adquirida: float,
    META_QUALIDADE: float = 10.0
):
    CORES = {
        "critico":     {"borda": "#c0392b", "titulo": "#c0392b", "icone": "🚨", "label": "CRÍTICO"},
        "importante":  {"borda": "#c87941", "titulo": "#c87941", "icone": "⚠️",  "label": "IMPORTANTE"},
        "nao_critico": {"borda": "#2d6fa3", "titulo": "#2d6fa3", "icone": "ℹ️",  "label": "ATENÇÃO"},
        "excelente":   {"borda": "#2d7a4f", "titulo": "#2d7a4f", "icone": "✅",  "label": "EXCELENTE"},
    }

    def card(nivel: str, titulo: str, problema: str, fundamentacao: str, acoes: list):
        c = CORES[nivel]
        acoes_html = "".join(f"<li>{a}</li>" for a in acoes)
        st.markdown(f"""
        <div style="background:#fff; border-left:5px solid {c['borda']}; padding:15px;
                    margin-bottom:12px; border-radius:4px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color:{c['titulo']}; margin-top:0;">
                {c['icone']} [{c['label']}] {titulo}
            </h4>
            <p style="margin-bottom:8px;"><b>Problema Identificado:</b> {problema}</p>
            <p style="margin-bottom:8px; font-size:0.9em; color:#4a4540;">
                <b>📚 Fundamentação:</b> <em>{fundamentacao}</em>
            </p>
            <div style="background:#fdfdfd; border:1px solid #e2ddd6; padding:10px; border-radius:4px;">
                <b style="color:#1a4d7a;">🚀 Proposta de Melhoria de Fluxo:</b>
                <ul style="margin-bottom:0;">{acoes_html}</ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pré-cálculo de métricas ───────────────────────────────────────────
    total      = len(df_est)
    total_adq  = int(df_est['Is_Adquirida'].sum()) if 'Is_Adquirida' in df_est.columns else 0
    total_grave= int(df_est['Is_Grave'].sum())     if 'Is_Grave'     in df_est.columns else 0
    taxa_grave = (total_grave / total) * 100 if total > 0 else 0.0

    loc_top = ""
    if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns and not df_est['LOCALIZAÇÃO DA LESÃO'].dropna().empty:
        loc_top = df_est['LOCALIZAÇÃO DA LESÃO'].value_counts().index[0]

    setor_vol = setor_grav_top = setor_adq_top = ""
    taxa_grav_max = 0.0
    if not df_matriz.empty:
        setor_vol      = df_matriz.sort_values('Volume_Total',  ascending=False).iloc[0]['SETORES']
        setor_grav_top = df_matriz.sort_values('Taxa_Gravidade',ascending=False).iloc[0]['SETORES']
        taxa_grav_max  = df_matriz['Taxa_Gravidade'].max()
        if 'Adquiridas' in df_matriz.columns:
            setor_adq_top = df_matriz.sort_values('Adquiridas', ascending=False).iloc[0]['SETORES']

    tendencia_crescente = False
    col_data = next((c for c in ['Carimbo de data/hora', 'DATA', 'DATA DE AVALIAÇÃO'] if c in df_est.columns), None)
    if col_data:
        df_temp = df_est.copy()
        df_temp['_dt'] = pd.to_datetime(df_temp[col_data], errors='coerce')
        df_temp = df_temp.dropna(subset=['_dt']).sort_values('_dt')
        if len(df_temp) >= 4:
            mid = len(df_temp) // 2
            taxa_rec  = df_temp.iloc[mid:]['Is_Adquirida'].mean() * 100
            taxa_ant  = df_temp.iloc[:mid]['Is_Adquirida'].mean() * 100
            tendencia_crescente = taxa_rec > taxa_ant

    pct_sem_local = 0.0
    if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns:
        pct_sem_local = (df_est['LOCALIZAÇÃO DA LESÃO'].isna().sum() / total) * 100 if total > 0 else 0.0

    pct_nao_estagiavel = 0.0
    if 'LESÃO POR PRESSÃO -  ESTAGIO' in df_est.columns:
        n_ne = df_est['LESÃO POR PRESSÃO -  ESTAGIO'].str.contains('ESTADIÁVEL', na=False).sum()
        pct_nao_estagiavel = (n_ne / total) * 100 if total > 0 else 0.0

    gini_setores = 0.0
    if not df_matriz.empty and 'Volume_Total' in df_matriz.columns:
        v = df_matriz['Volume_Total'].values.astype(float)
        v_sorted = np.sort(v)
        n = len(v_sorted)
        if n > 1 and v_sorted.sum() > 0:
            gini_setores = (2 * np.sum(np.arange(1, n + 1) * v_sorted) / (n * v_sorted.sum())) - (n + 1) / n

    pct_setores_criticos = 0.0
    if not df_matriz.empty:
        mv = df_matriz['Volume_Total'].mean()
        mg = df_matriz['Taxa_Gravidade'].mean()
        nc = ((df_matriz['Volume_Total'] >= mv) & (df_matriz['Taxa_Gravidade'] >= mg)).sum()
        pct_setores_criticos = (nc / len(df_matriz)) * 100

    # ── Definição das 20 estratégias ─────────────────────────────────────
    estrategias = []

    # 1. Taxa de Incidência Global
    def nivel_incidencia():
        if taxa_adquirida > META_QUALIDADE:
            return "critico", {
                "titulo": f"Taxa de Incidência de LPP Adquirida Elevada ({taxa_adquirida:.1f}%)",
                "problema": f"A incidência atingiu {taxa_adquirida:.1f}%, ultrapassando a meta de {META_QUALIDADE}%. Indica falha sistêmica nos processos de profilaxia primária.",
                "fundamentacao": "Protocolo MS/ANVISA: bundles de medidas reduzem a incidência hospitalar em até 60% quando implementados corretamente.",
                "acoes": [
                    "<b>Bundle Admissional:</b> Tornar obrigatória a aplicação da Escala de Braden no prontuário eletrônico em até 6h da admissão.",
                    "<b>Controle Visual:</b> Implantar relógios de reposicionamento (mudança de decúbito a cada 2h) fixados à beira leito.",
                    "<b>Auditoria de Insumos:</b> Revisar o estoque de coxins, filmes transparentes e coberturas profiláticas por setor semanalmente.",
                ]
            }
        elif taxa_adquirida > META_QUALIDADE * 0.6:
            return "importante", {
                "titulo": f"Taxa de Incidência em Zona de Atenção ({taxa_adquirida:.1f}%)",
                "problema": f"A taxa de {taxa_adquirida:.1f}% está se aproximando da margem crítica de {META_QUALIDADE}%. Requer intervenção preventiva.",
                "fundamentacao": "NPIAP/EPUAP: monitoramento ativo semanal é recomendado como estratégia de detecção precoce de surtos de incidência.",
                "acoes": [
                    "<b>Rounds Preventivos:</b> Reforçar os rounds de enfermagem com checklist de inspeção cutânea ao menos 2x/turno.",
                    "<b>Educação Continuada:</b> Organizar treinamento mensal para novos servidores sobre mapeamento de risco cutâneo.",
                ]
            }
        elif taxa_adquirida > 0:
            return "nao_critico", {
                "titulo": f"Taxa de Incidência dentro do Limite ({taxa_adquirida:.1f}%)",
                "problema": f"Incidência controlada em {taxa_adquirida:.1f}%, abaixo da meta. Casos adquiridos ainda merecem revisão individual.",
                "fundamentacao": "IHI: zero LPP evitável é o objetivo-alvo de excelência em segurança do paciente.",
                "acoes": [
                    "<b>Análise de Causa Raiz:</b> Realizar RCA individual para cada novo caso adquirido identificado.",
                    "<b>Monitoramento Contínuo:</b> Manter painel atualizado semanalmente para detectar tendências precocemente.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Zero LPP Adquirida na Internação Atual no Período",
                "problema": "Nenhuma lesão por pressão adquirida na internação atual foi registrada. Meta de excelência atingida.",
                "fundamentacao": "IHI: a meta 'zero harm' é alcançável com protocolos bem implementados e cultura de segurança consolidada.",
                "acoes": [
                    "<b>Reconhecimento:</b> Comunicar o resultado à equipe como reforço positivo do protocolo vigente.",
                    "<b>Documentação:</b> Registrar as práticas no relatório da CPGLP como referência para outros setores.",
                ]
            }
    estrategias.append(nivel_incidencia)

    # 2. Gravidade das Lesões
    def nivel_gravidade():
        if taxa_grave > 15.0:
            return "critico", {
                "titulo": f"Alta Proporção de Lesões Graves – Estágio III/IV/NE ({taxa_grave:.1f}%)",
                "problema": f"{taxa_grave:.1f}% das avaliações apresentam lesões graves, sinalizando risco iminente de complicações sistêmicas como sepse.",
                "fundamentacao": "NPIAP/EPUAP: lesões de grau elevado exigem superfícies de suporte ativo, desbridamento e suporte nutricional hiperproteico imediato.",
                "acoes": [
                    f"<b>Gestão de Ativos:</b> Criar fluxo prioritário de colchões pneumáticos para pacientes com Braden ≤ 12 no setor {setor_grav_top or 'identificado'}.",
                    "<b>Notificação Compulsória:</b> Garantir notificação de todas as LPPs Grau III e IV ao NSP em até 24h.",
                    "<b>Multidisciplinar:</b> Acionar nutrição clínica automaticamente para suplementação hiperproteica em pacientes com Grau II evoluindo para III.",
                ]
            }
        elif taxa_grave > 8.0:
            return "importante", {
                "titulo": f"Proporção Elevada de Lesões Graves ({taxa_grave:.1f}%)",
                "problema": f"A taxa de lesões graves está em {taxa_grave:.1f}%. Sugere pacientes avaliados tardiamente ou com progressão não controlada.",
                "fundamentacao": "Protocolo MS: avaliação diária e registro fotográfico permitem detectar progressão de estágio em até 48h.",
                "acoes": [
                    "<b>Registro Fotográfico:</b> Padronizar o registro fotográfico de todas as LPPs Estágio II ou superior.",
                    "<b>Round de Estomaterapia:</b> Realizar round semanal com enfermeira estomaterapeuta/CPGLP nos setores mais afetados.",
                ]
            }
        elif taxa_grave > 0:
            return "nao_critico", {
                "titulo": f"Gravidade Moderada Controlada ({taxa_grave:.1f}%)",
                "problema": f"Há {taxa_grave:.1f}% de lesões graves, compatível com a complexidade dos casos internados. Requer acompanhamento sistemático.",
                "fundamentacao": "NPIAP: acompanhamento contínuo é essencial para evitar que lesões Grau II evoluam para estágios superiores.",
                "acoes": [
                    "<b>Monitoramento Individual:</b> Assegurar que cada paciente com LPP Grau II+ possua plano de cuidados individualizado no prontuário.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Ausência de Lesões Graves no Período",
                "problema": "Nenhuma LPP de Estágio III, IV ou Não Estadiável foi registrada. Resultado de excelência clínica.",
                "fundamentacao": "IHI: prevenção de progressão é tão importante quanto a prevenção da incidência.",
                "acoes": [
                    "<b>Boas Práticas:</b> Compartilhar o protocolo de prevenção de progressão com outros setores como modelo de referência.",
                ]
            }
    estrategias.append(nivel_gravidade)

    # 3. Concentração por Setor (Gini)
    def nivel_concentracao():
        if gini_setores > 0.4:
            return "critico", {
                "titulo": f"Alta Concentração de Lesões em Poucos Setores (Gini={gini_setores:.2f})",
                "problema": f"Distribuição fortemente desigual entre setores (índice {gini_setores:.2f}). Um ou dois setores concentram a maior parte das ocorrências.",
                "fundamentacao": "Princípio de Pareto: 20% das unidades tendem a concentrar 80% dos eventos adversos evitáveis.",
                "acoes": [
                    f"<b>Força-Tarefa:</b> Deslocar temporariamente um enfermeiro especialista para o setor {setor_adq_top or setor_vol} durante 30 dias.",
                    "<b>Auditoria Estrutural:</b> Verificar adequação de insumos, quadro de pessoal e rotinas no setor mais crítico.",
                    "<b>Mapeamento de Risco:</b> Conduzir mapeamento de risco ambiental (leitos, posição, luminosidade) na unidade crítica.",
                ]
            }
        elif gini_setores > 0.25:
            return "importante", {
                "titulo": f"Concentração Moderada de Ocorrências por Setor (Gini={gini_setores:.2f})",
                "problema": "Alguns setores apresentam carga desproporcional de LPPs em relação ao volume de pacientes atendidos.",
                "fundamentacao": "Análise de causa raiz por setor permite identificar variáveis estruturais que explicam a concentração.",
                "acoes": [
                    "<b>Benchmarking Interno:</b> Comparar rotinas dos setores com menor taxa com os de maior concentração para identificar boas práticas replicáveis.",
                ]
            }
        elif gini_setores > 0:
            return "nao_critico", {
                "titulo": f"Distribuição Razoavelmente Uniforme por Setor (Gini={gini_setores:.2f})",
                "problema": "Distribuição de ocorrências moderada entre setores. Não há concentração extrema.",
                "fundamentacao": "Distribuição homogênea pode refletir protocolos bem disseminados, mas também pode mascarar falhas sistêmicas.",
                "acoes": [
                    "<b>Análise Segmentada:</b> Revisar individualmente os casos de cada setor para identificar padrões locais específicos.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Distribuição Uniforme de Ocorrências Entre Setores",
                "problema": "LPPs distribuídas proporcionalmente ao volume de cada setor, sem concentração anormal.",
                "fundamentacao": "Distribuição equitativa sugere que os protocolos estão sendo aplicados de forma homogênea em toda a instituição.",
                "acoes": [
                    "<b>Padronização:</b> Documentar o modelo atual como padrão institucional para próximas acreditações.",
                ]
            }
    estrategias.append(nivel_concentracao)

    # 4. LPPs Não Estadiáveis
    def nivel_nao_estagiavel():
        if pct_nao_estagiavel > 20.0:
            return "critico", {
                "titulo": f"Alta Taxa de LPPs Não Estadiáveis ({pct_nao_estagiavel:.1f}%)",
                "problema": f"{pct_nao_estagiavel:.1f}% das lesões são Não Estadiáveis (por esfacelo ou escara), comprometendo a qualidade do dado assistencial.",
                "fundamentacao": "NPIAP: LPPs Não Estadiáveis exigem desbridamento; manutenção > 7 dias representa atraso terapêutico.",
                "acoes": [
                    "<b>Protocolo de Desbridamento:</b> Criar fluxo de avaliação obrigatória da CPGLP em até 48h para toda lesão Não Estadiável.",
                    "<b>Capacitação:</b> Treinar a equipe na diferenciação entre Não Estadiável (NE) e Suspeita de Lesão Tissular Profunda (SLTP).",
                    "<b>Revisão Periódica:</b> Estabelecer revisão semanal de todas as LPPs NE para reclassificação.",
                ]
            }
        elif pct_nao_estagiavel > 10.0:
            return "importante", {
                "titulo": f"Taxa Moderada de LPPs Não Estadiáveis ({pct_nao_estagiavel:.1f}%)",
                "problema": f"Proporção de {pct_nao_estagiavel:.1f}% pode indicar dificuldade técnica na classificação ou atraso no desbridamento.",
                "fundamentacao": "Protocolo MS: estadiamento correto é pré-requisito para prescrição adequada de curativos e superfícies de suporte.",
                "acoes": [
                    "<b>Workshop:</b> Realizar oficina prática de estadiamento com casos reais para os enfermeiros de cada setor.",
                    "<b>Revisão de Prontuários:</b> Auditar prontuários de LPPs NE para verificar o tempo médio entre classificação e desbridamento.",
                ]
            }
        elif pct_nao_estagiavel > 0:
            return "nao_critico", {
                "titulo": f"Pequena Proporção de LPPs Não Estadiáveis ({pct_nao_estagiavel:.1f}%)",
                "problema": "Poucos casos NE no período. Situações pontuais que merecem acompanhamento individualizado.",
                "fundamentacao": "Desbridamento precoce de lesões NE reduz o risco de progressão para Grau IV em até 40% (EPUAP, 2019).",
                "acoes": [
                    "<b>Acompanhamento:</b> Assegurar reavaliação semanal dos casos NE até o estadiamento definitivo.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma LPP Classificada como Não Estadiável",
                "problema": "Todas as lesões estão devidamente estadiadas. Excelente qualidade do dado clínico.",
                "fundamentacao": "Classificação completa permite intervenção terapêutica mais precisa e reduz tempo de cicatrização.",
                "acoes": [
                    "<b>Boas Práticas:</b> Documentar o fluxo de estadiamento atual como modelo para acreditação hospitalar.",
                ]
            }
    estrategias.append(nivel_nao_estagiavel)

    # 5. Completude de Localização Anatômica
    def nivel_completude_local():
        if pct_sem_local > 20.0:
            return "critico", {
                "titulo": f"Preenchimento Inadequado de Localização Anatômica ({pct_sem_local:.1f}% sem dado)",
                "problema": f"{pct_sem_local:.1f}% dos registros sem localização compromete as análises de causa raiz.",
                "fundamentacao": "Resolução CFM 1638/2002 e LGPD: o prontuário deve conter todas as informações essenciais para continuidade do cuidado.",
                "acoes": [
                    "<b>Campo Obrigatório:</b> Solicitar à TI o bloqueio do envio do formulário sem o preenchimento da localização anatômica.",
                    "<b>Auditoria de Qualidade:</b> Incluir completude dos campos obrigatórios como indicador mensal da CPGLP.",
                    "<b>Treinamento:</b> Orientar a equipe sobre a importância do preenchimento completo.",
                ]
            }
        elif pct_sem_local > 10.0:
            return "importante", {
                "titulo": f"Preenchimento Parcial de Localização Anatômica ({pct_sem_local:.1f}%)",
                "problema": f"{pct_sem_local:.1f}% sem localização limita a análise anatômica de distribuição de lesões.",
                "fundamentacao": "Localização anatômica é variável-chave para identificar falhas de posicionamento e prescrição de superfícies de suporte.",
                "acoes": [
                    "<b>Feedback Mensal:</b> Divulgar o índice de completude por setor para promover responsabilização.",
                ]
            }
        elif pct_sem_local > 0:
            return "nao_critico", {
                "titulo": f"Pequena Lacuna no Preenchimento de Localização ({pct_sem_local:.1f}%)",
                "problema": "Poucos registros sem localização. Não compromete a análise geral, mas deve ser corrigido.",
                "fundamentacao": "Completude de 100% nos dados clínicos é o padrão recomendado por programas de acreditação ONA e JCI.",
                "acoes": [
                    "<b>Alinhamento:</b> Reforçar na próxima reunião da CPGLP a importância do preenchimento integral dos formulários.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "100% dos Registros com Localização Anatômica Preenchida",
                "problema": "Completude total nos dados de localização anatômica. Qualidade do registro excelente.",
                "fundamentacao": "Alta completude potencializa a capacidade analítica do painel e a precisão das intervenções.",
                "acoes": [
                    "<b>Reconhecimento:</b> Destacar a qualidade do preenchimento como critério positivo na avaliação de desempenho da equipe.",
                ]
            }
    estrategias.append(nivel_completude_local)

    # 6. Proporção de Setores no Quadrante Crítico
    def nivel_setores_criticos():
        if pct_setores_criticos > 40.0:
            return "critico", {
                "titulo": f"Maioria dos Setores no Quadrante de Alto Risco ({pct_setores_criticos:.0f}%)",
                "problema": f"{pct_setores_criticos:.0f}% dos setores estão no quadrante de alto volume e alta gravidade. Indica problema sistêmico institucional.",
                "fundamentacao": "Gestão Lean Healthcare: quando múltiplas unidades operam em estado crítico, o problema geralmente está em processos centralizados.",
                "acoes": [
                    "<b>Reunião de Crise:</b> Convocar reunião com a direção de enfermagem e CPGLP para revisão dos protocolos centrais.",
                    "<b>Diagnóstico de Carga:</b> Avaliar a relação paciente/enfermeiro em todos os setores críticos.",
                    "<b>Reestruturação:</b> Revisar o Protocolo Institucional de Prevenção de LPP à luz das diretrizes NPIAP 2019.",
                ]
            }
        elif pct_setores_criticos > 25.0:
            return "importante", {
                "titulo": f"Percentual Relevante de Setores em Risco ({pct_setores_criticos:.0f}%)",
                "problema": f"Cerca de {pct_setores_criticos:.0f}% dos setores estão em zona de risco elevado. Requer atenção antes de tornar-se sistêmico.",
                "fundamentacao": "Intervenção precoce em múltiplas unidades de médio risco é mais custo-efetiva do que resposta a crises.",
                "acoes": [
                    "<b>Plano por Unidade:</b> Elaborar planos de ação individualizados para cada setor no quadrante de risco.",
                    "<b>Benchmarking:</b> Identificar o setor com melhor desempenho e replicar suas práticas nos demais.",
                ]
            }
        elif pct_setores_criticos > 10.0:
            return "nao_critico", {
                "titulo": f"Poucos Setores em Zona de Risco ({pct_setores_criticos:.0f}%)",
                "problema": "Minoria dos setores no quadrante crítico, com a maioria dentro de parâmetros aceitáveis.",
                "fundamentacao": "Intervenção cirúrgica em poucas unidades tem maior impacto no resultado geral.",
                "acoes": [
                    "<b>Foco Pontual:</b> Concentrar esforços nos setores críticos sem dispersar recursos nas demais unidades.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhum Setor no Quadrante Crítico de Alto Risco",
                "problema": "Todos os setores operam com volume e gravidade dentro de parâmetros seguros.",
                "fundamentacao": "Ausência de setores em quadrante crítico é indicador de maturidade institucional no gerenciamento de LPPs.",
                "acoes": [
                    "<b>Manutenção:</b> Preservar os processos atuais como referência para ciclos futuros de acreditação.",
                ]
            }
    estrategias.append(nivel_setores_criticos)

    # 7. Tendência Temporal
    def nivel_tendencia():
        if tendencia_crescente and taxa_adquirida > META_QUALIDADE * 0.7:
            return "critico", {
                "titulo": "Tendência de Crescimento da Incidência Detectada",
                "problema": "Taxa de LPPs adquiridas em trajetória crescente no período. Combinação de tendência ascendente com taxa próxima da meta é sinal de alerta máximo.",
                "fundamentacao": "Epidemiologia hospitalar: uma semana de inação equivale, em média, a 3 semanas de recuperação do indicador.",
                "acoes": [
                    "<b>Reunião de Emergência:</b> Convocar reunião imediata da CPGLP com apresentação dos dados à liderança de enfermagem.",
                    "<b>Investigação Temporal:</b> Correlacionar o início da tendência com mudanças na equipe, insumos ou perfil epidemiológico.",
                    "<b>Intervenção Imediata:</b> Implementar checklist de prevenção diário com assinatura do enfermeiro responsável em todos os setores.",
                ]
            }
        elif tendencia_crescente:
            return "importante", {
                "titulo": "Leve Tendência de Alta na Incidência",
                "problema": "Incidência com crescimento moderado no período. Taxa ainda dentro do limite, mas o vetor é preocupante.",
                "fundamentacao": "Monitoramento de tendência é parte essencial do ciclo PDCA em gestão de qualidade hospitalar.",
                "acoes": [
                    "<b>Alerta ao Time:</b> Comunicar a tendência à equipe em reunião de passagem de plantão como medida preventiva.",
                    "<b>Investigação:</b> Verificar correlação com sazonalidade, aumento de ocupação ou rotatividade de equipe.",
                ]
            }
        elif taxa_adquirida > 0:
            return "nao_critico", {
                "titulo": "Tendência Estável ou Decrescente da Incidência",
                "problema": "Trajetória temporal estável ou em declínio. Os protocolos estão produzindo efeito positivo.",
                "fundamentacao": "Tendência decrescente sustentada por mais de 4 semanas indica que as intervenções implementadas estão sendo efetivas.",
                "acoes": [
                    "<b>Consolidação:</b> Identificar quais intervenções correlacionam com a queda e formalizá-las no protocolo permanente.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Sem Ocorrências ou Tendência Estável no Período",
                "problema": "Não foram identificadas ocorrências ou a tendência está estável e controlada.",
                "fundamentacao": "A análise temporal é um indicador de gestão avançada; sua manutenção agrega valor à inteligência do painel.",
                "acoes": [
                    "<b>Melhoria de Dados:</b> Garantir que o campo de data seja preenchido em todos os registros para habilitar análises de tendência.",
                ]
            }
    estrategias.append(nivel_tendencia)

    # 8. Proporção de Estágio I (detecção precoce)
    def nivel_estagio1():
        pct_e1 = 0.0
        if 'LESÃO POR PRESSÃO -  ESTAGIO' in df_est.columns:
            n_e1 = df_est['LESÃO POR PRESSÃO -  ESTAGIO'].str.fullmatch('ESTAGIO I', na=False).sum()
            pct_e1 = (n_e1 / total) * 100 if total > 0 else 0.0

        if pct_e1 > 50.0:
            return "excelente", {
                "titulo": f"Alta Proporção de Lesões em Estágio I ({pct_e1:.1f}%)",
                "problema": "Mais da metade das lesões no estágio mais inicial. Indica diagnóstico precoce e intervenção oportuna.",
                "fundamentacao": "EPUAP 2019: identificação precoce no Estágio I permite intervenções de baixo custo que evitam progressão em mais de 90% dos casos.",
                "acoes": [
                    "<b>Reconhecimento:</b> Comunicar à equipe que a alta taxa de diagnóstico precoce é indicador de qualidade assistencial.",
                    "<b>Manutenção:</b> Preservar os turnos de inspeção cutânea que permitem a detecção neste estágio.",
                ]
            }
        elif pct_e1 > 30.0:
            return "nao_critico", {
                "titulo": f"Proporção Razoável de Lesões Estágio I ({pct_e1:.1f}%)",
                "problema": "Diagnóstico em estágios iniciais ocorre em parte dos casos. Há espaço para ampliar a detecção precoce.",
                "fundamentacao": "Detecção no Estágio I é indicador de excelência assistencial e deve ser meta da CPGLP.",
                "acoes": [
                    "<b>Treinamento:</b> Capacitar a equipe na identificação do eritema não branqueável característico do Estágio I.",
                ]
            }
        elif pct_e1 > 0:
            return "importante", {
                "titulo": f"Baixa Proporção de Diagnóstico em Estágio I ({pct_e1:.1f}%)",
                "problema": "Poucos casos detectados no estágio mais precoce. Sugere inspeção cutânea rotineira insuficiente.",
                "fundamentacao": "Protocolo MS: inspeção cutânea a cada turno é recomendada para pacientes com risco moderado/alto no Braden.",
                "acoes": [
                    "<b>Revisão de Rotinas:</b> Verificar se a inspeção cutânea está sendo documentada a cada turno nos setores de alto risco.",
                    "<b>Checklist de Turno:</b> Incluir inspeção cutânea como campo obrigatório no registro de enfermagem.",
                ]
            }
        else:
            return "critico", {
                "titulo": "Nenhum Diagnóstico em Estágio I Registrado",
                "problema": "Ausência de LPPs Estágio I pode indicar subnotificação grave ou registro somente após progressão para estágios avançados.",
                "fundamentacao": "CFM: apenas 30-40% das lesões Estágio I são registradas em hospitais brasileiros devido à subnotificação.",
                "acoes": [
                    "<b>Auditoria de Subnotificação:</b> Cruzar registros de enfermagem com prescrições de curativo para identificar lesões não notificadas.",
                    "<b>Cultura de Notificação:</b> Reforçar que a notificação de LPP Estágio I é positiva e não constitui falta profissional.",
                    "<b>Revisão de Protocolo:</b> Verificar se o formulário inclui campo específico para Estágio I.",
                ]
            }
    estrategias.append(nivel_estagio1)

    # 9. Diversidade Anatômica
    def nivel_diversidade_anatomica():
        n_loc = 0
        if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns:
            n_loc = df_est['LOCALIZAÇÃO DA LESÃO'].nunique()

        if n_loc > 8:
            return "importante", {
                "titulo": f"Alta Diversidade de Localizações Anatômicas ({n_loc} regiões afetadas)",
                "problema": f"Lesões em {n_loc} regiões distintas indicam variabilidade de posicionamento ou LPPs relacionadas a dispositivos médicos.",
                "fundamentacao": "Quando múltiplas regiões atípicas são afetadas, equipamentos médicos (sondas, fixadores) devem ser investigados como fator causal.",
                "acoes": [
                    "<b>Análise por Dispositivo:</b> Investigar se há LPPs relacionadas a dispositivos médicos (LPPRD) nas localizações atípicas.",
                    f"<b>Foco Anatômico:</b> Direcionar o kit profilático para a localização mais prevalente: <b>{loc_top or 'identificada'}</b>.",
                    "<b>Checklist de Dispositivos:</b> Implementar inspeção diária de todas as interfaces entre dispositivos médicos e pele.",
                ]
            }
        elif n_loc > 4:
            return "nao_critico", {
                "titulo": f"Diversidade Moderada de Localizações ({n_loc} regiões)",
                "problema": f"Lesões em {n_loc} localizações anatômicas distintas, incluindo a mais prevalente: {loc_top or 'identificada'}.",
                "fundamentacao": "Distribuição anatômica moderada é esperada em ambientes de alta complexidade com pacientes acamados.",
                "acoes": [
                    f"<b>Prevenção Direcionada:</b> Priorizar a profilaxia para a localização <b>{loc_top or 'mais prevalente'}</b> no kit de admissão.",
                ]
            }
        elif n_loc > 1:
            return "nao_critico", {
                "titulo": f"Poucas Localizações Anatômicas Afetadas ({n_loc} regiões)",
                "problema": "Lesões concentradas em poucas regiões anatômicas, facilitando a intervenção profilática direcionada.",
                "fundamentacao": "Concentração anatômica permite uso de coberturas profiláticas específicas na admissão.",
                "acoes": [
                    f"<b>Kit Específico:</b> Padronizar a disponibilização de cobertura profilática para <b>{loc_top or 'região identificada'}</b> no kit de admissão.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Localização Anatômica Única ou Dados Insuficientes",
                "problema": "Apenas uma localização afetada ou dados insuficientes. Compatível com baixíssima incidência no período.",
                "fundamentacao": "Baixa variabilidade anatômica é compatível com controle efetivo de LPPs no período analisado.",
                "acoes": [
                    "<b>Monitoramento:</b> Manter acompanhamento mesmo em períodos de baixa incidência.",
                ]
            }
    estrategias.append(nivel_diversidade_anatomica)

    # 10. Setor de Maior Volume
    def nivel_setor_maior_volume():
        taxa_vol = 0.0
        if setor_vol and not df_matriz.empty:
            row = df_matriz[df_matriz['SETORES'] == setor_vol]
            if not row.empty and 'Adquiridas' in row.columns:
                taxa_vol = (row['Adquiridas'].values[0] / row['Volume_Total'].values[0]) * 100

        if taxa_vol > 20.0:
            return "critico", {
                "titulo": f"Setor de Maior Volume '{setor_vol}' com Alta Taxa de Adquiridas ({taxa_vol:.1f}%)",
                "problema": f"O setor com maior demanda ({setor_vol}) apresenta {taxa_vol:.1f}% de LPPs adquiridas. Alta demanda e alta incidência simultâneas indicam colapso do processo preventivo.",
                "fundamentacao": "AHRQ: alta taxa de ocupação correlaciona-se com aumento de LPPs quando a relação paciente/enfermeiro ultrapassa 6:1.",
                "acoes": [
                    f"<b>Avaliação de Lotação:</b> Revisar a escala de enfermagem do setor <b>{setor_vol}</b> para verificar se a carga de trabalho está dentro do limite seguro.",
                    "<b>Delegação Supervisionada:</b> Avaliar delegação da inspeção cutânea ao técnico de enfermagem com supervisão direta.",
                    "<b>Priorização de Recursos:</b> Garantir acesso prioritário a colchões preventivos e coberturas profiláticas para este setor.",
                ]
            }
        elif taxa_vol > 10.0:
            return "importante", {
                "titulo": f"Setor '{setor_vol}' com Taxa de Adquiridas em Atenção ({taxa_vol:.1f}%)",
                "problema": f"Setor de maior volume com taxa de adquiridas de {taxa_vol:.1f}%, merecendo monitoramento ativo.",
                "fundamentacao": "Setores de alta demanda requerem processos preventivos mais rigorosos para compensar a maior exposição ao risco.",
                "acoes": [
                    f"<b>Monitoramento Reforçado:</b> Incluir o setor <b>{setor_vol}</b> no round semanal da CPGLP como prioridade.",
                ]
            }
        elif taxa_vol > 0:
            return "nao_critico", {
                "titulo": f"Setor '{setor_vol}' com Taxa Controlada ({taxa_vol:.1f}%)",
                "problema": f"Apesar do alto volume, o setor {setor_vol} mantém taxa de adquiridas dentro do limite aceitável.",
                "fundamentacao": "Alta demanda com baixa incidência é indicador de excelência operacional no setor.",
                "acoes": [
                    "<b>Boas Práticas:</b> Documentar os processos preventivos do setor como modelo para unidades com menor desempenho.",
                ]
            }
        else:
            return "excelente", {
                "titulo": f"Setor de Maior Volume '{setor_vol}' sem LPPs Adquiridas",
                "problema": "O setor com mais avaliações não registrou nenhuma LPP adquirida. Resultado de excelência assistencial.",
                "fundamentacao": "Excelência em prevenção no setor de maior demanda é o benchmark ideal para toda a instituição.",
                "acoes": [
                    "<b>Divulgação:</b> Destacar o resultado em reunião geral como referência de prática segura.",
                ]
            }
    estrategias.append(nivel_setor_maior_volume)

    # 11. Proporção Estágio II
    def nivel_estagio2():
        pct_e2 = 0.0
        if 'LESÃO POR PRESSÃO -  ESTAGIO' in df_est.columns:
            n_e2 = df_est['LESÃO POR PRESSÃO -  ESTAGIO'].str.fullmatch('ESTAGIO II', na=False).sum()
            pct_e2 = (n_e2 / total) * 100 if total > 0 else 0.0

        if pct_e2 > 40.0:
            return "importante", {
                "titulo": f"Alta Proporção de Lesões Estágio II ({pct_e2:.1f}%)",
                "problema": f"{pct_e2:.1f}% das lesões em Estágio II pode indicar diagnóstico tardio do Estágio I ou progressão não controlada.",
                "fundamentacao": "NPIAP: coberturas hidrocoloides em Estágio II reduzem o tempo de cicatrização em 30-50%.",
                "acoes": [
                    "<b>Protocolo de Cobertura:</b> Revisar disponibilidade de coberturas específicas para Estágio II (hidrocoloides, espumas de poliuretano).",
                    "<b>Avaliação Nutricional:</b> Garantir avaliação nutricional em 100% dos pacientes com LPP Estágio II.",
                    "<b>Reavaliação Diária:</b> Estabelecer reavaliação diária documentada de todas as lesões Estágio II.",
                ]
            }
        elif pct_e2 > 0:
            return "nao_critico", {
                "titulo": f"Proporção de Lesões Estágio II Dentro do Esperado ({pct_e2:.1f}%)",
                "problema": f"Proporção de {pct_e2:.1f}% de Estágio II compatível com o perfil assistencial.",
                "fundamentacao": "Acompanhamento rigoroso do Estágio II é essencial para evitar progressão para estágios mais graves.",
                "acoes": [
                    "<b>Rastreamento:</b> Monitorar semanalmente se casos de Estágio II estão evoluindo para regressão.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma LPP Estágio II no Período Selecionado",
                "problema": "Ausência de lesões em Estágio II. Resultado muito positivo.",
                "fundamentacao": "Ausência de Estágio II com boa cobertura de avaliações sugere protocolos preventivos eficazes.",
                "acoes": [
                    "<b>Documentação:</b> Registrar no relatório mensal da CPGLP como indicador de qualidade.",
                ]
            }
    estrategias.append(nivel_estagio2)

    # 12. Razão Grave/Adquirida
    def nivel_razao_grave_adq():
        razao = (total_grave / total_adq) if total_adq > 0 else 0.0
        if razao > 0.5:
            return "critico", {
                "titulo": f"Alta Proporção de Lesões Graves entre as Adquiridas (Razão={razao:.2f})",
                "problema": f"Mais de {razao*100:.0f}% das LPPs adquiridas já evoluíram para graus graves. Indica falha na prevenção e no controle da progressão.",
                "fundamentacao": "NPIAP: progressão de Estágio I para III ocorre em média em 48-72h sem intervenção. É sinal de pressão contínua ininterrupta.",
                "acoes": [
                    "<b>Protocolo Anti-Progressão:</b> Criar gatilho automático de revisão do alívio de pressão ao diagnosticar LPP adquirida.",
                    "<b>Inspeção Intensificada:</b> Aumentar a frequência de avaliação de pacientes com LPP adquirida para 2x por turno.",
                    "<b>Registro de Evolução:</b> Tornar obrigatório o registro diário da evolução (melhorando/estável/piorando) de todas as LPPs adquiridas.",
                ]
            }
        elif razao > 0.25:
            return "importante", {
                "titulo": f"Progressão Moderada das Lesões Adquiridas (Razão={razao:.2f})",
                "problema": f"Cerca de {razao*100:.0f}% das LPPs adquiridas atingiram graus graves. A taxa de progressão merece investigação.",
                "fundamentacao": "Avaliação diária documentada é o preditor mais forte de não-progressão para estágios graves.",
                "acoes": [
                    "<b>Auditoria de Prescrições:</b> Revisar se prescrições de curativo para LPPs adquiridas estão sendo atualizadas conforme evolução.",
                ]
            }
        elif razao > 0:
            return "nao_critico", {
                "titulo": f"Baixa Progressão para Lesões Graves (Razão={razao:.2f})",
                "problema": f"Apenas {razao*100:.0f}% das LPPs adquiridas progrediram para estágios graves. Maioria sendo contida em estágios iniciais.",
                "fundamentacao": "Contenção de progressão reflete qualidade do tratamento e monitoramento.",
                "acoes": [
                    "<b>Monitoramento:</b> Manter o protocolo de reavaliação frequente que está evitando a progressão.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma Progressão para Grau Grave entre Lesões Adquiridas",
                "problema": "Nenhuma LPP adquirida evoluiu para graus graves no período. Resultado de excelência clínica.",
                "fundamentacao": "Controle simultâneo de incidência e progressão representa o nível mais elevado de qualidade assistencial em gestão de LPPs.",
                "acoes": [
                    "<b>Reconhecimento:</b> Comunicar o resultado como indicador de excelência no ciclo assistencial completo.",
                ]
            }
    estrategias.append(nivel_razao_grave_adq)

    # 13. Cobertura da Escala de Braden
    def nivel_braden():
        col_b = next((c for c in ['ESCALA BRADEN', 'CLASSIFICAÇÃO BRADEN', 'BRADEN'] if c in df_est.columns), None)
        if col_b:
            n_braden = df_est[col_b].notna().sum()
            # Excluir N/A textuais
            n_braden = df_est[col_b].replace({'N/A': np.nan, 'n/a': np.nan}).notna().sum()
            pct_braden = (n_braden / total) * 100 if total > 0 else 0.0

            if pct_braden < 60.0:
                return "critico", {
                    "titulo": f"Cobertura Crítica da Escala de Braden ({pct_braden:.1f}%)",
                    "problema": f"Apenas {pct_braden:.1f}% dos pacientes com avaliação de risco registrada. Ausência de estratificação impede alocação adequada de recursos preventivos.",
                    "fundamentacao": "Protocolo MS: Escala de Braden deve ser aplicada em 100% dos pacientes nas primeiras 6h da admissão e reavaliada a cada 72h.",
                    "acoes": [
                        "<b>Campo Obrigatório:</b> Solicitar ao sistema hospitalar que o campo Braden seja obrigatório na admissão.",
                        "<b>Auditoria:</b> Incluir cobertura da Escala de Braden como indicador mensal auditado pela CPGLP.",
                        "<b>Treinamento:</b> Capacitar a equipe de enfermagem na aplicação correta e documentação do Braden.",
                    ]
                }
            elif pct_braden < 80.0:
                return "importante", {
                    "titulo": f"Cobertura Parcial da Escala de Braden ({pct_braden:.1f}%)",
                    "problema": f"{pct_braden:.1f}% de cobertura insuficiente para garantir estratificação universal de risco.",
                    "fundamentacao": "Cobertura ≥ 95% da Escala de Braden é requisito para acreditação pela ONA e JCI.",
                    "acoes": [
                        "<b>Meta de Cobertura:</b> Estabelecer meta de 90% de cobertura para o próximo mês com acompanhamento semanal.",
                    ]
                }
            elif pct_braden < 95.0:
                return "nao_critico", {
                    "titulo": f"Boa Cobertura da Escala de Braden ({pct_braden:.1f}%)",
                    "problema": f"Cobertura de {pct_braden:.1f}% próxima da meta de excelência. Pequena lacuna a corrigir.",
                    "fundamentacao": "Cobertura quase universal permite identificação de risco em praticamente todos os pacientes.",
                    "acoes": [
                        "<b>Finalização:</b> Identificar setores com menor cobertura e reforçar a aplicação nos casos faltantes.",
                    ]
                }
            else:
                return "excelente", {
                    "titulo": f"Cobertura Excelente da Escala de Braden ({pct_braden:.1f}%)",
                    "problema": "Estratificação de risco realizada de forma praticamente universal. Excelência na avaliação de risco.",
                    "fundamentacao": "Alta cobertura do Braden é a base para todos os outros processos preventivos de LPP.",
                    "acoes": [
                        "<b>Manutenção:</b> Preservar os processos que garantem essa cobertura e utilizá-la em processos de acreditação.",
                    ]
                }
        else:
            return "nao_critico", {
                "titulo": "Dados da Escala de Braden Não Disponíveis no Dataset",
                "problema": "Coluna de Escala de Braden não encontrada, impossibilitando análise de cobertura de risco.",
                "fundamentacao": "A Escala de Braden é a ferramenta validada mais utilizada mundialmente para estratificação de risco de LPP.",
                "acoes": [
                    "<b>Integração de Dados:</b> Incluir a pontuação da Escala de Braden como campo exportável no sistema de avaliação.",
                ]
            }
    estrategias.append(nivel_braden)

    # 14. LPP em Calcâneo/Talão
    def nivel_calcaneo():
        pct_cal = 0.0
        if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns:
            n_cal = df_est['LOCALIZAÇÃO DA LESÃO'].str.contains('TALÃO|CALCÂNEO|CALCANEO|HEEL', na=False, case=False).sum()
            pct_cal = (n_cal / total) * 100 if total > 0 else 0.0

        if pct_cal > 20.0:
            return "critico", {
                "titulo": f"Alta Incidência de LPP em Calcâneo/Talão ({pct_cal:.1f}%)",
                "problema": f"{pct_cal:.1f}% das lesões no calcâneo. Indica ausência de protocolo de suspensão de calcâneo.",
                "fundamentacao": "NPIAP: o calcâneo é o segundo local mais afetado. A suspensão completa é a única estratégia eficaz — coxins devem ser posicionados sob a panturrilha, não sob o tornozelo.",
                "acoes": [
                    "<b>Protocolo de Suspensão:</b> Implementar protocolo de suspensão de calcâneo com suportes sob a panturrilha.",
                    "<b>Dispositivos:</b> Avaliar aquisição de botas de alívio de pressão para pacientes com Braden ≤ 14.",
                    "<b>Treinamento Visual:</b> Fixar cartaz ilustrativo do posicionamento correto nos postos de enfermagem.",
                ]
            }
        elif pct_cal > 10.0:
            return "importante", {
                "titulo": f"Proporção Elevada de LPP em Calcâneo ({pct_cal:.1f}%)",
                "problema": f"{pct_cal:.1f}% de lesões no calcâneo indica necessidade de reforço preventivo nessa região.",
                "fundamentacao": "Prevenção de LPP de calcâneo requer intervenção ativa; colchões antidecúbito não protegem os calcâneos.",
                "acoes": [
                    "<b>Inspeção Específica:</b> Incluir inspeção obrigatória dos calcâneos no checklist de turno de todos os pacientes acamados.",
                    "<b>Insumos:</b> Verificar disponibilidade de coberturas profiláticas para calcâneo nos setores mais afetados.",
                ]
            }
        elif pct_cal > 0:
            return "nao_critico", {
                "titulo": f"Proporção Moderada de LPP em Calcâneo ({pct_cal:.1f}%)",
                "problema": "Casos de calcâneo dentro do esperado para o perfil de internação.",
                "fundamentacao": "Monitoramento específico de LPP de calcâneo é recomendado por ser a segunda localização mais prevalente.",
                "acoes": [
                    "<b>Monitoramento:</b> Incluir calcâneo como localização de monitoramento prioritário no relatório mensal da CPGLP.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma LPP de Calcâneo Registrada",
                "problema": "Ausência de lesões em calcâneo/talão. Boa adesão ao protocolo de posicionamento.",
                "fundamentacao": "Ausência de LPP de calcâneo é indicador de excelência em posicionamento terapêutico.",
                "acoes": [
                    "<b>Manutenção:</b> Verificar e documentar qual prática de posicionamento está sustentando este resultado.",
                ]
            }
    estrategias.append(nivel_calcaneo)

    # 15. LPP Sacral
    def nivel_sacro():
        pct_sacro = 0.0
        if 'LOCALIZAÇÃO DA LESÃO' in df_est.columns:
            n_sacro = df_est['LOCALIZAÇÃO DA LESÃO'].str.contains('SACRO|SACRAL|CÓCCIX|COCCIX', na=False, case=False).sum()
            pct_sacro = (n_sacro / total) * 100 if total > 0 else 0.0

        if pct_sacro > 30.0:
            return "critico", {
                "titulo": f"Concentração Crítica de LPP na Região Sacral ({pct_sacro:.1f}%)",
                "problema": f"{pct_sacro:.1f}% das lesões na região sacral. Indica que o protocolo de reposicionamento não está sendo seguido.",
                "fundamentacao": "NPIAP: posição dorsal por mais de 2h é o principal fator causal de LPP sacral. Inclinação lateral a 30° é mais eficaz que 90°.",
                "acoes": [
                    "<b>Protocolo de 30°:</b> Implementar rotina de posicionamento em inclinação lateral de 30° com coxin lombar correto.",
                    "<b>Colchões:</b> Priorizar alocação de colchões pneumáticos para pacientes com Braden ≤ 14 em decúbito prolongado.",
                    "<b>Controle Visual:</b> Implementar cronômetro/relógio de reposicionamento a cada 2h nos setores com maior incidência sacral.",
                ]
            }
        elif pct_sacro > 15.0:
            return "importante", {
                "titulo": f"Alta Prevalência de LPP Sacral ({pct_sacro:.1f}%)",
                "problema": f"{pct_sacro:.1f}% de lesões sacrais acima do esperado. Reposicionamento e superfícies de suporte devem ser revisados.",
                "fundamentacao": "Superfícies de suporte de redistribuição de pressão reduzem a incidência de LPP sacral em 60% vs colchões padrão (EPUAP, 2019).",
                "acoes": [
                    "<b>Auditoria de Colchões:</b> Verificar estado de conservação dos colchões nos setores mais afetados e identificar necessidade de substituição.",
                ]
            }
        elif pct_sacro > 0:
            return "nao_critico", {
                "titulo": f"Prevalência Sacral Dentro do Esperado ({pct_sacro:.1f}%)",
                "problema": f"Proporção de LPPs sacrais ({pct_sacro:.1f}%) compatível com o perfil de pacientes acamados atendidos.",
                "fundamentacao": "Monitoramento contínuo da localização sacral é fundamental por ser a região mais vulnerável em decúbito dorsal.",
                "acoes": [
                    "<b>Monitoramento:</b> Manter controle e registrar a evolução das lesões sacrais existentes.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma LPP Sacral Registrada",
                "problema": "Ausência de lesões sacrais. Indica eficácia do protocolo de reposicionamento.",
                "fundamentacao": "Ausência de LPP sacral demonstra que o protocolo de mudança de decúbito está sendo executado com frequência e técnica adequadas.",
                "acoes": [
                    "<b>Documentação:</b> Registrar como indicador de excelência de posicionamento no relatório mensal da CPGLP.",
                ]
            }
    estrategias.append(nivel_sacro)

    # 16. Subnotificação (volume de registros)
    def nivel_subnotificacao():
        if total < 5:
            return "critico", {
                "titulo": f"Possível Subnotificação: Apenas {total} Avaliações no Período",
                "problema": f"Volume de {total} avaliações é muito baixo para um ambiente hospitalar ativo. Alta probabilidade de subnotificação de LPPs.",
                "fundamentacao": "Estudos nacionais: a subnotificação de LPP pode chegar a 70% dos casos reais, especialmente nos Estágios I e II.",
                "acoes": [
                    "<b>Investigação de Campo:</b> Realizar auditoria presencial em todos os setores para verificar LPPs não registradas.",
                    "<b>Cultura de Notificação:</b> Promover campanha reforçando que a notificação não implica responsabilização individual.",
                    "<b>Facilitação:</b> Verificar se o sistema de registro está acessível e de fácil utilização para toda a equipe.",
                ]
            }
        elif total < 15:
            return "importante", {
                "titulo": f"Volume de Avaliações Baixo ({total} registros)",
                "problema": f"{total} registros pode indicar subnotificação parcial ou filtro com escopo muito restrito.",
                "fundamentacao": "Volume adequado de notificações é pré-requisito para que qualquer análise estatística seja representativa.",
                "acoes": [
                    "<b>Verificação de Filtros:</b> Confirmar se os filtros aplicados estão restringindo demais o período ou setores.",
                    "<b>Estímulo à Notificação:</b> Divulgar o painel como ferramenta de apoio, não de fiscalização.",
                ]
            }
        elif total < 30:
            return "nao_critico", {
                "titulo": f"Volume Moderado de Avaliações ({total} registros)",
                "problema": f"{total} avaliações suficiente para análises exploratórias, mas requer cautela na generalização.",
                "fundamentacao": "Análises com n < 30 possuem maior variabilidade estatística; resultados devem ser interpretados com cautela.",
                "acoes": [
                    "<b>Ampliação de Período:</b> Considerar ampliar o período de análise para obter maior volume de dados.",
                ]
            }
        else:
            return "excelente", {
                "titulo": f"Volume Adequado de Avaliações ({total} registros)",
                "problema": "Volume de registros suficiente para análises estatísticas robustas e conclusões representativas.",
                "fundamentacao": "Volume representativo de dados é a base para que as análises do painel gerem insights acionáveis e confiáveis.",
                "acoes": [
                    "<b>Análise Avançada:</b> Com este volume, é possível realizar análises de correlação e predição mais sofisticadas para apoio à decisão.",
                ]
            }
    estrategias.append(nivel_subnotificacao)

    # 17. LPP Relacionada a Dispositivos Médicos (LPPRD)
    def nivel_lpprd():
        pct_lpprd = 0.0
        if 'CLASSIFICAÇÃO' in df_est.columns:
            n_lpprd = df_est['CLASSIFICAÇÃO'].str.contains('DISPOSITIVO|DEVICE|LPPRD', na=False, case=False).sum()
            pct_lpprd = (n_lpprd / total) * 100 if total > 0 else 0.0

        if pct_lpprd > 15.0:
            return "critico", {
                "titulo": f"Alta Taxa de LPP Relacionada a Dispositivos Médicos ({pct_lpprd:.1f}%)",
                "problema": f"{pct_lpprd:.1f}% das lesões associadas a dispositivos médicos. Indica falha no protocolo de interface dispositivo-pele.",
                "fundamentacao": "NPIAP 2019: LPPRD responde por até 30% das LPPs em UTI. Inspeção diária da pele sob dispositivos é obrigatória.",
                "acoes": [
                    "<b>Protocolo LPPRD:</b> Implementar checklist específico para inspeção da pele sob dispositivos em uso.",
                    "<b>Rotação de Dispositivos:</b> Estabelecer protocolo de rotação de fixação de dispositivos (ex: troca de lado da sonda nasoenteral a cada 24h).",
                    "<b>Coberturas Preventivas:</b> Padronizar coberturas finas de silicone sob interfaces de dispositivos de alto risco.",
                ]
            }
        elif pct_lpprd > 5.0:
            return "importante", {
                "titulo": f"Proporção Relevante de LPPRD ({pct_lpprd:.1f}%)",
                "problema": f"{pct_lpprd:.1f}% de lesões relacionadas a dispositivos merecem atenção preventiva específica.",
                "fundamentacao": "Prevenção de LPPRD requer treinamento específico diferente da prevenção de LPP por pressão/cisalhamento.",
                "acoes": [
                    "<b>Treinamento Específico:</b> Capacitar a equipe na prevenção e identificação de LPPRD.",
                ]
            }
        elif pct_lpprd > 0:
            return "nao_critico", {
                "titulo": f"Poucos Casos de LPPRD Identificados ({pct_lpprd:.1f}%)",
                "problema": "Casos isolados de LPPRD. Verificar se há subclassificação desta categoria.",
                "fundamentacao": "Correta identificação e classificação de LPPRD é fundamental para direcionamento de intervenções específicas.",
                "acoes": [
                    "<b>Verificação:</b> Revisar se a equipe está identificando e classificando corretamente as lesões relacionadas a dispositivos.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhuma LPPRD Registrada no Período",
                "problema": "Ausência de lesões relacionadas a dispositivos. Pode refletir excelência preventiva.",
                "fundamentacao": "Monitoramento específico de LPPRD é recomendado pela NPIAP mesmo em períodos de baixa incidência.",
                "acoes": [
                    "<b>Verificação:</b> Confirmar com a equipe se há LPPRD não classificadas nesta categoria.",
                ]
            }
    estrategias.append(nivel_lpprd)

    # 18. Mobilidade dos Pacientes
    def nivel_mobilidade():
        pct_acamado = 0.0
        if 'MOBILIDADE' in df_est.columns:
            n_acamado = df_est['MOBILIDADE'].str.contains('ACAMADO|IMÓVEL|IMOBILIZADO', na=False, case=False).sum()
            pct_acamado = (n_acamado / total) * 100 if total > 0 else 0.0

        if pct_acamado > 40.0:
            return "critico", {
                "titulo": f"Alta Proporção de Pacientes Acamados/Imóveis ({pct_acamado:.1f}%)",
                "problema": f"{pct_acamado:.1f}% dos pacientes avaliados estão em situação de imobilidade total. Representa risco máximo para desenvolvimento de LPP.",
                "fundamentacao": "Imobilidade é o fator de risco mais fortemente associado ao desenvolvimento de LPP. Pacientes acamados têm risco 4-7x maior (NPIAP, 2019).",
                "acoes": [
                    "<b>Superfícies de Suporte:</b> Garantir 100% dos pacientes acamados com Braden ≤ 14 em colchão de redistribuição de pressão.",
                    "<b>Fisioterapia:</b> Acionar programa de mobilização passiva/ativa para todos os pacientes acamados nas primeiras 24h de internação.",
                    "<b>Protocolo de Reposicionamento:</b> Implementar cronograma de reposicionamento a cada 2h com registro obrigatório no prontuário.",
                ]
            }
        elif pct_acamado > 20.0:
            return "importante", {
                "titulo": f"Proporção Relevante de Pacientes Acamados ({pct_acamado:.1f}%)",
                "problema": f"{pct_acamado:.1f}% de pacientes acamados representa carga preventiva significativa para a equipe.",
                "fundamentacao": "Pacientes acamados requerem avaliação de risco mais frequente e intervenções preventivas intensificadas.",
                "acoes": [
                    "<b>Escalonamento Preventivo:</b> Intensificar avaliações de risco para pacientes acamados — mínimo a cada 48h.",
                    "<b>Recursos:</b> Verificar se o quantitativo de colchões preventivos é suficiente para cobrir todos os pacientes acamados de alto risco.",
                ]
            }
        elif pct_acamado > 0:
            return "nao_critico", {
                "titulo": f"Proporção Moderada de Pacientes Acamados ({pct_acamado:.1f}%)",
                "problema": f"{pct_acamado:.1f}% de pacientes acamados dentro do esperado para o perfil assistencial.",
                "fundamentacao": "Monitoramento da mobilidade é variável preditora de risco e deve ser atualizado a cada turno.",
                "acoes": [
                    "<b>Atualização do Braden:</b> Garantir que qualquer mudança no status de mobilidade seja refletida na reavaliação do Braden.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhum Paciente Registrado como Acamado/Imóvel no Período",
                "problema": "Ausência de pacientes acamados ou dados de mobilidade não disponíveis. Perfil de menor risco para LPP.",
                "fundamentacao": "Mobilidade preservada é o principal fator protetor contra o desenvolvimento de LPP.",
                "acoes": [
                    "<b>Monitoramento:</b> Manter registros de mobilidade atualizados para rastreamento prospectivo de risco.",
                ]
            }
    estrategias.append(nivel_mobilidade)

    # 19. Suporte Nutricional
    def nivel_nutricao():
        pct_sne_npt = 0.0
        if 'NUTRIÇÃO' in df_est.columns:
            n_sne = df_est['NUTRIÇÃO'].str.contains('SNE|NPT|PARENTERAL|ENTERAL', na=False, case=False).sum()
            pct_sne_npt = (n_sne / total) * 100 if total > 0 else 0.0

        if pct_sne_npt > 30.0:
            return "critico", {
                "titulo": f"Alta Proporção de Pacientes com Suporte Nutricional Invasivo ({pct_sne_npt:.1f}%)",
                "problema": f"{pct_sne_npt:.1f}% dos pacientes em uso de SNE/NPT/dieta enteral. Pacientes em suporte nutricional invasivo têm maior risco de LPP por desnutrição e imobilidade associada.",
                "fundamentacao": "NPIAP: desnutrição é cofator independente para desenvolvimento de LPP. Pacientes em NPT ou com SNE prolongada apresentam comprometimento da integridade tissular.",
                "acoes": [
                    "<b>Triagem Nutricional:</b> Garantir que 100% dos pacientes com SNE/NPT sejam avaliados pela nutrição em até 48h da admissão.",
                    "<b>Suplementação:</b> Implementar protocolo de suplementação hiperproteica (1,2-1,5g proteína/kg/dia) para pacientes com Braden ≤ 14 em suporte nutricional.",
                    "<b>Monitoramento:</b> Realizar avaliação cutânea reforçada (2x/turno) nos pacientes com suporte nutricional invasivo.",
                ]
            }
        elif pct_sne_npt > 15.0:
            return "importante", {
                "titulo": f"Proporção Relevante de Pacientes com Suporte Enteral/Parenteral ({pct_sne_npt:.1f}%)",
                "problema": f"{pct_sne_npt:.1f}% dos pacientes em suporte nutricional invasivo. Perfil de risco nutricional aumentado para desenvolvimento de LPP.",
                "fundamentacao": "Terapia nutricional adequada reduz o tempo de cicatrização de LPPs em até 35% (ESPEN Guidelines, 2021).",
                "acoes": [
                    "<b>Protocolo Multidisciplinar:</b> Assegurar avaliação conjunta de enfermagem e nutrição para pacientes em SNE com LPP ativa.",
                ]
            }
        elif pct_sne_npt > 0:
            return "nao_critico", {
                "titulo": f"Poucos Pacientes com Suporte Nutricional Invasivo ({pct_sne_npt:.1f}%)",
                "problema": f"{pct_sne_npt:.1f}% dos pacientes em SNE/NPT. Volume controlado que permite atenção individualizada.",
                "fundamentacao": "Monitoramento nutricional individualizado para pacientes em suporte invasivo é recomendado pelo protocolo MS.",
                "acoes": [
                    "<b>Acompanhamento Individual:</b> Assegurar avaliação nutricional individualizada para cada paciente com SNE/NPT.",
                ]
            }
        else:
            return "excelente", {
                "titulo": "Nenhum Paciente com Suporte Nutricional Invasivo no Período",
                "problema": "Ausência de pacientes em SNE/NPT. Perfil nutricional mais favorável para prevenção de LPP.",
                "fundamentacao": "Nutrição oral eficaz é o melhor preditor de integridade tissular e resistência cutânea à pressão.",
                "acoes": [
                    "<b>Monitoramento:</b> Manter triagem nutricional na admissão para detecção precoce de risco.",
                ]
            }
    estrategias.append(nivel_nutricao)

    # 20. Score Global de Segurança Assistencial
    def nivel_score_global():
        score = 100.0
        score -= min(40, (taxa_adquirida / META_QUALIDADE) * 25)
        score -= min(20, taxa_grave * 0.5)
        score -= min(10, pct_nao_estagiavel * 0.3)
        score -= min(10, pct_sem_local * 0.2)
        score -= min(10, pct_setores_criticos * 0.1)
        score -= min(10, gini_setores * 10)
        score = max(0.0, min(100.0, score))

        if score < 50:
            return "critico", {
                "titulo": f"Score de Segurança Assistencial Crítico ({score:.0f}/100)",
                "problema": f"Combinação de múltiplos indicadores desfavoráveis resulta em Score de Segurança de {score:.0f}/100. Representa risco sistêmico para os pacientes.",
                "fundamentacao": "Gestão de Risco Clínico (ISO 31000): plano de ação abrangente é recomendado quando múltiplos indicadores estão comprometidos simultaneamente.",
                "acoes": [
                    "<b>Plano de Crise:</b> Elaborar Plano de Ação de Emergência para LPP com metas a 15, 30 e 90 dias, aprovado pela Direção de Enfermagem.",
                    "<b>Notificação ao NSP:</b> Reportar o cenário ao Núcleo de Segurança do Paciente para acompanhamento gerencial intensivo.",
                    "<b>Revisão Total:</b> Revisar o Protocolo Institucional de Prevenção de LPP à luz das diretrizes NPIAP/EPUAP mais recentes.",
                    "<b>Dashboard:</b> Implementar acompanhamento mensal de todos os indicadores com meta clara de evolução.",
                ]
            }
        elif score < 70:
            return "importante", {
                "titulo": f"Score de Segurança Assistencial em Atenção ({score:.0f}/100)",
                "problema": f"Score de {score:.0f}/100 indica que vários processos estão abaixo do ideal. Ação preventiva coordenada necessária.",
                "fundamentacao": "Ciclo PDCA: a fase de 'verificação' revelou múltiplos pontos de melhoria que devem ser endereçados no próximo ciclo de 'ação'.",
                "acoes": [
                    "<b>Reunião Mensal da CPGLP:</b> Pautar todos os indicadores abaixo da meta com plano de ação específico para cada um.",
                    "<b>Capacitação:</b> Programar treinamento abrangente sobre prevenção de LPP para toda a equipe.",
                ]
            }
        elif score < 85:
            return "nao_critico", {
                "titulo": f"Score de Segurança Assistencial Satisfatório ({score:.0f}/100)",
                "problema": f"Score de {score:.0f}/100 indica bom desempenho geral com pontos de melhoria identificados.",
                "fundamentacao": "Padrão satisfatório deve ser a base para busca da excelência; a melhoria contínua é o pilar do ciclo de acreditação hospitalar.",
                "acoes": [
                    "<b>Melhoria Contínua:</b> Focar nos indicadores específicos que estão contribuindo para o afastamento do score ideal.",
                    "<b>Meta:</b> Estabelecer meta de score ≥ 85 para o próximo trimestre com plano documentado.",
                ]
            }
        else:
            return "excelente", {
                "titulo": f"Score de Segurança Assistencial de Excelência ({score:.0f}/100)",
                "problema": f"Score de {score:.0f}/100 reflete gestão de alto padrão. Os processos assistenciais estão operando em nível de excelência.",
                "fundamentacao": "Excelência em segurança do paciente é reconhecida como indicador central em programas de acreditação ONA Nível III e JCI.",
                "acoes": [
                    "<b>Acreditação:</b> Utilizar os resultados deste painel como evidência para o processo de (re)acreditação hospitalar.",
                    "<b>Publicação Científica:</b> Considerar submissão dos resultados para publicação em revistas de qualidade hospitalar.",
                    "<b>Benchmark Externo:</b> Compartilhar as práticas com a rede hospitalar municipal/estadual como referência.",
                ]
            }
    estrategias.append(nivel_score_global)

    # ── Renderização Final ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Plano de Ação Estratégico Dinâmico")
    st.caption(
        "20 estratégias baseadas em evidências, geradas automaticamente a partir dos indicadores do período/filtro selecionado. "
        "Cada estratégia é classificada dinamicamente: 🚨 Crítico | ⚠️ Importante | ℹ️ Atenção | ✅ Excelente."
    )

    contagem = {"critico": 0, "importante": 0, "nao_critico": 0, "excelente": 0}
    resultados = []
    for fn in estrategias:
        nivel, dados = fn()
        contagem[nivel] += 1
        resultados.append((nivel, dados))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚨 Críticos",    contagem["critico"],     help="Requerem ação imediata")
    col2.metric("⚠️ Importantes", contagem["importante"],  help="Requerem ação planejada")
    col3.metric("ℹ️ Atenção",     contagem["nao_critico"], help="Monitorar")
    col4.metric("✅ Excelentes",  contagem["excelente"],   help="Manter")

    st.markdown("---")

    ordem = {"critico": 0, "importante": 1, "nao_critico": 2, "excelente": 3}
    resultados_ord = sorted(resultados, key=lambda x: ordem[x[0]])

    for nivel_g, label_g, desc_g in [
        ("critico",    "🚨 Alertas Críticos — Ação Imediata Requerida",      "Os itens abaixo requerem intervenção urgente para garantir a segurança do paciente."),
        ("importante", "⚠️ Pontos de Atenção — Ação Planejada Necessária",   "Estes itens estão em zona de risco e devem ser endereçados no curto prazo."),
        ("nao_critico","ℹ️ Monitoramento Contínuo — Dentro dos Limites",     "Indicadores sob controle que requerem vigilância ativa para manutenção."),
        ("excelente",  "✅ Indicadores de Excelência — Manter e Reconhecer", "Resultados positivos que devem ser reconhecidos e sustentados."),
    ]:
        grupo = [(n, d) for n, d in resultados_ord if n == nivel_g]
        if grupo:
            with st.expander(f"{label_g} ({len(grupo)} itens)", expanded=(nivel_g in ["critico", "importante"])):
                st.caption(desc_g)
                for nivel, dados in grupo:
                    card(
                        nivel=nivel,
                        titulo=dados["titulo"],
                        problema=dados["problema"],
                        fundamentacao=dados["fundamentacao"],
                        acoes=dados["acoes"]
                    )