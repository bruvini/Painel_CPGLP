import streamlit as st

def render_sobre():
    # Estilo customizado para os cartões de etapas e ícones
    st.markdown("""
        <style>
        .hero-container {
            padding: 2rem;
            border-radius: 1rem;
            background-color: #fdf0ee;
            border: 1px solid #c0392b;
            margin-bottom: 2rem;
        }
        .step-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e2ddd6;
            height: 100%;
        }
        .step-number {
            font-size: 2rem;
            color: #c0392b;
            opacity: 0.4;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HERO SECTION ---
    st.markdown(f"""
        <div class="hero-container">
            <h1 style='color: #1a1714; margin-bottom: 0;'>Painel de Indicadores CPGLP</h1>
            <h3 style='color: #c0392b; font-style: italic; margin-top: 0;'>Hospital Municipal São José</h3>
            <p style='color: #4a4540; font-size: 1.1rem; margin-top: 1rem;'>
                Sistema de monitoramento e análise estratégica para a <b>Comissão de Prevenção e Gerenciamento de Lesões de Pele (CPGLP)</b>. 
                Desenvolvido para apoiar a tomada de decisão baseada em dados e a melhoria contínua da qualidade assistencial em Joinville/SC.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- LGPD & SEGURANÇA ---
    st.info("""
        **🛡️ Conformidade LGPD — Lei Geral de Proteção de Dados** Este painel opera inteiramente no ambiente local do servidor/navegador. Nenhum dado é transmitido, armazenado em servidores externos ou compartilhado com terceiros. 
        Os arquivos CSV são lidos em memória e descartados ao encerrar a sessão. Não há coleta de dados pessoais identificáveis de pacientes.
    """)

    # --- COMO UTILIZAR ---
    st.subheader("🚀 Como utilizar o painel")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""<div class="step-card"><span class="step-number">01</span><br><b>Google Forms</b><br>Acesse as respostas do formulário CPGLP no Google Sheets.</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="step-card"><span class="step-number">02</span><br><b>Download CSV</b><br>Vá em Arquivo > Fazer download > Valores separados por vírgula (.csv).</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="step-card"><span class="step-number">03</span><br><b>Upload</b><br>Carregue o arquivo na barra lateral deste painel para processar os dados.</div>""", unsafe_allow_html=True)

    st.write("") # Espaçador

    # --- INDICADORES ---
    with st.expander("📊 Conheça os Indicadores Monitorados", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Lesões por Pressão (LPP)**
            - Estadiamento (I a IV e Não Estadiável)
            - Escalas Braden (Risco) e BWAT (Gravidade)
            - Classificação (Admissão vs. Adquirida)
            - Localização Anatômica e Mensuração
            """)
        with c2:
            st.markdown("""
            **Estomias e Outros**
            - Tipos de Estomia (Traqueo, Gastro, Colostomia, etc.)
            - Avaliação de Tecido e Mucosa
            - Queimaduras (Profundidade e Complexidade)
            - Desfecho Clínico e Antibioticoterapia
            """)

    # --- CRÉDITOS E CONTATO ---
    st.divider()
    footer_col1, footer_col2 = st.columns([2, 1])
    
    with footer_col1:
        st.markdown(f"""
            **Desenvolvido por:** **Enf. Bruno Vinícius da Silva** *Analista de Dados e Processos — NIR/HMSJ* Hospital Municipal São José · Joinville/SC
        """)
    
    with footer_col2:
        st.markdown("""
            **Contato:** [bruno.vinicius@joinville.sc.gov.br](mailto:bruno.vinicius@joinville.sc.gov.br)
        """)

    st.caption("Aviso: Este é um instrumento de apoio à gestão. As análises devem ser validadas pelo responsável técnico e não substituem a avaliação clínica individual.")