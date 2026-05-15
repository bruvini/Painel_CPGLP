import streamlit as st
import pandas as pd

# ==========================================
# IMPORTAÇÃO DOS MÓDULOS (PÁGINAS)
# ==========================================
from paginas.sobre import render_sobre
from paginas.indicadores import render_indicadores
from paginas.temporal import render_temporal
from paginas.financeiro import render_financeiro
from paginas.estrategica import render_estrategica

# ==========================================
# 1. CONFIGURAÇÃO GERAL DA PÁGINA
# ==========================================
# IMPORTANTE: st.set_page_config deve ser sempre o primeiro comando do Streamlit
st.set_page_config(
    page_title="CPGLP · Painel de Indicadores",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo global customizado
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. UPLOAD E PROCESSAMENTO DE DADOS
# ==========================================
st.sidebar.title("🏥 CPGLP · HMSJ")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Carregar exportação do Forms (.csv)", type=['csv'])

@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file, dtype=str)
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df = df.fillna('Não Informado')
            return df
        except Exception as e:
            st.sidebar.error(f"Erro ao ler o arquivo: {e}")
            return None
    return None

df_raw = load_data(uploaded_file)

# Bloqueia o app aqui se não houver arquivo carregado, exibindo a tela "Sobre" como padrão inicial
if df_raw is None:
    st.info("👈 Por favor, faça o upload do arquivo CSV na barra lateral para liberar as ferramentas de análise.")
    render_sobre()
    st.stop()

# ==========================================
# 3. FILTROS GLOBAIS (SIDEBAR)
# ==========================================
st.sidebar.subheader("Filtros Ativos")

def apply_filters(df):
    df_filtered = df.copy()
    cols_to_filter = ['ANO', 'MÊS', 'SETORES', 'TIPO DE LESÃO', 'AVALIAÇÃO', 'CLASSIFICAÇÃO']
    
    for col in cols_to_filter:
        if col in df.columns:
            options = ['Todos'] + sorted(list(df[col].unique()))
            selected = st.sidebar.selectbox(f"{col.title()}", options, index=0)
            if selected != 'Todos':
                df_filtered = df_filtered[df_filtered[col] == selected]
                
    return df_filtered

df_filtrado = apply_filters(df_raw)
st.sidebar.markdown(f"**Registros filtrados: {len(df_filtrado)} / {len(df_raw)}**")

# ==========================================
# 4. ROTEADOR DE ABAS (NAVEGAÇÃO)
# ==========================================
st.title("Painel de Indicadores Estratégicos")

tab_ind, tab_temp, tab_fin, tab_est, tab_sobre = st.tabs([
    "📊 Indicadores", 
    "📈 Série Temporal", 
    "💰 Impacto Financeiro", 
    "💡 Análise Estratégica", 
    "ℹ️ Sobre"
])

# Renderização condicional dentro de cada aba
with tab_ind:
    render_indicadores(df_filtrado)

with tab_temp:
    render_temporal(df_filtrado)

with tab_fin:
    render_financeiro(df_filtrado)

with tab_est:
    render_estrategica(df_filtrado)

with tab_sobre:
    render_sobre() # Esta página não precisa receber os dados