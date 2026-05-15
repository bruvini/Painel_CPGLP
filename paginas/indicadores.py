import streamlit as st
import pandas as pd
import plotly.express as px

def render_indicadores(df):
    if df.empty:
        st.warning("⚠️ Não existem dados para os filtros selecionados.")
        return

    # --- CÁLCULOS DE KPIs ---
    t = len(df)
    
    # Filtros lógicos
    lpp_df = df[df['TIPO DE LESÃO'].str.contains('PRESSÃO', na=False, case=False)]
    que_df = df[df['TIPO DE LESÃO'].str.contains('QUEIMADURA', na=False, case=False)]
    
    # KPIs específicos
    lpp_count = len(lpp_df)
    adq_count = len(df[df['CLASSIFICAÇÃO'] == 'ADQUIRIDA NA INTERNAÇÃO ATUAL'])
    setores_ativos = df['SETORES'].nunique()
    que_count = len(que_df)
    est_count = len(df[(df['TIPOS ESTOMIAS'].notna()) & (df['TIPOS ESTOMIAS'] != 'Não Informado')])
    altas = len(df[df['AVALIAÇÃO'] == 'ALTA HOSPITALAR'])
    antibio = len(df[df['ANTIBIÓTICO TERAPIA SISTÊMICO'] == 'SIM'])

    # --- SECÇÃO: VISÃO GERAL (KPIs) ---
    st.subheader("📊 Visão Geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Registos", t)
    c2.metric("LPP", lpp_count, f"{round((lpp_count/t)*100,1)}% do total" if t > 0 else None)
    c3.metric("LPP Adquiridas", adq_count, delta_color="inverse")
    c4.metric("Setores Ativos", setores_ativos)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Queimaduras", que_count)
    c6.metric("Estomias", est_count)
    c7.metric("Altas Hospitalares", altas)
    c8.metric("Com Antibiótico", antibio)

    st.markdown("---")

    # --- SECÇÃO: LESÕES POR PRESSÃO (LPP) ---
    st.subheader("🩹 Lesões por Pressão")
    
    col_lpp1, col_lpp2 = st.columns(2)
    
    with col_lpp1:
        if 'LESÃO POR PRESSÃO -  ESTAGIO' in df.columns:
            st.markdown("**Estadiamento LPP**")
            estagio_data = df['LESÃO POR PRESSÃO -  ESTAGIO'].value_counts().reset_index()
            estagio_data.columns = ['Categoria', 'Contagem'] # Padronizando nomes das colunas
            fig_estagio = px.bar(estagio_data, x='Contagem', y='Categoria', orientation='h',
                                 color_discrete_sequence=['#c0392b'], text_auto=True)
            fig_estagio.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_estagio, width="stretch")

    with col_lpp2:
        if 'CLASSIFICAÇÃO' in df.columns:
            st.markdown("**Classificação**")
            class_data = df['CLASSIFICAÇÃO'].value_counts().reset_index()
            class_data.columns = ['Categoria', 'Contagem']
            fig_class = px.pie(class_data, values='Contagem', names='Categoria', hole=0.5,
                               color_discrete_sequence=['#c87941', '#2d6fa3', '#2d7a4f'])
            fig_class.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_class, width="stretch")

    col_lpp3, col_lpp4 = st.columns(2)
    
    with col_lpp3:
        if 'ESCALA BRADEN' in df.columns:
            st.markdown("**Escala Braden — Risco**")
            braden_data = df['ESCALA BRADEN'].value_counts().reset_index()
            braden_data.columns = ['Categoria', 'Contagem']
            fig_braden = px.bar(braden_data, x='Contagem', y='Categoria', orientation='h',
                                color_discrete_sequence=['#2d6fa3'], text_auto=True)
            st.plotly_chart(fig_braden, width="stretch")

    with col_lpp4:
        if 'LOCALIZAÇÃO DA LESÃO' in df.columns:
            st.markdown("**Localização da Lesão**")
            loc_data = df['LOCALIZAÇÃO DA LESÃO'].value_counts().head(10).reset_index()
            loc_data.columns = ['Categoria', 'Contagem']
            fig_loc = px.bar(loc_data, x='Contagem', y='Categoria', orientation='h',
                             color_discrete_sequence=['#c0392b'], text_auto=True)
            fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_loc, width="stretch")

    st.markdown("---")

    # --- SECÇÃO: TIPO DE LESÃO & CLÍNICA ---
    st.subheader("🩺 Tipo de Lesão & Clínica")
    
    col_clin1, col_clin2 = st.columns(2)
    
    with col_clin1:
        if 'TIPO DE LESÃO' in df.columns:
            st.markdown("**Tipo de Lesão**")
            tipo_data = df['TIPO DE LESÃO'].value_counts().reset_index()
            tipo_data.columns = ['Categoria', 'Contagem']
            fig_tipo = px.bar(tipo_data, x='Contagem', y='Categoria', orientation='h',
                              color_discrete_sequence=['#c87941'], text_auto=True)
            st.plotly_chart(fig_tipo, width="stretch")

    with col_clin2:
        if 'DESBRIDAMENTO ' in df.columns:
            st.markdown("**Desbridamento**")
            desb_data = df['DESBRIDAMENTO '].value_counts().reset_index()
            desb_data.columns = ['Categoria', 'Contagem']
            fig_desb = px.bar(desb_data, x='Contagem', y='Categoria', orientation='h',
                              color_discrete_sequence=['#2d7a4f'], text_auto=True)
            st.plotly_chart(fig_desb, width="stretch")

    # --- SECÇÃO: SETORES & DESFECHO ---
    st.divider()
    st.subheader("🏢 Setores & Desfecho")
    
    col_set1, col_set2 = st.columns([2, 1])
    
    with col_set1:
        st.markdown("**Atendimentos por Setor**")
        setor_data = df['SETORES'].value_counts().reset_index()
        setor_data.columns = ['Categoria', 'Contagem']
        fig_setor = px.bar(setor_data, x='Contagem', y='Categoria', orientation='h',
                           color_discrete_sequence=['#2d7a4f'], text_auto=True)
        fig_setor.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_setor, width="stretch")
        
    with col_set2:
        st.markdown("**Desfecho Clínico**")
        desf_data = df['AVALIAÇÃO'].value_counts().reset_index()
        desf_data.columns = ['Categoria', 'Contagem']
        fig_desf = px.pie(desf_data, values='Contagem', names='Categoria', hole=0.5,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_desf, width="stretch")