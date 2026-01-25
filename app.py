import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime
from utils.ifdata_extractor import gerar_periodos, processar_todos_periodos, carregar_cores_aliases

st.set_page_config(page_title="Fica de Olho", page_icon="👁️", layout="wide")

CACHE_FILE = "data/dados_cache.pkl"
CACHE_INFO = "data/cache_info.txt"
ALIASES_PATH = "data/Aliases.xlsx"

def salvar_cache(dados_periodos, periodo_info):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(dados_periodos, f)
    with open(CACHE_INFO, 'w') as f:
        f.write(f"Última extração: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"Períodos: {periodo_info}\n")
        f.write(f"Total de períodos: {len(dados_periodos)}\n")

def carregar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def ler_info_cache():
    if os.path.exists(CACHE_INFO):
        with open(CACHE_INFO, 'r') as f:
            return f.read()
    return None

def carregar_aliases():
    if os.path.exists(ALIASES_PATH):
        return pd.read_excel(ALIASES_PATH)
    return None

st.markdown('<h1 style="text-align: center; color: #003366;">👁️ Fica de Olho</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Dashboard de Análise de Instituições Financeiras</p>', unsafe_allow_html=True)

# CARREGAR ALIASES AUTOMATICAMENTE (UMA VEZ)
if 'df_aliases' not in st.session_state:
    df_aliases = carregar_aliases()
    if df_aliases is not None:
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]

# CARREGAR CACHE AUTOMATICAMENTE (UMA VEZ)
if 'dados_periodos' not in st.session_state:
    dados_cache = carregar_cache()
    if dados_cache:
        st.session_state['dados_periodos'] = dados_cache

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Mostrar status dos aliases
    if 'df_aliases' in st.session_state:
        st.success(f"✅ {len(st.session_state['df_aliases'])} aliases carregados")
    else:
        st.warning("⚠️ Aliases não encontrados")
    
    # Mostrar status do cache
    info_cache = ler_info_cache()
    if info_cache:
        with st.expander("💾 Dados em Cache"):
            st.text(info_cache)
            if st.button("🗑️ Limpar Cache", type="secondary", use_container_width=True):
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                if os.path.exists(CACHE_INFO):
                    os.remove(CACHE_INFO)
                if 'dados_periodos' in st.session_state:
                    del st.session_state['dados_periodos']
                st.success("Cache limpo!")
                st.rerun()
    
    st.divider()
    
    # Upload manual opcional
    uploaded_file = st.file_uploader("📤 Substituir Aliases (opcional)", type=['xlsx'])
    
    if uploaded_file:
        df_aliases = pd.read_excel(uploaded_file)
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]
        st.success(f"✅ {len(df_aliases)} aliases carregados do upload")
    
    st.divider()
    
    # Seção de extração
    st.subheader("📅 Atualizar Dados")
    
    col1, col2 = st.columns(2)
    with col1:
        ano_i = st.selectbox("Ano Inicial", range(2015,2027), index=8)
        mes_i = st.selectbox("Trim. Inicial", ['03','06','09','12'])
    with col2:
        ano_f = st.selectbox("Ano Final", range(2015,2027), index=10)
        mes_f = st.selectbox("Trim. Final", ['03','06','09','12'], index=2)
    
    if 'dict_aliases' in st.session_state:
        if st.button("🚀 Extrair Novos Dados", type="primary", use_container_width=True):
            periodos = gerar_periodos(ano_i, mes_i, ano_f, mes_f)
            progress_bar = st.progress(0)
            status = st.empty()
            
            def update(i, total, p):
                progress_bar.progress((i+1)/total)
                status.text(f"{p[4:6]}/{p[:4]} ({i+1}/{total})")
            
            dados = processar_todos_periodos(periodos, st.session_state['dict_aliases'], update)
            st.session_state['dados_periodos'] = dados
            
            periodo_info = f"{periodos[0][4:6]}/{periodos[0][:4]} até {periodos[-1][4:6]}/{periodos[-1][:4]}"
            salvar_cache(dados, periodo_info)
            
            progress_bar.empty()
            status.empty()
            st.success(f"✅ {len(dados)} períodos extraídos e salvos!")
            st.rerun()
    else:
        st.warning("⚠️ Carregue os aliases primeiro")

if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
    df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
    st.success("✅ Dados carregados! Use o menu lateral ← para acessar os dashboards")
    
    st.subheader("📊 Visão Geral")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Instituições", df['Instituição'].nunique())
    c2.metric("Carteira Total", f"R$ {df['Carteira de Crédito'].sum()/1e9:.1f}B")
    c3.metric("ROE Médio", f"{df['ROE An. (%)'].mean()*100:.1f}%")
    c4.metric("Basileia Média", f"{df['Índice de Basileia'].mean():.1f}%")
    
    st.subheader("📋 TOP 10 Bancos")
    ultimo = max(st.session_state['dados_periodos'].keys())
    top = st.session_state['dados_periodos'][ultimo].nlargest(10, 'Carteira de Crédito')[['Instituição','Carteira de Crédito','ROE An. (%)','Alavancagem']].copy()
    top['Carteira de Crédito'] = top['Carteira de Crédito'].apply(lambda x: f"R$ {x/1e9:.2f}B")
    top['ROE An. (%)'] = top['ROE An. (%)'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    top['Alavancagem'] = top['Alavancagem'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
    st.dataframe(top, use_container_width=True, hide_index=True)
    
    periodos_disponiveis = sorted(df['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
    st.info(f"📅 Períodos disponíveis: {periodos_disponiveis[0]} até {periodos_disponiveis[-1]} ({len(periodos_disponiveis)} trimestres)")
else:
    st.markdown("---")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        ### 📊 Sobre o Fica de Olho
        
        O **Fica de Olho** é uma ferramenta de análise financeira que extrai, processa e visualiza dados 
        de instituições financeiras brasileiras de forma automatizada e interativa.
        
        #### 🎯 Funcionalidades
        
        - **Extração Automatizada**: Integração direta com a API IF.data do Banco Central do Brasil
        - **Análise Temporal**: Acompanhamento de métricas financeiras ao longo de múltiplos trimestres
        - **Visualização Interativa**: Gráficos de dispersão customizáveis com filtros dinâmicos
        - **Classificação Personalizada**: Sistema de aliases para renomear e categorizar instituições
        - **Métricas Calculadas**: ROE anualizado, alavancagem, funding gap, market share e índices de risco/retorno
        
        #### 📈 Dados Utilizados
        
        Todos os dados são extraídos da **API IF.data** do Banco Central do Brasil, incluindo:
        
        - Carteira de Crédito Classificada
        - Patrimônio Líquido e Lucro Líquido
        - Índice de Basileia
        - Captações e Ativo Total
        - Cadastro de Instituições Financeiras
        
        #### 🚀 Como Começar
        
        1. Configure o período de análise na barra lateral
        2. Clique em **"Extrair Novos Dados"** para buscar informações do BCB
        3. Acesse os dashboards no menu lateral após a extração
        4. Personalize visualizações e exporte análises
        """)
    
    with col2:
        st.info("""
        ### 💡 Primeira Vez?
        
        **Passo 1:** Configure as datas na barra lateral ←
        
        **Passo 2:** Clique em "🚀 Extrair Novos Dados"
        
        **Passo 3:** Aguarde o processamento (30-60 segundos)
        
        **Passo 4:** Explore os dashboards!
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Recursos Técnicos
        
        - **Python 3.10+**
        - **Streamlit** (interface)
        - **Pandas** (processamento)
        - **Plotly** (visualizações)
        - **API BCB Olinda**
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>
        Desenvolvido em 2026 por <strong>Matheus Prates, CFA</strong><br>
        Ferramenta de código aberto para análise do sistema financeiro brasileiro
    </div>
    """, unsafe_allow_html=True)
