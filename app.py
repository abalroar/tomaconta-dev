import streamlit as st
import pandas as pd
import pickle
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from utils.ifdata_extractor import gerar_periodos, processar_todos_periodos, carregar_cores_aliases

st.set_page_config(page_title="Fica de Olho", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS customizado com fonte Source Sans 3 e design limpo
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700;800&display=swap');
    
    /* Aplicar Source Sans 3 em TODOS os elementos */
    * {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    html, body, [class*="css"], div, span, p, label, input, select, textarea, button {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Headers e títulos */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar clean */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
    }
    
    [data-testid="stSidebar"] * {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Radio buttons estilo Perplexity */
    .stRadio > label {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #333 !important;
    }
    
    /* Botões */
    button[kind="primary"], button[kind="secondary"], .stButton button {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Selectbox, inputs */
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Métricas */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Markdown */
    .stMarkdown, .stMarkdown * {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        background-color: #f5f5f5 !important;
    }
    
    /* Captions */
    .stCaption {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Container do logo */
    .logo-container {
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
    }
    
    .logo-container img {
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Header principal customizado */
    .main-header {
        font-size: 6rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.1;
        font-family: 'Source Sans 3', sans-serif !important;
        letter-spacing: -0.03em;
    }
    
    .sub-header {
        font-size: 2rem;
        color: #666;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    .by-line {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 300;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Source Sans 3', sans-serif !important;
    }
    
    /* Info card na sidebar */
    .info-card {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    
    .info-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

CACHE_FILE = "data/dados_cache.pkl"
CACHE_INFO = "data/cache_info.txt"
ALIASES_PATH = "data/Aliases.xlsx"
LOGO_PATH = "data/logo.jpg"
CACHE_URL = "https://github.com/abalroar/ficadeolho/releases/download/v1.0-cache/dados_cache.pkl"
CACHE_INFO_URL = "https://github.com/abalroar/ficadeolho/releases/download/v1.0-cache/cache_info.txt"

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

def baixar_cache_inicial():
    """Baixa cache do GitHub Releases se não existir localmente"""
    cache_path = Path(CACHE_FILE)
    
    if not cache_path.exists():
        try:
            with st.spinner("Carregando dados do GitHub (10MB)..."):
                r = requests.get(CACHE_URL, timeout=120)
                if r.status_code == 200:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(r.content)
                    
                    r_info = requests.get(CACHE_INFO_URL, timeout=30)
                    if r_info.status_code == 200:
                        Path(CACHE_INFO).write_text(r_info.text)
                    
                    return True
                else:
                    st.warning(f"Cache não encontrado (HTTP {r.status_code})")
                    return False
        except Exception as e:
            st.error(f"Erro ao baixar cache: {e}")
            return False
    return True

def formatar_valor(valor, variavel):
    """Formata valor de acordo com o tipo de variável"""
    if pd.isna(valor):
        return "N/A"
    
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_razao = ['Alavancagem', 'Risco/Retorno']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']
    
    if variavel in vars_percentual:
        return f"{valor*100:.2f}%"
    elif variavel in vars_razao:
        return f"{valor:.2f}x"
    elif variavel in vars_monetarias:
        valor_mm = valor / 1e6
        return f"R$ {valor_mm:,.0f}MM".replace(",", ".")
    else:
        return f"{valor:.2f}"

def get_axis_format(variavel):
    """Retorna formato e multiplicador para eixos do gráfico"""
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']
    
    if variavel in vars_percentual:
        return {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
    elif variavel in vars_monetarias:
        return {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
    else:
        return {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}

def criar_mini_grafico(df_banco, variavel, titulo):
    """Cria mini gráfico para uma variável específica"""
    # Ordenar cronologicamente por ano e trimestre
    df_sorted = df_banco.copy()
    df_sorted['ano'] = df_sorted['Período'].str.split('/').str[1].astype(int)
    df_sorted['trimestre'] = df_sorted['Período'].str.split('/').str[0].astype(int)
    df_sorted = df_sorted.sort_values(['ano', 'trimestre'])
    
    # Determinar tipo de formatação
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']
    
    # Criar valores formatados para hover
    if variavel in vars_percentual:
        hover_values = df_sorted[variavel] * 100
        tickformat = '.2f'
        suffix = '%'
    elif variavel in vars_monetarias:
        hover_values = df_sorted[variavel] / 1e6
        tickformat = ',.0f'
        suffix = 'M'
    else:
        hover_values = df_sorted[variavel]
        tickformat = '.2f'
        suffix = ''
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar linha
    fig.add_trace(go.Scatter(
        x=df_sorted['Período'],
        y=hover_values,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='%{x}<br>%{y:' + tickformat + '}' + suffix + '<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=12, color='#333', family='Source Sans 3')),
        height=180,
        margin=dict(l=10, r=10, t=35, b=30),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#e0e0e0',
            tickformat=tickformat,
            ticksuffix=suffix
        ),
        hovermode='x',
        font=dict(family='Source Sans 3')
    )
    
    return fig

# Header com Logo ACIMA do título
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(LOGO_PATH, use_column_width=True)
else:
    st.markdown('<p style="font-size: 150px; text-align: center;">👁️</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Títulos centralizados
st.markdown('<p class="main-header">Fica de Olho</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dashboard de Análise de Instituições Financeiras Brasileiras</p>', unsafe_allow_html=True)
st.markdown('<p class="by-line">by Matheus Prates, CFA</p>', unsafe_allow_html=True)

# CARREGAR ALIASES AUTOMATICAMENTE
if 'df_aliases' not in st.session_state:
    df_aliases = carregar_aliases()
    if df_aliases is not None:
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]

# CARREGAR CACHE COM DOWNLOAD DO GITHUB
if 'dados_periodos' not in st.session_state:
    baixar_cache_inicial()
    dados_cache = carregar_cache()
    if dados_cache:
        st.session_state['dados_periodos'] = dados_cache

# SIDEBAR - ESTILO PERPLEXITY
with st.sidebar:
    st.title("Menu")
    
    # Menu com radio buttons (estilo Perplexity - clean e minimalista)
    menu = st.radio(
        "Navegação",
        ["📘 Sobre", "🏦 Análise Individual", "📊 Scatter Plot"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Status compacto
    if 'df_aliases' in st.session_state:
        st.markdown(f'<div class="info-card"><p><strong>{len(st.session_state["df_aliases"])} aliases</strong> carregados</p></div>', unsafe_allow_html=True)
    else:
        st.error("❌ Aliases não encontrados")
    
    # Cache info em expander
    info_cache = ler_info_cache()
    if info_cache:
        with st.expander("📦 Cache"):
            st.text(info_cache)
            if st.button("Limpar Cache", use_container_width=True):
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                if os.path.exists(CACHE_INFO):
                    os.remove(CACHE_INFO)
                if 'dados_periodos' in st.session_state:
                    del st.session_state['dados_periodos']
                st.rerun()
    
    st.markdown("---")
    
    # Extração de dados
    with st.expander("🔄 Atualizar Dados"):
        col1, col2 = st.columns(2)
        with col1:
            ano_i = st.selectbox("Ano Inicial", range(2015,2027), index=8, key="ano_i")
            mes_i = st.selectbox("Trim. Inicial", ['03','06','09','12'], key="mes_i")
        with col2:
            ano_f = st.selectbox("Ano Final", range(2015,2027), index=10, key="ano_f")
            mes_f = st.selectbox("Trim. Final", ['03','06','09','12'], index=2, key="mes_f")
        
        if 'dict_aliases' in st.session_state:
            if st.button("Extrair Dados", type="primary", use_container_width=True):
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
                st.success(f"✅ {len(dados)} períodos extraídos")
                st.rerun()
        else:
            st.warning("Carregue os aliases primeiro")

# CONTEÚDO PRINCIPAL
if menu == "📘 Sobre":
    st.markdown("---")
    
    st.markdown("""
    ### Sobre o Fica de Olho
    
    O **Fica de Olho** é uma ferramenta de análise financeira que extrai, processa e visualiza dados 
    de instituições financeiras brasileiras de forma automatizada e interativa.
    
    #### Funcionalidades
    
    - **Extração Automatizada**: Integração direta com a API IF.data do Banco Central do Brasil
    - **Análise Temporal**: Acompanhamento de métricas financeiras ao longo de múltiplos trimestres
    - **Visualização Interativa**: Gráficos de dispersão customizáveis com filtros dinâmicos
    - **Classificação Personalizada**: Sistema de aliases para renomear e categorizar instituições
    - **Métricas Calculadas**: ROE anualizado, alavancagem, funding gap, market share e índices de risco/retorno
    
    #### Dados Utilizados
    
    Todos os dados são extraídos da **API IF.data** do Banco Central do Brasil, incluindo:
    
    - Carteira de Crédito Classificada
    - Patrimônio Líquido e Lucro Líquido
    - Índice de Basileia
    - Captações e Ativo Total
    - Cadastro de Instituições Financeiras
    
    #### Como Começar
    
    1. Os dados já estão carregados automaticamente do GitHub
    2. Acesse **Análise Individual** ou **Scatter Plot** no menu lateral
    3. Para atualizar dados, configure período e clique em "Extrair Dados"
    4. Personalize visualizações usando os filtros disponíveis
    
    ---
    
    ### Recursos Técnicos
    
    - **Python 3.10+**
    - **Streamlit** (interface)
    - **Pandas** (processamento)
    - **Plotly** (visualizações)
    - **API BCB Olinda**
    """)
    
    st.markdown("---")
    st.markdown("**Desenvolvido em 2026 por Matheus Prates, CFA** | *Ferramenta de código aberto para análise do sistema financeiro brasileiro*")

elif menu == "🏦 Análise Individual":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
        
        if len(df) > 0 and 'Instituição' in df.columns:
            bancos_todos = df['Instituição'].dropna().unique().tolist()
            
            if 'dict_aliases' in st.session_state and st.session_state['dict_aliases']:
                bancos_com_alias = []
                bancos_sem_alias = []
                
                for banco in bancos_todos:
                    if banco in st.session_state['dict_aliases']:
                        alias = st.session_state['dict_aliases'][banco]
                        bancos_com_alias.append((alias, banco))
                    else:
                        bancos_sem_alias.append(banco)
                
                bancos_com_alias_sorted = [banco for alias, banco in sorted(bancos_com_alias)]
                bancos_sem_alias_sorted = sorted(bancos_sem_alias)
                bancos_disponiveis = bancos_com_alias_sorted + bancos_sem_alias_sorted
            else:
                bancos_disponiveis = sorted(bancos_todos)
            
            if len(bancos_disponiveis) > 0:
                banco_selecionado = st.selectbox("Selecione uma Instituição", bancos_disponiveis, key="banco_individual")
                
                if banco_selecionado:
                    df_banco = df[df['Instituição'] == banco_selecionado].copy()
                    
                    df_banco['ano'] = df_banco['Período'].str.split('/').str[1].astype(int)
                    df_banco['trimestre'] = df_banco['Período'].str.split('/').str[0].astype(int)
                    df_banco = df_banco.sort_values(['ano', 'trimestre'])
                    
                    st.markdown(f"## {banco_selecionado}")
                    
                    ultimo_periodo = df_banco['Período'].iloc[-1]
                    dados_ultimo = df_banco[df_banco['Período'] == ultimo_periodo].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Carteira de Crédito", formatar_valor(dados_ultimo.get('Carteira de Crédito'), 'Carteira de Crédito'))
                    with col2:
                        st.metric("ROE Anualizado", formatar_valor(dados_ultimo.get('ROE An. (%)'), 'ROE An. (%)'))
                    with col3:
                        st.metric("Índice de Basileia", formatar_valor(dados_ultimo.get('Índice de Basileia'), 'Índice de Basileia'))
                    with col4:
                        st.metric("Alavancagem", formatar_valor(dados_ultimo.get('Alavancagem'), 'Alavancagem'))
                    
                    st.markdown("---")
                    st.markdown("### Evolução Histórica das Variáveis")
                    
                    variaveis = [col for col in df_banco.columns if col not in ['Instituição', 'Período', 'ano', 'trimestre'] and df_banco[col].notna().any()]
                    
                    for i in range(0, len(variaveis), 3):
                        cols = st.columns(3)
                        for j, col_obj in enumerate(cols):
                            if i + j < len(variaveis):
                                var = variaveis[i + j]
                                with col_obj:
                                    fig = criar_mini_grafico(df_banco, var, var)
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Nenhuma instituição encontrada nos dados")
        else:
            st.warning("Dados incompletos ou vazios")
    else:
        st.info("Carregando dados automaticamente do GitHub...")
        st.markdown("Por favor, aguarde alguns segundos e recarregue a página")

elif menu == "📊 Scatter Plot":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
        
        colunas_numericas = [col for col in df.columns if col not in ['Instituição', 'Período'] and df[col].dtype in ['float64', 'int64']]
        periodos = sorted(df['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            var_x = st.selectbox("Eixo X", colunas_numericas, index=colunas_numericas.index('Índice de Basileia') if 'Índice de Basileia' in colunas_numericas else 0)
        with col2:
            var_y = st.selectbox("Eixo Y", colunas_numericas, index=colunas_numericas.index('ROE An. (%)') if 'ROE An. (%)' in colunas_numericas else 1)
        with col3:
            var_size = st.selectbox("Tamanho", colunas_numericas, index=colunas_numericas.index('Carteira de Crédito') if 'Carteira de Crédito' in colunas_numericas else 0)
        with col4:
            periodo_scatter = st.selectbox("Período", periodos, index=len(periodos)-1)
        with col5:
            top_n_scatter = st.slider("TOP N", 5, 50, 15)
        
        df_scatter = df[df['Período'] == periodo_scatter].nlargest(top_n_scatter, 'Carteira de Crédito')
        
        format_x = get_axis_format(var_x)
        format_y = get_axis_format(var_y)
        format_size = get_axis_format(var_size)
        
        df_scatter_plot = df_scatter.copy()
        df_scatter_plot['x_display'] = df_scatter_plot[var_x] * format_x['multiplicador']
        df_scatter_plot['y_display'] = df_scatter_plot[var_y] * format_y['multiplicador']
        df_scatter_plot['size_display'] = df_scatter_plot[var_size] * format_size['multiplicador']
        
        if 'dict_cores_personalizadas' in st.session_state and st.session_state['dict_cores_personalizadas']:
            color_map = st.session_state['dict_cores_personalizadas']
            fig_scatter = go.Figure()
            
            for instituicao in df_scatter_plot['Instituição'].unique():
                df_inst = df_scatter_plot[df_scatter_plot['Instituição'] == instituicao]
                cor = color_map.get(instituicao, '#1f77b4')
                
                fig_scatter.add_trace(go.Scatter(
                    x=df_inst['x_display'],
                    y=df_inst['y_display'],
                    mode='markers',
                    name=instituicao,
                    marker=dict(
                        size=df_inst['size_display'] / df_scatter_plot['size_display'].max() * 100,
                        color=cor,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{instituicao}</b><br>{var_x}: %{{x}}{format_x["ticksuffix"]}<br>{var_y}: %{{y}}{format_y["ticksuffix"]}<extra></extra>'
                ))
            
            fig_scatter.update_layout(
                title=f'{var_y} vs {var_x} - {periodo_scatter} (TOP {top_n_scatter})',
                xaxis_title=var_x,
                yaxis_title=var_y,
                height=650,
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                xaxis=dict(tickformat=format_x['tickformat'], ticksuffix=format_x['ticksuffix']),
                yaxis=dict(tickformat=format_y['tickformat'], ticksuffix=format_y['ticksuffix']),
                font=dict(family='Source Sans 3')
            )
        else:
            fig_scatter = px.scatter(
                df_scatter_plot, 
                x='x_display', 
                y='y_display', 
                size='size_display', 
                color='Instituição',
                hover_data=['Alavancagem', 'Índice de Basileia', 'ROE An. (%)'],
                title=f'{var_y} vs {var_x} - {periodo_scatter} (TOP {top_n_scatter})',
                labels={'x_display': var_x, 'y_display': var_y}
            )
            
            fig_scatter.update_layout(
                height=650,
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                xaxis=dict(tickformat=format_x['tickformat'], ticksuffix=format_x['ticksuffix'], title=var_x),
                yaxis=dict(tickformat=format_y['tickformat'], ticksuffix=format_y['ticksuffix'], title=var_y),
                font=dict(family='Source Sans 3')
            )
            
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color='white')))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Carregando dados automaticamente do GitHub...")
        st.markdown("Por favor, aguarde alguns segundos e recarregue a página")

# Footer
st.markdown("---")
st.caption("Desenvolvido em 2026 por Matheus Prates, CFA")
