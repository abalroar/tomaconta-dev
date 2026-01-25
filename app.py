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

st.set_page_config(page_title="fica de olho", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS customizado com fonte IBM Plex Sans (clean e thin como StockPeers)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;200;300;400;500;600;700&display=swap');
    
    /* Esconder ícone de rerun chato */
    button[kind="header"] {
        display: none !important;
    }
    
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Aplicar IBM Plex Sans em TODOS os elementos */
    * {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    html, body, [class*="css"], div, span, p, label, input, select, textarea, button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }
    
    /* Headers e títulos */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }
    
    /* Logo no sidebar */
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    
    .sidebar-logo img {
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100px;
    }
    
    /* Título no sidebar */
    .sidebar-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 300;
        color: #1f77b4;
        margin: 0.5rem 0 0.2rem 0;
        line-height: 1.2;
    }
    
    .sidebar-subtitle {
        text-align: center;
        font-size: 0.85rem;
        color: #666;
        margin: 0 0 0.2rem 0;
        line-height: 1.3;
    }
    
    .sidebar-author {
        text-align: center;
        font-size: 0.75rem;
        color: #888;
        font-style: italic;
        margin: 0 0 1.5rem 0;
    }
    
    /* Botões */
    button[kind="primary"], button[kind="secondary"], .stButton button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }
    
    /* Selectbox, inputs */
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }
    
    /* Métricas */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 300 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 400 !important;
    }
    
    /* Markdown */
    .stMarkdown, .stMarkdown * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }
    
    /* Captions */
    .stCaption {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 400 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    
    /* Cards de feature */
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    
    .feature-card h4 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
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
            with st.spinner("carregando dados do github (10mb)..."):
                r = requests.get(CACHE_URL, timeout=120)
                if r.status_code == 200:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(r.content)
                    
                    r_info = requests.get(CACHE_INFO_URL, timeout=30)
                    if r_info.status_code == 200:
                        Path(CACHE_INFO).write_text(r_info.text)
                    
                    return True
                else:
                    st.warning(f"cache não encontrado (http {r.status_code})")
                    return False
        except Exception as e:
            st.error(f"erro ao baixar cache: {e}")
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
    df_sorted = df_banco.copy()
    df_sorted['ano'] = df_sorted['Período'].str.split('/').str[1].astype(int)
    df_sorted['trimestre'] = df_sorted['Período'].str.split('/').str[0].astype(int)
    df_sorted = df_sorted.sort_values(['ano', 'trimestre'])
    
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']
    
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sorted['Período'],
        y=hover_values,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='%{x}<br>%{y:' + tickformat + '}' + suffix + '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=12, color='#333', family='IBM Plex Sans')),
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
        font=dict(family='IBM Plex Sans')
    )
    
    return fig

if 'df_aliases' not in st.session_state:
    df_aliases = carregar_aliases()
    if df_aliases is not None:
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]

if 'dados_periodos' not in st.session_state:
    baixar_cache_inicial()
    dados_cache = carregar_cache()
    if dados_cache:
        st.session_state['dados_periodos'] = dados_cache

with st.sidebar:
    # Logo e título no topo do sidebar
    if os.path.exists(LOGO_PATH):
        st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.image(LOGO_PATH, width=100)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="sidebar-title">fica de olho</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">análise de instituições<br>financeiras brasileiras</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-author">por matheus prates, cfa</p>', unsafe_allow_html=True)
    
    st.divider()
    
    st.title("navegação")
    
    if 'menu_atual' not in st.session_state:
        st.session_state['menu_atual'] = "sobre"
    
    if st.button("sobre", use_container_width=True, type="primary" if st.session_state['menu_atual'] == "sobre" else "secondary"):
        st.session_state['menu_atual'] = "sobre"
        st.rerun()
    
    if st.button("análise individual", use_container_width=True, type="primary" if st.session_state['menu_atual'] == "análise individual" else "secondary"):
        st.session_state['menu_atual'] = "análise individual"
        st.rerun()
    
    if st.button("scatter plot", use_container_width=True, type="primary" if st.session_state['menu_atual'] == "scatter plot" else "secondary"):
        st.session_state['menu_atual'] = "scatter plot"
        st.rerun()
    
    menu = st.session_state['menu_atual']
    
    st.divider()
    st.title("controle")
    
    if 'df_aliases' in st.session_state:
        st.success(f"{len(st.session_state['df_aliases'])} aliases carregados")
    else:
        st.error("aliases não encontrados")
    
    info_cache = ler_info_cache()
    if info_cache:
        with st.expander("informações do cache"):
            st.text(info_cache)
            if st.button("limpar cache", use_container_width=True):
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                if os.path.exists(CACHE_INFO):
                    os.remove(CACHE_INFO)
                if 'dados_periodos' in st.session_state:
                    del st.session_state['dados_periodos']
                st.rerun()
    
    st.divider()
    
    with st.expander("upload de aliases"):
        uploaded_file = st.file_uploader("selecione arquivo excel", type=['xlsx'], key="upload_aliases")
        
        if uploaded_file:
            df_aliases = pd.read_excel(uploaded_file)
            st.session_state['df_aliases'] = df_aliases
            st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
            st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
            st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]
            st.success("aliases atualizados com sucesso")
    
    st.divider()
    
    st.subheader("atualizar dados")
    
    col1, col2 = st.columns(2)
    with col1:
        ano_i = st.selectbox("ano inicial", range(2015,2027), index=8, key="ano_i")
        mes_i = st.selectbox("trimestre inicial", ['03','06','09','12'], key="mes_i")
    with col2:
        ano_f = st.selectbox("ano final", range(2015,2027), index=10, key="ano_f")
        mes_f = st.selectbox("trimestre final", ['03','06','09','12'], index=2, key="mes_f")
    
    if 'dict_aliases' in st.session_state:
        if st.button("extrair dados", type="primary", use_container_width=True):
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
            st.success(f"{len(dados)} períodos extraídos com sucesso")
            st.rerun()
    else:
        st.warning("carregue os aliases primeiro")

if menu == "sobre":
    st.markdown("---")
    
    st.markdown("""
    ## sobre a plataforma
    
    o **fica de olho** é uma plataforma de análise financeira que automatiza a extração, processamento e visualização 
    de dados de instituições financeiras brasileiras, oferecendo insights comparativos e históricos em tempo real.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>extração automatizada</h4>
            <p>integração direta com a api if.data do banco central do brasil para coleta de dados em tempo real.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>análise temporal</h4>
            <p>acompanhamento histórico de métricas financeiras ao longo de múltiplos trimestres com séries temporais.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>visualização interativa</h4>
            <p>gráficos de dispersão customizáveis com filtros dinâmicos e comparações multi-institucionais.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>classificação personalizada</h4>
            <p>sistema de aliases para renomear e categorizar instituições conforme critérios específicos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>métricas calculadas</h4>
            <p>roe anualizado, alavancagem, funding gap, market share e índices de risco/retorno automatizados.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>dados oficiais</h4>
            <p>fonte única e confiável: api if.data do banco central do brasil com atualização trimestral.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## dados disponíveis
    
    todas as informações são extraídas diretamente da **api if.data** do banco central, incluindo:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - carteira de crédito classificada
        - patrimônio líquido e resultado
        - índice de basileia e capital
        """)
    
    with col2:
        st.markdown("""
        - captações e funding
        - ativo total e composição
        - cadastro de instituições autorizadas
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## como utilizar
    
    1. **dados pré-carregados**: a plataforma já possui dados históricos prontos para análise
    2. **navegue pelas páginas**: use o menu lateral para acessar "análise individual" ou "scatter plot"
    3. **atualize quando necessário**: configure o período desejado e clique em "extrair dados"
    4. **personalize visualizações**: aplique filtros e ajuste variáveis conforme sua análise
    
    ---
    
    ## stack tecnológica
    
    - **python 3.10+** | linguagem base
    - **streamlit** | interface web interativa
    - **pandas** | processamento e análise de dados
    - **plotly** | visualizações dinâmicas
    - **api bcb olinda** | fonte oficial de dados
    """)
    
    st.markdown("---")
    st.caption("desenvolvido em 2026 por matheus prates, cfa | ferramenta open-source para análise do sistema financeiro brasileiro")

elif menu == "análise individual":
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
                banco_selecionado = st.selectbox("selecione uma instituição", bancos_disponiveis, key="banco_individual")
                
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
                        st.metric("carteira de crédito", formatar_valor(dados_ultimo.get('Carteira de Crédito'), 'Carteira de Crédito'))
                    with col2:
                        st.metric("roe anualizado", formatar_valor(dados_ultimo.get('ROE An. (%)'), 'ROE An. (%)'))
                    with col3:
                        st.metric("índice de basileia", formatar_valor(dados_ultimo.get('Índice de Basileia'), 'Índice de Basileia'))
                    with col4:
                        st.metric("alavancagem", formatar_valor(dados_ultimo.get('Alavancagem'), 'Alavancagem'))
                    
                    st.markdown("---")
                    st.markdown("### evolução histórica das variáveis")
                    
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
                st.warning("nenhuma instituição encontrada nos dados")
        else:
            st.warning("dados incompletos ou vazios")
    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "scatter plot":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
        
        colunas_numericas = [col for col in df.columns if col not in ['Instituição', 'Período'] and df[col].dtype in ['float64', 'int64']]
        periodos = sorted(df['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            var_x = st.selectbox("eixo x", colunas_numericas, index=colunas_numericas.index('Índice de Basileia') if 'Índice de Basileia' in colunas_numericas else 0)
        with col2:
            var_y = st.selectbox("eixo y", colunas_numericas, index=colunas_numericas.index('ROE An. (%)') if 'ROE An. (%)' in colunas_numericas else 1)
        with col3:
            var_size = st.selectbox("tamanho", colunas_numericas, index=colunas_numericas.index('Carteira de Crédito') if 'Carteira de Crédito' in colunas_numericas else 0)
        with col4:
            periodo_scatter = st.selectbox("período", periodos, index=len(periodos)-1)
        with col5:
            top_n_scatter = st.slider("top n", 5, 50, 15)
        
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
                title=f'{var_y} vs {var_x} - {periodo_scatter} (top {top_n_scatter})',
                xaxis_title=var_x,
                yaxis_title=var_y,
                height=650,
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                xaxis=dict(tickformat=format_x['tickformat'], ticksuffix=format_x['ticksuffix']),
                yaxis=dict(tickformat=format_y['tickformat'], ticksuffix=format_y['ticksuffix']),
                font=dict(family='IBM Plex Sans')
            )
        else:
            fig_scatter = px.scatter(
                df_scatter_plot, 
                x='x_display', 
                y='y_display', 
                size='size_display', 
                color='Instituição',
                hover_data=['Alavancagem', 'Índice de Basileia', 'ROE An. (%)'],
                title=f'{var_y} vs {var_x} - {periodo_scatter} (top {top_n_scatter})',
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
                font=dict(family='IBM Plex Sans')
            )
            
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color='white')))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

st.markdown("---")
st.caption("desenvolvido em 2026 por matheus prates, cfa")
