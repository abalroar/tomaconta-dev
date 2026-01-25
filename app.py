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

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
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
    }
</style>
""", unsafe_allow_html=True)

CACHE_FILE = "data/dados_cache.pkl"
CACHE_INFO = "data/cache_info.txt"
ALIASES_PATH = "data/Aliases.xlsx"
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
            with st.spinner("🔄 Carregando dados do GitHub (10MB)..."):
                r = requests.get(CACHE_URL, timeout=120)
                if r.status_code == 200:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(r.content)
                    
                    r_info = requests.get(CACHE_INFO_URL, timeout=30)
                    if r_info.status_code == 200:
                        Path(CACHE_INFO).write_text(r_info.text)
                    
                    return True
                else:
                    st.warning(f"⚠️ Cache não encontrado (HTTP {r.status_code})")
                    return False
        except Exception as e:
            st.error(f"❌ Erro ao baixar cache: {e}")
            return False
    return True

def criar_mini_grafico(df_banco, variavel, titulo):
    """Cria mini gráfico para uma variável específica"""
    df_sorted = df_banco.sort_values('Período')
    
    # Determinar tipo de formatação
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_razao = ['Alavancagem', 'Risco/Retorno']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar linha
    fig.add_trace(go.Scatter(
        x=df_sorted['Período'],
        y=df_sorted[variavel],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=12, color='#333')),
        height=180,
        margin=dict(l=10, r=10, t=35, b=30),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#e0e0e0',
            tickformat='.2%' if variavel in vars_percentual else (',.0f' if variavel in vars_monetarias else '.2f')
        ),
        hovermode='x'
    )
    
    return fig

def criar_grafico_evolucao(df, top_n=5):
    """Cria gráfico de evolução temporal das maiores instituições"""
    top_bancos = df.groupby('Instituição')['Carteira de Crédito'].mean().nlargest(top_n).index
    df_filtered = df[df['Instituição'].isin(top_bancos)].copy()
    
    fig = px.line(df_filtered, x='Período', y='Carteira de Crédito', 
                  color='Instituição', 
                  title=f'Evolução da Carteira de Crédito - TOP {top_n}',
                  labels={'Carteira de Crédito': 'Carteira (R$ bilhões)', 'Período': 'Trimestre'})
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_traces(line=dict(width=3))
    fig.update_yaxis(tickformat='.1f')
    
    return fig

# Header
st.markdown('<p class="main-header">👁️ Fica de Olho</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dashboard de Análise de Instituições Financeiras Brasileiras</p>', unsafe_allow_html=True)

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

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/financial-analytics.png", width=80)
    
    # Menu de navegação
    menu = st.radio(
        "📍 Navegação",
        ["🏠 Dashboard Principal", "🏦 Análise Individual", "ℹ️ Sobre o Fica de Olho"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.title("⚙️ Controle")
    
    # Status
    if 'df_aliases' in st.session_state:
        st.success(f"✅ {len(st.session_state['df_aliases'])} aliases")
    else:
        st.error("❌ Aliases não encontrados")
    
    # Cache info
    info_cache = ler_info_cache()
    if info_cache:
        with st.expander("💾 Cache"):
            st.text(info_cache)
            if st.button("🗑️ Limpar", use_container_width=True):
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                if os.path.exists(CACHE_INFO):
                    os.remove(CACHE_INFO)
                if 'dados_periodos' in st.session_state:
                    del st.session_state['dados_periodos']
                st.rerun()
    
    st.divider()
    
    # Upload opcional
    uploaded_file = st.file_uploader("📤 Upload Aliases", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file:
        df_aliases = pd.read_excel(uploaded_file)
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]
        st.success("✅ Aliases atualizados")
    
    st.divider()
    
    # Extração
    st.subheader("📅 Atualizar Dados")
    
    col1, col2 = st.columns(2)
    with col1:
        ano_i = st.selectbox("Ano", range(2015,2027), index=8, key="ano_i")
        mes_i = st.selectbox("Trim", ['03','06','09','12'], key="mes_i")
    with col2:
        ano_f = st.selectbox("Ano", range(2015,2027), index=10, key="ano_f")
        mes_f = st.selectbox("Trim", ['03','06','09','12'], index=2, key="mes_f")
    
    if 'dict_aliases' in st.session_state:
        if st.button("🚀 Extrair", type="primary", use_container_width=True):
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
            st.success(f"✅ {len(dados)} períodos!")
            st.rerun()
    else:
        st.warning("⚠️ Carregue aliases")

# CONTEÚDO PRINCIPAL - BASEADO NO MENU
if menu == "ℹ️ Sobre o Fica de Olho":
    # PÁGINA SOBRE
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
        
        1. Os dados já estão carregados automaticamente do GitHub
        2. Acesse o **Dashboard Principal** no menu lateral
        3. Para atualizar dados, configure período e clique em "Extrair Novos Dados"
        4. Personalize visualizações usando os filtros disponíveis
        """)
    
    with col2:
        st.info("""
        ### 💡 Primeira Vez?
        
        **Aguarde:** Os dados estão sendo baixados automaticamente...
        
        **Depois:** Clique em "Dashboard Principal" no menu lateral
        
        **Atualizar:** Configure período e clique em "Extrair Novos Dados"
        
        **Explorar:** Use os filtros para análises customizadas!
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

elif menu == "🏦 Análise Individual":
    # ANÁLISE INDIVIDUAL DE BANCO
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
        
        # Seletor de banco
        bancos_disponiveis = sorted(df['Instituição'].unique())
        banco_selecionado = st.selectbox("🏦 Selecione uma Instituição", bancos_disponiveis, key="banco_individual")
        
        if banco_selecionado:
            df_banco = df[df['Instituição'] == banco_selecionado].copy()
            df_banco = df_banco.sort_values('Período')
            
            # Header do banco
            st.markdown(f"## {banco_selecionado}")
            
            # Métricas do último período
            ultimo_periodo = df_banco['Período'].max()
            dados_ultimo = df_banco[df_banco['Período'] == ultimo_periodo].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Carteira de Crédito",
                    f"R$ {dados_ultimo['Carteira de Crédito']/1e6:.0f}M"
                )
            
            with col2:
                st.metric(
                    "📈 ROE Anualizado",
                    f"{dados_ultimo['ROE An. (%)']*100:.2f}%" if pd.notna(dados_ultimo['ROE An. (%)']) else "N/A"
                )
            
            with col3:
                st.metric(
                    "🛡️ Índice de Basileia",
                    f"{dados_ultimo['Índice de Basileia']:.2f}%" if pd.notna(dados_ultimo['Índice de Basileia']) else "N/A"
                )
            
            with col4:
                st.metric(
                    "⚖️ Alavancagem",
                    f"{dados_ultimo['Alavancagem']:.2f}x" if pd.notna(dados_ultimo['Alavancagem']) else "N/A"
                )
            
            st.markdown("---")
            st.markdown("### 📊 Evolução Histórica das Variáveis")
            
            # Variáveis disponíveis (excluindo Instituição e Período)
            variaveis = [col for col in df_banco.columns if col not in ['Instituição', 'Período'] and df_banco[col].notna().any()]
            
            # Criar grid de mini gráficos (3 por linha)
            for i in range(0, len(variaveis), 3):
                cols = st.columns(3)
                for j, col_obj in enumerate(cols):
                    if i + j < len(variaveis):
                        var = variaveis[i + j]
                        with col_obj:
                            fig = criar_mini_grafico(df_banco, var, var)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    else:
        st.info("🔄 **Carregando dados automaticamente do GitHub...**")
        st.markdown("#### Por favor, aguarde alguns segundos e recarregue a página")

elif menu == "🏠 Dashboard Principal":
    # DASHBOARD PRINCIPAL COM SCATTER PLOT
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
        
        # Calcular variações
        periodos = sorted(df['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
        ultimo = periodos[-1]
        penultimo = periodos[-2] if len(periodos) > 1 else ultimo
        
        df_ultimo = df[df['Período'] == ultimo]
        df_penultimo = df[df['Período'] == penultimo]
        
        carteira_atual = df_ultimo['Carteira de Crédito'].sum()
        carteira_anterior = df_penultimo['Carteira de Crédito'].sum()
        delta_carteira = ((carteira_atual / carteira_anterior) - 1) * 100 if carteira_anterior > 0 else 0
        
        roe_atual = df_ultimo['ROE An. (%)'].mean()
        roe_anterior = df_penultimo['ROE An. (%)'].mean()
        delta_roe = (roe_atual - roe_anterior) * 100 if pd.notna(roe_anterior) else 0
        
        # KPIs com delta
        st.markdown("### 📊 Indicadores Principais")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏦 Instituições", 
                f"{df_ultimo['Instituição'].nunique()}", 
                f"{df_ultimo['Instituição'].nunique() - df_penultimo['Instituição'].nunique()}"
            )
        
        with col2:
            st.metric(
                "💰 Carteira Total", 
                f"R$ {carteira_atual/1e9:.1f}B",
                f"{delta_carteira:+.1f}%"
            )
        
        with col3:
            st.metric(
                "📈 ROE Médio", 
                f"{roe_atual*100:.1f}%",
                f"{delta_roe:+.1f} p.p."
            )
        
        with col4:
            st.metric(
                "🛡️ Basileia Média", 
                f"{df_ultimo['Índice de Basileia'].mean():.1f}%",
                f"{df_ultimo['Índice de Basileia'].mean() - df_penultimo['Índice de Basileia'].mean():+.1f} p.p."
            )
        
        st.markdown("---")
        
        # SCATTER PLOT CUSTOMIZÁVEL
        st.markdown("### 🎯 Análise Comparativa (Scatter Plot)")
        
        # Variáveis numéricas disponíveis
        colunas_numericas = [col for col in df.columns if col not in ['Instituição', 'Período'] and df[col].dtype in ['float64', 'int64']]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_x = st.selectbox("Eixo X", colunas_numericas, index=colunas_numericas.index('Índice de Basileia') if 'Índice de Basileia' in colunas_numericas else 0)
        
        with col2:
            var_y = st.selectbox("Eixo Y", colunas_numericas, index=colunas_numericas.index('ROE An. (%)') if 'ROE An. (%)' in colunas_numericas else 1)
        
        with col3:
            periodo_scatter = st.selectbox("Período", periodos, index=len(periodos)-1)
        
        with col4:
            top_n_scatter = st.slider("TOP N Bancos", 5, 50, 15)
        
        # Criar scatter plot
        df_scatter = df[df['Período'] == periodo_scatter].nlargest(top_n_scatter, 'Carteira de Crédito')
        
        fig_scatter = px.scatter(
            df_scatter, 
            x=var_x, 
            y=var_y, 
            size='Carteira de Crédito', 
            color='Instituição',
            hover_data=['Alavancagem', 'Índice de Basileia', 'ROE An. (%)'],
            title=f'{var_y} vs {var_x} - {periodo_scatter} (TOP {top_n_scatter})',
            labels={var_x: var_x, var_y: var_y}
        )
        
        fig_scatter.update_layout(
            height=550,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color='white')))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # Tabs para conteúdo adicional
        tab1, tab2 = st.tabs(["📋 Rankings", "📈 Evolução Temporal"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 TOP 10 por Carteira de Crédito")
                top10 = df_ultimo.nlargest(10, 'Carteira de Crédito')[['Instituição','Carteira de Crédito']].copy()
                top10['Carteira de Crédito'] = top10['Carteira de Crédito'].apply(lambda x: f"R$ {x/1e9:.2f}B")
                st.dataframe(top10, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("💎 TOP 10 por ROE")
                top_roe = df_ultimo.nlargest(10, 'ROE An. (%)')[['Instituição','ROE An. (%)']].copy()
                top_roe['ROE An. (%)'] = top_roe['ROE An. (%)'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
                st.dataframe(top_roe, use_container_width=True, hide_index=True)
        
        with tab2:
            st.plotly_chart(criar_grafico_evolucao(df, top_n=5), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📅 Períodos Disponíveis", f"{len(periodos)}")
            with col2:
                st.metric("📆 Cobertura", f"{periodos[0]} → {periodos[-1]}")
    
    else:
        st.info("🔄 **Carregando dados automaticamente do GitHub...**")
        st.markdown("#### Por favor, aguarde alguns segundos e recarregue a página")

# Footer
st.markdown("---")
st.caption("💡 **Dica:** Use o menu lateral para navegar entre Dashboard, Análise Individual e Sobre")
