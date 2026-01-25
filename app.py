import streamlit as st
import pandas as pd
import pickle
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from utils.ifdata_extractor import gerar_periodos, processar_todos_periodos
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

st.set_page_config(page_title="fica de olho", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS - sidebar fixa, tipografia e ícones corrigidos
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;200;300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* ===========================================
       SIDEBAR SEMPRE VISÍVEL (sem toggle)
       =========================================== */
    [data-testid="stSidebar"] {
        transform: none !important;
        margin-left: 0 !important;
        position: relative !important;
    }

    /* Esconde botão de toggle da sidebar */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Esconde decoração do header */
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ===========================================
       TIPOGRAFIA - IBM Plex Sans
       IMPORTANTE: Não aplicar em spans de ícones
       =========================================== */
    html, body, .stApp {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* Elementos de texto (exclui spans que podem ser ícones) */
    div, p, label, input, select, textarea, button, h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    p, label, div { font-weight: 300 !important; }
    h1, h2, h3, h4, h5, h6 { font-weight: 500 !important; }
    button { font-weight: 400 !important; }

    /* ===========================================
       ÍCONES - Material Symbols (corrige texto)
       =========================================== */
    /* Ícone do expander */
    [data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"],
    [data-testid="stExpander"] summary > span:first-child,
    [data-testid="stExpanderToggleIcon"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 24px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* ===========================================
       LAYOUT - Sidebar e Expander
       =========================================== */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }

    /* Expander - corrige sobreposição */
    [data-testid="stExpander"] {
        margin-top: 1rem !important;
        overflow: visible !important;
    }

    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem 0 !important;
    }

    /* ===========================================
       ESTILOS CUSTOMIZADOS
       =========================================== */
    .sidebar-logo-container {
        width: 100%; display: flex; justify-content: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sidebar-logo-container img {
        border-radius: 50%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100px;
    }
    .sidebar-title {
        text-align: center; font-size: 1.8rem; font-weight: 300;
        color: #1f77b4; margin: 0.5rem 0 0.2rem 0;
    }
    .sidebar-subtitle {
        text-align: center; font-size: 0.85rem; color: #666;
        margin: 0 0 0.2rem 0;
    }
    .sidebar-author {
        text-align: center; font-size: 0.75rem; color: #888;
        font-style: italic; margin: 0 0 1rem 0;
    }

    .feature-card {
        background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;
        margin-bottom: 1rem; border-left: 4px solid #1f77b4;
    }
    .feature-card h4 { color: #1f77b4; margin-bottom: 0.5rem; }

    .stMetric {
        background-color: #f8f9fa; padding: 15px;
        border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 400 !important; }

    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-weight: 300 !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 300 !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 400 !important;
    }

    .stMarkdown p, .stMarkdown div {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }

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

    /* ============================================================
       CARDS E ELEMENTOS CUSTOMIZADOS
       ============================================================ */
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
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .feature-card p {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

CACHE_FILE = "data/dados_cache.pkl"
CACHE_INFO = "data/cache_info.txt"
ALIASES_PATH = "data/Aliases.xlsx"
LOGO_PATH = "data/logo.png"
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

# FIX PROBLEMA 3: Normalização de nomes de instituições
def normalizar_nome_instituicao(nome):
    """Normaliza nome removendo espaços extras e convertendo para uppercase"""
    if pd.isna(nome):
        return ""
    return " ".join(str(nome).split()).upper()



def normalizar_codigo_cor(cor_valor):
    if pd.isna(cor_valor):
        return None

    if isinstance(cor_valor, (int, np.integer)):
        cor_str = f"{int(cor_valor):06X}"
    elif isinstance(cor_valor, (float, np.floating)) and float(cor_valor).is_integer():
        cor_str = f"{int(cor_valor):06X}"
    else:
        cor_str = str(cor_valor).strip().upper()
        if cor_str.startswith('#'):
            cor_str = cor_str[1:]
        cor_str = cor_str.replace(" ", "")

    if len(cor_str) < 6:
        cor_str = cor_str.zfill(6)

    if len(cor_str) == 6 and all(c in '0123456789ABCDEF' for c in cor_str):
        return f"#{cor_str}"

    return None

# FIX PROBLEMA 3: Carregamento correto de cores com normalização
def carregar_cores_aliases_local(df_aliases):
    """Lê a cor do Aliases.xlsx e cria um dicionário de cores.

    Importante: mapeia tanto o valor da coluna 'Instituição' (nome original vindo do BCB)
    quanto o valor da coluna 'Alias Banco' (nome amigável que aparece no app),
    para que a cor seja aplicada em qualquer tela.
    """
    dict_cores = {}
    if df_aliases is None or df_aliases.empty:
        return dict_cores

    colunas_possiveis = ['Código Cor', 'Cor', 'Color', 'Hex', 'Código']
    coluna_cor = None
    for col in colunas_possiveis:
        if col in df_aliases.columns:
            coluna_cor = col
            break

    if coluna_cor is None:
        return dict_cores

    for _, row in df_aliases.iterrows():
        instituicao = row.get('Instituição')
        alias = row.get('Alias Banco')
        cor_valor = row.get(coluna_cor)

        cor_str = normalizar_codigo_cor(cor_valor)
        if not cor_str:
            continue

        if pd.notna(instituicao):
            dict_cores[normalizar_nome_instituicao(instituicao)] = cor_str
            # Também aceita busca pelo 'Alias Banco' (útil quando a coluna Instituição já vem renomeada)
            alias_banco = row.get('Alias Banco')
            if pd.notna(alias_banco):
                dict_cores[normalizar_nome_instituicao(alias_banco)] = cor_str

        # Também mapeia pelo alias (é o que aparece na UI)
        if pd.notna(alias):
            dict_cores[normalizar_nome_instituicao(alias)] = cor_str

    return dict_cores

def baixar_cache_inicial():
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
    vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
    vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']

    if variavel in vars_percentual:
        return {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
    elif variavel in vars_monetarias:
        return {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
    else:
        return {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}

# FIX PROBLEMA 3: Busca de cor com normalização
def obter_cor_banco(instituicao):
    if 'dict_cores_personalizadas' in st.session_state:
        instituicao_norm = normalizar_nome_instituicao(instituicao)
        if instituicao_norm in st.session_state['dict_cores_personalizadas']:
            return st.session_state['dict_cores_personalizadas'][instituicao_norm]
    return None

def criar_mini_grafico(df_banco, variavel, titulo):
    df_sorted = df_banco.copy()
    df_sorted['ano'] = df_sorted['Período'].str.split('/').str[1].astype(int)
    df_sorted['trimestre'] = df_sorted['Período'].str.split('/').str[0].astype(int)
    df_sorted = df_sorted.sort_values(['ano', 'trimestre'])

    instituicao = df_sorted['Instituição'].iloc[0]
    cor_banco = obter_cor_banco(instituicao)
    if not cor_banco:
        cor_banco = '#1f77b4'

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
        line=dict(color=cor_banco, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({int(cor_banco[1:3], 16)}, {int(cor_banco[3:5], 16)}, {int(cor_banco[5:7], 16)}, 0.2)',
        hovertemplate='%{x}<br>%{y:' + tickformat + '}' + suffix + '<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=12, color='#333', family='IBM Plex Sans')),
        height=180,
        margin=dict(l=10, r=10, t=35, b=30),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#e0e0e0', tickformat=tickformat, ticksuffix=suffix),
        hovermode='x',
        font=dict(family='IBM Plex Sans')
    )

    return fig

def gerar_scorecard_pdf(banco_selecionado, df_banco, periodo_inicial, periodo_final):
    df_sorted = df_banco.copy()
    df_sorted['ano'] = df_sorted['Período'].str.split('/').str[1].astype(int)
    df_sorted['trimestre'] = df_sorted['Período'].str.split('/').str[0].astype(int)
    df_sorted = df_sorted.sort_values(['ano', 'trimestre'])

    instituicao = df_sorted['Instituição'].iloc[0]
    cor_banco = obter_cor_banco(instituicao)
    if not cor_banco:
        cor_banco = '#1f77b4'

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.4*inch, bottomMargin=0.4*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=28, textColor=colors.HexColor('#1f77b4'), spaceAfter=4, alignment=TA_LEFT, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#666666'), spaceAfter=16, alignment=TA_LEFT)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1f77b4'), spaceAfter=12, spaceBefore=12, fontName='Helvetica-Bold')

    story = []
    story.append(Paragraph(banco_selecionado, title_style))
    story.append(Paragraph(f"Análise de {periodo_inicial} até {periodo_final}", subtitle_style))

    ultimo_periodo = df_sorted['Período'].iloc[-1]
    dados_ultimo = df_sorted[df_sorted['Período'] == ultimo_periodo].iloc[0]

    metricas_principais = [
        ('Carteira de Crédito', 'Carteira de Crédito'),
        ('ROE Anualizado', 'ROE An. (%)'),
        ('Índice de Basileia', 'Índice de Basileia'),
        ('Alavancagem', 'Alavancagem'),
    ]

    metricas_data = []
    for label, col in metricas_principais:
        valor = formatar_valor(dados_ultimo.get(col), col)
        metricas_data.append([label, valor])

    metrics_table_data = [
        [
            Paragraph(f'<font size="9"><b>{metricas_data[0][0]}</b></font><br/><font size="14"><b>{metricas_data[0][1]}</b></font>', styles['Normal']),
            Paragraph(f'<font size="9"><b>{metricas_data[1][0]}</b></font><br/><font size="14"><b>{metricas_data[1][1]}</b></font>', styles['Normal'])
        ],
        [
            Paragraph(f'<font size="9"><b>{metricas_data[2][0]}</b></font><br/><font size="14"><b>{metricas_data[2][1]}</b></font>', styles['Normal']),
            Paragraph(f'<font size="9"><b>{metricas_data[3][0]}</b></font><br/><font size="14"><b>{metricas_data[3][1]}</b></font>', styles['Normal'])
        ]
    ]

    metrics_table = Table(metrics_table_data, colWidths=[3.25*inch, 3.25*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Evolução Histórica das Variáveis", section_style))

    variaveis = [col for col in df_sorted.columns if col not in ['Instituição', 'Período', 'ano', 'trimestre'] and df_sorted[col].notna().sum() > 0]

    cor_rgb = tuple(int(cor_banco[i:i+2], 16) for i in (1, 3, 5))
    cor_rgb_norm = tuple(c/255 for c in cor_rgb)

    def criar_figura_grafico(df_plot, variavel, titulo):
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=100)

        vars_percentual = ['ROE An. (%)', 'Índice de Basileia', 'Crédito/Captações', 'Funding Gap', 'Carteira/Ativo', 'Market Share Carteira']
        vars_monetarias = ['Carteira de Crédito', 'Lucro Líquido', 'Patrimônio Líquido', 'Captações', 'Ativo Total']

        y_values = df_plot[variavel].values
        x_labels = df_plot['Período'].values
        x_pos = np.arange(len(x_labels))

        if variavel in vars_percentual:
            y_display = y_values * 100
            suffix = '%'
        elif variavel in vars_monetarias:
            y_display = y_values / 1e6
            suffix = 'M'
        else:
            y_display = y_values
            suffix = ''

        ax.fill_between(x_pos, y_display, alpha=0.3, color=cor_rgb_norm)
        ax.plot(x_pos, y_display, color=cor_rgb_norm, linewidth=2, marker='o', markersize=3)
        ax.set_title(titulo, fontsize=11, fontweight='bold', color='#333333', pad=8)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')

        if len(x_labels) > 4:
            step = max(1, len(x_labels) // 4)
            ax.set_xticks(x_pos[::step])
            ax.set_xticklabels(x_labels[::step], fontsize=7, rotation=45)
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=7, rotation=45)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}{suffix}'))
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, linestyle='--', color='#e0e0e0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig

    figs_por_linha = 3
    for i in range(0, len(variaveis), figs_por_linha):
        graficos_linha = variaveis[i:i+figs_por_linha]
        row_images = []
        for var in graficos_linha:
            try:
                fig = criar_figura_grafico(df_sorted, var, var)
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
                img_buffer.seek(0)
                img = Image(img_buffer, width=2.2*inch, height=1.6*inch)
                row_images.append(img)
                plt.close(fig)
            except:
                row_images.append(Paragraph(f"[Erro: {var}]", styles['Normal']))
        while len(row_images) < figs_por_linha:
            row_images.append(Spacer(1, 0))
        img_table = Table([row_images], colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.15*inch))

    story.append(Spacer(1, 0.1*inch))
    rodape = Paragraph(f"<font size='8'><i>Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | Fonte: API IF.DATA - BCB</i></font>", styles['Normal'])
    story.append(rodape)

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        return None

if 'df_aliases' not in st.session_state:
    df_aliases = carregar_aliases()
    if df_aliases is not None:
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases_local(df_aliases)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]

if 'dados_periodos' not in st.session_state:
    baixar_cache_inicial()
    dados_cache = carregar_cache()
    if dados_cache:
        st.session_state['dados_periodos'] = dados_cache

with st.sidebar:
    # FIX PROBLEMA 1: Logo centralizado com HTML inline
    if os.path.exists(LOGO_PATH):
        logo_base64 = base64.b64encode(Path(LOGO_PATH).read_bytes()).decode("utf-8")
        st.markdown(
            f'<div class="sidebar-logo-container"><img src="data:image/png;base64,{logo_base64}" width="100" /></div>',
            unsafe_allow_html=True
        )
    st.markdown('<p class="sidebar-title">fica de olho</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">análise de instituições financeiras brasileiras</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-author">por matheus prates, cfa</p>', unsafe_allow_html=True)

    st.markdown("")

    if 'menu_atual' not in st.session_state:
        st.session_state['menu_atual'] = "Sobre"

    menu = st.segmented_control(
        "navegação",
        ["Sobre", "Análise Individual", "Scatter Plot"],
        default=st.session_state['menu_atual'],
        label_visibility="collapsed"
    )

    if menu != st.session_state['menu_atual']:
        st.session_state['menu_atual'] = menu
        st.rerun()

    st.markdown("")

    with st.expander("controle avançado"):
        if 'df_aliases' in st.session_state:
            st.success(f"{len(st.session_state['df_aliases'])} aliases carregados")
        else:
            st.error("aliases não encontrados")

        info_cache = ler_info_cache()
        if info_cache:
            st.caption(info_cache.replace('\n', ' • '))

        st.markdown("---")
        st.markdown("**atualizar dados**")

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

if menu == "Sobre":
    st.markdown("""
    ## sobre a plataforma

    o fica de olho é uma plataforma de análise financeira que automatiza a extração, processamento e visualização de dados de instituições financeiras brasileiras, oferecendo insights comparativos e históricos.
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
    2. **navegue pelas páginas**: use o menu lateral para acessar análise individual ou scatter plot
    3. **atualize quando necessário**: configure o período desejado e clique em extrair dados
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

elif menu == "Análise Individual":
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

                    periodos_disponiveis = sorted(df_banco['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
                    if len(periodos_disponiveis) >= 2:
                        pdf_buffer = gerar_scorecard_pdf(banco_selecionado, df_banco, periodos_disponiveis[0], periodos_disponiveis[-1])
                        if pdf_buffer:
                            st.download_button(
                                label="baixar scorecard",
                                data=pdf_buffer.getvalue(),
                                file_name=f"scorecard_{banco_selecionado.replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )

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

elif menu == "Scatter Plot":
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
            opcoes_tamanho = ['Tamanho fixo'] + colunas_numericas
            default_idx = opcoes_tamanho.index('Carteira de Crédito') if 'Carteira de Crédito' in opcoes_tamanho else 1
            var_size = st.selectbox("tamanho", opcoes_tamanho, index=default_idx)
        with col4:
            periodo_scatter = st.selectbox("período", periodos, index=len(periodos)-1)
        with col5:
            top_n_scatter = st.slider("top n", 5, 50, 15)

        df_scatter = df[df['Período'] == periodo_scatter].nlargest(top_n_scatter, 'Carteira de Crédito')

        format_x = get_axis_format(var_x)
        format_y = get_axis_format(var_y)

        df_scatter_plot = df_scatter.copy()
        df_scatter_plot['x_display'] = df_scatter_plot[var_x] * format_x['multiplicador']
        df_scatter_plot['y_display'] = df_scatter_plot[var_y] * format_y['multiplicador']

        if var_size == 'Tamanho fixo':
            tamanho_constante = 50
        else:
            format_size = get_axis_format(var_size)
            df_scatter_plot['size_display'] = df_scatter_plot[var_size] * format_size['multiplicador']

        fig_scatter = go.Figure()
        cores_plotly = px.colors.qualitative.Plotly
        idx_cor = 0

        for instituicao in df_scatter_plot['Instituição'].unique():
            df_inst = df_scatter_plot[df_scatter_plot['Instituição'] == instituicao]
            cor = obter_cor_banco(instituicao)
            if not cor:
                cor = cores_plotly[idx_cor % len(cores_plotly)]
                idx_cor += 1

            if var_size == 'Tamanho fixo':
                marker_size = tamanho_constante
            else:
                marker_size = df_inst['size_display'] / df_scatter_plot['size_display'].max() * 100

            fig_scatter.add_trace(go.Scatter(
                x=df_inst['x_display'],
                y=df_inst['y_display'],
                mode='markers',
                name=instituicao,
                marker=dict(size=marker_size, color=cor, opacity=1.0, line=dict(width=1, color='white')),
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

        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

st.markdown("---")
st.caption("desenvolvido em 2026 por matheus prates, cfa")
