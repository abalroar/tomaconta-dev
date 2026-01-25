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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

st.set_page_config(page_title="fica de olho", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;200;300;400;500;600;700&display=swap');

    button[kind="header"],
    [data-testid="stStatusWidget"],
    [data-testid="stAppViewBlockContainer"] button[kind="header"],
    button[data-testid="baseButton-header"],
    .stApp > header,
    [data-testid="stDecoration"],
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    * {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    html, body, [class*="css"], div, span, p, label, input, select, textarea, button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .sidebar-logo img {
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100px;
        margin: 0 auto;
        display: block;
    }

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
        margin: 0 0 1rem 0;
    }

    button[kind="primary"], button[kind="secondary"], .stButton button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
    }

    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-family: 'IBM Plex Sans', sans-serif !important;
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

    .stMarkdown, .stMarkdown * {
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

    /* ========== FIX SOBREPOSIÇÃO SIDEBAR ========== */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }

    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSegmentedControl"] {
        margin-bottom: 1.5rem !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stExpander"] {
        margin-top: 1rem !important;
        clear: both !important;
    }

    [data-testid="stSidebar"] .row-widget {
        margin-top: 0 !important;
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

def carregar_cores_aliases_local(df_aliases):
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
        cor_valor = row.get(coluna_cor)

        if pd.notna(instituicao) and pd.notna(cor_valor):
            cor_str = str(cor_valor).strip().upper()
            if not cor_str.startswith('#'):
                cor_str = '#' + cor_str
            if len(cor_str) == 7 and all(c in '0123456789ABCDEF' for c in cor_str[1:]):
                dict_cores[instituicao] = cor_str

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

def obter_cor_banco(instituicao):
    if 'dict_cores_personalizadas' in st.session_state and instituicao in st.session_state['dict_cores_personalizadas']:
        return st.session_state['dict_cores_personalizadas'][instituicao]
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

# ========== SIDEBAR - CORREÇÃO DEFINITIVA ==========
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.image(LOGO_PATH, width=100)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">fica de olho</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">análise de instituições financeiras brasileiras</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-author">por matheus prates, cfa</p>', unsafe_allow_html=True)

    st.markdown("")  # Espaço limpo sem divider

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

    st.markdown("")  # Espaço limpo após segmented_control

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
