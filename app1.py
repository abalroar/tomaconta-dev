import importlib
import streamlit as st
import pandas as pd
import os
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import time
import html as _html_mod

# === PERFORMANCE TIMER ===
_perf_timers = {}

def _perf_start(label: str):
    """Inicia timer de performance para uma etapa."""
    _perf_timers[label] = time.perf_counter()

def _perf_end(label: str) -> float:
    """Finaliza timer e retorna tempo em segundos."""
    if label in _perf_timers:
        elapsed = time.perf_counter() - _perf_timers[label]
        del _perf_timers[label]
        return elapsed
    return 0.0

def _perf_log(label: str) -> str:
    """Finaliza timer e retorna mensagem formatada."""
    elapsed = _perf_end(label)
    return f"[PERF] {label}: {elapsed:.3f}s"
import utils  # garante pacote utils carregado
importlib.invalidate_caches()

from utils.ifdata_extractor import (
    gerar_periodos,
    processar_todos_periodos,
    construir_mapa_codinst,
    construir_mapa_codinst_multiperiodo,
    get_log_file_path,
    parece_codigo_instituicao,
)
# Sistema unificado de cache (capital e principal)
from utils.ifdata_cache import (
    # Cache principal (compat)
    carregar_cache,
    salvar_cache,
    get_cache_info,
    # Cache de capital
    gerar_periodos_capital,
    processar_todos_periodos_capital,
    salvar_cache_capital,
    carregar_cache_capital,
    get_capital_cache_info,
    ler_info_cache_capital,
    get_campos_capital_info,
    # Gerenciador unificado
    CacheManager,
    get_manager as get_cache_manager,
    DERIVED_METRICS,
    DERIVED_METRICS_FORMAT,
    DERIVED_METRICS_FORMULAS,
    build_derived_metrics,
    load_derived_metrics_slice,
)
import io
import base64
import subprocess
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import xlsxwriter
from PIL import Image as PILImage
from io import BytesIO

st.set_page_config(page_title="🏦 👀 toma.conta!", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS - sidebar fixa, tipografia e ícones corrigidos
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;200;300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* ===========================================
       SIDEBAR - esconde completamente quando recolhido
       =========================================== */
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] > div {
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

# Diretório base relativo ao app.py (funciona independente do nome do repo)
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"

ALIASES_PATH = str(DATA_DIR / "Aliases.xlsx")
LOGO_PATH = str(DATA_DIR / "logo.jpg")

# Senha para proteger a funcionalidade de atualização de cache
SENHA_ADMIN = "m4th3u$987"

VARS_PERCENTUAL = [
    'ROE Ac. YTD an. (%)',
    'Índice de Basileia',
    'Índice de CET1',
    'Crédito/Captações (%)',
    'Crédito/Ativo (%)',
    'Índice de Imobilização',
    'Perda Esperada / Carteira Bruta',
    'Perda Esperada / (Carteira C4 + C5)',
    'Desp PDD Anualizada / Carteira Bruta',
    'Carteira de Créd. Class. C4+C5 / Carteira Bruta',
    # Variáveis de Capital (Relatório 5)
    'Índice de Capital Principal',
    'Índice de Capital Principal (CET1)',
    'Índice de Capital Nível I',
    'Razão de Alavancagem',
    # Métricas derivadas
    *DERIVED_METRICS,
]
VARS_RAZAO = ['Crédito/PL (%)', 'Ativo/PL']
VARS_MOEDAS = [
    'Carteira de Crédito',
    'Carteira de Crédito Bruta',
    'Ativos Líquidos',
    'Depósitos Totais',
    'Perda Esperada',
    'Carteira de Créd. Class. C4+C5',
    'Lucro Líquido Acumulado YTD',
    'Patrimônio Líquido',
    'Captações',
    'Ativo Total',
    'Títulos e Valores Mobiliários',
    'Passivo Exigível',
    'Patrimônio de Referência',
    'Patrimônio de Referência para Comparação com o RWA (e)',
    # Variáveis de Capital (Relatório 5)
    'Capital Principal',
    'Capital Complementar',
    'Capital Nível II',
    'RWA Total',
    'RWA Crédito',
    'RWA Mercado',
    'RWA Operacional',
    'Exposição Total',
    'Adicional de Capital Principal',
]
VARS_CONTAGEM = ['Número de Agências', 'Número de Postos de Atendimento']

PEERS_TABELA_LAYOUT = [
    {
        "section": "Balanço",
        "rows": [
            {
                "label": "Ativo Total",
                "data_keys": ["Ativo Total"],
                "format_key": "Ativo Total",
            },
            {
                "label": "Ativos Líquidos",
                "data_keys": [],
                "format_key": "Ativos Líquidos",
            },
            {
                "label": "Carteira de Crédito Bruta",
                "data_keys": [],
                "format_key": "Carteira de Crédito Bruta",
            },
            {
                "label": "Depósitos Totais",
                "data_keys": [],
                "format_key": "Depósitos Totais",
            },
            {
                "label": "Patrimônio Líquido (PL)",
                "data_keys": ["Patrimônio Líquido"],
                "format_key": "Patrimônio Líquido",
            },
        ],
    },
    {
        "section": "Qualidade Carteira",
        "rows": [
            {
                "label": "Perda Esperada",
                "data_keys": [],
                "format_key": "Perda Esperada",
            },
            {
                "label": "Perda Esperada / Carteira Bruta",
                "data_keys": [],
                "format_key": "Perda Esperada / Carteira Bruta",
            },
            {
                "label": "Carteira de Créd. Class. C4+C5",
                "data_keys": [],
                "format_key": "Carteira de Créd. Class. C4+C5",
            },
            {
                "label": "Carteira de Créd. Class. C4+C5 / Carteira Bruta",
                "data_keys": [],
                "format_key": "Carteira de Créd. Class. C4+C5 / Carteira Bruta",
            },
            {
                "label": "Perda Esperada / (Carteira C4 + C5)",
                "data_keys": [],
                "format_key": "Perda Esperada / (Carteira C4 + C5)",
            },
            {
                "label": "Desp PDD Anualizada / Carteira Bruta",
                "data_keys": [],
                "format_key": "Desp PDD Anualizada / Carteira Bruta",
            },
            {
                "label": "Desp PDD / NII (ref: período acumulado)",
                "data_keys": [],
                "format_key": "Desp PDD / NII",
                "todo": "TODO: Integrar despesa PDD/NII (período acumulado) a partir das fontes do projeto.",
            },
        ],
    },
    {
        "section": "Alavancagem",
        "rows": [
            {
                "label": "Ativo / PL",
                "data_keys": ["Ativo/PL", "Ativo / PL"],
                "format_key": "Ativo/PL",
                "todo": "TODO: Integrar Ativo/PL a partir das fontes do projeto (sem criar fórmula nova).",
            },
            {
                "label": "Crédito / PL",
                "data_keys": ["Crédito/PL (%)", "Crédito/PL"],
                "format_key": "Crédito/PL (%)",
            },
            {
                "label": "Índice de Capital Principal (CET1)",
                "data_keys": [],
                "format_key": "Índice de Capital Principal",
            },
            {
                "label": "Índice de Basileia Total",
                "data_keys": [],
                "format_key": "Índice de Basileia",
            },
        ],
    },
    {
        "section": "Desempenho",
        "rows": [
            {
                "label": "Lucro Líquido Acumulado",
                "data_keys": ["Lucro Líquido Acumulado YTD"],
                "format_key": "Lucro Líquido Acumulado YTD",
            },
            {
                "label": "ROE AC. Anualizado (%)",
                "data_keys": ["ROE Ac. YTD an. (%)"],
                "format_key": "ROE Ac. YTD an. (%)",
            },
        ],
    },
]

# Variáveis disponíveis para ponderação (variáveis de tamanho/volume em valores absolutos)
# Mapeamento: label exibido -> nome da coluna no DataFrame
VARIAVEIS_PONDERACAO = {
    'Média Simples': None,  # Sem ponderação
    'Ativo Total': 'Ativo Total',
    'Carteira de Crédito': 'Carteira de Crédito',
    'Patrimônio Líquido': 'Patrimônio Líquido',
    'Patrimônio de Referência': 'Patrimônio de Referência',
    'Captações': 'Captações',
    'Passivo Exigível': 'Passivo Exigível',
    'RWA Total': 'RWA Total',
}

def calcular_media_ponderada(df, coluna_valor, coluna_peso=None):
    """Calcula média simples ou ponderada de uma coluna.

    Args:
        df: DataFrame com os dados
        coluna_valor: Nome da coluna cujos valores serão promediados
        coluna_peso: Nome da coluna de peso (None para média simples)

    Returns:
        float: Média calculada
    """
    if df.empty or coluna_valor not in df.columns:
        return 0

    valores = df[coluna_valor].dropna()
    if valores.empty:
        return 0

    # Média simples se não houver coluna de peso
    if coluna_peso is None:
        return valores.mean()

    if coluna_peso not in df.columns:
        return valores.mean()

    # Filtrar apenas registros com peso e valor válidos
    mask = df[coluna_valor].notna() & df[coluna_peso].notna() & (df[coluna_peso] > 0)
    df_valid = df[mask]

    if df_valid.empty:
        return valores.mean()

    # Calcular média ponderada: sum(valor * peso) / sum(peso)
    soma_ponderada = (df_valid[coluna_valor] * df_valid[coluna_peso]).sum()
    soma_pesos = df_valid[coluna_peso].sum()

    if soma_pesos == 0:
        return valores.mean()

    return soma_ponderada / soma_pesos

def get_label_media(coluna_peso):
    """Retorna o label descritivo do tipo de média."""
    if coluna_peso is None:
        return 'Média'
    # Abreviar nomes longos
    abreviacoes = {
        'Ativo Total': 'Ativo Total',
        'Carteira de Crédito': 'Cart. Crédito',
        'Patrimônio Líquido': 'PL',
        'Patrimônio de Referência': 'PR',
        'Captações': 'Captações',
        'Passivo Exigível': 'Passivo',
        'RWA Total': 'RWA',
    }
    nome_abrev = abreviacoes.get(coluna_peso, coluna_peso)
    return f'Média (pond. {nome_abrev})'

def get_cache_info_detalhado():
    """Retorna informações detalhadas sobre o arquivo de cache."""
    cache_manager = get_cache_manager()
    cache = cache_manager.get_cache("principal") if cache_manager else None
    info = {
        'existe': False,
        'caminho': '',
        'tamanho': 0,
        'tamanho_formatado': '0 B',
        'data_modificacao': None,
        'data_formatada': 'N/A',
        'fonte': 'nenhuma',
        'formato': 'N/A',
    }
    if cache and cache.existe():
        if cache.arquivo_dados.exists():
            arquivo_dados = cache.arquivo_dados
            info['formato'] = 'parquet'
        else:
            arquivo_dados = cache.arquivo_dados_pickle
            info['formato'] = 'pickle'

        info['existe'] = True
        info['caminho'] = str(arquivo_dados.resolve())
        stat = arquivo_dados.stat()
        info['tamanho'] = stat.st_size
        # Formatar tamanho
        tamanho = stat.st_size
        for unidade in ['B', 'KB', 'MB', 'GB']:
            if tamanho < 1024:
                info['tamanho_formatado'] = f"{tamanho:.1f} {unidade}"
                break
            tamanho /= 1024
        # Data de modificação
        info['data_modificacao'] = datetime.fromtimestamp(stat.st_mtime)
        info['data_formatada'] = info['data_modificacao'].strftime('%d/%m/%Y %H:%M:%S')

        if cache.arquivo_metadata.exists():
            try:
                metadata = json.loads(cache.arquivo_metadata.read_text())
                info['fonte'] = metadata.get('fonte', 'desconhecida')
            except Exception:
                info['fonte'] = 'desconhecida'
    return info


def _fator_anualizacao(mes: int) -> float:
    """Retorna fator de anualização por mês: Mar=4, Jun=2, Set=12/9, Dez=1."""
    if mes == 3:
        return 4.0
    elif mes == 6:
        return 2.0
    elif mes == 9:
        return 12 / 9
    elif mes == 12:
        return 1.0
    return 12 / mes if mes and mes > 0 else 1.0


def _calcular_roe_anualizado(ll_ytd, pl_t, pl_dez_anterior, mes):
    """ROE Ac. Anualizado = (LL_YTD × fator) / ((PL_t + PL_Dez_anterior) / 2).

    Aceita escalares ou pandas Series.
    Retorna None (escalar) ou NaN (Series) quando PL médio <= 0 ou dados faltantes.
    """
    fator = _fator_anualizacao(mes)
    pl_medio = (pl_t + pl_dez_anterior) / 2

    if isinstance(pl_medio, pd.Series):
        return (fator * ll_ytd) / pl_medio.where(pl_medio > 0, np.nan)

    # Escalar
    for v in [ll_ytd, pl_t, pl_dez_anterior]:
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
    pl_medio_f = float(pl_medio)
    if pl_medio_f <= 0:
        return None
    return float(ll_ytd) * fator / pl_medio_f


def recalcular_metricas_derivadas(dados_periodos):
    """Recalcula métricas derivadas para dados carregados do cache.

    SEMPRE recalcula as métricas para garantir que estejam corretas,
    mesmo que a coluna já exista (pode ter valores NaN de cache antigo).
    """
    if not dados_periodos:
        return dados_periodos

    # Mapeamento de nomes antigos para novos
    RENOMEAR_COLUNAS = {
        'Lucro Líquido': 'Lucro Líquido Acumulado YTD',
        'ROE An. (%)': 'ROE Ac. YTD an. (%)',
        'Crédito/PL': 'Crédito/PL (%)',
    }

    # Colunas obsoletas a serem removidas
    COLUNAS_OBSOLETAS = ['Risco/Retorno', 'Funding Gap (%)']

    dados_atualizados = {}
    for periodo, df in dados_periodos.items():
        df_atualizado = df.copy()

        # Renomear colunas antigas para novos nomes
        colunas_para_renomear = {
            old: new for old, new in RENOMEAR_COLUNAS.items()
            if old in df_atualizado.columns and new not in df_atualizado.columns
        }
        if colunas_para_renomear:
            df_atualizado = df_atualizado.rename(columns=colunas_para_renomear)

        # Remover colunas obsoletas
        colunas_remover = [col for col in COLUNAS_OBSOLETAS if col in df_atualizado.columns]
        if colunas_remover:
            df_atualizado = df_atualizado.drop(columns=colunas_remover)

        # Extrair mês do período para cálculo do fator de anualização
        periodo_str = None
        if 'Período' in df_atualizado.columns and len(df_atualizado) > 0:
            periodo_str = df_atualizado['Período'].iloc[0]
        mes = _extrair_mes_periodo(periodo_str, periodo)
        if mes is None:
            mes = 12

        # ROE Anualizado com PL Médio - SEMPRE recalcular
        # Fórmula: (LL_YTD × fator) / ((PL_t + PL_Dez_ano_anterior) / 2)
        # Se PL médio <= 0 ou dados faltantes → N/A.
        #
        # IMPORTANTE — Acumulação do LL para Set (período 3 / mês 9):
        # O BCB (Relatório 1) publica o LL de Set como Jul-Set (Q3, 3 meses),
        # NÃO como YTD Jan-Set. Para obter o YTD, somamos o valor de Jun
        # (que é Jan-Jun acumulado, 6 meses): YTD = Q3 + Jun = Jan-Set (9 meses).
        # Essa acumulação é feita UMA ÚNICA VEZ aqui; downstream
        # (_ajustar_lucro_acumulado_peers, etc.) lê o valor já acumulado.
        if "Lucro Líquido Acumulado YTD" in df_atualizado.columns and "Patrimônio Líquido" in df_atualizado.columns:
            ll_col = "Lucro Líquido Acumulado YTD"
            pl_col = "Patrimônio Líquido"
            ll_ytd = df_atualizado[ll_col].copy()

            # Para Set (mes=9): LL do BCB é só Q3 (Jul-Set); somar Jun (YTD Jan-Jun)
            if mes == 9:
                parsed = _parse_periodo(periodo_str) if periodo_str else None
                if parsed:
                    parte_p, ano_p, _ = parsed
                    per_jun = f"2/{ano_p}" if 1 <= int(parte_p) <= 4 else f"6/{ano_p}"
                    df_jun = dados_periodos.get(per_jun)
                    if df_jun is not None and ll_col in df_jun.columns and "Instituição" in df_jun.columns:
                        jun_map = df_jun.set_index("Instituição")[ll_col]
                        if "Instituição" in df_atualizado.columns:
                            ll_ytd = ll_ytd + df_atualizado["Instituição"].map(jun_map)

            # PL médio: (PL_t + PL_Dez_ano_anterior) / 2
            pl_t = df_atualizado[pl_col].copy()
            pl_dez_anterior = pd.Series(np.nan, index=df_atualizado.index)
            parsed_per = _parse_periodo(periodo_str) if periodo_str else _parse_periodo(periodo)
            if parsed_per and "Instituição" in df_atualizado.columns:
                _, ano_p, ano_len_p = parsed_per
                ano_ant = ano_p - 1
                per_dez = f"4/{str(ano_ant)[-2:]}" if ano_len_p == 2 else f"4/{ano_ant}"
                df_dez = dados_periodos.get(per_dez)
                if df_dez is not None and pl_col in df_dez.columns and "Instituição" in df_dez.columns:
                    pl_dez_anterior = df_atualizado["Instituição"].map(
                        df_dez.set_index("Instituição")[pl_col]
                    )

            df_atualizado["ROE Ac. YTD an. (%)"] = _calcular_roe_anualizado(
                ll_ytd, pl_t, pl_dez_anterior, mes
            )
            # Armazenar LL YTD acumulado (Set: Q3 + Jun) na coluna para uso downstream
            df_atualizado[ll_col] = ll_ytd

        # Crédito/PL - SEMPRE recalcular
        if "Carteira de Crédito" in df_atualizado.columns and "Patrimônio Líquido" in df_atualizado.columns:
            df_atualizado["Crédito/PL (%)"] = (
                df_atualizado["Carteira de Crédito"].fillna(0) /
                df_atualizado["Patrimônio Líquido"].replace(0, np.nan)
            )

        # Crédito/Captações - SEMPRE recalcular
        if "Carteira de Crédito" in df_atualizado.columns and "Captações" in df_atualizado.columns:
            df_atualizado["Crédito/Captações (%)"] = (
                df_atualizado["Carteira de Crédito"].fillna(0) /
                df_atualizado["Captações"].replace(0, np.nan)
            )

        # Crédito/Ativo (%) - SEMPRE recalcular
        if "Carteira de Crédito" in df_atualizado.columns and "Ativo Total" in df_atualizado.columns:
            df_atualizado["Crédito/Ativo (%)"] = (
                df_atualizado["Carteira de Crédito"].fillna(0) /
                df_atualizado["Ativo Total"].replace(0, np.nan)
            )

        # Migrar nome antigo Carteira/Ativo (%) para Crédito/Ativo (%) se existir
        if "Carteira/Ativo (%)" in df_atualizado.columns and "Crédito/Ativo (%)" not in df_atualizado.columns:
            df_atualizado["Crédito/Ativo (%)"] = df_atualizado["Carteira/Ativo (%)"]

        dados_atualizados[periodo] = df_atualizado

    return dados_atualizados


def _extrair_mes_periodo(periodo_preferencial: Optional[str], periodo_fallback: Optional[str]) -> Optional[int]:
    for periodo_raw in (periodo_preferencial, periodo_fallback):
        if not periodo_raw:
            continue
        periodo_str = str(periodo_raw)
        if "/" in periodo_str:
            parte = periodo_str.split("/")[0].strip()
            if parte.isdigit():
                parte_int = int(parte)
                if 1 <= parte_int <= 4:
                    return {1: 3, 2: 6, 3: 9, 4: 12}.get(parte_int)
                if 1 <= parte_int <= 12:
                    return parte_int
        if periodo_str.isdigit() and len(periodo_str) >= 6:
            mes = periodo_str[4:6]
            if mes.isdigit():
                return int(mes)
    return None

# Variáveis de capital que serão mescladas com os dados principais
VARS_CAPITAL_MERGE = [
    'Capital Principal',
    'Capital Complementar',
    'Capital Nível II',
    'RWA Total',
    'RWA Crédito',
    'RWA Mercado',
    'RWA Operacional',
    'Exposição Total',
    'Índice de Capital Principal',
    'Índice de Capital Nível I',
    'Razão de Alavancagem',
    'Adicional de Capital Principal',
]

def normalizar_colunas_capital(df_capital: pd.DataFrame) -> pd.DataFrame:
    mapa_colunas = {
        'Capital Principal': ['Capital Principal', 'Capital Principal para Comparação com RWA (a)'],
        'RWA Total': [
            'RWA Total',
            'Ativos Ponderados pelo Risco (RWA) (j) = (f) + (g) + (h) + (i)',
            'Ativos Ponderados pelo Risco (RWA) (j)',
            'RWA',
        ],
    }
    df_capital = df_capital.copy()
    for nome_padrao, alternativas in mapa_colunas.items():
        if nome_padrao in df_capital.columns:
            continue
        alternativa = next((col for col in alternativas if col in df_capital.columns), None)
        if alternativa:
            df_capital = df_capital.rename(columns={alternativa: nome_padrao})
    return df_capital

def mesclar_dados_capital(dados_periodos, dados_capital):
    """Mescla dados de capital (Relatório 5) com os dados principais.

    Permite que as variáveis de capital estejam disponíveis nas abas
    Resumo, Análise Individual, Série Histórica, Scatter Plot, Deltas e Brincar.
    """
    if not dados_periodos or not dados_capital:
        return dados_periodos

    dados_mesclados = {}

    for periodo, df_principal in dados_periodos.items():
        df_merged = df_principal.copy()

        # Encontrar período correspondente no cache de capital
        # O formato pode ser diferente, então precisamos normalizar
        if periodo in dados_capital:
            df_capital = dados_capital[periodo]
        else:
            # Tentar formatos alternativos (ex: "03/2024" vs "202403")
            periodo_alt = None
            for p in dados_capital.keys():
                # Normalizar ambos para comparação
                p_norm = p.replace('/', '')
                periodo_norm = periodo.replace('/', '')
                if p_norm == periodo_norm or p == periodo:
                    periodo_alt = p
                    break
            if periodo_alt:
                df_capital = dados_capital[periodo_alt]
            else:
                dados_mesclados[periodo] = df_merged
                continue

        df_capital = normalizar_colunas_capital(df_capital)

        # Fazer merge por nome da instituição
        if 'Instituição' in df_capital.columns:
            # Selecionar apenas as colunas de capital que queremos mesclar
            colunas_para_merge = ['Instituição'] + [c for c in VARS_CAPITAL_MERGE if c in df_capital.columns]

            if len(colunas_para_merge) > 1:  # Tem pelo menos uma variável de capital
                df_capital_subset = df_capital[colunas_para_merge].drop_duplicates(subset=['Instituição'])

                # Fazer merge preservando os dados principais
                df_merged = df_merged.merge(
                    df_capital_subset,
                    on='Instituição',
                    how='left',
                    suffixes=('', '_capital')
                )

                # Se houver colunas duplicadas, manter a versão do capital (mais atualizada/específica)
                for col in VARS_CAPITAL_MERGE:
                    if f'{col}_capital' in df_merged.columns:
                        df_merged[col] = df_merged[f'{col}_capital']
                        df_merged = df_merged.drop(columns=[f'{col}_capital'])

        dados_mesclados[periodo] = df_merged

    # Renomear coluna para display (CET1) sem afetar chaves internas do cache
    for periodo, df_m in dados_mesclados.items():
        if 'Índice de Capital Principal' in df_m.columns:
            df_m.rename(columns={'Índice de Capital Principal': 'Índice de Capital Principal (CET1)'}, inplace=True)

    return dados_mesclados

def ler_info_cache():
    cache_manager = get_cache_manager()
    cache = cache_manager.get_cache("principal") if cache_manager else None
    if not cache or not cache.arquivo_metadata.exists():
        return None

    try:
        metadata = json.loads(cache.arquivo_metadata.read_text())
    except Exception:
        return None

    linhas = []
    if metadata.get("timestamp_salvamento"):
        try:
            ts = datetime.fromisoformat(metadata["timestamp_salvamento"])
            linhas.append(f"Última extração: {ts.strftime('%d/%m/%Y %H:%M')}")
        except ValueError:
            linhas.append(f"Última extração: {metadata['timestamp_salvamento']}")

    if metadata.get("total_periodos") is not None:
        linhas.append(f"Total de períodos: {metadata.get('total_periodos')}")

    if metadata.get("fonte"):
        linhas.append(f"Fonte: {metadata.get('fonte')}")

    return "\n".join(linhas) if linhas else None

def forcar_recarregar_cache():
    """Força o recarregamento do cache do disco, ignorando session_state."""
    dados = carregar_cache()
    if dados:
        dados = recalcular_metricas_derivadas(dados)
        if 'dict_aliases' in st.session_state:
            mapa_codigos = None
            periodos_disponiveis = sorted(dados.keys())
            if periodos_disponiveis:
                mapa_codigos = construir_mapa_codinst(periodos_disponiveis[-1])
            dados = aplicar_aliases_em_periodos(
                dados,
                st.session_state['dict_aliases'],
                mapa_codigos=mapa_codigos,
            )
        st.session_state['dados_periodos'] = dados
        st.session_state['cache_fonte'] = 'local (recarregado)'
        return True
    return False

def upload_cache_github(cache_manager: CacheManager, tipo_cache: str, gh_token: str = None) -> Tuple[bool, str]:
    """Publica cache (parquet + metadata) no GitHub Releases.

    Retorna (sucesso, mensagem).
    """
    cache = cache_manager.get_cache(tipo_cache) if cache_manager else None
    if cache is None:
        return False, f"cache '{tipo_cache}' não encontrado"

    data_path = cache.arquivo_dados
    metadata_path = cache.arquivo_metadata

    if not data_path.exists():
        return False, f"arquivo parquet não encontrado para '{tipo_cache}' (gere o cache com pyarrow)"
    if not metadata_path.exists():
        return False, f"metadata.json não encontrada para '{tipo_cache}'"

    repo = os.getenv("TOMACONTA_RELEASE_REPO", "abalroar/tomaconta")
    tag = "v1.0-cache"
    asset_data_name = f"{tipo_cache}_dados.parquet"
    asset_metadata_name = f"{tipo_cache}_metadata.json"

    cache_size = data_path.stat().st_size

    try:
        result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True, timeout=10)
        gh_available = result.returncode == 0

        if gh_available:
            subprocess.run(
                ['gh', 'release', 'delete-asset', tag, asset_data_name, '-y', '-R', repo],
                capture_output=True, text=True, timeout=30
            )
            subprocess.run(
                ['gh', 'release', 'delete-asset', tag, asset_metadata_name, '-y', '-R', repo],
                capture_output=True, text=True, timeout=30
            )

            result = subprocess.run(
                ['gh', 'release', 'upload', tag, f"{data_path}#{asset_data_name}", '--clobber', '-R', repo],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return False, f"erro ao fazer upload do cache: {result.stderr}"

            subprocess.run(
                ['gh', 'release', 'upload', tag, f"{metadata_path}#{asset_metadata_name}", '--clobber', '-R', repo],
                capture_output=True, text=True, timeout=30
            )

            return True, f"cache '{tipo_cache}' enviado ({cache_size / 1024 / 1024:.1f} MB)"

    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        return False, "timeout ao executar gh CLI"
    except Exception as e:
        return False, f"erro ao usar gh CLI: {str(e)}"

    if gh_token:
        try:
            headers = {
                'Authorization': f'token {gh_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
            r = requests.get(release_url, headers=headers, timeout=30)
            if r.status_code != 200:
                return False, f"release '{tag}' não encontrada no github"

            release_data = r.json()
            upload_url = release_data['upload_url'].replace('{?name,label}', '')

            for asset in release_data.get('assets', []):
                if asset['name'] in [asset_data_name, asset_metadata_name]:
                    delete_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
                    requests.delete(delete_url, headers=headers, timeout=30)

            upload_headers = {
                'Authorization': f'token {gh_token}',
                'Content-Type': 'application/octet-stream'
            }

            with open(data_path, 'rb') as f:
                r = requests.post(
                    f"{upload_url}?name={asset_data_name}",
                    headers=upload_headers,
                    data=f,
                    timeout=300
                )
                if r.status_code not in [200, 201]:
                    return False, f"erro ao fazer upload do cache: {r.status_code}"

            with open(metadata_path, 'rb') as f:
                r = requests.post(
                    f"{upload_url}?name={asset_metadata_name}",
                    headers=upload_headers,
                    data=f,
                    timeout=60
                )
                if r.status_code not in [200, 201]:
                    return False, f"erro ao fazer upload da metadata: {r.status_code}"

            return True, f"cache '{tipo_cache}' enviado ({cache_size / 1024 / 1024:.1f} MB)"

        except Exception as e:
            return False, f"erro ao usar API do github: {str(e)}"

    return False, "gh CLI não disponível e nenhum token fornecido"


def preparar_download_cache_local(cache_manager: CacheManager, tipo_cache: str) -> Optional[dict]:
    """Prepara bytes do cache local para download (parquet ou pickle)."""
    cache = cache_manager.get_cache(tipo_cache) if cache_manager else None
    if cache is None:
        return None

    if cache.arquivo_dados.exists():
        data_path = cache.arquivo_dados
    elif cache.arquivo_dados_pickle.exists():
        data_path = cache.arquivo_dados_pickle
    else:
        return None

    return {
        "data": data_path.read_bytes(),
        "file_name": f"{tipo_cache}_cache{data_path.suffix}",
        "mime": "application/octet-stream",
    }

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_aliases():
    """Carrega aliases do Excel com cache de 1 hora."""
    _perf_start("carregar_aliases")
    if os.path.exists(ALIASES_PATH):
        df = pd.read_excel(ALIASES_PATH)
        print(_perf_log("carregar_aliases"))
        return df
    print(_perf_log("carregar_aliases"))
    return None

# FIX PROBLEMA 3: Normalização de nomes de instituições
def normalizar_nome_instituicao(nome):
    """Normaliza nome removendo espaços extras e convertendo para uppercase"""
    if pd.isna(nome):
        return ""
    return " ".join(str(nome).split()).upper()

@st.cache_data(ttl=3600, show_spinner=False)
def construir_dict_aliases_normalizado(_df_aliases_hash: str, df_aliases_data: tuple):
    """Constrói dicionário de aliases com nomes normalizados para lookup robusto.

    Usa hash do dataframe para cache (evita recomputar a cada rerun).
    Mapeia tanto o nome original (Instituição) quanto variantes normalizadas
    para o alias amigável (Alias Banco).
    """
    _perf_start("construir_dict_aliases")
    dict_norm = {}
    if not df_aliases_data:
        print(_perf_log("construir_dict_aliases"))
        return dict_norm

    # Reconstruir DataFrame a partir da tupla (para cache funcionar)
    instituicoes, aliases = df_aliases_data

    for i in range(len(instituicoes)):
        instituicao = instituicoes[i]
        alias = aliases[i]

        if pd.notna(instituicao) and pd.notna(alias):
            # Mapeamento direto
            dict_norm[instituicao] = alias
            # Mapeamento normalizado (uppercase, sem espaços extras)
            dict_norm[normalizar_nome_instituicao(instituicao)] = alias
            # Mapeamento sem acentos (simplificado)
            nome_simples = instituicao.upper().strip()
            dict_norm[nome_simples] = alias

    print(_perf_log("construir_dict_aliases"))
    return dict_norm


def _preparar_aliases_para_cache(df_aliases):
    """Prepara dados do DataFrame para funções cacheadas."""
    if df_aliases is None or df_aliases.empty:
        return "", ()
    # Hash baseado no conteúdo
    content_hash = str(hash(tuple(df_aliases['Instituição'].fillna('').tolist())))
    # Dados como tupla (hashável)
    instituicoes = tuple(df_aliases['Instituição'].tolist())
    aliases = tuple(df_aliases['Alias Banco'].tolist())
    return content_hash, (instituicoes, aliases)

def aplicar_aliases_em_periodos(dados_periodos, dict_aliases, mapa_codigos=None):
    if not dados_periodos:
        return dados_periodos
    dados_corrigidos = {}

    for periodo, df in dados_periodos.items():
        if 'Instituição' not in df.columns:
            dados_corrigidos[periodo] = df
            continue

        df_corrigido = df.copy()

        if mapa_codigos:
            df_corrigido['Instituição'] = df_corrigido['Instituição'].apply(
                lambda nome: mapa_codigos.get(str(nome).strip(), nome) if pd.notna(nome) else nome
            )

        df_corrigido['Instituição'] = df_corrigido['Instituição'].apply(
            lambda nome: dict_aliases.get(nome, nome) if pd.notna(nome) else nome
        )

        dados_corrigidos[periodo] = df_corrigido

    return dados_corrigidos


def resolver_nomes_instituicoes_capital(df_capital, dict_aliases, df_aliases=None, dados_periodos=None):
    """Resolve nomes de instituições que estão como códigos [IF xxxxx] no cache de capital.

    Usa múltiplas estratégias de lookup:
    1. Busca direta no dict_aliases (nome original -> alias)
    2. Busca por CodInst no df_aliases se disponível
    3. Busca no dados_periodos (cache principal) como fallback
    4. Mantém o nome original se não encontrar correspondência

    Args:
        df_capital: DataFrame com dados de capital
        dict_aliases: Dicionário nome original -> alias
        df_aliases: DataFrame do Aliases.xlsx com mapeamentos
        dados_periodos: Dicionário do cache principal para fallback de nomes

    Returns:
        DataFrame com nomes de instituições resolvidos
    """
    import re

    if df_capital is None or df_capital.empty:
        return df_capital

    if 'Instituição' not in df_capital.columns:
        return df_capital

    df_result = df_capital.copy()

    # Construir mapa CodInst -> Nome a partir do df_aliases se disponível
    mapa_codinst_nome = {}
    if df_aliases is not None and not df_aliases.empty:
        # Verificar se há coluna de código no aliases
        colunas_codigo = ['CodInst', 'Código', 'Cod', 'CNPJ']
        col_codigo = None
        for col in colunas_codigo:
            if col in df_aliases.columns:
                col_codigo = col
                break

        if col_codigo and 'Alias Banco' in df_aliases.columns:
            for _, row in df_aliases.iterrows():
                cod = row.get(col_codigo)
                alias = row.get('Alias Banco')
                if pd.notna(cod) and pd.notna(alias):
                    mapa_codinst_nome[str(cod).strip()] = alias

    # Construir mapa adicional a partir do dados_periodos (cache principal) como fallback
    mapa_dados_periodos = {}
    if dados_periodos:
        for periodo, df_periodo in dados_periodos.items():
            if 'Instituição' in df_periodo.columns and 'CodInst' in df_periodo.columns:
                for _, row in df_periodo.iterrows():
                    cod = row.get('CodInst')
                    nome = row.get('Instituição')
                    # Só usar se o nome não for um placeholder
                    if pd.notna(cod) and pd.notna(nome) and not str(nome).startswith('[IF'):
                        cod_str = str(int(cod)) if isinstance(cod, float) else str(cod)
                        if cod_str not in mapa_dados_periodos:
                            mapa_dados_periodos[cod_str] = nome

    def resolver_nome(nome, codinst=None):
        """Resolve um nome de instituição."""
        if pd.isna(nome):
            return nome

        nome_str = str(nome).strip()

        # 1. Busca direta no dict_aliases
        if dict_aliases and nome_str in dict_aliases:
            return dict_aliases[nome_str]

        # 2. Verificar se é um placeholder [IF xxxxx] - agora aceita alfanuméricos
        match = re.match(r'\[IF\s+([A-Za-z0-9]+)\]', nome_str)
        if match:
            cod_extraido = match.group(1)

            # Tentar resolver pelo código extraído no mapa de aliases
            if mapa_codinst_nome and cod_extraido in mapa_codinst_nome:
                return mapa_codinst_nome[cod_extraido]

            # Tentar resolver no mapa do dados_periodos
            if mapa_dados_periodos and cod_extraido in mapa_dados_periodos:
                return mapa_dados_periodos[cod_extraido]

            # Se temos CodInst na linha, usar para lookup
            if codinst is not None and pd.notna(codinst):
                cod_str = str(int(codinst)) if isinstance(codinst, float) else str(codinst)
                if mapa_codinst_nome and cod_str in mapa_codinst_nome:
                    return mapa_codinst_nome[cod_str]
                if mapa_dados_periodos and cod_str in mapa_dados_periodos:
                    return mapa_dados_periodos[cod_str]

        # 3. Se temos CodInst, tentar lookup direto
        if codinst is not None and pd.notna(codinst):
            cod_str = str(int(codinst)) if isinstance(codinst, float) else str(codinst)
            if mapa_codinst_nome and cod_str in mapa_codinst_nome:
                return mapa_codinst_nome[cod_str]
            if mapa_dados_periodos and cod_str in mapa_dados_periodos:
                return mapa_dados_periodos[cod_str]

        # 4. Manter nome original se não encontrou correspondência
        return nome_str

    # Aplicar resolução de nomes
    if 'CodInst' in df_result.columns:
        df_result['Instituição'] = df_result.apply(
            lambda row: resolver_nome(row['Instituição'], row.get('CodInst')),
            axis=1
        )
    else:
        df_result['Instituição'] = df_result['Instituição'].apply(
            lambda nome: resolver_nome(nome)
        )

    return df_result



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
@st.cache_data(ttl=3600, show_spinner=False)
def carregar_cores_aliases_local(_df_hash: str, cores_data: tuple):
    """Lê a cor do Aliases.xlsx e cria um dicionário de cores.

    Usa hash do dataframe para cache (evita recomputar a cada rerun).
    Importante: mapeia tanto o valor da coluna 'Instituição' (nome original vindo do BCB)
    quanto o valor da coluna 'Alias Banco' (nome amigável que aparece no app),
    para que a cor seja aplicada em qualquer tela.
    """
    _perf_start("carregar_cores_aliases")
    dict_cores = {}
    if not cores_data:
        print(_perf_log("carregar_cores_aliases"))
        return dict_cores

    instituicoes, aliases, cores = cores_data

    for i in range(len(instituicoes)):
        instituicao = instituicoes[i]
        alias = aliases[i]
        cor_valor = cores[i]

        cor_str = normalizar_codigo_cor(cor_valor)
        if not cor_str:
            continue

        if pd.notna(instituicao):
            dict_cores[normalizar_nome_instituicao(instituicao)] = cor_str
            # Também aceita busca pelo 'Alias Banco' (útil quando a coluna Instituição já vem renomeada)
            if pd.notna(alias):
                dict_cores[normalizar_nome_instituicao(alias)] = cor_str

        # Também mapeia pelo alias (é o que aparece na UI)
        if pd.notna(alias):
            dict_cores[normalizar_nome_instituicao(alias)] = cor_str

    print(_perf_log("carregar_cores_aliases"))
    return dict_cores


def _preparar_cores_para_cache(df_aliases):
    """Prepara dados de cores do DataFrame para função cacheada."""
    if df_aliases is None or df_aliases.empty:
        return "", ()

    # Encontrar coluna de cor
    colunas_possiveis = ['Código Cor', 'Cor', 'Color', 'Hex', 'Código']
    coluna_cor = None
    for col in colunas_possiveis:
        if col in df_aliases.columns:
            coluna_cor = col
            break

    if coluna_cor is None:
        return "", ()

    # Hash baseado no conteúdo
    content_hash = str(hash(tuple(df_aliases['Instituição'].fillna('').tolist())))
    # Dados como tupla (hashável)
    instituicoes = tuple(df_aliases['Instituição'].tolist())
    aliases = tuple(df_aliases['Alias Banco'].tolist())
    cores = tuple(df_aliases[coluna_cor].tolist())
    return content_hash, (instituicoes, aliases, cores)

def verificar_caches_github() -> dict:
    """Verifica quais caches existem no GitHub Releases.

    Retorna dict com status de cada cache no GitHub (sem autenticação, apenas leitura pública).
    Verifica todos os 8 tipos de cache disponíveis.
    """
    repo = os.getenv("TOMACONTA_RELEASE_REPO", "abalroar/tomaconta")
    tag = "v1.0-cache"

    # Todos os tipos de cache
    tipos_cache = ['principal', 'capital', 'ativo', 'passivo', 'dre',
                   'carteira_pf', 'carteira_pj', 'carteira_instrumentos']

    result = {
        'release_existe': False,
        'repo': repo,
        'tag': tag,
        'erro': None,
        'caches': {}
    }

    # Inicializar todos os caches como não existentes
    for tipo in tipos_cache:
        result['caches'][tipo] = {'existe': False, 'tamanho': 0, 'tamanho_fmt': 'N/A'}

    # Manter compatibilidade com código antigo
    result['cache_principal'] = result['caches']['principal']
    result['cache_capital'] = result['caches']['capital']

    try:
        release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        r = requests.get(release_url, timeout=10)

        if r.status_code == 404:
            result['erro'] = f"Release '{tag}' não encontrada"
            return result
        elif r.status_code != 200:
            result['erro'] = f"Erro ao acessar release: {r.status_code}"
            return result

        result['release_existe'] = True
        release_data = r.json()

        for asset in release_data.get('assets', []):
            size = asset.get('size', 0)
            size_fmt = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
            nome_asset = asset.get('name', '')

            # Identificar tipo de cache pelo nome do asset
            for tipo in tipos_cache:
                if nome_asset.startswith(f'{tipo}_dados'):
                    result['caches'][tipo] = {
                        'existe': True,
                        'tamanho': size,
                        'tamanho_fmt': size_fmt,
                        'nome_asset': nome_asset
                    }
                    break

        # Atualizar referências de compatibilidade
        result['cache_principal'] = result['caches']['principal']
        result['cache_capital'] = result['caches']['capital']

    except requests.exceptions.Timeout:
        result['erro'] = "Timeout ao verificar GitHub"
    except Exception as e:
        result['erro'] = str(e)

    return result


def ordenar_periodos(periodos, reverso=False):
    def chave_periodo(valor):
        try:
            mes, ano = valor.split('/')
            return (int(ano), int(mes))
        except (ValueError, AttributeError):
            return (0, 0)

    return sorted(periodos, key=chave_periodo, reverse=reverso)


def periodo_para_exibicao(periodo_trimestre: str) -> str:
    """Converte período trimestral (1/2025) para formato mês abreviado (Mar/25).

    Args:
        periodo_trimestre: Período no formato "trimestre/ano" (ex: "1/2025")

    Returns:
        Período no formato "Mês/AA" (ex: "Mar/25")
    """
    if not periodo_trimestre or '/' not in str(periodo_trimestre):
        return str(periodo_trimestre)
    try:
        trimestre, ano = str(periodo_trimestre).split('/')
        meses_map = {'1': 'Mar', '2': 'Jun', '3': 'Set', '4': 'Dez'}
        mes = meses_map.get(trimestre.strip(), trimestre)
        ano_curto = ano[-2:] if len(ano) == 4 else ano
        return f"{mes}/{ano_curto}"
    except Exception:
        return str(periodo_trimestre)


def formatar_periodos_lista(periodos: list) -> list:
    """Converte lista de períodos para formato de exibição (Mar/25).

    Args:
        periodos: Lista de períodos no formato "trimestre/ano"

    Returns:
        Lista de períodos formatados
    """
    return [periodo_para_exibicao(p) for p in periodos]


def ordenar_bancos_com_alias(bancos: list, dict_aliases: dict = None) -> list:
    """Ordena bancos com alias primeiro (A-Z), depois sem alias (A-Z).

    Args:
        bancos: Lista de nomes de bancos
        dict_aliases: Dicionário de aliases {nome_original: alias}

    Returns:
        Lista ordenada de bancos
    """
    if not dict_aliases:
        return sorted(bancos)

    aliases_set = set(dict_aliases.values())

    bancos_com_alias = []
    bancos_sem_alias = []

    for banco in bancos:
        if banco in aliases_set:
            bancos_com_alias.append(banco)
        else:
            bancos_sem_alias.append(banco)

    def sort_key(nome):
        primeiro_char = nome[0].lower() if nome else 'z'
        if primeiro_char.isdigit():
            return (1, nome.lower())
        return (0, nome.lower())

    bancos_com_alias_sorted = sorted(bancos_com_alias, key=sort_key)
    bancos_sem_alias_sorted = sorted(bancos_sem_alias, key=sort_key)

    return bancos_com_alias_sorted + bancos_sem_alias_sorted


# --- Defaults helpers ---

# Nomes-alvo para defaults (slug normalizado → possíveis matches)
_BANCOS_DEFAULT_SLUGS = [
    ("itau", "itaú"),
    ("santander",),
    ("banco do brasil", "bb"),
    ("btg", "btg pactual"),
    ("bradesco",),
]


def _encontrar_bancos_default(bancos_disponiveis: list, slugs=None) -> list:
    """Encontra bancos nos disponíveis por matching fuzzy de substrings."""
    if slugs is None:
        slugs = _BANCOS_DEFAULT_SLUGS
    resultado = []
    for grupo in slugs:
        encontrado = None
        for slug in grupo:
            for banco in bancos_disponiveis:
                if slug == banco.lower() or slug in banco.lower():
                    encontrado = banco
                    break
            if encontrado:
                break
        if encontrado and encontrado not in resultado:
            resultado.append(encontrado)
    return resultado


def _encontrar_periodo(periodos: list, trimestre: int, ano: int) -> Optional[str]:
    """Encontra um período na lista pelo trimestre e ano."""
    for p in periodos:
        if f"{trimestre}/{ano}" == p or f"{trimestre}/{str(ano)[-2:]}" == p:
            return p
    return None


def _is_variavel_percentual(variavel: str) -> bool:
    if not variavel:
        return False
    if variavel in VARS_PERCENTUAL:
        return True
    return "Basileia" in variavel


def _formatar_percentual(valor, decimais: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        valor_float = float(valor)
    except Exception:
        return "N/A"
    valor_float *= 100
    return f"{valor_float:.{decimais}f}%"


def formatar_valor(valor, variavel):
    if pd.isna(valor) or valor == 0:
        return "N/A"

    if _is_variavel_percentual(variavel):
        return _formatar_percentual(valor)
    elif variavel in VARS_RAZAO:
        return f"{valor:.2f}x"
    elif variavel in VARS_MOEDAS:
        valor_mm = valor / 1e6
        return f"R$ {valor_mm:,.0f}MM".replace(",", ".")
    elif variavel in VARS_CONTAGEM:
        return f"{valor:,.0f}".replace(",", ".")
    else:
        return f"{valor:.2f}"

def get_axis_format(variavel, serie: Optional[pd.Series] = None):
    if _is_variavel_percentual(variavel):
        return {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
    elif variavel in VARS_MOEDAS:
        return {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
    elif variavel in VARS_CONTAGEM:
        return {'tickformat': ',.0f', 'ticksuffix': '', 'multiplicador': 1}
    else:
        return {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}


def _normalizar_percentual_display(serie: pd.Series) -> pd.Series:
    serie_num = pd.to_numeric(serie, errors="coerce")
    if serie_num.empty:
        return serie_num
    return serie_num * 100


def _calcular_valores_display(serie: pd.Series, variavel: str, format_info: dict) -> pd.Series:
    if _is_variavel_percentual(variavel):
        return _normalizar_percentual_display(serie)
    return serie * format_info['multiplicador']


def _normalizar_label_peers(texto: str) -> str:
    if texto is None:
        return ""
    return (
        str(texto)
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
        .lower()
        .replace(".", "")
        .replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def _resolver_coluna_peers(df: pd.DataFrame, candidatos: list) -> Optional[str]:
    if df is None or df.empty:
        return None
    for candidato in candidatos:
        if candidato in df.columns:
            return candidato
    candidatos_norm = [_normalizar_label_peers(c) for c in candidatos]
    for col in df.columns:
        col_norm = _normalizar_label_peers(col)
        if col_norm in candidatos_norm:
            return col
    for col in df.columns:
        col_norm = _normalizar_label_peers(col)
        if any(cand in col_norm for cand in candidatos_norm):
            return col
    return None


def _formatar_valor_peers(valor, format_key: str, coluna_origem: Optional[str] = None) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    if format_key == "Ativo/PL":
        try:
            return f"{float(valor):.2f}x"
        except Exception:
            return "—"
    if format_key == "Crédito/PL (%)":
        try:
            return f"{float(valor):.2f}x"
        except Exception:
            return "—"
    if coluna_origem and "(%)" in coluna_origem:
        try:
            return _formatar_percentual(float(valor))
        except Exception:
            return "—"
    return formatar_valor(valor, format_key)


def _fmt_tooltip_mm(valor) -> str:
    """Formata valor numérico como R$ MM para tooltip."""
    if valor is None:
        return "N/A"
    try:
        v = float(valor)
        if pd.isna(v):
            return "N/A"
        return f"R$ {v / 1e6:,.0f} MM".replace(",", ".")
    except Exception:
        return "N/A"


def _tooltip_roe_peers(df, banco, periodo, coluna_lucro, coluna_pl, valor_roe):
    """Tooltip com memória de cálculo do ROE Ac. Anualizado."""
    lucro = _obter_valor_peers(df, banco, periodo, coluna_lucro)
    lucro_num = _coerce_numeric_value(lucro)
    pl_t = _coerce_numeric_value(_obter_valor_peers(df, banco, periodo, coluna_pl))
    parsed = _parse_periodo(periodo)
    if not parsed:
        return ""
    _, ano, ano_len = parsed
    ano_ant = ano - 1
    per_dez = f"4/{str(ano_ant)[-2:]}" if ano_len == 2 else f"4/{ano_ant}"
    pl_dez = _coerce_numeric_value(_obter_valor_peers(df, banco, per_dez, coluna_pl))
    mes = _extrair_mes_periodo(periodo, periodo)
    fator = _fator_anualizacao(mes) if mes else 1
    per_exib = periodo_para_exibicao(periodo)
    dez_exib = periodo_para_exibicao(per_dez)
    fmt = _fmt_tooltip_mm
    lines = [
        f"LL YTD: {fmt(lucro_num)}",
        f"Fator: x{fator:.4g} (12/{mes})",
        f"PL ({per_exib}): {fmt(pl_t)}",
        f"PL ({dez_exib}): {fmt(pl_dez)}",
    ]
    if pl_t is not None and pl_dez is not None and not pd.isna(pl_t) and not pd.isna(pl_dez):
        pl_medio = (float(pl_t) + float(pl_dez)) / 2
        lines.append(f"PL Medio: {fmt(pl_medio)}")
    if valor_roe is not None and not pd.isna(valor_roe):
        v = float(valor_roe)
        if abs(v) <= 1:
            v *= 100
        lines.append(f"ROE = {v:.2f}%")
    else:
        lines.append("ROE = N/A")
    return "\n".join(lines)


def _tooltip_ll_peers(df, banco, periodo, coluna, valor):
    """Tooltip para Lucro Líquido (BCB Rel. 1 já publica YTD acumulado)."""
    fmt = _fmt_tooltip_mm
    return f"LL YTD: {fmt(valor)}"


def _tooltip_ratio_peers(label, valor_num, valor_den, valor_ratio):
    """Tooltip para métricas do tipo razão (Ativo/PL, Crédito/PL, ratios %)."""
    fmt = _fmt_tooltip_mm
    _NOMES_COMPONENTES = {
        "Ativo / PL": ("Ativo Total", "PL"),
        "Crédito / PL": ("Carteira de Crédito", "PL"),
        "Perda Esperada / Carteira Bruta": ("Perda Esperada", "Carteira Bruta"),
        "Carteira de Créd. Class. C4+C5 / Carteira Bruta": ("C4+C5", "Carteira Bruta"),
        "Perda Esperada / (Carteira C4 + C5)": ("Perda Esperada", "C4+C5"),
        "Desp PDD Anualizada / Carteira Bruta": ("Desp PDD Anual.", "Carteira Bruta"),
    }
    lines = []
    if label in _NOMES_COMPONENTES:
        n_name, d_name = _NOMES_COMPONENTES[label]
        lines.append(f"{n_name}: {fmt(valor_num)}")
        lines.append(f"{d_name}: {fmt(valor_den)}")
    else:
        lines.append(f"Numerador: {fmt(valor_num)}")
        lines.append(f"Denominador: {fmt(valor_den)}")
    if valor_ratio is not None and not pd.isna(valor_ratio):
        # Leverage ratios (x/PL) show as Nx; other ratios as %
        if "/ PL" in label or "/PL" in label:
            lines.append(f"= {float(valor_ratio):.2f}x")
        else:
            lines.append(f"= {float(valor_ratio) * 100:.2f}%")
    else:
        lines.append("= N/A")
    return "\n".join(lines)


def _parse_periodo(periodo: str) -> Optional[Tuple[str, int, int]]:
    if not periodo or "/" not in str(periodo):
        return None
    partes = str(periodo).split("/")
    if len(partes) != 2:
        return None
    parte, ano_str = partes
    try:
        ano = int(ano_str)
    except ValueError:
        return None
    return parte, ano, len(ano_str)


def _periodo_ano_anterior(periodo: str) -> Optional[str]:
    parsed = _parse_periodo(periodo)
    if not parsed:
        return None
    parte, ano, ano_len = parsed
    ano_anterior = ano - 1
    if ano_len == 2:
        return f"{parte}/{str(ano_anterior)[-2:]}"
    return f"{parte}/{ano_anterior}"


def _periodo_mesma_estrutura(periodo: str, novo_parte: int) -> Optional[str]:
    parsed = _parse_periodo(periodo)
    if not parsed:
        return None
    parte, ano, ano_len = parsed
    nova_parte = str(novo_parte).zfill(len(str(parte)))
    if ano_len == 2:
        return f"{nova_parte}/{str(ano)[-2:]}"
    return f"{nova_parte}/{ano}"


def _obter_valor_peers(df: pd.DataFrame, banco: str, periodo: str, coluna: Optional[str]):
    if coluna is None or df is None or df.empty:
        return None
    df_cell = df[(df["Instituição"] == banco) & (df["Período"] == periodo)]
    if df_cell.empty:
        return None
    return df_cell.iloc[0].get(coluna)


def _coerce_numeric_value(valor):
    if valor is None or pd.isna(valor):
        return None
    if isinstance(valor, str):
        cleaned = valor.replace(".", "").replace(",", ".")
        valor = cleaned
    return pd.to_numeric(valor, errors="coerce")


def _somar_valores(valores: list) -> Optional[float]:
    numeros = []
    for valor in valores:
        val_num = _coerce_numeric_value(valor)
        if val_num is not None and not pd.isna(val_num):
            numeros.append(float(val_num))
    if not numeros:
        return None
    return float(sum(numeros))


def _aplicar_aliases_df(df: Optional[pd.DataFrame], dict_aliases: dict) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not dict_aliases:
        return df
    df_out = df.copy()
    if "Instituição" in df_out.columns:
        df_out["Instituição"] = df_out["Instituição"].apply(
            lambda nome: dict_aliases.get(nome, dict_aliases.get(normalizar_nome_instituicao(nome), nome))
            if pd.notna(nome) else nome
        )
    return df_out


@st.cache_data(ttl=3600, show_spinner=False)
def _carregar_cache_relatorio(tipo_cache: str) -> Optional[pd.DataFrame]:
    manager = get_cache_manager()
    if manager is None:
        return None
    resultado = manager.carregar(tipo_cache)
    if resultado.sucesso and resultado.dados is not None:
        return resultado.dados
    return None


def _preparar_metricas_extra_peers(
    bancos: list,
    periodos: list,
    cache_ativo: Optional[pd.DataFrame],
    cache_passivo: Optional[pd.DataFrame],
    cache_carteira_pf: Optional[pd.DataFrame],
    cache_carteira_pj: Optional[pd.DataFrame],
    cache_carteira_instr: Optional[pd.DataFrame],
    cache_dre: Optional[pd.DataFrame],
    cache_capital: Optional[pd.DataFrame] = None,
) -> dict:
    extra = {
        "Carteira de Crédito Bruta": {},
        "Ativos Líquidos": {},
        "Depósitos Totais": {},
        "Perda Esperada": {},
        "Perda Esperada / Carteira Bruta": {},
        "Carteira de Créd. Class. C4+C5": {},
        "Carteira de Créd. Class. C4+C5 / Carteira Bruta": {},
        "Perda Esperada / (Carteira C4 + C5)": {},
        "Desp PDD Anualizada": {},
        "Desp PDD Anualizada / Carteira Bruta": {},
        "Índice de Capital Principal (CET1)": {},
        "Índice de Basileia Total": {},
    }
    periodos_base = {_periodo_ano_anterior(periodo) for periodo in periodos}
    periodos_ext = [p for p in periodos + sorted(periodos_base) if p]

    col_pf_total = _resolver_coluna_peers(
        cache_carteira_pf,
        [
            "Total da Carteira de Pessoa Física",
            "Total da Carteira Pessoa Física",
            "Total da Carteira PF",
            "Total da Carteira de Pessoa Fisica",
        ],
    )
    col_pj_total = _resolver_coluna_peers(
        cache_carteira_pj,
        [
            "Total da Carteira de Pessoa Jurídica",
            "Total da Carteira Pessoa Jurídica",
            "Total da Carteira PJ",
            "Total da Carteira de Pessoa Juridica",
        ],
    )

    # Ativos Líquidos: Disponibilidades (a) + Aplicações Interfinanceiras de Liquidez (b)
    # + TVM (c) do relatório de Ativo (Rel. 2)
    col_disp_ativo = _resolver_coluna_peers(
        cache_ativo,
        ["Disponibilidades (a)", "Disponibilidades", "Disponibilidades (a) ="],
    )
    col_aplic_ativo = _resolver_coluna_peers(
        cache_ativo,
        [
            "Aplicações Interfinanceiras de Liquidez (b)",
            "Aplicacoes Interfinanceiras de Liquidez (b)",
            "Aplicações Interfinanceiras de Liquidez",
        ],
    )
    col_tvm_ativo = _resolver_coluna_peers(
        cache_ativo,
        [
            "Títulos e Valores Mobiliários (c)",
            "Titulos e Valores Mobiliarios (c)",
            "Títulos e Valores Mobiliários e Instrumentos Financeiros Derivativos (c)",
            "Títulos e Valores Mobiliários",
        ],
    )

    # Depósitos Totais: Depósitos (a) do relatório de Passivo (Rel. 3)
    # Nota: BCB mudou nome de "Depósito Total (a)" (até 2024) para "Depósitos (a)" (2025+)
    col_depositos_passivo = _resolver_coluna_peers(
        cache_passivo,
        [
            "Depósitos (a)",
            "Depósito Total (a)",
            "Depositos (a)",
            "Deposito Total (a)",
            "Depósitos",
            "Depositos",
        ],
    )

    perda_colunas_base = [
        "Perda Esperada (e2)",
        "Hedge de Valor Justo (e3)",
        "Ajuste a Valor Justo (e4)",
        "Perda Esperada (f2)",
        "Hedge de Valor Justo (f3)",
        "Perda Esperada (g2)",
        "Hedge de Valor Justo (g3)",
        "Ajuste a Valor Justo (g4)",
        "Perda Esperada (h2)",
    ]
    perda_colunas = []
    for coluna in perda_colunas_base:
        col_resolvida = _resolver_coluna_peers(cache_ativo, [coluna])
        if col_resolvida and col_resolvida not in perda_colunas:
            perda_colunas.append(col_resolvida)

    col_c4 = _resolver_coluna_peers(cache_carteira_instr, ["C4"])
    col_c5 = _resolver_coluna_peers(cache_carteira_instr, ["C5"])

    col_desp_pdd = _resolver_coluna_peers(
        cache_dre,
        [
            "Resultado com Perda Esperada (f)",
            "Resultado com Perda Esperada",
            "Desp. PDD",
            "Despesa com Perda Esperada",
        ],
    )

    # Capital: colunas para Índice de Capital Principal e Índice de Basileia Total
    col_cap_principal = _resolver_coluna_peers(
        cache_capital,
        ["Capital Principal", "Capital Principal para Comparação com RWA (a)"],
    )
    col_cap_complementar = _resolver_coluna_peers(
        cache_capital,
        ["Capital Complementar", "Capital Complementar (b)"],
    )
    col_cap_nivel2 = _resolver_coluna_peers(
        cache_capital,
        ["Capital Nível II", "Capital Nível II (d)", "Capital Nivel II"],
    )
    col_rwa_total = _resolver_coluna_peers(
        cache_capital,
        [
            "RWA Total",
            "Ativos Ponderados pelo Risco (RWA) (j) = (f) + (g) + (h) + (i)",
            "Ativos Ponderados pelo Risco (RWA) (j)",
            "RWA",
        ],
    )
    col_indice_cap_principal = _resolver_coluna_peers(
        cache_capital,
        [
            "Índice de Capital Principal",
            "Índice de Capital Principal (l) = (a) / (j)",
        ],
    )
    col_indice_basileia_precalc = _resolver_coluna_peers(
        cache_capital,
        [
            "Índice de Basileia",
            "Índice de Basileia Capital",
            "Índice de Basileia (n) = (e) / (j)",
        ],
    )

    for banco in bancos:
        for periodo in periodos_ext:
            chave = (banco, periodo)
            valor_pf = _obter_valor_peers(cache_carteira_pf, banco, periodo, col_pf_total)
            valor_pj = _obter_valor_peers(cache_carteira_pj, banco, periodo, col_pj_total)
            carteira_bruta = _somar_valores([valor_pf, valor_pj])
            extra["Carteira de Crédito Bruta"][chave] = carteira_bruta

            # Ativos Líquidos = Disponibilidades (a) + Aplicações Interfinanceiras (b) + TVM (c)
            # do relatório de Ativo (Rel. 2)
            ativos_liquidos = _somar_valores([
                _obter_valor_peers(cache_ativo, banco, periodo, col_disp_ativo),
                _obter_valor_peers(cache_ativo, banco, periodo, col_aplic_ativo),
                _obter_valor_peers(cache_ativo, banco, periodo, col_tvm_ativo),
            ])
            extra["Ativos Líquidos"][chave] = ativos_liquidos

            # Depósitos Totais = Depósitos (a) do relatório de Passivo (Rel. 3)
            depositos_totais = _obter_valor_peers(cache_passivo, banco, periodo, col_depositos_passivo)
            extra["Depósitos Totais"][chave] = _coerce_numeric_value(depositos_totais)

            perda_vals = [
                _obter_valor_peers(cache_ativo, banco, periodo, col)
                for col in perda_colunas
            ]
            perda_esperada = _somar_valores(perda_vals)
            extra["Perda Esperada"][chave] = perda_esperada
            extra["Perda Esperada / Carteira Bruta"][chave] = _calcular_ratio_peers(perda_esperada, carteira_bruta)

            valor_c4 = _obter_valor_peers(cache_carteira_instr, banco, periodo, col_c4)
            valor_c5 = _obter_valor_peers(cache_carteira_instr, banco, periodo, col_c5)
            carteira_c4_c5 = _somar_valores([valor_c4, valor_c5])
            extra["Carteira de Créd. Class. C4+C5"][chave] = carteira_c4_c5
            extra["Carteira de Créd. Class. C4+C5 / Carteira Bruta"][chave] = _calcular_ratio_peers(
                carteira_c4_c5,
                carteira_bruta,
            )
            extra["Perda Esperada / (Carteira C4 + C5)"][chave] = _calcular_ratio_peers(
                perda_esperada,
                carteira_c4_c5,
            )

            # DRE (Rel. 4): Set/Dez são semestrais; acumular YTD antes de anualizar
            desp_pdd_ytd = _acumular_dre_ytd_peers(cache_dre, banco, periodo, col_desp_pdd)
            desp_pdd_anual = _anualizar_valor_dre(desp_pdd_ytd, periodo)
            extra["Desp PDD Anualizada"][chave] = desp_pdd_anual
            extra["Desp PDD Anualizada / Carteira Bruta"][chave] = _calcular_ratio_peers(
                desp_pdd_anual,
                carteira_bruta,
            )

            # Capital: Índice de Capital Principal e Índice de Basileia Total
            # Prioridade: calcular da composição (Capital Principal / RWA);
            # fallback: usar valor pré-calculado do cache de capital.
            indice_cap_principal = None
            if col_cap_principal and col_rwa_total:
                val_cp = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_cap_principal)
                )
                val_rwa = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_rwa_total)
                )
                if val_cp is not None and val_rwa is not None and not pd.isna(val_cp) and not pd.isna(val_rwa) and float(val_rwa) != 0:
                    indice_cap_principal = float(val_cp) / float(val_rwa)
            if indice_cap_principal is None and col_indice_cap_principal:
                val_precalc = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_indice_cap_principal)
                )
                if val_precalc is not None and not pd.isna(val_precalc):
                    indice_cap_principal = float(val_precalc)
            extra["Índice de Capital Principal (CET1)"][chave] = indice_cap_principal

            indice_basileia = None
            if col_cap_principal and col_cap_complementar and col_cap_nivel2 and col_rwa_total:
                val_cp = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_cap_principal)
                )
                val_cc = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_cap_complementar)
                )
                val_n2 = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_cap_nivel2)
                )
                val_rwa = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_rwa_total)
                )
                if (
                    val_cp is not None and val_cc is not None and val_n2 is not None and val_rwa is not None
                    and not pd.isna(val_cp) and not pd.isna(val_cc) and not pd.isna(val_n2) and not pd.isna(val_rwa)
                    and float(val_rwa) != 0
                ):
                    indice_basileia = (float(val_cp) + float(val_cc) + float(val_n2)) / float(val_rwa)
            if indice_basileia is None and col_indice_basileia_precalc:
                val_precalc = _coerce_numeric_value(
                    _obter_valor_peers(cache_capital, banco, periodo, col_indice_basileia_precalc)
                )
                if val_precalc is not None and not pd.isna(val_precalc):
                    indice_basileia = float(val_precalc)
            extra["Índice de Basileia Total"][chave] = indice_basileia

    return extra


def _calcular_ratio_peers(valor_num, valor_den) -> Optional[float]:
    if valor_num is None or valor_den is None:
        return None
    if pd.isna(valor_num) or pd.isna(valor_den):
        return None
    try:
        valor_den_float = float(valor_den)
        if valor_den_float == 0:
            return None
        return float(valor_num) / valor_den_float
    except Exception:
        return None


def _acumular_dre_ytd_peers(
    cache_dre: Optional[pd.DataFrame],
    banco: str,
    periodo: str,
    coluna: Optional[str],
) -> Optional[float]:
    """Converte valor DRE bruto para YTD acumulado.

    O BCB (Relatório 4) publica dados DRE de forma semestral:
      - Mar (parte=1): Jan-Mar → já é YTD (3 meses)
      - Jun (parte=2): Jan-Jun → já é YTD (6 meses)
      - Set (parte=3): Jul-Set → precisa somar Jun (Jan-Jun) para YTD (9 meses)
      - Dez (parte=4): Jul-Dez → precisa somar Jun (Jan-Jun) para YTD (12 meses)

    Retorna valor numérico YTD ou None se dados insuficientes.
    """
    valor_raw = _obter_valor_peers(cache_dre, banco, periodo, coluna)
    if valor_raw is None or pd.isna(valor_raw):
        return None
    valor_num = _coerce_numeric_value(valor_raw)
    if valor_num is None or pd.isna(valor_num):
        return None
    parsed = _parse_periodo(periodo)
    if not parsed:
        return float(valor_num)
    parte, _, _ = parsed
    try:
        parte_int = int(parte)
    except ValueError:
        return float(valor_num)
    # Determinar se precisa acumular (Set e Dez precisam de Jun)
    if 1 <= parte_int <= 4:
        precisa_jun = parte_int in (3, 4)
        parte_jun = 2
    elif 1 <= parte_int <= 12:
        precisa_jun = parte_int in (9, 12)
        parte_jun = 6
    else:
        return float(valor_num)
    if not precisa_jun:
        return float(valor_num)
    periodo_jun = _periodo_mesma_estrutura(periodo, parte_jun)
    if not periodo_jun:
        return None
    valor_jun = _obter_valor_peers(cache_dre, banco, periodo_jun, coluna)
    if valor_jun is None or pd.isna(valor_jun):
        return None
    valor_jun_num = _coerce_numeric_value(valor_jun)
    if valor_jun_num is None or pd.isna(valor_jun_num):
        return None
    return float(valor_num) + float(valor_jun_num)


def _anualizar_valor_dre(valor, periodo: str) -> Optional[float]:
    """Anualiza valor DRE já acumulado YTD: valor_ytd / meses * 12."""
    if valor is None or pd.isna(valor):
        return None
    valor_num = _coerce_numeric_value(valor)
    if valor_num is None or pd.isna(valor_num):
        return None
    parsed = _parse_periodo(periodo)
    if not parsed:
        return float(valor_num)
    parte, _, _ = parsed
    try:
        parte_int = int(parte)
    except ValueError:
        return float(valor_num)
    if 1 <= parte_int <= 4:
        meses = parte_int * 3
    elif 1 <= parte_int <= 12:
        meses = parte_int
    else:
        meses = None
    if not meses:
        return float(valor_num)
    if meses == 0:
        return None
    return float(valor_num) / meses * 12


def _ajustar_lucro_acumulado_peers(
    df: pd.DataFrame,
    banco: str,
    periodo: str,
    coluna: Optional[str],
):
    """Retorna LL YTD do período.

    A acumulação Set (Q3 + Jun) já foi feita em recalcular_metricas_derivadas(),
    que sobrescreve a coluna 'Lucro Líquido Acumulado YTD' com o valor YTD.
    Aqui apenas lemos o valor já acumulado — NÃO somar Jun novamente.
    """
    return _obter_valor_peers(df, banco, periodo, coluna)


def _calcular_roe_anualizado_peers(
    df: pd.DataFrame,
    banco: str,
    periodo: str,
    coluna_lucro: Optional[str],
    coluna_pl: Optional[str],
) -> Optional[float]:
    """ROE Ac. Anualizado = (LL_YTD × fator) / ((PL_t + PL_Dez_anterior) / 2).

    Usa PL médio entre período atual (t) e Dez do ano anterior.
    Retorna None quando PL médio <= 0 ou qualquer componente faltar.
    O LL já está acumulado YTD no DataFrame (feito em recalcular_metricas_derivadas).
    """
    lucro = _ajustar_lucro_acumulado_peers(df, banco, periodo, coluna_lucro)
    lucro_num = _coerce_numeric_value(lucro)
    if lucro_num is None or pd.isna(lucro_num):
        return None

    # PL_t
    pl_t = _coerce_numeric_value(_obter_valor_peers(df, banco, periodo, coluna_pl))
    if pl_t is None or pd.isna(pl_t):
        return None

    # PL_Dez_ano_anterior
    parsed = _parse_periodo(periodo)
    if not parsed:
        return None
    _, ano, ano_len = parsed
    ano_ant = ano - 1
    per_dez = f"4/{str(ano_ant)[-2:]}" if ano_len == 2 else f"4/{ano_ant}"
    pl_dez = _coerce_numeric_value(_obter_valor_peers(df, banco, per_dez, coluna_pl))
    if pl_dez is None or pd.isna(pl_dez):
        return None

    mes = _extrair_mes_periodo(periodo, periodo)
    return _calcular_roe_anualizado(float(lucro_num), float(pl_t), float(pl_dez), mes)


def _montar_tabela_peers(
    df: pd.DataFrame,
    bancos: list,
    periodos: list,
    caches_extras: Optional[dict] = None,
):
    """Monta estrutura da tabela peers com valores por banco/período."""
    valores = {}
    colunas_usadas = {}
    faltas = set()
    delta_flags = {}
    tooltips = {}
    coluna_ativo = _resolver_coluna_peers(df, ["Ativo Total"])
    coluna_pl = _resolver_coluna_peers(df, ["Patrimônio Líquido"])
    coluna_lucro = _resolver_coluna_peers(df, ["Lucro Líquido Acumulado YTD", "Lucro Líquido"])
    extra_values = _preparar_metricas_extra_peers(
        bancos,
        periodos,
        (caches_extras or {}).get("ativo"),
        (caches_extras or {}).get("passivo"),
        (caches_extras or {}).get("carteira_pf"),
        (caches_extras or {}).get("carteira_pj"),
        (caches_extras or {}).get("carteira_instrumentos"),
        (caches_extras or {}).get("dre"),
        (caches_extras or {}).get("capital"),
    )

    for section in PEERS_TABELA_LAYOUT:
        for row in section["rows"]:
            label = row["label"]
            candidatos = row.get("data_keys", [])
            coluna = _resolver_coluna_peers(df, candidatos) if candidatos else None
            if label == "Ativo / PL":
                coluna = None
            colunas_usadas[label] = coluna
            if coluna is None and row.get("todo"):
                faltas.add(label)

            for banco in bancos:
                for periodo in periodos:
                    chave = (label, banco, periodo)
                    valor = None
                    tip = ""
                    # Mapeamento de ratios → (chave numerador, chave denominador)
                    _RATIO_COMPONENTS = {
                        "Perda Esperada / Carteira Bruta": ("Perda Esperada", "Carteira de Crédito Bruta"),
                        "Carteira de Créd. Class. C4+C5 / Carteira Bruta": ("Carteira de Créd. Class. C4+C5", "Carteira de Crédito Bruta"),
                        "Perda Esperada / (Carteira C4 + C5)": ("Perda Esperada", "Carteira de Créd. Class. C4+C5"),
                        "Desp PDD Anualizada / Carteira Bruta": ("Desp PDD Anualizada", "Carteira de Crédito Bruta"),
                    }
                    if label in extra_values and label in _RATIO_COMPONENTS:
                        valor = extra_values[label].get((banco, periodo))
                        num_key, den_key = _RATIO_COMPONENTS[label]
                        valor_num = extra_values.get(num_key, {}).get((banco, periodo))
                        valor_den = extra_values.get(den_key, {}).get((banco, periodo))
                        tip = _tooltip_ratio_peers(label, valor_num, valor_den, valor)
                    elif label in extra_values:
                        valor = extra_values[label].get((banco, periodo))
                        if label in ("Índice de Capital Principal (CET1)", "Índice de Basileia Total"):
                            if valor is not None and not pd.isna(valor):
                                tip = f"{label}: {float(valor) * 100:.2f}%"
                            else:
                                tip = f"{label}: N/A"
                        else:
                            tip = f"{label}: {_fmt_tooltip_mm(valor)}" if valor is not None else ""
                    elif label == "ROE AC. Anualizado (%)":
                        valor = _calcular_roe_anualizado_peers(
                            df,
                            banco,
                            periodo,
                            coluna_lucro,
                            coluna_pl,
                        )
                        tip = _tooltip_roe_peers(df, banco, periodo, coluna_lucro, coluna_pl, valor)
                    elif label == "Ativo / PL":
                        valor_ativo = _obter_valor_peers(df, banco, periodo, coluna_ativo)
                        valor_pl = _obter_valor_peers(df, banco, periodo, coluna_pl)
                        valor = _calcular_ratio_peers(valor_ativo, valor_pl)
                        tip = _tooltip_ratio_peers(label, valor_ativo, valor_pl, valor)
                    elif label == "Crédito / PL":
                        coluna_credito = _resolver_coluna_peers(df, ["Carteira de Crédito"])
                        valor_credito = _obter_valor_peers(df, banco, periodo, coluna_credito)
                        valor_pl_v = _obter_valor_peers(df, banco, periodo, coluna_pl)
                        valor = _calcular_ratio_peers(valor_credito, valor_pl_v)
                        tip = _tooltip_ratio_peers(label, valor_credito, valor_pl_v, valor)
                    elif coluna:
                        if label == "Lucro Líquido Acumulado":
                            valor = _ajustar_lucro_acumulado_peers(df, banco, periodo, coluna)
                            tip = _tooltip_ll_peers(df, banco, periodo, coluna, valor)
                        else:
                            valor = _obter_valor_peers(df, banco, periodo, coluna)
                            tip = f"{coluna}: {_fmt_tooltip_mm(valor)}" if valor is not None else ""
                    valores[chave] = valor
                    tooltips[chave] = tip

                    delta_flag = None
                    periodo_base = _periodo_ano_anterior(periodo)
                    if periodo_base and label in extra_values:
                        valor_base = extra_values[label].get((banco, periodo_base))
                    elif periodo_base and label == "ROE AC. Anualizado (%)":
                        valor_base = _calcular_roe_anualizado_peers(
                            df,
                            banco,
                            periodo_base,
                            coluna_lucro,
                            coluna_pl,
                        )
                    elif periodo_base and coluna:
                        if label == "Lucro Líquido Acumulado":
                            valor_base = _ajustar_lucro_acumulado_peers(df, banco, periodo_base, coluna)
                        else:
                            valor_base = _obter_valor_peers(df, banco, periodo_base, coluna)
                    elif periodo_base and label == "Ativo / PL":
                        valor_ativo_base = _obter_valor_peers(df, banco, periodo_base, coluna_ativo)
                        valor_pl_base = _obter_valor_peers(df, banco, periodo_base, coluna_pl)
                        valor_base = _calcular_ratio_peers(valor_ativo_base, valor_pl_base)
                    elif periodo_base and label == "Crédito / PL":
                        coluna_credito_b = _resolver_coluna_peers(df, ["Carteira de Crédito"])
                        valor_credito_b = _obter_valor_peers(df, banco, periodo_base, coluna_credito_b)
                        valor_pl_b = _obter_valor_peers(df, banco, periodo_base, coluna_pl)
                        valor_base = _calcular_ratio_peers(valor_credito_b, valor_pl_b)
                    else:
                        valor_base = None
                    if valor_base is not None and valor is not None and not pd.isna(valor) and not pd.isna(valor_base):
                        delta = valor - valor_base
                        if delta > 0:
                            delta_flag = "up"
                        elif delta < 0:
                            delta_flag = "down"
                    delta_flags[chave] = delta_flag

    return valores, colunas_usadas, faltas, delta_flags, tooltips


def _render_peers_table_html(
    bancos: list,
    periodos: list,
    valores: dict,
    colunas_usadas: dict,
    delta_flags: dict,
    tooltips: Optional[dict] = None,
):
    colunas_total = 1 + len(bancos) * len(periodos)
    html = """
    <style>
    .peers-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin-top: 10px;
        table-layout: auto;
    }
    .peers-table th, .peers-table td {
        border: 1px solid #ddd;
        padding: 6px 10px;
        text-align: right;
        vertical-align: top;
        white-space: nowrap;
    }
    .peers-table th {
        background-color: #f5f5f5;
        font-weight: 600;
        text-align: center;
        white-space: normal;
    }
    .peers-table td:first-child {
        text-align: left;
        font-weight: 500;
        white-space: nowrap;
        width: 1%;
        padding-right: 8px;
    }
    .peers-table thead tr:first-child th {
        background-color: #4a4a4a;
        color: white;
    }
    .peers-table thead tr:nth-child(2) th {
        background-color: #6a6a6a;
        color: white;
    }
    .peer-section {
        background-color: #4a90e2;
        color: white;
        font-weight: 600;
        text-align: left !important;
    }
    .peer-item td:first-child {
        padding-left: 18px;
        font-weight: 400;
    }
    .peer-zebra {
        background-color: #f8f9fa;
    }
    .delta-pos { color: #28a745; margin-left: 4px; }
    .delta-neg { color: #dc3545; margin-left: 4px; }
    .peers-table td.has-tip {
        position: relative;
        cursor: help;
    }
    .peers-table .tip-text {
        display: none;
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: #fff;
        padding: 8px 10px;
        border-radius: 4px;
        font-size: 11px;
        white-space: normal;
        z-index: 9999;
        min-width: 220px;
        max-width: 340px;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        pointer-events: none;
        line-height: 1.5;
    }
    .peers-table td.has-tip:hover .tip-text {
        display: block;
    }
    </style>
    <table class="peers-table">
    <thead>
    <tr>
        <th rowspan="2">R$ MM e %</th>
    """

    for banco in bancos:
        html += f'<th colspan="{len(periodos)}">{banco}</th>'
    html += "</tr><tr>"

    for _ in bancos:
        for periodo in periodos:
            html += f"<th>{periodo_para_exibicao(periodo)}</th>"
    html += "</tr></thead><tbody>"

    zebra_idx = 0
    for section in PEERS_TABELA_LAYOUT:
        html += f'<tr><td class="peer-section" colspan="{colunas_total}">{section["section"]}</td></tr>'
        for row in section["rows"]:
            zebra_class = "peer-zebra" if zebra_idx % 2 == 0 else ""
            zebra_idx += 1
            html += f'<tr class="peer-item {zebra_class}"><td>{row["label"]}</td>'

            for banco in bancos:
                for periodo in periodos:
                    chave = (row["label"], banco, periodo)
                    coluna = colunas_usadas.get(row["label"])
                    valor = valores.get(chave)
                    valor_fmt = _formatar_valor_peers(valor, row["format_key"], coluna_origem=coluna)

                    delta_html = ""
                    delta_flag = delta_flags.get(chave)
                    if delta_flag == "up":
                        delta_html = ' <span class="delta-pos">▲</span>'
                    elif delta_flag == "down":
                        delta_html = ' <span class="delta-neg">▼</span>'
                    tip = (tooltips or {}).get(chave, "") if tooltips else ""
                    if tip:
                        tip_html = _html_mod.escape(tip).replace("\n", "<br>")
                        html += f'<td class="has-tip">{valor_fmt}{delta_html}<span class="tip-text">{tip_html}</span></td>'
                    else:
                        html += f"<td>{valor_fmt}{delta_html}</td>"

            html += "</tr>"

    html += "</tbody></table>"
    return html


def _gerar_imagem_peers_tabela(
    bancos: list,
    periodos: list,
    valores: dict,
    colunas_usadas: dict,
    delta_flags: dict,
    scale: float = 1.0,
):
    """Gera imagem PNG da tabela peers para exportação."""
    header_row_1 = ["R$ MM e %"]
    for banco in bancos:
        for idx in range(len(periodos)):
            header_row_1.append(banco if idx == 0 else "")
    header_row_2 = [""]
    for _ in bancos:
        for periodo in periodos:
            header_row_2.append(periodo_para_exibicao(periodo))

    rows = [header_row_1, header_row_2]
    delta_flags_rows = []
    row_styles = ["header", "subheader"]

    for section in PEERS_TABELA_LAYOUT:
        rows.append([section["section"]] + [""] * (len(bancos) * len(periodos)))
        row_styles.append("section")
        delta_flags_rows.append([None] * len(rows[-1]))

        for row in section["rows"]:
            linha = [row["label"]]
            deltas = [None]
            for banco in bancos:
                for periodo in periodos:
                    chave = (row["label"], banco, periodo)
                    coluna = colunas_usadas.get(row["label"])
                    valor = valores.get(chave)
                    valor_fmt = _formatar_valor_peers(valor, row["format_key"], coluna_origem=coluna)
                    delta_flag = None
                    delta_flag = delta_flags.get(chave)
                    if delta_flag == "up":
                        valor_fmt = f"{valor_fmt} ▲"
                    elif delta_flag == "down":
                        valor_fmt = f"{valor_fmt} ▼"
                    linha.append(valor_fmt)
                    deltas.append(delta_flag)
            rows.append(linha)
            row_styles.append("data")
            delta_flags_rows.append(deltas)

    n_rows = len(rows)
    n_cols = len(rows[0])

    col_widths = []
    for col_idx in range(n_cols):
        max_len = max(len(str(row[col_idx])) for row in rows)
        base = 0.16 if col_idx == 0 else 0.12
        col_widths.append(max(base, min(0.35, max_len * 0.012)))

    fig_width = max(10, sum(col_widths) * 10 * scale)
    fig_height = max(3, n_rows * 0.32 * scale)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        cellLoc="right",
        colWidths=col_widths,
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10 * scale)
    table.scale(1, 1.2 * scale)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_idx == 0:
            cell.set_facecolor("#4a4a4a")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        elif row_idx == 1:
            cell.set_facecolor("#6a6a6a")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        else:
            style = row_styles[row_idx]
            if style == "section":
                cell.set_facecolor("#4a90e2")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                if col_idx == 0:
                    cell.get_text().set_ha("left")
                else:
                    cell.get_text().set_text("")
            else:
                cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f8f9fa")
                if col_idx == 0:
                    cell.get_text().set_ha("left")
                delta_flag = delta_flags_rows[row_idx - 2][col_idx] if row_idx >= 2 else None
                if delta_flag == "up":
                    cell.get_text().set_color("#28a745")
                elif delta_flag == "down":
                    cell.get_text().set_color("#dc3545")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(180 * scale), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _gerar_excel_peers_tabela(
    bancos: list,
    periodos: list,
    valores: dict,
    colunas_usadas: dict,
    delta_flags: dict,
) -> BytesIO:
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    worksheet = workbook.add_worksheet("peers_tabela")

    n_cols = 1 + len(bancos) * len(periodos)
    border = {"border": 1, "border_color": "#dddddd"}
    header_fmt = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#4a4a4a", "font_color": "white", **border}
    )
    subheader_fmt = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#6a6a6a", "font_color": "white", **border}
    )
    section_fmt = workbook.add_format(
        {"bold": True, "align": "left", "valign": "vcenter", "bg_color": "#4a90e2", "font_color": "white", **border}
    )
    row_even = workbook.add_format({"align": "right", "valign": "vcenter", "bg_color": "#f8f9fa", **border})
    row_odd = workbook.add_format({"align": "right", "valign": "vcenter", "bg_color": "#ffffff", **border})
    row_even_label = workbook.add_format({"align": "left", "valign": "vcenter", "bg_color": "#f8f9fa", **border})
    row_odd_label = workbook.add_format({"align": "left", "valign": "vcenter", "bg_color": "#ffffff", **border})
    row_even_up = workbook.add_format(
        {"align": "right", "valign": "vcenter", "bg_color": "#f8f9fa", "font_color": "#28a745", **border}
    )
    row_even_down = workbook.add_format(
        {"align": "right", "valign": "vcenter", "bg_color": "#f8f9fa", "font_color": "#dc3545", **border}
    )
    row_odd_up = workbook.add_format(
        {"align": "right", "valign": "vcenter", "bg_color": "#ffffff", "font_color": "#28a745", **border}
    )
    row_odd_down = workbook.add_format(
        {"align": "right", "valign": "vcenter", "bg_color": "#ffffff", "font_color": "#dc3545", **border}
    )

    worksheet.set_column(0, 0, 34)
    worksheet.set_column(1, max(1, n_cols - 1), 16)

    row_idx = 0
    worksheet.write(row_idx, 0, "R$ MM e %", header_fmt)
    col_idx = 1
    for banco in bancos:
        start_col = col_idx
        end_col = col_idx + len(periodos) - 1
        if start_col <= end_col:
            worksheet.merge_range(row_idx, start_col, row_idx, end_col, banco, header_fmt)
        col_idx = end_col + 1
    row_idx += 1

    worksheet.write(row_idx, 0, "", subheader_fmt)
    col_idx = 1
    for _ in bancos:
        for periodo in periodos:
            worksheet.write(row_idx, col_idx, periodo_para_exibicao(periodo), subheader_fmt)
            col_idx += 1
    row_idx += 1

    zebra_idx = 0
    for section in PEERS_TABELA_LAYOUT:
        worksheet.merge_range(row_idx, 0, row_idx, n_cols - 1, section["section"], section_fmt)
        row_idx += 1
        for row in section["rows"]:
            is_even = zebra_idx % 2 == 0
            label_fmt = row_even_label if is_even else row_odd_label
            base_fmt = row_even if is_even else row_odd
            base_fmt_up = row_even_up if is_even else row_odd_up
            base_fmt_down = row_even_down if is_even else row_odd_down

            worksheet.write(row_idx, 0, row["label"], label_fmt)
            col_idx = 1
            for banco in bancos:
                for periodo in periodos:
                    chave = (row["label"], banco, periodo)
                    coluna = colunas_usadas.get(row["label"])
                    valor = valores.get(chave)
                    valor_fmt = _formatar_valor_peers(valor, row["format_key"], coluna_origem=coluna)
                    delta_flag = delta_flags.get(chave)
                    cell_fmt = base_fmt
                    if delta_flag == "up":
                        valor_fmt = f"{valor_fmt} ▲"
                        cell_fmt = base_fmt_up
                    elif delta_flag == "down":
                        valor_fmt = f"{valor_fmt} ▼"
                        cell_fmt = base_fmt_down
                    worksheet.write(row_idx, col_idx, valor_fmt, cell_fmt)
                    col_idx += 1
            row_idx += 1
            zebra_idx += 1

    workbook.close()
    output.seek(0)
    return output


def _gerar_excel_peers_dados_puros(
    bancos: list,
    periodos: list,
    valores: dict,
) -> BytesIO:
    """Exporta tabela Peers no mesmo layout matricial, mas com valores numéricos puros."""
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    worksheet = workbook.add_worksheet("dados_puros")

    n_cols = 1 + len(bancos) * len(periodos)
    border = {"border": 1, "border_color": "#dddddd"}
    header_fmt = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#4a4a4a", "font_color": "white", **border}
    )
    subheader_fmt = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#6a6a6a", "font_color": "white", **border}
    )
    section_fmt = workbook.add_format(
        {"bold": True, "align": "left", "valign": "vcenter", "bg_color": "#4a90e2", "font_color": "white", **border}
    )
    label_fmt = workbook.add_format({"align": "left", "valign": "vcenter", **border})
    num_fmt = workbook.add_format({"align": "right", "valign": "vcenter", "num_format": "0.000000", **border})
    empty_fmt = workbook.add_format({"align": "right", "valign": "vcenter", **border})

    worksheet.set_column(0, 0, 34)
    worksheet.set_column(1, max(1, n_cols - 1), 18)

    # Cabeçalho: bancos
    row_idx = 0
    worksheet.write(row_idx, 0, "Dados Puros", header_fmt)
    col_idx = 1
    for banco in bancos:
        start_col = col_idx
        end_col = col_idx + len(periodos) - 1
        if start_col <= end_col:
            worksheet.merge_range(row_idx, start_col, row_idx, end_col, banco, header_fmt)
        col_idx = end_col + 1
    row_idx += 1

    # Sub-cabeçalho: períodos
    worksheet.write(row_idx, 0, "", subheader_fmt)
    col_idx = 1
    for _ in bancos:
        for periodo in periodos:
            worksheet.write(row_idx, col_idx, periodo_para_exibicao(periodo), subheader_fmt)
            col_idx += 1
    row_idx += 1

    # Dados por seção/indicador
    for section in PEERS_TABELA_LAYOUT:
        worksheet.merge_range(row_idx, 0, row_idx, n_cols - 1, section["section"], section_fmt)
        row_idx += 1
        for row in section["rows"]:
            worksheet.write(row_idx, 0, row["label"], label_fmt)
            col_idx = 1
            for banco in bancos:
                for periodo in periodos:
                    chave = (row["label"], banco, periodo)
                    valor = valores.get(chave)
                    if valor is not None and not pd.isna(valor):
                        worksheet.write_number(row_idx, col_idx, float(valor), num_fmt)
                    else:
                        worksheet.write(row_idx, col_idx, "", empty_fmt)
                    col_idx += 1
            row_idx += 1

    workbook.close()
    output.seek(0)
    return output


def _mapear_colunas_capital(df_capital: pd.DataFrame):
    mapa_colunas_capital = {
        'Capital Principal': ['Capital Principal', 'Capital Principal para Comparação com RWA (a)'],
        'Capital Complementar': ['Capital Complementar', 'Capital Complementar (b)'],
        'Capital Nível II': ['Capital Nível II', 'Capital Nível II (d)'],
        'RWA Total': ['RWA Total', 'Ativos Ponderados pelo Risco (RWA) (j) = (f) + (g) + (h) + (i)', 'Ativos Ponderados pelo Risco (RWA) (j)', 'RWA'],
        'Índice de Basileia Capital': ['Índice de Basileia Capital', 'Índice de Basileia (n) = (e) / (j)', 'Índice de Basileia'],
    }
    colunas_encontradas = {}
    colunas_faltantes = []
    for nome_padrao, alternativas in mapa_colunas_capital.items():
        encontrada = None
        for alt in alternativas:
            if alt in df_capital.columns:
                encontrada = alt
                break
        if encontrada:
            colunas_encontradas[nome_padrao] = encontrada
        else:
            colunas_faltantes.append(nome_padrao)
    colunas_composicao = ['Capital Principal', 'Capital Complementar', 'Capital Nível II', 'RWA Total']
    faltantes_composicao = [c for c in colunas_composicao if c in colunas_faltantes]
    tem_indice_basileia_precalc = 'Índice de Basileia Capital' in colunas_encontradas
    return colunas_encontradas, colunas_faltantes, faltantes_composicao, tem_indice_basileia_precalc


def _preparar_df_capital_base() -> pd.DataFrame:
    if 'dados_capital' not in st.session_state or not st.session_state['dados_capital']:
        return pd.DataFrame()
    df_capital = pd.concat(st.session_state['dados_capital'].values(), ignore_index=True)
    df_capital = normalizar_colunas_capital(df_capital)
    dict_aliases = st.session_state.get('dict_aliases', {})
    df_aliases = st.session_state.get('df_aliases', None)
    dados_periodos = st.session_state.get('dados_periodos', None)
    df_capital = resolver_nomes_instituicoes_capital(df_capital, dict_aliases, df_aliases, dados_periodos)
    return df_capital


def _calcular_basileia_periodo(
    df_capital: pd.DataFrame,
    periodo: str,
    colunas_encontradas: dict,
) -> Tuple[pd.DataFrame, dict]:
    df_periodo_cap = df_capital[df_capital['Período'] == periodo].copy()
    info = {"usou_precalc": False, "mensagem": None}

    pode_calcular_composicao = all(
        col in colunas_encontradas
        for col in ['Capital Principal', 'Capital Complementar', 'Capital Nível II', 'RWA Total']
    )

    if pode_calcular_composicao:
        col_capital_principal = colunas_encontradas['Capital Principal']
        col_capital_complementar = colunas_encontradas['Capital Complementar']
        col_capital_nivel2 = colunas_encontradas['Capital Nível II']
        col_rwa_total = colunas_encontradas['RWA Total']

        df_periodo_cap['RWA_valido'] = (
            df_periodo_cap[col_rwa_total].notna() &
            (df_periodo_cap[col_rwa_total] != 0)
        )

        df_periodo_cap['CET1 (%)'] = np.where(
            df_periodo_cap['RWA_valido'],
            (df_periodo_cap[col_capital_principal] / df_periodo_cap[col_rwa_total]) * 100,
            np.nan
        )
        df_periodo_cap['AT1 (%)'] = np.where(
            df_periodo_cap['RWA_valido'],
            (df_periodo_cap[col_capital_complementar] / df_periodo_cap[col_rwa_total]) * 100,
            np.nan
        )
        df_periodo_cap['T2 (%)'] = np.where(
            df_periodo_cap['RWA_valido'],
            (df_periodo_cap[col_capital_nivel2] / df_periodo_cap[col_rwa_total]) * 100,
            np.nan
        )

        df_periodo_cap['Índice de Basileia Total (%)'] = (
            df_periodo_cap['CET1 (%)'] +
            df_periodo_cap['AT1 (%)'] +
            df_periodo_cap['T2 (%)']
        )
    else:
        df_periodo_cap['Índice de Basileia Total (%)'] = np.nan
        df_periodo_cap['CET1 (%)'] = np.nan
        df_periodo_cap['AT1 (%)'] = np.nan
        df_periodo_cap['T2 (%)'] = np.nan
        df_periodo_cap['RWA_valido'] = False

    if 'Índice de Basileia Capital' in colunas_encontradas:
        col_indice_basileia = colunas_encontradas['Índice de Basileia Capital']
        valores_ib = pd.to_numeric(df_periodo_cap[col_indice_basileia], errors="coerce")
        max_ib = valores_ib.dropna().max() if not valores_ib.dropna().empty else None
        fator_ib = 100 if max_ib is not None and max_ib <= 1 else 1
        mask_preencher = df_periodo_cap['Índice de Basileia Total (%)'].isna() & valores_ib.notna()
        if mask_preencher.any():
            df_periodo_cap.loc[mask_preencher, 'Índice de Basileia Total (%)'] = valores_ib[mask_preencher] * fator_ib
            df_periodo_cap.loc[mask_preencher, 'RWA_valido'] = True
            info["usou_precalc"] = True
            if pode_calcular_composicao:
                info["mensagem"] = "ℹ️ Alguns bancos usam Índice de Basileia pré-calculado (composição CET1/AT1/T2 não disponível)"
            else:
                info["mensagem"] = "ℹ️ Usando Índice de Basileia pré-calculado (composição CET1/AT1/T2 não disponível)"
        return df_periodo_cap, info

    info["mensagem"] = "Não foi possível calcular o Índice de Basileia. Verifique se o cache possui as colunas necessárias."
    return pd.DataFrame(), info

def adicionar_indice_cet1(df_base: pd.DataFrame) -> pd.DataFrame:
    if df_base.empty or "Índice de CET1" in df_base.columns:
        return df_base
    if "CET1 (%)" in df_base.columns:
        df_base = df_base.copy()
        df_base["Índice de CET1"] = df_base["CET1 (%)"] / 100
        return df_base
    if "Capital Principal" not in df_base.columns or "RWA Total" not in df_base.columns:
        return df_base
    df_base = df_base.copy()
    df_base["Índice de CET1"] = (
        df_base["Capital Principal"] / df_base["RWA Total"].replace(0, np.nan)
    )
    return df_base


def normalizar_periodo_chave(periodo: str) -> str:
    if periodo is None:
        return ""
    periodo_str = str(periodo)
    if "/" in periodo_str:
        return periodo_str
    if periodo_str.isdigit() and len(periodo_str) == 6:
        return f"{periodo_str[4:6]}/{periodo_str[:4]}"
    if periodo_str.isdigit() and len(periodo_str) == 5:
        return f"0{periodo_str[3:5]}/{periodo_str[:4]}"
    return periodo_str


def construir_cet1_capital(
    dados_capital: dict,
    dict_aliases: Optional[dict] = None,
    df_aliases: Optional[pd.DataFrame] = None,
    dados_periodos: Optional[dict] = None,
) -> pd.DataFrame:
    if not dados_capital:
        return pd.DataFrame()
    registros = []
    for periodo, df_capital in dados_capital.items():
        if df_capital is None or df_capital.empty:
            continue
        df_capital = normalizar_colunas_capital(df_capital)
        if dict_aliases:
            df_capital = resolver_nomes_instituicoes_capital(
                df_capital, dict_aliases, df_aliases, dados_periodos
            )
        if "Instituição" not in df_capital.columns:
            continue
        if "CET1 (%)" in df_capital.columns:
            df_temp = df_capital[["Instituição", "CET1 (%)"]].copy()
            df_temp["Índice de CET1"] = df_temp["CET1 (%)"] / 100
        elif "Capital Principal" in df_capital.columns and "RWA Total" in df_capital.columns:
            df_temp = df_capital[["Instituição", "Capital Principal", "RWA Total"]].copy()
            df_temp["Índice de CET1"] = (
                df_temp["Capital Principal"] / df_temp["RWA Total"].replace(0, np.nan)
            )
        else:
            continue
        df_temp["Período"] = normalizar_periodo_chave(periodo)
        registros.append(df_temp[["Período", "Instituição", "Índice de CET1"]])
    if not registros:
        return pd.DataFrame()
    return pd.concat(registros, ignore_index=True)


def obter_cet1_periodo(
    periodo: str,
    dados_capital: dict,
    dict_aliases: dict,
    df_aliases: Optional[pd.DataFrame] = None,
    dados_periodos: Optional[dict] = None,
) -> pd.DataFrame:
    if not dados_capital:
        return pd.DataFrame()
    periodo_norm = normalizar_periodo_chave(periodo)
    chave_periodo = None
    for chave in dados_capital.keys():
        if normalizar_periodo_chave(chave) == periodo_norm:
            chave_periodo = chave
            break
    if chave_periodo is None:
        return pd.DataFrame()

    df_capital = dados_capital.get(chave_periodo)
    if df_capital is None or df_capital.empty:
        return pd.DataFrame()

    df_capital = normalizar_colunas_capital(df_capital)
    df_capital = resolver_nomes_instituicoes_capital(
        df_capital, dict_aliases, df_aliases, dados_periodos
    )

    if "CET1 (%)" in df_capital.columns:
        df_temp = df_capital[["Instituição", "CET1 (%)"]].copy()
        df_temp["Índice de CET1"] = df_temp["CET1 (%)"] / 100
    elif "Capital Principal" in df_capital.columns and "RWA Total" in df_capital.columns:
        df_temp = df_capital[["Instituição", "Capital Principal", "RWA Total"]].copy()
        df_temp["Índice de CET1"] = (
            df_temp["Capital Principal"] / df_temp["RWA Total"].replace(0, np.nan)
        )
    else:
        return pd.DataFrame()

    return df_temp[["Instituição", "Índice de CET1"]]


@st.cache_resource(show_spinner=False)
def _carregar_logo_base64(logo_path: str, target_width: int = 200) -> str:
    """Carrega e processa o logo uma única vez, retornando base64.

    Usa cache_resource para manter em memória entre reruns.
    """
    _perf_start("carregar_logo")
    if not os.path.exists(logo_path):
        print(_perf_log("carregar_logo"))
        return ""

    logo_image = PILImage.open(logo_path)
    if logo_image.width < target_width:
        ratio = target_width / logo_image.width
        new_height = int(logo_image.height * ratio)
        logo_image = logo_image.resize((target_width, new_height), PILImage.LANCZOS)
    buffer = BytesIO()
    logo_image.save(buffer, format="PNG", optimize=True)
    logo_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    print(_perf_log("carregar_logo"))
    return logo_base64


@st.cache_data(show_spinner=False)
def _get_dados_concatenados(periodos_hash: str, dados_keys: tuple) -> pd.DataFrame:
    """Concatena todos os DataFrames de períodos uma única vez.

    Evita pd.concat() repetido em cada página/rerun.
    Usa hash dos períodos para invalidar cache quando dados mudam.
    """
    _perf_start("concat_dados")
    if 'dados_periodos' not in st.session_state:
        print(_perf_log("concat_dados"))
        return pd.DataFrame()

    df = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
    print(_perf_log("concat_dados"))
    return df


def get_dados_concatenados() -> pd.DataFrame:
    """Retorna DataFrame concatenado de todos os períodos (com cache)."""
    if 'dados_periodos' not in st.session_state or not st.session_state['dados_periodos']:
        return pd.DataFrame()

    # Hash baseado nas chaves dos períodos E nas colunas para invalidação de cache
    # Isso garante que o cache seja invalidado quando métricas derivadas são recalculadas
    # ou quando dados de capital são mesclados
    periodos_keys = tuple(sorted(st.session_state['dados_periodos'].keys()))

    # Incluir colunas do primeiro DataFrame no hash para detectar mudanças
    primeiro_periodo = next(iter(st.session_state['dados_periodos'].values()))
    colunas_hash = tuple(sorted(primeiro_periodo.columns.tolist()))

    # Flag de mesclagem de capital também invalida o cache
    capital_mesclado = st.session_state.get('_dados_capital_mesclados', False)

    periodos_hash = str(hash((periodos_keys, colunas_hash, capital_mesclado)))

    return _get_dados_concatenados(periodos_hash, periodos_keys)


def _get_cache_data_mtime(cache_obj) -> Optional[float]:
    if cache_obj is None:
        return None
    if cache_obj.arquivo_dados.exists():
        return cache_obj.arquivo_dados.stat().st_mtime
    if cache_obj.arquivo_dados_pickle.exists():
        return cache_obj.arquivo_dados_pickle.stat().st_mtime
    return None


def _load_cache_metadata(cache_obj) -> dict:
    if cache_obj is None or not cache_obj.arquivo_metadata.exists():
        return {}
    try:
        return json.loads(cache_obj.arquivo_metadata.read_text())
    except Exception:
        return {}


def ensure_derived_metrics_cache() -> Tuple[Optional[object], Optional[str], dict]:
    manager = get_cache_manager()
    cache_derivado = manager.get_cache("derived_metrics") if manager else None
    if cache_derivado is None:
        return None, "cache de métricas derivadas não configurado", {}

    mtime_derivado = _get_cache_data_mtime(cache_derivado)
    dre_cache = manager.get_cache("dre")
    principal_cache = manager.get_cache("principal")
    mtime_dre = _get_cache_data_mtime(dre_cache)
    mtime_principal = _get_cache_data_mtime(principal_cache)

    precisa_recalcular = True
    if mtime_derivado is not None:
        referencia = max([t for t in [mtime_dre, mtime_principal] if t is not None], default=None)
        if referencia is None or mtime_derivado >= referencia:
            precisa_recalcular = False

    if not precisa_recalcular:
        return cache_derivado, None, _load_cache_metadata(cache_derivado)

    resultado_dre = manager.carregar("dre") if manager else None
    if not resultado_dre or not resultado_dre.sucesso or resultado_dre.dados is None:
        msg = resultado_dre.mensagem if resultado_dre else "falha ao carregar DRE"
        return cache_derivado, msg, {}

    resultado_principal = manager.carregar("principal") if manager else None
    if not resultado_principal or not resultado_principal.sucesso or resultado_principal.dados is None:
        msg = resultado_principal.mensagem if resultado_principal else "falha ao carregar principal"
        return cache_derivado, msg, {}

    _perf_start("derived_metrics_build")
    df_derived, stats = build_derived_metrics(resultado_dre.dados, resultado_principal.dados)
    elapsed = _perf_end("derived_metrics_build")

    info_extra = {
        "denominador_zero_ou_nan": stats.denominador_zero_ou_nan,
        "period_type": stats.period_type,
        "periodos_detectados": stats.periodos_detectados,
        "tempo_execucao_s": round(elapsed, 3),
    }
    cache_derivado.salvar_local(df_derived, fonte="derivado", info_extra=info_extra)
    return cache_derivado, None, _load_cache_metadata(cache_derivado)


def carregar_metricas_derivadas_slice(periodos=None, instituicoes=None, metricas=None) -> pd.DataFrame:
    cache_derivado, erro, _ = ensure_derived_metrics_cache()
    if cache_derivado is None or erro:
        return pd.DataFrame()
    return load_derived_metrics_slice(
        cache_derivado,
        periodos=periodos,
        instituicoes=instituicoes,
        metricas=metricas,
    )


def _df_mem_mb(df: Optional[pd.DataFrame]) -> float:
    if df is None or df.empty:
        return 0.0
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def anexar_metricas_derivadas_periodo(df_periodo: pd.DataFrame, periodo: str):
    if df_periodo is None or df_periodo.empty:
        return df_periodo, {}
    _perf_start("scatter_derived_load")
    df_derived = carregar_metricas_derivadas_slice(
        periodos=[periodo],
        instituicoes=df_periodo["Instituição"].dropna().unique().tolist(),
        metricas=DERIVED_METRICS,
    )
    tempo = _perf_end("scatter_derived_load")
    if df_derived.empty:
        df_out = df_periodo.copy()
        for metric in DERIVED_METRICS:
            if metric not in df_out.columns:
                df_out[metric] = pd.NA
        return df_out, {"tempo_s": round(tempo, 3), "linhas": 0, "mem_mb": 0.0}
    df_pivot = df_derived.pivot_table(
        index="Instituição",
        columns="Métrica",
        values="Valor",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None
    df_out = df_periodo.merge(df_pivot, on="Instituição", how="left")
    for metric in DERIVED_METRICS:
        if metric not in df_out.columns:
            df_out[metric] = pd.NA
    diag = {
        "tempo_s": round(tempo, 3),
        "linhas": len(df_derived),
        "mem_mb": _df_mem_mb(df_derived),
    }
    return df_out, diag


# FIX PROBLEMA 3: Busca de cor com normalização
def obter_cor_banco(instituicao):
    if 'dict_cores_personalizadas' in st.session_state:
        instituicao_norm = normalizar_nome_instituicao(instituicao)
        if instituicao_norm in st.session_state['dict_cores_personalizadas']:
            return st.session_state['dict_cores_personalizadas'][instituicao_norm]
    return None

def criar_mini_grafico(df_banco, variavel, titulo, tipo='linha'):
    df_sorted = df_banco.copy()
    if 'ano' not in df_sorted.columns:
        df_sorted['ano'] = df_sorted['Período'].str.split('/').str[1].astype(int)
        df_sorted['trimestre'] = df_sorted['Período'].str.split('/').str[0].astype(int)
    df_sorted = df_sorted.sort_values(['ano', 'trimestre'])

    instituicao = df_sorted['Instituição'].iloc[0]
    cor_banco = obter_cor_banco(instituicao)
    if not cor_banco:
        cor_banco = '#1f77b4'

    if variavel in VARS_PERCENTUAL:
        hover_values = df_sorted[variavel] * 100
        tickformat = '.2f'
        suffix = '%'
    elif variavel in VARS_MOEDAS:
        hover_values = df_sorted[variavel] / 1e6
        tickformat = ',.0f'
        suffix = 'M'
    elif variavel in VARS_CONTAGEM:
        hover_values = df_sorted[variavel]
        tickformat = ',.0f'
        suffix = ''
    else:
        hover_values = df_sorted[variavel]
        tickformat = '.2f'
        suffix = ''

    fig = go.Figure()

    if tipo == 'barra':
        fig.add_trace(go.Bar(
            x=df_sorted['Período'],
            y=hover_values,
            marker=dict(color=cor_banco, opacity=0.8),
            hovertemplate='%{x}<br>%{y:' + tickformat + '}' + suffix + '<extra></extra>'
        ))
    else:
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

# === INICIALIZAÇÃO COM CACHE E PERFORMANCE LOGS ===
_perf_start("init_total")

if 'df_aliases' not in st.session_state:
    _perf_start("init_aliases")
    df_aliases = carregar_aliases()
    if df_aliases is not None:
        st.session_state['df_aliases'] = df_aliases
        st.session_state['dict_aliases'] = dict(zip(df_aliases['Instituição'], df_aliases['Alias Banco']))
        # Usar funções cacheadas com dados preparados
        alias_hash, alias_data = _preparar_aliases_para_cache(df_aliases)
        st.session_state['dict_aliases_norm'] = construir_dict_aliases_normalizado(alias_hash, alias_data)
        cores_hash, cores_data = _preparar_cores_para_cache(df_aliases)
        st.session_state['dict_cores_personalizadas'] = carregar_cores_aliases_local(cores_hash, cores_data)
        st.session_state['colunas_classificacao'] = [c for c in df_aliases.columns if c not in ['Instituição','Alias Banco','Cor','Código Cor']]
    print(_perf_log("init_aliases"))


def carregar_dados_periodos():
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        return
    _perf_start("init_dados_periodos")
    cache_manager = get_cache_manager()
    resultado_principal = cache_manager.carregar("principal")
    cache_principal = cache_manager.get_cache("principal")
    dados_cache = cache_principal.carregar_formato_antigo() if cache_principal else None
    if dados_cache:
        _perf_start("recalc_metricas")
        dados_cache = recalcular_metricas_derivadas(dados_cache)
        print(_perf_log("recalc_metricas"))
        if 'dict_aliases' in st.session_state:
            _perf_start("aplicar_aliases")
            dados_cache = aplicar_aliases_em_periodos(
                dados_cache,
                st.session_state['dict_aliases'],
                mapa_codigos=None,  # Evita chamada HTTP à API Olinda
            )
            print(_perf_log("aplicar_aliases"))
        st.session_state['dados_periodos'] = dados_cache
        if 'cache_fonte' not in st.session_state:
            st.session_state['cache_fonte'] = resultado_principal.fonte if resultado_principal else 'desconhecida'
        if 'dados_periodos_erro' in st.session_state:
            del st.session_state['dados_periodos_erro']
    else:
        if resultado_principal and not resultado_principal.sucesso:
            st.session_state['dados_periodos_erro'] = resultado_principal.mensagem
            st.session_state['dados_periodos_fonte'] = resultado_principal.fonte
    print(_perf_log("init_dados_periodos"))


def carregar_dados_capital():
    if 'dados_capital' in st.session_state and st.session_state['dados_capital']:
        return
    _perf_start("init_dados_capital")
    cache_manager = get_cache_manager()
    resultado_capital = cache_manager.carregar("capital")
    cache_capital = cache_manager.get_cache("capital")
    dados_capital = cache_capital.carregar_formato_antigo() if cache_capital else None
    if dados_capital:
        if 'dict_aliases' in st.session_state:
            dados_capital = aplicar_aliases_em_periodos(
                dados_capital,
                st.session_state['dict_aliases'],
                mapa_codigos=None,  # Evita chamada HTTP à API Olinda
            )
        st.session_state['dados_capital'] = dados_capital
        st.session_state['capital_cache_fonte'] = resultado_capital.fonte if resultado_capital else 'desconhecida'
    print(_perf_log("init_dados_capital"))


print(_perf_log("init_total"))

# Menu horizontal no topo
if 'menu_atual' not in st.session_state:
    st.session_state['menu_atual'] = "Sobre"

# Header e menu centralizados
st.markdown("""
<style>
    /* Remove padding extra do topo */
    .main .block-container {
        padding-top: 1rem !important;
    }

    .header-nav {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .header-nav [data-testid="stSegmentedControl"] > div {
        justify-content: center;
    }

    .header-logo {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 0.5rem;
    }

    .header-logo img {
        width: 200px;
        height: auto;
        image-rendering: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header usando colunas Streamlit para garantir centralização
_, col_header, _ = st.columns([1, 3, 1])
with col_header:
    # OTIMIZAÇÃO: Usar função cacheada para processamento do logo
    logo_base64 = _carregar_logo_base64(LOGO_PATH, target_width=200)
    if logo_base64:
        st.markdown(
            f"""
            <div class="header-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="toma.conta logo" />
            </div>
            """,
            unsafe_allow_html=True
        )

    # Título e subtítulos centralizados via HTML
    st.markdown("""
        <div style="text-align: center; margin-top: -0.5rem;">
            <p style="font-size: 3.6rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.2rem;">toma.conta</p>
            <p style="font-size: 1.6rem; color: #666; margin-bottom: 0.1rem;">análise de instituições financeiras brasileiras</p>
            <p style="font-size: 0.9rem; color: #888; font-style: italic; margin-bottom: 0.5rem;">por matheus prates, cfa</p>
        </div>
    """, unsafe_allow_html=True)

# Lista de opções do menu principal (análise)
MENU_PRINCIPAL = [
    "Rankings",
    "Peers (Tabela)",
    "Scatter Plot",
    "DRE",
    "Carteira 4.966",
    "Taxas de Juros por Produto",
    "Crie sua métrica!",
]

# Lista de opções do menu secundário (utilitários)
MENU_SECUNDARIO = ["Sobre", "Atualizar Base", "Glossário"]

TODOS_MENUS = MENU_PRINCIPAL + MENU_SECUNDARIO

# Validar menu_atual
if st.session_state['menu_atual'] not in TODOS_MENUS:
    if st.session_state['menu_atual'] == "Taxas de Juros":
        st.session_state['menu_atual'] = "Taxas de Juros por Produto"
    elif st.session_state['menu_atual'] == "Atualização Base":
        st.session_state['menu_atual'] = "Atualizar Base"
    elif st.session_state['menu_atual'] == "Painel":
        st.session_state['menu_atual'] = "Rankings"
    else:
        st.session_state['menu_atual'] = "Sobre"

menu_atual = st.session_state['menu_atual']

# Carregamento sob demanda para reduzir uso de RAM
menus_precisam_principal = {
    "Peers (Tabela)",
    "Scatter Plot",
    "Rankings",
    "Crie sua métrica!",
}
if menu_atual in menus_precisam_principal:
    carregar_dados_periodos()

# Callbacks para navegação entre menus (evita conflito)
def _on_main_menu_change():
    """Callback quando menu principal é clicado."""
    sel = st.session_state.get('nav_main')
    if sel is not None and sel in MENU_PRINCIPAL:
        st.session_state['menu_atual'] = sel
        # Limpar seleção do menu secundário
        if 'nav_sec' in st.session_state:
            st.session_state['nav_sec'] = None

def _on_sec_menu_change():
    """Callback quando menu secundário é clicado."""
    sel = st.session_state.get('nav_sec')
    if sel is not None and sel in MENU_SECUNDARIO:
        st.session_state['menu_atual'] = sel
        # Limpar seleção do menu principal
        if 'nav_main' in st.session_state:
            st.session_state['nav_main'] = None

# Configurar valores iniciais nos widgets (antes de renderizar)
if menu_atual in MENU_PRINCIPAL:
    st.session_state['nav_main'] = menu_atual
    st.session_state['nav_sec'] = None
else:
    st.session_state['nav_main'] = None
    st.session_state['nav_sec'] = menu_atual

# Menu principal (análise)
st.markdown('<div class="header-nav">', unsafe_allow_html=True)
st.segmented_control(
    "menu principal",
    MENU_PRINCIPAL,
    label_visibility="collapsed",
    key="nav_main",
    on_change=_on_main_menu_change
)
st.markdown('</div>', unsafe_allow_html=True)

# Menu secundário (utilitários)
st.markdown('<div class="header-nav">', unsafe_allow_html=True)
st.segmented_control(
    "menu secundário",
    MENU_SECUNDARIO,
    label_visibility="collapsed",
    key="nav_sec",
    on_change=_on_sec_menu_change
)
st.markdown('</div>', unsafe_allow_html=True)

# Usar menu_atual (já atualizado pelos callbacks)
menu = st.session_state['menu_atual']

st.markdown("---")

# Sidebar apenas para informações básicas
with st.sidebar:
    st.markdown('<p class="sidebar-title">toma.conta</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">análise de instituições financeiras brasileiras</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-author">por matheus prates, cfa</p>', unsafe_allow_html=True)

    st.markdown("")

    with st.expander("controle avançado"):
        if 'df_aliases' in st.session_state:
            st.success(f"{len(st.session_state['df_aliases'])} aliases carregados")
        else:
            st.error("aliases não encontrados")

        # Informações detalhadas do cache
        st.markdown("**status do cache**")
        cache_info = get_cache_info_detalhado()
        fonte = st.session_state.get('cache_fonte', 'desconhecida')

        if cache_info['existe']:
            st.caption(f"**caminho:** `{cache_info['caminho']}`")
            st.caption(f"**modificado:** {cache_info['data_formatada']}")
            st.caption(f"**tamanho:** {cache_info['tamanho_formatado']}")
            st.caption(f"**fonte:** {fonte}")

            # Mostrar info do cache_info.txt se existir
            info_cache = ler_info_cache()
            if info_cache:
                st.caption(f"{info_cache.replace(chr(10), ' | ')}")
        else:
            st.warning("cache não encontrado no disco")

        # Botão para forçar recarregamento do cache local
        if st.button("recarregar cache do disco", width='stretch'):
            if forcar_recarregar_cache():
                st.success("cache recarregado do disco com sucesso!")
                st.rerun()
            else:
                st.error("falha ao recarregar cache - arquivo não existe")

        st.markdown("---")
        st.markdown("**diagnóstico**")
        st.toggle(
            "modo diagnóstico",
            value=False,
            key="modo_diagnostico",
            help="exibe memória aproximada, tamanho do recorte do cache derivado e tempos de execução",
        )

        st.markdown("---")
        st.markdown("**atualizar dados (admin)**")

        senha_input = st.text_input("senha de administrador", type="password", key="senha_admin")

        if senha_input == SENHA_ADMIN:
            col1, col2 = st.columns(2)
            with col1:
                ano_i = st.selectbox("ano inicial", range(2015,2028), index=8, key="ano_i")
                mes_i = st.selectbox("trimestre inicial", ['03','06','09','12'], key="mes_i")
            with col2:
                ano_f = st.selectbox("ano final", range(2015,2028), index=10, key="ano_f")
                mes_f = st.selectbox("trimestre final", ['03','06','09','12'], index=2, key="mes_f")

            if 'dict_aliases' in st.session_state:
                if st.button("extrair dados do BCB", type="primary", width='stretch'):
                    periodos = gerar_periodos(ano_i, mes_i, ano_f, mes_f)
                    progress_bar = st.progress(0)
                    status = st.empty()
                    save_status = st.empty()

                    def update(i, total, p):
                        progress_bar.progress((i+1)/total)
                        status.text(f"extraindo {p[4:6]}/{p[:4]} ({i+1}/{total})")

                    # Callback para salvamento progressivo (a cada 5 períodos)
                    def save_progress(dados_parciais, info):
                        save_status.text(f"💾 salvando {len(dados_parciais)} períodos...")
                        salvar_cache(dados_parciais, info, incremental=True)
                        save_status.text(f"✓ {len(dados_parciais)} períodos salvos no cache")

                    st.info(f"🔄 iniciando extração de {len(periodos)} períodos. salvamento progressivo a cada 5 períodos.")

                    dados = processar_todos_periodos(
                        periodos,
                        st.session_state['dict_aliases'],
                        progress_callback=update,
                        save_callback=save_progress,
                        save_interval=5
                    )

                    if not dados:
                        progress_bar.empty()
                        status.empty()
                        save_status.empty()
                        st.error("falha ao extrair dados: nenhum período retornou dados válidos.")
                    else:
                        periodo_info = f"{periodos[0][4:6]}/{periodos[0][:4]} até {periodos[-1][4:6]}/{periodos[-1][:4]}"
                        cache_salvo = salvar_cache(dados, periodo_info, incremental=True)

                        # Atualizar session_state com merge dos dados existentes + novos
                        if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
                            # Merge: dados existentes + novos (novos sobrescrevem)
                            dados_merged = st.session_state['dados_periodos'].copy()
                            dados_merged.update(dados)
                            st.session_state['dados_periodos'] = dados_merged
                        else:
                            st.session_state['dados_periodos'] = dados

                        st.session_state['cache_fonte'] = 'extração local'

                        progress_bar.empty()
                        status.empty()
                        save_status.empty()
                        st.success(f"✓ {len(dados)} períodos extraídos! cache total: {len(st.session_state['dados_periodos'])} períodos")
                        st.info(f"cache salvo em: {cache_salvo['caminho']}")
                        st.info(f"tamanho: {cache_salvo['tamanho_formatado']}")
                        st.rerun()

                st.markdown("---")
                st.markdown("**status dos caches**")

                # Verificar status no GitHub
                with st.spinner("verificando github..."):
                    gh_status = verificar_caches_github()

                col_status1, col_status2 = st.columns(2)

                with col_status1:
                    st.caption("**Cache Principal (dados.parquet/pickle)**")
                    # Local
                    cache_info_local = get_cache_info()
                    if cache_info_local['existe']:
                        st.caption(f"📁 Local: ✅ {cache_info_local['tamanho_formatado']}")
                    else:
                        st.caption("📁 Local: ❌ não existe")
                    # GitHub
                    if gh_status['cache_principal']['existe']:
                        st.caption(f"☁️ GitHub: ✅ {gh_status['cache_principal']['tamanho_fmt']}")
                    else:
                        st.caption("☁️ GitHub: ❌ não existe")

                with col_status2:
                    st.caption("**Cache Capital (dados.parquet/pickle)**")
                    # Local
                    capital_info_local = get_capital_cache_info()
                    if capital_info_local['existe']:
                        st.caption(f"📁 Local: ✅ {capital_info_local['tamanho_formatado']} ({capital_info_local['n_periodos']} períodos)")
                    else:
                        st.caption("📁 Local: ❌ não existe")
                    # GitHub
                    if gh_status['cache_capital']['existe']:
                        st.caption(f"☁️ GitHub: ✅ {gh_status['cache_capital']['tamanho_fmt']}")
                    else:
                        st.caption("☁️ GitHub: ⚠️ **NÃO EXISTE** - envie abaixo!")

                if gh_status['erro']:
                    st.warning(f"⚠️ Erro ao verificar GitHub: {gh_status['erro']}")

                st.markdown("---")
                st.markdown("**publicar caches no github**")
                st.caption("envia os caches locais para github releases para persistência")

                # Tentar usar token do Streamlit Secrets primeiro
                token_from_secrets = st.secrets.get("GITHUB_TOKEN", None) if hasattr(st, 'secrets') else None
                if token_from_secrets:
                    st.caption("✓ usando GITHUB_TOKEN dos Secrets")

                gh_token = st.text_input("github token (opcional)", type="password", key="gh_token",
                                        help="token com permissão 'repo'. deixe em branco para usar Secrets ou gh CLI")

                col_upload1, col_upload2 = st.columns(2)

                with col_upload1:
                    if st.button("📦 enviar cache PRINCIPAL", width='stretch', help="Envia dados_cache"):
                        token_final = gh_token if gh_token else token_from_secrets
                        with st.spinner("enviando cache principal para github releases..."):
                            sucesso, mensagem = upload_cache_github(get_cache_manager(), "principal", token_final)
                            if sucesso:
                                st.success(f"✅ Cache PRINCIPAL: {mensagem}")
                            else:
                                st.error(f"❌ Cache PRINCIPAL: {mensagem}")

                with col_upload2:
                    # Botão para enviar cache de capital separadamente
                    capital_info = get_capital_cache_info()
                    btn_disabled = not capital_info['existe']
                    btn_help = "Envia cache de capital" if capital_info['existe'] else "Cache de capital não existe localmente"

                    if st.button("💰 enviar cache CAPITAL", width='stretch', disabled=btn_disabled, help=btn_help):
                        token_final = gh_token if gh_token else token_from_secrets
                        with st.spinner("enviando cache de capital para github releases..."):
                            sucesso, mensagem = upload_cache_github(get_cache_manager(), "capital", token_final)
                            if sucesso:
                                st.success(f"✅ Cache CAPITAL: {mensagem}")
                            else:
                                st.error(f"❌ Cache CAPITAL: {mensagem}")

                # =============================================================
                # SEÇÃO ISOLADA: EXTRAÇÃO DE DADOS DE CAPITAL
                # Cache separado (parquet/pickle), sem impacto no fluxo principal
                # =============================================================
                st.markdown("---")
                st.markdown("**extrair capital (relatório 5)**")
                st.caption("extrai informações de capital (índices, RWA, alavancagem) - cache separado")

                # Mostrar status do cache de capital
                capital_cache_info = get_capital_cache_info()
                if capital_cache_info['existe']:
                    st.caption(f"📊 cache capital: {capital_cache_info['n_periodos']} períodos | {capital_cache_info['tamanho_formatado']}")
                    st.caption(f"📅 atualizado: {capital_cache_info['data_formatada']}")

                    # Botão para baixar o cache localmente (backup)
                    cache_download = preparar_download_cache_local(get_cache_manager(), "capital")
                    if cache_download:
                        st.download_button(
                            label="📥 baixar cache de capital (backup local)",
                            data=cache_download["data"],
                            file_name=cache_download["file_name"],
                            mime=cache_download["mime"],
                            width='stretch',
                            help="Baixe uma cópia do cache antes que o Streamlit reinicie"
                        )
                else:
                    st.caption("📊 cache capital: não existe ainda")

                col_cap1, col_cap2 = st.columns(2)
                with col_cap1:
                    ano_cap_i = st.selectbox("ano inicial", range(2015, 2028), index=8, key="ano_cap_i")
                    mes_cap_i = st.selectbox("trim. inicial", ['03', '06', '09', '12'], key="mes_cap_i")
                with col_cap2:
                    ano_cap_f = st.selectbox("ano final", range(2015, 2028), index=10, key="ano_cap_f")
                    mes_cap_f = st.selectbox("trim. final", ['03', '06', '09', '12'], index=2, key="mes_cap_f")

                # Opção de atualização completa (sobrescreve cache antigo)
                atualizar_completo_capital = st.checkbox(
                    "🔄 atualização completa (sobrescrever cache antigo)",
                    value=False,
                    key="atualizar_completo_capital",
                    help="Marque para apagar o cache existente e extrair tudo do zero. Use quando houver problemas com nomes ou dados antigos."
                )

                if st.button("extrair dados de capital", type="secondary", width='stretch', key="btn_extrair_capital"):
                    periodos_cap = gerar_periodos_capital(ano_cap_i, mes_cap_i, ano_cap_f, mes_cap_f)
                    progress_bar_cap = st.progress(0)
                    status_cap = st.empty()
                    save_status_cap = st.empty()

                    # Se atualização completa, deletar cache existente primeiro
                    incremental_mode = not atualizar_completo_capital
                    if atualizar_completo_capital:
                        cache_manager = get_cache_manager()
                        resultado_limpar = cache_manager.limpar("capital")
                        if resultado_limpar.sucesso:
                            st.info("🗑️ cache antigo de capital removido para atualização completa")

                    def update_cap(i, total, p):
                        progress_bar_cap.progress((i + 1) / total)
                        status_cap.text(f"extraindo capital {p[4:6]}/{p[:4]} ({i + 1}/{total})")

                    def save_progress_cap(dados_parciais, info):
                        save_status_cap.text(f"💾 salvando {len(dados_parciais)} períodos de capital...")
                        salvar_cache_capital(dados_parciais, info, incremental=incremental_mode)
                        save_status_cap.text(f"✓ {len(dados_parciais)} períodos de capital salvos")

                    modo_texto = "completa" if atualizar_completo_capital else "incremental"
                    st.info(f"🔄 iniciando extração de capital ({modo_texto}): {len(periodos_cap)} períodos")

                    # Usar dict_aliases se disponível
                    aliases_para_capital = st.session_state.get('dict_aliases', {})

                    dados_capital = processar_todos_periodos_capital(
                        periodos_cap,
                        dict_aliases=aliases_para_capital,
                        progress_callback=update_cap,
                        save_callback=save_progress_cap,
                        save_interval=5
                    )

                    if not dados_capital:
                        progress_bar_cap.empty()
                        status_cap.empty()
                        save_status_cap.empty()
                        st.error("falha ao extrair dados de capital: nenhum período retornou dados válidos.")
                    else:
                        periodo_info_cap = f"capital {periodos_cap[0][4:6]}/{periodos_cap[0][:4]} até {periodos_cap[-1][4:6]}/{periodos_cap[-1][:4]}"
                        cache_capital_salvo = salvar_cache_capital(dados_capital, periodo_info_cap, incremental=incremental_mode)

                        progress_bar_cap.empty()
                        status_cap.empty()
                        save_status_cap.empty()

                        st.success(f"✓ {len(dados_capital)} períodos de capital extraídos!")
                        st.info(f"cache capital salvo em: {cache_capital_salvo['caminho']}")
                        st.info(f"tamanho: {cache_capital_salvo['tamanho_formatado']} | total: {cache_capital_salvo['n_periodos']} períodos")

                        # Upload para GitHub Releases para persistência
                        # Usa token do Streamlit Secrets se disponível
                        github_token = st.secrets.get("GITHUB_TOKEN", None) if hasattr(st, 'secrets') else None
                        with st.spinner("enviando cache de capital para github releases..."):
                            sucesso_upload, msg_upload = upload_cache_github(get_cache_manager(), "capital", github_token)
                            if sucesso_upload:
                                st.success(f"☁️ {msg_upload}")
                            else:
                                st.warning(f"⚠️ cache local salvo, mas falha no upload: {msg_upload}")

                        # Atualizar session_state com novos dados
                        st.session_state['dados_capital'] = dados_capital

                        # Mostrar campos extraídos
                        with st.expander("campos extraídos"):
                            campos = get_campos_capital_info()
                            for original, exibido in campos.items():
                                st.caption(f"• {exibido} ← _{original}_")

            else:
                st.warning("carregue os aliases primeiro")
        elif senha_input:
            st.error("senha incorreta")

if menu == "Sobre":
    st.markdown("""
    ## sobre a plataforma

    o **toma.conta** é uma plataforma completa de análise financeira do sistema bancário brasileiro. com dados oficiais do banco central, oferece visualizações interativas, comparações entre instituições, análise de capital regulatório e ferramentas para criação de métricas customizadas — tudo em uma interface intuitiva e com exportação profissional.
    """)

    st.markdown("### módulos de análise")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>ranking comparativo</h4>
            <p>rankings e gráficos de composição para comparar instituições em um período específico. inclui médias ponderadas do grupo e diferenças calculadas automaticamente.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>histórico individual</h4>
            <p>análise detalhada de uma instituição ao longo do tempo com mini-gráficos de tendência. exportação em pdf scorecard com visão de 4 anos.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>histórico peers</h4>
            <p>séries temporais comparativas entre múltiplas instituições. ideal para acompanhar a evolução de concorrentes ao longo dos trimestres.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>scatter plot</h4>
            <p>gráficos de dispersão com 3 dimensões (eixo x, eixo y e tamanho da bolha). visualize relações entre variáveis e movimentação entre períodos.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>deltas (antes e depois)</h4>
            <p>análise de variações absolutas e percentuais entre dois períodos. identifique rapidamente quem cresceu, encolheu ou mudou de patamar.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>capital regulatório</h4>
            <p>dados do relatório 5 do bcb: capital principal, complementar, nível ii, rwa por tipo de risco, razão de alavancagem e índices de capital.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>crie sua métrica</h4>
            <p>construa indicadores customizados combinando variáveis com operações matemáticas. visualize como ranking, scatter plot ou análise de deltas.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>glossário técnico</h4>
            <p>definições detalhadas de todas as variáveis, métricas calculadas e informações sobre consolidação prudencial (visão conglomerado).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### indicadores disponíveis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **variáveis de balanço**
        - ativo total
        - carteira de crédito (líquida)
        - títulos e valores mobiliários
        - passivo exigível
        - captações
        - patrimônio líquido
        - lucro líquido ac. ytd
        - patrimônio de referência
        """)

    with col2:
        st.markdown("""
        **capital regulatório**
        - capital principal (tier 1)
        - capital complementar
        - capital nível ii
        - rwa total / crédito / mercado / operacional
        - exposição total
        - índices de capital
        - razão de alavancagem
        """)

    with col3:
        st.markdown("""
        **métricas calculadas**
        - roe anualizado (%)
        - crédito / pl (%)
        - crédito / captações (%)
        - crédito / ativo (%)

        **índices**
        - índice de basileia
        - índice de imobilização
        - número de agências e pacs

        **taxas de juros**
        - taxa de juros por produto e modalidade (pf e pj)
        """)

    st.markdown("---")

    st.markdown("### recursos avançados")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>filtros inteligentes</h4>
            <p>selecione instituições por top n (por qualquer indicador) ou lista customizada. 8 opções de ponderação para médias do grupo.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>nomenclatura personalizada</h4>
            <p>renomeie instituições com nomes-fantasia e defina cores personalizadas para análises segmentadas.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>exportação profissional</h4>
            <p>exporte para excel (multi-abas) ou csv. gere pdf scorecards com histórico de 4 anos para apresentações e relatórios.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>dados oficiais</h4>
            <p>fonte única: api if.data e relatório 5 do banco central. dados consolidados (visão conglomerado) com atualização trimestral.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### como utilizar

    1. **dados pré-carregados**: a plataforma já possui dados históricos prontos para análise imediata
    2. **navegue pelos módulos**: use o menu superior para acessar rankings, histórico, scatter plot, deltas ou capital regulatório
    3. **aplique filtros**: selecione instituições por top n ou lista customizada
    4. **personalize análises**: ajuste variáveis, períodos e ponderações conforme sua necessidade
    5. **exporte resultados**: baixe em excel, csv ou gere pdf scorecards para compartilhar

    ---

    ### stack tecnológica

    | componente | função |
    |------------|--------|
    | **python 3.10+** | linguagem base |
    | **streamlit** | interface web interativa |
    | **pandas** | processamento e análise de dados |
    | **numpy** | computação numérica e vetorização |
    | **plotly** | visualizações dinâmicas e interativas |
    | **matplotlib** | gráficos auxiliares e exportações |
    | **reportlab** | geração de pdfs e scorecards |
    | **openpyxl / xlsxwriter** | exportação avançada para excel |
    | **pillow** | tratamento de imagens e assets |
    | **requests** | integrações http e consumo de apis |
    | **api bcb olinda** | fonte oficial de dados |
    """)

    st.markdown("---")
    st.caption("desenvolvido em 2026 por matheus prates, cfa | ferramenta open-source para análise do sistema financeiro brasileiro")

elif False and menu == "Painel":
    st.markdown("## resumo comparativo por período")
    st.caption("compare múltiplas instituições em um único trimestre, com ranking e média do grupo selecionado.")

    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = get_dados_concatenados()  # OTIMIZAÇÃO: usar cache

        indicadores_config = {
            'Ativo Total': ['Ativo Total'],
            'Carteira de Crédito': ['Carteira de Crédito'],
            'Títulos e Valores Mobiliários': ['Títulos e Valores Mobiliários'],
            'Passivo Exigível': ['Passivo Exigível'],
            'Captações': ['Captações'],
            'Patrimônio Líquido': ['Patrimônio Líquido'],
            'Lucro Líquido Acumulado YTD': ['Lucro Líquido Acumulado YTD'],
            'Índice de CET1': ['Índice de CET1'],
            'Patrimônio de Referência': [
                'Patrimônio de Referência para Comparação com o RWA (e)',
                'Patrimônio de Referência',
            ],
            'Índice de Basileia': ['Índice de Basileia'],
            'Índice de Imobilização': ['Índice de Imobilização'],
            'Número de Agências': ['Número de Agências'],
            'Número de Postos de Atendimento': ['Número de Postos de Atendimento'],
            # Variáveis de Capital (Relatório 5)
            'RWA Total': ['RWA Total'],
            'Capital Principal': ['Capital Principal'],
            'Índice de Capital Principal (CET1)': ['Índice de Capital Principal (CET1)', 'Índice de Capital Principal'],
            'Índice de Capital Nível I': ['Índice de Capital Nível I'],
            'Razão de Alavancagem': ['Razão de Alavancagem'],
        }

        indicadores_disponiveis = {}
        for label, colunas in indicadores_config.items():
            coluna_valida = next((col for col in colunas if col in df.columns), None)
            if coluna_valida:
                indicadores_disponiveis[label] = coluna_valida
        if 'Índice de CET1' not in indicadores_disponiveis:
            df_cet1_check = construir_cet1_capital(
                st.session_state.get("dados_capital", {}),
                st.session_state.get("dict_aliases", {}),
                st.session_state.get("df_aliases"),
                st.session_state.get("dados_periodos"),
            )
            if not df_cet1_check.empty:
                indicadores_disponiveis['Índice de CET1'] = 'Índice de CET1'
        if 'Índice de CET1' not in indicadores_disponiveis:
            df_cet1_check = construir_cet1_capital(
                st.session_state.get("dados_capital", {}),
                st.session_state.get("dict_aliases", {}),
                st.session_state.get("df_aliases"),
                st.session_state.get("dados_periodos"),
            )
            if not df_cet1_check.empty:
                indicadores_disponiveis['Índice de CET1'] = 'Índice de CET1'

        if not indicadores_disponiveis:
            st.warning("nenhum dos indicadores requeridos foi encontrado nos dados atuais.")
        else:
            periodos = ordenar_periodos(df['Período'].dropna().unique(), reverso=True)

            componentes_indicador = {
                'Patrimônio de Referência': [
                    'RWA Crédito',
                    'RWA Contraparte',
                    'RWA Operacional',
                    'RWA Mercado',
                    'RWA Outros',
                ]
            }

            col_periodo, col_indicador, col_media = st.columns([1.2, 2, 1.8])
            with col_periodo:
                periodo_resumo = st.selectbox(
                    "período",
                    periodos,
                    index=0,
                    key="periodo_resumo",
                    format_func=periodo_para_exibicao
                )
            with col_indicador:
                indicador_label = st.selectbox(
                    "indicador",
                    list(indicadores_disponiveis.keys()),
                    key="indicador_resumo"
                )
            componentes_disponiveis = [
                col for col in componentes_indicador.get(indicador_label, []) if col in df.columns
            ]
            tipo_grafico = "Composição (100%)" if componentes_disponiveis else None
            with col_media:
                tipo_media_label = st.selectbox(
                    "ponderar média por",
                    list(VARIAVEIS_PONDERACAO.keys()),
                    index=0,
                    key="tipo_media_resumo"
                )
                coluna_peso_resumo = VARIAVEIS_PONDERACAO[tipo_media_label]

            df_periodo = df[df['Período'] == periodo_resumo].copy()
            if indicador_label == "Índice de CET1":
                df_cet1_periodo = obter_cet1_periodo(
                    periodo_resumo,
                    st.session_state.get("dados_capital", {}),
                    st.session_state.get("dict_aliases", {}),
                    st.session_state.get("df_aliases"),
                    st.session_state.get("dados_periodos"),
                )
                if not df_cet1_periodo.empty:
                    df_periodo = df_periodo.merge(
                        df_cet1_periodo,
                        on="Instituição",
                        how="left",
                    )
            if indicador_label == "Índice de CET1":
                df_cet1_periodo = obter_cet1_periodo(
                    periodo_resumo,
                    st.session_state.get("dados_capital", {}),
                    st.session_state.get("dict_aliases", {}),
                    st.session_state.get("df_aliases"),
                    st.session_state.get("dados_periodos"),
                )
                if not df_cet1_periodo.empty:
                    df_periodo = df_periodo.merge(
                        df_cet1_periodo,
                        on="Instituição",
                        how="left",
                    )
            df_periodo_universo = df_periodo.copy()

            bancos_todos = df_periodo['Instituição'].dropna().unique().tolist()
            dict_aliases = st.session_state.get('dict_aliases', {})
            bancos_todos = ordenar_bancos_com_alias(bancos_todos, dict_aliases)

            indicador_col = indicadores_disponiveis[indicador_label]
            coluna_selecao = indicador_col
            if componentes_disponiveis:
                df_periodo['total_componentes'] = df_periodo[componentes_disponiveis].sum(axis=1, skipna=True)
                df_periodo_universo['total_componentes'] = df_periodo_universo[componentes_disponiveis].sum(axis=1, skipna=True)
                coluna_selecao = 'total_componentes'

            bancos_default = []
            if usar_top_universo:
                df_universo_valid = df_periodo_universo.dropna(subset=[coluna_selecao]).copy()
                if df_universo_valid.empty:
                    st.warning("não há dados disponíveis para calcular o top N do universo.")
                else:
                    df_universo_top = df_universo_valid.sort_values(coluna_selecao, ascending=False).head(top_universo_n)
                    bancos_default = df_universo_top['Instituição'].tolist()

            bancos_default = [banco for banco in bancos_default if banco in bancos_todos]

            with col_bancos:
                bancos_selecionados = st.multiselect(
                    "selecionar instituições (até 40)",
                    bancos_todos,
                    default=bancos_default,
                    key="bancos_resumo"
                )

            col_top, col_ordem, col_sort = st.columns([1.4, 1.4, 1.8])
            with col_top:
                usar_top_n = st.toggle("usar top/bottom n", value=True, key="usar_top_resumo")
                top_n_resumo = st.selectbox("n", [10, 15, 20], index=0, key="top_n_resumo")
            with col_ordem:
                direcao_top = st.radio(
                    "top/bottom",
                    ["Top", "Bottom"],
                    horizontal=True,
                    key="top_bottom_resumo"
                )
            with col_sort:
                modo_ordenacao = st.radio(
                    "ordenação",
                    ["Ordenar por valor", "Manter ordem de seleção"],
                    horizontal=True,
                    key="ordenacao_resumo"
                )

            format_info = get_axis_format(indicador_col)

            def formatar_numero(valor, fmt_info, incluir_sinal=False):
                if pd.isna(valor):
                    return "N/A"
                valor_formatado = format(valor, fmt_info['tickformat'])
                if incluir_sinal and valor > 0:
                    valor_formatado = f"+{valor_formatado}"
                return f"{valor_formatado}{fmt_info['ticksuffix']}"

            max_bancos = 40
            if bancos_selecionados and len(bancos_selecionados) > max_bancos:
                st.warning(f"limite de {max_bancos} instituições excedido; exibindo as primeiras {max_bancos}.")
                bancos_selecionados = bancos_selecionados[:max_bancos]

            if usar_top_n or not bancos_selecionados:
                df_periodo_valid = df_periodo.dropna(subset=[coluna_selecao]).copy()
                if df_periodo_valid.empty:
                    st.info("não há dados suficientes para o período e indicador selecionados.")
                else:
                    ascending = direcao_top == "Bottom"
                    df_selecionado = df_periodo_valid.sort_values(coluna_selecao, ascending=ascending).head(top_n_resumo)
            else:
                df_selecionado = df_periodo[df_periodo['Instituição'].isin(bancos_selecionados)].copy()

            df_selecionado = df_selecionado.dropna(subset=[coluna_selecao])

            if df_selecionado.empty:
                st.info("selecione instituições ou ajuste os filtros para visualizar o resumo.")
            else:
                df_selecionado['valor_display'] = df_selecionado[indicador_col] * format_info['multiplicador']
                media_display = calcular_media_ponderada(df_selecionado, 'valor_display', coluna_peso_resumo)
                label_media = get_label_media(coluna_peso_resumo)

                if modo_ordenacao == "Ordenar por valor":
                    ordenar_asc = direcao_top == "Bottom"
                    if tipo_grafico == "Composição (100%)" and componentes_disponiveis:
                        df_selecionado = df_selecionado.sort_values(coluna_selecao, ascending=ordenar_asc)
                    else:
                        df_selecionado = df_selecionado.sort_values('valor_display', ascending=ordenar_asc)
                elif bancos_selecionados:
                    ordem = bancos_selecionados
                    df_selecionado['ordem'] = pd.Categorical(df_selecionado['Instituição'], categories=ordem, ordered=True)
                    df_selecionado = df_selecionado.sort_values('ordem')

                if componentes_disponiveis:
                    df_componentes = df_selecionado[['Instituição'] + componentes_disponiveis].copy()
                    df_componentes['total'] = df_componentes[componentes_disponiveis].sum(axis=1, skipna=True)
                    df_componentes = df_componentes[df_componentes['total'] > 0]

                    if df_componentes.empty:
                        st.info("não há dados suficientes para exibir a composição selecionada.")
                    else:
                        df_percent = df_componentes.copy()
                        df_percent[componentes_disponiveis] = df_percent[componentes_disponiveis].div(
                            df_componentes['total'],
                            axis=0
                        ) * 100

                        fig_resumo = go.Figure()
                        for componente in componentes_disponiveis:
                            fig_resumo.add_trace(go.Bar(
                                x=df_percent['Instituição'],
                                y=df_percent[componente],
                                name=componente,
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    f"{componente}: %{{y:.1f}}%<extra></extra>"
                                )
                            ))

                        fig_resumo.update_layout(
                            title=f"{indicador_label} - {periodo_resumo} ({len(df_percent)} instituições)",
                            xaxis_title="instituições",
                            yaxis_title="participação (%)",
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='white',
                            height=max(650, len(df_percent) * 24),
                            barmode='stack',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            xaxis=dict(tickangle=-45),
                            yaxis=dict(tickformat='.1f', ticksuffix='%'),
                            font=dict(family='IBM Plex Sans')
                        )

                        st.plotly_chart(fig_resumo, width='stretch', config={'displayModeBar': False})

                        df_componentes['ranking'] = df_componentes['total'].rank(method='first', ascending=False).astype(int)
                        # Merge com df_selecionado para ter acesso às colunas de peso
                        colunas_peso_possiveis = [v for v in VARIAVEIS_PONDERACAO.values() if v is not None]
                        colunas_peso_disponiveis = ['Instituição'] + [c for c in colunas_peso_possiveis if c in df_selecionado.columns]
                        if len(colunas_peso_disponiveis) > 1:
                            df_componentes_merge = df_componentes.merge(
                                df_selecionado[colunas_peso_disponiveis].drop_duplicates(),
                                on='Instituição',
                                how='left'
                            )
                        else:
                            df_componentes_merge = df_componentes.copy()
                        media_grupo_raw = calcular_media_ponderada(df_componentes_merge, 'total', coluna_peso_resumo)
                        df_export_base = df_componentes.copy()
                        df_export_base['Período'] = periodo_resumo
                        df_export_base['Indicador'] = indicador_label
                        df_export_base['Valor'] = df_export_base['total']
                        df_export_base['Média do Grupo'] = media_grupo_raw
                        df_export_base['Tipo de Média'] = tipo_media_label
                        df_export_base['Diferença vs Média'] = df_export_base['Valor'] - media_grupo_raw
                        df_export_base = df_export_base[[
                            'Período',
                            'Instituição',
                            'Indicador',
                            'Valor',
                            'ranking',
                            'Média do Grupo',
                            'Tipo de Média',
                            'Diferença vs Média'
                        ]].rename(columns={'ranking': 'Ranking'})

                        df_export_comp = df_percent.melt(
                            id_vars=['Instituição'],
                            value_vars=componentes_disponiveis,
                            var_name='Componente',
                            value_name='Participação (%)'
                        )
                        df_export_comp['Período'] = periodo_resumo
                        df_export_comp['Indicador'] = indicador_label

                        buffer_excel = BytesIO()
                        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                            df_export_base.to_excel(writer, index=False, sheet_name='resumo')
                            df_export_comp.to_excel(writer, index=False, sheet_name='composicao')
                        buffer_excel.seek(0)

                        st.download_button(
                            label="Exportar Excel",
                            data=buffer_excel,
                            file_name=f"resumo_{periodo_resumo.replace('/', '-')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="exportar_resumo_excel"
                        )
                else:
                    st.info("composição disponível apenas para indicadores com componentes detalhados (ex.: Patrimônio de Referência).")
    else:
        erro_cache = st.session_state.get('dados_periodos_erro')
        if erro_cache:
            st.error("dados principais indisponíveis no cache.")
            st.caption(f"detalhe: {erro_cache}")
            st.caption("abra 'Atualizar Base' e publique o cache no GitHub Releases.")
        else:
            st.info("carregando dados automaticamente do github...")
            st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "Peers (Tabela)":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = get_dados_concatenados()

        if len(df) > 0 and 'Instituição' in df.columns:
            bancos_todos = df['Instituição'].dropna().unique().tolist()
            dict_aliases = st.session_state.get('dict_aliases', {})
            bancos_disponiveis = ordenar_bancos_com_alias(bancos_todos, dict_aliases)
            periodos_disponiveis = ordenar_periodos(df['Período'].dropna().unique())
            periodos_dropdown = ordenar_periodos(df['Período'].dropna().unique(), reverso=True)

            if bancos_disponiveis and periodos_disponiveis:
                st.markdown("### Peers (Tabela)")
                st.caption("comparativo multi-bancos com períodos sincronizados.")

                _default_peers_bancos = _encontrar_bancos_default(
                    bancos_disponiveis, [("itau", "itaú")]
                )
                if not _default_peers_bancos:
                    _default_peers_bancos = bancos_disponiveis[:1]

                _default_peers_periodos = []
                for _tri, _ano in [(3, 2025), (2, 2025), (1, 2025)]:
                    _p = _encontrar_periodo(periodos_dropdown, _tri, _ano)
                    if _p:
                        _default_peers_periodos.append(_p)
                if not _default_peers_periodos:
                    _default_peers_periodos = periodos_dropdown[:3]

                col_bancos, col_periodos = st.columns([2, 2])
                with col_bancos:
                    bancos_selecionados = st.multiselect(
                        "selecionar instituições (até 5)",
                        bancos_disponiveis,
                        default=_default_peers_bancos,
                        max_selections=5,
                        key="peers_tabela_bancos",
                    )
                with col_periodos:
                    periodos_selecionados = st.multiselect(
                        "selecionar períodos (até 3)",
                        periodos_dropdown,
                        default=_default_peers_periodos,
                        max_selections=3,
                        key="peers_tabela_periodos",
                        format_func=periodo_para_exibicao,
                    )

                if bancos_selecionados and periodos_selecionados:
                    periodos_selecionados = ordenar_periodos(periodos_selecionados, reverso=True)
                    cache_ativo = _carregar_cache_relatorio("ativo")
                    cache_passivo = _carregar_cache_relatorio("passivo")
                    cache_carteira_pf = _carregar_cache_relatorio("carteira_pf")
                    cache_carteira_pj = _carregar_cache_relatorio("carteira_pj")
                    cache_carteira_instr = _carregar_cache_relatorio("carteira_instrumentos")
                    cache_dre = _carregar_cache_relatorio("dre")
                    cache_capital = _carregar_cache_relatorio("capital")

                    cache_ativo = _aplicar_aliases_df(cache_ativo, dict_aliases)
                    cache_passivo = _aplicar_aliases_df(cache_passivo, dict_aliases)
                    cache_carteira_pf = _aplicar_aliases_df(cache_carteira_pf, dict_aliases)
                    cache_carteira_pj = _aplicar_aliases_df(cache_carteira_pj, dict_aliases)
                    cache_carteira_instr = _aplicar_aliases_df(cache_carteira_instr, dict_aliases)
                    cache_dre = _aplicar_aliases_df(cache_dre, dict_aliases)
                    cache_capital = _aplicar_aliases_df(cache_capital, dict_aliases)

                    valores, colunas_usadas, faltas, delta_flags, tooltips = _montar_tabela_peers(
                        df,
                        bancos_selecionados,
                        periodos_selecionados,
                        caches_extras={
                            "ativo": cache_ativo,
                            "passivo": cache_passivo,
                            "carteira_pf": cache_carteira_pf,
                            "carteira_pj": cache_carteira_pj,
                            "carteira_instrumentos": cache_carteira_instr,
                            "dre": cache_dre,
                            "capital": cache_capital,
                        },
                    )

                    if faltas:
                        st.info(
                            "⚠️ Algumas métricas ainda aguardam integração (ver TODOs no código): "
                            + ", ".join(sorted(faltas))
                        )

                    html_tabela = _render_peers_table_html(
                        bancos_selecionados,
                        periodos_selecionados,
                        valores,
                        colunas_usadas,
                        delta_flags,
                        tooltips,
                    )
                    st.markdown(html_tabela, unsafe_allow_html=True)

                    with st.expander("📥 exportar"):
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            st.caption("Tabela formatada (layout visual)")
                            excel_buffer = _gerar_excel_peers_tabela(
                                bancos_selecionados,
                                periodos_selecionados,
                                valores,
                                colunas_usadas,
                                delta_flags,
                            )
                            st.download_button(
                                label="baixar Excel",
                                data=excel_buffer,
                                file_name="peers_tabela.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="peers_tabela_excel",
                            )
                        with col_exp2:
                            st.caption("Dados puros (sem formatação)")
                            excel_raw = _gerar_excel_peers_dados_puros(
                                bancos_selecionados,
                                periodos_selecionados,
                                valores,
                            )
                            st.download_button(
                                label="baixar Dados Puros",
                                data=excel_raw,
                                file_name="peers_dados_puros.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="peers_dados_puros_excel",
                            )

                    st.markdown(
                        """
                        <div style="font-size: 12px; color: #666; margin-top: 12px;">
                            <strong>mini-glossário:</strong><br>
                            <br>
                            <em>Balanço</em><br>
                            <strong>Ativo Total</strong> = Ativo Total do balanço principal (Rel. 1).<br>
                            <strong>Ativos Líquidos</strong> = Disponibilidades (a) + Aplicações Interfinanceiras de Liquidez (b) + Títulos e Valores Mobiliários (c) no relatório de Ativo (Rel. 2).<br>
                            <strong>Carteira de Crédito Bruta</strong> = Total da Carteira de Pessoa Física (Rel. 11) + Total da Carteira de Pessoa Jurídica (Rel. 13).<br>
                            <strong>Depósitos Totais</strong> = Depósitos (a) no relatório de Passivo (Rel. 3).<br>
                            <strong>Patrimônio Líquido (PL)</strong> = Patrimônio Líquido do balanço principal (Rel. 1).<br>
                            <br>
                            <em>Qualidade Carteira</em><br>
                            <strong>Perda Esperada</strong> = Soma das linhas Perda Esperada (e2), Hedge de Valor Justo (e3), Ajuste a Valor Justo (e4), Perda Esperada (f2), Hedge de Valor Justo (f3), Perda Esperada (g2), Hedge de Valor Justo (g3), Ajuste a Valor Justo (g4) e Perda Esperada (h2) no relatório de Ativo (Rel. 2).<br>
                            <strong>Perda Esperada / Carteira Bruta</strong> = Perda Esperada ÷ Carteira de Crédito Bruta.<br>
                            <strong>Carteira de Créd. Class. C4+C5</strong> = Soma das linhas C4 e C5 do relatório de Carteira 4.966 (Rel. 16).<br>
                            <strong>Carteira de Créd. Class. C4+C5 / Carteira Bruta</strong> = (C4 + C5) ÷ Carteira de Crédito Bruta.<br>
                            <strong>Perda Esperada / (Carteira C4 + C5)</strong> = Perda Esperada ÷ (C4 + C5).<br>
                            <strong>Desp PDD Anualizada / Carteira Bruta</strong> = (Resultado com Perda Esperada anualizado) ÷ Carteira de Crédito Bruta. Anualização: valor acumulado YTD × (12 / meses do período).<br>
                            <strong>Desp PDD / NII (ref: período acumulado)</strong> = Desp. PDD acumulada ÷ NII (resultado de intermediação financeira) acumulado.<br>
                            <br>
                            <em>Alavancagem</em><br>
                            <strong>Ativo / PL</strong> = Ativo Total ÷ Patrimônio Líquido.<br>
                            <strong>Crédito / PL</strong> = Carteira de Crédito (Rel. 1) ÷ Patrimônio Líquido.<br>
                            <strong>Índice de Capital Principal (CET1)</strong> = Capital Principal ÷ RWA Total, extraído do relatório de Informações de Capital (Rel. 5).<br>
                            <strong>Índice de Basileia Total</strong> = (Capital Principal + Capital Complementar + Capital Nível II) ÷ RWA Total (Rel. 5). Equivale à soma CET1 + AT1 + T2.<br>
                            <br>
                            <em>Desempenho</em><br>
                            <strong>Lucro Líquido Acumulado</strong> = Lucro Líquido acumulado no ano (YTD) até o fim do período (Rel. 1).<br>
                            <strong>ROE AC. Anualizado (%)</strong> = (LL YTD × fator de anualização) ÷ PL Médio, onde PL Médio = (PL no período + PL em Dez do ano anterior) / 2. O LL YTD de Set é obtido somando Jun (Jan-Jun) ao Set (Jul-Sep). Fator: Mar=4, Jun=2, Set=12/9, Dez=1. Se PL médio ≤ 0 ou dado faltante: N/A.<br>
                            <br>
                            <strong>Δ (▲/▼)</strong> = Variação vs. mesmo período do ano anterior.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("selecione instituições e períodos para visualizar a tabela.")
            else:
                st.warning("nenhuma instituição ou período disponível nos dados.")
        else:
            st.warning("dados incompletos ou vazios.")
    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "Scatter Plot":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = get_dados_concatenados()  # OTIMIZAÇÃO: usar cache

        colunas_base = [
            col for col in df.columns
            if col not in ['Instituição', 'Período'] and pd.api.types.is_numeric_dtype(df[col])
        ]
        colunas_numericas = colunas_base + [m for m in DERIVED_METRICS if m not in colunas_base]
        periodos = ordenar_periodos(df['Período'].unique(), reverso=True)

        # Lista de todos os bancos disponíveis com ordenação por alias
        bancos_todos = df['Instituição'].dropna().unique().tolist()
        dict_aliases = st.session_state.get('dict_aliases', {})
        todos_bancos = ordenar_bancos_com_alias(bancos_todos, dict_aliases)

        # Primeira linha: variáveis dos eixos e tamanho
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            var_x = st.selectbox("eixo x", colunas_numericas, index=colunas_numericas.index('Índice de Basileia') if 'Índice de Basileia' in colunas_numericas else 0)
        with col2:
            var_y = st.selectbox("eixo y", colunas_numericas, index=colunas_numericas.index('ROE Ac. YTD an. (%)') if 'ROE Ac. YTD an. (%)' in colunas_numericas else 1)
        with col3:
            opcoes_tamanho = ['Tamanho Fixo'] + colunas_numericas
            var_size = st.selectbox("tamanho", opcoes_tamanho, index=0)
        with col4:
            periodo_scatter = st.selectbox("período", periodos, index=0, format_func=periodo_para_exibicao)

        # Segunda linha: Top N e variável de ordenação
        col_t1, col_t2, col_t3 = st.columns([1, 1, 2])

        with col_t1:
            top_n_scatter = st.slider("top n", 5, 50, 5)
        with col_t2:
            var_top_n = st.selectbox("top n por", colunas_numericas, index=colunas_numericas.index('Carteira de Crédito') if 'Carteira de Crédito' in colunas_numericas else 0)

        # Terceira linha: Seleção de bancos
        col_f = st.columns(1)[0]
        with col_f:
            bancos_selecionados = st.multiselect(
                "selecionar bancos",
                todos_bancos,
                key="bancos_multiselect"
            )

        # Aplica filtros ao dataframe
        df_periodo = df[df['Período'] == periodo_scatter]
        df_periodo, diag_scatter_derived = anexar_metricas_derivadas_periodo(df_periodo, periodo_scatter)

        if bancos_selecionados:
            # Usa os bancos selecionados no multiselect
            df_scatter = df_periodo[df_periodo['Instituição'].isin(bancos_selecionados)]
        else:
            # Usa top N pela variável selecionada (remove NaN antes)
            df_periodo_valid = df_periodo.dropna(subset=[var_top_n])
            df_scatter = df_periodo_valid.nlargest(top_n_scatter, var_top_n)

        format_x = get_axis_format(var_x, df_scatter[var_x] if var_x in df_scatter.columns else None)
        format_y = get_axis_format(var_y, df_scatter[var_y] if var_y in df_scatter.columns else None)

        df_scatter_plot = df_scatter.copy()
        df_scatter_plot['x_display'] = _calcular_valores_display(df_scatter_plot[var_x], var_x, format_x)
        df_scatter_plot['y_display'] = _calcular_valores_display(df_scatter_plot[var_y], var_y, format_y)

        if var_size == 'Tamanho Fixo':
            tamanho_constante = 25
        else:
            format_size = get_axis_format(var_size, df_scatter[var_size] if var_size in df_scatter.columns else None)
            df_scatter_plot['size_display'] = _calcular_valores_display(df_scatter_plot[var_size], var_size, format_size)

        fig_scatter = go.Figure()
        cores_plotly = px.colors.qualitative.Plotly
        idx_cor = 0

        for instituicao in df_scatter_plot['Instituição'].unique():
            df_inst = df_scatter_plot[df_scatter_plot['Instituição'] == instituicao]
            cor = obter_cor_banco(instituicao)
            if not cor:
                cor = cores_plotly[idx_cor % len(cores_plotly)]
                idx_cor += 1

            if var_size == 'Tamanho Fixo':
                marker_size = tamanho_constante
            else:
                marker_size = df_inst['size_display'] / df_scatter_plot['size_display'].max() * 100

            fig_scatter.add_trace(go.Scatter(
                x=df_inst['x_display'],
                y=df_inst['y_display'],
                mode='markers',
                name=instituicao,
                marker=dict(size=marker_size, color=cor, opacity=1.0, line=dict(width=1, color='white')),
                hovertemplate=f'<b>{instituicao}</b><br>{var_x}: %{{x:{format_x["tickformat"]}}}{format_x["ticksuffix"]}<br>{var_y}: %{{y:{format_y["tickformat"]}}}{format_y["ticksuffix"]}<extra></extra>'
            ))

        # Título dinâmico - Scatter Plot t=1
        st.markdown("#### Scatter Plot t=1")
        if bancos_selecionados:
            titulo_scatter = f'{var_y} vs {var_x} - {periodo_scatter} ({len(df_scatter)} bancos)'
        else:
            titulo_scatter = f'{var_y} vs {var_x} - {periodo_scatter} (top {top_n_scatter} por {var_top_n})'

        fig_scatter.update_layout(
            title=titulo_scatter,
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

        st.plotly_chart(fig_scatter, width='stretch')

        if st.session_state.get("modo_diagnostico"):
            with st.expander("diagnóstico scatter n=1"):
                st.caption(f"Memória df_periodo: {_df_mem_mb(df_periodo):.2f} MB")
                st.caption(f"Memória df_scatter: {_df_mem_mb(df_scatter):.2f} MB")
                st.caption(f"Recorte derivado: {diag_scatter_derived.get('linhas', 0)} linhas")
                st.caption(f"Memória recorte derivado: {diag_scatter_derived.get('mem_mb', 0):.2f} MB")
                st.caption(f"Tempo recorte derivado: {diag_scatter_derived.get('tempo_s', 0):.3f}s")

        # ============================================================
        # SCATTER PLOT t=2 - Comparação entre dois períodos
        # ============================================================
        st.markdown("---")
        st.markdown("#### Scatter Plot t=2")
        st.caption("Visualize a movimentação dos bancos entre dois períodos")

        # Seletores para os dois períodos
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            var_x_n2 = st.selectbox(
                "eixo x",
                colunas_numericas,
                index=colunas_numericas.index('Índice de Basileia') if 'Índice de Basileia' in colunas_numericas else 0,
                key="var_x_n2"
            )
        with col_p2:
            var_y_n2 = st.selectbox(
                "eixo y",
                colunas_numericas,
                index=colunas_numericas.index('ROE Ac. YTD an. (%)') if 'ROE Ac. YTD an. (%)' in colunas_numericas else 1,
                key="var_y_n2"
            )
        with col_p3:
            # Período inicial (Dez/2024 por padrão)
            _p_dez24 = _encontrar_periodo(periodos, 4, 2024)
            _idx_ini_n2 = periodos.index(_p_dez24) if _p_dez24 and _p_dez24 in periodos else (min(1, len(periodos) - 1) if len(periodos) > 1 else 0)
            periodo_inicial = st.selectbox(
                "período inicial",
                periodos,
                index=_idx_ini_n2,
                key="periodo_inicial_n2",
                format_func=periodo_para_exibicao
            )
        with col_p4:
            # Período subsequente (Set/2025 por padrão)
            _p_set25_n2 = _encontrar_periodo(periodos, 3, 2025)
            _idx_sub_n2 = periodos.index(_p_set25_n2) if _p_set25_n2 and _p_set25_n2 in periodos else 0
            periodo_subseq = st.selectbox(
                "período subsequente",
                periodos,
                index=_idx_sub_n2,
                key="periodo_subseq_n2",
                format_func=periodo_para_exibicao
            )

        # Segunda linha: Top N e tamanho
        col_n2_t1, col_n2_t2, col_n2_t3 = st.columns([1, 1, 2])

        with col_n2_t1:
            top_n_scatter_n2 = st.slider("top n", 5, 50, 5, key="top_n_n2")
        with col_n2_t2:
            var_top_n_n2 = st.selectbox(
                "top n por",
                colunas_numericas,
                index=colunas_numericas.index('Carteira de Crédito') if 'Carteira de Crédito' in colunas_numericas else 0,
                key="var_top_n_n2"
            )
        with col_n2_t3:
            opcoes_tamanho_n2 = ['Tamanho Fixo'] + colunas_numericas
            var_size_n2 = st.selectbox("tamanho", opcoes_tamanho_n2, index=0, key="var_size_n2")

        # Terceira linha: Seleção de bancos
        col_n2_f = st.columns(1)[0]
        with col_n2_f:
            bancos_selecionados_n2 = st.multiselect(
                "selecionar bancos",
                todos_bancos,
                key="bancos_multiselect_n2"
            )

        # Validação: períodos devem ser diferentes
        if periodo_inicial == periodo_subseq:
            st.warning("Selecione dois períodos diferentes para visualizar a movimentação.")
        else:
            # Filtra dados para os dois períodos
            df_p1 = df[df['Período'] == periodo_inicial].copy()
            df_p2 = df[df['Período'] == periodo_subseq].copy()

            # Aplica seleção de bancos ou top N
            if bancos_selecionados_n2:
                df_p1 = df_p1[df_p1['Instituição'].isin(bancos_selecionados_n2)]
                df_p2 = df_p2[df_p2['Instituição'].isin(bancos_selecionados_n2)]
            else:
                # Usa top N do período subsequente (mais recente)
                df_p2_valid = df_p2.dropna(subset=[var_top_n_n2])
                top_bancos = df_p2_valid.nlargest(top_n_scatter_n2, var_top_n_n2)['Instituição'].tolist()
                df_p1 = df_p1[df_p1['Instituição'].isin(top_bancos)]
                df_p2 = df_p2[df_p2['Instituição'].isin(top_bancos)]

            # Encontra bancos presentes em ambos os períodos
            bancos_comuns = set(df_p1['Instituição'].unique()) & set(df_p2['Instituição'].unique())

            if len(bancos_comuns) == 0:
                st.warning("Nenhum banco encontrado em ambos os períodos selecionados.")
            else:
                # Formatos dos eixos
                format_x_n2 = get_axis_format(var_x_n2, df_p1[var_x_n2] if var_x_n2 in df_p1.columns else None)
                format_y_n2 = get_axis_format(var_y_n2, df_p1[var_y_n2] if var_y_n2 in df_p1.columns else None)

                # Prepara dados com valores de exibição
                df_p1['x_display'] = _calcular_valores_display(df_p1[var_x_n2], var_x_n2, format_x_n2)
                df_p1['y_display'] = _calcular_valores_display(df_p1[var_y_n2], var_y_n2, format_y_n2)
                df_p2['x_display'] = _calcular_valores_display(df_p2[var_x_n2], var_x_n2, format_x_n2)
                df_p2['y_display'] = _calcular_valores_display(df_p2[var_y_n2], var_y_n2, format_y_n2)

                # Tamanho dos pontos
                if var_size_n2 != 'Tamanho Fixo':
                    format_size_n2 = get_axis_format(var_size_n2, df_p1[var_size_n2] if var_size_n2 in df_p1.columns else None)
                    df_p1['size_display'] = _calcular_valores_display(df_p1[var_size_n2], var_size_n2, format_size_n2)
                    df_p2['size_display'] = _calcular_valores_display(df_p2[var_size_n2], var_size_n2, format_size_n2)
                    max_size = max(df_p1['size_display'].max(), df_p2['size_display'].max())

                fig_scatter_n2 = go.Figure()
                cores_plotly = px.colors.qualitative.Plotly
                idx_cor_n2 = 0

                for instituicao in sorted(bancos_comuns):
                    # Dados do período inicial
                    row_p1 = df_p1[df_p1['Instituição'] == instituicao]
                    # Dados do período subsequente
                    row_p2 = df_p2[df_p2['Instituição'] == instituicao]

                    if row_p1.empty or row_p2.empty:
                        continue

                    x1 = row_p1['x_display'].values[0]
                    y1 = row_p1['y_display'].values[0]
                    x2 = row_p2['x_display'].values[0]
                    y2 = row_p2['y_display'].values[0]

                    # Cor do banco
                    cor = obter_cor_banco(instituicao)
                    if not cor:
                        cor = cores_plotly[idx_cor_n2 % len(cores_plotly)]
                        idx_cor_n2 += 1

                    # Tamanho dos marcadores
                    if var_size_n2 == 'Tamanho Fixo':
                        marker_size_p1 = 20
                        marker_size_p2 = 20
                    else:
                        marker_size_p1 = row_p1['size_display'].values[0] / max_size * 80 if max_size > 0 else 20
                        marker_size_p2 = row_p2['size_display'].values[0] / max_size * 80 if max_size > 0 else 20

                    # Adiciona linha conectando os dois pontos (seta)
                    fig_scatter_n2.add_trace(go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(color=cor, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Ponto do período inicial (círculo vazio/anel)
                    fig_scatter_n2.add_trace(go.Scatter(
                        x=[x1],
                        y=[y1],
                        mode='markers',
                        name=f'{instituicao} ({periodo_inicial})',
                        marker=dict(
                            size=marker_size_p1,
                            color='white',
                            opacity=1.0,
                            line=dict(width=3, color=cor)
                        ),
                        hovertemplate=f'<b>{instituicao}</b> ({periodo_inicial})<br>{var_x_n2}: %{{x:{format_x_n2["tickformat"]}}}{format_x_n2["ticksuffix"]}<br>{var_y_n2}: %{{y:{format_y_n2["tickformat"]}}}{format_y_n2["ticksuffix"]}<extra></extra>',
                        legendgroup=instituicao,
                        showlegend=False
                    ))

                    # Ponto do período subsequente (círculo cheio com seta)
                    fig_scatter_n2.add_trace(go.Scatter(
                        x=[x2],
                        y=[y2],
                        mode='markers',
                        name=instituicao,
                        marker=dict(
                            size=marker_size_p2,
                            color=cor,
                            opacity=1.0,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        hovertemplate=f'<b>{instituicao}</b> ({periodo_subseq})<br>{var_x_n2}: %{{x:{format_x_n2["tickformat"]}}}{format_x_n2["ticksuffix"]}<br>{var_y_n2}: %{{y:{format_y_n2["tickformat"]}}}{format_y_n2["ticksuffix"]}<extra></extra>',
                        legendgroup=instituicao,
                        showlegend=True
                    ))

                    # Adiciona seta (annotation) para indicar direção
                    fig_scatter_n2.add_annotation(
                        x=x2,
                        y=y2,
                        ax=x1,
                        ay=y1,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=cor,
                        opacity=0.7
                    )

                # Título do gráfico n=2
                if bancos_selecionados_n2:
                    titulo_scatter_n2 = f'{var_y_n2} vs {var_x_n2} - {periodo_inicial} → {periodo_subseq} ({len(bancos_comuns)} bancos)'
                else:
                    titulo_scatter_n2 = f'{var_y_n2} vs {var_x_n2} - {periodo_inicial} → {periodo_subseq} (top {top_n_scatter_n2} por {var_top_n_n2})'

                fig_scatter_n2.update_layout(
                    title=titulo_scatter_n2,
                    xaxis_title=var_x_n2,
                    yaxis_title=var_y_n2,
                    height=650,
                    plot_bgcolor='#f8f9fa',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                    xaxis=dict(tickformat=format_x_n2['tickformat'], ticksuffix=format_x_n2['ticksuffix']),
                    yaxis=dict(tickformat=format_y_n2['tickformat'], ticksuffix=format_y_n2['ticksuffix']),
                    font=dict(family='IBM Plex Sans')
                )

                st.plotly_chart(fig_scatter_n2, width='stretch')

                # Legenda explicativa
                st.caption("○ Círculo vazio = período inicial | ● Círculo cheio = período subsequente | → Seta indica direção da movimentação")

                if st.session_state.get("modo_diagnostico"):
                    with st.expander("diagnóstico scatter n=2"):
                        st.caption(f"Memória df_p1: {_df_mem_mb(df_p1):.2f} MB")
                        st.caption(f"Memória df_p2: {_df_mem_mb(df_p2):.2f} MB")
                        st.caption(f"Recorte derivado p1: {diag_scatter_derived_p1.get('linhas', 0)} linhas")
                        st.caption(f"Recorte derivado p2: {diag_scatter_derived_p2.get('linhas', 0)} linhas")
                        st.caption(f"Memória recorte derivado p1: {diag_scatter_derived_p1.get('mem_mb', 0):.2f} MB")
                        st.caption(f"Memória recorte derivado p2: {diag_scatter_derived_p2.get('mem_mb', 0):.2f} MB")
                        st.caption(f"Tempo recorte derivado p1: {diag_scatter_derived_p1.get('tempo_s', 0):.3f}s")
                        st.caption(f"Tempo recorte derivado p2: {diag_scatter_derived_p2.get('tempo_s', 0):.3f}s")

    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "Rankings":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        if not st.session_state.get('_dados_capital_mesclados'):
            carregar_dados_capital()
            if 'dados_capital' in st.session_state and st.session_state['dados_capital']:
                st.session_state['dados_periodos'] = mesclar_dados_capital(
                    st.session_state['dados_periodos'],
                    st.session_state['dados_capital']
                )
                st.session_state['_dados_capital_mesclados'] = True

        df = get_dados_concatenados()  # OTIMIZAÇÃO: usar cache
        df = adicionar_indice_cet1(df)
        df_cet1 = construir_cet1_capital(
            st.session_state.get("dados_capital", {}),
            st.session_state.get("dict_aliases", {}),
            st.session_state.get("df_aliases"),
            st.session_state.get("dados_periodos"),
        )
        if not df_cet1.empty:
            df = df.merge(
                df_cet1,
                on=["Período", "Instituição"],
                how="left",
                suffixes=("", "_cet1"),
            )
            if "Índice de CET1" not in df.columns and "Índice de CET1_cet1" in df.columns:
                df = df.rename(columns={"Índice de CET1_cet1": "Índice de CET1"})
            elif "Índice de CET1_cet1" in df.columns:
                df["Índice de CET1"] = df["Índice de CET1"].fillna(df["Índice de CET1_cet1"])
                df = df.drop(columns=["Índice de CET1_cet1"])

        periodos_disponiveis = ordenar_periodos(df['Período'].dropna().unique())
        periodos_dropdown = ordenar_periodos(df['Período'].dropna().unique(), reverso=True)

        st.markdown("### ranking")
        opcoes_grafico = ["Ranking (barras)", "Deltas (barras)", "Deltas (pontos)"]
        if st.session_state.get("grafico_rankings_toggle_v2") not in opcoes_grafico:
            st.session_state["grafico_rankings_toggle_v2"] = opcoes_grafico[0]
        grafico_escolhido = st.radio(
            "gráfico",
            opcoes_grafico,
            key="grafico_rankings_toggle_v2",
            index=0,
            horizontal=True
        )
        grafico_base = "Ranking" if grafico_escolhido.startswith("Ranking") else "Deltas (antes e depois)"
        modo_visualizacao_deltas = None
        if grafico_base == "Deltas (antes e depois)":
            modo_visualizacao_deltas = (
                "Barras (delta)"
                if grafico_escolhido == "Deltas (barras)"
                else "Pontos (antes/depois)"
            )

        indicadores_config = {
            'Ativo Total': ['Ativo Total'],
            'Carteira de Crédito': ['Carteira de Crédito'],
            'Captações': ['Captações'],
            'Patrimônio Líquido': ['Patrimônio Líquido'],
            'Índice de Capital Principal (CET1)': ['Índice de Capital Principal (CET1)', 'Índice de Capital Principal'],
            'Índice de Basileia': ['Índice de Basileia'],
            'Lucro Líquido Acumulado YTD': ['Lucro Líquido Acumulado YTD'],
            'ROE Ac. Anualizado (%)': ['ROE Ac. YTD an. (%)'],
        }

        indicadores_disponiveis = {}
        for label, colunas in indicadores_config.items():
            coluna_valida = next((col for col in colunas if col in df.columns), None)
            if coluna_valida:
                indicadores_disponiveis[label] = coluna_valida

        if not indicadores_disponiveis:
            st.warning("nenhum dos indicadores requeridos foi encontrado nos dados atuais.")
        else:
            periodos = ordenar_periodos(df['Período'].dropna().unique(), reverso=True)
            ordem_prioritaria = [
                'Ativo Total',
                'Carteira de Crédito',
                'Captações',
                'Patrimônio Líquido',
                'Índice de Capital Principal (CET1)',
                'Índice de Basileia',
                'Lucro Líquido Acumulado YTD',
                'ROE Ac. Anualizado (%)',
            ]
            indicadores_ordenados = [i for i in ordem_prioritaria if i in indicadores_disponiveis]
            indicadores_restantes = [i for i in indicadores_disponiveis.keys() if i not in indicadores_ordenados]
            indicadores_ordenados.extend(indicadores_restantes)

            col_periodo, col_indicador, col_media = st.columns([1.2, 2, 1.8])
            with col_periodo:
                _idx_periodo_rank = 0
                _p_set25 = _encontrar_periodo(periodos, 3, 2025)
                if _p_set25 and _p_set25 in periodos:
                    _idx_periodo_rank = periodos.index(_p_set25)
                periodo_resumo = st.selectbox(
                    "período",
                    periodos,
                    index=_idx_periodo_rank,
                    key="periodo_resumo",
                    format_func=periodo_para_exibicao
                )
            with col_indicador:
                indicador_label = st.selectbox(
                    "indicador",
                    indicadores_ordenados,
                    key="indicador_resumo"
                )
            with col_media:
                tipo_media_label = st.selectbox(
                    "ponderar média por",
                    list(VARIAVEIS_PONDERACAO.keys()),
                    index=0,
                    key="tipo_media_resumo"
                )
                coluna_peso_resumo = VARIAVEIS_PONDERACAO[tipo_media_label]

            col_bancos = st.columns([1])[0]

            df_periodo = df[df['Período'] == periodo_resumo].copy()

            bancos_todos = df_periodo['Instituição'].dropna().unique().tolist()
            dict_aliases = st.session_state.get('dict_aliases', {})
            bancos_todos = ordenar_bancos_com_alias(bancos_todos, dict_aliases)

            indicador_col = indicadores_disponiveis[indicador_label]

            _default_bancos_rank = _encontrar_bancos_default(bancos_todos)
            with col_bancos:
                bancos_selecionados = st.multiselect(
                    "selecionar instituições (até 40)",
                    bancos_todos,
                    default=_default_bancos_rank,
                    key="bancos_resumo",
                    max_selections=40
                )

            col_ordem, col_sort = st.columns([1.4, 1.8])
            with col_ordem:
                direcao_top = st.radio(
                    "ordem",
                    ["Maior → Menor", "Menor → Maior"],
                    horizontal=True,
                    key="ordem_resumo"
                )
            with col_sort:
                modo_ordenacao = st.radio(
                    "ordenação",
                    ["Ordenar por valor", "Manter ordem de seleção"],
                    horizontal=True,
                    key="ordenacao_resumo"
                )

            format_info = get_axis_format(indicador_col)

            def formatar_numero(valor, fmt_info, incluir_sinal=False):
                if pd.isna(valor):
                    return "N/A"
                valor_formatado = format(valor, fmt_info['tickformat'])
                if incluir_sinal and valor > 0:
                    valor_formatado = f"+{valor_formatado}"
                return f"{valor_formatado}{fmt_info['ticksuffix']}"

            if bancos_selecionados:
                df_selecionado = df_periodo[df_periodo['Instituição'].isin(bancos_selecionados)].copy()
            else:
                df_selecionado = pd.DataFrame()

            if indicador_col in df_selecionado.columns:
                df_selecionado = df_selecionado.dropna(subset=[indicador_col])
            else:
                df_selecionado = pd.DataFrame()

            if grafico_base == "Ranking":
                if indicador_label == "Índice de Basileia":
                    df_capital_base = _preparar_df_capital_base()
                    if df_capital_base.empty:
                        st.info("dados de capital não disponíveis para o ranking.")
                    else:
                        colunas_encontradas_cap, _, _, _ = _mapear_colunas_capital(df_capital_base)
                        df_periodo_cap, basileia_info = _calcular_basileia_periodo(
                            df_capital_base,
                            periodo_resumo,
                            colunas_encontradas_cap,
                        )
                        if basileia_info.get("mensagem") and basileia_info.get("usou_precalc"):
                            st.caption(basileia_info["mensagem"])
                        elif basileia_info.get("mensagem") and df_periodo_cap.empty:
                            st.error(basileia_info["mensagem"])
                            df_periodo_cap = pd.DataFrame()

                        if not df_periodo_cap.empty:
                            colunas_peso_possiveis = [v for v in VARIAVEIS_PONDERACAO.values() if v is not None]
                            colunas_peso = ['Instituição'] + colunas_peso_possiveis
                            colunas_disponiveis = [c for c in colunas_peso if c in df_periodo.columns]
                            if len(colunas_disponiveis) > 1:
                                df_peso = df_periodo[colunas_disponiveis].drop_duplicates(subset=['Instituição'])
                                df_periodo_cap = df_periodo_cap.merge(df_peso, on='Instituição', how='left')

                        if bancos_selecionados:
                            df_selecionado_cap = df_periodo_cap[
                                df_periodo_cap['Instituição'].isin(bancos_selecionados)
                            ].copy()
                        else:
                            df_selecionado_cap = pd.DataFrame()

                        df_selecionado_cap = df_selecionado_cap.dropna(subset=['Índice de Basileia Total (%)'])

                        if df_selecionado_cap.empty:
                            st.info("selecione instituições ou ajuste os filtros para visualizar o ranking.")
                        else:
                            if modo_ordenacao == "Ordenar por valor":
                                ordenar_asc = direcao_top == "Menor → Maior"
                                df_selecionado_cap = df_selecionado_cap.sort_values(
                                    'Índice de Basileia Total (%)', ascending=ordenar_asc
                                )
                            elif bancos_selecionados:
                                ordem = bancos_selecionados
                                df_selecionado_cap['ordem'] = pd.Categorical(
                                    df_selecionado_cap['Instituição'], categories=ordem, ordered=True
                                )
                                df_selecionado_cap = df_selecionado_cap.sort_values('ordem')

                            media_basileia = calcular_media_ponderada(
                                df_selecionado_cap, 'Índice de Basileia Total (%)', coluna_peso_resumo
                            )
                            media_cet1 = calcular_media_ponderada(
                                df_selecionado_cap, 'CET1 (%)', coluna_peso_resumo
                            )
                            media_at1 = calcular_media_ponderada(
                                df_selecionado_cap, 'AT1 (%)', coluna_peso_resumo
                            )
                            media_t2 = calcular_media_ponderada(
                                df_selecionado_cap, 'T2 (%)', coluna_peso_resumo
                            )
                            label_media = get_label_media(coluna_peso_resumo)

                            df_selecionado_cap['Ranking'] = df_selecionado_cap['Índice de Basileia Total (%)'].rank(
                                method='first', ascending=False
                            ).astype(int)
                            df_selecionado_cap['Diferença vs Média (%)'] = (
                                df_selecionado_cap['Índice de Basileia Total (%)'] - media_basileia
                            )

                            n_bancos = len(df_selecionado_cap)
                            cores_componentes = {
                                'CET1 (%)': '#1f77b4',
                                'AT1 (%)': '#ff7f0e',
                                'T2 (%)': '#2ca02c'
                            }

                            fig_basileia = go.Figure()
                            for componente, cor in cores_componentes.items():
                                nome_display = componente.replace(' (%)', '')
                                fig_basileia.add_trace(go.Bar(
                                    x=df_selecionado_cap['Instituição'],
                                    y=df_selecionado_cap[componente],
                                    name=nome_display,
                                    marker_color=cor,
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        f"{nome_display}: %{{y:.2f}}%<extra></extra>"
                                    )
                                ))

                            fig_basileia.add_trace(go.Scatter(
                                x=df_selecionado_cap['Instituição'],
                                y=df_selecionado_cap['Índice de Basileia Total (%)'],
                                mode='text',
                                text=df_selecionado_cap['Índice de Basileia Total (%)'].apply(lambda x: f"{x:.2f}%"),
                                textposition='top center',
                                textfont=dict(size=10, color='#333'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                            fig_basileia.add_trace(go.Scatter(
                                x=df_selecionado_cap['Instituição'],
                                y=[media_basileia] * n_bancos,
                                mode='lines',
                                name=f'{label_media} ({media_basileia:.2f}%)',
                                line=dict(color='#3498db', dash='dash', width=2),
                                hovertemplate=f"{label_media}: {media_basileia:.2f}%<extra></extra>"
                            ))

                            MINIMO_REGULATORIO = 10.5
                            fig_basileia.add_trace(go.Scatter(
                                x=df_selecionado_cap['Instituição'],
                                y=[MINIMO_REGULATORIO] * n_bancos,
                                mode='lines',
                                name=f'Mínimo Regulatório ({MINIMO_REGULATORIO:.1f}%)',
                                line=dict(color='#e74c3c', dash='solid', width=2),
                                hovertemplate=f"Mínimo Regulatório: {MINIMO_REGULATORIO:.1f}%<extra></extra>"
                            ))

                            fig_basileia.update_layout(
                                title=f"Índice de Basileia Total - {periodo_resumo} ({n_bancos} instituições)",
                                xaxis_title="instituições",
                                yaxis_title="índice (%)",
                                plot_bgcolor='#f8f9fa',
                                paper_bgcolor='white',
                                height=max(650, n_bancos * 24),
                                barmode='stack',
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                xaxis=dict(tickangle=-45),
                                yaxis=dict(tickformat='.2f', ticksuffix='%'),
                                font=dict(family='IBM Plex Sans')
                            )

                            st.plotly_chart(fig_basileia, width='stretch', config={'displayModeBar': False})

                            df_export_capital = df_selecionado_cap[[
                                'Instituição', 'CET1 (%)', 'AT1 (%)', 'T2 (%)',
                                'Índice de Basileia Total (%)', 'Ranking', 'Diferença vs Média (%)'
                            ]].copy()
                            df_export_capital.insert(0, 'Período', periodo_resumo)
                            df_export_capital['Tipo de Média'] = tipo_media_label
                            df_export_capital['Média CET1 (%)'] = round(media_cet1, 2)
                            df_export_capital['Média AT1 (%)'] = round(media_at1, 2)
                            df_export_capital['Média T2 (%)'] = round(media_t2, 2)
                            df_export_capital['Média Basileia (%)'] = round(media_basileia, 2)
                            df_export_capital['Mínimo Regulatório (%)'] = MINIMO_REGULATORIO

                            for col in ['CET1 (%)', 'AT1 (%)', 'T2 (%)', 'Índice de Basileia Total (%)', 'Diferença vs Média (%)']:
                                df_export_capital[col] = df_export_capital[col].apply(
                                    lambda x: round(x, 2) if pd.notna(x) else None
                                )

                            with st.expander("exportar dados (excel)"):
                                buffer_excel = BytesIO()
                                with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                    df_export_capital.to_excel(writer, index=False, sheet_name='indice_basileia')
                                buffer_excel.seek(0)

                                st.download_button(
                                    label="exportar excel",
                                    data=buffer_excel,
                                    file_name=f"indice_basileia_{periodo_resumo.replace('/', '-')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="exportar_resumo_excel_basileia"
                                )
                else:
                    if df_selecionado.empty:
                        st.info("selecione instituições ou ajuste os filtros para visualizar o ranking.")
                    else:
                        df_selecionado['valor_display'] = df_selecionado[indicador_col] * format_info['multiplicador']
                        media_display = calcular_media_ponderada(df_selecionado, 'valor_display', coluna_peso_resumo)
                        label_media = get_label_media(coluna_peso_resumo)

                        if modo_ordenacao == "Ordenar por valor":
                            ordenar_asc = direcao_top == "Menor → Maior"
                            df_selecionado = df_selecionado.sort_values('valor_display', ascending=ordenar_asc)
                        elif bancos_selecionados:
                            ordem = bancos_selecionados
                            df_selecionado['ordem'] = pd.Categorical(df_selecionado['Instituição'], categories=ordem, ordered=True)
                            df_selecionado = df_selecionado.sort_values('ordem')

                        df_selecionado['ranking'] = df_selecionado[indicador_col].rank(method='first', ascending=False).astype(int)
                        df_selecionado['diff_media'] = df_selecionado['valor_display'] - media_display

                        if media_display and media_display != 0:
                            df_selecionado['diff_pct'] = (df_selecionado['valor_display'] / media_display - 1) * 100
                            df_selecionado['diff_pct_text'] = df_selecionado['diff_pct'].map(lambda v: f"{v:.1f}%")
                        else:
                            df_selecionado['diff_pct_text'] = "N/A"

                        df_selecionado['valor_text'] = df_selecionado['valor_display'].map(
                            lambda v: formatar_numero(v, format_info)
                        )
                        df_selecionado['diff_text'] = df_selecionado['diff_media'].map(
                            lambda v: formatar_numero(v, format_info, incluir_sinal=True)
                        )

                        n_bancos = len(df_selecionado)
                        orientacao_horizontal = n_bancos > 15
                        altura_grafico = max(650, n_bancos * 24) if orientacao_horizontal else 650

                        cores_plotly = px.colors.qualitative.Plotly
                        cores_barras = []
                        idx_cor = 0
                        for banco in df_selecionado['Instituição']:
                            cor = obter_cor_banco(banco)
                            if not cor:
                                cor = cores_plotly[idx_cor % len(cores_plotly)]
                                idx_cor += 1
                            cores_barras.append(cor)

                        fig_resumo = go.Figure()
                        banco_hover = "%{y}" if orientacao_horizontal else "%{x}"
                        fig_resumo.add_trace(go.Bar(
                            x=df_selecionado['valor_display'] if orientacao_horizontal else df_selecionado['Instituição'],
                            y=df_selecionado['Instituição'] if orientacao_horizontal else df_selecionado['valor_display'],
                            marker=dict(color=cores_barras, opacity=0.85),
                            name=indicador_label,
                            orientation='h' if orientacao_horizontal else 'v',
                            customdata=np.stack([
                                df_selecionado['ranking'],
                                df_selecionado['diff_text'],
                                df_selecionado['diff_pct_text'],
                                df_selecionado['valor_text'],
                            ], axis=-1),
                            hovertemplate=(
                                f"<b>{banco_hover}</b><br>"
                                f"{indicador_label}: %{{customdata[3]}}<br>"
                                "Ranking: %{customdata[0]}<br>"
                                "Diferença vs média: %{customdata[1]}<br>"
                                "Diferença vs média (%): %{customdata[2]}"
                                "<extra></extra>"
                            )
                        ))

                        if orientacao_horizontal:
                            fig_resumo.add_trace(go.Scatter(
                                x=[media_display] * len(df_selecionado),
                                y=df_selecionado['Instituição'],
                                mode='lines',
                                name=label_media,
                                line=dict(color='#1f77b4', dash='dash')
                            ))
                        else:
                            fig_resumo.add_trace(go.Scatter(
                                x=df_selecionado['Instituição'],
                                y=[media_display] * len(df_selecionado),
                                mode='lines',
                                name=label_media,
                                line=dict(color='#1f77b4', dash='dash')
                            ))

                        fig_resumo.update_layout(
                            title=f"{indicador_label} - {periodo_resumo} ({len(df_selecionado)} instituições)",
                            xaxis_title=indicador_label if orientacao_horizontal else "instituições",
                            yaxis_title="instituições" if orientacao_horizontal else indicador_label,
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='white',
                            height=altura_grafico,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            xaxis=dict(
                                tickangle=-45 if not orientacao_horizontal else 0,
                                tickformat=format_info['tickformat'] if orientacao_horizontal else None,
                                ticksuffix=format_info['ticksuffix'] if orientacao_horizontal else None
                            ),
                            yaxis=dict(
                                tickformat=format_info['tickformat'] if not orientacao_horizontal else None,
                                ticksuffix=format_info['ticksuffix'] if not orientacao_horizontal else None
                            ),
                            font=dict(family='IBM Plex Sans')
                        )

                        st.plotly_chart(fig_resumo, width='stretch', config={'displayModeBar': False})

                        media_grupo_raw = calcular_media_ponderada(df_selecionado, indicador_col, coluna_peso_resumo)
                        df_export = df_selecionado.copy()
                        df_export['Período'] = periodo_resumo
                        df_export['Indicador'] = indicador_label
                        df_export['Valor'] = df_export[indicador_col]
                        df_export['Média do Grupo'] = media_grupo_raw
                        df_export['Tipo de Média'] = tipo_media_label
                        df_export['Diferença vs Média'] = df_export['Valor'] - media_grupo_raw
                        df_export = df_export[[
                            'Período',
                            'Instituição',
                            'Indicador',
                            'Valor',
                            'ranking',
                            'Média do Grupo',
                            'Tipo de Média',
                            'Diferença vs Média'
                        ]].rename(columns={'ranking': 'Ranking'})

                        with st.expander("exportar dados (excel)"):
                            buffer_excel = BytesIO()
                            with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                df_export.to_excel(writer, index=False, sheet_name='ranking')
                            buffer_excel.seek(0)

                            st.download_button(
                                label="exportar excel",
                                data=buffer_excel,
                                file_name=f"ranking_{periodo_resumo.replace('/', '-')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="exportar_resumo_excel"
                            )

            if grafico_base == "Deltas (antes e depois)":
                st.markdown("---")

                st.markdown("### deltas (antes e depois)")
                variaveis_selecionadas_delta = [indicador_label]
                delta_colunas_map = {label: col for label, col in indicadores_disponiveis.items()}

                periodo_inicial_delta = periodo_resumo
                periodos_subsequentes = [
                    periodo for periodo in periodos_dropdown if periodo != periodo_inicial_delta
                ]
                if not periodos_subsequentes:
                    periodos_subsequentes = periodos_dropdown

                # ===== LINHA 2: Seleção de período subsequente e tipo de variação =====
                col_p2, col_tipo_var = st.columns([2, 1])
                with col_p2:
                    periodo_subsequente_delta = st.selectbox(
                        "período subsequente",
                        periodos_subsequentes,
                        index=0,
                        key="periodo_subsequente_delta",
                        format_func=periodo_para_exibicao
                    )
                    periodo_valido = periodo_subsequente_delta != periodo_inicial_delta
                    if not periodo_valido:
                        st.warning("selecione um período subsequente diferente do período inicial.")
                with col_tipo_var:
                    tipo_variacao = st.radio(
                        "ordenar por",
                        ["Δ absoluto", "Δ %"],
                        index=1,
                        key="tipo_variacao_delta",
                        horizontal=True
                    )
                bancos_selecionados_delta = bancos_selecionados

                if periodo_valido and variaveis_selecionadas_delta and bancos_selecionados_delta:
                    df_inicial = df[df['Período'] == periodo_inicial_delta].copy()
                    df_subsequente = df[df['Período'] == periodo_subsequente_delta].copy()

                    if modo_visualizacao_deltas == "Pontos (antes/depois)":
                        st.markdown("---")
                        col_escala1, col_escala2, col_escala3 = st.columns([1, 1, 2])

                        with col_escala1:
                            if 'delta_escala_modo' not in st.session_state:
                                st.session_state['delta_escala_modo'] = 'Auto (zoom)'

                            modo_escala = st.radio(
                                "escala do eixo Y",
                                ["Auto (zoom)", "Zero baseline", "Manual"],
                                index=["Auto (zoom)", "Zero baseline", "Manual"].index(st.session_state['delta_escala_modo']),
                                key="delta_escala_modo_radio",
                                horizontal=True
                            )
                            st.session_state['delta_escala_modo'] = modo_escala

                        with col_escala2:
                            if 'delta_escala_margem' not in st.session_state:
                                st.session_state['delta_escala_margem'] = 10

                            if modo_escala == "Auto (zoom)":
                                margem_pct = st.slider(
                                    "margem (%)",
                                    0, 50, st.session_state['delta_escala_margem'],
                                    key="delta_margem_slider",
                                    help="Margem adicional acima/abaixo dos valores"
                                )
                                st.session_state['delta_escala_margem'] = margem_pct

                        with col_escala3:
                            if modo_escala == "Manual":
                                col_min, col_max = st.columns(2)
                                with col_min:
                                    if 'delta_y_min' not in st.session_state:
                                        st.session_state['delta_y_min'] = 0.0
                                    y_min_manual = st.number_input(
                                        "Y mínimo",
                                        value=st.session_state['delta_y_min'],
                                        key="delta_y_min_input"
                                    )
                                    st.session_state['delta_y_min'] = y_min_manual
                                with col_max:
                                    if 'delta_y_max' not in st.session_state:
                                        st.session_state['delta_y_max'] = 100.0
                                    y_max_manual = st.number_input(
                                        "Y máximo",
                                        value=st.session_state['delta_y_max'],
                                        key="delta_y_max_input"
                                    )
                                    st.session_state['delta_y_max'] = y_max_manual

                        st.markdown("---")

                    for variavel in variaveis_selecionadas_delta:
                        coluna_variavel = delta_colunas_map.get(variavel, variavel)
                        if coluna_variavel not in df.columns:
                            st.warning(f"variável '{variavel}' não encontrada nos dados")
                            continue

                        format_info = get_axis_format(variavel)

                        dados_grafico = []
                        for instituicao in bancos_selecionados_delta:
                            valor_ini = df_inicial[df_inicial['Instituição'] == instituicao][coluna_variavel].values
                            valor_sub = df_subsequente[df_subsequente['Instituição'] == instituicao][coluna_variavel].values

                            if len(valor_ini) > 0 and len(valor_sub) > 0:
                                v_ini = valor_ini[0]
                                v_sub = valor_sub[0]

                                if pd.isna(v_ini) or pd.isna(v_sub):
                                    continue

                                delta_absoluto = v_sub - v_ini

                                if variavel in VARS_PERCENTUAL:
                                    delta_texto = f"{delta_absoluto * 100:+.2f}pp"
                                elif variavel in VARS_MOEDAS:
                                    delta_texto = f"R$ {delta_absoluto/1e6:+,.0f}MM".replace(",", ".")
                                else:
                                    delta_texto = f"{delta_absoluto:+.2f}"

                                if v_ini == 0:
                                    if delta_absoluto > 0:
                                        variacao_pct = float('inf')
                                        variacao_texto = "Valor Inicial 0 - ∞"
                                    elif delta_absoluto < 0:
                                        variacao_pct = float('-inf')
                                        variacao_texto = "Valor Inicial 0 - ∞"
                                    else:
                                        variacao_pct = 0
                                        variacao_texto = "0.0%"
                                elif v_ini < 0 and v_sub > 0:
                                    variacao_pct = ((v_sub - v_ini) / abs(v_ini)) * 100
                                    variacao_texto = f"{variacao_pct:+.1f}% (inversão)"
                                elif v_ini > 0 and v_sub < 0:
                                    variacao_pct = ((v_sub - v_ini) / abs(v_ini)) * 100
                                    variacao_texto = f"{variacao_pct:+.1f}% (inversão)"
                                else:
                                    variacao_pct = ((v_sub - v_ini) / abs(v_ini)) * 100
                                    variacao_texto = f"{variacao_pct:+.1f}%"

                                dados_grafico.append({
                                    'instituicao': instituicao,
                                    'valor_ini': v_ini,
                                    'valor_sub': v_sub,
                                    'delta': delta_absoluto,
                                    'delta_texto': delta_texto,
                                    'variacao_pct': variacao_pct if not (variacao_pct == float('inf') or variacao_pct == float('-inf')) else (1e10 if variacao_pct > 0 else -1e10),
                                    'variacao_texto': variacao_texto
                                })

                        if not dados_grafico:
                            st.info(f"sem dados disponíveis para '{variavel}' nos períodos selecionados")
                            continue

                        if tipo_variacao == "Δ %":
                            dados_grafico = sorted(dados_grafico, key=lambda x: x['variacao_pct'], reverse=True)
                        else:
                            dados_grafico = sorted(dados_grafico, key=lambda x: x['delta'], reverse=True)

                        if modo_visualizacao_deltas == "Pontos (antes/depois)":
                            fig_delta = go.Figure()
                            todos_y = []
                            for dado in dados_grafico:
                                todos_y.append(dado['valor_ini'] * format_info['multiplicador'])
                                todos_y.append(dado['valor_sub'] * format_info['multiplicador'])

                            for i, dado in enumerate(dados_grafico):
                                inst = dado['instituicao']
                                y_ini = dado['valor_ini'] * format_info['multiplicador']
                                y_sub = dado['valor_sub'] * format_info['multiplicador']
                                delta_positivo = dado['delta'] > 0

                                cor_sub = '#2E7D32' if delta_positivo else '#7B1E3A'

                                fig_delta.add_trace(go.Scatter(
                                    x=[inst, inst],
                                    y=[y_ini, y_sub],
                                    mode='lines',
                                    line=dict(color='#9E9E9E', width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))

                                fig_delta.add_trace(go.Scatter(
                                    x=[inst],
                                    y=[y_ini],
                                    mode='markers',
                                    marker=dict(size=12, color='#424242', line=dict(width=1, color='white')),
                                    name=periodo_inicial_delta if i == 0 else None,
                                    showlegend=(i == 0),
                                    legendgroup='inicial',
                                    hovertemplate=f'<b>{inst}</b><br>{periodo_inicial_delta}: %{{y:{format_info["tickformat"]}}}{format_info["ticksuffix"]}<extra></extra>'
                                ))

                                fig_delta.add_trace(go.Scatter(
                                    x=[inst],
                                    y=[y_sub],
                                    mode='markers',
                                    marker=dict(size=12, color=cor_sub, line=dict(width=1, color='white')),
                                    name=periodo_subsequente_delta if i == 0 else None,
                                    showlegend=(i == 0),
                                    legendgroup='subsequente',
                                    customdata=[[dado['delta_texto'], dado['variacao_texto']]],
                                    hovertemplate=f'<b>{inst}</b><br>{periodo_subsequente_delta}: %{{y:{format_info["tickformat"]}}}{format_info["ticksuffix"]}<br>Δ: %{{customdata[0]}}<br>Variação: %{{customdata[1]}}<extra></extra>'
                                ))

                            titulo_delta = f"{variavel}: {periodo_inicial_delta} → {periodo_subsequente_delta}"

                            yaxis_config = dict(
                                showgrid=True,
                                gridcolor='#e0e0e0',
                                tickformat=format_info['tickformat'],
                                ticksuffix=format_info['ticksuffix'],
                                title=variavel
                            )

                            if todos_y:
                                y_min_dados = min(todos_y)
                                y_max_dados = max(todos_y)
                                y_range = y_max_dados - y_min_dados if y_max_dados != y_min_dados else abs(y_max_dados) * 0.1 or 1

                                if modo_escala == "Zero baseline":
                                    yaxis_config['range'] = [min(0, y_min_dados - y_range * 0.05), max(0, y_max_dados + y_range * 0.05)]
                                elif modo_escala == "Auto (zoom)":
                                    margem = y_range * (margem_pct / 100)
                                    yaxis_config['range'] = [y_min_dados - margem, y_max_dados + margem]
                                elif modo_escala == "Manual":
                                    yaxis_config['range'] = [y_min_manual, y_max_manual]

                            fig_delta.update_layout(
                                title=dict(
                                    text=titulo_delta,
                                    font=dict(size=16, family='IBM Plex Sans')
                                ),
                                height=max(400, len(dados_grafico) * 25 + 150),
                                plot_bgcolor='#f8f9fa',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="left",
                                    x=0
                                ),
                                xaxis=dict(
                                    showgrid=False,
                                    tickangle=45 if len(dados_grafico) > 10 else 0,
                                    tickfont=dict(size=10)
                                ),
                                yaxis=yaxis_config,
                                font=dict(family='IBM Plex Sans'),
                                margin=dict(l=60, r=20, t=80, b=100)
                            )

                            st.markdown(f"### {variavel}")
                            st.plotly_chart(fig_delta, width='stretch', config={'displayModeBar': False})
                        else:
                            valores_finitos = []
                            for dado in dados_grafico:
                                if tipo_variacao == "Δ %":
                                    if np.isfinite(dado['variacao_pct']):
                                        valores_finitos.append(abs(dado['variacao_pct']))

                            cap_visual = (max(valores_finitos) * 1.2) if valores_finitos else 1

                            for dado in dados_grafico:
                                if tipo_variacao == "Δ %":
                                    if np.isfinite(dado['variacao_pct']):
                                        dado['valor_plot'] = dado['variacao_pct']
                                    else:
                                        dado['valor_plot'] = cap_visual if dado['variacao_pct'] > 0 else -cap_visual
                                else:
                                    dado['valor_plot'] = dado['delta'] * format_info['multiplicador']

                            n_bancos = len(dados_grafico)
                            orientacao_horizontal = n_bancos > 15

                            insts = [d['instituicao'] for d in dados_grafico]
                            valores_plot = [d['valor_plot'] for d in dados_grafico]
                            cores_barras = ['#2E7D32' if d['delta'] > 0 else '#7B1E3A' for d in dados_grafico]

                            eixo_tickformat = '.1f' if tipo_variacao == "Δ %" else format_info['tickformat']
                            eixo_ticksuffix = '%' if tipo_variacao == "Δ %" else format_info['ticksuffix']
                            eixo_titulo = "Δ %" if tipo_variacao == "Δ %" else "Δ absoluto"

                            fig_barras = go.Figure()
                            fig_barras.add_trace(go.Bar(
                                x=valores_plot if orientacao_horizontal else insts,
                                y=insts if orientacao_horizontal else valores_plot,
                                orientation='h' if orientacao_horizontal else 'v',
                                marker=dict(color=cores_barras, opacity=0.9),
                                customdata=np.stack([
                                    [d['delta_texto'] for d in dados_grafico],
                                    [d['variacao_texto'] for d in dados_grafico],
                                ], axis=-1),
                                hovertemplate=(
                                    "<b>%{y}</b><br>" if orientacao_horizontal else "<b>%{x}</b><br>"
                                ) + "Δ: %{customdata[0]}<br>Variação: %{customdata[1]}<extra></extra>"
                            ))

                            fig_barras.update_layout(
                                title=dict(
                                    text=f"{variavel}: {periodo_inicial_delta} → {periodo_subsequente_delta}",
                                    font=dict(size=16, family='IBM Plex Sans')
                                ),
                                height=max(450, len(dados_grafico) * 25 + 160),
                                plot_bgcolor='#f8f9fa',
                                paper_bgcolor='white',
                                showlegend=False,
                                xaxis=dict(
                                    showgrid=True if not orientacao_horizontal else True,
                                    zeroline=True,
                                    zerolinecolor='#444',
                                    tickformat=eixo_tickformat if orientacao_horizontal else None,
                                    ticksuffix=eixo_ticksuffix if orientacao_horizontal else None,
                                    title=eixo_titulo if orientacao_horizontal else None
                                ),
                                yaxis=dict(
                                    showgrid=False if orientacao_horizontal else True,
                                    zeroline=True,
                                    zerolinecolor='#444',
                                    tickformat=eixo_tickformat if not orientacao_horizontal else None,
                                    ticksuffix=eixo_ticksuffix if not orientacao_horizontal else None,
                                    title=eixo_titulo if not orientacao_horizontal else None
                                ),
                                font=dict(family='IBM Plex Sans'),
                                margin=dict(l=60, r=20, t=80, b=100)
                            )

                            st.markdown(f"### {variavel}")
                            st.plotly_chart(fig_barras, width='stretch', config={'displayModeBar': False})

                        if bancos_selecionados_delta:
                            idx_ini_hist = periodos_disponiveis.index(periodo_inicial_delta)
                            idx_fin_hist = periodos_disponiveis.index(periodo_subsequente_delta)
                            if idx_ini_hist > idx_fin_hist:
                                idx_ini_hist, idx_fin_hist = idx_fin_hist, idx_ini_hist
                            periodos_hist = periodos_disponiveis[idx_ini_hist:idx_fin_hist + 1]

                            df_hist = df[
                                df['Período'].isin(periodos_hist)
                                & df['Instituição'].isin(bancos_selecionados_delta)
                            ].copy()

                            if not df_hist.empty and variavel in df_hist.columns:
                                format_hist = get_axis_format(variavel)
                                fig_hist = go.Figure()
                                for instituicao in bancos_selecionados_delta:
                                    df_banco = df_hist[df_hist['Instituição'] == instituicao].copy()
                                    if df_banco.empty:
                                        continue
                                    df_banco['ano'] = df_banco['Período'].str.split('/').str[1].astype(int)
                                    df_banco['trimestre'] = df_banco['Período'].str.split('/').str[0].astype(int)
                                    df_banco = df_banco.sort_values(['ano', 'trimestre'])
                                    y_values = df_banco[variavel] * format_hist['multiplicador']
                                    cor_banco = obter_cor_banco(instituicao) or None

                                    if variavel == 'Lucro Líquido Acumulado YTD':
                                        fig_hist.add_trace(go.Bar(
                                            x=df_banco['Período'],
                                            y=y_values,
                                            name=instituicao,
                                            marker=dict(color=cor_banco),
                                            hovertemplate=(
                                                f'<b>{instituicao}</b><br>%{{x}}<br>'
                                                f'%{{y:{format_hist["tickformat"]}}}{format_hist["ticksuffix"]}<extra></extra>'
                                            )
                                        ))
                                    else:
                                        fig_hist.add_trace(go.Scatter(
                                            x=df_banco['Período'],
                                            y=y_values,
                                            mode='lines',
                                            name=instituicao,
                                            line=dict(width=2, color=cor_banco),
                                            hovertemplate=(
                                                f'<b>{instituicao}</b><br>%{{x}}<br>'
                                                f'%{{y:{format_hist["tickformat"]}}}{format_hist["ticksuffix"]}<extra></extra>'
                                            )
                                        ))

                                st.markdown("### evolução histórica (dados brutos)")
                                fig_hist.update_layout(
                                    height=320,
                                    margin=dict(l=10, r=10, t=40, b=30),
                                    plot_bgcolor='#f8f9fa',
                                    paper_bgcolor='white',
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                    xaxis=dict(
                                        showgrid=False,
                                        tickmode='array' if variavel == 'Lucro Líquido Acumulado YTD' else None,
                                        tickvals=periodos_hist if variavel == 'Lucro Líquido Acumulado YTD' else None,
                                        ticktext=periodos_hist if variavel == 'Lucro Líquido Acumulado YTD' else None,
                                        categoryorder='array' if variavel == 'Lucro Líquido Acumulado YTD' else None,
                                        categoryarray=periodos_hist if variavel == 'Lucro Líquido Acumulado YTD' else None
                                    ),
                                    yaxis=dict(
                                        showgrid=True,
                                        gridcolor='#e0e0e0',
                                        tickformat=format_hist['tickformat'],
                                        ticksuffix=format_hist['ticksuffix']
                                    ),
                                    font=dict(family='IBM Plex Sans'),
                                    barmode='group' if variavel == 'Lucro Líquido Acumulado YTD' else None
                                )

                                st.plotly_chart(fig_hist, width='stretch', config={'displayModeBar': False})

                                df_export_hist = df_hist.pivot_table(
                                    index='Instituição',
                                    columns='Período',
                                    values=variavel,
                                    aggfunc='first'
                                ).reset_index()
                                colunas_ordenadas = ['Instituição'] + periodos_hist
                                df_export_hist = df_export_hist.reindex(columns=colunas_ordenadas)

                                buffer_excel_hist = BytesIO()
                                with pd.ExcelWriter(buffer_excel_hist, engine='xlsxwriter') as writer:
                                    df_export_hist.to_excel(writer, index=False, sheet_name='historico')
                                buffer_excel_hist.seek(0)

                                st.download_button(
                                    label="exportar histórico excel",
                                    data=buffer_excel_hist,
                                    file_name=f"Historico_{variavel}_{periodo_inicial_delta.replace('/', '-')}_{periodo_subsequente_delta.replace('/', '-')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"exportar_historico_delta_{variavel}"
                                )

                        with st.expander("exportar dados (excel)"):
                            df_resumo = pd.DataFrame(dados_grafico)
                            df_resumo = df_resumo.rename(columns={
                                'instituicao': 'Instituição',
                                'valor_ini': periodo_inicial_delta,
                                'valor_sub': periodo_subsequente_delta,
                                'delta_texto': 'Delta',
                                'variacao_texto': 'Variação %'
                            })
                            df_resumo = df_resumo[['Instituição', periodo_inicial_delta, periodo_subsequente_delta, 'Delta', 'Variação %']]
                            st.dataframe(df_resumo, use_container_width=True)

                            buffer_excel = BytesIO()
                            with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                df_resumo.to_excel(writer, index=False, sheet_name='deltas')
                            buffer_excel.seek(0)

                            nome_variavel = variavel.replace(' ', '_').replace('/', '_')
                            st.download_button(
                                label="exportar excel",
                                data=buffer_excel,
                                file_name=f"Deltas_{variavel}_{periodo_inicial_delta.replace('/', '-')}_{periodo_subsequente_delta.replace('/', '-')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"exportar_excel_delta_{variavel}"
                            )
                elif not periodo_valido:
                    pass  # Já exibiu warning acima
                elif not variaveis_selecionadas_delta:
                    st.info("selecione ao menos uma variável para análise")
                else:
                    st.info("selecione instituições para comparar")

    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "DRE":
    st.markdown("### Demonstração de Resultado (DRE)")
    st.caption("Tabela DRE a partir de Mar/25 com YTD irregular e YoY.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_dre_data():
        manager = get_cache_manager()
        resultado = manager.carregar("dre")
        if resultado.sucesso and resultado.dados is not None:
            return resultado.dados, None
        return None, resultado.mensagem

    def normalize_sources(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace("|", ";").split(";")]
            return [p for p in parts if p]
        return []

    def load_dre_mapping():
        return [
            {
                "label": "Resultado de Intermediação Financeira Bruto",
                "sources": [
                    "Rendas de Aplicações Interfinanceiras de Liquidez (a)",
                    "Rendas de Títulos e Valores Mobiliários (b)",
                    "Rendas de Operações de Crédito (c)",
                    "Rendas de Arrendamento Financeiro (d)",
                    "Rendas de Outras Operações com Características de Concessão de Crédito (e)",
                ],
                "concept": "Resultado de intermediação financeira bruto (a+b+c+d+e).",
                "original_label": "Resultado de Intermediação Financeira Bruto",
            },
            {
                "label": "Rec. Aplicações Interfinanceiras Liquidez",
                "sources": ["Rendas de Aplicações Interfinanceiras de Liquidez (a)"],
                "concept": "Receitas de aplicações interfinanceiras de liquidez.",
                "original_label": "Rendas de Aplicações Interfinanceiras de Liquidez (a)",
                "is_child": True,
            },
            {
                "label": "Rec. TVMs",
                "sources": ["Rendas de Títulos e Valores Mobiliários (b)"],
                "concept": "Receitas de títulos e valores mobiliários.",
                "original_label": "Rendas de Títulos e Valores Mobiliários (b)",
                "is_child": True,
            },
            {
                "label": "Rec. Crédito",
                "sources": ["Rendas de Operações de Crédito (c)"],
                "concept": "Receitas de operações de crédito.",
                "original_label": "Rendas de Operações de Crédito (c)",
                "is_child": True,
            },
            {
                "label": "Rec. Arrendamento Financeiro",
                "sources": ["Rendas de Arrendamento Financeiro (d)"],
                "concept": "Receitas de arrendamento financeiro.",
                "original_label": "Rendas de Arrendamento Financeiro (d)",
                "is_child": True,
            },
            {
                "label": "Rec. Outras Operações c/ Características de Crédito",
                "sources": ["Rendas de Outras Operações com Características de Concessão de Crédito (e)"],
                "concept": "Receitas de outras operações com características de crédito.",
                "original_label": "Rendas de Outras Operações com Características de Concessão de Crédito (e)",
                "is_child": True,
            },
            {
                "label": "Desp. PDD",
                "sources": ["Resultado com Perda Esperada (f)"],
                "concept": "Despesa com perdas esperadas (PDD).",
                "original_label": "Resultado com Perda Esperada (f)",
            },
            {
                "label": "Desp. Captação",
                "sources": ["Despesas de Captações (g)"],
                "concept": "Despesas de captação.",
                "original_label": "Despesas de Captações (g)",
            },
            {
                "label": "Desp PDD / NIM bruta",
                "derived_metric": "Desp PDD / NIM bruta",
                "format": "pct",
                "concept": "Desp. PDD dividido pela NIM bruta (Rec. Crédito + Rec. Arrendamento Financeiro + Rec. Outras Operações c/ Características de Crédito).",
            },
            {
                "label": "Desp PDD / Resultado Intermediação Fin. Bruto",
                "derived_metric": "Desp PDD / Resultado Intermediação Fin. Bruto",
                "format": "pct",
                "concept": "Desp. PDD dividido pelo Resultado de Intermediação Financeira Bruto.",
            },
            {
                "label": "Desp Captação / Captação",
                "derived_metric": "Desp Captação / Captação",
                "format": "pct",
                "concept": "Desp. Captação anualizada dividida por Captações.",
            },
            {
                "label": "Desp. Dívida Elegível a Capital",
                "sources": ["Despesas de Instrumentos de Dívida Elegíveis a Capital (h)"],
                "concept": "Despesas com dívida elegível a capital.",
                "original_label": "Despesas de Instrumentos de Dívida Elegíveis a Capital (h)",
            },
            {
                "label": "Res. Derivativos",
                "sources": ["Resultado com Derivativos (i)"],
                "concept": "Resultado com derivativos.",
                "original_label": "Resultado com Derivativos (i)",
            },
            {
                "label": "Outros Res. Intermediação Financeira",
                "sources": ["Outros Resultados de Intermediação Financeira (j)"],
                "concept": "Outros resultados de intermediação financeira.",
                "original_label": "Outros Resultados de Intermediação Financeira (j)",
            },
            {
                "label": "Resultado Int. Financeira Líquido",
                "sources": [
                    "Rendas de Aplicações Interfinanceiras de Liquidez (a)",
                    "Rendas de Títulos e Valores Mobiliários (b)",
                    "Rendas de Operações de Crédito (c)",
                    "Rendas de Arrendamento Financeiro (d)",
                    "Rendas de Outras Operações com Características de Concessão de Crédito (e)",
                    "Resultado com Perda Esperada (f)",
                    "Despesas de Captações (g)",
                    "Despesas de Instrumentos de Dívida Elegíveis a Capital (h)",
                    "Resultado com Derivativos (i)",
                    "Outros Resultados de Intermediação Financeira (j)",
                ],
                "concept": "Resultado de intermediação financeira líquido.",
                "original_label": "Resultado de Intermediação Financeira (k) = (a) + (b) + (c) + (d) + (e) + (f) + (g) + (h) + (i) + (j)",
            },
            {
                "label": "Resultado Transações Pgto",
                "sources": ["Resultado com Transações de Pagamento (l)"],
                "concept": "Resultado com transações de pagamento.",
                "original_label": "Resultado com Transações de Pagamento (l)",
            },
            {
                "label": "Renda Tarifas Bancárias",
                "sources": ["Rendas de Tarifas Bancárias (m)"],
                "concept": "Receitas de tarifas bancárias.",
                "original_label": "Rendas de Tarifas Bancárias (m)",
            },
            {
                "label": "Outras Prestações de Serviços",
                "sources": ["Outras Rendas de Prestação de Serviços (n)"],
                "concept": "Outras receitas de prestação de serviços.",
                "original_label": "Outras Rendas de Prestação de Serviços (n)",
            },
            {
                "label": "Desp. Pessoal",
                "sources": ["Despesas de Pessoal (o)"],
                "concept": "Despesas com pessoal.",
                "original_label": "Despesas de Pessoal (o)",
            },
            {
                "label": "Desp. Adm",
                "sources": ["Despesas Administrativas (p)"],
                "concept": "Despesas administrativas.",
                "original_label": "Despesas Administrativas (p)",
            },
            {
                "label": "Desp. PDD Outras Operações",
                "sources": ["Resultado com Perdas Esperadas de Outras Operações (q)"],
                "concept": "Perdas esperadas de outras operações.",
                "original_label": "Resultado com Perdas Esperadas de Outras Operações (q)",
            },
            {
                "label": "Desp. JSCP Cooperativas",
                "sources": ["Despesas de Juros Sobre Capital Próprio de Cooperativas (r)"],
                "concept": "Juros sobre capital próprio (cooperativas).",
                "original_label": "Despesas de Juros Sobre Capital Próprio de Cooperativas (r)",
            },
            {
                "label": "Desp. Tributárias",
                "sources": ["Despesas Tributárias (s)"],
                "concept": "Despesas tributárias.",
                "original_label": "Despesas Tributárias (s)",
            },
            {
                "label": "Res. Participação Controladas",
                "sources": ["Resultado de Participações (t)"],
                "concept": "Resultado de participações em controladas/coligadas.",
                "original_label": "Resultado de Participações (t)",
            },
            {
                "label": "Outras Receitas",
                "sources": ["Outras Receitas (u)"],
                "concept": "Outras receitas.",
                "original_label": "Outras Receitas (u)",
            },
            {
                "label": "Outras Despesas",
                "sources": ["Outras Despesas (v)"],
                "concept": "Outras despesas.",
                "original_label": "Outras Despesas (v)",
            },
            {
                "label": "IR/CSLL",
                "sources": ["Imposto de Renda e Contribuição Social (y)"],
                "concept": "Imposto de renda e contribuição social.",
                "original_label": "Imposto de Renda e Contribuição Social (y)",
            },
            {
                "label": "Res. Participação Lucro",
                "sources": ["Participações no Lucro (z)"],
                "concept": "Participações no lucro.",
                "original_label": "Participações no Lucro (z)",
            },
            {
                "label": "Lucro Líquido Período Acumulado",
                "sources": ["Lucro Líquido (aa) = (x) + (y) + (z)"],
                "concept": "Lucro líquido acumulado no período.",
                "original_label": "Lucro Líquido (aa) = (x) + (y) + (z)",
            },
        ]

    def find_column(df, source_name: str):
        if source_name in df.columns:
            return source_name
        target = source_name.strip().lower()
        for col in df.columns:
            if str(col).strip().lower() == target:
                return col
        for col in df.columns:
            if target in str(col).strip().lower():
                return col
        return None

    def coerce_numeric(series: pd.Series) -> pd.Series:
        if series is None:
            return series
        if series.dtype == object:
            cleaned = (
                series.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            return pd.to_numeric(cleaned, errors="coerce")
        return pd.to_numeric(series, errors="coerce")

    def detectar_colunas_basicas(df: pd.DataFrame):
        col_periodo = None
        for candidato in ["Período", "Periodo", "PERIODO", "PERÍODO"]:
            if candidato in df.columns:
                col_periodo = candidato
                break
        if col_periodo is None:
            for col in df.columns:
                if "period" in str(col).lower():
                    col_periodo = col
                    break
        col_inst = None
        for candidato in ["Instituição", "Instituicao", "INSTITUICAO", "INSTITUIÇÃO"]:
            if candidato in df.columns:
                col_inst = candidato
                break
        if col_inst is None:
            for col in df.columns:
                if "institu" in str(col).lower():
                    col_inst = col
                    break
        return col_periodo, col_inst

    def parse_periodo(periodo_val):
        if periodo_val is None:
            return None, None
        texto = str(periodo_val).strip()
        if "/" in texto:
            partes = texto.split("/")
            if len(partes) >= 2 and partes[0].isdigit() and partes[1].isdigit():
                parte1 = int(partes[0])
                ano = int(partes[1])
                if 1 <= parte1 <= 4:
                    mes = {1: 3, 2: 6, 3: 9, 4: 12}.get(parte1)
                else:
                    mes = parte1
                return ano, mes
        if texto.isdigit():
            if len(texto) == 6:
                ano = int(texto[:4])
                mes = int(texto[4:])
                return ano, mes
            if len(texto) == 8:
                ano = int(texto[:4])
                mes = int(texto[4:6])
                return ano, mes
        return None, None

    def compute_ytd_irregular(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[["ano", "mes"]] = df["Periodo"].apply(
            lambda x: pd.Series(parse_periodo(x))
        )
        df["ytd"] = pd.NA
        for (instituicao, label, ano), grupo in df.groupby(["Instituicao", "Label", "ano"]):
            valores_mes = {row["mes"]: row["valor"] for _, row in grupo.iterrows()}
            for idx, row in grupo.iterrows():
                mes = row["mes"]
                valor = row["valor"]
                ytd_val = pd.NA
                if pd.isna(valor):
                    ytd_val = pd.NA
                elif mes in (3, 6):
                    ytd_val = valor
                elif mes in (9, 12):
                    valor_jun = valores_mes.get(6)
                    if valor_jun is None or pd.isna(valor_jun):
                        ytd_val = pd.NA
                    else:
                        ytd_val = valor + valor_jun
                else:
                    ytd_val = valor
                df.at[idx, "ytd"] = ytd_val
        return df

    def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["yoy"] = pd.NA
        for (instituicao, label, mes), grupo in df.groupby(["Instituicao", "Label", "mes"]):
            valores_ano = {
                row["ano"]: row["ytd"]
                for _, row in grupo.iterrows()
                if row["ano"] is not None
            }
            for idx, row in grupo.iterrows():
                ano = row["ano"]
                if ano is None or pd.isna(row["ytd"]):
                    continue
                anterior = valores_ano.get(ano - 1)
                if anterior is None or pd.isna(anterior) or anterior == 0:
                    continue
                df.at[idx, "yoy"] = (row["ytd"] / anterior) - 1
        return df

    def compute_line_values(df_base: pd.DataFrame, entry: dict) -> pd.DataFrame:
        if entry.get("derived_metric"):
            return pd.DataFrame()
        fontes = normalize_sources(entry.get("sources", []))
        colunas = []
        for fonte in fontes:
            col = find_column(df_base, fonte)
            if col:
                colunas.append(col)
        if not colunas:
            return pd.DataFrame()
        series_list = [coerce_numeric(df_base[col]) for col in colunas]
        valores = pd.concat(series_list, axis=1).sum(axis=1, min_count=1)
        df_out = df_base[["Instituicao", "Periodo"]].copy()
        df_out["Label"] = entry["label"]
        df_out["valor"] = valores
        return df_out

    def formatar_valor_br(valor, decimais=0):
        if pd.isna(valor) or valor is None:
            return "—"
        try:
            if decimais == 0:
                return f"{valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{valor:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return "—"

    def formatar_percentual(valor, decimais=1):
        if pd.isna(valor) or valor is None:
            return "—"
        try:
            return f"{valor * 100:.{decimais}f}%"
        except Exception:
            return "—"

    def render_table_like_carteira_4966(df_linhas: pd.DataFrame, entradas: list, periodos: list, formato_por_label: dict, tooltip_por_label: dict):
        html_tabela = """
        <style>
        .carteira-table {
            width: 88%;
            margin: 10px auto 0 auto;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: auto;
        }
        .carteira-table th, .carteira-table td {
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: right;
            vertical-align: top;
        }
        .carteira-table th {
            background-color: #f5f5f5;
            font-weight: 600;
        }
        .carteira-table td:first-child {
            text-align: left;
            font-weight: 500;
            white-space: nowrap;
            width: 1%;
            padding-right: 8px;
        }
        .carteira-table thead tr:first-child th {
            background-color: #4a4a4a;
            color: white;
            text-align: center;
        }
        .carteira-table thead tr:nth-child(2) th {
            background-color: #6a6a6a;
            color: white;
        }
        .dre-info {
            font-size: 12px;
            color: #666;
            margin-left: 6px;
            cursor: help;
        }
        .dre-subitem {
            font-size: 12px;
        }
        .dre-subitem td:first-child {
            padding-left: 18px;
        }
        .dre-negative {
            color: #7a1e2b;
            font-weight: 600;
        }
        </style>
        <table class="carteira-table">
        <thead>
        <tr>
        <th rowspan="2">Item</th>
        """

        for periodo in periodos:
            html_tabela += f'<th colspan="2">{periodo}</th>'
        html_tabela += "</tr><tr>"
        for _ in periodos:
            html_tabela += "<th>YTD</th><th>Δ% YoY</th>"
        html_tabela += "</tr></thead><tbody>"

        for entry in entradas:
            label = entry["label"]
            label_exib = entry.get("label_exib", label)
            tooltip = tooltip_por_label.get(label, "")
            info_html = f'<span class="dre-info" title="{tooltip}">ⓘ</span>' if tooltip else ""
            row_class = "dre-subitem" if entry.get("is_child") else ""
            html_tabela += f"<tr class=\"{row_class}\"><td>{label_exib} {info_html}</td>"
            linha = df_linhas[df_linhas["Label"] == label]
            for periodo in periodos:
                cell = linha[linha["PeriodoExib"] == periodo]
                if not cell.empty:
                    ytd_val = cell["ytd"].iloc[0]
                    yoy_val = cell["yoy"].iloc[0]
                else:
                    ytd_val = pd.NA
                    yoy_val = pd.NA
                formato = formato_por_label.get(label, "num")
                if formato == "pct":
                    ytd_fmt = formatar_percentual(ytd_val, decimais=2)
                    ytd_neg = False
                else:
                    ytd_fmt = formatar_valor_br(ytd_val)
                    ytd_neg = pd.notna(ytd_val) and ytd_val < 0
                yoy_fmt = formatar_percentual(yoy_val, decimais=1)
                yoy_neg = pd.notna(yoy_val) and yoy_val < 0
                ytd_span = f"<span class=\"dre-negative\">{ytd_fmt}</span>" if ytd_neg else ytd_fmt
                yoy_span = f"<span class=\"dre-negative\">{yoy_fmt}</span>" if yoy_neg else yoy_fmt
                html_tabela += f"<td>{ytd_span}</td><td>{yoy_span}</td>"
            html_tabela += "</tr>"

        html_tabela += "</tbody></table>"
        st.markdown(html_tabela, unsafe_allow_html=True)

    df_dre, dre_msg = load_dre_data()
    if df_dre is None or df_dre.empty:
        detalhe = f" ({dre_msg})" if dre_msg else ""
        st.warning(f"Dados DRE não disponíveis no cache. Atualize a base no menu 'Atualizar Base'.{detalhe}")
        with st.expander("Limites de recursos do Streamlit Community Cloud"):
            st.markdown(
                """
                Os limites atuais (fev/2024) são aproximadamente:
                - CPU: 0,078 cores mínimo, 2 cores máximo
                - Memória: 690MB mínimo, 2,7GB máximo
                - Armazenamento: sem mínimo, 50GB máximo

                Sintomas comuns de limite excedido:
                - App lento (throttling)
                - Mensagem "🤯 This app has gone over its resource limits."
                - Mensagem "😦 Oh no."

                Apps educacionais, open-source ou de impacto social podem solicitar aumento de recursos.
                """
            )
    else:
        col_periodo, col_inst = detectar_colunas_basicas(df_dre)
        if col_periodo is None:
            st.warning("Coluna de período não encontrada nos dados DRE.")
        else:
            mapping_entries = load_dre_mapping()

            if not mapping_entries:
                st.warning("Mapeamento DRE não encontrado ou vazio.")
            else:
                colunas_necessarias = {col_periodo}
                if col_inst:
                    colunas_necessarias.add(col_inst)
                for entry in mapping_entries:
                    for fonte in entry.get("sources", []):
                        col_encontrada = find_column(df_dre, fonte)
                        if col_encontrada:
                            colunas_necessarias.add(col_encontrada)
                colunas_necessarias = [c for c in df_dre.columns if c in colunas_necessarias]
                if colunas_necessarias:
                    df_dre = df_dre[colunas_necessarias].copy()

                df_base = df_dre.copy()
                if col_inst is None:
                    df_base["Instituicao"] = df_base.get("CodInst", "Instituição")
                else:
                    df_base = df_base.rename(columns={col_inst: "Instituicao"})
                df_base = df_base.rename(columns={col_periodo: "Periodo"})

                df_base[["ano", "mes"]] = df_base["Periodo"].apply(
                    lambda x: pd.Series(parse_periodo(x))
                )
                df_new = df_base[
                    (df_base["ano"].fillna(0) > 2025)
                    | ((df_base["ano"] == 2025) & (df_base["mes"].fillna(0) >= 3))
                ].copy()

                instit_col = "Instituicao"
                _dict_aliases_dre = st.session_state.get('dict_aliases', {})
                instituicoes = ordenar_bancos_com_alias(
                    df_base[instit_col].dropna().unique().tolist(), _dict_aliases_dre
                )
                anos_disponiveis = sorted(df_base["ano"].dropna().unique().astype(int).tolist())

                if not instituicoes or not anos_disponiveis:
                    st.warning("Dados DRE sem instituições ou períodos válidos.")
                    st.stop()

                _default_dre = _encontrar_bancos_default(instituicoes, [("itau", "itaú")])
                _idx_dre = instituicoes.index(_default_dre[0]) if _default_dre else 0

                col_inst, col_ano = st.columns([1, 1])
                with col_inst:
                    instituicao_selecionada = st.selectbox(
                        "Instituição",
                        instituicoes,
                        index=_idx_dre,
                        key="dre_instituicao"
                    )
                with col_ano:
                    ano_selecionado = st.selectbox(
                        "Ano",
                        anos_disponiveis[::-1],
                        index=0,
                        key="dre_ano"
                    )

                df_values = []
                for entry in mapping_entries:
                    df_entry = compute_line_values(df_new, entry)
                    if not df_entry.empty:
                        df_values.append(df_entry)

                if not df_values:
                    st.warning("Nenhuma linha DRE foi encontrada com o mapeamento atual.")
                else:
                    df_valores = pd.concat(df_values, ignore_index=True)
                    df_ytd = compute_ytd_irregular(df_valores)
                    df_ytd = compute_yoy(df_ytd)

                    df_ytd["PeriodoExib"] = df_ytd["Periodo"].apply(periodo_para_exibicao)

                    formato_por_label = {entry["label"]: entry.get("format", "num") for entry in mapping_entries}
                    tooltip_por_label = {}
                    entradas_com_label = []
                    for entry in mapping_entries:
                        fonte_original = entry.get("original_label")
                        fontes = [fonte_original] if fonte_original else entry.get("sources", [])
                        fontes_fmt = ", ".join([f for f in fontes if f])
                        tooltip_parts = []
                        if entry.get("concept"):
                            tooltip_parts.append(entry["concept"])
                        if entry.get("derived_metric"):
                            formula = DERIVED_METRICS_FORMULAS.get(entry["label"])
                            if formula:
                                tooltip_parts.append(f"Fórmula: {formula}")
                        if fontes_fmt:
                            tooltip_parts.append(f"Fontes: {fontes_fmt}")
                        if entry.get("ytd_note"):
                            tooltip_parts.append("Nota YTD: set/dez = jun + período.")
                        tooltip_por_label[entry["label"]] = " | ".join(tooltip_parts)

                        label_exib = entry["label"]
                        entrada_copy = entry.copy()
                        entrada_copy["label_exib"] = label_exib
                        entradas_com_label.append(entrada_copy)

                    df_filtrado = df_ytd[
                        (df_ytd["Instituicao"] == instituicao_selecionada)
                        & (df_ytd["ano"] == int(ano_selecionado))
                    ].copy()

                    diag_info = {}
                    if st.session_state.get("modo_diagnostico"):
                        diag_info["df_base_mb"] = _df_mem_mb(df_base)
                        diag_info["df_valores_mb"] = _df_mem_mb(df_valores)
                        diag_info["df_ytd_mb"] = _df_mem_mb(df_ytd)

                    _perf_start("dre_derived_load")
                    df_derived_slice = carregar_metricas_derivadas_slice(
                        instituicoes=[instituicao_selecionada],
                        metricas=DERIVED_METRICS,
                    )
                    tempo_derived = _perf_end("dre_derived_load")

                    if not df_derived_slice.empty:
                        df_derived_slice = df_derived_slice.rename(
                            columns={"Métrica": "Label", "Valor": "valor", "Instituição": "Instituicao", "Período": "Periodo"}
                        )
                        df_derived_slice["Periodo"] = df_derived_slice["Periodo"].astype(str)
                        df_derived_slice[["ano", "mes"]] = df_derived_slice["Periodo"].apply(
                            lambda x: pd.Series(parse_periodo(x))
                        )
                        df_derived_slice = compute_ytd_irregular(df_derived_slice)
                        df_derived_slice = compute_yoy(df_derived_slice)
                        df_derived_slice["PeriodoExib"] = df_derived_slice["Periodo"].apply(periodo_para_exibicao)
                        df_derived_filtrado = df_derived_slice[
                            (df_derived_slice["Instituicao"] == instituicao_selecionada)
                            & (df_derived_slice["ano"] == int(ano_selecionado))
                        ].copy()
                        df_filtrado = pd.concat([df_filtrado, df_derived_filtrado], ignore_index=True)

                    if st.session_state.get("modo_diagnostico"):
                        diag_info["derived_slice_mb"] = _df_mem_mb(df_derived_slice)
                        diag_info["derived_slice_rows"] = len(df_derived_slice)
                        diag_info["derived_load_s"] = round(tempo_derived, 3)

                    periodos_ordem = ["Set", "Jun", "Mar", "Dez"]
                    periodos_disponiveis = []
                    for mes in [9, 6, 3, 12]:
                        periodo_texto = periodo_para_exibicao(f"{int(mes/3)}/{ano_selecionado}")
                        periodos_disponiveis.append(periodo_texto)

                    render_table_like_carteira_4966(
                        df_filtrado,
                        entradas_com_label,
                        periodos_disponiveis,
                        formato_por_label,
                        tooltip_por_label
                    )

                    if st.session_state.get("modo_diagnostico"):
                        with st.expander("diagnóstico DRE"):
                            st.caption(f"Memória df_base: {diag_info.get('df_base_mb', 0):.2f} MB")
                            st.caption(f"Memória df_valores: {diag_info.get('df_valores_mb', 0):.2f} MB")
                            st.caption(f"Memória df_ytd: {diag_info.get('df_ytd_mb', 0):.2f} MB")
                            st.caption(f"Memória recorte derivado: {diag_info.get('derived_slice_mb', 0):.2f} MB")
                            st.caption(f"Linhas recorte derivado: {diag_info.get('derived_slice_rows', 0)}")
                            st.caption(f"Tempo recorte derivado: {diag_info.get('derived_load_s', 0):.3f}s")

                    st.markdown("#### Exportar DRE (formato simples)")
                    df_export = []
                    for entry in entradas_com_label:
                        label = entry["label"]
                        linha = {"Item": label}
                        for periodo in periodos_disponiveis:
                            cell = df_filtrado[
                                (df_filtrado["Label"] == label)
                                & (df_filtrado["PeriodoExib"] == periodo)
                            ]
                            if not cell.empty:
                                ytd_val = cell["ytd"].iloc[0]
                                yoy_val = cell["yoy"].iloc[0]
                            else:
                                ytd_val = pd.NA
                                yoy_val = pd.NA
                            linha[f"{periodo} YTD"] = ytd_val
                            linha[f"{periodo} Δ% YoY"] = yoy_val
                        df_export.append(linha)

                    df_export = pd.DataFrame(df_export)
                    buffer_excel = BytesIO()
                    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                        df_export.to_excel(writer, index=False, sheet_name="DRE")
                        worksheet = writer.sheets["DRE"]
                        for idx, col in enumerate(df_export.columns):
                            max_len = max(
                                len(str(col)),
                                df_export[col].astype(str).map(len).max()
                            )
                            worksheet.set_column(idx, idx, min(max_len + 2, 40))
                    buffer_excel.seek(0)

                    st.download_button(
                        label="Baixar Excel (DRE)",
                        data=buffer_excel,
                        file_name=f"DRE_{instituicao_selecionada.replace(' ', '_')}_{ano_selecionado}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dre_download_excel"
                    )

elif menu == "Carteira 4.966":
    # =========================================================================
    # ABA CARTEIRA 4.966 - Classificação de Instrumentos Financeiros (Res. 4.966)
    # =========================================================================

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_carteira_4966_data():
        """Carrega dados da Carteira 4.966 (Relatório 16) com cache."""
        from utils.ifdata_cache import get_manager
        manager = get_manager()
        resultado = manager.carregar("carteira_instrumentos")
        if resultado.sucesso and resultado.dados is not None:
            return resultado.dados
        return None

    def periodo_para_exibicao_mes(periodo_trimestre: str) -> str:
        """Converte período trimestral (1/2025) para formato mês abreviado (mar/25)."""
        if not periodo_trimestre or '/' not in periodo_trimestre:
            return periodo_trimestre
        try:
            trimestre, ano = periodo_trimestre.split('/')
            meses_map = {'1': 'mar', '2': 'jun', '3': 'set', '4': 'dez'}
            mes = meses_map.get(trimestre, trimestre)
            ano_curto = ano[-2:] if len(ano) == 4 else ano
            return f"{mes}/{ano_curto}"
        except:
            return periodo_trimestre

    def formatar_valor_br(valor, decimais=0):
        """Formata valor numérico no padrão brasileiro (pontos como separador de milhar)."""
        if pd.isna(valor) or valor is None:
            return "-"
        try:
            if decimais == 0:
                return f"{valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"{valor:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def formatar_percentual(valor, decimais=1):
        """Formata percentual com casas decimais."""
        if pd.isna(valor) or valor is None:
            return "-"
        try:
            return f"{valor * 100:.{decimais}f}%"
        except:
            return "-"

    # Mapeamento das linhas da tabela para colunas da API
    LINHAS_CARTEIRA_4966 = [
        ("C1", "C1"),
        ("C2", "C2"),
        ("C3", "C3"),
        ("C4", "C4"),
        ("C5", "C5"),
        ("Carteira Total", "Total Geral"),
        ("Carteira Não Informada", "Carteira não Informada ou não se Aplica"),
        ("Carteira no Exterior", "Total Exterior"),
        ("Total não Individualizado", "Total não Individualizado"),
    ]

    df_carteira = load_carteira_4966_data()

    if df_carteira is not None and not df_carteira.empty:
        # Verificar colunas disponíveis
        colunas_disponiveis = df_carteira.columns.tolist()

        # Obter lista de instituições e períodos
        if 'Instituição' in df_carteira.columns:
            _dict_aliases_cart = st.session_state.get('dict_aliases', {})
            instituicoes = ordenar_bancos_com_alias(
                df_carteira['Instituição'].dropna().unique().tolist(), _dict_aliases_cart
            )
        else:
            instituicoes = []

        col_periodo = 'Período' if 'Período' in df_carteira.columns else 'Periodo'
        if col_periodo in df_carteira.columns:
            periodos_disponiveis = ordenar_periodos(df_carteira[col_periodo].dropna().unique(), reverso=True)
        else:
            periodos_disponiveis = []

        if instituicoes and periodos_disponiveis:
            st.markdown("### Carteira de Crédito por Instrumentos Financeiros (Res. 4.966)")
            st.caption("Classificação conforme estágios de risco de crédito - C1 a C5")

            # Seletores
            col_inst, col_periodos = st.columns([1, 2])

            with col_inst:
                instituicao_selecionada = st.selectbox(
                    "Instituição",
                    instituicoes,
                    key="carteira_4966_instituicao"
                )

            # Filtrar períodos disponíveis para a instituição selecionada
            df_inst = df_carteira[df_carteira['Instituição'] == instituicao_selecionada]
            periodos_inst = ordenar_periodos(df_inst[col_periodo].dropna().unique(), reverso=True)

            with col_periodos:
                periodos_selecionados = st.multiselect(
                    "Períodos (selecione 2 ou mais para comparação)",
                    periodos_inst,
                    default=periodos_inst[:2] if len(periodos_inst) >= 2 else periodos_inst,
                    key="carteira_4966_periodos",
                    help="Selecione múltiplos períodos para comparar. O delta será calculado em relação ao período anterior na lista."
                )

            if periodos_selecionados and len(periodos_selecionados) >= 1:
                # Ordenar períodos do mais antigo para o mais recente (para cálculo de deltas)
                periodos_ordenados = ordenar_periodos(periodos_selecionados, reverso=False)

                # Construir dados da tabela
                dados_tabela = []

                linha_colunas = {}
                for nome_linha, coluna_api in LINHAS_CARTEIRA_4966:
                    linha_dados = {"Tipo de Carteira": nome_linha}

                    # Verificar se a coluna existe nos dados
                    coluna_encontrada = None
                    for col in colunas_disponiveis:
                        if col.lower().strip() == coluna_api.lower().strip():
                            coluna_encontrada = col
                            break

                    if coluna_encontrada is None:
                        # Tentar busca parcial
                        for col in colunas_disponiveis:
                            if coluna_api.lower() in col.lower():
                                coluna_encontrada = col
                                break
                    linha_colunas[nome_linha] = coluna_encontrada

                    for periodo in periodos_ordenados:
                        df_periodo = df_inst[df_inst[col_periodo] == periodo]

                        if not df_periodo.empty and coluna_encontrada and coluna_encontrada in df_periodo.columns:
                            valor = df_periodo[coluna_encontrada].iloc[0]
                        else:
                            valor = None

                        # Obter Total Geral para calcular percentual
                        total_geral = None
                        for col in colunas_disponiveis:
                            if 'total geral' in col.lower():
                                if not df_periodo.empty and col in df_periodo.columns:
                                    total_geral = df_periodo[col].iloc[0]
                                break

                        periodo_exib = periodo_para_exibicao_mes(periodo)
                        linha_dados[f"{periodo_exib}"] = valor

                        # Calcular percentual
                        if valor is not None and total_geral is not None and total_geral != 0:
                            pct = valor / total_geral
                        else:
                            pct = None
                        linha_dados[f"{periodo_exib} %"] = pct

                    dados_tabela.append(linha_dados)

                df_resultado = pd.DataFrame(dados_tabela)

                # Criar tabela estilizada com HTML
                st.markdown("---")

                # Construir HTML da tabela
                html_tabela = """
                <style>
                .carteira-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                    margin-top: 10px;
                }
                .carteira-table th, .carteira-table td {
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: right;
                }
                .carteira-table th {
                    background-color: #f5f5f5;
                    font-weight: 600;
                }
                .carteira-table td:first-child {
                    text-align: left;
                    font-weight: 500;
                }
                .carteira-table tr.total-row {
                    background-color: #e8f4e8;
                    font-weight: bold;
                }
                .carteira-table tr.total-row td {
                    font-weight: bold;
                }
                .carteira-table thead tr:first-child th {
                    background-color: #4a4a4a;
                    color: white;
                    text-align: center;
                }
                .carteira-table thead tr:nth-child(2) th {
                    background-color: #6a6a6a;
                    color: white;
                }
                .delta-pos { color: #28a745; }
                .delta-neg { color: #dc3545; }
                </style>
                <table class="carteira-table">
                <thead>
                <tr>
                <th rowspan="2">Tipo de Carteira</th>
                """

                # Cabeçalhos agrupados por período
                for periodo in periodos_ordenados:
                    periodo_exib = periodo_para_exibicao_mes(periodo)
                    html_tabela += f'<th colspan="2">{periodo_exib}</th>'

                html_tabela += "</tr><tr>"

                for _ in periodos_ordenados:
                    html_tabela += '<th>Valor</th><th>% Carteira Total</th>'

                html_tabela += "</tr></thead><tbody>"

                # Linhas de dados
                for idx, row in df_resultado.iterrows():
                    tipo = row["Tipo de Carteira"]
                    is_total = tipo == "Carteira Total"
                    row_class = 'class="total-row"' if is_total else ''

                    html_tabela += f"<tr {row_class}><td>{tipo}</td>"

                    for i, periodo in enumerate(periodos_ordenados):
                        periodo_exib = periodo_para_exibicao_mes(periodo)
                        valor = row.get(f"{periodo_exib}")
                        pct = row.get(f"{periodo_exib} %")

                        valor_fmt = formatar_valor_br(valor)
                        pct_fmt = formatar_percentual(pct) if pct is not None else "-"

                        # Se for linhas especiais (Não Informada, Exterior, Não Individualizado), não mostrar percentual
                        if tipo in ["Carteira Não Informada", "Carteira no Exterior", "Total não Individualizado"]:
                            pct_fmt = "-"

                        # Adicionar delta visual se houver período anterior
                        delta_html = ""
                        periodo_base = _periodo_ano_anterior(periodo)
                        coluna_base = linha_colunas.get(tipo)
                        valor_base = None
                        if periodo_base and coluna_base:
                            df_base = df_inst[df_inst[col_periodo] == periodo_base]
                            if not df_base.empty and coluna_base in df_base.columns:
                                valor_base = df_base[coluna_base].iloc[0]
                        if valor_base is not None and valor is not None:
                            delta = valor - valor_base
                            if delta > 0:
                                delta_html = f' <span class="delta-pos">▲</span>'
                            elif delta < 0:
                                delta_html = f' <span class="delta-neg">▼</span>'

                        html_tabela += f"<td>{valor_fmt}{delta_html}</td><td>{pct_fmt}</td>"
                    html_tabela += "</tr>"

                html_tabela += "</tbody></table>"

                st.markdown(html_tabela, unsafe_allow_html=True)

                # Tabela de auditoria (dataframe simples)
                with st.expander("Dados para auditoria"):
                    st.dataframe(df_resultado, width='stretch')

                # Exportação Excel
                st.markdown("---")

                def criar_excel_carteira_4966(df, periodos):
                    """Cria arquivo Excel com layout similar ao visual da tabela."""
                    import io
                    buffer = io.BytesIO()
                    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
                    worksheet = workbook.add_worksheet("Carteira 4.966")

                    n_cols = 1 + len(periodos) * 2
                    border = {"border": 1, "border_color": "#dddddd"}
                    header_fmt = workbook.add_format(
                        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#4a4a4a", "font_color": "white", **border}
                    )
                    subheader_fmt = workbook.add_format(
                        {"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#6a6a6a", "font_color": "white", **border}
                    )
                    row_even = workbook.add_format({"align": "right", "valign": "vcenter", "bg_color": "#f8f9fa", **border})
                    row_odd = workbook.add_format({"align": "right", "valign": "vcenter", "bg_color": "#ffffff", **border})
                    row_even_label = workbook.add_format({"align": "left", "valign": "vcenter", "bg_color": "#f8f9fa", **border})
                    row_odd_label = workbook.add_format({"align": "left", "valign": "vcenter", "bg_color": "#ffffff", **border})
                    total_row = workbook.add_format(
                        {"align": "right", "valign": "vcenter", "bg_color": "#e8f4e8", "bold": True, **border}
                    )
                    total_row_label = workbook.add_format(
                        {"align": "left", "valign": "vcenter", "bg_color": "#e8f4e8", "bold": True, **border}
                    )

                    worksheet.set_column(0, 0, 30)
                    worksheet.set_column(1, max(1, n_cols - 1), 16)

                    row_idx = 0
                    worksheet.write(row_idx, 0, "Tipo de Carteira", header_fmt)
                    col_idx = 1
                    for periodo in periodos:
                        periodo_exib = periodo_para_exibicao_mes(periodo)
                        worksheet.merge_range(row_idx, col_idx, row_idx, col_idx + 1, periodo_exib, header_fmt)
                        col_idx += 2
                    row_idx += 1

                    worksheet.write(row_idx, 0, "", subheader_fmt)
                    col_idx = 1
                    for _ in periodos:
                        worksheet.write(row_idx, col_idx, "Valor", subheader_fmt)
                        worksheet.write(row_idx, col_idx + 1, "% Carteira Total", subheader_fmt)
                        col_idx += 2
                    row_idx += 1

                    zebra_idx = 0
                    for _, row in df.iterrows():
                        tipo = row["Tipo de Carteira"]
                        is_total = tipo == "Carteira Total"
                        is_even = zebra_idx % 2 == 0
                        label_fmt = total_row_label if is_total else (row_even_label if is_even else row_odd_label)
                        cell_fmt = total_row if is_total else (row_even if is_even else row_odd)

                        worksheet.write(row_idx, 0, tipo, label_fmt)
                        col_idx = 1
                        for periodo in periodos:
                            periodo_exib = periodo_para_exibicao_mes(periodo)
                            valor = row.get(f"{periodo_exib}")
                            pct = row.get(f"{periodo_exib} %")
                            valor_fmt = formatar_valor_br(valor)
                            if tipo in ["Carteira Não Informada", "Carteira no Exterior", "Total não Individualizado"]:
                                pct_fmt = "-"
                            else:
                                pct_fmt = formatar_percentual(pct) if pct is not None else "-"
                            worksheet.write(row_idx, col_idx, valor_fmt, cell_fmt)
                            worksheet.write(row_idx, col_idx + 1, pct_fmt, cell_fmt)
                            col_idx += 2
                        row_idx += 1
                        zebra_idx += 1

                    workbook.close()
                    buffer.seek(0)
                    return buffer.getvalue()

                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    excel_data = criar_excel_carteira_4966(df_resultado, periodos_ordenados)
                    periodos_str = "_".join([periodo_para_exibicao_mes(p).replace("/", "-") for p in periodos_ordenados])
                    st.download_button(
                        label="Baixar Excel (Tabela Atual)",
                        data=excel_data,
                        file_name=f"Carteira_4966_{instituicao_selecionada.replace(' ', '_')[:30]}_{periodos_str}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_carteira_4966"
                    )

                with col_btn2:
                    # Exportar dados brutos da instituição
                    df_inst_export = df_inst.copy()
                    buffer_raw = io.BytesIO()
                    with pd.ExcelWriter(buffer_raw, engine='openpyxl') as writer:
                        df_inst_export.to_excel(writer, index=False, sheet_name='Dados Brutos')
                    buffer_raw.seek(0)

                    st.download_button(
                        label="Baixar Excel (Dados Brutos)",
                        data=buffer_raw.getvalue(),
                        file_name=f"Carteira_4966_{instituicao_selecionada.replace(' ', '_')[:30]}_dados_brutos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_carteira_4966_raw"
                    )

            else:
                st.info("Selecione ao menos um período para visualizar os dados.")

        else:
            st.warning("Dados de Carteira 4.966 não contêm instituições ou períodos válidos.")
            st.caption(f"Colunas disponíveis: {colunas_disponiveis[:10]}...")

    else:
        st.warning("Dados da Carteira 4.966 (Relatório 16) não disponíveis.")
        st.info("Os dados precisam ser extraídos via 'Atualização Base' no painel de administração.")

        # Mostrar informações sobre o cache
        from utils.ifdata_cache import get_manager
        manager = get_manager()
        info = manager.info("carteira_instrumentos")
        if info and not info.get("erro"):
            st.caption(f"Status do cache: {info}")

elif menu == "Taxas de Juros por Produto":
    # =========================================================================
    # ABA TAXAS DE JUROS POR PRODUTO - Extração direta da API do BCB
    # Dropdowns cascateados: Segmento → Produto → Bancos
    # =========================================================================
    import requests

    # URL da API do BCB
    API_TAXAS_URL = "https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata/TaxasJurosDiariaPorInicioPeriodo"

    def formatar_modalidade(nome: str) -> str:
        """Formata nome da modalidade para exibição."""
        if not nome:
            return nome
        preposicoes = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com', 'sem', 'por'}
        palavras = nome.lower().split()
        resultado = []
        for i, palavra in enumerate(palavras):
            if i == 0 or palavra not in preposicoes:
                resultado.append(palavra.capitalize())
            else:
                resultado.append(palavra)
        return ' '.join(resultado)

    @st.cache_data(ttl=3600, show_spinner="Buscando dados do BCB (12 meses)...")
    def buscar_taxas_bcb_historico() -> pd.DataFrame:
        """Busca dados de taxas de juros dos últimos 12 meses da API do BCB.
        Retorna apenas a última data de cada mês para otimizar performance."""
        from datetime import date

        # Data de 13 meses atrás (para garantir 12 meses completos)
        data_inicio = (date.today() - timedelta(days=400)).strftime('%Y-%m-%d')

        # Tentar diferentes formatos de URL
        urls_tentar = [
            f"{API_TAXAS_URL}?$format=json&$top=500000",
            f"{API_TAXAS_URL}(dataInicioPeriodo=@dataInicioPeriodo)?$format=json&$top=500000&@dataInicioPeriodo='{data_inicio}'",
        ]

        response = None
        for url in urls_tentar:
            try:
                response = requests.get(url, timeout=120)
                if response.status_code == 200:
                    break
            except:
                continue

        if response is None or response.status_code != 200:
            return pd.DataFrame()

        try:
            dados = response.json()

            if 'value' not in dados or not dados['value']:
                return pd.DataFrame()

            df = pd.DataFrame(dados['value'])

            # Renomear colunas
            df = df.rename(columns={
                'InicioPeriodo': 'Início Período',
                'FimPeriodo': 'Fim Período',
                'Segmento': 'Segmento',
                'Modalidade': 'Produto',
                'Posicao': 'Posição',
                'InstituicaoFinanceira': 'Instituição Financeira',
                'TaxaJurosAoMes': 'Taxa Mensal (%)',
                'TaxaJurosAoAno': 'Taxa Anual (%)'
            })

            # Converter datas
            df['Fim Período'] = pd.to_datetime(df['Fim Período'])

            # Filtrar últimos 12 meses
            data_limite = pd.Timestamp.today() - pd.DateOffset(months=12)
            df = df[df['Fim Período'] >= data_limite]

            # Criar coluna Ano-Mês para agrupar
            df['AnoMes'] = df['Fim Período'].dt.to_period('M')

            # Para cada combinação (Segmento, Produto, Instituição, AnoMes), pegar a última data
            idx = df.groupby(['Segmento', 'Produto', 'Instituição Financeira', 'AnoMes'])['Fim Período'].idxmax()
            df_mensal = df.loc[idx].copy()

            # Criar coluna de data formatada para o gráfico
            df_mensal['Mês'] = df_mensal['Fim Período'].dt.strftime('%b/%y')

            return df_mensal

        except Exception as e:
            return pd.DataFrame()

    st.markdown("### Taxas de Juros por Produto")
    st.caption("Histórico dos últimos 12 meses - API do Banco Central do Brasil")

    with st.expander("ℹ️ Sobre os dados", expanded=False):
        st.markdown("""
        **Fonte:** API do Banco Central do Brasil - Taxas de Juros

        **Histórico:** Última data disponível de cada mês (últimos 12 meses).

        **Posição:** Ranking da instituição para aquele produto/período.
        Posição 1 = menor taxa (melhor para o cliente).
        """)

    st.markdown("---")

    # =============================================================
    # BUSCAR DADOS HISTÓRICOS (12 MESES)
    # =============================================================
    df_taxas = buscar_taxas_bcb_historico()

    if df_taxas.empty:
        st.warning("Nenhum dado encontrado. A API do BCB pode estar indisponível.")
        st.info("Tente novamente em alguns minutos.")
    else:
        # Data mais recente disponível
        data_mais_recente = df_taxas['Fim Período'].max()
        meses_disponiveis = df_taxas['AnoMes'].nunique()

        st.success(f"✅ {len(df_taxas):,} registros | {meses_disponiveis} meses | Até: {data_mais_recente.strftime('%d/%m/%Y')}")

        st.markdown("---")

        # Usar dados da data mais recente para seleção de segmento/produto
        df_recente = df_taxas[df_taxas['Fim Período'] == data_mais_recente]

        # =============================================================
        # DROPDOWN 1: SEGMENTO (PF / PJ)
        # =============================================================
        segmentos = sorted(df_recente['Segmento'].dropna().unique().tolist())

        segmento_sel = st.selectbox(
            "1️⃣ Selecione o Segmento",
            options=segmentos,
            key="tj_segmento",
            help="PF = Pessoa Física, PJ = Pessoa Jurídica"
        )

        if segmento_sel:
            df_seg_recente = df_recente[df_recente['Segmento'] == segmento_sel]

            # =============================================================
            # DROPDOWN 2: PRODUTO (filtrado pelo segmento)
            # =============================================================
            produtos = sorted(df_seg_recente['Produto'].dropna().unique().tolist())

            produto_sel = st.selectbox(
                "2️⃣ Selecione o Produto",
                options=produtos,
                format_func=formatar_modalidade,
                key="tj_produto"
            )

            if produto_sel:
                # Filtrar dados históricos para o segmento e produto
                df_prod_hist = df_taxas[
                    (df_taxas['Segmento'] == segmento_sel) &
                    (df_taxas['Produto'] == produto_sel)
                ]

                st.markdown(f"#### {formatar_modalidade(produto_sel)}")

                # =============================================================
                # MULTISELECT 3: BANCOS (baseado na data mais recente)
                # =============================================================
                df_prod_recente = df_prod_hist[df_prod_hist['Fim Período'] == data_mais_recente]

                # Ordenar por posição (menor = melhor taxa)
                if 'Posição' in df_prod_recente.columns and not df_prod_recente.empty:
                    df_prod_ord = df_prod_recente.sort_values('Posição', ascending=True)
                else:
                    df_prod_ord = df_prod_recente.sort_values('Instituição Financeira')

                bancos_disponiveis = df_prod_ord['Instituição Financeira'].unique().tolist()

                # Top 10 como default
                top_10 = bancos_disponiveis[:10]

                # Ordenar lista com aliases primeiro
                dict_aliases = st.session_state.get('dict_aliases', {})
                bancos_ordenados = ordenar_bancos_com_alias(bancos_disponiveis, dict_aliases)

                bancos_sel = st.multiselect(
                    "3️⃣ Selecione os Bancos (máx 15)",
                    options=bancos_ordenados,
                    default=[b for b in top_10 if b in bancos_ordenados][:10],
                    max_selections=15,
                    key="tj_bancos",
                    help="Top 10 por menor taxa pré-selecionados"
                )

                if not bancos_sel:
                    st.warning("Selecione ao menos um banco.")
                else:
                    # =============================================================
                    # TIPO DE TAXA
                    # =============================================================
                    tipo_taxa = st.radio(
                        "Tipo de taxa",
                        ["Taxa Mensal (%)", "Taxa Anual (%)"],
                        horizontal=True,
                        key="tj_tipo_taxa"
                    )

                    # Filtrar dados históricos para os bancos selecionados
                    df_chart = df_prod_hist[df_prod_hist['Instituição Financeira'].isin(bancos_sel)].copy()
                    df_chart = df_chart.sort_values('Fim Período')

                    # =============================================================
                    # GRÁFICO DE LINHAS - HISTÓRICO 12 MESES
                    # =============================================================
                    fig = px.line(
                        df_chart,
                        x='Fim Período',
                        y=tipo_taxa,
                        color='Instituição Financeira',
                        title=f'{formatar_modalidade(produto_sel)} - {tipo_taxa} (Últimos 12 meses)',
                        labels={
                            'Fim Período': 'Data',
                            tipo_taxa: tipo_taxa,
                            'Instituição Financeira': 'Instituição'
                        },
                        template='plotly_white',
                        markers=True
                    )

                    fig.update_layout(
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.35,
                            xanchor="center",
                            x=0.5
                        ),
                        xaxis_title="",
                        yaxis_title=tipo_taxa,
                        hovermode='x unified',
                        margin=dict(b=100)
                    )

                    fig.update_xaxes(tickformat="%b/%y")
                    fig.update_traces(marker=dict(size=6))

                    st.plotly_chart(fig, width='stretch')

                    # =============================================================
                    # TABELA DE DADOS (data mais recente)
                    # =============================================================
                    with st.expander("📋 Ver ranking atual"):
                        df_rank = df_prod_recente[df_prod_recente['Instituição Financeira'].isin(bancos_sel)]
                        cols_mostrar = ['Posição', 'Instituição Financeira', 'Taxa Mensal (%)', 'Taxa Anual (%)']
                        cols_disp = [c for c in cols_mostrar if c in df_rank.columns]
                        df_display = df_rank[cols_disp].sort_values('Posição' if 'Posição' in cols_disp else tipo_taxa)
                        st.caption(f"Ranking em {data_mais_recente.strftime('%d/%m/%Y')}:")
                        st.dataframe(df_display, width='stretch', hide_index=True)

                    # =============================================================
                    # EXPORTAÇÃO
                    # =============================================================
                    with st.expander("📥 Exportar dados"):
                        csv_data = df_chart.to_csv(index=False, sep=';', decimal=',')
                        st.download_button(
                            label="⬇️ Baixar CSV (histórico)",
                            data=csv_data,
                            file_name=f"taxas_hist_{segmento_sel}_{produto_sel[:20]}.csv",
                            mime="text/csv",
                            key="tj_download_csv"
                        )

elif menu == "Crie sua métrica!":
    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
        df = get_dados_concatenados()  # OTIMIZAÇÃO: usar cache

        colunas_numericas = [col for col in df.columns if col not in ['Instituição', 'Período'] and df[col].dtype in ['float64', 'int64']]
        periodos_disponiveis = ordenar_periodos(df['Período'].dropna().unique())
        periodos_dropdown = ordenar_periodos(df['Período'].dropna().unique(), reverso=True)

        # Lista de todos os bancos disponíveis com ordenação por alias
        bancos_todos = df['Instituição'].dropna().unique().tolist()
        dict_aliases = st.session_state.get('dict_aliases', {})
        todos_bancos = ordenar_bancos_com_alias(bancos_todos, dict_aliases)

        st.markdown("### construtor de métricas derivadas")
        st.caption("monte uma métrica personalizada combinando variáveis com operações matemáticas")

        # ===== CONSTRUTOR DE FÓRMULA PASSO-A-PASSO =====
        # Inicializa session state para a fórmula
        if 'brincar_formula_steps' not in st.session_state:
            st.session_state['brincar_formula_steps'] = []
        if 'brincar_nome_metrica' not in st.session_state:
            st.session_state['brincar_nome_metrica'] = "Métrica Personalizada"

        col_nome, col_formato = st.columns([2, 1])
        with col_nome:
            nome_metrica = st.text_input(
                "nome da métrica",
                value=st.session_state['brincar_nome_metrica'],
                key="nome_metrica_input"
            )
            st.session_state['brincar_nome_metrica'] = nome_metrica

        with col_formato:
            formato_resultado = st.selectbox(
                "formato do resultado",
                ["Auto", "Valor bruto (R$)", "Percentual (%)", "Múltiplo (x)", "Número"],
                key="formato_resultado_brincar"
            )

        st.markdown("---")
        st.markdown("**construir fórmula**")

        # Interface de construção
        col_var, col_op, col_add = st.columns([2, 1, 1])

        with col_var:
            var_nova = st.selectbox(
                "variável",
                colunas_numericas,
                key="var_nova_brincar"
            )

        with col_op:
            operacao = st.selectbox(
                "operação seguinte",
                ["(fim)", "+", "-", "×", "÷"],
                key="operacao_brincar"
            )

        with col_add:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("adicionar", key="btn_add_step", width='stretch'):
                st.session_state['brincar_formula_steps'].append({
                    'variavel': var_nova,
                    'operacao': operacao if operacao != "(fim)" else None
                })
                st.rerun()

        # Exibir fórmula atual
        if st.session_state['brincar_formula_steps']:
            formula_texto = ""
            for i, step in enumerate(st.session_state['brincar_formula_steps']):
                formula_texto += f"**{step['variavel']}**"
                if step['operacao']:
                    formula_texto += f" {step['operacao']} "

            col_formula, col_limpar = st.columns([3, 1])
            with col_formula:
                st.markdown(f"Fórmula: {formula_texto}")
            with col_limpar:
                if st.button("limpar fórmula", key="btn_limpar_formula"):
                    st.session_state['brincar_formula_steps'] = []
                    st.rerun()
        else:
            st.info("adicione variáveis para construir sua fórmula")

        # Função para calcular a métrica derivada
        def calcular_metrica_derivada(df_input, steps):
            if not steps:
                return None

            # Inicializa com a primeira variável
            resultado = df_input[steps[0]['variavel']].copy()

            for i in range(len(steps) - 1):
                op = steps[i]['operacao']
                proxima_var = steps[i + 1]['variavel']

                if op == '+':
                    resultado = resultado + df_input[proxima_var]
                elif op == '-':
                    resultado = resultado - df_input[proxima_var]
                elif op == '×':
                    resultado = resultado * df_input[proxima_var]
                elif op == '÷':
                    # Evita divisão por zero
                    resultado = resultado / df_input[proxima_var].replace(0, np.nan)

            return resultado

        # Função para detectar se é uma razão/divisão
        def formula_eh_divisao(steps):
            for step in steps:
                if step['operacao'] == '÷':
                    return True
            return False

        # ===== SELEÇÃO DE BANCOS =====
        st.markdown("---")
        st.markdown("**seleção de instituições**")

        col_modo, col_config = st.columns([1, 3])

        with col_modo:
            modo_selecao_brincar = st.radio(
                "modo de seleção",
                ["Top N", "Personalizado"],
                index=0,
                key="modo_selecao_brincar"
            )

        bancos_selecionados_brincar = []

        with col_config:
            if modo_selecao_brincar == "Top N":
                col_slider, col_var_ord = st.columns(2)
                with col_slider:
                    top_n_brincar = st.slider("quantidade de bancos", 5, 40, 15, key="top_n_brincar")
                with col_var_ord:
                    var_ordenacao_brincar = st.selectbox(
                        "ordenar por",
                        colunas_numericas,
                        index=colunas_numericas.index('Carteira de Crédito') if 'Carteira de Crédito' in colunas_numericas else 0,
                        key="var_ordenacao_brincar"
                    )
                # Obtém top N bancos do período mais recente
                periodo_mais_recente = periodos_disponiveis[-1]
                df_recente = df[df['Período'] == periodo_mais_recente].copy()
                df_recente_valid = df_recente.dropna(subset=[var_ordenacao_brincar])
                bancos_top_n = df_recente_valid.nlargest(top_n_brincar, var_ordenacao_brincar)['Instituição'].tolist()
                bancos_selecionados_brincar = bancos_top_n

            else:  # Personalizado
                bancos_custom_brincar = st.multiselect(
                    "adicionar/remover bancos",
                    todos_bancos,
                    max_selections=40,
                    key="bancos_custom_brincar"
                )
                bancos_selecionados_brincar = bancos_custom_brincar

        # ===== TIPO DE VISUALIZAÇÃO =====
        st.markdown("---")
        st.markdown("**visualização**")

        tipo_visualizacao = st.radio(
            "tipo de gráfico",
            ["Scatter Plot", "Deltas", "Ranking (barras)"],
            horizontal=True,
            key="tipo_viz_brincar"
        )

        # ===== CALCULAR E VISUALIZAR =====
        if st.session_state['brincar_formula_steps'] and bancos_selecionados_brincar:
            steps = st.session_state['brincar_formula_steps']

            # Verifica se a fórmula está completa (última operação é None ou não tem operação pendente)
            formula_completa = steps[-1]['operacao'] is None

            if not formula_completa:
                st.warning("adicione mais uma variável ou selecione '(fim)' como operação para completar a fórmula")
            else:
                # ===== SCATTER PLOT =====
                if tipo_visualizacao == "Scatter Plot":
                    st.markdown("---")
                    col_periodo, col_eixo_x, col_eixo_y, col_tamanho = st.columns(4)

                    with col_periodo:
                        periodo_scatter_brincar = st.selectbox(
                            "período",
                            periodos_dropdown,
                            index=0,
                            key="periodo_scatter_brincar",
                            format_func=periodo_para_exibicao
                        )

                    with col_eixo_x:
                        opcoes_eixo = ['Métrica Derivada'] + colunas_numericas
                        eixo_x_brincar = st.selectbox(
                            "eixo X",
                            opcoes_eixo,
                            index=0,
                            key="eixo_x_brincar"
                        )

                    with col_eixo_y:
                        eixo_y_brincar = st.selectbox(
                            "eixo Y",
                            opcoes_eixo,
                            index=min(1, len(opcoes_eixo) - 1),
                            key="eixo_y_brincar"
                        )

                    with col_tamanho:
                        opcoes_tamanho = ['Tamanho Fixo', 'Métrica Derivada'] + colunas_numericas
                        var_tamanho_brincar = st.selectbox(
                            "tamanho",
                            opcoes_tamanho,
                            index=0,
                            key="var_tamanho_brincar"
                        )

                    # Filtrar dados para o período
                    df_periodo = df[df['Período'] == periodo_scatter_brincar].copy()
                    df_scatter_brincar = df_periodo[df_periodo['Instituição'].isin(bancos_selecionados_brincar)].copy()

                    # Calcular métrica derivada
                    df_scatter_brincar['Métrica Derivada'] = calcular_metrica_derivada(df_scatter_brincar, steps)

                    # Remover linhas com NaN na métrica
                    df_scatter_brincar = df_scatter_brincar.dropna(subset=['Métrica Derivada'])

                    if len(df_scatter_brincar) > 0:
                        # Preparar dados para o gráfico
                        def get_format_for_var(var_name):
                            if var_name == 'Métrica Derivada':
                                if formato_resultado == "Percentual (%)":
                                    return {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
                                elif formato_resultado == "Valor bruto (R$)":
                                    return {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                                elif formato_resultado == "Múltiplo (x)":
                                    return {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                                elif formato_resultado == "Auto":
                                    if formula_eh_divisao(steps):
                                        return {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                                    else:
                                        return {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                                else:
                                    return {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}
                            else:
                                serie_ref = df_scatter_brincar[var_name] if var_name in df_scatter_brincar.columns else None
                                return get_axis_format(var_name, serie_ref)

                        format_x = get_format_for_var(eixo_x_brincar)
                        format_y = get_format_for_var(eixo_y_brincar)

                        df_scatter_brincar['x_display'] = df_scatter_brincar[eixo_x_brincar] * format_x['multiplicador']
                        df_scatter_brincar['y_display'] = df_scatter_brincar[eixo_y_brincar] * format_y['multiplicador']

                        if var_tamanho_brincar == 'Tamanho Fixo':
                            tamanho_constante = 25
                        else:
                            format_size = get_format_for_var(var_tamanho_brincar)
                            df_scatter_brincar['size_display'] = df_scatter_brincar[var_tamanho_brincar].abs() * format_size['multiplicador']

                        fig_scatter_brincar = go.Figure()
                        cores_plotly = px.colors.qualitative.Plotly
                        idx_cor = 0

                        for instituicao in df_scatter_brincar['Instituição'].unique():
                            df_inst = df_scatter_brincar[df_scatter_brincar['Instituição'] == instituicao]
                            cor = obter_cor_banco(instituicao)
                            if not cor:
                                cor = cores_plotly[idx_cor % len(cores_plotly)]
                                idx_cor += 1

                            if var_tamanho_brincar == 'Tamanho Fixo':
                                marker_size = tamanho_constante
                            else:
                                max_size = df_scatter_brincar['size_display'].max()
                                if max_size > 0:
                                    marker_size = df_inst['size_display'] / max_size * 100
                                else:
                                    marker_size = 25

                            eixo_x_label = nome_metrica if eixo_x_brincar == 'Métrica Derivada' else eixo_x_brincar
                            eixo_y_label = nome_metrica if eixo_y_brincar == 'Métrica Derivada' else eixo_y_brincar

                            fig_scatter_brincar.add_trace(go.Scatter(
                                x=df_inst['x_display'],
                                y=df_inst['y_display'],
                                mode='markers',
                                name=instituicao,
                                marker=dict(size=marker_size, color=cor, opacity=1.0, line=dict(width=1, color='white')),
                                hovertemplate=f'<b>{instituicao}</b><br>{eixo_x_label}: %{{x:{format_x["tickformat"]}}}{format_x["ticksuffix"]}<br>{eixo_y_label}: %{{y:{format_y["tickformat"]}}}{format_y["ticksuffix"]}<extra></extra>'
                            ))

                        titulo_scatter = f'{nome_metrica}: {eixo_y_brincar} vs {eixo_x_brincar} - {periodo_scatter_brincar}'

                        fig_scatter_brincar.update_layout(
                            title=titulo_scatter,
                            xaxis_title=eixo_x_label,
                            yaxis_title=eixo_y_label,
                            height=650,
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='white',
                            showlegend=True,
                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                            xaxis=dict(tickformat=format_x['tickformat'], ticksuffix=format_x['ticksuffix']),
                            yaxis=dict(tickformat=format_y['tickformat'], ticksuffix=format_y['ticksuffix']),
                            font=dict(family='IBM Plex Sans')
                        )

                        st.plotly_chart(fig_scatter_brincar, width='stretch')

                        # Tabela e exportação
                        with st.expander("ver dados e exportar"):
                            # Prepara dados para exportação
                            df_export = df_scatter_brincar[['Instituição', eixo_x_brincar, eixo_y_brincar]].copy()
                            df_export['Período'] = periodo_scatter_brincar

                            # Adiciona componentes da fórmula
                            componentes = list(set([s['variavel'] for s in steps]))
                            for comp in componentes:
                                if comp not in df_export.columns:
                                    df_export[comp] = df_scatter_brincar[comp]

                            # Reordena colunas
                            cols_ordem = ['Período', 'Instituição', 'Métrica Derivada'] + [c for c in df_export.columns if c not in ['Período', 'Instituição', 'Métrica Derivada', eixo_x_brincar, eixo_y_brincar]]
                            if eixo_x_brincar != 'Métrica Derivada':
                                cols_ordem.append(eixo_x_brincar)
                            if eixo_y_brincar != 'Métrica Derivada' and eixo_y_brincar not in cols_ordem:
                                cols_ordem.append(eixo_y_brincar)

                            df_export['Métrica Derivada'] = df_scatter_brincar['Métrica Derivada']
                            df_export = df_export[[c for c in cols_ordem if c in df_export.columns]]

                            st.dataframe(df_export, width='stretch', hide_index=True)

                            col_excel, col_csv = st.columns(2)
                            with col_excel:
                                buffer_excel = BytesIO()
                                with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                    df_export.to_excel(writer, index=False, sheet_name='dados')
                                buffer_excel.seek(0)
                                st.download_button(
                                    label="Exportar Excel",
                                    data=buffer_excel,
                                    file_name=f"Brincar_Scatter_{nome_metrica.replace(' ', '_')}_{periodo_scatter_brincar.replace('/', '-')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="export_excel_scatter_brincar"
                                )
                            with col_csv:
                                csv_data = df_export.to_csv(index=False)
                                st.download_button(
                                    label="Exportar CSV",
                                    data=csv_data,
                                    file_name=f"Brincar_Scatter_{nome_metrica.replace(' ', '_')}_{periodo_scatter_brincar.replace('/', '-')}.csv",
                                    mime="text/csv",
                                    key="export_csv_scatter_brincar"
                                )
                    else:
                        st.warning("sem dados válidos para exibir no scatter plot")

                # ===== DELTAS =====
                elif tipo_visualizacao == "Deltas":
                    st.markdown("---")
                    col_p1, col_p2, col_tipo_var = st.columns([2, 2, 1])

                    with col_p1:
                        indice_inicial_brincar = 1 if len(periodos_dropdown) > 1 else 0
                        periodo_inicial_brincar = st.selectbox(
                            "período inicial",
                            periodos_dropdown,
                            index=indice_inicial_brincar,
                            key="periodo_inicial_brincar",
                            format_func=periodo_para_exibicao
                        )
                    with col_p2:
                        periodo_subsequente_brincar = st.selectbox(
                            "período subsequente",
                            periodos_dropdown,
                            index=0,
                            key="periodo_subsequente_brincar",
                            format_func=periodo_para_exibicao
                        )
                    with col_tipo_var:
                        tipo_variacao_brincar = st.radio(
                            "ordenar por",
                            ["Δ absoluto", "Δ %"],
                            index=1,
                            key="tipo_variacao_brincar",
                            horizontal=True
                        )

                    # Validação de períodos
                    idx_ini = periodos_disponiveis.index(periodo_inicial_brincar)
                    idx_sub = periodos_disponiveis.index(periodo_subsequente_brincar)
                    periodo_valido = idx_sub > idx_ini

                    if not periodo_valido:
                        st.warning("o período subsequente deve ser posterior ao período inicial")
                    else:
                        # Filtra dados para os dois períodos
                        df_inicial = df[df['Período'] == periodo_inicial_brincar].copy()
                        df_subsequente = df[df['Período'] == periodo_subsequente_brincar].copy()

                        # Calcular métrica derivada para ambos os períodos
                        df_inicial_calc = df_inicial[df_inicial['Instituição'].isin(bancos_selecionados_brincar)].copy()
                        df_subsequente_calc = df_subsequente[df_subsequente['Instituição'].isin(bancos_selecionados_brincar)].copy()

                        df_inicial_calc['Métrica Derivada'] = calcular_metrica_derivada(df_inicial_calc, steps)
                        df_subsequente_calc['Métrica Derivada'] = calcular_metrica_derivada(df_subsequente_calc, steps)

                        # Determinar formato
                        if formato_resultado == "Percentual (%)":
                            format_info = {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
                        elif formato_resultado == "Valor bruto (R$)":
                            format_info = {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                        elif formato_resultado == "Múltiplo (x)":
                            format_info = {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                        elif formato_resultado == "Auto":
                            if formula_eh_divisao(steps):
                                format_info = {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                            else:
                                format_info = {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                        else:
                            format_info = {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}

                        # Prepara dados para o gráfico
                        dados_grafico_brincar = []
                        for instituicao in bancos_selecionados_brincar:
                            valor_ini = df_inicial_calc[df_inicial_calc['Instituição'] == instituicao]['Métrica Derivada'].values
                            valor_sub = df_subsequente_calc[df_subsequente_calc['Instituição'] == instituicao]['Métrica Derivada'].values

                            if len(valor_ini) > 0 and len(valor_sub) > 0:
                                v_ini = valor_ini[0]
                                v_sub = valor_sub[0]

                                if pd.isna(v_ini) or pd.isna(v_sub):
                                    continue

                                delta_absoluto = v_sub - v_ini

                                # Formata delta texto
                                if formato_resultado == "Percentual (%)" or (formato_resultado == "Auto" and formula_eh_divisao(steps)):
                                    delta_texto = f"{delta_absoluto * format_info['multiplicador']:+.2f}{format_info['ticksuffix']}"
                                elif formato_resultado == "Valor bruto (R$)" or (formato_resultado == "Auto" and not formula_eh_divisao(steps)):
                                    delta_texto = f"R$ {delta_absoluto/1e6:+,.0f}MM".replace(",", ".")
                                else:
                                    delta_texto = f"{delta_absoluto:+.2f}"

                                # Variação percentual
                                if v_ini == 0:
                                    if delta_absoluto > 0:
                                        variacao_pct = float('inf')
                                        variacao_texto = "+∞"
                                    elif delta_absoluto < 0:
                                        variacao_pct = float('-inf')
                                        variacao_texto = "-∞"
                                    else:
                                        variacao_pct = 0
                                        variacao_texto = "0.0%"
                                else:
                                    variacao_pct = ((v_sub - v_ini) / abs(v_ini)) * 100
                                    variacao_texto = f"{variacao_pct:+.1f}%"

                                dados_grafico_brincar.append({
                                    'instituicao': instituicao,
                                    'valor_ini': v_ini,
                                    'valor_sub': v_sub,
                                    'delta': delta_absoluto,
                                    'delta_texto': delta_texto,
                                    'variacao_pct': variacao_pct if not (variacao_pct == float('inf') or variacao_pct == float('-inf')) else (1e10 if variacao_pct > 0 else -1e10),
                                    'variacao_texto': variacao_texto
                                })

                        if dados_grafico_brincar:
                            # Ordena pela variação
                            if tipo_variacao_brincar == "Δ %":
                                dados_grafico_brincar = sorted(dados_grafico_brincar, key=lambda x: x['variacao_pct'], reverse=True)
                            else:
                                dados_grafico_brincar = sorted(dados_grafico_brincar, key=lambda x: x['delta'], reverse=True)

                            # Cria o gráfico estilo lollipop
                            fig_delta_brincar = go.Figure()

                            for i, dado in enumerate(dados_grafico_brincar):
                                inst = dado['instituicao']
                                y_ini = dado['valor_ini'] * format_info['multiplicador']
                                y_sub = dado['valor_sub'] * format_info['multiplicador']
                                delta_positivo = dado['delta'] > 0

                                cor_sub = '#2E7D32' if delta_positivo else '#C62828'

                                # Linha conectando os dois pontos
                                fig_delta_brincar.add_trace(go.Scatter(
                                    x=[inst, inst],
                                    y=[y_ini, y_sub],
                                    mode='lines',
                                    line=dict(color='#9E9E9E', width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))

                                # Bolinha do período inicial
                                fig_delta_brincar.add_trace(go.Scatter(
                                    x=[inst],
                                    y=[y_ini],
                                    mode='markers',
                                    marker=dict(size=12, color='#424242', line=dict(width=1, color='white')),
                                    name=periodo_inicial_brincar if i == 0 else None,
                                    showlegend=(i == 0),
                                    legendgroup='inicial',
                                    hovertemplate=f'<b>{inst}</b><br>{periodo_inicial_brincar}: %{{y:{format_info["tickformat"]}}}{format_info["ticksuffix"]}<extra></extra>'
                                ))

                                # Bolinha do período subsequente
                                fig_delta_brincar.add_trace(go.Scatter(
                                    x=[inst],
                                    y=[y_sub],
                                    mode='markers',
                                    marker=dict(size=12, color=cor_sub, line=dict(width=1, color='white')),
                                    name=periodo_subsequente_brincar if i == 0 else None,
                                    showlegend=(i == 0),
                                    legendgroup='subsequente',
                                    customdata=[[dado['delta_texto'], dado['variacao_texto']]],
                                    hovertemplate=f'<b>{inst}</b><br>{periodo_subsequente_brincar}: %{{y:{format_info["tickformat"]}}}{format_info["ticksuffix"]}<br>Δ: %{{customdata[0]}}<br>Variação: %{{customdata[1]}}<extra></extra>'
                                ))

                            titulo_delta_brincar = f"{nome_metrica}: {periodo_inicial_brincar} → {periodo_subsequente_brincar}"

                            fig_delta_brincar.update_layout(
                                title=dict(
                                    text=titulo_delta_brincar,
                                    font=dict(size=16, family='IBM Plex Sans')
                                ),
                                height=max(400, len(dados_grafico_brincar) * 25 + 150),
                                plot_bgcolor='#f8f9fa',
                                paper_bgcolor='white',
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="left",
                                    x=0
                                ),
                                xaxis=dict(
                                    showgrid=False,
                                    tickangle=45 if len(dados_grafico_brincar) > 10 else 0,
                                    tickfont=dict(size=10)
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='#e0e0e0',
                                    tickformat=format_info['tickformat'],
                                    ticksuffix=format_info['ticksuffix'],
                                    title=nome_metrica
                                ),
                                font=dict(family='IBM Plex Sans'),
                                margin=dict(l=60, r=20, t=80, b=100)
                            )

                            st.plotly_chart(fig_delta_brincar, width='stretch', config={'displayModeBar': False})

                            # Tabela e exportação
                            with st.expander("ver dados e exportar"):
                                df_resumo_brincar = pd.DataFrame(dados_grafico_brincar)
                                df_resumo_brincar = df_resumo_brincar.rename(columns={
                                    'instituicao': 'Instituição',
                                    'valor_ini': periodo_inicial_brincar,
                                    'valor_sub': periodo_subsequente_brincar,
                                    'delta_texto': 'Delta',
                                    'variacao_texto': 'Variação %'
                                })

                                # Adiciona componentes da fórmula
                                componentes = list(set([s['variavel'] for s in steps]))
                                df_export_delta = df_resumo_brincar[['Instituição', periodo_inicial_brincar, periodo_subsequente_brincar, 'Delta', 'Variação %']].copy()

                                st.dataframe(df_export_delta, width='stretch', hide_index=True)

                                col_excel, col_csv = st.columns(2)
                                with col_excel:
                                    buffer_excel = BytesIO()
                                    with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                        df_export_delta.to_excel(writer, index=False, sheet_name='deltas')
                                    buffer_excel.seek(0)
                                    st.download_button(
                                        label="Exportar Excel",
                                        data=buffer_excel,
                                        file_name=f"Brincar_Deltas_{nome_metrica.replace(' ', '_')}_{periodo_inicial_brincar.replace('/', '-')}_{periodo_subsequente_brincar.replace('/', '-')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="export_excel_delta_brincar"
                                    )
                                with col_csv:
                                    csv_data = df_export_delta.to_csv(index=False)
                                    st.download_button(
                                        label="Exportar CSV",
                                        data=csv_data,
                                        file_name=f"Brincar_Deltas_{nome_metrica.replace(' ', '_')}_{periodo_inicial_brincar.replace('/', '-')}_{periodo_subsequente_brincar.replace('/', '-')}.csv",
                                        mime="text/csv",
                                        key="export_csv_delta_brincar"
                                    )
                        else:
                            st.info("sem dados válidos para exibir")

                # ===== RANKING (BARRAS) =====
                elif tipo_visualizacao == "Ranking (barras)":
                    st.markdown("---")
                    col_periodo_rank, col_ordem, col_media = st.columns([2, 1, 1])

                    with col_periodo_rank:
                        periodo_ranking = st.selectbox(
                            "período",
                            periodos_dropdown,
                            index=0,
                            key="periodo_ranking_brincar",
                            format_func=periodo_para_exibicao
                        )

                    with col_ordem:
                        ordem_ranking = st.radio(
                            "ordenação",
                            ["Maior → Menor", "Menor → Maior"],
                            horizontal=True,
                            key="ordem_ranking_brincar"
                        )

                    with col_media:
                        mostrar_media = st.checkbox("mostrar média do grupo", value=True, key="mostrar_media_brincar")

                    # Filtrar dados para o período
                    df_periodo_rank = df[df['Período'] == periodo_ranking].copy()
                    df_ranking = df_periodo_rank[df_periodo_rank['Instituição'].isin(bancos_selecionados_brincar)].copy()

                    # Calcular métrica derivada
                    df_ranking['Métrica Derivada'] = calcular_metrica_derivada(df_ranking, steps)
                    df_ranking = df_ranking.dropna(subset=['Métrica Derivada'])

                    if len(df_ranking) > 0:
                        # Ordenar
                        ascending = ordem_ranking == "Menor → Maior"
                        df_ranking = df_ranking.sort_values('Métrica Derivada', ascending=ascending)

                        # Determinar formato
                        if formato_resultado == "Percentual (%)":
                            format_info = {'tickformat': '.2f', 'ticksuffix': '%', 'multiplicador': 100}
                        elif formato_resultado == "Valor bruto (R$)":
                            format_info = {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                        elif formato_resultado == "Múltiplo (x)":
                            format_info = {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                        elif formato_resultado == "Auto":
                            if formula_eh_divisao(steps):
                                format_info = {'tickformat': '.2f', 'ticksuffix': 'x', 'multiplicador': 1}
                            else:
                                format_info = {'tickformat': ',.0f', 'ticksuffix': 'M', 'multiplicador': 1/1e6}
                        else:
                            format_info = {'tickformat': '.2f', 'ticksuffix': '', 'multiplicador': 1}

                        # Preparar valores para exibição
                        df_ranking['valor_display'] = df_ranking['Métrica Derivada'] * format_info['multiplicador']

                        # Cores por instituição
                        cores = []
                        cores_plotly = px.colors.qualitative.Plotly
                        idx_cor = 0
                        for inst in df_ranking['Instituição']:
                            cor = obter_cor_banco(inst)
                            if not cor:
                                cor = cores_plotly[idx_cor % len(cores_plotly)]
                                idx_cor += 1
                            cores.append(cor)

                        fig_ranking = go.Figure()

                        fig_ranking.add_trace(go.Bar(
                            x=df_ranking['Instituição'],
                            y=df_ranking['valor_display'],
                            marker=dict(color=cores, line=dict(width=1, color='white')),
                            hovertemplate='<b>%{x}</b><br>' + nome_metrica + ': %{y:' + format_info['tickformat'] + '}' + format_info['ticksuffix'] + '<extra></extra>'
                        ))

                        # Adicionar linha de média
                        if mostrar_media:
                            media_valor = df_ranking['valor_display'].mean()
                            fig_ranking.add_hline(
                                y=media_valor,
                                line_dash="dash",
                                line_color="#FF6B6B",
                                line_width=2,
                                annotation_text=f"Média: {media_valor:{format_info['tickformat']}}{format_info['ticksuffix']}",
                                annotation_position="top right",
                                annotation_font=dict(color="#FF6B6B", size=12)
                            )

                        titulo_ranking = f"{nome_metrica} - Ranking {periodo_ranking}"

                        fig_ranking.update_layout(
                            title=dict(
                                text=titulo_ranking,
                                font=dict(size=16, family='IBM Plex Sans')
                            ),
                            height=max(400, len(df_ranking) * 25 + 150),
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='white',
                            showlegend=False,
                            xaxis=dict(
                                showgrid=False,
                                tickangle=45 if len(df_ranking) > 10 else 0,
                                tickfont=dict(size=10)
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='#e0e0e0',
                                tickformat=format_info['tickformat'],
                                ticksuffix=format_info['ticksuffix'],
                                title=nome_metrica
                            ),
                            font=dict(family='IBM Plex Sans'),
                            margin=dict(l=60, r=20, t=60, b=100)
                        )

                        st.plotly_chart(fig_ranking, width='stretch', config={'displayModeBar': False})

                        # Tabela e exportação
                        with st.expander("ver dados e exportar"):
                            # Prepara dados para exportação
                            df_export_rank = df_ranking[['Instituição']].copy()
                            df_export_rank['Período'] = periodo_ranking
                            df_export_rank[nome_metrica] = df_ranking['Métrica Derivada']

                            # Adiciona componentes da fórmula
                            componentes = list(set([s['variavel'] for s in steps]))
                            for comp in componentes:
                                if comp in df_ranking.columns:
                                    df_export_rank[comp] = df_ranking[comp].values

                            # Reordena colunas
                            cols_ordem = ['Período', 'Instituição', nome_metrica] + componentes
                            df_export_rank = df_export_rank[[c for c in cols_ordem if c in df_export_rank.columns]]

                            st.dataframe(df_export_rank, width='stretch', hide_index=True)

                            col_excel, col_csv = st.columns(2)
                            with col_excel:
                                buffer_excel = BytesIO()
                                with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                                    df_export_rank.to_excel(writer, index=False, sheet_name='ranking')
                                buffer_excel.seek(0)
                                st.download_button(
                                    label="Exportar Excel",
                                    data=buffer_excel,
                                    file_name=f"Brincar_Ranking_{nome_metrica.replace(' ', '_')}_{periodo_ranking.replace('/', '-')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="export_excel_ranking_brincar"
                                )
                            with col_csv:
                                csv_data = df_export_rank.to_csv(index=False)
                                st.download_button(
                                    label="Exportar CSV",
                                    data=csv_data,
                                    file_name=f"Brincar_Ranking_{nome_metrica.replace(' ', '_')}_{periodo_ranking.replace('/', '-')}.csv",
                                    mime="text/csv",
                                    key="export_csv_ranking_brincar"
                                )
                    else:
                        st.warning("sem dados válidos para exibir no ranking")

        elif not st.session_state['brincar_formula_steps']:
            st.info("construa uma fórmula adicionando variáveis para começar a análise")
        else:
            st.info("selecione instituições para comparar")

    else:
        st.info("carregando dados automaticamente do github...")
        st.markdown("por favor, aguarde alguns segundos e recarregue a página")

elif menu == "Atualizar Base":
    st.markdown("## Atualização Base")
    st.markdown("painel de controle unificado para extração de dados do IFData/BCB")
    st.markdown("---")

    # Importar o gerenciador de cache unificado
    from utils.ifdata_cache import CacheManager, CACHES_INFO, gerar_periodos_trimestrais as gerar_periodos_cache

    # Inicializar gerenciador
    if 'cache_manager' not in st.session_state:
        st.session_state['cache_manager'] = CacheManager()
    cache_manager = st.session_state['cache_manager']

    if 'df_aliases' in st.session_state:
        st.success(f"{len(st.session_state['df_aliases'])} aliases carregados")
    else:
        st.error("aliases não encontrados")

    # =============================================================
    # STATUS DE TODOS OS CACHES (LOCAL + GITHUB)
    # =============================================================
    st.markdown("### Status dos Caches")

    # Verificar status no GitHub Releases
    github_status = verificar_caches_github()
    gh_caches = github_status.get('caches', {})

    with st.expander("ver status de todos os caches", expanded=True):
        caches_disponiveis = cache_manager.listar_caches()
        caches_info = CACHES_INFO

        # Criar tabela UNIFICADA de status (Local + GitHub + Persistência)
        status_data = []
        for tipo_cache in caches_disponiveis:
            info = cache_manager.info(tipo_cache)
            cache_info = caches_info.get(tipo_cache, {})
            gh_info = gh_caches.get(tipo_cache, {})

            existe_local = info.get("existe", False)
            existe_github = gh_info.get("existe", False)

            # Determinar situação de PERSISTÊNCIA (o que importa no Streamlit Cloud)
            if existe_github:
                persistencia = "☁️ Persistido"  # Vai sobreviver ao restart
            elif existe_local:
                persistencia = "⚠️ Efêmero"  # Vai sumir no restart
            else:
                persistencia = "❌ Ausente"

            status_data.append({
                "Cache": cache_info.get("nome_exibicao", tipo_cache),
                "Local": "✅" if existe_local else "❌",
                "GitHub": "☁️" if existe_github else "❌",
                "Persistência": persistencia,
                "Períodos": str(info.get("total_periodos", 0)) if existe_local else "-",
                "Registros": str(info.get("total_registros", 0)) if existe_local else "-",
                "Tamanho GH": gh_info.get("tamanho_fmt", "-") if existe_github else "-",
            })

        df_status = pd.DataFrame(status_data)
        st.dataframe(df_status, width='stretch', hide_index=True)

        # Legenda
        st.caption("""
        **Legenda:**
        - **Local**: Existe no filesystem (efêmero no Streamlit Cloud)
        - **GitHub**: Publicado no GitHub Releases (persistente)
        - **☁️ Persistido**: Dados seguros, serão recuperados após restart
        - **⚠️ Efêmero**: Só existe local, será perdido no restart - PUBLIQUE!
        """)

        # Resumo e alertas
        total_local = sum(1 for s in status_data if s["Local"] == "✅")
        total_github = sum(1 for s in status_data if s["GitHub"] == "☁️")
        total_efemero = sum(1 for s in status_data if s["Persistência"] == "⚠️ Efêmero")

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Cache Local", f"{total_local}/8")
        with col_r2:
            st.metric("GitHub Releases", f"{total_github}/8")
        with col_r3:
            if total_efemero > 0:
                st.metric("⚠️ Efêmeros", f"{total_efemero}", delta="Publicar!", delta_color="inverse")
            else:
                st.metric("✅ Efêmeros", "0")

        if total_efemero > 0:
            caches_efemeros = [s["Cache"] for s in status_data if s["Persistência"] == "⚠️ Efêmero"]
            st.warning(f"⚠️ **Atenção:** Os caches a seguir existem apenas localmente e serão perdidos no restart: **{', '.join(caches_efemeros)}**. Publique-os no GitHub!")

        if not github_status.get('release_existe'):
            st.error(f"❌ Release não acessível: {github_status.get('erro', 'erro desconhecido')}")
        else:
            st.caption(f"📦 Repositório: `{github_status.get('repo')}` | Tag: `{github_status.get('tag')}`")

    st.markdown("---")
    st.markdown("### Extração de Dados (Admin)")

    senha_input = st.text_input("senha de administrador", type="password", key="senha_admin_atualizacao_nova")

    if senha_input == SENHA_ADMIN:

        # =============================================================
        # SELEÇÃO DO CACHE A ATUALIZAR
        # =============================================================
        st.markdown("#### 1. Selecione o cache a atualizar")

        # Opções de cache com descrição
        opcoes_cache = {
            "principal": "Resumo (Rel. 1) - variáveis selecionadas",
            "capital": "Capital Regulatório (Rel. 5) - variáveis selecionadas",
            "ativo": "Ativo (Rel. 2) - TODAS as variáveis",
            "passivo": "Passivo (Rel. 3) - TODAS as variáveis",
            "dre": "DRE (Rel. 4) - TODAS as variáveis",
            "carteira_pf": "Carteira PF (Rel. 11) - TODAS as variáveis",
            "carteira_pj": "Carteira PJ (Rel. 13) - TODAS as variáveis",
            "carteira_instrumentos": "Carteira Instrumentos 4.966 (Rel. 16) - TODAS as variáveis",
            "taxas_juros": "Taxas de Juros (API BCB) - TODOS produtos/instituições",
        }

        cache_selecionado = st.selectbox(
            "cache para atualizar",
            options=list(opcoes_cache.keys()),
            format_func=lambda x: opcoes_cache[x],
            key="cache_selecionado"
        )

        # Mostrar status do cache selecionado
        info_selecionado = cache_manager.info(cache_selecionado)
        if info_selecionado.get("existe"):
            st.info(f"Cache atual: {info_selecionado.get('total_periodos', 0)} períodos, {info_selecionado.get('total_registros', 0):,} registros")
        else:
            st.warning(f"Cache '{cache_selecionado}' não existe ainda")

        # Flag para identificar se é extração de taxas de juros
        is_taxas_juros = (cache_selecionado == "taxas_juros")

        # =============================================================
        # MODO DE ATUALIZAÇÃO
        # =============================================================
        st.markdown("#### 2. Modo de atualização")

        modo_atualizacao = st.radio(
            "modo",
            options=["incremental", "overwrite"],
            format_func=lambda x: "Incremental (adiciona/atualiza períodos)" if x == "incremental" else "Overwrite (substitui todo o cache)",
            horizontal=True,
            key="modo_atualizacao"
        )

        if modo_atualizacao == "overwrite":
            st.warning("Modo OVERWRITE: todos os dados existentes serão substituídos!")

        # =============================================================
        # SELEÇÃO DE PERÍODOS
        # =============================================================
        st.markdown("#### 3. Selecione o período de extração")

        # Taxas de Juros usa seleção de data (diário), demais usam trimestral
        if is_taxas_juros:
            st.caption("⚠️ Taxas de Juros: extração completa de TODOS os produtos e TODAS as instituições")

            col_data1, col_data2 = st.columns(2)

            hoje = datetime.now()
            data_minima_tj = hoje - timedelta(days=365*3)  # 3 anos de histórico

            with col_data1:
                data_inicio_tj = st.date_input(
                    "Data inicial",
                    value=hoje - timedelta(days=365),  # 1 ano atrás por padrão
                    min_value=data_minima_tj.date(),
                    max_value=hoje.date(),
                    key="taxas_juros_data_inicio_extracao",
                    format="DD/MM/YYYY"
                )

            with col_data2:
                data_fim_tj = st.date_input(
                    "Data final",
                    value=hoje.date(),
                    min_value=data_minima_tj.date(),
                    max_value=hoje.date(),
                    key="taxas_juros_data_fim_extracao",
                    format="DD/MM/YYYY"
                )

            st.caption(f"Período selecionado: {data_inicio_tj.strftime('%d/%m/%Y')} até {data_fim_tj.strftime('%d/%m/%Y')}")
            st.info("A extração inclui paginação completa ($skip/$top) para garantir que todos os dados sejam capturados sem truncamento.")

            # Não precisa de periodos_extrair para taxas_juros
            periodos_extrair = None

        else:
            col1, col2 = st.columns(2)
            with col1:
                ano_i = st.selectbox("ano inicial", range(2015, 2029), index=8, key="ano_i_unificado")
                mes_i = st.selectbox("trimestre inicial", ['03', '06', '09', '12'], key="mes_i_unificado")
            with col2:
                ano_f = st.selectbox("ano final", range(2015, 2029), index=10, key="ano_f_unificado")
                mes_f = st.selectbox("trimestre final", ['03', '06', '09', '12'], index=2, key="mes_f_unificado")

            periodos_extrair = gerar_periodos_cache(ano_i, mes_i, ano_f, mes_f)
            st.caption(f"Serão extraídos {len(periodos_extrair)} períodos: {periodos_extrair[0][4:6]}/{periodos_extrair[0][:4]} até {periodos_extrair[-1][4:6]}/{periodos_extrair[-1][:4]}")

        # =============================================================
        # CONFIGURAÇÕES AVANÇADAS
        # =============================================================
        if not is_taxas_juros:
            with st.expander("configurações avançadas"):
                intervalo_save = st.slider(
                    "salvar a cada N períodos",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="O cache será salvo parcialmente a cada N períodos extraídos para evitar perda de dados",
                    key="intervalo_save"
                )

                st.caption("Nota: a extração usa Tipo de Instituição 1 (Conglomerados Prudenciais e Instituições Independentes)")
        else:
            intervalo_save = 1  # Não usado para taxas de juros

        # =============================================================
        # CONFIGURAÇÃO DO TOKEN GITHUB (para publicação)
        # =============================================================
        st.markdown("#### Token GitHub (para publicação)")

        # Verificar se há token nos secrets do Streamlit
        token_from_secrets = None
        try:
            token_from_secrets = st.secrets.get("GITHUB_TOKEN")
        except Exception:
            pass

        if token_from_secrets:
            st.success("✅ Token GitHub configurado via Streamlit Secrets")
            gh_token_final = token_from_secrets
        else:
            st.info("💡 Configure `GITHUB_TOKEN` nos Secrets do Streamlit Cloud para upload automático")
            gh_token_manual = st.text_input(
                "ou insira token manualmente (permissão 'repo')",
                type="password",
                key="gh_token_unificado",
                help="Token com permissão 'repo'. Configure nos Secrets para não precisar digitar."
            )
            gh_token_final = gh_token_manual if gh_token_manual else None

        # Armazenar no session_state para usar em outras partes
        st.session_state['_gh_token_unificado'] = gh_token_final

        # =============================================================
        # BOTÃO DE EXTRAÇÃO
        # =============================================================
        st.markdown("#### 4. Executar extração")

        # Para taxas de juros, não precisa de aliases; para outros, precisa
        pode_extrair = is_taxas_juros or ('dict_aliases' in st.session_state)

        if pode_extrair:

            if st.button(f"Extrair dados de {opcoes_cache[cache_selecionado]}", type="primary", width='stretch', key="btn_extrair_unificado"):

                # Containers para UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                save_status = st.empty()
                log_container = st.container()
                error_container = st.container()
                erros_encontrados = []
                logs_extracao = []

                # =============================================================
                # EXTRAÇÃO ESPECIAL PARA TAXAS DE JUROS
                # =============================================================
                if is_taxas_juros:
                    from utils.ifdata_cache import TaxasJurosCache

                    def callback_progresso_tj(progress, message):
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(message)

                    def callback_log_tj(message):
                        logs_extracao.append(message)
                        with log_container:
                            st.caption(f"📝 {message}")

                    st.info(f"Iniciando extração de Taxas de Juros de {data_inicio_tj.strftime('%d/%m/%Y')} até {data_fim_tj.strftime('%d/%m/%Y')}")

                    try:
                        # Obter o cache de taxas de juros
                        cache_taxas = cache_manager.get_cache("taxas_juros")

                        if cache_taxas is None:
                            st.error("Cache de Taxas de Juros não configurado corretamente")
                        else:
                            # Executar extração completa com paginação
                            resultado = cache_taxas.extrair_completo(
                                data_inicio=data_inicio_tj.strftime('%Y-%m-%d'),
                                data_fim=data_fim_tj.strftime('%Y-%m-%d'),
                                progress_callback=callback_progresso_tj,
                                log_callback=callback_log_tj
                            )

                            # Limpar UI de progresso
                            progress_bar.empty()
                            status_text.empty()

                            if resultado.sucesso:
                                # Salvar cache
                                save_status.text("Salvando cache...")

                                # Se modo overwrite, limpar antes
                                if modo_atualizacao == "overwrite":
                                    cache_taxas.limpar_local()

                                # Salvar os dados
                                save_result = cache_taxas.salvar_local(resultado.dados, resultado.metadata)

                                save_status.empty()

                                if save_result.sucesso:
                                    st.success(f"✅ Extração e salvamento concluídos: {resultado.mensagem}")

                                    # Mostrar estatísticas
                                    meta = resultado.metadata or {}
                                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                    with col_stat1:
                                        st.metric("Total de linhas", f"{meta.get('total_registros', 0):,}")
                                    with col_stat2:
                                        st.metric("Produtos", meta.get('produtos_unicos', 0))
                                    with col_stat3:
                                        st.metric("Instituições", meta.get('instituicoes_unicas', 0))
                                    with col_stat4:
                                        st.metric("Períodos (datas)", meta.get('periodos_unicos', 0))

                                    # Verificar truncamento
                                    if meta.get('truncado'):
                                        st.warning("⚠️ AVISO: Possível truncamento detectado! Verifique se todos os dados foram extraídos.")
                                    else:
                                        st.success("✅ Extração completa sem truncamento")

                                    # Mostrar logs em expander
                                    with st.expander("📋 Log de extração", expanded=False):
                                        for log_line in logs_extracao:
                                            st.text(log_line)
                                else:
                                    st.error(f"Erro ao salvar cache: {save_result.mensagem}")
                            else:
                                st.error(f"Erro na extração: {resultado.mensagem}")

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Erro durante extração: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                # =============================================================
                # EXTRAÇÃO PADRÃO PARA OUTROS CACHES
                # =============================================================
                else:
                    def callback_progresso(i, total, periodo):
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"extraindo {periodo[4:6]}/{periodo[:4]} ({i + 1}/{total})")

                    def callback_salvamento(info):
                        save_status.text(f"salvando... {info}")

                    def callback_erro(periodo, mensagem):
                        erros_encontrados.append(f"{periodo[4:6]}/{periodo[:4]}: {mensagem}")
                        with error_container:
                            st.warning(f"Erro em {periodo[4:6]}/{periodo[:4]}: {mensagem[:100]}...")

                    st.info(f"iniciando extração de {len(periodos_extrair)} períodos para '{cache_selecionado}'. Salvamento a cada {intervalo_save} períodos.")

                    try:
                        # Usar o gerenciador unificado para extração
                        resultado = cache_manager.extrair_periodos_com_salvamento(
                            tipo=cache_selecionado,
                            periodos=periodos_extrair,
                            modo=modo_atualizacao,
                            intervalo_salvamento=intervalo_save,
                            callback_progresso=callback_progresso,
                            callback_salvamento=callback_salvamento,
                            callback_erro=callback_erro,
                            dict_aliases=st.session_state.get('dict_aliases', {})
                        )

                        # Limpar UI de progresso
                        progress_bar.empty()
                        status_text.empty()
                        save_status.empty()

                        if resultado.sucesso:
                            st.success(f"Extração concluída: {resultado.mensagem}")

                            # Mostrar estatísticas
                            metadata = resultado.metadata or {}
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Períodos extraídos", f"{metadata.get('periodos_extraidos', 0)}/{metadata.get('periodos_total', 0)}")
                            with col_stat2:
                                st.metric("Registros totais", f"{metadata.get('total_registros', 0):,}")
                            with col_stat3:
                                st.metric("Modo", metadata.get('modo', 'N/A'))

                            # Mostrar erros se houver
                            if metadata.get('erros'):
                                with st.expander(f"erros encontrados ({len(metadata['erros'])})", expanded=False):
                                    for erro in metadata['erros']:
                                        st.caption(f"- {erro}")

                            # =============================================================
                            # DOWNLOAD IMEDIATO DO CACHE
                            # =============================================================
                            st.markdown("---")
                            st.markdown("#### Download do cache")
                            st.caption("Faça download imediato do cache para backup caso a publicação no GitHub falhe")

                            col_dl1, col_dl2 = st.columns(2)

                            with col_dl1:
                                # Download com fallback (parquet/csv/pickle)
                                download_payload = cache_manager.get_dados_para_download(cache_selecionado)
                                if download_payload:
                                    st.download_button(
                                        label=f"Download ({download_payload['label']})",
                                        data=download_payload["data"],
                                        file_name=f"{cache_selecionado}_cache{download_payload['ext']}",
                                        mime=download_payload["mime"],
                                        key="download_parquet"
                                    )
                                    if download_payload["label"].lower() != "parquet":
                                        st.caption("fallback usado por indisponibilidade do parquet")

                            with col_dl2:
                                # Download CSV
                                dados_csv = cache_manager.get_dados_para_download_csv(cache_selecionado)
                                if dados_csv:
                                    st.download_button(
                                        label="Download (CSV)",
                                        data=dados_csv,
                                        file_name=f"{cache_selecionado}_cache.csv",
                                        mime="text/csv",
                                        key="download_csv"
                                    )

                            # Atualizar session_state para caches principais
                            if cache_selecionado == "principal" and resultado.dados is not None:
                                # Converter para formato antigo se necessário
                                from utils.ifdata_cache import PrincipalCache
                                pc = PrincipalCache(cache_manager.base_dir)
                                dados_dict = pc.carregar_formato_antigo()
                                if dados_dict:
                                    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos'] and modo_atualizacao == "incremental":
                                        st.session_state['dados_periodos'].update(dados_dict)
                                    else:
                                        st.session_state['dados_periodos'] = dados_dict
                                    st.session_state['cache_fonte'] = 'extração local'

                            elif cache_selecionado == "capital" and resultado.dados is not None:
                                from utils.ifdata_cache import CapitalCache
                                cc = CapitalCache(cache_manager.base_dir)
                                dados_dict = cc.carregar_formato_antigo()
                                if dados_dict:
                                    st.session_state['dados_capital'] = dados_dict

                        else:
                            st.error(f"Extração falhou: {resultado.mensagem}")
                            if resultado.metadata and resultado.metadata.get('erros'):
                                with st.expander("detalhes dos erros"):
                                    for erro in resultado.metadata['erros']:
                                        st.caption(f"- {erro}")

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        save_status.empty()
                        st.error(f"Erro durante extração: {str(e)}")

                        import traceback
                        with st.expander("traceback completo"):
                            st.code(traceback.format_exc())

        else:
            st.warning("carregue os aliases primeiro (verifique a conexão com Google Sheets)")

        # =============================================================
        # SEÇÃO: PUBLICAR NO GITHUB
        # =============================================================
        st.markdown("---")
        st.markdown("### Publicar cache no GitHub")
        st.caption("Envia o cache local para GitHub Releases (tag v1.0-cache) para uso permanente")

        # Recuperar token do session_state
        token_para_upload = st.session_state.get('_gh_token_unificado')

        if not token_para_upload:
            st.warning("⚠️ Nenhum token GitHub disponível. Configure nos Secrets ou insira manualmente acima.")

        col_pub1, col_pub2 = st.columns([3, 1])
        with col_pub1:
            if st.button(f"📤 Enviar '{cache_selecionado}' para GitHub", width='stretch', key="btn_enviar_github_unificado", disabled=not token_para_upload):
                with st.spinner(f"enviando cache '{cache_selecionado}' para github releases..."):
                    sucesso, mensagem = upload_cache_github(
                        cache_manager,
                        cache_selecionado,
                        token_para_upload
                    )
                    if sucesso:
                        st.success(f"✅ {mensagem}")
                        st.balloons()
                    else:
                        st.error(f"❌ {mensagem}")
        with col_pub2:
            st.caption(f"Token: {'✅' if token_para_upload else '❌'}")

        # =============================================================
        # SEÇÃO LEGACY: EXTRAÇÃO PRINCIPAL (COMPATIBILIDADE)
        # =============================================================
        st.markdown("---")
        with st.expander("extração legacy (sistema antigo)", expanded=False):
            st.caption("Use esta opção apenas para compatibilidade com o sistema antigo")

            col_leg1, col_leg2 = st.columns(2)
            with col_leg1:
                ano_leg_i = st.selectbox("ano inicial", range(2015, 2028), index=8, key="ano_leg_i")
                mes_leg_i = st.selectbox("trim. inicial", ['03', '06', '09', '12'], key="mes_leg_i")
            with col_leg2:
                ano_leg_f = st.selectbox("ano final", range(2015, 2028), index=10, key="ano_leg_f")
                mes_leg_f = st.selectbox("trim. final", ['03', '06', '09', '12'], index=2, key="mes_leg_f")

            if st.button("extrair (sistema legado)", key="btn_extrair_legado"):
                periodos = gerar_periodos(ano_leg_i, mes_leg_i, ano_leg_f, mes_leg_f)
                progress_bar = st.progress(0)
                status = st.empty()
                save_status = st.empty()

                def update(i, total, p):
                    progress_bar.progress((i+1)/total)
                    status.text(f"extraindo {p[4:6]}/{p[:4]} ({i+1}/{total})")

                def save_progress(dados_parciais, info):
                    save_status.text(f"salvando {len(dados_parciais)} períodos...")
                    salvar_cache(dados_parciais, info, incremental=True)
                    save_status.text(f"salvos {len(dados_parciais)} períodos")

                dados = processar_todos_periodos(
                    periodos,
                    st.session_state['dict_aliases'],
                    progress_callback=update,
                    save_callback=save_progress,
                    save_interval=5
                )

                if not dados:
                    progress_bar.empty()
                    status.empty()
                    save_status.empty()
                    st.error("falha ao extrair dados")
                else:
                    periodo_info = f"{periodos[0][4:6]}/{periodos[0][:4]} até {periodos[-1][4:6]}/{periodos[-1][:4]}"
                    cache_salvo = salvar_cache(dados, periodo_info, incremental=True)

                    if 'dados_periodos' in st.session_state and st.session_state['dados_periodos']:
                        dados_merged = st.session_state['dados_periodos'].copy()
                        dados_merged.update(dados)
                        st.session_state['dados_periodos'] = dados_merged
                    else:
                        st.session_state['dados_periodos'] = dados

                    st.session_state['cache_fonte'] = 'extração local'

                    progress_bar.empty()
                    status.empty()
                    save_status.empty()
                    st.success(f"{len(dados)} períodos extraídos!")
                    st.rerun()

    elif senha_input:
        st.error("senha incorreta")

elif menu == "Glossário":
    st.markdown("""
    ## **Sobre os Dados Apresentados**

    Os dados são obtidos via API Olinda do IFdata do Banco Central, na visão **Conglomerado**.

    **O que isso significa:**

    **Conglomerado Prudencial** é um grupo de instituições financeiras e entidades autorizadas, controladas por uma mesma instituição líder, que deve consolidar suas demonstrações contábeis para fins regulatórios.

    O Banco Central supervisiona o grupo como um todo, considerando o somatório das instituições, e decisões sobre liquidez ou insolvência afetam todas as entidades do conglomerado.

    **Exemplos:**
    - O Banco PAN faz parte do Conglomerado Prudencial BTG Pactual. O próprio PAN esclarece que, por estar consolidado no prudencial do BTG, o índice de Basileia "individual" deixou de ser formalmente reportado.
    - O Banco Digio faz parte do conglomerado financeiro do Bradesco, então em leituras "por conglomerado" você não verá o Digio como um grupo separado.
    - O Conglomerado Prudencial Original inclui Banco Original, PicPay Bank e a IP do PicPay (além de outros veículos do grupo). Dependendo do recorte, você verá o conglomerado (ou o líder) e não cada entidade isoladamente.

    Em caso de dúvidas, consulte o site do Banco Central.

    ---

    ## **Por que os dados do IFdata (balanço etc.) começam em Mar-2015?**

    A partir de **Mar-2015**, usamos a visão **Conglomerado Prudencial**, criada pelo Banco Central naquela data.
    Antes disso, os dados eram reportados na visão **Conglomerado Financeiro**.

    Para instituições com apenas um CNPJ, a comparação entre as duas visões pode até ser possível.
    Para conglomerados com muitas instituições, **não é comparável**.
    O mais adequado é usar a soma do todo: **Conglomerado Prudencial**.

    ---

    ## **Informações de Capital Regulatório**

    **Capital Principal:** Parcela do capital de melhor qualidade e imediatamente disponível para absorver perdas (base regulatória usada para comparação com o RWA).

    **Capital Complementar:** Instrumentos de capital e dívida perpétuos elegíveis como patrimônio regulatório (complementam o Capital Principal na formação do Nível I).

    **Capital Nível II:** Parcela do capital composta por instrumentos subordinados, elegíveis como patrimônio regulatório, aptos a absorver perdas durante o funcionamento da instituição.

    **Patrimônio de Referência:** Montante de capital regulatório total formado pela soma do Patrimônio de Referência Nível I e do Capital Nível II (usado para comparação com o RWA).

    **RWA Crédito:** Parcela dos ativos ponderados pelo risco (RWA) referente à exposição ao risco de crédito (pode ser apurado pela abordagem padronizada ou por modelo interno IRB, quando aplicável).

    **RWA Mercado:** Parcela do RWA referente à exposição ao risco de mercado, composta pela soma das parcelas de risco cambial, commodities, juros/cupons, ações e CVA (ajuste de valor de crédito da contraparte em derivativos), entre outras componentes do relatório.

    **RWA Operacional:** Parcela do RWA referente à exposição ao risco operacional.

    **RWA Total:** Soma das parcelas de RWA de Crédito, Mercado, Operacional e (quando existir no relatório) a parcela relativa a serviços de pagamento.

    **Exposição Total:** Exposição total sem ponderação de risco (definição regulatória usada no cálculo da razão de alavancagem, conforme Circular 3.748/2015).

    **Índice de Capital Principal (CET1):** Relação entre Capital Principal e RWA Total (Capital Principal / RWA Total).

    **Índice de Capital Nível I:** Relação entre Patrimônio de Referência Nível I e RWA Total (Nível I / RWA Total).

    **Índice de Basileia:** Relação entre Patrimônio de Referência e RWA Total (Patrimônio de Referência / RWA Total).

    **Adicional de Capital Principal:** Requerimento de adicional de capital principal (ACP), apurado pela soma de ACP Conservação, ACP Contracíclico e ACP Sistêmico.

    ---

    ## **Variáveis de Balanço**

    **Ativo Total:** Padrão COSIF.

    **Carteira de Crédito:** Valor líquido, já descontada a PDD.

    **Títulos e Valores Mobiliários:** Títulos de Renda Fixa + Aplicação em COEs + Cotas de Fundos de Curto Prazo e Fundos de Investimentos, já descontados de Perda Incorrida, Perda Esperada e Ajuste a Valor Justo.

    **Passivo Exigível:** Passivo Total, incluindo Depósitos, Compromissadas, Outros Instrumentos de Dívida, Relações Interfinanceiras, Relações Interdependências, Derivativos, Provisões (Cíveis, Fiscais, Trabalhistas) e Outras Obrigações.

    **Captações:** Todos os tipos de passivos, exceto (i) Títulos de Dívida Elegíveis a Capital e (ii) Dívidas Subordinadas Elegíveis a Capital.

    **Patrimônio Líquido:** Padrão COSIF.

    **Lucro Líquido Acumulado YTD:** Lucro líquido acumulado entre janeiro do ano-competência até o final do semestre de referência (ex: 09/2025 refere-se ao lucro acumulado no ano de 2025, até 30/09/2025).

    ---

    ## **Índices e Percentuais**

    **Índice de Imobilização:** Ativo Permanente dividido pelo Patrimônio de Referência.

    ---

    ## **Métricas Calculadas**

    **ROE Ac. YTD an. (%):** (LL YTD × fator de anualização) ÷ PL Médio. PL Médio = (PL no período + PL em Dez do ano anterior) / 2. Fator: Mar=4, Jun=2, Set≈1.33, Dez=1. N/A se PL médio ≤ 0 ou dado faltante.

    **Crédito/PL (%):** Carteira de Crédito Líquida dividida pelo Patrimônio Líquido.

    **Crédito/Captações (%):** Carteira de Crédito Líquida dividida pelas Captações.

    **Crédito/Ativo (%):** Carteira de Crédito Líquida dividida pelo Ativo Total.

    **Desp PDD / NIM bruta (%):** Desp. PDD dividido pela NIM bruta (Rec. Crédito + Rec. Arrendamento Financeiro + Rec. Outras Operações c/ Características de Crédito). Fórmula: Desp. PDD / (Rec. Crédito + Rec. Arrendamento Financeiro + Rec. Outras Operações c/ Características de Crédito).

    **Desp PDD / Resultado Intermediação Fin. Bruto (%):** Desp. PDD dividido pelo Resultado de Intermediação Financeira Bruto. Fórmula: Desp. PDD / Resultado de Intermediação Financeira Bruto.

    **Desp Captação / Captação (%):** Desp. Captação anualizada dividida por Captações. Fórmula: (Desp. Captação * (12 / meses_do_período)) / Captações.
    """)
