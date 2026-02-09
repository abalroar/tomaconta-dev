"""
derived_metrics.py - Cache e cálculo de métricas derivadas (formato LONG/TIDY)

Cria um cache separado para indicadores derivados calculados a partir de:
- DRE (Relatório 4)
- Resumo/Principal (Relatório 1) para Captações

Formato LONG/TIDY:
    Instituição | Período | Métrica | Valor | Unidade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .base import BaseCache, CacheConfig

logger = logging.getLogger("ifdata_cache")


DERIVED_METRICS: List[str] = []

DERIVED_METRICS_FORMAT: Dict[str, str] = {}

DERIVED_METRICS_FORMULAS: Dict[str, str] = {}


DRE_REQUIRED_COLUMNS = {
    "desp_pdd": "Resultado com Perda Esperada (f)",
    "rec_credito": "Rendas de Operações de Crédito (c)",
    "rec_arrendamento": "Rendas de Arrendamento Financeiro (d)",
    "rec_outras": "Rendas de Outras Operações com Características de Concessão de Crédito (e)",
    "rec_liquidez": "Rendas de Aplicações Interfinanceiras de Liquidez (a)",
    "rec_tvm": "Rendas de Títulos e Valores Mobiliários (b)",
    "desp_captacao": "Despesas de Captações (g)",
}


DERIVED_CACHE_CONFIG = CacheConfig(
    nome="derived_metrics",
    descricao="Métricas derivadas (DRE + Resumo)",
    subdir="derived_metrics",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base=None,
    max_idade_horas=None,
    colunas_obrigatorias=["Instituição", "Período", "Métrica", "Valor"],
)


class DerivedMetricsCache(BaseCache):
    """Cache dedicado para métricas derivadas."""

    def __init__(self, base_dir: Path):
        super().__init__(DERIVED_CACHE_CONFIG, base_dir)

    def baixar_remoto(self):
        return self._unsupported("Cache derivado não possui fonte remota")

    def extrair_periodo(self, periodo: str, **kwargs):
        return self._unsupported("Cache derivado não suporta extração direta")

    def _unsupported(self, mensagem: str):
        from .base import CacheResult

        return CacheResult(sucesso=False, mensagem=mensagem, fonte="nenhum")


@dataclass
class DerivedMetricsStats:
    denominador_zero_ou_nan: Dict[str, int]
    periodos_detectados: List[str]
    period_type: str
    total_registros: int


def _normalize_label(texto: str) -> str:
    if texto is None:
        return ""
    return (
        str(texto)
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


def _find_column(df: pd.DataFrame, label: str) -> Optional[str]:
    if label in df.columns:
        return label
    target = _normalize_label(label)
    for col in df.columns:
        if _normalize_label(col) == target:
            return col
    for col in df.columns:
        if target in _normalize_label(col):
            return col
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
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


def _parse_periodo(periodo_val: str) -> Tuple[Optional[int], Optional[int]]:
    """Retorna (ano, mes) a partir do formato '1/2025' ou '202503'."""
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


def _detect_period_type(periodos: Iterable[str]) -> str:
    for periodo in periodos:
        texto = str(periodo)
        if "/" in texto:
            return "trimestral"
        if texto.isdigit() and len(texto) in (6, 8):
            return "mensal"
    return "desconhecido"


def _safe_ratio(
    numerador: pd.Series,
    denominador: pd.Series,
    metric_label: str,
    contadores: Dict[str, int],
) -> pd.Series:
    denom_invalid = denominador.isna() | (denominador == 0)
    contadores[metric_label] = int(denom_invalid.sum())
    resultado = numerador / denominador
    resultado = resultado.mask(denom_invalid)
    return resultado


def _prepare_base_dre(df_dre: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    col_periodo = _find_column(df_dre, "Período") or _find_column(df_dre, "Periodo")
    col_inst = _find_column(df_dre, "Instituição") or _find_column(df_dre, "Instituicao")
    if col_periodo is None or col_inst is None:
        raise ValueError("Colunas de período ou instituição não encontradas no DRE")

    colunas = {"Instituição": col_inst, "Período": col_periodo}
    for key, label in DRE_REQUIRED_COLUMNS.items():
        col = _find_column(df_dre, label)
        if col:
            colunas[key] = col

    df_base = df_dre[list(dict.fromkeys(colunas.values()))].copy()
    df_base = df_base.rename(columns={col_inst: "Instituição", col_periodo: "Período"})

    for key, col in colunas.items():
        if key in ("Instituição", "Período"):
            continue
        df_base[col] = _coerce_numeric(df_base[col])

    return df_base, colunas


def _prepare_base_principal(df_principal: pd.DataFrame) -> pd.DataFrame:
    col_periodo = _find_column(df_principal, "Período") or _find_column(df_principal, "Periodo")
    col_inst = _find_column(df_principal, "Instituição") or _find_column(df_principal, "Instituicao")
    col_captacoes = _find_column(df_principal, "Captações") or _find_column(df_principal, "Captação")

    if col_periodo is None or col_inst is None or col_captacoes is None:
        raise ValueError("Colunas necessárias (Período/Instituição/Captações) não encontradas no principal")

    df_base = df_principal[[col_inst, col_periodo, col_captacoes]].copy()
    df_base = df_base.rename(
        columns={
            col_inst: "Instituição",
            col_periodo: "Período",
            col_captacoes: "Captações",
        }
    )
    df_base["Captações"] = _coerce_numeric(df_base["Captações"])
    return df_base


def build_derived_metrics(
    df_dre: pd.DataFrame,
    df_principal: pd.DataFrame,
) -> Tuple[pd.DataFrame, DerivedMetricsStats]:
    """Calcula métricas derivadas no formato LONG/TIDY."""
    period_type = _detect_period_type(
        df_dre["Período"].dropna().unique() if "Período" in df_dre.columns else []
    )
    df_final = pd.DataFrame(
        columns=["Instituição", "Período", "Métrica", "Valor", "Unidade"]
    )
    stats = DerivedMetricsStats(
        denominador_zero_ou_nan={},
        periodos_detectados=[],
        period_type=period_type,
        total_registros=0,
    )
    return df_final, stats


def load_derived_metrics_slice(
    cache: DerivedMetricsCache,
    periodos: Optional[Iterable[str]] = None,
    instituicoes: Optional[Iterable[str]] = None,
    metricas: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Carrega recortes do cache derivado sem carregar todo o parquet em RAM."""
    if not cache.arquivo_dados.exists() and not cache.arquivo_dados_pickle.exists():
        return pd.DataFrame()

    filtros = []
    if periodos:
        filtros.append(("Período", "in", list(periodos)))
    if instituicoes:
        filtros.append(("Instituição", "in", list(instituicoes)))
    if metricas:
        filtros.append(("Métrica", "in", list(metricas)))

    if cache.arquivo_dados.exists():
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(cache.arquivo_dados)
            if filtros:
                tabela = dataset.to_table(filter=_build_arrow_filter(filtros))
            else:
                tabela = dataset.to_table()
            df = tabela.to_pandas()
        except Exception as e:
            logger.warning(f"Falha ao ler parquet com filtros ({e}); usando fallback completo")
            df = pd.read_parquet(cache.arquivo_dados)
            df = _apply_filters(df, filtros)
    else:
        import pickle

        with open(cache.arquivo_dados_pickle, "rb") as f:
            df = pickle.load(f)
        df = _apply_filters(df, filtros)

    return df


def _apply_filters(df: pd.DataFrame, filtros: List[Tuple[str, str, list]]) -> pd.DataFrame:
    if not filtros:
        return df
    df_out = df
    for col, op, valores in filtros:
        if col not in df_out.columns:
            continue
        if op == "in":
            df_out = df_out[df_out[col].isin(valores)]
    return df_out


def _build_arrow_filter(filtros: List[Tuple[str, str, list]]):
    import pyarrow.dataset as ds

    filtro_final = None
    for col, op, valores in filtros:
        if op != "in":
            continue
        cond = ds.field(col).isin(valores)
        filtro_final = cond if filtro_final is None else filtro_final & cond
    return filtro_final
