"""
integration.py - Camada de compatibilidade com o sistema antigo

Fornece funcoes com assinaturas compativeis com o codigo existente no app1.py
para facilitar a migracao gradual.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable, Any

from .orchestrator import (
    obter_dados_capital,
    extrair_e_salvar_periodos,
    gerar_periodos_trimestrais,
    get_info,
)
from .storage import carregar_cache, salvar_cache, cache_existe, limpar_cache
from .config import CACHE_DATA_FILE, CACHE_METADATA_FILE, CAMPOS_CAPITAL, LOG_PREFIX

logger = logging.getLogger("capital_cache.integration")


def baixar_cache_capital_inicial() -> Tuple[bool, str]:
    """Baixa o cache de capital do GitHub se nao existir localmente.

    Compativel com a funcao antiga do app1.py.

    Returns:
        Tupla (sucesso, fonte) onde fonte pode ser 'local', 'github' ou 'nenhuma'
    """
    # Se cache local existe, usar ele
    if cache_existe():
        logger.info(f"{LOG_PREFIX} Cache local encontrado")
        return True, 'local'

    # Tentar obter dados (vai tentar GitHub)
    resultado = obter_dados_capital()

    if resultado.sucesso:
        if resultado.fonte == "cache_local":
            return True, 'local'
        elif resultado.fonte == "github":
            return True, 'github'
        elif resultado.fonte == "cache_local_obsoleto":
            return True, 'local'

    return False, 'nenhuma'


def carregar_cache_capital() -> Optional[Dict[str, Any]]:
    """Carrega o cache de capital do disco.

    Compativel com a funcao antiga do capital_extractor.py.
    Retorna dicionario {periodo: DataFrame} ou None.

    Returns:
        Dicionario com dados por periodo ou None se cache nao existir
    """
    df, metadata, msg = carregar_cache()

    if df is None:
        logger.warning(f"{LOG_PREFIX} {msg}")
        return None

    # Converter DataFrame unico em dicionario por periodo (formato antigo)
    if "Periodo" not in df.columns:
        logger.warning(f"{LOG_PREFIX} DataFrame sem coluna Periodo")
        return None

    dados_por_periodo = {}
    for periodo in df["Periodo"].unique():
        dados_por_periodo[str(periodo)] = df[df["Periodo"] == periodo].copy()

    logger.info(f"{LOG_PREFIX} Cache carregado: {len(dados_por_periodo)} periodos")
    return dados_por_periodo


def salvar_cache_capital(
    dados_periodos: Dict[str, Any],
    periodo_info: str,
    incremental: bool = True
) -> Dict[str, Any]:
    """Salva dados de capital no cache.

    Compativel com a funcao antiga do capital_extractor.py.

    Args:
        dados_periodos: Dicionario {periodo: DataFrame}
        periodo_info: String descrevendo o periodo extraido
        incremental: Se True, mescla com dados existentes

    Returns:
        Dicionario com informacoes do cache salvo
    """
    import pandas as pd

    # Se incremental, carregar dados existentes primeiro
    dados_finais = {}

    if incremental:
        dados_existentes = carregar_cache_capital()
        if dados_existentes:
            dados_finais = dados_existentes.copy()
            logger.info(f"{LOG_PREFIX} Merge: {len(dados_existentes)} periodos existentes")

    # Atualizar com novos dados
    for periodo, df in dados_periodos.items():
        dados_finais[str(periodo)] = df

    # Concatenar todos em um unico DataFrame
    if not dados_finais:
        return {
            "caminho": str(CACHE_DATA_FILE),
            "tamanho_formatado": "0 B",
            "n_periodos": 0,
            "sucesso": False,
        }

    df_concat = pd.concat(list(dados_finais.values()), ignore_index=True)

    # Salvar usando novo sistema
    sucesso, msg = salvar_cache(
        df_concat,
        fonte="bcb_api",
        periodos_extraidos=list(dados_finais.keys()),
        info_extra={"periodo_info": periodo_info, "incremental": incremental}
    )

    # Retornar info no formato esperado pelo codigo antigo
    info = get_info()

    tamanho = info.get("tamanho_bytes", 0)
    if tamanho >= 1024 * 1024:
        tamanho_fmt = f"{tamanho / 1024 / 1024:.1f} MB"
    elif tamanho >= 1024:
        tamanho_fmt = f"{tamanho / 1024:.1f} KB"
    else:
        tamanho_fmt = f"{tamanho} B"

    return {
        "caminho": str(CACHE_DATA_FILE),
        "tamanho_formatado": tamanho_fmt,
        "n_periodos": info.get("total_periodos", 0),
        "sucesso": sucesso,
        "mensagem": msg,
    }


def gerar_periodos_capital(
    ano_ini: int,
    mes_ini: str,
    ano_fin: int,
    mes_fin: str
) -> list:
    """Gera lista de periodos trimestrais.

    Compativel com a funcao antiga do capital_extractor.py.

    Args:
        ano_ini: Ano inicial
        mes_ini: Mes inicial ('03', '06', '09', '12')
        ano_fin: Ano final
        mes_fin: Mes final ('03', '06', '09', '12')

    Returns:
        Lista de periodos no formato "YYYYMM"
    """
    return gerar_periodos_trimestrais(
        ano_ini, int(mes_ini),
        ano_fin, int(mes_fin)
    )


def processar_todos_periodos_capital(
    periodos: list,
    dict_aliases: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    save_callback: Optional[Callable[[Dict, str], None]] = None,
    save_interval: int = 5,
) -> Optional[Dict[str, Any]]:
    """Processa multiplos periodos de capital.

    Compativel com a funcao antiga do capital_extractor.py.

    Args:
        periodos: Lista de periodos "YYYYMM"
        dict_aliases: Dicionario de aliases para instituicoes
        progress_callback: Funcao (i, total, periodo) chamada a cada periodo
        save_callback: Funcao (dados, info) para salvamento intermediario
        save_interval: Intervalo entre salvamentos intermediarios

    Returns:
        Dicionario {periodo: DataFrame} ou None em caso de falha
    """
    logger.info(f"{LOG_PREFIX} Iniciando extracao de {len(periodos)} periodos...")

    # Usar o orquestrador para extrair
    resultado = extrair_e_salvar_periodos(
        periodos=periodos,
        dict_aliases=dict_aliases,
        callback_progresso=progress_callback,
        incremental=True,
    )

    if not resultado.sucesso or resultado.df is None:
        logger.error(f"{LOG_PREFIX} Falha na extracao: {resultado.mensagem}")
        return None

    # Converter DataFrame em dicionario por periodo (formato antigo)
    df = resultado.df
    if "Periodo" not in df.columns:
        logger.error(f"{LOG_PREFIX} DataFrame sem coluna Periodo")
        return None

    dados_por_periodo = {}
    for periodo in df["Periodo"].unique():
        dados_por_periodo[str(periodo)] = df[df["Periodo"] == periodo].copy()

    logger.info(f"{LOG_PREFIX} Extracao concluida: {len(dados_por_periodo)} periodos")
    return dados_por_periodo


def get_capital_cache_info() -> Dict[str, Any]:
    """Retorna informacoes do cache de capital.

    Compativel com a funcao antiga do capital_extractor.py.

    Returns:
        Dicionario com informacoes do cache
    """
    info = get_info()

    # Formatar tamanho
    tamanho = info.get("tamanho_bytes", 0)
    if tamanho >= 1024 * 1024:
        tamanho_fmt = f"{tamanho / 1024 / 1024:.1f} MB"
    elif tamanho >= 1024:
        tamanho_fmt = f"{tamanho / 1024:.1f} KB"
    else:
        tamanho_fmt = f"{tamanho} B"

    return {
        "existe": info.get("existe", False),
        "caminho": str(CACHE_DATA_FILE),
        "tamanho_bytes": tamanho,
        "tamanho_formatado": tamanho_fmt,
        "n_periodos": info.get("total_periodos", 0),
        "data_extracao": info.get("timestamp_extracao"),
        "fonte": info.get("fonte"),
    }


def get_campos_capital_info() -> Dict[str, str]:
    """Retorna mapeamento de campos de capital.

    Compativel com a funcao antiga do capital_extractor.py.

    Returns:
        Dicionario {nome_original: nome_exibicao}
    """
    return CAMPOS_CAPITAL.copy()


def limpar_cache_capital() -> Tuple[bool, str]:
    """Remove cache de capital.

    Returns:
        Tupla (sucesso, mensagem)
    """
    return limpar_cache()
