"""
storage.py - Armazenamento e carregamento de cache

Responsavel por salvar e carregar dados do cache local.
Inclui validacoes explicitas para garantir integridade.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, Any

from .config import (
    CACHE_DIR,
    CACHE_DATA_FILE,
    CACHE_METADATA_FILE,
    LOG_PREFIX,
)

logger = logging.getLogger("capital_cache.storage")


class StorageError(Exception):
    """Erro durante operacao de storage."""
    pass


def _ensure_cache_dir() -> Path:
    """Garante que o diretorio de cache existe.

    Returns:
        Path do diretorio de cache

    Raises:
        StorageError: Se nao conseguir criar o diretorio
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not CACHE_DIR.exists():
            raise StorageError(f"Falha ao criar diretorio: {CACHE_DIR}")
        return CACHE_DIR
    except OSError as e:
        raise StorageError(f"Erro ao criar diretorio de cache: {e}")


def salvar_cache(
    df: pd.DataFrame,
    fonte: str,
    periodos_extraidos: Optional[list] = None,
    info_extra: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Salva dados no cache local com validacao.

    Args:
        df: DataFrame com dados de capital
        fonte: Origem dos dados ("bcb_api", "github", etc)
        periodos_extraidos: Lista de periodos contidos nos dados
        info_extra: Informacoes adicionais para metadata

    Returns:
        Tupla (sucesso, mensagem)

    Raises:
        StorageError: Se a validacao pos-gravacao falhar
    """
    logger.info(f"{LOG_PREFIX} Salvando cache ({len(df)} registros)...")

    # Validar entrada
    if df is None or df.empty:
        return False, "DataFrame vazio ou None, nada a salvar"

    # Garantir diretorio
    _ensure_cache_dir()

    # Preparar metadata
    timestamp = datetime.now().isoformat()
    periodos = periodos_extraidos or df["Periodo"].unique().tolist() if "Periodo" in df.columns else []

    metadata = {
        "timestamp_extracao": timestamp,
        "fonte": fonte,
        "total_registros": len(df),
        "total_periodos": len(periodos),
        "periodos": sorted([str(p) for p in periodos]),
        "colunas": list(df.columns),
    }
    if info_extra:
        metadata.update(info_extra)

    # Salvar dados em parquet
    try:
        df.to_parquet(CACHE_DATA_FILE, index=False, engine="pyarrow")
        logger.debug(f"{LOG_PREFIX} Dados salvos em {CACHE_DATA_FILE}")
    except Exception as e:
        return False, f"Erro ao salvar parquet: {e}"

    # Validar gravacao - CRITICO
    if not CACHE_DATA_FILE.exists():
        raise StorageError(f"CRITICO: Arquivo nao existe apos gravacao: {CACHE_DATA_FILE}")

    file_size = CACHE_DATA_FILE.stat().st_size
    if file_size == 0:
        raise StorageError(f"CRITICO: Arquivo vazio apos gravacao: {CACHE_DATA_FILE}")

    # Reler e validar
    try:
        df_check = pd.read_parquet(CACHE_DATA_FILE)
        if len(df_check) != len(df):
            raise StorageError(
                f"CRITICO: Contagem difere apos releitura. "
                f"Original: {len(df)}, Relido: {len(df_check)}"
            )
        logger.debug(f"{LOG_PREFIX} Validacao pos-gravacao OK: {len(df_check)} registros")
    except Exception as e:
        raise StorageError(f"CRITICO: Falha ao reler arquivo: {e}")

    # Salvar metadata
    try:
        with open(CACHE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"{LOG_PREFIX} Metadata salvo em {CACHE_METADATA_FILE}")
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Erro ao salvar metadata (dados salvos OK): {e}")

    msg = (
        f"Cache salvo com sucesso. "
        f"Arquivo: {CACHE_DATA_FILE.name} ({file_size:,} bytes), "
        f"{len(df)} registros, {len(periodos)} periodos"
    )
    logger.info(f"{LOG_PREFIX} {msg}")

    return True, msg


def carregar_cache() -> Tuple[Optional[pd.DataFrame], Optional[Dict], str]:
    """Carrega dados do cache local.

    Returns:
        Tupla (DataFrame, metadata, mensagem)
        DataFrame e metadata serao None se cache nao existir ou estiver invalido
    """
    logger.info(f"{LOG_PREFIX} Carregando cache local...")

    # Verificar existencia
    if not CACHE_DATA_FILE.exists():
        return None, None, f"Cache nao encontrado: {CACHE_DATA_FILE}"

    # Verificar tamanho
    file_size = CACHE_DATA_FILE.stat().st_size
    if file_size == 0:
        return None, None, f"Arquivo de cache vazio: {CACHE_DATA_FILE}"

    # Carregar dados
    try:
        df = pd.read_parquet(CACHE_DATA_FILE)
    except Exception as e:
        return None, None, f"Erro ao ler parquet: {e}"

    if df.empty:
        return None, None, "Cache carregado mas DataFrame vazio"

    # Carregar metadata
    metadata = None
    if CACHE_METADATA_FILE.exists():
        try:
            with open(CACHE_METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Erro ao ler metadata: {e}")

    msg = f"Cache carregado: {len(df)} registros"
    if metadata and "timestamp_extracao" in metadata:
        msg += f", extraido em {metadata['timestamp_extracao']}"
    if metadata and "fonte" in metadata:
        msg += f", fonte: {metadata['fonte']}"

    logger.info(f"{LOG_PREFIX} {msg}")

    return df, metadata, msg


def cache_existe() -> bool:
    """Verifica se existe cache local valido.

    Returns:
        True se cache existe e tem tamanho > 0
    """
    return CACHE_DATA_FILE.exists() and CACHE_DATA_FILE.stat().st_size > 0


def get_cache_info() -> Dict[str, Any]:
    """Retorna informacoes sobre o cache local.

    Returns:
        Dicionario com informacoes do cache
    """
    info = {
        "existe": False,
        "caminho_dados": str(CACHE_DATA_FILE),
        "caminho_metadata": str(CACHE_METADATA_FILE),
        "tamanho_bytes": 0,
        "timestamp_extracao": None,
        "fonte": None,
        "total_registros": None,
        "total_periodos": None,
        "periodos": [],
    }

    if not cache_existe():
        return info

    info["existe"] = True
    info["tamanho_bytes"] = CACHE_DATA_FILE.stat().st_size
    info["data_modificacao"] = datetime.fromtimestamp(
        CACHE_DATA_FILE.stat().st_mtime
    ).isoformat()

    # Carregar metadata se existir
    if CACHE_METADATA_FILE.exists():
        try:
            with open(CACHE_METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            info.update({
                "timestamp_extracao": metadata.get("timestamp_extracao"),
                "fonte": metadata.get("fonte"),
                "total_registros": metadata.get("total_registros"),
                "total_periodos": metadata.get("total_periodos"),
                "periodos": metadata.get("periodos", []),
            })
        except Exception:
            pass

    return info


def limpar_cache() -> Tuple[bool, str]:
    """Remove arquivos de cache local.

    Returns:
        Tupla (sucesso, mensagem)
    """
    removidos = []
    erros = []

    for arquivo in [CACHE_DATA_FILE, CACHE_METADATA_FILE]:
        if arquivo.exists():
            try:
                arquivo.unlink()
                removidos.append(arquivo.name)
            except Exception as e:
                erros.append(f"{arquivo.name}: {e}")

    if erros:
        return False, f"Erros ao remover: {', '.join(erros)}"

    if removidos:
        return True, f"Removidos: {', '.join(removidos)}"

    return True, "Nenhum arquivo de cache para remover"
