"""
extractor.py - Extracao de dados de capital

Responsavel por buscar dados da API do BCB (Olinda) ou do GitHub.
Inclui validacao de integridade dos dados extraidos.
"""

import requests
import pandas as pd
import time
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime

from .config import (
    BCB_BASE_URL,
    CAMPOS_CAPITAL,
    CAMPOS_CAPITAL_NORMALIZADO,
    HTTP_TIMEOUT,
    HTTP_RETRIES,
    HTTP_BACKOFF,
    REMOTE_CACHE_URL,
    REMOTE_METADATA_URL,
    LOG_PREFIX,
)

logger = logging.getLogger("capital_cache.extractor")


class ExtractionError(Exception):
    """Erro durante extracao de dados."""
    pass


class ValidationError(Exception):
    """Erro de validacao de dados extraidos."""
    pass


def _fetch_json(url: str, timeout: int = HTTP_TIMEOUT, retries: int = HTTP_RETRIES) -> Optional[dict]:
    """Faz requisicao HTTP com retry e backoff exponencial.

    Args:
        url: URL para requisicao
        timeout: Timeout em segundos
        retries: Numero de tentativas

    Returns:
        Dicionario com dados JSON ou None em caso de erro

    Raises:
        ExtractionError: Se todas as tentativas falharem
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            logger.debug(f"{LOG_PREFIX} Tentativa {attempt + 1}/{retries + 1}: {url[:80]}...")
            response = requests.get(url, timeout=timeout)

            if response.status_code == 429:
                wait_time = HTTP_BACKOFF * (2 ** attempt) * 2
                logger.warning(f"{LOG_PREFIX} Rate limit (429). Aguardando {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                wait_time = HTTP_BACKOFF * (2 ** attempt)
                logger.warning(f"{LOG_PREFIX} Erro servidor ({response.status_code}). Aguardando {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            record_count = len(data.get("value", [])) if isinstance(data, dict) else 0
            logger.debug(f"{LOG_PREFIX} Sucesso: {record_count} registros")

            return data

        except requests.Timeout as e:
            last_error = e
            wait_time = HTTP_BACKOFF * (2 ** attempt)
            logger.warning(f"{LOG_PREFIX} Timeout. Tentativa {attempt + 1}/{retries + 1}. Aguardando {wait_time:.1f}s...")
            time.sleep(wait_time)

        except requests.RequestException as e:
            last_error = e
            if attempt >= retries:
                raise ExtractionError(f"Erro HTTP apos {retries + 1} tentativas: {e}")
            wait_time = HTTP_BACKOFF * (2 ** attempt)
            logger.warning(f"{LOG_PREFIX} Erro: {e}. Aguardando {wait_time:.1f}s...")
            time.sleep(wait_time)

        except ValueError as e:
            last_error = e
            if attempt >= retries:
                raise ExtractionError(f"Erro ao decodificar JSON: {e}")
            time.sleep(HTTP_BACKOFF * (attempt + 1))

    if last_error:
        raise ExtractionError(f"Falha apos {retries + 1} tentativas: {last_error}")
    return None


def extrair_cadastro_bcb(ano_mes: str) -> pd.DataFrame:
    """Extrai cadastro de instituicoes do BCB para um periodo.

    Args:
        ano_mes: Periodo no formato "YYYYMM" (ex: "202312")

    Returns:
        DataFrame com colunas: CodInst, NomeInstituicao

    Raises:
        ExtractionError: Se a extracao falhar
        ValidationError: Se os dados estiverem invalidos
    """
    url = (
        f"{BCB_BASE_URL}/IfDataCadastro?"
        f"$filter=AnoMes eq '{ano_mes}' and TipoInstituicao eq 'CI'"
        f"&$select=CodInst,NomeInstituicao"
        f"&$format=json"
    )

    logger.info(f"{LOG_PREFIX} Extraindo cadastro BCB para {ano_mes}...")

    data = _fetch_json(url)
    if not data or "value" not in data:
        raise ExtractionError(f"Resposta invalida do BCB para cadastro {ano_mes}")

    df = pd.DataFrame(data["value"])

    if df.empty:
        raise ValidationError(f"Cadastro vazio para periodo {ano_mes}")

    required_cols = ["CodInst", "NomeInstituicao"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Colunas ausentes no cadastro: {missing}")

    logger.info(f"{LOG_PREFIX} Cadastro extraido: {len(df)} instituicoes")
    return df[required_cols]


def extrair_valores_capital_bcb(ano_mes: str) -> pd.DataFrame:
    """Extrai valores de capital do BCB (Relatorio Tipo 5).

    Args:
        ano_mes: Periodo no formato "YYYYMM" (ex: "202312")

    Returns:
        DataFrame com colunas: CodInst, NomeColuna, Valor

    Raises:
        ExtractionError: Se a extracao falhar
        ValidationError: Se os dados estiverem invalidos
    """
    url = (
        f"{BCB_BASE_URL}/IfDataValores?"
        f"$filter=AnoMes eq '{ano_mes}' and Relatorio eq '5' and TipoInstituicao eq 'CI'"
        f"&$select=CodInst,NomeColuna,Valor"
        f"&$format=json"
    )

    logger.info(f"{LOG_PREFIX} Extraindo valores de capital BCB para {ano_mes}...")

    data = _fetch_json(url)
    if not data or "value" not in data:
        raise ExtractionError(f"Resposta invalida do BCB para valores {ano_mes}")

    df = pd.DataFrame(data["value"])

    if df.empty:
        raise ValidationError(f"Valores de capital vazios para periodo {ano_mes}")

    required_cols = ["CodInst", "NomeColuna", "Valor"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Colunas ausentes nos valores: {missing}")

    logger.info(f"{LOG_PREFIX} Valores extraidos: {len(df)} registros")
    return df[required_cols]


def processar_periodo_capital(
    ano_mes: str,
    dict_aliases: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Processa dados de capital para um periodo completo.

    Combina cadastro + valores, filtra campos relevantes, pivota e renomeia.

    Args:
        ano_mes: Periodo no formato "YYYYMM"
        dict_aliases: Dicionario de aliases para renomear instituicoes

    Returns:
        DataFrame processado com uma linha por instituicao e colunas de capital

    Raises:
        ExtractionError: Se a extracao falhar
        ValidationError: Se os dados estiverem invalidos
    """
    logger.info(f"{LOG_PREFIX} Processando periodo {ano_mes}...")

    # Extrair dados
    df_cadastro = extrair_cadastro_bcb(ano_mes)
    df_valores = extrair_valores_capital_bcb(ano_mes)

    # Normalizar nomes de colunas (remover quebras de linha)
    df_valores["NomeColuna"] = df_valores["NomeColuna"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Filtrar apenas campos de capital desejados
    campos_normalizado = {" ".join(k.split()): v for k, v in CAMPOS_CAPITAL.items()}
    df_valores = df_valores[df_valores["NomeColuna"].isin(campos_normalizado.keys())]

    if df_valores.empty:
        raise ValidationError(f"Nenhum campo de capital encontrado para {ano_mes}")

    # Mapear para nomes de exibicao
    df_valores["NomeExibicao"] = df_valores["NomeColuna"].map(campos_normalizado)

    # Pivotar: uma linha por instituicao, uma coluna por metrica
    df_pivot = df_valores.pivot_table(
        index="CodInst",
        columns="NomeExibicao",
        values="Valor",
        aggfunc="first"
    ).reset_index()

    # Merge com cadastro para obter nomes
    df_final = df_pivot.merge(df_cadastro, on="CodInst", how="left")

    # Aplicar aliases se fornecidos
    if dict_aliases:
        df_final["NomeInstituicao"] = df_final["NomeInstituicao"].replace(dict_aliases)

    # Adicionar coluna de periodo
    df_final["Periodo"] = ano_mes

    # Reordenar colunas
    cols_first = ["Periodo", "CodInst", "NomeInstituicao"]
    cols_rest = [c for c in df_final.columns if c not in cols_first]
    df_final = df_final[cols_first + sorted(cols_rest)]

    logger.info(f"{LOG_PREFIX} Periodo {ano_mes} processado: {len(df_final)} instituicoes, {len(cols_rest)} metricas")

    return df_final


def baixar_cache_github() -> Tuple[Optional[pd.DataFrame], Optional[dict], str]:
    """Baixa cache de capital do GitHub Releases.

    Returns:
        Tupla (DataFrame, metadata_dict, mensagem)
        DataFrame e metadata serao None se download falhar

    Note:
        Nao levanta excecao em caso de falha, apenas retorna None
    """
    logger.info(f"{LOG_PREFIX} Tentando baixar cache do GitHub...")

    try:
        # Baixar dados
        response = requests.get(REMOTE_CACHE_URL, timeout=HTTP_TIMEOUT)
        if response.status_code != 200:
            return None, None, f"HTTP {response.status_code} ao baixar dados"

        # Salvar temporariamente para ler com pandas
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        df = pd.read_parquet(tmp_path)

        # Limpar arquivo temporario
        import os
        os.unlink(tmp_path)

        # Baixar metadata
        metadata = None
        try:
            resp_meta = requests.get(REMOTE_METADATA_URL, timeout=30)
            if resp_meta.status_code == 200:
                metadata = resp_meta.json()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Falha ao baixar metadata: {e}")

        logger.info(f"{LOG_PREFIX} Cache baixado do GitHub: {len(df)} registros")
        return df, metadata, "Cache baixado com sucesso do GitHub"

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Falha ao baixar do GitHub: {e}")
        return None, None, f"Erro ao baixar do GitHub: {e}"


def validar_dataframe(df: pd.DataFrame, contexto: str = "") -> Tuple[bool, str]:
    """Valida integridade de um DataFrame de capital.

    Args:
        df: DataFrame a validar
        contexto: Descricao do contexto para mensagens de erro

    Returns:
        Tupla (valido, mensagem)
    """
    if df is None:
        return False, f"{contexto}: DataFrame e None"

    if df.empty:
        return False, f"{contexto}: DataFrame vazio"

    required_cols = ["Periodo", "CodInst", "NomeInstituicao"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"{contexto}: Colunas obrigatorias ausentes: {missing}"

    if df["CodInst"].isna().all():
        return False, f"{contexto}: Todos os CodInst sao nulos"

    if df["Periodo"].isna().all():
        return False, f"{contexto}: Todos os Periodos sao nulos"

    return True, f"{contexto}: Validacao OK ({len(df)} registros, {df['Periodo'].nunique()} periodos)"
