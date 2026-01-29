"""
orchestrator.py - Orquestrador do sistema de cache de capital

Coordena o fluxo entre cache local e fontes remotas (GitHub, BCB).
Decide quando usar cache ou quando extrair novos dados.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Callable, Any

from .config import CACHE_MAX_AGE_HOURS, LOG_PREFIX
from .storage import (
    carregar_cache,
    salvar_cache,
    cache_existe,
    get_cache_info,
    limpar_cache,
)
from .extractor import (
    processar_periodo_capital,
    baixar_cache_github,
    validar_dataframe,
    ExtractionError,
    ValidationError,
)

logger = logging.getLogger("capital_cache.orchestrator")


class CacheResult:
    """Resultado de operacao de cache com metadados."""

    def __init__(
        self,
        df: Optional[pd.DataFrame],
        fonte: str,
        sucesso: bool,
        mensagem: str,
        metadata: Optional[Dict] = None
    ):
        self.df = df
        self.fonte = fonte  # "cache_local", "github", "bcb_api", "nenhum"
        self.sucesso = sucesso
        self.mensagem = mensagem
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        status = "OK" if self.sucesso else "FALHA"
        registros = len(self.df) if self.df is not None else 0
        return f"CacheResult({status}, fonte={self.fonte}, registros={registros})"


def _cache_esta_fresco(max_age_hours: int = CACHE_MAX_AGE_HOURS) -> Tuple[bool, str]:
    """Verifica se o cache local esta dentro da validade.

    Args:
        max_age_hours: Idade maxima em horas

    Returns:
        Tupla (fresco, mensagem)
    """
    info = get_cache_info()

    if not info["existe"]:
        return False, "Cache nao existe"

    timestamp_str = info.get("timestamp_extracao")
    if not timestamp_str:
        # Sem timestamp, usar data de modificacao do arquivo
        data_mod_str = info.get("data_modificacao")
        if not data_mod_str:
            return False, "Sem informacao de data do cache"
        timestamp = datetime.fromisoformat(data_mod_str)
    else:
        timestamp = datetime.fromisoformat(timestamp_str)

    idade = datetime.now() - timestamp
    idade_horas = idade.total_seconds() / 3600

    if idade_horas <= max_age_hours:
        return True, f"Cache fresco (idade: {idade_horas:.1f}h, limite: {max_age_hours}h)"
    else:
        return False, f"Cache obsoleto (idade: {idade_horas:.1f}h, limite: {max_age_hours}h)"


def obter_dados_capital(
    forcar_extracao: bool = False,
    forcar_github: bool = False,
    max_age_hours: Optional[int] = None,
    dict_aliases: Optional[Dict[str, str]] = None,
) -> CacheResult:
    """Obtem dados de capital usando a melhor fonte disponivel.

    Ordem de prioridade:
    1. Cache local (se fresco e nao forcado)
    2. GitHub (se disponivel)
    3. Retorna erro (nao extrai do BCB automaticamente)

    Args:
        forcar_extracao: Se True, ignora cache e vai direto para GitHub
        forcar_github: Se True, baixa do GitHub mesmo com cache valido
        max_age_hours: Idade maxima do cache em horas (None = usar padrao)
        dict_aliases: Aliases para renomear instituicoes

    Returns:
        CacheResult com dados e metadados
    """
    max_age = max_age_hours if max_age_hours is not None else CACHE_MAX_AGE_HOURS

    # 1. Tentar cache local (se nao forcar)
    if not forcar_extracao and not forcar_github:
        fresco, msg_fresco = _cache_esta_fresco(max_age)

        if fresco:
            df, metadata, msg_load = carregar_cache()
            if df is not None:
                valido, msg_val = validar_dataframe(df, "cache_local")
                if valido:
                    logger.info(f"{LOG_PREFIX} Usando cache local: {msg_fresco}")
                    return CacheResult(
                        df=df,
                        fonte="cache_local",
                        sucesso=True,
                        mensagem=f"Dados carregados do cache local. {msg_fresco}",
                        metadata=metadata
                    )
                else:
                    logger.warning(f"{LOG_PREFIX} Cache invalido: {msg_val}")
        else:
            logger.info(f"{LOG_PREFIX} {msg_fresco}")

    # 2. Tentar GitHub
    logger.info(f"{LOG_PREFIX} Tentando baixar do GitHub...")
    df_github, metadata_github, msg_github = baixar_cache_github()

    if df_github is not None:
        valido, msg_val = validar_dataframe(df_github, "github")
        if valido:
            # Aplicar aliases se fornecidos
            if dict_aliases and "NomeInstituicao" in df_github.columns:
                df_github["NomeInstituicao"] = df_github["NomeInstituicao"].replace(dict_aliases)

            # Salvar no cache local
            sucesso_save, msg_save = salvar_cache(
                df_github,
                fonte="github",
                info_extra={"github_metadata": metadata_github}
            )

            if sucesso_save:
                logger.info(f"{LOG_PREFIX} Cache atualizado do GitHub: {msg_save}")
            else:
                logger.warning(f"{LOG_PREFIX} Falha ao salvar cache do GitHub: {msg_save}")

            return CacheResult(
                df=df_github,
                fonte="github",
                sucesso=True,
                mensagem=f"Dados baixados do GitHub. {msg_github}",
                metadata=metadata_github
            )
        else:
            logger.warning(f"{LOG_PREFIX} Dados do GitHub invalidos: {msg_val}")
    else:
        logger.warning(f"{LOG_PREFIX} Falha ao baixar do GitHub: {msg_github}")

    # 3. Se tudo falhar, tentar usar cache local mesmo obsoleto
    if cache_existe():
        df, metadata, msg_load = carregar_cache()
        if df is not None:
            logger.warning(f"{LOG_PREFIX} Usando cache local obsoleto como fallback")
            return CacheResult(
                df=df,
                fonte="cache_local_obsoleto",
                sucesso=True,
                mensagem=f"AVISO: Usando cache local obsoleto. GitHub indisponivel. {msg_load}",
                metadata=metadata
            )

    # 4. Nenhuma fonte disponivel
    return CacheResult(
        df=None,
        fonte="nenhum",
        sucesso=False,
        mensagem="Nenhuma fonte de dados disponivel. Cache local nao existe e GitHub falhou."
    )


def extrair_e_salvar_periodos(
    periodos: List[str],
    dict_aliases: Optional[Dict[str, str]] = None,
    callback_progresso: Optional[Callable[[int, int, str], None]] = None,
    callback_erro: Optional[Callable[[str, Exception], None]] = None,
    incremental: bool = True,
) -> CacheResult:
    """Extrai dados do BCB para multiplos periodos e salva no cache.

    Args:
        periodos: Lista de periodos no formato "YYYYMM"
        dict_aliases: Aliases para renomear instituicoes
        callback_progresso: Funcao chamada com (atual, total, periodo)
        callback_erro: Funcao chamada com (periodo, excecao) em caso de erro
        incremental: Se True, mescla com dados existentes

    Returns:
        CacheResult com dados extraidos
    """
    logger.info(f"{LOG_PREFIX} Iniciando extracao de {len(periodos)} periodos...")

    # Carregar dados existentes se incremental
    dados_existentes = {}
    if incremental and cache_existe():
        df_exist, _, _ = carregar_cache()
        if df_exist is not None and "Periodo" in df_exist.columns:
            for periodo in df_exist["Periodo"].unique():
                dados_existentes[str(periodo)] = df_exist[df_exist["Periodo"] == periodo]
            logger.info(f"{LOG_PREFIX} Carregados {len(dados_existentes)} periodos existentes para merge")

    # Extrair novos periodos
    dados_novos = {}
    erros = []

    for i, periodo in enumerate(periodos):
        if callback_progresso:
            callback_progresso(i + 1, len(periodos), periodo)

        try:
            df_periodo = processar_periodo_capital(periodo, dict_aliases)
            dados_novos[periodo] = df_periodo
            logger.info(f"{LOG_PREFIX} Periodo {periodo} extraido: {len(df_periodo)} registros")

        except (ExtractionError, ValidationError) as e:
            logger.error(f"{LOG_PREFIX} Erro no periodo {periodo}: {e}")
            erros.append((periodo, str(e)))
            if callback_erro:
                callback_erro(periodo, e)

    if not dados_novos:
        return CacheResult(
            df=None,
            fonte="bcb_api",
            sucesso=False,
            mensagem=f"Nenhum periodo extraido com sucesso. Erros: {erros}"
        )

    # Mesclar com existentes
    dados_finais = dados_existentes.copy()
    novos_count = 0
    atualizados_count = 0

    for periodo, df in dados_novos.items():
        if periodo in dados_finais:
            atualizados_count += 1
        else:
            novos_count += 1
        dados_finais[periodo] = df

    # Concatenar todos os periodos
    df_final = pd.concat(list(dados_finais.values()), ignore_index=True)

    # Salvar
    sucesso, msg_save = salvar_cache(
        df_final,
        fonte="bcb_api",
        periodos_extraidos=list(dados_finais.keys()),
        info_extra={
            "novos_periodos": novos_count,
            "atualizados_periodos": atualizados_count,
            "erros": erros,
        }
    )

    if not sucesso:
        return CacheResult(
            df=df_final,
            fonte="bcb_api",
            sucesso=False,
            mensagem=f"Extracao OK mas falha ao salvar: {msg_save}"
        )

    msg = (
        f"Extracao concluida. "
        f"{novos_count} novos, {atualizados_count} atualizados, "
        f"{len(dados_finais)} total, {len(erros)} erros"
    )

    return CacheResult(
        df=df_final,
        fonte="bcb_api",
        sucesso=True,
        mensagem=msg,
        metadata={
            "novos": novos_count,
            "atualizados": atualizados_count,
            "total": len(dados_finais),
            "erros": erros,
        }
    )


def gerar_periodos_trimestrais(
    ano_inicio: int,
    mes_inicio: int,
    ano_fim: int,
    mes_fim: int
) -> List[str]:
    """Gera lista de periodos trimestrais (03, 06, 09, 12).

    Args:
        ano_inicio: Ano inicial
        mes_inicio: Mes inicial (sera ajustado para trimestre)
        ano_fim: Ano final
        mes_fim: Mes final (sera ajustado para trimestre)

    Returns:
        Lista de periodos no formato "YYYYMM"
    """
    meses_trimestrais = [3, 6, 9, 12]

    # Ajustar mes inicial para proximo trimestre
    mes_ini_ajustado = min([m for m in meses_trimestrais if m >= mes_inicio], default=12)
    if mes_ini_ajustado < mes_inicio:
        ano_inicio += 1
        mes_ini_ajustado = 3

    # Ajustar mes final para trimestre anterior ou igual
    mes_fim_ajustado = max([m for m in meses_trimestrais if m <= mes_fim], default=3)

    periodos = []
    ano_atual = ano_inicio
    mes_atual = mes_ini_ajustado

    while (ano_atual < ano_fim) or (ano_atual == ano_fim and mes_atual <= mes_fim_ajustado):
        periodos.append(f"{ano_atual}{mes_atual:02d}")

        # Proximo trimestre
        idx = meses_trimestrais.index(mes_atual)
        if idx == 3:  # Dezembro
            ano_atual += 1
            mes_atual = 3
        else:
            mes_atual = meses_trimestrais[idx + 1]

    return periodos


# Funcoes de conveniencia para uso externo
def get_info() -> Dict[str, Any]:
    """Retorna informacoes do cache."""
    return get_cache_info()


def limpar() -> Tuple[bool, str]:
    """Limpa o cache local."""
    return limpar_cache()
