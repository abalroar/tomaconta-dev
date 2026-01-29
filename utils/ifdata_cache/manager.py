"""
manager.py - Gerenciador central de caches

Coordena multiplos tipos de cache e fornece interface unificada.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCache, CacheResult

logger = logging.getLogger("ifdata_cache")


class CacheManager:
    """Gerenciador central de todos os caches do IFData."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Args:
            base_dir: Diretorio base do projeto. Se None, detecta automaticamente.
        """
        if base_dir is None:
            # Detectar diretorio base (3 niveis acima: utils/ifdata_cache/manager.py)
            base_dir = Path(__file__).parent.parent.parent.resolve()

        self.base_dir = base_dir
        self._caches: Dict[str, BaseCache] = {}
        self._registrar_caches_padrao()

    def _registrar_caches_padrao(self):
        """Registra os caches padrao do sistema."""
        # Importar aqui para evitar dependencia circular
        from .principal import PrincipalCache
        from .capital import CapitalCache

        self.registrar(PrincipalCache(self.base_dir))
        self.registrar(CapitalCache(self.base_dir))

    def registrar(self, cache: BaseCache):
        """Registra um novo tipo de cache.

        Args:
            cache: Instancia de cache a registrar
        """
        nome = cache.config.nome
        self._caches[nome] = cache
        logger.debug(f"[CACHE_MANAGER] Registrado: {nome}")

    def listar_caches(self) -> List[str]:
        """Lista nomes de todos os caches registrados."""
        return list(self._caches.keys())

    def get_cache(self, tipo: str) -> Optional[BaseCache]:
        """Retorna instancia de cache pelo tipo."""
        return self._caches.get(tipo)

    def carregar(self, tipo: str, forcar_remoto: bool = False) -> CacheResult:
        """Carrega dados de um tipo de cache.

        Args:
            tipo: Nome do tipo de cache (ex: "principal", "capital")
            forcar_remoto: Se True, ignora cache local

        Returns:
            CacheResult com dados ou erro
        """
        cache = self._caches.get(tipo)
        if cache is None:
            return CacheResult(
                sucesso=False,
                mensagem=f"Tipo de cache desconhecido: {tipo}. Disponiveis: {self.listar_caches()}",
                fonte="nenhum"
            )

        return cache.carregar(forcar_remoto=forcar_remoto)

    def salvar(
        self,
        tipo: str,
        dados,
        fonte: str = "desconhecida",
        **kwargs
    ) -> CacheResult:
        """Salva dados em um tipo de cache.

        Args:
            tipo: Nome do tipo de cache
            dados: DataFrame a salvar
            fonte: Identificador da fonte dos dados
            **kwargs: Argumentos extras passados para salvar_local

        Returns:
            CacheResult
        """
        cache = self._caches.get(tipo)
        if cache is None:
            return CacheResult(
                sucesso=False,
                mensagem=f"Tipo de cache desconhecido: {tipo}",
                fonte="nenhum"
            )

        return cache.salvar_local(dados, fonte=fonte, info_extra=kwargs.get("info_extra"))

    def info(self, tipo: str = None) -> Dict[str, Any]:
        """Retorna informacoes sobre cache(s).

        Args:
            tipo: Nome do tipo de cache. Se None, retorna info de todos.

        Returns:
            Dicionario com informacoes
        """
        if tipo is not None:
            cache = self._caches.get(tipo)
            if cache is None:
                return {"erro": f"Tipo desconhecido: {tipo}"}
            return cache.get_info()

        # Retornar info de todos
        return {nome: cache.get_info() for nome, cache in self._caches.items()}

    def limpar(self, tipo: str = None) -> CacheResult:
        """Limpa cache(s).

        Args:
            tipo: Nome do tipo de cache. Se None, limpa todos.

        Returns:
            CacheResult
        """
        if tipo is not None:
            cache = self._caches.get(tipo)
            if cache is None:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Tipo desconhecido: {tipo}",
                    fonte="nenhum"
                )
            return cache.limpar_local()

        # Limpar todos
        resultados = []
        for nome, cache in self._caches.items():
            resultado = cache.limpar_local()
            resultados.append(f"{nome}: {resultado.mensagem}")

        return CacheResult(
            sucesso=True,
            mensagem="; ".join(resultados),
            fonte="nenhum"
        )

    def extrair_periodos(
        self,
        tipo: str,
        periodos: List[str],
        callback_progresso=None,
        **kwargs
    ) -> CacheResult:
        """Extrai dados de multiplos periodos da API.

        Args:
            tipo: Nome do tipo de cache
            periodos: Lista de periodos "YYYYMM"
            callback_progresso: Funcao (i, total, periodo) chamada a cada periodo
            **kwargs: Argumentos extras para extracao

        Returns:
            CacheResult com todos os dados
        """
        import pandas as pd

        cache = self._caches.get(tipo)
        if cache is None:
            return CacheResult(
                sucesso=False,
                mensagem=f"Tipo desconhecido: {tipo}",
                fonte="nenhum"
            )

        dados_todos = []
        erros = []

        for i, periodo in enumerate(periodos):
            if callback_progresso:
                callback_progresso(i, len(periodos), periodo)

            resultado = cache.extrair_periodo(periodo, **kwargs)

            if resultado.sucesso and resultado.dados is not None:
                dados_todos.append(resultado.dados)
            else:
                erros.append(f"{periodo}: {resultado.mensagem}")
                logger.warning(f"[CACHE:{tipo.upper()}] Erro em {periodo}: {resultado.mensagem}")

        if not dados_todos:
            return CacheResult(
                sucesso=False,
                mensagem=f"Nenhum periodo extraido. Erros: {'; '.join(erros[:3])}",
                fonte="nenhum"
            )

        # Concatenar todos os periodos
        df_final = pd.concat(dados_todos, ignore_index=True)

        logger.info(f"[CACHE:{tipo.upper()}] Extraidos {len(dados_todos)}/{len(periodos)} periodos, {len(df_final)} registros")

        return CacheResult(
            sucesso=True,
            mensagem=f"Extraidos {len(dados_todos)} periodos, {len(df_final)} registros",
            dados=df_final,
            metadata={"periodos_extraidos": len(dados_todos), "erros": erros},
            fonte="api"
        )

    def existe(self, tipo: str) -> bool:
        """Verifica se cache de um tipo existe."""
        cache = self._caches.get(tipo)
        if cache is None:
            return False
        return cache.existe()

    def cache_valido(self, tipo: str) -> tuple:
        """Verifica se cache de um tipo esta valido."""
        cache = self._caches.get(tipo)
        if cache is None:
            return False, f"Tipo desconhecido: {tipo}"
        return cache.cache_valido()
