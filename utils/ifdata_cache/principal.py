"""
principal.py - Cache de dados principais do IFData (Relatório 1 - Resumo)

Implementa cache para o Relatório 1 do IFData com variáveis selecionadas.
Produz dados no formato exato que os gráficos do app1.py esperam.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .base import BaseCache, CacheConfig, CacheResult

logger = logging.getLogger("ifdata_cache")

# Configuração do cache principal
PRINCIPAL_CONFIG = CacheConfig(
    nome="principal",
    descricao="Dados gerais das instituições (Relatório 1 - Resumo)",
    subdir="principal",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta/releases/download/v1.0-cache",
    max_idade_horas=168.0,  # 7 dias
    colunas_obrigatorias=["Período"],  # Formato de exibição
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=1,
)


class PrincipalCache(BaseCache):
    """Cache de dados principais do IFData (Resumo).

    Produz dados com:
    - Coluna "Instituição" (nome da instituição)
    - Coluna "Período" no formato "1/2024" (trimestre/ano)
    - Métricas financeiras no formato esperado pelos gráficos
    """

    def __init__(self, base_dir: Path):
        super().__init__(PRINCIPAL_CONFIG, base_dir)
        self.github_data_url = f"{self.config.github_url_base}/dados_cache.pkl"

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados do GitHub Releases."""
        self._log("info", "Tentando baixar do GitHub...")

        try:
            response = requests.get(self.github_data_url, timeout=120)

            if response.status_code == 404:
                self._log("warning", "Cache não encontrado no GitHub (404)")
                return CacheResult(
                    sucesso=False,
                    mensagem="Cache não existe no GitHub",
                    fonte="nenhum"
                )

            response.raise_for_status()

            import pickle
            import io

            dados_dict = pickle.load(io.BytesIO(response.content))

            # Converter de {periodo: DataFrame} para DataFrame único
            if isinstance(dados_dict, dict):
                dfs = []
                for periodo, df in dados_dict.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Garantir coluna Período
                        if "Período" not in df.columns:
                            df = df.copy()
                            df["Período"] = str(periodo)
                        dfs.append(df)

                if dfs:
                    df_final = pd.concat(dfs, ignore_index=True)
                else:
                    return CacheResult(
                        sucesso=False,
                        mensagem="Arquivo do GitHub vazio",
                        fonte="nenhum"
                    )
            elif isinstance(dados_dict, pd.DataFrame):
                df_final = dados_dict
            else:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Formato inesperado: {type(dados_dict)}",
                    fonte="nenhum"
                )

            self._log("info", f"Baixado do GitHub: {len(df_final)} registros")

            return CacheResult(
                sucesso=True,
                mensagem=f"Baixado do GitHub: {len(df_final)} registros",
                dados=df_final,
                fonte="github"
            )

        except requests.RequestException as e:
            self._log("error", f"Erro de rede: {e}")
            return CacheResult(sucesso=False, mensagem=f"Erro de rede: {e}", fonte="nenhum")
        except Exception as e:
            self._log("error", f"Erro: {e}")
            return CacheResult(sucesso=False, mensagem=f"Erro: {e}", fonte="nenhum")

    def extrair_periodo(
        self,
        periodo: str,
        dict_aliases: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> CacheResult:
        """Extrai dados de um período da API do BCB.

        Usa o extrator autônomo para produzir dados no formato dos gráficos.

        Args:
            periodo: Período no formato YYYYMM (ex: "202312")
            dict_aliases: Dicionário de aliases para instituições

        Returns:
            CacheResult com DataFrame ou erro
        """
        self._log("info", f"Extraindo período {periodo}...")

        try:
            # Usar extrator autônomo
            from .extractor import extrair_resumo

            df = extrair_resumo(periodo, dict_aliases)

            if df is None or df.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Sem dados para período {periodo}",
                    fonte="nenhum"
                )

            self._log("info", f"Período {periodo}: {len(df)} instituições")

            return CacheResult(
                sucesso=True,
                mensagem=f"Extraído {periodo}: {len(df)} registros",
                dados=df,
                metadata={
                    "periodo": periodo,
                    "n_registros": len(df),
                    "colunas": list(df.columns)
                },
                fonte="api"
            )

        except Exception as e:
            self._log("error", f"Erro ao extrair {periodo}: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro: {e}",
                fonte="nenhum"
            )

    # =========================================================================
    # COMPATIBILIDADE COM SISTEMA ANTIGO
    # =========================================================================

    def carregar_formato_antigo(self) -> Optional[dict]:
        """Carrega e retorna no formato antigo {periodo: DataFrame}.

        O período é no formato de exibição ("1/2024").
        """
        resultado = self.carregar()
        if not resultado.sucesso or resultado.dados is None:
            return None

        df = resultado.dados
        if "Período" not in df.columns:
            return None

        dados_dict = {}
        for periodo in df["Período"].unique():
            dados_dict[str(periodo)] = df[df["Período"] == periodo].copy()

        return dados_dict

    def salvar_formato_antigo(
        self,
        dados_dict: dict,
        fonte: str = "api",
        info_extra: Optional[dict] = None
    ) -> CacheResult:
        """Salva a partir do formato antigo {periodo: DataFrame}."""
        if not dados_dict:
            return CacheResult(
                sucesso=False,
                mensagem="Dicionário vazio",
                fonte="nenhum"
            )

        dfs = []
        for periodo, df in dados_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df_copy = df.copy()
                if "Período" not in df_copy.columns:
                    df_copy["Período"] = str(periodo)
                dfs.append(df_copy)

        if not dfs:
            return CacheResult(
                sucesso=False,
                mensagem="Nenhum DataFrame válido",
                fonte="nenhum"
            )

        df_final = pd.concat(dfs, ignore_index=True)
        return self.salvar_local(df_final, fonte=fonte, info_extra=info_extra)
