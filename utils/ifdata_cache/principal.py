"""
principal.py - Cache de dados principais do IFData

Implementa cache para os relatorios 1-4 do IFData (dados gerais das instituicoes).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .base import BaseCache, CacheConfig, CacheResult

logger = logging.getLogger("ifdata_cache")

# Configuracao do cache principal
PRINCIPAL_CONFIG = CacheConfig(
    nome="principal",
    descricao="Dados gerais das instituicoes financeiras (Relatorios 1-4)",
    subdir="principal",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta/releases/download/v1.0-cache",
    max_idade_horas=168.0,  # 7 dias
    colunas_obrigatorias=["Periodo", "CodInst", "NomeInstituicao"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=None,  # Multiplos relatorios
)


class PrincipalCache(BaseCache):
    """Cache de dados principais do IFData."""

    def __init__(self, base_dir: Path):
        super().__init__(PRINCIPAL_CONFIG, base_dir)

        # URLs especificas
        self.github_data_url = f"{self.config.github_url_base}/dados_cache.pkl"
        self.github_info_url = f"{self.config.github_url_base}/cache_info.txt"

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados do GitHub Releases."""
        self._log("info", "Tentando baixar do GitHub...")

        try:
            # Baixar arquivo pickle do GitHub (formato antigo)
            response = requests.get(self.github_data_url, timeout=120)

            if response.status_code == 404:
                self._log("warning", "Cache nao encontrado no GitHub (404)")
                return CacheResult(
                    sucesso=False,
                    mensagem="Cache nao existe no GitHub",
                    fonte="nenhum"
                )

            response.raise_for_status()

            # Carregar pickle (formato antigo do sistema)
            import pickle
            import io

            dados_dict = pickle.load(io.BytesIO(response.content))

            # Converter de {periodo: DataFrame} para DataFrame unico
            if isinstance(dados_dict, dict):
                dfs = []
                for periodo, df in dados_dict.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if "Periodo" not in df.columns:
                            df = df.copy()
                            df["Periodo"] = str(periodo)
                        dfs.append(df)

                if dfs:
                    df_final = pd.concat(dfs, ignore_index=True)
                else:
                    return CacheResult(
                        sucesso=False,
                        mensagem="Arquivo do GitHub vazio ou invalido",
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
            self._log("error", f"Erro de rede ao baixar: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro de rede: {e}",
                fonte="nenhum"
            )
        except Exception as e:
            self._log("error", f"Erro ao processar dados: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro ao processar: {e}",
                fonte="nenhum"
            )

    def extrair_periodo(self, periodo: str, **kwargs) -> CacheResult:
        """Extrai dados de um periodo da API do BCB.

        Nota: Esta funcao e um placeholder. A extracao completa
        requer logica mais complexa que esta no app1.py (processar_periodo).
        """
        self._log("info", f"Extracao de periodo {periodo} nao implementada diretamente")
        return CacheResult(
            sucesso=False,
            mensagem="Use a funcao processar_periodo do app1.py para extracao",
            fonte="nenhum"
        )

    # =========================================================================
    # COMPATIBILIDADE COM SISTEMA ANTIGO
    # =========================================================================

    def carregar_formato_antigo(self) -> Optional[dict]:
        """Carrega e retorna no formato antigo {periodo: DataFrame}.

        Para compatibilidade com codigo existente no app1.py.
        """
        resultado = self.carregar()
        if not resultado.sucesso or resultado.dados is None:
            return None

        df = resultado.dados
        if "Periodo" not in df.columns:
            return None

        # Converter para formato antigo
        dados_dict = {}
        for periodo in df["Periodo"].unique():
            dados_dict[str(periodo)] = df[df["Periodo"] == periodo].copy()

        return dados_dict

    def salvar_formato_antigo(
        self,
        dados_dict: dict,
        fonte: str = "api",
        info_extra: Optional[dict] = None
    ) -> CacheResult:
        """Salva a partir do formato antigo {periodo: DataFrame}.

        Para compatibilidade com codigo existente no app1.py.
        """
        if not dados_dict:
            return CacheResult(
                sucesso=False,
                mensagem="Dicionario de dados vazio",
                fonte="nenhum"
            )

        # Converter para DataFrame unico
        dfs = []
        for periodo, df in dados_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df_copy = df.copy()
                if "Periodo" not in df_copy.columns:
                    df_copy["Periodo"] = str(periodo)
                dfs.append(df_copy)

        if not dfs:
            return CacheResult(
                sucesso=False,
                mensagem="Nenhum DataFrame valido no dicionario",
                fonte="nenhum"
            )

        df_final = pd.concat(dfs, ignore_index=True)
        return self.salvar_local(df_final, fonte=fonte, info_extra=info_extra)
