"""
capital.py - Cache de dados de capital regulatório (Relatório 5)

Implementa cache para o Relatório 5 do IFData com variáveis de capital.
Produz dados no formato exato que os gráficos do app1.py esperam.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import BaseCache, CacheConfig, CacheResult

logger = logging.getLogger("ifdata_cache")

# Mapeamento de campos do Relatório 5
CAMPOS_CAPITAL = {
    "Capital Principal para Comparação com RWA (a)": "Capital Principal",
    "Capital Complementar (b)": "Capital Complementar",
    "Patrimônio de Referência Nível I para Comparação com RWA (c) = (a) + (b)": "Patrimônio de Referência Nível I",
    "Capital Nível II (d)": "Capital Nível II",
    "RWA para Risco de Crédito (f)": "RWA Crédito",
    "RWA para Risco de Mercado (g) = (g1) + (g2) + (g3) + (g4) + (g5) + (g6)": "RWA Mercado",
    "RWA para Risco Operacional (h)": "RWA Operacional",
    "Ativos Ponderados pelo Risco (RWA) (j) = (f) + (g) + (h) + (i)": "RWA Total",
    "Exposição Total (k)": "Exposição Total",
    "Índice de Capital Principal (l) = (a) / (j)": "Índice de Capital Principal",
    "Índice de Capital Nível I (m) = (c) / (j)": "Índice de Capital Nível I",
    "Índice de Basileia (n) = (e) / (j)": "Índice de Basileia",
    "Adicional de Capital Principal": "Adicional de Capital Principal",
    "IRRBB": "IRRBB",
    "Razão de Alavancagem (o) = (c) / (k)": "Razão de Alavancagem",
    "Índice de Imobilização (p)": "Índice de Imobilização",
}

# Configuração do cache de capital
CAPITAL_CONFIG = CacheConfig(
    nome="capital",
    descricao="Dados de capital regulatório (Relatório 5)",
    subdir="capital",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Período"],
    campos_mapeamento=CAMPOS_CAPITAL,
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=5,
)


class CapitalCache(BaseCache):
    """Cache de dados de capital regulatório do IFData.

    Produz dados com:
    - Coluna "Instituição" (nome da instituição)
    - Coluna "Período" no formato "1/2024" (trimestre/ano)
    - Métricas de capital no formato esperado pelos gráficos
    """

    def __init__(self, base_dir: Path):
        super().__init__(CAPITAL_CONFIG, base_dir)
        self.github_data_url = f"{self.config.github_url_base}/capital_cache.pkl"

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados do GitHub Releases."""
        self._log("info", "Tentando baixar do GitHub...")

        try:
            response = requests.get(self.github_data_url, timeout=120)

            if response.status_code == 404:
                self._log("warning", "Cache de capital não encontrado no GitHub")
                return CacheResult(
                    sucesso=False,
                    mensagem="Cache de capital não existe no GitHub",
                    fonte="nenhum"
                )

            response.raise_for_status()

            import pickle
            import io

            dados_dict = pickle.load(io.BytesIO(response.content))

            # Converter para DataFrame único
            if isinstance(dados_dict, dict):
                dfs = []
                for periodo, df in dados_dict.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
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
        """Extrai dados de capital de um período da API do BCB.

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
            from .extractor import extrair_capital

            df = extrair_capital(periodo, dict_aliases)

            if df is None or df.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Sem dados de capital para {periodo}",
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
        """Carrega e retorna no formato antigo {periodo: DataFrame}."""
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


def gerar_periodos_trimestrais(
    ano_ini: int,
    mes_ini: int,
    ano_fin: int,
    mes_fin: int
) -> List[str]:
    """Gera lista de períodos trimestrais.

    Args:
        ano_ini: Ano inicial
        mes_ini: Mês inicial (3, 6, 9, ou 12)
        ano_fin: Ano final
        mes_fin: Mês final (3, 6, 9, ou 12)

    Returns:
        Lista de períodos no formato YYYYMM
    """
    meses_validos = [3, 6, 9, 12]
    periodos = []

    for ano in range(ano_ini, ano_fin + 1):
        for mes in meses_validos:
            if ano == ano_ini and mes < mes_ini:
                continue
            if ano == ano_fin and mes > mes_fin:
                continue
            periodos.append(f"{ano}{mes:02d}")

    return periodos
