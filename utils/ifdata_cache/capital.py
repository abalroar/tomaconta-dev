"""
capital.py - Cache de dados de capital regulatorio

Implementa cache para o Relatorio 5 do IFData (dados de capital).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import BaseCache, CacheConfig, CacheResult

logger = logging.getLogger("ifdata_cache")

# Mapeamento de campos do Relatorio 5 (capital)
CAMPOS_CAPITAL = {
    "Capital Principal para Comparação com RWA (a)": "Capital Principal",
    "Capital Complementar (b)": "Capital Complementar",
    "Patrimônio de Referência Nível I": "Patrimônio de Referência",
    "Capital Nível II (d)": "Capital Nível II",
    "RWA para Risco de Crédito (f)": "RWA Crédito",
    "RWA para Risco de Mercado (g)": "RWA Mercado",
    "RWA para Risco Operacional (h)": "RWA Operacional",
    "Ativos Ponderados pelo Risco (RWA) (j)": "RWA Total",
    "Índice de Capital Principal (l)": "Índice de Capital Principal",
    "Índice de Capital Nível I (m)": "Índice de Capital Nível I",
    "Índice de Basileia (n)": "Índice de Basileia",
    "Exposição Total (k)": "Exposição Total",
    "Razão de Alavancagem (o)": "Razão de Alavancagem",
    "Índice de Imobilização (p)": "Índice de Imobilização",
    "Adicional de Capital Principal": "Adicional de Capital Principal",
    "IRRBB": "IRRBB",
}

# Configuracao do cache de capital
CAPITAL_CONFIG = CacheConfig(
    nome="capital",
    descricao="Dados de capital regulatorio (Relatorio 5)",
    subdir="capital",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta/releases/download/v1.0-cache",
    max_idade_horas=168.0,  # 7 dias
    colunas_obrigatorias=["Periodo", "CodInst"],
    campos_mapeamento=CAMPOS_CAPITAL,
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=5,
)


class CapitalCache(BaseCache):
    """Cache de dados de capital regulatorio do IFData."""

    def __init__(self, base_dir: Path):
        super().__init__(CAPITAL_CONFIG, base_dir)

        # URLs especificas
        self.github_data_url = f"{self.config.github_url_base}/capital_cache.pkl"
        self.github_info_url = f"{self.config.github_url_base}/capital_cache_info.txt"

        # API endpoints
        self.api_cadastro = f"{self.config.api_url}/IfDataCadastro"
        self.api_valores = f"{self.config.api_url}/IfDataValores"

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados do GitHub Releases."""
        self._log("info", "Tentando baixar do GitHub...")

        try:
            response = requests.get(self.github_data_url, timeout=120)

            if response.status_code == 404:
                self._log("warning", "Cache de capital nao encontrado no GitHub (404)")
                return CacheResult(
                    sucesso=False,
                    mensagem="Cache de capital nao existe no GitHub",
                    fonte="nenhum"
                )

            response.raise_for_status()

            # Carregar pickle (formato antigo)
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
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro de rede: {e}",
                fonte="nenhum"
            )
        except Exception as e:
            self._log("error", f"Erro ao processar: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro: {e}",
                fonte="nenhum"
            )

    def extrair_periodo(
        self,
        periodo: str,
        dict_aliases: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> CacheResult:
        """Extrai dados de capital de um periodo da API do BCB.

        Args:
            periodo: Periodo no formato "YYYYMM" (ex: "202312")
            dict_aliases: Dicionario de aliases para nomes de instituicoes

        Returns:
            CacheResult com DataFrame ou erro
        """
        self._log("info", f"Extraindo periodo {periodo}...")

        try:
            ano = periodo[:4]
            mes = periodo[4:6]

            # 1. Buscar cadastro (nomes das instituicoes)
            df_cadastro = self._buscar_cadastro(ano, mes)
            if df_cadastro is None or df_cadastro.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Sem dados de cadastro para {periodo}",
                    fonte="nenhum"
                )

            # 2. Buscar valores de capital (Relatorio 5)
            df_valores = self._buscar_valores(ano, mes)
            if df_valores is None or df_valores.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Sem dados de capital para {periodo}",
                    fonte="nenhum"
                )

            # 3. Processar e combinar dados
            df_final = self._processar_dados(df_cadastro, df_valores, periodo, dict_aliases)

            if df_final is None or df_final.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Erro ao processar dados de {periodo}",
                    fonte="nenhum"
                )

            self._log("info", f"Periodo {periodo}: {len(df_final)} instituicoes")

            return CacheResult(
                sucesso=True,
                mensagem=f"Extraido {periodo}: {len(df_final)} registros",
                dados=df_final,
                fonte="api"
            )

        except Exception as e:
            self._log("error", f"Erro em {periodo}: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro: {e}",
                fonte="nenhum"
            )

    def _buscar_cadastro(self, ano: str, mes: str) -> Optional[pd.DataFrame]:
        """Busca cadastro de instituicoes na API."""
        url = (
            f"{self.api_cadastro}?"
            f"$filter=AnoMes eq '{ano}{mes}'"
            f"&$select=CodInst,NomeInstituicao"
            f"&$format=json"
        )

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "value" not in data:
                return None

            return pd.DataFrame(data["value"])

        except Exception as e:
            self._log("warning", f"Erro ao buscar cadastro: {e}")
            return None

    def _buscar_valores(self, ano: str, mes: str) -> Optional[pd.DataFrame]:
        """Busca valores de capital (Relatorio 5) na API."""
        url = (
            f"{self.api_valores}?"
            f"$filter=AnoMes eq '{ano}{mes}' and Relatorio eq '5'"
            f"&$select=CodInst,NomeColuna,Valor"
            f"&$format=json"
        )

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()

            if "value" not in data:
                return None

            return pd.DataFrame(data["value"])

        except Exception as e:
            self._log("warning", f"Erro ao buscar valores: {e}")
            return None

    def _processar_dados(
        self,
        df_cadastro: pd.DataFrame,
        df_valores: pd.DataFrame,
        periodo: str,
        dict_aliases: Optional[Dict[str, str]] = None
    ) -> Optional[pd.DataFrame]:
        """Processa e combina dados de cadastro e valores."""
        try:
            # Filtrar apenas campos de capital que queremos
            campos_desejados = list(CAMPOS_CAPITAL.keys())
            df_filtrado = df_valores[df_valores["NomeColuna"].isin(campos_desejados)].copy()

            if df_filtrado.empty:
                return None

            # Pivotar para ter uma coluna por campo
            df_pivot = df_filtrado.pivot_table(
                index="CodInst",
                columns="NomeColuna",
                values="Valor",
                aggfunc="first"
            ).reset_index()

            # Renomear colunas
            rename_map = {k: v for k, v in CAMPOS_CAPITAL.items() if k in df_pivot.columns}
            df_pivot = df_pivot.rename(columns=rename_map)

            # Juntar com cadastro para ter nomes
            df_final = df_pivot.merge(df_cadastro, on="CodInst", how="left")

            # Adicionar periodo
            df_final["Periodo"] = periodo

            # Aplicar aliases se fornecido
            if dict_aliases and "NomeInstituicao" in df_final.columns:
                df_final["NomeInstituicao"] = df_final["NomeInstituicao"].apply(
                    lambda x: dict_aliases.get(x, x) if pd.notna(x) else x
                )

            # Reordenar colunas
            cols_inicio = ["Periodo", "CodInst", "NomeInstituicao"]
            outras_cols = [c for c in df_final.columns if c not in cols_inicio]
            df_final = df_final[cols_inicio + outras_cols]

            return df_final

        except Exception as e:
            self._log("error", f"Erro ao processar: {e}")
            return None

    # =========================================================================
    # COMPATIBILIDADE COM SISTEMA ANTIGO
    # =========================================================================

    def carregar_formato_antigo(self) -> Optional[dict]:
        """Carrega e retorna no formato antigo {periodo: DataFrame}."""
        resultado = self.carregar()
        if not resultado.sucesso or resultado.dados is None:
            return None

        df = resultado.dados
        if "Periodo" not in df.columns:
            return None

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
        """Salva a partir do formato antigo {periodo: DataFrame}."""
        if not dados_dict:
            return CacheResult(
                sucesso=False,
                mensagem="Dicionario vazio",
                fonte="nenhum"
            )

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
                mensagem="Nenhum DataFrame valido",
                fonte="nenhum"
            )

        df_final = pd.concat(dfs, ignore_index=True)
        return self.salvar_local(df_final, fonte=fonte, info_extra=info_extra)


# Funcoes de conveniencia para compatibilidade
def gerar_periodos_trimestrais(
    ano_ini: int,
    mes_ini: int,
    ano_fin: int,
    mes_fin: int
) -> List[str]:
    """Gera lista de periodos trimestrais (03, 06, 09, 12).

    Args:
        ano_ini: Ano inicial
        mes_ini: Mes inicial (1-12)
        ano_fin: Ano final
        mes_fin: Mes final (1-12)

    Returns:
        Lista de periodos no formato "YYYYMM"
    """
    meses_validos = [3, 6, 9, 12]
    periodos = []

    for ano in range(ano_ini, ano_fin + 1):
        for mes in meses_validos:
            # Verificar limites
            if ano == ano_ini and mes < mes_ini:
                continue
            if ano == ano_fin and mes > mes_fin:
                continue

            periodos.append(f"{ano}{mes:02d}")

    return periodos
