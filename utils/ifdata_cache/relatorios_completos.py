"""
relatorios_completos.py - Caches para relatórios que extraem todas as variáveis

Este módulo implementa caches para os relatórios que extraem TODAS as variáveis:
- Relatório 2: Ativo
- Relatório 3: Passivo
- Relatório 4: Demonstração de Resultado (DRE)
- Relatório 11: Carteira de crédito ativa PF - modalidade e prazo
- Relatório 13: Carteira de crédito ativa PJ - modalidade e prazo
- Relatório 14: Carteira de crédito ativa - por carteiras de instrumentos financeiros

Todos usam a mesma lógica: extrair TODAS as variáveis do relatório da API.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import BaseCache, CacheConfig, CacheResult
from .unified_extractor import (
    extrair_cadastro,
    extrair_valores,
    normalizar_nome_coluna,
    ExtractionError
)

logger = logging.getLogger("ifdata_cache")


# =============================================================================
# CLASSE BASE PARA RELATORIOS COMPLETOS
# =============================================================================
class RelatorioCompletoCache(BaseCache):
    """Cache base para relatórios que extraem todas as variáveis."""

    def __init__(self, config: CacheConfig, base_dir: Path, relatorio_num: int):
        super().__init__(config, base_dir)
        self.relatorio_num = relatorio_num

        # URLs do GitHub
        self.github_data_url = f"{self.config.github_url_base}/{self.config.nome}_cache.pkl"

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados do GitHub Releases."""
        self._log("info", "Tentando baixar do GitHub...")

        try:
            response = requests.get(self.github_data_url, timeout=120)

            if response.status_code == 404:
                self._log("warning", f"Cache {self.config.nome} não encontrado no GitHub (404)")
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Cache {self.config.nome} não existe no GitHub",
                    fonte="nenhum"
                )

            response.raise_for_status()

            # Carregar pickle (formato antigo)
            import pickle
            import io

            dados_dict = pickle.load(io.BytesIO(response.content))

            # Converter para DataFrame único
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
        """Extrai dados de um período da API do BCB.

        Extrai TODAS as variáveis do relatório.
        """
        self._log("info", f"Extraindo período {periodo}...")

        try:
            # 1. Buscar cadastro (nomes das instituições)
            df_cadastro = extrair_cadastro(periodo)

            # 2. Buscar valores do relatório
            df_valores = extrair_valores(periodo, self.relatorio_num)

            if df_valores.empty:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Sem dados para {periodo}",
                    fonte="nenhum"
                )

            # 3. Normalizar nomes de colunas
            if "NomeColuna" in df_valores.columns:
                df_valores["NomeColuna"] = df_valores["NomeColuna"].apply(normalizar_nome_coluna)

            # 4. Pivotar dados (TODAS as variáveis)
            df_pivot = df_valores.pivot_table(
                index="CodInst",
                columns="NomeColuna",
                values="Saldo",
                aggfunc="sum"
            ).reset_index()
            df_pivot.columns.name = None

            # 5. Adicionar nomes de instituições
            if not df_cadastro.empty and "CodInst" in df_cadastro.columns:
                col_nome = None
                for candidato in ["NomeInstituicao", "NomeInstituição"]:
                    if candidato in df_cadastro.columns:
                        col_nome = candidato
                        break

                if col_nome:
                    df_nomes = df_cadastro[["CodInst", col_nome]].drop_duplicates()
                    df_nomes = df_nomes.rename(columns={col_nome: "NomeInstituicao"})
                    df_pivot = df_pivot.merge(df_nomes, on="CodInst", how="left")

            # 6. Preencher nomes faltantes
            if "NomeInstituicao" not in df_pivot.columns:
                df_pivot["NomeInstituicao"] = df_pivot["CodInst"].apply(lambda x: f"[IF {x}]")
            else:
                df_pivot["NomeInstituicao"] = df_pivot.apply(
                    lambda row: row["NomeInstituicao"] if pd.notna(row["NomeInstituicao"])
                    else f"[IF {row['CodInst']}]",
                    axis=1
                )

            # 7. Aplicar aliases se fornecido
            if dict_aliases:
                df_pivot["NomeInstituicao"] = df_pivot["NomeInstituicao"].apply(
                    lambda x: dict_aliases.get(x, x) if pd.notna(x) else x
                )

            # 8. Adicionar período
            df_pivot["Periodo"] = periodo

            # 9. Reordenar colunas
            cols_inicio = ["Periodo", "CodInst", "NomeInstituicao"]
            outras_cols = sorted([c for c in df_pivot.columns if c not in cols_inicio])
            df_pivot = df_pivot[cols_inicio + outras_cols]

            # 10. Remover linhas sem dados
            colunas_numericas = [c for c in df_pivot.columns if c not in cols_inicio]
            if colunas_numericas:
                df_pivot = df_pivot.dropna(subset=colunas_numericas, how="all")

            self._log("info", f"Período {periodo}: {len(df_pivot)} instituições, {len(colunas_numericas)} variáveis")

            return CacheResult(
                sucesso=True,
                mensagem=f"Extraído {periodo}: {len(df_pivot)} registros",
                dados=df_pivot,
                metadata={
                    "periodo": periodo,
                    "relatorio": self.relatorio_num,
                    "n_instituicoes": len(df_pivot),
                    "n_variaveis": len(colunas_numericas)
                },
                fonte="api"
            )

        except Exception as e:
            self._log("error", f"Erro em {periodo}: {e}")
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
                mensagem="Dicionário vazio",
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
                mensagem="Nenhum DataFrame válido",
                fonte="nenhum"
            )

        df_final = pd.concat(dfs, ignore_index=True)
        return self.salvar_local(df_final, fonte=fonte, info_extra=info_extra)


# =============================================================================
# CONFIGURACOES ESPECÍFICAS DE CADA RELATORIO
# =============================================================================

# Relatório 2: Ativo
ATIVO_CONFIG = CacheConfig(
    nome="ativo",
    descricao="Composição detalhada do Ativo (Relatório 2)",
    subdir="ativo",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=2,
)

# Relatório 3: Passivo
PASSIVO_CONFIG = CacheConfig(
    nome="passivo",
    descricao="Composição detalhada do Passivo (Relatório 3)",
    subdir="passivo",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=3,
)

# Relatório 4: DRE
DRE_CONFIG = CacheConfig(
    nome="dre",
    descricao="Demonstração de Resultado do Exercício (Relatório 4)",
    subdir="dre",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=4,
)

# Relatório 11: Carteira PF
CARTEIRA_PF_CONFIG = CacheConfig(
    nome="carteira_pf",
    descricao="Carteira de crédito ativa PF - modalidade e prazo (Relatório 11)",
    subdir="carteira_pf",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=11,
)

# Relatório 13: Carteira PJ
CARTEIRA_PJ_CONFIG = CacheConfig(
    nome="carteira_pj",
    descricao="Carteira de crédito ativa PJ - modalidade e prazo (Relatório 13)",
    subdir="carteira_pj",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=13,
)

# Relatório 14: Carteira Instrumentos Financeiros
CARTEIRA_INSTRUMENTOS_CONFIG = CacheConfig(
    nome="carteira_instrumentos",
    descricao="Carteira de crédito ativa - por instrumentos financeiros (Relatório 14)",
    subdir="carteira_instrumentos",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base="https://github.com/abalroar/tomaconta-dev/releases/download/v1.0-cache",
    max_idade_horas=168.0,
    colunas_obrigatorias=["Periodo", "CodInst"],
    api_url="https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata",
    relatorio_tipo=14,
)


# =============================================================================
# CLASSES DE CACHE ESPECIFICAS
# =============================================================================

class AtivoCache(RelatorioCompletoCache):
    """Cache de dados de Ativo (Relatório 2)."""

    def __init__(self, base_dir: Path):
        super().__init__(ATIVO_CONFIG, base_dir, relatorio_num=2)


class PassivoCache(RelatorioCompletoCache):
    """Cache de dados de Passivo (Relatório 3)."""

    def __init__(self, base_dir: Path):
        super().__init__(PASSIVO_CONFIG, base_dir, relatorio_num=3)


class DRECache(RelatorioCompletoCache):
    """Cache de Demonstração de Resultado (Relatório 4)."""

    def __init__(self, base_dir: Path):
        super().__init__(DRE_CONFIG, base_dir, relatorio_num=4)


class CarteiraPFCache(RelatorioCompletoCache):
    """Cache de Carteira de Crédito PF (Relatório 11)."""

    def __init__(self, base_dir: Path):
        super().__init__(CARTEIRA_PF_CONFIG, base_dir, relatorio_num=11)


class CarteiraPJCache(RelatorioCompletoCache):
    """Cache de Carteira de Crédito PJ (Relatório 13)."""

    def __init__(self, base_dir: Path):
        super().__init__(CARTEIRA_PJ_CONFIG, base_dir, relatorio_num=13)


class CarteiraInstrumentosCache(RelatorioCompletoCache):
    """Cache de Carteira de Crédito - Instrumentos Financeiros (Relatório 14)."""

    def __init__(self, base_dir: Path):
        super().__init__(CARTEIRA_INSTRUMENTOS_CONFIG, base_dir, relatorio_num=14)


# =============================================================================
# FUNCOES DE CONVENIENCIA
# =============================================================================

def listar_relatorios_completos() -> List[Dict]:
    """Lista todos os relatórios completos disponíveis."""
    return [
        {"numero": 2, "nome": "ativo", "descricao": "Composição do Ativo"},
        {"numero": 3, "nome": "passivo", "descricao": "Composição do Passivo"},
        {"numero": 4, "nome": "dre", "descricao": "Demonstração de Resultado"},
        {"numero": 11, "nome": "carteira_pf", "descricao": "Carteira de Crédito PF"},
        {"numero": 13, "nome": "carteira_pj", "descricao": "Carteira de Crédito PJ"},
        {"numero": 14, "nome": "carteira_instrumentos", "descricao": "Carteira - Instrumentos Financeiros"},
    ]
