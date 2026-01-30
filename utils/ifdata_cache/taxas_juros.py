"""
taxas_juros.py - Cache para dados de Taxas de Juros do BCB

Integra dados de taxas de juros ao sistema de cache do IFData.
API: TaxasJurosDiariaPorInicioPeriodo
Periodicidade: Janelas de 5 dias úteis (rolling window)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseCache, CacheConfig, CacheResult

logger = logging.getLogger("ifdata_cache")

# URL da API do BCB para taxas de juros
API_URL = "https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata/TaxasJurosDiariaPorInicioPeriodo"

# Timeout padrão para requisições (segundos)
REQUEST_TIMEOUT = 120

# Máximo de registros por requisição
MAX_RECORDS = 150000


# Configuração do cache
TAXAS_JUROS_CONFIG = CacheConfig(
    nome="taxas_juros",
    descricao="Taxas de Juros por Produto e Instituição (API BCB)",
    subdir="taxas_juros",
    arquivo_dados="dados.parquet",
    arquivo_metadata="metadata.json",
    github_url_base=None,  # Sem suporte a download remoto por enquanto
    max_idade_horas=24.0,  # Dados atualizados diariamente
    colunas_obrigatorias=["Fim Período", "Instituição Financeira"],
    api_url=API_URL,
)


class TaxasJurosCache(BaseCache):
    """Cache para dados de taxas de juros do BCB."""

    def __init__(self, base_dir: Path):
        super().__init__(TAXAS_JUROS_CONFIG, base_dir)

    def baixar_remoto(self) -> CacheResult:
        """Baixa dados de fonte remota (API BCB).

        Para taxas de juros, não há cache no GitHub, então tentamos
        carregar os últimos 90 dias da API diretamente.
        """
        try:
            data_inicio = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            resultado = self._extrair_periodo_api(data_inicio)
            if resultado.sucesso:
                resultado.fonte = "api"
            return resultado
        except Exception as e:
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro ao baixar dados: {e}",
                fonte="nenhum"
            )

    def extrair_periodo(
        self,
        periodo: str,
        data_fim: Optional[str] = None,
        **kwargs
    ) -> CacheResult:
        """Extrai dados de um período específico da API.

        Para taxas de juros, o período é uma data no formato YYYY-MM-DD.
        """
        return self._extrair_periodo_api(periodo, data_fim)

    def _extrair_periodo_api(
        self,
        data_inicio: str,
        data_fim: Optional[str] = None
    ) -> CacheResult:
        """Extrai dados da API do BCB."""
        if data_fim is None:
            data_fim = datetime.now().strftime('%Y-%m-%d')

        params = {
            "$format": "json",
            "$top": MAX_RECORDS,
            "dataInicioPeriodo": f"'{data_inicio}'"
        }

        try:
            self._log("info", f"Extraindo dados de {data_inicio} a {data_fim}...")
            response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT * 2)

            if response.status_code != 200:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Erro HTTP: {response.status_code}",
                    fonte="nenhum"
                )

            dados = response.json()

            if 'value' not in dados or not dados['value']:
                return CacheResult(
                    sucesso=False,
                    mensagem="Nenhum dado retornado pela API",
                    fonte="nenhum"
                )

            df = pd.DataFrame(dados['value'])

            # Converter datas
            df['InicioPeriodo'] = pd.to_datetime(df['InicioPeriodo'])
            df['FimPeriodo'] = pd.to_datetime(df['FimPeriodo'])

            # Filtrar por data final
            data_fim_dt = pd.to_datetime(data_fim)
            df = df[df['FimPeriodo'] <= data_fim_dt]

            # Selecionar colunas relevantes
            colunas_manter = [
                'InicioPeriodo', 'FimPeriodo', 'Segmento', 'Modalidade',
                'Posicao', 'InstituicaoFinanceira', 'TaxaJurosAoMes', 'TaxaJurosAoAno', 'cnpj8'
            ]
            colunas_existentes = [c for c in colunas_manter if c in df.columns]
            df = df[colunas_existentes]

            # Renomear colunas
            df = df.rename(columns={
                'InicioPeriodo': 'Início Período',
                'FimPeriodo': 'Fim Período',
                'Segmento': 'Segmento',
                'Modalidade': 'Produto',
                'Posicao': 'Posição',
                'InstituicaoFinanceira': 'Instituição Financeira',
                'TaxaJurosAoMes': 'Taxa Mensal (%)',
                'TaxaJurosAoAno': 'Taxa Anual (%)',
                'cnpj8': 'CNPJ'
            })

            # Ordenar por data e produto
            df = df.sort_values(['Fim Período', 'Produto', 'Posição'], ascending=[False, True, True])

            self._log("info", f"Extraídos {len(df)} registros de taxas de juros")

            return CacheResult(
                sucesso=True,
                mensagem=f"Extraídos {len(df)} registros",
                dados=df,
                fonte="api"
            )

        except requests.exceptions.Timeout:
            return CacheResult(
                sucesso=False,
                mensagem="Timeout ao extrair dados do BCB",
                fonte="nenhum"
            )
        except Exception as e:
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro na extração: {e}",
                fonte="nenhum"
            )

    def _validar_dados(self, dados: Any) -> Tuple[bool, str]:
        """Valida dados específicos de taxas de juros."""
        if dados is None:
            return False, "Dados são None"

        if not isinstance(dados, pd.DataFrame):
            return False, f"Esperado DataFrame, recebido {type(dados)}"

        if dados.empty:
            return False, "DataFrame vazio"

        # Verificar colunas obrigatórias
        colunas_obrig = ['Fim Período', 'Instituição Financeira']
        for col in colunas_obrig:
            if col not in dados.columns:
                return False, f"Coluna obrigatória ausente: {col}"

        return True, "OK"

    def extrair_com_filtros(
        self,
        data_inicio: str,
        data_fim: Optional[str] = None,
        modalidades: Optional[List[str]] = None,
        instituicoes: Optional[List[str]] = None,
        progress_callback=None
    ) -> CacheResult:
        """Extrai dados com filtros opcionais.

        Args:
            data_inicio: Data de início no formato 'YYYY-MM-DD'
            data_fim: Data de fim no formato 'YYYY-MM-DD'
            modalidades: Lista de modalidades para filtrar
            instituicoes: Lista de instituições para filtrar
            progress_callback: Função de callback para progresso
        """
        if progress_callback:
            progress_callback(0.1, "Conectando à API do BCB...")

        resultado = self._extrair_periodo_api(data_inicio, data_fim)

        if not resultado.sucesso:
            return resultado

        df = resultado.dados

        if progress_callback:
            progress_callback(0.7, "Aplicando filtros...")

        # Filtrar por modalidades se especificado
        if modalidades and len(modalidades) > 0:
            df = df[df['Produto'].isin(modalidades)]

        # Filtrar por instituições se especificado
        if instituicoes and len(instituicoes) > 0:
            df = df[df['Instituição Financeira'].isin(instituicoes)]

        if progress_callback:
            progress_callback(1.0, "Extração concluída!")

        return CacheResult(
            sucesso=True,
            mensagem=f"Extraídos {len(df)} registros",
            dados=df,
            fonte="api"
        )


def buscar_modalidades_disponiveis(dias_amostra: int = 60) -> List[str]:
    """Busca todas as modalidades (produtos) disponíveis na API."""
    data_inicio = (datetime.now() - timedelta(days=dias_amostra)).strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": 50000,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dados = response.json()
            if 'value' in dados and dados['value']:
                df = pd.DataFrame(dados['value'])
                if 'Modalidade' in df.columns:
                    return sorted(df['Modalidade'].unique().tolist())

        return []

    except Exception as e:
        logger.error(f"Erro ao buscar modalidades: {e}")
        return []


def buscar_instituicoes_disponiveis(dias_amostra: int = 60) -> List[str]:
    """Busca todas as instituições disponíveis na API."""
    data_inicio = (datetime.now() - timedelta(days=dias_amostra)).strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": 100000,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dados = response.json()
            if 'value' in dados and dados['value']:
                df = pd.DataFrame(dados['value'])
                if 'InstituicaoFinanceira' in df.columns:
                    return sorted(df['InstituicaoFinanceira'].unique().tolist())

        return []

    except Exception as e:
        logger.error(f"Erro ao buscar instituições: {e}")
        return []


def formatar_nome_modalidade(nome: str) -> str:
    """Formata o nome da modalidade para exibição mais curta."""
    substituicoes = {
        ' - Pré-fixado': ' (Pré)',
        ' - Pós-fixado referenciado em moeda estrangeira': ' (Pós M.E.)',
        ' - Pós-fixado referenciado em juros flutuantes': ' (Pós Flutuante)',
        ' - Pós-fixado referenciado em TR': ' (Pós TR)',
        ' - Pós-fixado referenciado em IPCA': ' (Pós IPCA)',
        'Adiantamento sobre contratos de câmbio (ACC)': 'ACC',
        'Capital de giro com prazo até 365 dias': 'Capital Giro até 365d',
        'Capital de giro com prazo superior a 365 dias': 'Capital Giro > 365d',
        'Cartão de crédito - parcelado': 'Cartão Parcelado',
        'Cartão de crédito - rotativo total': 'Cartão Rotativo',
        'Crédito pessoal consignado INSS': 'Consignado INSS',
        'Crédito pessoal consignado privado': 'Consignado Privado',
        'Crédito pessoal consignado público': 'Consignado Público',
        'Crédito pessoal não consignado': 'Crédito Pessoal',
        'Antecipação de faturas de cartão de crédito': 'Antecipação Fatura',
        'Aquisição de outros bens': 'Aquisição Outros',
        'Aquisição de veículos': 'Aquisição Veículos',
        'Desconto de cheques': 'Desconto Cheques',
        'Desconto de duplicatas': 'Desconto Duplicatas',
        'Conta garantida': 'Conta Garantida',
        'Cheque especial': 'Cheque Especial',
    }

    resultado = nome
    for original, abreviado in substituicoes.items():
        resultado = resultado.replace(original, abreviado)

    return resultado


# Modalidades conhecidas (fallback)
MODALIDADES_CONHECIDAS = [
    "Adiantamento sobre contratos de câmbio (ACC) - Pós-fixado referenciado em moeda estrangeira",
    "Antecipação de faturas de cartão de crédito - Pré-fixado",
    "Aquisição de outros bens - Pré-fixado",
    "Aquisição de veículos - Pré-fixado",
    "Capital de giro com prazo até 365 dias - Pré-fixado",
    "Capital de giro com prazo até 365 dias - Pós-fixado referenciado em juros flutuantes",
    "Capital de giro com prazo superior a 365 dias - Pré-fixado",
    "Capital de giro com prazo superior a 365 dias - Pós-fixado referenciado em juros flutuantes",
    "Cartão de crédito - parcelado - Pré-fixado",
    "Cartão de crédito - rotativo total - Pré-fixado",
    "Cheque especial - Pré-fixado",
    "Conta garantida - Pré-fixado",
    "Conta garantida - Pós-fixado referenciado em juros flutuantes",
    "Crédito pessoal consignado INSS - Pré-fixado",
    "Crédito pessoal consignado privado - Pré-fixado",
    "Crédito pessoal consignado público - Pré-fixado",
    "Crédito pessoal não consignado - Pré-fixado",
    "Desconto de cheques - Pré-fixado",
    "Desconto de duplicatas - Pré-fixado",
    "Vendor - Pré-fixado",
]
