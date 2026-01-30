"""
Extrator de Taxas de Juros do Banco Central do Brasil
======================================================

Este módulo fornece funções para extrair dados de taxas de juros
por produto e por instituição financeira da API do BCB.

API utilizada: TaxasJurosDiariaPorInicioPeriodo
URL base: https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata

Periodicidade: Os dados são divulgados em janelas de 5 dias úteis consecutivos.
Cada registro representa um período de ~5 dias (segunda a sexta ou similar).
Para construir séries temporais, usa-se a data final (FimPeriodo) como referência.

Estrutura dos dados:
- InicioPeriodo: Data de início do período
- FimPeriodo: Data de fim do período
- Segmento: PESSOA FÍSICA ou PESSOA JURÍDICA
- Modalidade: Nome do produto de crédito
- Posicao: Ranking da instituição para aquele produto/período
- InstituicaoFinanceira: Nome da instituição
- TaxaJurosAoMes: Taxa de juros mensal (%)
- TaxaJurosAoAno: Taxa de juros anual (%)
- cnpj8: CNPJ da instituição (8 primeiros dígitos)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging

# Configuração de logging
logger = logging.getLogger("taxas_juros_extractor")
logger.setLevel(logging.INFO)

# URL da API do BCB para taxas de juros
API_URL = "https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata/TaxasJurosDiariaPorInicioPeriodo"

# Timeout padrão para requisições (segundos)
REQUEST_TIMEOUT = 120

# Máximo de registros por requisição
MAX_RECORDS = 100000


def buscar_modalidades_disponiveis(dias_amostra: int = 30) -> Tuple[List[str], pd.DataFrame]:
    """
    Busca todas as modalidades (produtos) disponíveis na API do BCB.

    Args:
        dias_amostra: Número de dias para buscar amostra de dados

    Returns:
        Tuple com lista de modalidades únicas e DataFrame com amostra de dados
    """
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
                    modalidades = sorted(df['Modalidade'].unique())
                    logger.info(f"Encontradas {len(modalidades)} modalidades")
                    return modalidades, df

        logger.error(f"Erro ao acessar API: status {response.status_code}")
        return [], pd.DataFrame()

    except requests.exceptions.Timeout:
        logger.error("Timeout ao acessar API do BCB")
        return [], pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro de conexão: {e}")
        return [], pd.DataFrame()


def buscar_instituicoes_por_modalidade(
    modalidade: str,
    df_amostra: Optional[pd.DataFrame] = None,
    dias_busca: int = 180
) -> List[str]:
    """
    Busca todas as instituições que oferecem uma determinada modalidade de crédito.

    Args:
        modalidade: Nome da modalidade (produto)
        df_amostra: DataFrame com amostra prévia de dados (opcional)
        dias_busca: Dias para buscar se df_amostra estiver vazio

    Returns:
        Lista de nomes de instituições financeiras
    """
    if df_amostra is not None and not df_amostra.empty:
        df_filtrado = df_amostra[df_amostra['Modalidade'] == modalidade]
        if not df_filtrado.empty:
            instituicoes = sorted(df_filtrado['InstituicaoFinanceira'].unique())
            logger.info(f"Encontradas {len(instituicoes)} instituições para '{modalidade}'")
            return instituicoes

    # Buscar dados se não houver amostra
    data_inicio = (datetime.now() - timedelta(days=dias_busca)).strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": MAX_RECORDS,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dados = response.json()
            if 'value' in dados and dados['value']:
                df = pd.DataFrame(dados['value'])
                df_filtrado = df[df['Modalidade'] == modalidade]
                if not df_filtrado.empty:
                    instituicoes = sorted(df_filtrado['InstituicaoFinanceira'].unique())
                    logger.info(f"Encontradas {len(instituicoes)} instituições para '{modalidade}'")
                    return instituicoes

        return []

    except Exception as e:
        logger.error(f"Erro ao buscar instituições: {e}")
        return []


def buscar_todas_instituicoes(dias_amostra: int = 60) -> List[str]:
    """
    Busca todas as instituições financeiras disponíveis na API.

    Args:
        dias_amostra: Número de dias para buscar amostra de dados

    Returns:
        Lista de nomes de instituições financeiras
    """
    data_inicio = (datetime.now() - timedelta(days=dias_amostra)).strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": MAX_RECORDS,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dados = response.json()
            if 'value' in dados and dados['value']:
                df = pd.DataFrame(dados['value'])
                if 'InstituicaoFinanceira' in df.columns:
                    instituicoes = sorted(df['InstituicaoFinanceira'].unique())
                    logger.info(f"Encontradas {len(instituicoes)} instituições no total")
                    return instituicoes

        return []

    except Exception as e:
        logger.error(f"Erro ao buscar instituições: {e}")
        return []


def buscar_periodos_disponiveis(dias_busca: int = 90) -> List[Tuple[str, str]]:
    """
    Busca os períodos disponíveis na API.

    Args:
        dias_busca: Número de dias para buscar

    Returns:
        Lista de tuplas (InicioPeriodo, FimPeriodo) ordenadas por data
    """
    data_inicio = (datetime.now() - timedelta(days=dias_busca)).strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": 10000,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dados = response.json()
            if 'value' in dados and dados['value']:
                df = pd.DataFrame(dados['value'])
                periodos = df[['InicioPeriodo', 'FimPeriodo']].drop_duplicates()
                periodos = periodos.sort_values('FimPeriodo', ascending=False)
                return list(periodos.itertuples(index=False, name=None))

        return []

    except Exception as e:
        logger.error(f"Erro ao buscar períodos: {e}")
        return []


def extrair_taxas_juros(
    data_inicio: str,
    data_fim: Optional[str] = None,
    modalidades: Optional[List[str]] = None,
    instituicoes: Optional[List[str]] = None,
    progress_callback=None
) -> pd.DataFrame:
    """
    Extrai dados de taxas de juros da API do BCB.

    Args:
        data_inicio: Data de início no formato 'YYYY-MM-DD'
        data_fim: Data de fim no formato 'YYYY-MM-DD' (opcional, padrão: hoje)
        modalidades: Lista de modalidades para filtrar (opcional)
        instituicoes: Lista de instituições para filtrar (opcional)
        progress_callback: Função de callback para atualizar progresso

    Returns:
        DataFrame com os dados de taxas de juros
    """
    if data_fim is None:
        data_fim = datetime.now().strftime('%Y-%m-%d')

    params = {
        "$format": "json",
        "$top": MAX_RECORDS,
        "dataInicioPeriodo": f"'{data_inicio}'"
    }

    if progress_callback:
        progress_callback(0.1, "Conectando à API do BCB...")

    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT * 2)

        if progress_callback:
            progress_callback(0.5, "Processando dados recebidos...")

        if response.status_code != 200:
            logger.error(f"Erro HTTP: {response.status_code}")
            return pd.DataFrame()

        dados = response.json()

        if 'value' not in dados or not dados['value']:
            logger.warning("Nenhum dado retornado pela API")
            return pd.DataFrame()

        df = pd.DataFrame(dados['value'])

        # Converter datas
        df['InicioPeriodo'] = pd.to_datetime(df['InicioPeriodo'])
        df['FimPeriodo'] = pd.to_datetime(df['FimPeriodo'])

        # Filtrar por data final
        data_fim_dt = pd.to_datetime(data_fim)
        df = df[df['FimPeriodo'] <= data_fim_dt]

        # Filtrar por modalidades se especificado
        if modalidades and len(modalidades) > 0:
            df = df[df['Modalidade'].isin(modalidades)]

        # Filtrar por instituições se especificado
        if instituicoes and len(instituicoes) > 0:
            df = df[df['InstituicaoFinanceira'].isin(instituicoes)]

        if progress_callback:
            progress_callback(0.9, "Finalizando extração...")

        # Selecionar apenas colunas relevantes (remover códigos internos)
        colunas_manter = [
            'InicioPeriodo', 'FimPeriodo', 'Segmento', 'Modalidade',
            'Posicao', 'InstituicaoFinanceira', 'TaxaJurosAoMes', 'TaxaJurosAoAno', 'cnpj8'
        ]
        colunas_existentes = [c for c in colunas_manter if c in df.columns]
        df = df[colunas_existentes]

        # Renomear colunas para português mais amigável
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

        logger.info(f"Extraídos {len(df)} registros de taxas de juros")

        if progress_callback:
            progress_callback(1.0, "Extração concluída!")

        return df

    except requests.exceptions.Timeout:
        logger.error("Timeout ao extrair dados do BCB")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro na extração: {e}")
        return pd.DataFrame()


def criar_tabela_pivot_taxas(
    df: pd.DataFrame,
    valor_coluna: str = 'Taxa Mensal (%)',
    usar_data_fim: bool = True
) -> pd.DataFrame:
    """
    Cria uma tabela pivot com datas nas linhas e instituições nas colunas.

    Args:
        df: DataFrame com dados de taxas
        valor_coluna: Coluna com os valores (Taxa Mensal ou Taxa Anual)
        usar_data_fim: Se True, usa FimPeriodo como data de referência

    Returns:
        DataFrame pivotado
    """
    if df.empty:
        return pd.DataFrame()

    col_data = 'Fim Período' if usar_data_fim else 'Início Período'

    # Criar pivot
    df_pivot = df.pivot_table(
        index=col_data,
        columns='Instituição Financeira',
        values=valor_coluna,
        aggfunc='mean'
    ).reset_index()

    # Renomear coluna de data
    df_pivot = df_pivot.rename(columns={col_data: 'Data'})

    # Ordenar colunas de instituições
    colunas_inst = sorted([c for c in df_pivot.columns if c != 'Data'])
    df_pivot = df_pivot[['Data'] + colunas_inst]

    # Ordenar por data
    df_pivot = df_pivot.sort_values('Data', ascending=False)

    return df_pivot


def criar_tabela_por_produto(
    df: pd.DataFrame,
    valor_coluna: str = 'Taxa Mensal (%)'
) -> Dict[str, pd.DataFrame]:
    """
    Cria tabelas pivot separadas por produto.

    Args:
        df: DataFrame com dados de taxas
        valor_coluna: Coluna com os valores

    Returns:
        Dicionário com produto como chave e DataFrame pivotado como valor
    """
    if df.empty:
        return {}

    resultado = {}

    produtos = df['Produto'].unique()

    for produto in produtos:
        df_produto = df[df['Produto'] == produto]
        df_pivot = criar_tabela_pivot_taxas(df_produto, valor_coluna)
        if not df_pivot.empty:
            resultado[produto] = df_pivot

    return resultado


def formatar_nome_modalidade(nome: str) -> str:
    """
    Formata o nome da modalidade para exibição mais curta.

    Args:
        nome: Nome completo da modalidade

    Returns:
        Nome formatado/abreviado
    """
    # Mapeamento de abreviações
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


def get_info_periodicidade() -> str:
    """
    Retorna informação sobre a periodicidade dos dados.

    Returns:
        String explicativa sobre a periodicidade
    """
    return """
    **Periodicidade dos dados:**

    Os dados são divulgados em janelas de 5 dias úteis consecutivos (rolling window).
    Por exemplo: 12/01/2026 a 16/01/2026 representa os dados agregados desse período.

    Cada instituição pode aparecer ou não em determinado período, dependendo se
    realizou operações naquela modalidade de crédito.

    Para construção de séries temporais, utiliza-se a data final (Fim Período)
    como data de referência.
    """


# Constantes exportadas
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

# Mapeamento de segmentos
SEGMENTOS = {
    "PF": "PESSOA FÍSICA",
    "PJ": "PESSOA JURÍDICA",
}
