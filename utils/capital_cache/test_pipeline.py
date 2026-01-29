#!/usr/bin/env python3
"""
test_pipeline.py - Testes do pipeline de cache de capital

Executa testes de validacao do sistema completo.
Pode ser executado diretamente: python -m utils.capital_cache.test_pipeline
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging para testes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_pipeline")


def print_header(titulo: str):
    """Imprime cabecalho de secao."""
    print("\n" + "=" * 60)
    print(f" {titulo}")
    print("=" * 60)


def print_result(nome: str, passou: bool, detalhes: str = ""):
    """Imprime resultado de teste."""
    status = "PASSOU" if passou else "FALHOU"
    simbolo = "[OK]" if passou else "[X]"
    print(f"  {simbolo} {nome}: {status}")
    if detalhes:
        print(f"      {detalhes}")


def test_imports():
    """Testa se todos os modulos podem ser importados."""
    print_header("TESTE: Imports")

    resultados = []

    try:
        from utils.capital_cache import config
        print_result("config", True)
        resultados.append(True)
    except Exception as e:
        print_result("config", False, str(e))
        resultados.append(False)

    try:
        from utils.capital_cache import extractor
        print_result("extractor", True)
        resultados.append(True)
    except Exception as e:
        print_result("extractor", False, str(e))
        resultados.append(False)

    try:
        from utils.capital_cache import storage
        print_result("storage", True)
        resultados.append(True)
    except Exception as e:
        print_result("storage", False, str(e))
        resultados.append(False)

    try:
        from utils.capital_cache import orchestrator
        print_result("orchestrator", True)
        resultados.append(True)
    except Exception as e:
        print_result("orchestrator", False, str(e))
        resultados.append(False)

    try:
        from utils.capital_cache import (
            obter_dados_capital,
            extrair_e_salvar_periodos,
            CacheResult,
        )
        print_result("__init__ exports", True)
        resultados.append(True)
    except Exception as e:
        print_result("__init__ exports", False, str(e))
        resultados.append(False)

    return all(resultados)


def test_config():
    """Testa configuracoes e caminhos."""
    print_header("TESTE: Configuracoes")

    from utils.capital_cache.config import (
        PROJECT_ROOT,
        CACHE_DIR,
        CACHE_DATA_FILE,
        CACHE_METADATA_FILE,
        CAMPOS_CAPITAL,
    )

    resultados = []

    # Verificar que PROJECT_ROOT existe
    passou = PROJECT_ROOT.exists()
    print_result("PROJECT_ROOT existe", passou, str(PROJECT_ROOT))
    resultados.append(passou)

    # Verificar que CACHE_DIR e filho de PROJECT_ROOT
    passou = str(CACHE_DIR).startswith(str(PROJECT_ROOT))
    print_result("CACHE_DIR dentro de PROJECT_ROOT", passou, str(CACHE_DIR))
    resultados.append(passou)

    # Verificar que CAMPOS_CAPITAL tem entradas
    passou = len(CAMPOS_CAPITAL) > 0
    print_result("CAMPOS_CAPITAL populado", passou, f"{len(CAMPOS_CAPITAL)} campos")
    resultados.append(passou)

    return all(resultados)


def test_storage_operations():
    """Testa operacoes de storage."""
    print_header("TESTE: Storage")

    import pandas as pd
    from utils.capital_cache.storage import (
        salvar_cache,
        carregar_cache,
        cache_existe,
        get_cache_info,
        limpar_cache,
    )
    from utils.capital_cache.config import CACHE_DIR

    resultados = []

    # Garantir diretorio limpo
    limpar_cache()

    # Teste 1: Verificar que cache nao existe inicialmente
    passou = not cache_existe()
    print_result("Cache nao existe inicialmente", passou)
    resultados.append(passou)

    # Teste 2: Criar DataFrame de teste
    df_teste = pd.DataFrame({
        "Periodo": ["202312", "202312", "202403", "202403"],
        "CodInst": [1, 2, 1, 2],
        "NomeInstituicao": ["Banco A", "Banco B", "Banco A", "Banco B"],
        "Capital Principal": [1000, 2000, 1100, 2100],
        "RWA Total": [5000, 10000, 5500, 10500],
    })

    # Teste 3: Salvar cache
    sucesso, msg = salvar_cache(df_teste, fonte="teste")
    passou = sucesso
    print_result("Salvar cache", passou, msg)
    resultados.append(passou)

    # Teste 4: Verificar que cache existe
    passou = cache_existe()
    print_result("Cache existe apos salvar", passou)
    resultados.append(passou)

    # Teste 5: Carregar cache
    df_carregado, metadata, msg = carregar_cache()
    passou = df_carregado is not None and len(df_carregado) == len(df_teste)
    print_result("Carregar cache", passou, f"{len(df_carregado) if df_carregado is not None else 0} registros")
    resultados.append(passou)

    # Teste 6: Verificar integridade dos dados
    if df_carregado is not None:
        passou = list(df_carregado.columns) == list(df_teste.columns)
        print_result("Colunas preservadas", passou)
        resultados.append(passou)

        passou = df_carregado["Capital Principal"].sum() == df_teste["Capital Principal"].sum()
        print_result("Valores preservados", passou)
        resultados.append(passou)

    # Teste 7: Verificar metadata
    passou = metadata is not None and metadata.get("fonte") == "teste"
    print_result("Metadata correto", passou, f"fonte={metadata.get('fonte') if metadata else None}")
    resultados.append(passou)

    # Teste 8: get_cache_info
    info = get_cache_info()
    passou = info["existe"] and info["total_registros"] == 4
    print_result("get_cache_info", passou, f"registros={info.get('total_registros')}")
    resultados.append(passou)

    # Teste 9: Limpar cache
    sucesso, msg = limpar_cache()
    passou = sucesso and not cache_existe()
    print_result("Limpar cache", passou, msg)
    resultados.append(passou)

    return all(resultados)


def test_extractor_validation():
    """Testa funcoes de validacao do extractor."""
    print_header("TESTE: Extractor Validation")

    import pandas as pd
    from utils.capital_cache.extractor import validar_dataframe

    resultados = []

    # Teste 1: DataFrame valido
    df_valido = pd.DataFrame({
        "Periodo": ["202312"],
        "CodInst": [1],
        "NomeInstituicao": ["Banco A"],
    })
    valido, msg = validar_dataframe(df_valido, "teste_valido")
    passou = valido
    print_result("DataFrame valido aceito", passou)
    resultados.append(passou)

    # Teste 2: DataFrame None
    valido, msg = validar_dataframe(None, "teste_none")
    passou = not valido
    print_result("DataFrame None rejeitado", passou)
    resultados.append(passou)

    # Teste 3: DataFrame vazio
    df_vazio = pd.DataFrame()
    valido, msg = validar_dataframe(df_vazio, "teste_vazio")
    passou = not valido
    print_result("DataFrame vazio rejeitado", passou)
    resultados.append(passou)

    # Teste 4: DataFrame sem colunas obrigatorias
    df_incompleto = pd.DataFrame({"OutraColuna": [1, 2, 3]})
    valido, msg = validar_dataframe(df_incompleto, "teste_incompleto")
    passou = not valido
    print_result("DataFrame incompleto rejeitado", passou)
    resultados.append(passou)

    return all(resultados)


def test_orchestrator_gerar_periodos():
    """Testa geracao de periodos trimestrais."""
    print_header("TESTE: Geracao de Periodos")

    from utils.capital_cache.orchestrator import gerar_periodos_trimestrais

    resultados = []

    # Teste 1: Ano completo
    periodos = gerar_periodos_trimestrais(2023, 1, 2023, 12)
    esperado = ["202303", "202306", "202309", "202312"]
    passou = periodos == esperado
    print_result("Ano completo 2023", passou, f"{periodos}")
    resultados.append(passou)

    # Teste 2: Multiplos anos
    periodos = gerar_periodos_trimestrais(2022, 6, 2023, 6)
    passou = "202206" in periodos and "202306" in periodos
    print_result("Multiplos anos", passou, f"{len(periodos)} periodos")
    resultados.append(passou)

    # Teste 3: Periodo parcial
    periodos = gerar_periodos_trimestrais(2023, 4, 2023, 9)
    passou = periodos == ["202306", "202309"]
    print_result("Periodo parcial", passou, f"{periodos}")
    resultados.append(passou)

    return all(resultados)


def test_orchestrator_cache_result():
    """Testa classe CacheResult."""
    print_header("TESTE: CacheResult")

    import pandas as pd
    from utils.capital_cache.orchestrator import CacheResult

    resultados = []

    # Teste 1: CacheResult de sucesso
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = CacheResult(df=df, fonte="teste", sucesso=True, mensagem="OK")
    passou = result.sucesso and result.fonte == "teste" and result.df is not None
    print_result("CacheResult sucesso", passou)
    resultados.append(passou)

    # Teste 2: CacheResult de falha
    result = CacheResult(df=None, fonte="nenhum", sucesso=False, mensagem="Erro")
    passou = not result.sucesso and result.df is None
    print_result("CacheResult falha", passou)
    resultados.append(passou)

    # Teste 3: Repr
    passou = "OK" in repr(CacheResult(df=df, fonte="x", sucesso=True, mensagem=""))
    print_result("CacheResult repr", passou)
    resultados.append(passou)

    return all(resultados)


def test_full_pipeline():
    """Testa pipeline completo (sem acesso a rede)."""
    print_header("TESTE: Pipeline Completo (local)")

    import pandas as pd
    from utils.capital_cache import (
        obter_dados_capital,
        salvar_cache,
        limpar_cache,
        cache_existe,
    )

    resultados = []

    # Limpar estado
    limpar_cache()

    # Teste 1: Obter dados sem cache (deve falhar graciosamente)
    resultado = obter_dados_capital()
    # Pode falhar se GitHub nao estiver acessivel, mas nao deve dar excecao
    passou = isinstance(resultado.mensagem, str)
    print_result("obter_dados_capital sem cache", passou, resultado.mensagem[:60])
    resultados.append(passou)

    # Teste 2: Criar cache manualmente e testar leitura
    df_teste = pd.DataFrame({
        "Periodo": ["202312"],
        "CodInst": [1],
        "NomeInstituicao": ["Banco Teste"],
        "Capital Principal": [1000],
    })
    salvar_cache(df_teste, fonte="teste_pipeline")

    resultado = obter_dados_capital()
    passou = resultado.sucesso and resultado.fonte == "cache_local"
    print_result("obter_dados_capital com cache", passou, resultado.fonte)
    resultados.append(passou)

    # Teste 3: Verificar dados retornados
    if resultado.df is not None:
        passou = len(resultado.df) == 1 and resultado.df["NomeInstituicao"].iloc[0] == "Banco Teste"
        print_result("Dados corretos", passou)
        resultados.append(passou)

    # Limpar
    limpar_cache()

    return all(resultados)


def run_all_tests():
    """Executa todos os testes."""
    print("\n")
    print("*" * 60)
    print(" TESTES DO SISTEMA DE CACHE DE CAPITAL")
    print(f" Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 60)

    resultados = {}

    resultados["imports"] = test_imports()
    resultados["config"] = test_config()
    resultados["storage"] = test_storage_operations()
    resultados["extractor_validation"] = test_extractor_validation()
    resultados["gerar_periodos"] = test_orchestrator_gerar_periodos()
    resultados["cache_result"] = test_orchestrator_cache_result()
    resultados["full_pipeline"] = test_full_pipeline()

    # Resumo
    print_header("RESUMO")

    total = len(resultados)
    passou = sum(1 for v in resultados.values() if v)
    falhou = total - passou

    for nome, resultado in resultados.items():
        status = "PASSOU" if resultado else "FALHOU"
        print(f"  {nome}: {status}")

    print("\n" + "-" * 40)
    print(f"  Total: {total} | Passou: {passou} | Falhou: {falhou}")
    print("-" * 40)

    if falhou == 0:
        print("\n  TODOS OS TESTES PASSARAM\n")
        return 0
    else:
        print(f"\n  {falhou} TESTE(S) FALHARAM\n")
        return 1


if __name__ == "__main__":
    # Adicionar diretorio raiz ao path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    exit_code = run_all_tests()
    sys.exit(exit_code)
