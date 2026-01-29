"""
capital_cache - Sistema robusto de cache de dados de capital

Modulos:
    config: Configuracoes, caminhos e constantes
    extractor: Extracao de dados do BCB e GitHub
    storage: Armazenamento e carregamento de cache local
    orchestrator: Coordenacao do fluxo entre fontes

Uso basico:
    from utils.capital_cache import obter_dados_capital, extrair_e_salvar_periodos

    # Obter dados (usa cache se disponivel, senao baixa do GitHub)
    resultado = obter_dados_capital()
    if resultado.sucesso:
        df = resultado.df
        print(f"Fonte: {resultado.fonte}")

    # Extrair novos periodos do BCB
    resultado = extrair_e_salvar_periodos(["202312", "202403"])
"""

from .orchestrator import (
    obter_dados_capital,
    extrair_e_salvar_periodos,
    gerar_periodos_trimestrais,
    get_info,
    limpar,
    CacheResult,
)

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

from .config import (
    CACHE_DIR,
    CACHE_DATA_FILE,
    CACHE_METADATA_FILE,
    REMOTE_CACHE_URL,
    CAMPOS_CAPITAL,
)

# Funcoes de compatibilidade com o sistema antigo
from .integration import (
    baixar_cache_capital_inicial,
    carregar_cache_capital,
    salvar_cache_capital,
    gerar_periodos_capital,
    processar_todos_periodos_capital,
    get_capital_cache_info,
    get_campos_capital_info,
    limpar_cache_capital,
)

__all__ = [
    # Orchestrator
    "obter_dados_capital",
    "extrair_e_salvar_periodos",
    "gerar_periodos_trimestrais",
    "get_info",
    "limpar",
    "CacheResult",
    # Storage
    "carregar_cache",
    "salvar_cache",
    "cache_existe",
    "get_cache_info",
    "limpar_cache",
    # Extractor
    "processar_periodo_capital",
    "baixar_cache_github",
    "validar_dataframe",
    "ExtractionError",
    "ValidationError",
    # Config
    "CACHE_DIR",
    "CACHE_DATA_FILE",
    "CACHE_METADATA_FILE",
    "REMOTE_CACHE_URL",
    "CAMPOS_CAPITAL",
    # Integration (compatibilidade)
    "baixar_cache_capital_inicial",
    "carregar_cache_capital",
    "salvar_cache_capital",
    "gerar_periodos_capital",
    "processar_todos_periodos_capital",
    "get_capital_cache_info",
    "get_campos_capital_info",
    "limpar_cache_capital",
]
