"""
ifdata_cache - Sistema unificado de cache para dados do IFData/BCB

Arquitetura extensivel que suporta multiplos tipos de cache:
- principal: Dados gerais das instituicoes (Relatorios 1-4)
- capital: Dados de capital regulatorio (Relatorio 5)
- (extensivel para outros relatorios)

Uso basico:
    from utils.ifdata_cache import CacheManager

    # Obter gerenciador
    manager = CacheManager()

    # Carregar dados de capital
    resultado = manager.carregar("capital")
    if resultado.sucesso:
        df = resultado.dados

    # Carregar dados principais
    resultado = manager.carregar("principal")

    # Listar caches disponiveis
    print(manager.listar_caches())
"""

from .base import (
    CacheConfig,
    CacheResult,
    BaseCache,
)

from .manager import CacheManager

from .principal import PrincipalCache
from .capital import CapitalCache, CAMPOS_CAPITAL, gerar_periodos_trimestrais

# Funcoes de compatibilidade com sistema antigo
from .compat import (
    # Cache Principal
    baixar_cache_inicial,
    carregar_cache,
    salvar_cache,
    get_cache_info,
    # Cache Capital
    baixar_cache_capital_inicial,
    carregar_cache_capital,
    salvar_cache_capital,
    gerar_periodos_capital,
    processar_todos_periodos_capital,
    get_capital_cache_info,
    get_campos_capital_info,
    ler_info_cache_capital,
)

# Instancia global para uso simplificado
_manager = None


def get_manager() -> CacheManager:
    """Retorna instancia global do gerenciador de cache."""
    global _manager
    if _manager is None:
        _manager = CacheManager()
    return _manager


# Funcoes de conveniencia
def carregar(tipo: str) -> CacheResult:
    """Carrega cache do tipo especificado."""
    return get_manager().carregar(tipo)


def salvar(tipo: str, dados, **kwargs) -> CacheResult:
    """Salva dados no cache do tipo especificado."""
    return get_manager().salvar(tipo, dados, **kwargs)


def info(tipo: str = None) -> dict:
    """Retorna informacoes do cache (ou todos se tipo=None)."""
    return get_manager().info(tipo)


def limpar(tipo: str = None) -> CacheResult:
    """Limpa cache do tipo especificado (ou todos se tipo=None)."""
    return get_manager().limpar(tipo)


__all__ = [
    # Classes principais
    "CacheConfig",
    "CacheResult",
    "BaseCache",
    "CacheManager",
    # Implementacoes
    "PrincipalCache",
    "CapitalCache",
    "CAMPOS_CAPITAL",
    "gerar_periodos_trimestrais",
    # Funcoes de conveniencia
    "get_manager",
    "carregar",
    "salvar",
    "info",
    "limpar",
    # Compatibilidade - Principal
    "baixar_cache_inicial",
    "carregar_cache",
    "salvar_cache",
    "get_cache_info",
    # Compatibilidade - Capital
    "baixar_cache_capital_inicial",
    "carregar_cache_capital",
    "salvar_cache_capital",
    "gerar_periodos_capital",
    "processar_todos_periodos_capital",
    "get_capital_cache_info",
    "get_campos_capital_info",
    "ler_info_cache_capital",
]
