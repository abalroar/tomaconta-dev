"""
config.py - Configuracoes do sistema de cache de capital

Centraliza todas as constantes, caminhos e URLs usados pelo sistema.
"""

from pathlib import Path
from typing import Dict

# =============================================================================
# DIRETORIOS E CAMINHOS
# =============================================================================

# Diretorio raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Diretorio de cache
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "capital"

# Arquivos de cache
CACHE_DATA_FILE = CACHE_DIR / "capital_data.parquet"
CACHE_METADATA_FILE = CACHE_DIR / "capital_metadata.json"

# =============================================================================
# URLs REMOTAS (GITHUB)
# =============================================================================

# URL base do repositorio no GitHub
GITHUB_REPO = "abalroar/tomaconta"
GITHUB_RELEASE_TAG = "v1.0-cache"

# URLs de download do cache remoto
REMOTE_CACHE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/capital_data.parquet"
REMOTE_METADATA_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/capital_metadata.json"

# =============================================================================
# API DO BANCO CENTRAL (OLINDA)
# =============================================================================

BCB_BASE_URL = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata"

# Endpoints
BCB_CADASTRO_ENDPOINT = f"{BCB_BASE_URL}/IfDataCadastro"
BCB_VALORES_ENDPOINT = f"{BCB_BASE_URL}/IfDataValores"

# =============================================================================
# CONFIGURACOES DE REDE
# =============================================================================

# Timeouts em segundos
HTTP_TIMEOUT = 120
HTTP_RETRIES = 3
HTTP_BACKOFF = 2.0

# =============================================================================
# CONFIGURACOES DE CACHE
# =============================================================================

# Tempo maximo em horas antes de considerar cache obsoleto
CACHE_MAX_AGE_HOURS = 24 * 7  # 7 dias

# =============================================================================
# MAPEAMENTO DE CAMPOS DE CAPITAL
# =============================================================================

# Campos extraidos da API BCB -> Nome para exibicao
CAMPOS_CAPITAL: Dict[str, str] = {
    "Capital Principal para Comparação com RWA (a)": "Capital Principal",
    "Capital Complementar (b)": "Capital Complementar",
    "Patrimônio de Referência Nível I para Comparação com RWA (c) = (a) + (b)": "Patrimônio de Referência",
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

# Versao normalizada para match com API
CAMPOS_CAPITAL_NORMALIZADO: Dict[str, str] = {
    " ".join(k.split()): v for k, v in CAMPOS_CAPITAL.items()
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_PREFIX = "[CAPITAL_CACHE]"
