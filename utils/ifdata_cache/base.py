"""
base.py - Classes base para o sistema de cache

Define a interface comum para todos os tipos de cache.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ifdata_cache")


@dataclass
class CacheConfig:
    """Configuracao de um tipo de cache."""

    # Identificador unico do cache
    nome: str

    # Descricao para logs e UI
    descricao: str

    # Diretorio relativo para armazenamento (dentro de data/cache/)
    subdir: str

    # Nome do arquivo de dados
    arquivo_dados: str = "data.parquet"

    # Nome do arquivo de metadados
    arquivo_metadata: str = "metadata.json"

    # URL base para download do GitHub (None = sem suporte remoto)
    github_url_base: Optional[str] = None

    # Tempo maximo de cache em horas (None = sem expiracao)
    max_idade_horas: Optional[float] = 168.0  # 7 dias

    # Colunas obrigatorias para validacao
    colunas_obrigatorias: List[str] = field(default_factory=lambda: ["Periodo", "CodInst"])

    # Mapeamento de campos para extracao (nome_api -> nome_exibicao)
    campos_mapeamento: Dict[str, str] = field(default_factory=dict)

    # URL da API para extracao (None = sem suporte a extracao direta)
    api_url: Optional[str] = None

    # Tipo de relatorio IFData (1-5)
    relatorio_tipo: Optional[int] = None


@dataclass
class CacheResult:
    """Resultado de uma operacao de cache."""

    sucesso: bool
    mensagem: str
    dados: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    fonte: str = "nenhum"  # cache_local, github, api, nenhum

    def __repr__(self):
        status = "OK" if self.sucesso else "ERRO"
        n_registros = len(self.dados) if self.dados is not None else 0
        return f"CacheResult({status}, fonte={self.fonte}, registros={n_registros})"


class BaseCache(ABC):
    """Classe base abstrata para implementacoes de cache."""

    def __init__(self, config: CacheConfig, base_dir: Path):
        """
        Args:
            config: Configuracao do cache
            base_dir: Diretorio base do projeto (onde fica data/)
        """
        self.config = config
        self.base_dir = base_dir
        self.cache_dir = base_dir / "data" / "cache" / config.subdir
        self.arquivo_dados = self.cache_dir / config.arquivo_dados
        self.arquivo_metadata = self.cache_dir / config.arquivo_metadata

        # Prefixo para logs
        self._log_prefix = f"[CACHE:{config.nome.upper()}]"

    def _log(self, nivel: str, msg: str):
        """Log com prefixo padronizado."""
        getattr(logger, nivel)(f"{self._log_prefix} {msg}")

    def _garantir_diretorio(self):
        """Cria diretorio de cache se nao existir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # OPERACOES DE CACHE LOCAL
    # =========================================================================

    def existe(self) -> bool:
        """Verifica se cache local existe."""
        return self.arquivo_dados.exists() and self.arquivo_metadata.exists()

    def carregar_local(self) -> CacheResult:
        """Carrega dados do cache local."""
        if not self.existe():
            return CacheResult(
                sucesso=False,
                mensagem="Cache local nao existe",
                fonte="nenhum"
            )

        try:
            # Carregar dados
            df = pd.read_parquet(self.arquivo_dados)

            # Carregar metadata
            with open(self.arquivo_metadata, "r") as f:
                metadata = json.load(f)

            # Validar dados
            valido, msg = self._validar_dados(df)
            if not valido:
                return CacheResult(
                    sucesso=False,
                    mensagem=f"Cache corrompido: {msg}",
                    fonte="nenhum"
                )

            self._log("info", f"Cache local carregado: {len(df)} registros")

            return CacheResult(
                sucesso=True,
                mensagem=f"Carregado do cache local: {len(df)} registros",
                dados=df,
                metadata=metadata,
                fonte="cache_local"
            )

        except Exception as e:
            self._log("error", f"Erro ao carregar cache local: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro ao carregar: {e}",
                fonte="nenhum"
            )

    def salvar_local(
        self,
        dados: pd.DataFrame,
        fonte: str = "desconhecida",
        info_extra: Optional[Dict] = None
    ) -> CacheResult:
        """Salva dados no cache local."""
        # Validar dados
        valido, msg = self._validar_dados(dados)
        if not valido:
            return CacheResult(
                sucesso=False,
                mensagem=f"Dados invalidos: {msg}",
                fonte="nenhum"
            )

        try:
            self._garantir_diretorio()

            # Salvar dados
            dados.to_parquet(self.arquivo_dados, index=False)

            # Criar metadata
            metadata = {
                "timestamp_salvamento": datetime.now().isoformat(),
                "fonte": fonte,
                "total_registros": len(dados),
                "colunas": list(dados.columns),
            }

            # Adicionar periodos se disponivel
            if "Periodo" in dados.columns:
                periodos = sorted(dados["Periodo"].unique().tolist())
                metadata["periodos"] = [str(p) for p in periodos]
                metadata["total_periodos"] = len(periodos)

            # Adicionar info extra
            if info_extra:
                metadata["extra"] = info_extra

            # Salvar metadata
            with open(self.arquivo_metadata, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Verificar que foi salvo corretamente
            if not self.arquivo_dados.exists():
                raise IOError("Arquivo de dados nao foi criado")

            tamanho = self.arquivo_dados.stat().st_size
            self._log("info", f"Cache salvo: {len(dados)} registros, {tamanho:,} bytes")

            return CacheResult(
                sucesso=True,
                mensagem=f"Salvo com sucesso: {len(dados)} registros",
                dados=dados,
                metadata=metadata,
                fonte="cache_local"
            )

        except Exception as e:
            self._log("error", f"Erro ao salvar cache: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro ao salvar: {e}",
                fonte="nenhum"
            )

    def limpar_local(self) -> CacheResult:
        """Remove arquivos de cache local."""
        removidos = []

        try:
            if self.arquivo_dados.exists():
                self.arquivo_dados.unlink()
                removidos.append(self.config.arquivo_dados)

            if self.arquivo_metadata.exists():
                self.arquivo_metadata.unlink()
                removidos.append(self.config.arquivo_metadata)

            if removidos:
                self._log("info", f"Cache limpo: {', '.join(removidos)}")
                return CacheResult(
                    sucesso=True,
                    mensagem=f"Removidos: {', '.join(removidos)}",
                    fonte="nenhum"
                )
            else:
                return CacheResult(
                    sucesso=True,
                    mensagem="Cache ja estava vazio",
                    fonte="nenhum"
                )

        except Exception as e:
            self._log("error", f"Erro ao limpar cache: {e}")
            return CacheResult(
                sucesso=False,
                mensagem=f"Erro ao limpar: {e}",
                fonte="nenhum"
            )

    def get_info(self) -> Dict[str, Any]:
        """Retorna informacoes sobre o cache."""
        info = {
            "nome": self.config.nome,
            "descricao": self.config.descricao,
            "existe": self.existe(),
            "diretorio": str(self.cache_dir),
            "arquivo_dados": str(self.arquivo_dados),
        }

        if self.existe():
            try:
                with open(self.arquivo_metadata, "r") as f:
                    metadata = json.load(f)

                info.update({
                    "tamanho_bytes": self.arquivo_dados.stat().st_size,
                    "timestamp_salvamento": metadata.get("timestamp_salvamento"),
                    "fonte": metadata.get("fonte"),
                    "total_registros": metadata.get("total_registros"),
                    "total_periodos": metadata.get("total_periodos"),
                    "periodos": metadata.get("periodos"),
                })

                # Calcular idade do cache
                if metadata.get("timestamp_salvamento"):
                    ts = datetime.fromisoformat(metadata["timestamp_salvamento"])
                    idade = datetime.now() - ts
                    info["idade_horas"] = idade.total_seconds() / 3600

            except Exception as e:
                info["erro_metadata"] = str(e)

        return info

    def cache_valido(self) -> Tuple[bool, str]:
        """Verifica se cache existe e nao expirou."""
        if not self.existe():
            return False, "Cache nao existe"

        info = self.get_info()

        # Verificar expiracao
        if self.config.max_idade_horas and "idade_horas" in info:
            if info["idade_horas"] > self.config.max_idade_horas:
                return False, f"Cache expirado (idade: {info['idade_horas']:.1f}h)"

        return True, "Cache valido"

    # =========================================================================
    # VALIDACAO
    # =========================================================================

    def _validar_dados(self, dados: Any) -> Tuple[bool, str]:
        """Valida dados antes de salvar/apos carregar."""
        if dados is None:
            return False, "Dados sao None"

        if not isinstance(dados, pd.DataFrame):
            return False, f"Esperado DataFrame, recebido {type(dados)}"

        if dados.empty:
            return False, "DataFrame vazio"

        # Verificar colunas obrigatorias
        for col in self.config.colunas_obrigatorias:
            if col not in dados.columns:
                return False, f"Coluna obrigatoria ausente: {col}"

        return True, "OK"

    # =========================================================================
    # METODOS ABSTRATOS (implementados pelas subclasses)
    # =========================================================================

    @abstractmethod
    def baixar_remoto(self) -> CacheResult:
        """Baixa dados de fonte remota (GitHub/API)."""
        pass

    @abstractmethod
    def extrair_periodo(self, periodo: str, **kwargs) -> CacheResult:
        """Extrai dados de um periodo especifico da API."""
        pass

    # =========================================================================
    # METODO PRINCIPAL DE CARREGAMENTO
    # =========================================================================

    def carregar(self, forcar_remoto: bool = False) -> CacheResult:
        """Carrega dados usando a melhor fonte disponivel.

        Ordem de prioridade:
        1. Cache local (se valido e nao forcar_remoto)
        2. Download remoto (GitHub)
        3. Falha

        Args:
            forcar_remoto: Se True, ignora cache local

        Returns:
            CacheResult com dados ou erro
        """
        # Tentar cache local primeiro
        if not forcar_remoto:
            valido, msg = self.cache_valido()
            if valido:
                resultado = self.carregar_local()
                if resultado.sucesso:
                    self._log("info", f"Usando cache local: {msg}")
                    return resultado

        # Tentar baixar do remoto
        self._log("info", "Tentando baixar de fonte remota...")
        resultado = self.baixar_remoto()

        if resultado.sucesso:
            # Salvar localmente
            self.salvar_local(resultado.dados, fonte=resultado.fonte)
            return resultado

        # Tentar cache local mesmo expirado como fallback
        if self.existe():
            self._log("warning", "Usando cache local expirado como fallback")
            resultado = self.carregar_local()
            if resultado.sucesso:
                resultado.fonte = "cache_local_expirado"
                return resultado

        return CacheResult(
            sucesso=False,
            mensagem="Nenhuma fonte de dados disponivel",
            fonte="nenhum"
        )
