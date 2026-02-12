"""Loader e cache local para Conglomerados Prudenciais (BLOPRUDENCIAL)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger("ifdata_cache.bloprudencial")

BASE_URL = "https://www.bcb.gov.br/content/estabilidadefinanceira/cosif/Conglomerados-prudenciais"
FILE_SUFFIX = "BLOPRUDENCIAL.csv.zip"
DEFAULT_CACHE_DIR = Path("data/cache/bcb_bloprudencial")
LOADER_VERSION = "v1"


# -----------------------------------------------------------------------------
# Utilitários base
# -----------------------------------------------------------------------------
def _validate_yyyymm(yyyymm: str) -> str:
    yyyymm = str(yyyymm).strip()
    if not re.fullmatch(r"\d{6}", yyyymm):
        raise ValueError(f"yyyymm inválido: {yyyymm}. Esperado formato YYYYMM")
    ano = int(yyyymm[:4])
    mes = int(yyyymm[4:6])
    if ano < 1900 or mes < 1 or mes > 12:
        raise ValueError(f"yyyymm inválido: {yyyymm}")
    return yyyymm


def _ensure_dirs(base_cache_dir: Path) -> Dict[str, Path]:
    zips = base_cache_dir / "zips"
    csv = base_cache_dir / "csv"
    meta = base_cache_dir / "meta"
    zips.mkdir(parents=True, exist_ok=True)
    csv.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    return {"base": base_cache_dir, "zips": zips, "csv": csv, "meta": meta}


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _guess_encoding(sample: bytes) -> str:
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _guess_delimiter(header_text: str) -> str:
    count_semicolon = header_text.count(";")
    count_comma = header_text.count(",")
    return ";" if count_semicolon >= count_comma else ","


def _coerce_numeric_columns(df: pd.DataFrame) -> List[str]:
    converted: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        serie = df[col]
        if serie.dtype != object:
            continue

        s = serie.astype(str).str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
        if s.dropna().empty:
            continue

        s_norm = (
            s.str.replace(r"\s+", "", regex=True)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        parsed = pd.to_numeric(s_norm, errors="coerce")
        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.80:
            df[col] = parsed
            converted.append(col)

    return converted


# -----------------------------------------------------------------------------
# API pública
# -----------------------------------------------------------------------------
def build_bloprudencial_url(yyyymm: str) -> str:
    """Retorna a URL BLOPRUDENCIAL para uma competência YYYYMM."""
    yyyymm = _validate_yyyymm(yyyymm)
    return f"{BASE_URL}/{yyyymm}{FILE_SUFFIX}"


def download_bloprudencial_zip(
    yyyymm: str,
    cache_dir: str | Path,
    force_refresh: bool = False,
    timeout: int = 120,
) -> Path:
    """Baixa o ZIP BLOPRUDENCIAL para cache local (ou reutiliza cache hit)."""
    yyyymm = _validate_yyyymm(yyyymm)
    dirs = _ensure_dirs(Path(cache_dir))
    url = build_bloprudencial_url(yyyymm)
    zip_path = dirs["zips"] / f"{yyyymm}{FILE_SUFFIX}"

    if zip_path.exists() and not force_refresh:
        logger.info("[BLOPRUDENCIAL] cache hit ZIP %s (%d bytes)", zip_path.name, zip_path.stat().st_size)
        return zip_path

    t0 = time.perf_counter()
    logger.info("[BLOPRUDENCIAL] cache miss ZIP %s -> GET %s", zip_path.name, url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    dt = time.perf_counter() - t0
    logger.info(
        "[BLOPRUDENCIAL] download concluído %s (%d bytes) em %.2fs",
        zip_path.name,
        zip_path.stat().st_size,
        dt,
    )
    return zip_path


def extract_bloprudencial_csv(zip_path: Path, extracted_dir: str | Path) -> Path:
    """Extrai CSV interno do ZIP para diretório local, reutilizando se ZIP não mudou."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP não encontrado: {zip_path}")

    extracted_dir = Path(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    zip_hash = _file_sha256(zip_path)
    sig_path = extracted_dir / f"{zip_path.stem}.sha256"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"ZIP sem CSV interno: {zip_path}")
        internal_csv = csv_names[0]

    target_csv = extracted_dir / Path(internal_csv).name
    if target_csv.exists() and sig_path.exists() and sig_path.read_text().strip() == zip_hash:
        logger.info("[BLOPRUDENCIAL] CSV extraído reutilizado: %s", target_csv.name)
        return target_csv

    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            content = zf.read(name)
            out = extracted_dir / Path(name).name
            out.write_bytes(content)
            extracted_count += 1

    sig_path.write_text(zip_hash)
    logger.info(
        "[BLOPRUDENCIAL] extração concluída zip=%s csv_interno=%s arquivos_extraidos=%d",
        zip_path.name,
        Path(internal_csv).name,
        extracted_count,
    )
    return target_csv


def inspect_bloprudencial_csv(csv_path: Path) -> Dict[str, Any]:
    """Inspeciona rapidamente CSV BLOPRUDENCIAL (header/delimiter/encoding/linhas/dtypes)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    with csv_path.open("rb") as f:
        sample = f.read(16384)

    encoding_guess = _guess_encoding(sample)
    text_sample = sample.decode(encoding_guess, errors="replace")
    first_line = text_sample.splitlines()[0] if text_sample.splitlines() else ""
    delimiter_guess = _guess_delimiter(first_line)

    line_count = 0
    with csv_path.open("rb") as f:
        for _ in f:
            line_count += 1

    encodings_try = [encoding_guess] + [e for e in ("utf-8", "latin-1", "utf-8-sig") if e != encoding_guess]
    sample_df: Optional[pd.DataFrame] = None
    used_encoding = encoding_guess
    for enc in encodings_try:
        try:
            sample_df = pd.read_csv(csv_path, sep=delimiter_guess, encoding=enc, nrows=200)
            used_encoding = enc
            break
        except Exception:
            continue

    if sample_df is None:
        raise ValueError(f"Falha ao inspecionar CSV: {csv_path}")

    info = {
        "path": str(csv_path),
        "first_line": first_line,
        "delimiter_guess": delimiter_guess,
        "encoding_guess": encoding_guess,
        "encoding_used": used_encoding,
        "line_count": line_count,
        "columns": sample_df.columns.tolist(),
        "dtypes_initial": {c: str(t) for c, t in sample_df.dtypes.items()},
    }

    logger.info(
        "[BLOPRUDENCIAL] inspect csv=%s encoding=%s delimiter=%s linhas=%d colunas=%d",
        csv_path.name,
        info["encoding_used"],
        info["delimiter_guess"],
        info["line_count"],
        len(info["columns"]),
    )
    return info


def load_bloprudencial_df(
    yyyymm: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Orquestra download->extração->inspeção->load DataFrame numérico."""
    yyyymm = _validate_yyyymm(yyyymm)
    dirs = _ensure_dirs(Path(cache_dir))

    zip_path = download_bloprudencial_zip(yyyymm, cache_dir=dirs["base"], force_refresh=force_refresh)
    csv_path = extract_bloprudencial_csv(zip_path, extracted_dir=dirs["csv"])
    inspect = inspect_bloprudencial_csv(csv_path)

    delimiter = inspect["delimiter_guess"]
    encodings_try = [inspect["encoding_used"], "utf-8", "latin-1", "utf-8-sig"]
    encodings_try = list(dict.fromkeys(encodings_try))

    df: Optional[pd.DataFrame] = None
    last_exc: Optional[Exception] = None
    for enc in encodings_try:
        try:
            df = pd.read_csv(csv_path, sep=delimiter, encoding=enc)
            logger.info("[BLOPRUDENCIAL] read_csv ok encoding=%s linhas=%d", enc, len(df))
            break
        except Exception as exc:
            last_exc = exc
            continue

    if df is None:
        raise ValueError(f"Falha ao carregar CSV com encodings tentados {encodings_try}: {last_exc}")

    converted_cols = _coerce_numeric_columns(df)
    logger.info("[BLOPRUDENCIAL] colunas convertidas para numérico: %d", len(converted_cols))
    if converted_cols:
        logger.debug("[BLOPRUDENCIAL] colunas convertidas: %s", converted_cols)

    df.attrs["bloprudencial"] = {
        "yyyymm": yyyymm,
        "zip_path": str(zip_path),
        "csv_path": str(csv_path),
        "inspect": inspect,
        "converted_numeric_columns": converted_cols,
        "loader_version": LOADER_VERSION,
    }
    return df


def preload_bloprudencial(
    yyyymm_list: List[str],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Pré-carrega lista de competências BLOPRUDENCIAL com log por competência."""
    result: Dict[str, pd.DataFrame] = {}
    total = len(yyyymm_list)
    for i, ym in enumerate(yyyymm_list, start=1):
        logger.info("[BLOPRUDENCIAL] preload %d/%d competência=%s", i, total, ym)
        result[ym] = load_bloprudencial_df(ym, cache_dir=cache_dir, force_refresh=force_refresh)
    return result


try:
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def load_bloprudencial_df_cached(
        yyyymm: str,
        cache_dir: str = str(DEFAULT_CACHE_DIR),
        force_refresh: bool = False,
        loader_version: str = LOADER_VERSION,
    ) -> pd.DataFrame:
        """Wrapper cacheado para uso no Streamlit (chaveado por competência/versão)."""
        _ = loader_version
        return load_bloprudencial_df(yyyymm=yyyymm, cache_dir=cache_dir, force_refresh=force_refresh)

except Exception:
    def load_bloprudencial_df_cached(
        yyyymm: str,
        cache_dir: str = str(DEFAULT_CACHE_DIR),
        force_refresh: bool = False,
        loader_version: str = LOADER_VERSION,
    ) -> pd.DataFrame:
        _ = loader_version
        return load_bloprudencial_df(yyyymm=yyyymm, cache_dir=cache_dir, force_refresh=force_refresh)
