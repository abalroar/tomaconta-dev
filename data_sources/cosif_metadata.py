from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

CACHE_PATH = Path("data/cache/cosif/cosif_metadata.json")


def ifdata_to_cosif_code(n: str | int) -> str:
    """Converte conta IFData (10 dígitos) para formato COSIF: A.B.C.DE.FG.HI-J."""
    digits = re.sub(r"\D", "", str(n or ""))
    if len(digits) != 10:
        raise ValueError(f"Conta IFData inválida (esperado 10 dígitos): {n!r}")
    a, b, c, de, fg, hi, j = digits[0], digits[1], digits[2], digits[3:5], digits[5:7], digits[7:9], digits[9]
    return f"{a}.{b}.{c}.{de}.{fg}.{hi}-{j}"


def normalize_digits(code: str | int) -> str:
    return re.sub(r"\D", "", str(code or ""))


def _ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict[str, Dict[str, str]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    items = payload.get("items")
    return items if isinstance(items, dict) else {}


def _save_cache(items: Dict[str, Dict[str, str]]) -> None:
    _ensure_cache_dir()
    CACHE_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "source": "cosif_public_site",
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _request_text(url: str, timeout_s: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; toma.conta/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _strip_tags(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_labeled_field(text: str, labels: Iterable[str]) -> str:
    for label in labels:
        pattern = rf"(?:^|\s){re.escape(label)}\s*[:\-]\s*(.+?)(?=\s(?:T[íi]tulo|Fun[cç][aã]o|Base normativa|Observa[cç][õo]es?|Conta|$)\b|$)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .;")
    return ""


def _candidate_urls(cosif_code: str) -> list[str]:
    q = urllib.parse.quote(cosif_code)
    return [
        f"https://www3.bcb.gov.br/aplica/cosif/conta/{q}",
        f"https://www3.bcb.gov.br/aplica/cosif/manual/contas/{q}.htm",
        f"https://www3.bcb.gov.br/aplica/cosif/manual/contas/{q}.html",
        f"https://www3.bcb.gov.br/aplica/cosif?conta={q}",
        f"https://www3.bcb.gov.br/aplica/cosif?codigo={q}",
    ]


def fetch_cosif_metadata(cosif_code: str) -> Dict[str, str]:
    """Busca metadados oficiais COSIF para uma conta pontuada (quando disponível)."""
    normalized = normalize_digits(cosif_code)
    if len(normalized) != 10:
        raise ValueError(f"Código COSIF inválido: {cosif_code!r}")

    code = ifdata_to_cosif_code(normalized)
    last_error = ""
    for url in _candidate_urls(code):
        try:
            html = _request_text(url)
        except Exception as exc:
            last_error = str(exc)
            continue

        text = _strip_tags(html)
        titulo = _extract_labeled_field(text, ["Título", "Titulo"]) or ""
        funcao = _extract_labeled_field(text, ["Função", "Funcao", "Finalidade"]) or ""
        base_normativa = _extract_labeled_field(text, ["Base normativa", "Base legal", "Normativo"]) or ""

        # fallback: se não tiver label explícito para título, usa descrição central quando houver
        if not titulo:
            m_title = re.search(rf"{re.escape(code)}\s+([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ0-9\-\s]{{8,}})", text)
            if m_title:
                titulo = m_title.group(1).strip()

        if titulo or funcao or base_normativa:
            return {
                "cosif_code": code,
                "titulo": titulo,
                "funcao": funcao,
                "base_normativa": base_normativa,
                "source_url": url,
                "source_status": "ok",
            }

    return {
        "cosif_code": code,
        "titulo": "",
        "funcao": "",
        "base_normativa": "",
        "source_url": "",
        "source_status": f"not_found_or_unreachable:{last_error}" if last_error else "not_found",
    }


def get_cosif_metadata_for_accounts(accounts_ifdata: Iterable[str | int], force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    """Retorna metadados por conta IFData (10 dígitos), com cache persistido."""
    cache = {} if force_refresh else _load_cache()
    changed = False

    for account in accounts_ifdata:
        digits = normalize_digits(account)
        if len(digits) != 10:
            continue
        if digits in cache and cache[digits].get("source_status") == "ok" and not force_refresh:
            continue
        cache[digits] = fetch_cosif_metadata(ifdata_to_cosif_code(digits))
        changed = True

    if changed:
        _save_cache(cache)

    return cache
