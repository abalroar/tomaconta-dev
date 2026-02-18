import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = PROJECT_ROOT / "data" / "balanco_4060_schema.json"
MAP_8_TO_10_PATH = PROJECT_ROOT / "data" / "balanco_4060_map_8_to_10.csv"


def _find_header_idx(path: Path) -> int:
    with path.open("r", encoding="latin-1", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("#DATA_BASE"):
                return i
    raise ValueError(f"Header #DATA_BASE not found in {path}")


def read_blo_prudencial_csv(path: Path) -> pd.DataFrame:
    """Lê CSV BLOPRUDENCIAL com header iniciado em '#DATA_BASE' e decimal com vírgula."""
    header_idx = _find_header_idx(path)
    df = pd.read_csv(path, sep=";", encoding="latin-1", skiprows=header_idx, decimal=",")
    df = df.rename(columns={"#DATA_BASE": "DATA_BASE"})
    df["CONTA"] = df["CONTA"].astype(str)
    return df


def load_schema(schema_path: Path = SCHEMA_PATH) -> List[dict]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_map_8_to_10(map_path: Path = MAP_8_TO_10_PATH) -> Dict[Tuple[str, str], str]:
    """Retorna mapeamento (CONTA_8, NOME_CONTA) -> CONTA_10."""
    if not map_path.exists():
        return {}
    df = pd.read_csv(map_path)
    mapping = {}
    for _, row in df.iterrows():
        key = (str(row["CONTA_8"]), str(row["NOME_CONTA"]))
        mapping[key] = str(row["CONTA_10"])
    return mapping


def normalize_conta_10(df: pd.DataFrame, mapping_8_to_10: Optional[Dict[Tuple[str, str], str]] = None) -> pd.DataFrame:
    """Normaliza coluna CONTA para 10 dígitos. Usa mapping 8->10 por NOME_CONTA quando aplicável."""
    df = df.copy()
    df["CONTA"] = df["CONTA"].astype(str)
    lengths = df["CONTA"].map(len)

    if (lengths == 10).all():
        df["CONTA_10"] = df["CONTA"]
        return df

    if mapping_8_to_10 is None:
        mapping_8_to_10 = load_map_8_to_10()

    def map_row(row):
        conta = str(row["CONTA"])
        if len(conta) == 10:
            return conta
        if len(conta) == 8:
            key = (conta, str(row.get("NOME_CONTA", "")))
            return mapping_8_to_10.get(key)
        return None

    df["CONTA_10"] = df.apply(map_row, axis=1)
    return df


def build_balanco_padronizado(
    df: pd.DataFrame,
    schema: List[dict],
    cod_congl: str,
    data_base: str,
) -> pd.DataFrame:
    """Retorna balanço padronizado para (COD_CONGL, DATA_BASE) com NaN quando ausente."""
    df = df.copy()
    df["COD_CONGL"] = df["COD_CONGL"].astype(str)
    df["DATA_BASE"] = df["DATA_BASE"].astype(str)

    if "CONTA_10" not in df.columns:
        df = normalize_conta_10(df)

    df_sel = df[(df["COD_CONGL"] == str(cod_congl)) & (df["DATA_BASE"] == str(data_base))].copy()

    out_rows = []
    for row in schema:
        contas = set(row.get("contas", []))
        if df_sel.empty or not contas:
            saldo = pd.NA
        else:
            mask = df_sel["CONTA_10"].isin(contas)
            if not mask.any():
                saldo = pd.NA
            else:
                saldo = df_sel.loc[mask, "SALDO"].sum()
        out_rows.append(
            {
                "id": row.get("id"),
                "section": row.get("section"),
                "level": row.get("level"),
                "label": row.get("label"),
                "conta_base": row.get("conta_base"),
                "saldo": saldo,
            }
        )

    return pd.DataFrame(out_rows)
