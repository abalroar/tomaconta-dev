import json
import re
from pathlib import Path

import pandas as pd

CANON_PATH = Path("202509BLOPRUDENCIAL 2.CSV")
LEGACY_PATH = Path("202412BLOPRUDENCIAL.CSV")


def read_blo(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="latin-1", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("#DATA_BASE"):
                header_idx = i
                break
        else:
            raise ValueError(f"Header #DATA_BASE not found in {path}")
    df = pd.read_csv(path, sep=";", encoding="latin-1", skiprows=header_idx, decimal=",")
    df = df.rename(columns={"#DATA_BASE": "DATA_BASE"})
    df["CONTA"] = df["CONTA"].astype(str)
    return df


def build_group_map(contas_set, group_set, root_by_digit):
    # Assign each account to nearest group (by zero suffix) or root by first digit
    group_map = {}
    for c in contas_set:
        grp = None
        for k in range(1, 10):
            cand = c[:-k] + ("0" * k)
            if cand == c:
                continue
            if cand in group_set:
                grp = cand
                break
        if not grp:
            grp = root_by_digit.get(c[0])
        group_map[c] = grp
    return group_map


def slugify_section(section: str) -> str:
    s = section.lower()
    s = s.replace("patrimônio líquido", "pl")
    s = s.replace("patrimonio liquido", "pl")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main():
    df = read_blo(CANON_PATH)
    df["CONTA"] = df["CONTA"].astype(str).str.zfill(10)
    df["COD_CONGL"] = df["COD_CONGL"].astype(str)

    df["SECTION_DIGIT"] = df["CONTA"].str[0]
    df_bal = df[df["SECTION_DIGIT"].isin(["1", "2", "4", "6"])].copy()

    # frequency by COD_CONGL
    freq = df_bal.groupby("CONTA")["COD_CONGL"].nunique().sort_values(ascending=False)

    top_n = 60
    top_accounts = freq.head(top_n).index.tolist()

    contas_set = set(df_bal["CONTA"].unique())

    # group accounts = trailing zeros >=5
    u = df_bal[["CONTA", "NOME_CONTA"]].drop_duplicates().copy()
    u["TZ"] = 10 - u["CONTA"].str.rstrip("0").str.len()
    group_set = set(u[u["TZ"] >= 5]["CONTA"])

    root_by_digit = {
        "1": "1000000009",
        "2": "2000000008",
        "4": "4000000006",
        "6": "6000000004",
    }
    roots = [v for v in root_by_digit.values() if v in contas_set]

    # select details and group ancestors
    selected_detail = [
        c for c in top_accounts if c in contas_set and c not in group_set and c not in roots
    ]

    selected_groups = set()
    for c in top_accounts:
        if c in group_set:
            selected_groups.add(c)
        for k in range(1, 10):
            cand = c[:-k] + ("0" * k)
            if cand == c:
                continue
            if cand in group_set:
                selected_groups.add(cand)
                break

    u_idx = df_bal[["CONTA", "NOME_CONTA"]].drop_duplicates().set_index("CONTA")
    group_map = build_group_map(contas_set, group_set, root_by_digit)

    rows = []
    sec_by_digit = {"1": "Ativo", "2": "Ativo", "4": "Passivo", "6": "Patrimônio Líquido"}

    def add_row(code, section, level, label, contas):
        slug = f"{slugify_section(section)}_{code}"
        rows.append(
            {
                "id": slug,
                "section": section,
                "level": level,
                "label": label,
                "conta_base": code,
                "contas": sorted(set(contas)),
            }
        )

    # level 1: roots
    for r in roots:
        section = sec_by_digit.get(r[0])
        label = u_idx.loc[r, "NOME_CONTA"] if r in u_idx.index else r
        contas = [c for c in contas_set if c[0] == r[0]]
        add_row(r, section, 1, label, contas)

    # level 2: groups
    for g in sorted(selected_groups):
        if g not in contas_set or g in roots:
            continue
        section = sec_by_digit.get(g[0])
        label = u_idx.loc[g, "NOME_CONTA"] if g in u_idx.index else g
        contas = [c for c in contas_set if group_map.get(c) == g]
        if not contas:
            continue
        add_row(g, section, 2, label, contas)

    # level 3: detail
    for c in sorted(selected_detail):
        section = sec_by_digit.get(c[0])
        label = u_idx.loc[c, "NOME_CONTA"] if c in u_idx.index else c
        add_row(c, section, 3, label, [c])

    sec_order = {"Ativo": 1, "Passivo": 2, "Patrimônio Líquido": 3}
    rows.sort(key=lambda r: (sec_order.get(r["section"], 9), r["level"], r["conta_base"]))

    Path("data/balanco_4060_schema.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table_rows = []
    for r in rows:
        for conta in r["contas"]:
            nome = u_idx.loc[conta, "NOME_CONTA"] if conta in u_idx.index else None
            table_rows.append(
                {
                    "Section": r["section"],
                    "Level": r["level"],
                    "Label": r["label"],
                    "CONTA": conta,
                    "COSIF": None,
                    "NOME_CONTA": nome,
                    "Observacoes": None,
                    "ContaBase": r["conta_base"],
                    "Id": r["id"],
                }
            )
    pd.DataFrame(table_rows).to_csv("data/balanco_4060_schema_table.csv", index=False)

    # mapping 8->10 by unique NOME_CONTA
    df8 = read_blo(LEGACY_PATH)
    df8["CONTA"] = df8["CONTA"].astype(str)
    u8 = df8[["CONTA", "NOME_CONTA"]].drop_duplicates()
    name_to_code10 = (
        df_bal[["CONTA", "NOME_CONTA"]].drop_duplicates().groupby("NOME_CONTA")["CONTA"].nunique()
    )
    unique_names = set(name_to_code10[name_to_code10 == 1].index)
    u8u = u8[u8["NOME_CONTA"].isin(unique_names)]
    u10 = df_bal[["CONTA", "NOME_CONTA"]].drop_duplicates()
    mapping = u8u.merge(u10, on="NOME_CONTA", suffixes=("_8", "_10"))
    name_to_code8 = u8.groupby("NOME_CONTA")["CONTA"].nunique()
    mapping = mapping[mapping["NOME_CONTA"].map(name_to_code8) == 1]
    mapping.to_csv("data/balanco_4060_map_8_to_10.csv", index=False)

    print(f"Schema rows: {len(rows)}")
    print(
        "Levels:",
        pd.Series([r["level"] for r in rows]).value_counts().sort_index().to_dict(),
    )
    print(f"Mapping 8->10 rows: {len(mapping)}")


if __name__ == "__main__":
    main()
