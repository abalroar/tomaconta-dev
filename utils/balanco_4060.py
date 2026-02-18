import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = PROJECT_ROOT / "data" / "balanco_4060_schema.json"
MAP_8_TO_10_PATH = PROJECT_ROOT / "data" / "balanco_4060_map_8_to_10.csv"
SCHEMA_RULES_PATH = PROJECT_ROOT / "data" / "balanco_4060_schema_rules.json"
NAME_MAP_PATH = PROJECT_ROOT / "data" / "ClassificacaoCompleta4060_mapeamento.xlsx"


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


def default_schema_rules() -> dict:
    return {
        "version": "v2",
        "description": "Balanço 4060 resumido com linhas derivadas e subgrupos por liquidez.",
        "sections_order": ["Ativo", "Passivo", "Patrimônio Líquido"],
        "lines": [
            # ATIVO - linha principais
            {"id": "ativo_liquidos", "label": "Ativos Líquidos", "section": "Ativo", "level": 1, "order": 10, "derived": True},
            {"id": "ativo_credito", "label": "Operações de Crédito", "section": "Ativo", "level": 1, "order": 20, "derived": True},
            {"id": "ativo_derivativos", "label": "Derivativos", "section": "Ativo", "level": 1, "order": 30, "derived": False, "rule": {"section_digits": ["1"], "name_regex": [r"DERIVAT"]}},
            {"id": "ativo_outros_creditos", "label": "Outros Créditos", "section": "Ativo", "level": 1, "order": 40, "derived": True},
            {"id": "ativo_permanente", "label": "Ativo Permanente", "section": "Ativo", "level": 1, "order": 50, "derived": True},
            {"id": "ativo_total", "label": "ATIVO TOTAL", "section": "Ativo", "level": 1, "order": 99, "derived": True, "is_total": True},

            # ATIVO - sublinhas (Ativos Líquidos)
            {"id": "ativo_caixa", "label": "Caixa e Disponibilidades", "section": "Ativo", "level": 2, "order": 11, "parent_id": "ativo_liquidos",
             "rule": {"section_digits": ["1"], "name_regex": [r"DISPONIBILIDADES", r"\\bCAIXA\\b"]}},
            {"id": "ativo_aplic_interfin", "label": "Aplicações Interfinanceiras", "section": "Ativo", "level": 2, "order": 12, "parent_id": "ativo_liquidos",
             "rule": {"section_digits": ["1"], "name_regex": [r"APLICAÇÕES INTERFINANCEIRAS", r"APLICACOES INTERFINANCEIRAS"]}},
            {"id": "ativo_tvm", "label": "Títulos e Valores Mobiliários", "section": "Ativo", "level": 2, "order": 13, "parent_id": "ativo_liquidos",
             "rule": {"section_digits": ["1"], "name_regex": [r"T[IÍ]TULOS E VALORES MOBILI[ÁA]RIOS"]}},

            # ATIVO - sublinhas (Crédito)
            {"id": "ativo_emprestimos", "label": "Empréstimos", "section": "Ativo", "level": 2, "order": 21, "parent_id": "ativo_credito",
             "rule": {"section_digits": ["1"], "name_regex": [r"EMPR[ÉE]STIMOS"]}},
            {"id": "ativo_financiamentos", "label": "Financiamentos", "section": "Ativo", "level": 2, "order": 22, "parent_id": "ativo_credito",
             "rule": {"section_digits": ["1"], "name_regex": [r"FINANCIAMENTOS"]}},
            {"id": "ativo_arrendamento", "label": "Arrendamento Financeiro", "section": "Ativo", "level": 2, "order": 23, "parent_id": "ativo_credito",
             "rule": {"section_digits": ["1"], "name_regex": [r"ARRENDAMENTO"]}},
            {"id": "ativo_credito_outros", "label": "Outras Operações de Crédito", "section": "Ativo", "level": 2, "order": 24, "parent_id": "ativo_credito",
             "rule": {"section_digits": ["1"], "name_regex": [r"CARACTER[IÍ]STICAS DE CONCESS[ÃA]O DE CR[ÉE]DITO", r"ADIANTAMENTOS A DEPOSITANTES"]}},

            # ATIVO - sublinhas (Outros Créditos)
            {"id": "ativo_credito_tributario", "label": "Crédito Tributário", "section": "Ativo", "level": 2, "order": 41, "parent_id": "ativo_outros_creditos",
             "rule": {"section_digits": ["1"], "name_regex": [r"IMPOSTOS E CONTRIBUI[CÇ][ÕO]ES", r"CR[ÉE]DITO PRESUMIDO", r"TRIBUT"]}},
            {"id": "ativo_depositos_judiciais", "label": "Depósitos Judiciais", "section": "Ativo", "level": 2, "order": 42, "parent_id": "ativo_outros_creditos",
             "rule": {"section_digits": ["1"], "name_regex": [r"DEP[ÓO]SITOS JUDICIAIS"]}},
            {"id": "ativo_bens_nao_uso", "label": "Bens Não de Uso Próprio", "section": "Ativo", "level": 2, "order": 43, "parent_id": "ativo_outros_creditos",
             "rule": {"section_digits": ["1"], "name_regex": [r"BENS.*N[AÃ]O DE USO", r"OUTROS VALORES E BENS"]}},
            {"id": "ativo_outros_creditos_residual", "label": "Outros Créditos (Residual)", "section": "Ativo", "level": 2, "order": 49, "parent_id": "ativo_outros_creditos",
             "rule": {"section_digits": ["1"], "residual": True}},

            # ATIVO - sublinhas (Permanente)
            {"id": "ativo_imobilizado", "label": "Imobilizado e Bens de Uso", "section": "Ativo", "level": 2, "order": 51, "parent_id": "ativo_permanente",
             "rule": {"section_digits": ["2"], "name_regex": [r"IMOBILIZAD", r"BENS DE USO", r"TERRENOS"]}},
            {"id": "ativo_intangivel", "label": "Intangível", "section": "Ativo", "level": 2, "order": 52, "parent_id": "ativo_permanente",
             "rule": {"section_digits": ["2"], "name_regex": [r"INTANG"]}},
            {"id": "ativo_diferido", "label": "Diferido / Software", "section": "Ativo", "level": 2, "order": 53, "parent_id": "ativo_permanente",
             "rule": {"section_digits": ["2"], "name_regex": [r"DIFERID", r"SOFTWARE", r"DESENVOLVIMENTO"]}},
            {"id": "ativo_permanente_residual", "label": "Permanente (Residual)", "section": "Ativo", "level": 2, "order": 59, "parent_id": "ativo_permanente",
             "rule": {"section_digits": ["2"], "residual": True}},

            # PASSIVO
            {"id": "passivo_depositos", "label": "Depósitos", "section": "Passivo", "level": 1, "order": 10, "derived": True},
            {"id": "passivo_titulos", "label": "Títulos Emitidos", "section": "Passivo", "level": 1, "order": 20, "derived": True},
            {"id": "passivo_captacoes", "label": "Captações / Empréstimos e Repasses", "section": "Passivo", "level": 1, "order": 30, "derived": True},
            {"id": "passivo_derivativos", "label": "Derivativos", "section": "Passivo", "level": 1, "order": 40, "derived": False, "rule": {"section_digits": ["4"], "name_regex": [r"DERIVAT"]}},
            {"id": "passivo_outras_obr", "label": "Outras Obrigações", "section": "Passivo", "level": 1, "order": 50, "derived": True},
            {"id": "passivo_total", "label": "PASSIVO TOTAL", "section": "Passivo", "level": 1, "order": 99, "derived": True, "is_total": True},

            # PASSIVO - sublinhas (Depósitos)
            {"id": "passivo_dep_vista", "label": "Depósitos à Vista", "section": "Passivo", "level": 2, "order": 11, "parent_id": "passivo_depositos",
             "rule": {"section_digits": ["4"], "name_regex": [r"DEP[ÓO]SITOS .*VISTA"]}},
            {"id": "passivo_dep_prazo", "label": "Depósitos a Prazo", "section": "Passivo", "level": 2, "order": 12, "parent_id": "passivo_depositos",
             "rule": {"section_digits": ["4"], "name_regex": [r"DEP[ÓO]SITOS .*PRAZO"]}},
            {"id": "passivo_dep_poupanca", "label": "Depósitos de Poupança", "section": "Passivo", "level": 2, "order": 13, "parent_id": "passivo_depositos",
             "rule": {"section_digits": ["4"], "name_regex": [r"POUPAN[CÇ]A"]}},
            {"id": "passivo_dep_interfin", "label": "Depósitos Interfinanceiros", "section": "Passivo", "level": 2, "order": 14, "parent_id": "passivo_depositos",
             "rule": {"section_digits": ["4"], "name_regex": [r"INTERFINANCEIR"]}},
            {"id": "passivo_dep_outros", "label": "Outros Depósitos", "section": "Passivo", "level": 2, "order": 19, "parent_id": "passivo_depositos",
             "rule": {"section_digits": ["4"], "name_regex": [r"DEP[ÓO]SITOS"], "residual": True}},

            # PASSIVO - sublinhas (Títulos)
            {"id": "passivo_lf", "label": "Letras Financeiras (LF)", "section": "Passivo", "level": 2, "order": 21, "parent_id": "passivo_titulos",
             "rule": {"section_digits": ["4"], "name_regex": [r"LETRA(S)? FINANCEIRA"]}},
            {"id": "passivo_lci_lca", "label": "LCI / LCA", "section": "Passivo", "level": 2, "order": 22, "parent_id": "passivo_titulos",
             "rule": {"section_digits": ["4"], "name_regex": [r"LCI", r"LCA", r"LETRA DE CR[ÉE]DITO"]}},
            {"id": "passivo_titulos_outros", "label": "Outros Títulos Emitidos", "section": "Passivo", "level": 2, "order": 29, "parent_id": "passivo_titulos",
             "rule": {"section_digits": ["4"], "name_regex": [r"T[IÍ]TULOS", r"VALORES MOBILI[ÁA]RIOS"], "residual": True}},

            # PASSIVO - sublinhas (Captações)
            {"id": "passivo_capt_local", "label": "Empréstimos e Repasses - Local", "section": "Passivo", "level": 2, "order": 31, "parent_id": "passivo_captacoes",
             "rule": {"section_digits": ["4"], "name_regex": [r"EMPR[ÉE]STIMOS", r"REPASSES"], "exclude_regex": [r"EXTERIOR"]}},
            {"id": "passivo_capt_exterior", "label": "Empréstimos e Repasses - Exterior", "section": "Passivo", "level": 2, "order": 32, "parent_id": "passivo_captacoes",
             "rule": {"section_digits": ["4"], "name_regex": [r"EXTERIOR"]}},
            {"id": "passivo_capt_residual", "label": "Captações (Residual)", "section": "Passivo", "level": 2, "order": 39, "parent_id": "passivo_captacoes",
             "rule": {"section_digits": ["4"], "residual": True}},

            # PASSIVO - sublinhas (Outras obrigações)
            {"id": "passivo_outras_obr_residual", "label": "Outras Obrigações (Residual)", "section": "Passivo", "level": 2, "order": 59, "parent_id": "passivo_outras_obr",
             "rule": {"section_digits": ["4"], "residual": True}},

            # PL
            {"id": "pl_total", "label": "Patrimônio Líquido", "section": "Patrimônio Líquido", "level": 1, "order": 10, "derived": True, "is_total": True},
            {"id": "pl_capital", "label": "Capital Social", "section": "Patrimônio Líquido", "level": 2, "order": 11, "parent_id": "pl_total",
             "rule": {"section_digits": ["6"], "name_regex": [r"CAPITAL"]}},
            {"id": "pl_reservas", "label": "Reservas", "section": "Patrimônio Líquido", "level": 2, "order": 12, "parent_id": "pl_total",
             "rule": {"section_digits": ["6"], "name_regex": [r"RESERVAS"]}},
            {"id": "pl_ajustes", "label": "Ajustes de Avaliação Patrimonial", "section": "Patrimônio Líquido", "level": 2, "order": 13, "parent_id": "pl_total",
             "rule": {"section_digits": ["6"], "name_regex": [r"AJUSTES"]}},
            {"id": "pl_tesouraria", "label": "Ações em Tesouraria", "section": "Patrimônio Líquido", "level": 2, "order": 14, "parent_id": "pl_total",
             "rule": {"section_digits": ["6"], "name_regex": [r"TESOURARIA"]}},
            {"id": "pl_outros", "label": "Outros (Residual)", "section": "Patrimônio Líquido", "level": 2, "order": 19, "parent_id": "pl_total",
             "rule": {"section_digits": ["6"], "residual": True}},
        ],
    }


def load_schema_rules(schema_rules_path: Path = SCHEMA_RULES_PATH) -> dict:
    if schema_rules_path.exists():
        return json.loads(schema_rules_path.read_text(encoding="utf-8"))
    return default_schema_rules()


def load_name_mapping_xlsx(path: Path = NAME_MAP_PATH) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_excel(path, sheet_name="ClassificacaoCompleta4060")
    if "CONTA" not in df.columns or "NOME_CONTA_CORRIGIDO" not in df.columns:
        # fallback: tentar nomes originais
        if "NOME_CONTA" not in df.columns:
            return {}
        df["NOME_CONTA_CORRIGIDO"] = df["NOME_CONTA"]
    df["CONTA"] = df["CONTA"].astype(str).str.replace(r"\D", "", regex=True)
    return dict(zip(df["CONTA"], df["NOME_CONTA_CORRIGIDO"].astype(str)))


def _normalize_name(val: str) -> str:
    return str(val or "").upper()


def _compile_regex_list(patterns: Iterable[str]) -> List[re.Pattern]:
    compiled = []
    for pat in patterns:
        if not pat:
            continue
        compiled.append(re.compile(pat, re.IGNORECASE))
    return compiled


def build_group_accounts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Cria tabela de contas-grupo e mapeia conta->grupo."""
    df = df.copy()
    df["CONTA_10"] = df["CONTA_10"].astype(str)
    df["SECTION_DIGIT"] = df["CONTA_10"].str[0]
    df = df[df["SECTION_DIGIT"].isin(["1", "2", "4", "6"])]

    u = df[["CONTA_10", "NOME_CONTA", "SECTION_DIGIT"]].drop_duplicates().copy()
    u["TZ"] = 10 - u["CONTA_10"].str.rstrip("0").str.len()
    group_set = set(u[u["TZ"] >= 5]["CONTA_10"].tolist())

    root_by_digit = {
        "1": "1000000009",
        "2": "2000000008",
        "4": "4000000006",
        "6": "6000000004",
    }
    group_set.update({v for v in root_by_digit.values() if v in u["CONTA_10"].values})

    def find_group(code: str) -> Optional[str]:
        for k in range(1, 10):
            cand = code[:-k] + ("0" * k)
            if cand == code:
                continue
            if cand in group_set:
                return cand
        return root_by_digit.get(code[0])

    group_map: Dict[str, str] = {}
    for code in u["CONTA_10"].unique():
        grp = find_group(code)
        if grp:
            group_map[code] = grp

    groups = u[u["CONTA_10"].isin(group_set)].copy()
    groups = groups.rename(columns={"CONTA_10": "group_code", "NOME_CONTA": "group_name"})
    return groups[["group_code", "group_name", "SECTION_DIGIT"]].drop_duplicates(), group_map


def assign_groups_to_lines(groups: pd.DataFrame, lines: List[dict]) -> Dict[str, List[str]]:
    remaining = groups.copy()
    remaining["name_norm"] = remaining["group_name"].map(_normalize_name)
    assigned: Dict[str, List[str]] = {}

    for line in sorted(lines, key=lambda r: (r.get("section", ""), r.get("level", 0), r.get("order", 0))):
        rule = line.get("rule") or {}
        if line.get("derived"):
            continue
        section_digits = set(rule.get("section_digits") or [])
        subset = remaining.copy()
        if section_digits:
            subset = subset[subset["SECTION_DIGIT"].isin(section_digits)]

        regex_list = _compile_regex_list(rule.get("name_regex") or [])
        exclude_list = _compile_regex_list(rule.get("exclude_regex") or [])
        residual = bool(rule.get("residual"))

        if residual:
            matched = subset
        else:
            if not regex_list:
                continue
            mask = pd.Series([False] * len(subset), index=subset.index)
            for rg in regex_list:
                mask = mask | subset["name_norm"].str.contains(rg)
            if exclude_list:
                for ex in exclude_list:
                    mask = mask & ~subset["name_norm"].str.contains(ex)
            matched = subset[mask]

        if matched.empty:
            assigned[line["id"]] = []
            continue

        assigned[line["id"]] = matched["group_code"].tolist()
        remaining = remaining.drop(index=matched.index)

    return assigned


def build_schema_from_rules(df: pd.DataFrame, rules: dict) -> List[dict]:
    """Gera schema com contas a partir de regras."""
    if "CONTA_10" not in df.columns:
        df = normalize_conta_10(df)

    groups, group_map = build_group_accounts(df)
    lines = rules.get("lines", [])
    group_assignment = assign_groups_to_lines(groups, lines)

    # Build accounts per line
    line_accounts: Dict[str, List[str]] = {}
    for line in lines:
        line_id = line["id"]
        if line.get("derived"):
            line_accounts[line_id] = []
            continue
        group_codes = group_assignment.get(line_id, [])
        if not group_codes:
            line_accounts[line_id] = []
            continue
        accounts = [c for c, g in group_map.items() if g in set(group_codes)]
        line_accounts[line_id] = sorted(set(accounts))

    # Derived lines aggregate children
    children_map: Dict[str, List[str]] = {}
    for line in lines:
        parent = line.get("parent_id")
        if parent:
            children_map.setdefault(parent, []).append(line["id"])

    def collect_accounts(line_id: str) -> List[str]:
        if line_id in line_accounts and line_accounts[line_id]:
            return line_accounts[line_id]
        children = children_map.get(line_id, [])
        acc = []
        for child in children:
            acc.extend(collect_accounts(child))
        return sorted(set(acc))

    # section-level accounts (para totais)
    section_accounts: Dict[str, List[str]] = {}
    if "SECTION_DIGIT" not in df.columns:
        df["SECTION_DIGIT"] = df["CONTA_10"].str[0]
    for sec, digits in {
        "Ativo": ["1", "2"],
        "Passivo": ["4"],
        "Patrimônio Líquido": ["6"],
    }.items():
        section_accounts[sec] = df.loc[df["SECTION_DIGIT"].isin(digits), "CONTA_10"].dropna().astype(str).unique().tolist()

    schema: List[dict] = []
    for line in lines:
        line_id = line["id"]
        contas = collect_accounts(line_id)
        if line.get("is_total"):
            contas = section_accounts.get(line.get("section"), contas)
        schema.append(
            {
                "id": line_id,
                "section": line.get("section"),
                "level": line.get("level"),
                "label": line.get("label"),
                "order": line.get("order", 0),
                "parent_id": line.get("parent_id"),
                "derived": bool(line.get("derived")),
                "is_total": bool(line.get("is_total")),
                "contas": contas,
            }
        )

    return schema


def build_balanco_padronizado(
    df: pd.DataFrame,
    schema: List[dict],
    cod_congl: str,
    data_base: str,
) -> pd.DataFrame:
    df = df.copy()
    df["COD_CONGL"] = df["COD_CONGL"].astype(str)
    df["DATA_BASE"] = df["DATA_BASE"].astype(str)

    if "CONTA_10" not in df.columns:
        df = normalize_conta_10(df)

    df_sel = df[(df["COD_CONGL"] == str(cod_congl)) & (df["DATA_BASE"] == str(data_base))].copy()

    rows = []
    for row in schema:
        contas = row.get("contas") or []
        if df_sel.empty or not contas:
            saldo = pd.NA
        else:
            mask = df_sel["CONTA_10"].isin(contas)
            if not mask.any():
                saldo = pd.NA
            else:
                saldo = df_sel.loc[mask, "SALDO"].sum()
        rows.append(
            {
                "id": row.get("id"),
                "section": row.get("section"),
                "level": row.get("level"),
                "label": row.get("label"),
                "order": row.get("order", 0),
                "parent_id": row.get("parent_id"),
                "derived": row.get("derived"),
                "is_total": row.get("is_total"),
                "saldo": saldo,
            }
        )
    return pd.DataFrame(rows)


def build_mapping_table(df: pd.DataFrame, schema: List[dict]) -> pd.DataFrame:
    if "CONTA_10" not in df.columns:
        df = normalize_conta_10(df)
    u = df[["CONTA_10", "NOME_CONTA"]].drop_duplicates()
    rows = []
    for line in schema:
        if line.get("is_total"):
            continue
        for conta in line.get("contas") or []:
            nome = u.loc[u["CONTA_10"] == conta, "NOME_CONTA"]
            rows.append(
                {
                    "Section": line.get("section"),
                    "Level": line.get("level"),
                    "Label": line.get("label"),
                    "LineId": line.get("id"),
                    "ParentId": line.get("parent_id"),
                    "Derived": bool(line.get("derived")),
                    "CONTA": conta,
                    "NOME_CONTA": nome.iloc[0] if not nome.empty else None,
                    "Prefixo": conta[:3],
                }
            )
    return pd.DataFrame(rows)


def build_hierarchical_lines(
    df: pd.DataFrame,
    rules: dict,
    name_map: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Constrói linhas hierárquicas: derivadas -> grupos -> contas."""
    if name_map is None:
        name_map = {}
    if "CONTA_10" not in df.columns:
        df = normalize_conta_10(df)

    groups, group_map = build_group_accounts(df)
    lines = rules.get("lines", [])
    group_assignment = assign_groups_to_lines(groups, lines)

    # helper for label
    def _label_for(code: str, fallback: str) -> str:
        return name_map.get(code, fallback)

    # build u for account names
    u = df[["CONTA_10", "NOME_CONTA"]].drop_duplicates().set_index("CONTA_10")

    out = []
    # derived lines (level 1 or as defined)
    for line in lines:
        out.append(
            {
                "id": line["id"],
                "parent_id": line.get("parent_id"),
                "section": line.get("section"),
                "level": line.get("level"),
                "order": line.get("order", 0),
                "label": line.get("label"),
                "type": "derived",
                "is_total": bool(line.get("is_total")),
            }
        )

    # group lines
    for line in lines:
        line_id = line["id"]
        groups_for_line = group_assignment.get(line_id, [])
        for gcode in groups_for_line:
            gname = groups.loc[groups["group_code"] == gcode, "group_name"]
            gname = gname.iloc[0] if not gname.empty else gcode
            out.append(
                {
                    "id": f"{line_id}:{gcode}",
                    "parent_id": line_id,
                    "section": line.get("section"),
                    "level": (line.get("level") or 1) + 1,
                    "order": line.get("order", 0) * 100 + 1,
                    "label": _label_for(gcode, gname),
                    "type": "group",
                    "group_code": gcode,
                    "is_total": False,
                }
            )

    # account lines
    for line in lines:
        line_id = line["id"]
        groups_for_line = group_assignment.get(line_id, [])
        for gcode in groups_for_line:
            # all accounts mapped to group
            contas = [c for c, g in group_map.items() if g == gcode]
            for conta in contas:
                nome = u.loc[conta, "NOME_CONTA"] if conta in u.index else conta
                out.append(
                    {
                        "id": f"{line_id}:{gcode}:{conta}",
                        "parent_id": f"{line_id}:{gcode}",
                        "section": line.get("section"),
                        "level": (line.get("level") or 1) + 2,
                        "order": line.get("order", 0) * 100 + 2,
                        "label": _label_for(conta, nome),
                        "type": "account",
                        "conta": conta,
                        "group_code": gcode,
                        "is_total": False,
                    }
                )

    return out
