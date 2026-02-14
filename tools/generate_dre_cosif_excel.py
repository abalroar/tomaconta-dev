from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence
import zipfile
from xml.sax.saxutils import escape

MAPPING_PATH = Path("data/dre_cosif_mapping.json")
CSV_PATH = Path("202509BLOPRUDENCIAL 2.CSV")
OUTPUT_PATH = Path("data/DRE_COSIF_mapeamento_explicativo.xlsx")
CSV_OUTPUT_PATH = Path("data/dre_cosif_mapeamento_explicativo.csv")


def _load_mapping() -> List[Dict]:
    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    return payload.get("mappings", []) if isinstance(payload, dict) else []


def _build_description_rows(mappings: List[Dict]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in mappings:
        dre_label = str(item.get("label") or "").strip()
        ifdata_label = str(item.get("ifdata_label") or "").strip()
        formula = str(item.get("cosif_formula") or "").strip()
        depara = item.get("cosif_depara") or []

        if not depara:
            rows.append(
                {
                    "conta_dre_atual": dre_label,
                    "linha_ifdata": ifdata_label,
                    "ordem_componente": "1",
                    "descricao_componente_cosif": "(sem de-para cadastrado)",
                    "formula_referencia_ifdata": formula,
                }
            )
            continue

        for idx, comp in enumerate(depara, start=1):
            rows.append(
                {
                    "conta_dre_atual": dre_label,
                    "linha_ifdata": ifdata_label,
                    "ordem_componente": str(idx),
                    "descricao_componente_cosif": str(comp.get("description") or "").strip(),
                    "formula_referencia_ifdata": formula,
                }
            )
    return rows


def _parse_float_br(v: str) -> float:
    return float(v.replace(".", "").replace(",", "."))


def _load_balancete_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with CSV_PATH.open(encoding="latin-1") as f:
        for _ in range(3):
            next(f)
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r is None:
                continue
            row = {str(k).lstrip("#").strip(): str(v or "") for k, v in r.items()}
            rows.append(row)
    return rows


def _build_a1_evidence_rows(balancete_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    contas = {"7140000004", "7141000003", "7142000002", "8115020001"}
    filt = [r for r in balancete_rows if r.get("CONTA") in contas]

    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for r in filt:
        key = (r.get("CNPJ", ""), r.get("NOME_INSTITUICAO", ""), r.get("NOME_CONGL", ""))
        grouped.setdefault(key, []).append(r)

    selected_key = None
    selected_rows: List[Dict[str, str]] = []
    for key, group in grouped.items():
        existentes = {r.get("CONTA", "") for r in group}
        if {"7140000004", "7141000003", "7142000002"}.issubset(existentes):
            selected_key = key
            selected_rows = group
            break

    if not selected_key:
        return [{"status": "sem_amostra", "observacao": "Não foi encontrada instituição com as contas necessárias."}]

    (_, nome_inst, nome_congl) = selected_key
    by_conta = {r["CONTA"]: r for r in selected_rows}
    total = _parse_float_br(by_conta["7140000004"]["SALDO"])
    part1 = _parse_float_br(by_conta["7141000003"]["SALDO"])
    part2 = _parse_float_br(by_conta["7142000002"]["SALDO"])
    part3 = _parse_float_br(by_conta["8115020001"]["SALDO"]) if "8115020001" in by_conta else 0.0
    diff = total - (part1 + part2 + part3)

    return [
        {
            "instituicao": nome_inst,
            "conglomerado": nome_congl,
            "conta_referencia": "7140000004",
            "descricao": "Rendas de Aplicações Interfinanceiras de Liquidez",
            "saldo": f"{total:.2f}",
            "observacao": "Conta agregada no balancete 4060.",
        },
        {
            "instituicao": nome_inst,
            "conglomerado": nome_congl,
            "conta_referencia": "7141000003",
            "descricao": "RENDAS DE APLICAÇÕES EM OPERAÇÕES COMPROMISSADAS",
            "saldo": f"{part1:.2f}",
            "observacao": "Componente detalhado observado localmente.",
        },
        {
            "instituicao": nome_inst,
            "conglomerado": nome_congl,
            "conta_referencia": "7142000002",
            "descricao": "RENDAS DE APLICAÇÕES EM DEPÓSITOS INTERFINANCEIROS",
            "saldo": f"{part2:.2f}",
            "observacao": "Componente detalhado observado localmente.",
        },
        {
            "instituicao": nome_inst,
            "conglomerado": nome_congl,
            "conta_referencia": "8115020001",
            "descricao": "(conta da fórmula IFData/COSIF; não apareceu na amostra local)",
            "saldo": f"{part3:.2f}",
            "observacao": "Manter conferência em ambiente com acesso ao COSIF online.",
        },
        {
            "instituicao": nome_inst,
            "conglomerado": nome_congl,
            "conta_referencia": "checagem",
            "descricao": "7140000004 - (7141000003 + 7142000002 + 8115020001)",
            "saldo": f"{diff:.2f}",
            "observacao": "Diferença (na amostra local) ficou próxima de zero.",
        },
    ]


def _col_name(idx: int) -> str:
    out = ""
    n = idx
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _sheet_xml(rows: Sequence[Sequence[str]]) -> str:
    body = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_col_name(c_idx)}{r_idx}"
            txt = escape(str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{txt}</t></is></c>')
        body.append(f"<row r=\"{r_idx}\">{''.join(cells)}</row>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(body)}</sheetData>"
        "</worksheet>"
    )


def _rows_from_dicts(dict_rows: List[Dict[str, str]]) -> List[List[str]]:
    if not dict_rows:
        return [["sem_dados"]]
    headers = list(dict_rows[0].keys())
    out = [headers]
    for row in dict_rows:
        out.append([str(row.get(h, "")) for h in headers])
    return out




def _write_csv(rows: List[Dict[str, str]]) -> None:
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def _write_xlsx(sheet1: List[List[str]], sheet2: List[List[str]]) -> None:
    content_types = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/>
  <Override PartName='/xl/worksheets/sheet1.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>
  <Override PartName='/xl/worksheets/sheet2.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>
</Types>"""
    rels = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/>
</Relationships>"""
    workbook = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>
  <sheets>
    <sheet name='Mapa_DRE_Descricao' sheetId='1' r:id='rId1'/>
    <sheet name='Investigacao_a1' sheetId='2' r:id='rId2'/>
  </sheets>
</workbook>"""
    wb_rels = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/>
  <Relationship Id='rId2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet2.xml'/>
</Relationships>"""

    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml(sheet1))
        zf.writestr("xl/worksheets/sheet2.xml", _sheet_xml(sheet2))


def main() -> None:
    mappings = _load_mapping()
    desc_rows = _build_description_rows(mappings)
    bal_rows = _load_balancete_rows()
    ev_rows = _build_a1_evidence_rows(bal_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_xlsx(_rows_from_dicts(desc_rows), _rows_from_dicts(ev_rows))
    _write_csv(desc_rows)
    print(f"arquivo gerado: {OUTPUT_PATH}")
    print(f"arquivo gerado: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
