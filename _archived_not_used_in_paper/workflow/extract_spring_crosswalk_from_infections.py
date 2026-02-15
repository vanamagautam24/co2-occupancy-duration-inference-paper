#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def col_to_idx(ref: str) -> int:
    letters = "".join([c for c in ref if c.isalpha()])
    v = 0
    for ch in letters:
        v = v * 26 + (ord(ch.upper()) - 64)
    return v - 1


def excel_serial_to_date(v: str | None) -> str | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        f = float(v)
    except Exception:
        return None
    dt = datetime(1899, 12, 30) + timedelta(days=f)
    return dt.date().isoformat()


def parse_sheet_rows(xlsx_path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{{{NS_PKG}}}Relationship")}

        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall(f"{{{NS_MAIN}}}si"):
                txt = "".join((t.text or "") for t in si.iter(f"{{{NS_MAIN}}}t"))
                shared.append(txt)

        sheet = None
        sheets = wb.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            raise RuntimeError("No sheets found in workbook.")
        for s in sheets.findall(f"{{{NS_MAIN}}}sheet"):
            if s.attrib.get("name") == sheet_name:
                sheet = s
                break
        if sheet is None:
            raise RuntimeError(f"Sheet not found: {sheet_name}")

        rid = sheet.attrib.get(f"{{{NS_REL}}}id")
        target = rel_map[rid]
        sheet_path = "xl/" + target.split("xl/")[-1]
        root = ET.fromstring(z.read(sheet_path))

        out: list[list[str]] = []
        for row in root.findall(f".//{{{NS_MAIN}}}sheetData/{{{NS_MAIN}}}row"):
            cells = row.findall(f"{{{NS_MAIN}}}c")
            vals: dict[int, str] = {}
            max_idx = -1
            for c in cells:
                ref = c.attrib.get("r", "A1")
                idx = col_to_idx(ref)
                max_idx = max(max_idx, idx)
                ctype = c.attrib.get("t")
                v = c.find(f"{{{NS_MAIN}}}v")
                txt = ""
                if v is not None and v.text is not None:
                    txt = v.text
                    if ctype == "s":
                        try:
                            txt = shared[int(txt)]
                        except Exception:
                            pass
                vals[idx] = txt
            if max_idx >= 0:
                out.append([vals.get(i, "") for i in range(max_idx + 1)])
    return out


def extract_crosswalk(rows: list[list[str]]) -> list[tuple[str, str, str | None, str | None, str, str]]:
    header_indices: list[int] = []
    for i, r in enumerate(rows):
        text = " | ".join(str(x) for x in r)
        if "Roommate pair" in text and "Monitor ID" in text and "start date file" in text:
            header_indices.append(i)
    if not header_indices:
        raise RuntimeError("Could not find header rows with Roommate pair/Monitor ID in sheet.")

    all_entries: list[tuple[str, str, str | None, str | None, str, str]] = []

    for hi, header_i in enumerate(header_indices):
        header = rows[header_i]
        end_i = header_indices[hi + 1] if hi + 1 < len(header_indices) else len(rows)

        try:
            idx_pair = header.index("Roommate pair")
            idx_monitor = header.index("Monitor ID")
            idx_start = header.index("start date file")
            idx_end = header.index("end date file")
        except ValueError as e:
            raise RuntimeError(f"Missing required column in header row {header_i+1}: {e}") from e

        section_room_type = "shared" if hi == 0 else "single"
        section_label = f"header_row_{header_i+1}"

        for r in rows[header_i + 1 : end_i]:
            pair = r[idx_pair] if idx_pair < len(r) else ""
            mon = r[idx_monitor] if idx_monitor < len(r) else ""
            st = r[idx_start] if idx_start < len(r) else ""
            en = r[idx_end] if idx_end < len(r) else ""

            pair = str(pair).strip()
            mon = str(mon).strip()
            if not pair or not mon:
                continue
            if "/" not in pair:
                continue
            if not mon.isdigit():
                continue

            ids = [x.strip() for x in pair.split("/") if x.strip().isdigit()]
            if not ids:
                continue

            st_iso = excel_serial_to_date(st)
            en_iso = excel_serial_to_date(en)
            for sid in ids:
                all_entries.append((sid, mon, st_iso, en_iso, section_room_type, section_label))

    deduped: list[tuple[str, str, str | None, str | None, str, str]] = []
    seen = set()
    for rec in all_entries:
        if rec in seen:
            continue
        seen.add(rec)
        deduped.append(rec)
    return deduped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract subject->sensor crosswalk from infections workbook.")
    p.add_argument(
        "--infile",
        default="InfectionsDatabase_2024_2025.xlsx",
        help="Input XLSX workbook path.",
    )
    p.add_argument(
        "--sheet",
        default="Spring 2025 -  Flu only",
        help="Sheet name to extract from.",
    )
    p.add_argument(
        "--outfile",
        default="data/subject_sensor_map_from_infections_spring2025.csv",
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_sheet_rows(infile, args.sheet)
    crosswalk = extract_crosswalk(rows)

    with outfile.open("w", encoding="utf-8") as f:
        f.write("subject_id,sensor_id,map_start_date,map_end_date,room_type,source_section\n")
        for sid, mon, st, en, room_type, section in crosswalk:
            f.write(f"{sid},{mon},{st or ''},{en or ''},{room_type},{section}\n")

    print(f"wrote {outfile} rows={len(crosswalk)}")


if __name__ == "__main__":
    main()

