import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


NUMBER_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def column_name(index):
    name = ""

    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name

    return name


def is_numeric_cell(value, column):
    if value == "":
        return False

    if column.endswith("_timestamp_ns"):
        return False

    if column in {"heart_rate_raw_line", "raw_eye_dtype"}:
        return False

    return bool(NUMBER_RE.fullmatch(value))


def inline_string(value):
    escaped = escape(value)
    preserve = ' xml:space="preserve"' if value.strip() != value else ""
    return f'<is><t{preserve}>{escaped}</t></is>'


def cell_xml(row_index, col_index, value, column, header=False):
    ref = f"{column_name(col_index)}{row_index}"

    if value == "":
        return ""

    if not header and is_numeric_cell(value, column):
        return f'<c r="{ref}"><v>{value}</v></c>'

    return f'<c r="{ref}" t="inlineStr">{inline_string(value)}</c>'


def worksheet_xml(headers, rows):
    max_col = len(headers)
    max_row = len(rows) + 1
    dimension = f"A1:{column_name(max_col)}{max_row}"
    xml_rows = []

    header_cells = [
        cell_xml(1, col_index, header, header, header=True)
        for col_index, header in enumerate(headers, start=1)
    ]
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_offset, row in enumerate(rows, start=2):
        cells = [
            cell_xml(row_offset, col_index, row.get(header, ""), header)
            for col_index, header in enumerate(headers, start=1)
        ]
        xml_rows.append(f'<row r="{row_offset}">{"".join(cells)}</row>')

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetData>
    {"".join(xml_rows)}
  </sheetData>
  <autoFilter ref="{dimension}"/>
</worksheet>'''


def write_xlsx(csv_path, xlsx_path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not headers:
        raise ValueError(f"No headers found in {csv_path}")

    files = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Raw Data" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>''',
        "xl/worksheets/sheet1.xml": worksheet_xml(headers, rows),
    }

    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    return len(rows), len(headers)


def main():
    parser = argparse.ArgumentParser(description="Convert a CSV file to a basic Excel .xlsx workbook.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("xlsx_path", nargs="?", type=Path)
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    xlsx_path = (
        args.xlsx_path.expanduser().resolve()
        if args.xlsx_path
        else csv_path.with_suffix(".xlsx")
    )

    row_count, col_count = write_xlsx(csv_path, xlsx_path)
    print(f"wrote: {xlsx_path}")
    print(f"rows: {row_count}")
    print(f"columns: {col_count}")


if __name__ == "__main__":
    main()
