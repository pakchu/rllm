from __future__ import annotations

import io
import json
import urllib.request
import zlib
from dataclasses import asdict
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from training import build_daily_treasury_fiscal_flow_source as dts


def _pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode() + body + b"\nendobj\n"


def _stream_object(number: int, content: bytes, *, filter_name: str = "FlateDecode") -> bytes:
    compressed = zlib.compress(content) if filter_name == "FlateDecode" else content
    dictionary = f"<< /Length {len(compressed)} /Filter /{filter_name} >>\n".encode()
    return _pdf_object(
        number,
        dictionary + b"stream\n" + compressed + b"\nendstream",
    )


def _text(x: float, y: float, value: str, *, size: float = 6.5) -> bytes:
    escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return (
        f"BT /F1 {size} Tf 1 0 0 1 {x} {y} Tm ({escaped}) Tj ET\n".encode()
    )


def _minimal_dts_pdf(report_date: str = "Friday, December 29, 2023") -> bytes:
    operations = [
            _text(100, 800, "DAILY TREASURY STATEMENT", size=14.5),
            _text(100, 780, report_date, size=12),
            _text(100, 740, "TABLE I - Operating Cash Balance", size=12),
            _text(280, 712, "1/"),
            _text(40, 710, "Treasury General Account Closing Balance"),
            _text(330, 710, "100"),
            _text(380, 710, "100"),
            _text(430, 710, "100"),
            _text(100, 680, "TABLE II - Deposits and Withdrawals", size=12),
            _text(40, 650, "Deposit A"),
            _text(180, 650, "1"),
            _text(220, 650, "2"),
            _text(260, 650, "3"),
            _text(320, 650, "Withdrawal A"),
            _text(450, 650, "4"),
            _text(490, 650, "5"),
            _text(530, 650, "6"),
            _text(100, 600, "TABLE IIIA - Public Debt Transactions", size=12),
            _text(40, 570, "Issue A"),
            _text(180, 570, "7"),
            _text(220, 570, "8"),
            _text(260, 570, "9"),
            _text(320, 570, "Redemption A"),
            _text(450, 570, "10"),
            _text(490, 570, "11"),
            _text(530, 570, "12"),
        ]
    first_content = b"".join(operations[:10])
    second_content = b"".join(operations[10:])
    return b"".join(
        [
            b"%PDF-1.4\n",
            _pdf_object(1, b"<< /Type /Page /Contents [2 0 R 3 0 R] >>"),
            _stream_object(2, b"q\n" + first_content),
            _stream_object(3, second_content + b"Q\n"),
            _pdf_object(4, b"<< /CreationDate (D:20240102160000-05'00') >>"),
            b"%%EOF\n",
        ]
    )


def _published_report(date_value: date) -> dict[str, str]:
    compact = date_value.strftime("%Y%m%d")
    return {
        "report_date": date_value.strftime(
            "%a %b %d %Y 00:00:00 GMT+0000 (Coordinated Universal Time)"
        ),
        "path": (
            "/static-data/published-reports/dts/"
            f"DailyTreasuryStatement_{compact}.pdf"
        ),
    }


def _page_data(rows: list[dict[str, str]]) -> bytes:
    return json.dumps(
        {"result": {"pageContext": {"config": {"publishedReports": rows}}}}
    ).encode()


def _write_bound_source(path: Path, *, url: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    receipt = dts.FetchReceipt(
        url=url,
        status=200,
        final_url=url,
        retrieved_at_utc="2026-07-20T00:00:00+00:00",
        etag='"fixture"',
        last_modified="Mon, 20 Jul 2026 00:00:00 GMT",
        redirect_chain=(),
        byte_length=len(payload),
        sha256=dts.sha256_bytes(payload),
    )
    dts._receipt_path(path).write_bytes(dts.canonical_json(asdict(receipt)))


def _xlsx_fixture(*, corrupt_header: bool = False) -> bytes:
    strings = [
        "Wrong Date Header" if corrupt_header else "Date of DTS Change",
        "Table Name",
        "Table Section",
        "Change",
        "2",
        "Deposit",
        "pre-cap change",
        "post-cap change",
        "DTS Date",
        "Table Number",
        "Entity",
    ]
    shared = "".join(f"<si><t>{value}</t></si>" for value in strings)
    sheet1 = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>
      <c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>
    <row r="2"><c r="A2"><v>45289</v></c><c r="B2" t="s"><v>4</v></c>
      <c r="C2" t="s"><v>5</v></c><c r="D2" t="s"><v>6</v></c></row>
    <row r="3"><c r="A3"><v>45293</v></c><c r="B3" t="s"><v>4</v></c>
      <c r="C3" t="s"><v>5</v></c><c r="D3" t="s"><v>7</v></c></row>
  </sheetData>
</worksheet>"""
    sheet2 = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData><row r="1"><c r="A1" t="s"><v>8</v></c>
    <c r="B1" t="s"><v>9</v></c><c r="C1" t="s"><v>2</v></c>
    <c r="D1" t="s"><v>10</v></c>
    <c r="E1" t="s"><v>3</v></c></row></sheetData>
</worksheet>"""
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            "<?xml version=\"1.0\"?><sst "
            "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            f"{shared}</sst>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet1)
        archive.writestr("xl/worksheets/sheet2.xml", sheet2)
        archive.writestr(
            "xl/workbook.xml",
            "<?xml version=\"1.0\"?><workbook "
            "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            "<sheets><sheet name=\"DTS Changes\" sheetId=\"1\" r:id=\"rId1\"/>"
            "<sheet name=\"Full - Table\" sheetId=\"2\" r:id=\"rId2\"/>"
            "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            "<?xml version=\"1.0\"?><Relationships "
            "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" "
            "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/worksheet\" "
            "Target=\"worksheets/sheet1.xml\"/>"
            "<Relationship Id=\"rId2\" "
            "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/worksheet\" "
            "Target=\"worksheets/sheet2.xml\"/>"
            "</Relationships>",
        )
    return buffer.getvalue()


def test_report_url_and_published_index_enforce_the_pre2024_cap() -> None:
    with pytest.raises(ValueError, match="outside the frozen cap"):
        dts.report_url(date(2024, 1, 2))

    payload = _page_data(
        [_published_report(date(2023, 12, 29)), _published_report(date(2024, 1, 2))]
    )
    document = json.loads(payload)
    document["result"]["pageContext"]["config"]["publishedReports"][1][
        "report_date"
    ] = {"postcap_schema": "must not affect pre-cap logic"}
    payload = json.dumps(document).encode()
    rows = dts.parse_published_reports(
        payload, start=date(2023, 12, 29), end=date(2023, 12, 29)
    )
    assert [row.record_date for row in rows] == [date(2023, 12, 29)]
    assert all("2024" not in row.url for row in rows)


def test_redirect_handler_rejects_cross_host_and_postcap_targets() -> None:
    handler = dts._SameHostRedirectHandler(allow_metadata=False)
    request = urllib.request.Request(dts.report_url(date(2023, 12, 29)))
    with pytest.raises(ValueError, match="unsafe Fiscal Data redirect"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "https://example.com/report.pdf"
        )
    with pytest.raises(ValueError, match="outside the frozen cap"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://fiscaldata.treasury.gov/static-data/published-reports/dts/"
            "DailyTreasuryStatement_20240102.pdf",
        )


def test_published_index_rejects_duplicate_report_dates() -> None:
    row = _published_report(date(2023, 6, 30))
    with pytest.raises(ValueError, match="duplicate published DTS report date"):
        dts.parse_published_reports(
            _page_data([row, row]), start=date(2023, 6, 30), end=date(2023, 6, 30)
        )


def test_source_clock_uses_causal_execution_boundaries() -> None:
    available, execution, stage = dts.source_clock(date(2022, 12, 30))
    assert available.isoformat() == "2023-01-03T21:00:00+00:00"
    assert execution.isoformat() == "2023-01-03T21:05:00+00:00"
    assert stage == "selection"

    available, execution, stage = dts.source_clock(date(2023, 12, 29))
    assert available.isoformat() == "2024-01-02T21:00:00+00:00"
    assert execution.isoformat() == "2024-01-02T21:05:00+00:00"
    assert stage == "boundary_quarantine"


def test_literal_string_parser_handles_nested_escapes_octal_and_continuation() -> None:
    payload = b"(A \\(nested\\) (ok)\\nB\\053C\\\nD) trailing"
    parsed, end = dts._pdf_literal_string(payload, 0)
    assert parsed == b"A (nested) (ok)\nB+CD"
    assert payload[end:].startswith(b" trailing")

    parsed, _ = dts._pdf_literal_string(b"(A\\0053B)", 0)
    assert parsed == b"A\x053B"
    with pytest.raises(ValueError, match="octal escape exceeds one byte"):
        dts._pdf_literal_string(b"(A\\400B)", 0)


def test_content_parser_handles_tj_arrays_and_graphics_state() -> None:
    content = b"".join(
        [
            b"q 1 0 0 1 100 200 cm ",
            b"BT /F1 10 Tf 1 0 0 1 5 6 Tm ",
            b"[(Daily) -120 ( Treasury) 40 ( Statement)] TJ ET Q ",
            b"BT /F1 8 Tf 1 0 0 1 1 2 Tm (Plain) Tj ET",
        ]
    )
    cells = dts.parse_pdf_content_cells(content, page_number=1)
    assert [(cell.text, cell.x, cell.y) for cell in cells] == [
        ("Daily Treasury Statement", 105.0, 206.0),
        ("Plain", 1.0, 2.0),
    ]
    with pytest.raises(ValueError, match="unsupported PDF content operator"):
        dts.parse_pdf_content_cells(b"123 ZZ", page_number=1)


def test_pdf_parser_supports_content_arrays_and_extracts_all_required_sides() -> None:
    parsed = dts.parse_dts_pdf(
        _minimal_dts_pdf(), expected_record_date=date(2023, 12, 29)
    )
    assert parsed.table_ids_found == ("I", "II", "IIIA")
    assert parsed.creation_metadata_raw == ("D:20240102160000-05'00'",)
    assert len(parsed.table_i_rows) == 1
    assert parsed.table_i_rows[0].published_values_usd_millions_json == "[100,100,100]"
    assert parsed.table_i_rows[0].footnote_markers == "1/"
    assert {(row.table_id, row.side) for row in parsed.rows} == {
        ("II", "deposit"),
        ("II", "withdrawal"),
        ("IIIA", "issue"),
        ("IIIA", "redemption"),
    }
    assert {row.research_stage for row in parsed.rows} == {"boundary_quarantine"}


def test_pdf_parser_rejects_filename_report_date_mismatch() -> None:
    with pytest.raises(ValueError, match="filename/report date mismatch"):
        dts.parse_dts_pdf(
            _minimal_dts_pdf("Thursday, December 28, 2023"),
            expected_record_date=date(2023, 12, 29),
        )


def test_pdf_parser_rejects_encryption_and_unsupported_filters() -> None:
    encrypted = b"%PDF-1.4\n1 0 obj\n<< /Encrypt true >>\nendobj\n%%EOF\n"
    with pytest.raises(ValueError, match="encrypted DTS PDFs"):
        dts._parse_pdf_objects(encrypted)

    payload = b"raw"
    item = dts._PdfObject(
        number=1,
        generation=0,
        offset=0,
        body=b"",
        dictionary=b"<< /Length 3 /Filter /DCTDecode >>",
        stream_start=0,
        stream_end_hint=3,
    )
    with pytest.raises(ValueError, match="unsupported filters"):
        dts._decode_pdf_stream(payload + b"\nendstream", item, {1: item})


def test_announcements_parser_uses_excel_dates_and_filters_postcap_rows() -> None:
    rows = dts.parse_announcements(_xlsx_fixture())
    assert len(rows) == 1
    assert rows[0].effective_date == date(2023, 12, 29)
    assert rows[0].change == "pre-cap change"
    assert dts._excel_serial_to_date("44927") == date(2023, 1, 1)
    with pytest.raises(ValueError, match="header changed"):
        dts.parse_announcements(_xlsx_fixture(corrupt_header=True))


def test_canonical_serialization_is_stable_and_unicode_is_normalized() -> None:
    first = dts.canonical_json({"b": 2, "a": "é"})
    second = dts.canonical_json({"a": "é", "b": 2})
    assert first == second == b'{"a":"\xc3\xa9","b":2}\n'
    assert dts._mechanical_label_normalization("Cafe\u0301 —  A") == "Café - A"


def test_gzip_serialization_uses_zero_timestamp(tmp_path: Path) -> None:
    first = tmp_path / "a.csv.gz"
    second = tmp_path / "b.csv.gz"
    dts.write_gzip(first, b"x,y\n1,2\n")
    dts.write_gzip(second, b"x,y\n1,2\n")
    assert first.read_bytes() == second.read_bytes()


def test_source_build_is_deterministic_and_keeps_postcap_metadata_out_of_logic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "source"
    raw = output / "raw"
    page_payload = _page_data(
            [
                _published_report(date(2023, 12, 29)),
                _published_report(date(2024, 1, 2)),
            ]
        )
    _write_bound_source(
        raw / "page-data.json", url=dts.DATASET_PAGE_DATA_URL, payload=page_payload
    )
    announcement_payload = _xlsx_fixture()
    _write_bound_source(
        raw / "DailyTreasuryStatement_Announcements.xlsx",
        url=dts.ANNOUNCEMENTS_URL,
        payload=announcement_payload,
    )
    report_payload = _minimal_dts_pdf()
    _write_bound_source(
        raw / "reports" / "20231229.pdf",
        url=dts.report_url(date(2023, 12, 29)),
        payload=report_payload,
    )
    config = dts.BuildConfig(
        start_date="2023-12-29",
        end_date="2023-12-29",
        output_dir=str(output),
        max_workers=1,
        request_pace_seconds=0,
    )

    first = dts.build_source(config)
    manifest_before = (output / "source_manifest.json").read_bytes()
    rows_before = (output / "daily_treasury_fiscal_flow_rows.csv.gz").read_bytes()
    second = dts.build_source(config)

    assert first == second
    assert first["calendar_coverage_gate_pass"] is True
    assert first["source_quality_gates_evaluated"] is False
    assert first["all_source_quality_gates_pass"] is None
    assert first["next_stage_authorized"] is None
    assert first["stage_report_counts"] == {"boundary_quarantine": 1}
    assert first["protocol"]["post_2023_report_opened"] is False
    assert first["protocol"]["current_metadata_postcap_rows_used_in_logic"] is False
    assert (output / "source_manifest.json").read_bytes() == manifest_before
    assert (output / "daily_treasury_fiscal_flow_rows.csv.gz").read_bytes() == rows_before
    manifest = json.loads(manifest_before)
    assert [row["record_date"] for row in manifest["reports"]] == ["2023-12-29"]
    assert manifest["metadata"]["precap_announcement_count"] == 1


def test_unbound_cached_source_bytes_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "20231229.pdf"
    target.write_bytes(_minimal_dts_pdf())
    with pytest.raises(ValueError, match="unbound cached source bytes"):
        dts._load_or_fetch(
            url=dts.report_url(date(2023, 12, 29)),
            target=target,
            allow_metadata=False,
            config=dts.BuildConfig(request_pace_seconds=0),
        )
