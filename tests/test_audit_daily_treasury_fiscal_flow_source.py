from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import audit_daily_treasury_fiscal_flow_source as audit


def _snapshot(
    *,
    rows: tuple[dict[str, str], ...] = (),
    table_i_rows: tuple[dict[str, str], ...] = (),
    announcements: tuple[dict[str, str], ...] = (),
    report_dates: tuple[str, ...] = (),
) -> audit.SourceSnapshot:
    reports = [
        {
            "record_date": report_date,
            "source_available_not_before_utc": datetime.combine(
                date.fromisoformat(report_date) + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).replace(hour=21).isoformat(),
        }
        for report_date in report_dates
    ]
    return audit.SourceSnapshot(
        source_dir=Path("/source-only-fixture"),
        manifest={"reports": reports},
        build_report={},
        rows=rows,
        table_i_rows=table_i_rows,
        announcements=announcements,
    )


def _amount_row(**overrides: str) -> dict[str, str]:
    row = {
        "record_date": "2023-01-03",
        "table_id": "II",
        "side": "deposit",
        "today_amount_usd_millions": "1234",
        "today_amount_literal": "1,234",
        "month_to_date_amount_usd_millions": "",
        "month_to_date_amount_literal": "-",
        "fiscal_year_to_date_amount_usd_millions": "-12",
        "fiscal_year_to_date_amount_literal": "(12)",
    }
    row.update(overrides)
    return row


def _write_physical_inventory_fixture(root: Path) -> dict[str, dict[str, object]]:
    raw = root / "raw"
    reports_dir = raw / "reports"
    reports_dir.mkdir(parents=True)
    for name in (
        "daily_treasury_fiscal_flow_rows.csv.gz",
        "daily_treasury_operating_cash_rows.csv.gz",
        "precap_schema_announcements.csv.gz",
        "source_build_report.json",
        "source_manifest.json",
    ):
        (root / name).write_bytes(b"")
    (raw / "page-data.json").write_bytes(b"page")
    (raw / "DailyTreasuryStatement_Announcements.xlsx").write_bytes(b"announcements")
    report_date = "2023-12-29"
    (reports_dir / "20231229.pdf").write_bytes(b"pdf")
    receipt_specs = {
        audit.source.DATASET_PAGE_DATA_URL: raw / "page-data.json.receipt.json",
        audit.source.ANNOUNCEMENTS_URL: (
            raw / "DailyTreasuryStatement_Announcements.xlsx.receipt.json"
        ),
        audit.source.report_url(date.fromisoformat(report_date)): (
            reports_dir / "20231229.pdf.receipt.json"
        ),
    }
    receipts = []
    for url, path in receipt_specs.items():
        receipt = {"url": url, "final_url": url}
        path.write_text(json.dumps(receipt), encoding="utf-8")
        receipts.append(receipt)
    (root / "receipt_log.jsonl").write_text(
        "".join(json.dumps(receipt) + "\n" for receipt in receipts),
        encoding="utf-8",
    )
    return {report_date: {"record_date": report_date}}


def test_rounding_tolerance_accepts_zero_components_and_exact_boundary() -> None:
    assert audit.rounding_tolerance(0) == 0.5
    assert audit.rounding_tolerance(2) == 1.5
    with pytest.raises(ValueError, match="non-negative"):
        audit.rounding_tolerance(-1)

    rows: list[dict[str, object]] = []
    audit._append_reconciliation(
        rows,
        record_date="2023-01-03",
        table_id="II",
        side="deposit",
        check_name="boundary",
        amount_column="today",
        observed=2,
        components=[1],
    )
    audit._append_reconciliation(
        rows,
        record_date="2023-01-03",
        table_id="II",
        side="deposit",
        check_name="outside_boundary",
        amount_column="today",
        observed=3,
        components=[1],
    )
    assert [row["status"] for row in rows] == ["PASS", "FAIL"]


def test_published_accounting_anomaly_is_exactly_quarantined() -> None:
    rows: list[dict[str, object]] = []
    audit._append_reconciliation(
        rows,
        record_date="2020-04-01",
        table_id="II",
        side="both",
        check_name="account_deposits_minus_withdrawals_equals_net_change",
        amount_column="month_to_date",
        observed=-8324,
        components=[75371, 83786],
        signs=[1, -1],
    )
    near_misses = (
        ("2020-04-02", "account_deposits_minus_withdrawals_equals_net_change", "month_to_date", -8324, [75371, 83786]),
        ("2020-04-01", "different_check", "month_to_date", -8324, [75371, 83786]),
        ("2020-04-01", "account_deposits_minus_withdrawals_equals_net_change", "today", -8324, [75371, 83786]),
        ("2020-04-01", "account_deposits_minus_withdrawals_equals_net_change", "month_to_date", -8323, [75371, 83786]),
        ("2020-04-01", "account_deposits_minus_withdrawals_equals_net_change", "month_to_date", -8324, [75370, 83786]),
    )
    for record_date, check_name, amount_column, observed, components in near_misses:
        audit._append_reconciliation(
            rows,
            record_date=record_date,
            table_id="II",
            side="both",
            check_name=check_name,
            amount_column=amount_column,
            observed=observed,
            components=components,
            signs=[1, -1],
        )
    assert rows[0]["status"] == "PUBLISHED_SOURCE_ANOMALY"
    assert all(row["status"] == "FAIL" for row in rows[1:])


def test_toolchain_and_physical_inventory_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": audit.source.SCHEMA_VERSION,
        "parser_version": audit.source.PARSER_VERSION,
        "toolchain": audit._active_toolchain(),
    }
    assert audit._verify_toolchain_binding(manifest) == []
    manifest["toolchain"] = {**manifest["toolchain"], "parser_source_sha256": "0" * 64}
    assert "toolchain" in " ".join(audit._verify_toolchain_binding(manifest))

    reports = _write_physical_inventory_fixture(tmp_path)
    assert audit._verify_physical_inventory(tmp_path, reports) == []

    postcap = tmp_path / "raw" / "reports" / "20240102.pdf"
    postcap.write_bytes(b"forbidden")
    errors = audit._verify_physical_inventory(tmp_path, reports)
    assert any("raw report inventory mismatch" in error for error in errors)
    postcap.unlink()

    receipt_path = tmp_path / "raw" / "page-data.json.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["final_url"] = "https://example.com/escaped"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    errors = audit._verify_physical_inventory(tmp_path, reports)
    assert any("escaped source allowlist" in error for error in errors)


def test_numeric_literal_roundtrip_passes_and_detects_tampering() -> None:
    summaries, errors, checked = audit.audit_numeric_roundtrip(
        _snapshot(rows=(_amount_row(),))
    )
    assert checked == 3
    assert not errors
    assert summaries[0]["status"] == "PASS"

    summaries, errors, _ = audit.audit_numeric_roundtrip(
        _snapshot(rows=(_amount_row(today_amount_literal="1,235"),))
    )
    assert errors
    assert summaries[0]["failed_cell_count"] == 1
    assert summaries[0]["status"] == "FAIL"


def test_duplicate_audit_is_side_aware() -> None:
    base = {
        "record_date": "2023-01-03",
        "table_id": "II",
        "normalized_category_label": "Shared label",
        "raw_category_label": "Shared label",
        "parent_section": "TGA",
    }
    rows, errors = audit.audit_duplicate_labels(
        _snapshot(
            rows=(
                {**base, "side": "deposit"},
                {**base, "side": "withdrawal"},
            )
        )
    )
    assert rows == []
    assert errors == []

    rows, errors = audit.audit_duplicate_labels(
        _snapshot(
            rows=(
                {**base, "side": "deposit"},
                {**base, "side": "deposit"},
            )
        )
    )
    assert len(rows) == 1
    assert errors


def test_stable_transition_detection_filters_sparse_presence_and_pairs_labels() -> None:
    first = date(2023, 1, 1)
    report_dates = tuple((first + timedelta(days=index)).isoformat() for index in range(14))
    rows: list[dict[str, str]] = []
    table_i_rows: list[dict[str, str]] = []
    for index, report_date in enumerate(report_dates):
        parent = "Before" if index < 7 else "After"
        rows.append(
            {
                "record_date": report_date,
                "table_id": "II",
                "side": "deposit",
                "normalized_category_label": "Stable row",
                "raw_category_label": "Stable row",
                "parent_section": parent,
            }
        )
        raw_label = "Dept - Misc" if index < 7 else "dept Misc"
        rows.append(
            {
                "record_date": report_date,
                "table_id": "II",
                "side": "withdrawal",
                "normalized_category_label": raw_label,
                "raw_category_label": raw_label,
                "parent_section": "TGA",
            }
        )
        if 5 <= index < 10:
            rows.append(
                {
                    "record_date": report_date,
                    "table_id": "II",
                    "side": "deposit",
                    "normalized_category_label": "Sparse activity row",
                    "raw_category_label": "Sparse activity row",
                    "parent_section": "TGA",
                }
            )
        if index >= 7:
            rows.append(
                {
                    "record_date": report_date,
                    "table_id": "II",
                    "side": "deposit",
                    "normalized_category_label": "New Revenue Channel",
                    "raw_category_label": "New Revenue Channel",
                    "parent_section": "TGA",
                }
            )
        table_i_rows.append(
            {
                "record_date": report_date,
                "normalized_category_label": "Stable cash row",
                "raw_category_label": "Stable cash row",
                "schema_variant": (
                    "legacy_four_column" if index < 7 else "modern_three_column"
                ),
            }
        )

    announcement = {
        "effective_date": report_dates[7],
        "table_name": "2",
        "table_section": "Deposit",
        "entity": "New Revenue Channel",
        "change": "Add New Revenue Channel to Table 2 deposits.",
        "sheet_name": "DTS Changes",
        "source_row": "1",
    }
    snapshot = _snapshot(
        rows=tuple(rows),
        table_i_rows=tuple(table_i_rows),
        announcements=(announcement,),
        report_dates=report_dates,
    )
    transitions, errors = audit.detect_schema_transitions(snapshot)
    assert not errors
    assert {row["transition_type"] for row in transitions} == {
        "column_schema_change",
        "stable_birth",
        "stable_label_change",
        "stable_parent_section_change",
    }
    assert all(
        row["normalized_category_label"] != "Sparse activity row"
        for row in transitions
    )
    birth = next(
        row
        for row in transitions
        if row["normalized_category_label"] == "New Revenue Channel"
    )
    assert birth["report_date"] == report_dates[7]
    assert birth["source_available_date_new_york"] == report_dates[8]
    assert birth["support_before_reports"] == 5
    assert birth["support_after_reports"] == 5
    _, errors = audit.reconcile_announcements(snapshot, [birth])
    assert not errors
    assert birth["status"] == "RECONCILED_ANNOUNCEMENT"


def test_announcement_scope_handles_global_and_excludes_other_table_iii() -> None:
    assert ("II", "deposit") in audit._announcement_scope(
        "2 & 4", "Deposit"
    )
    assert audit._announcement_scope(
        "Various",
        "Various",
        "All references to Federal Reserve Account throughout the DTS will change.",
    ) == {
        ("I", "operating_cash"),
        ("II", "deposit"),
        ("II", "withdrawal"),
        ("IIIA", "issue"),
        ("IIIA", "redemption"),
    }
    assert audit._announcement_scope(
        "3", "", "The asterisk in Table III-C will change."
    ) == set()


def test_global_announcement_reconciles_parent_transition() -> None:
    report_date = "2021-10-01"
    announcement = {
        "effective_date": report_date,
        "table_name": "Various",
        "table_section": "Various",
        "entity": "",
        "change": (
            "All references to Federal Reserve Account throughout the DTS will be "
            "changed to Treasury General Account (TGA)."
        ),
        "sheet_name": "DTS Changes",
        "source_row": "1",
    }
    transition: dict[str, object] = {
        "report_date": report_date,
        "source_available_date_new_york": "2021-10-04",
        "table_id": "II",
        "side": "deposit",
        "transition_type": "stable_parent_section_change",
        "normalized_category_label": "Federal Reserve Earnings",
        "before_label": "Federal Reserve Earnings",
        "after_label": "Federal Reserve Earnings",
        "before_parent_section": "Federal Reserve Account",
        "after_parent_section": "Treasury General Account (TGA)",
        "support_before_reports": 5,
        "support_after_reports": 5,
        "announcement_match_count": 0,
        "status": "UNRECONCILED",
    }
    announcement_rows, errors = audit.reconcile_announcements(
        _snapshot(
            announcements=(announcement,),
            report_dates=(report_date,),
        ),
        [transition],
    )
    assert not errors
    assert announcement_rows[0]["status"] == "MATCHED_TRANSITION"
    assert transition["status"] == "RECONCILED_ANNOUNCEMENT"


def test_source_equivalent_label_pair_reconciles_without_announcement() -> None:
    report_date = "2023-02-14"
    transition: dict[str, object] = {
        "report_date": report_date,
        "source_available_date_new_york": "2023-02-15",
        "table_id": "II",
        "side": "withdrawal",
        "transition_type": "stable_label_change",
        "normalized_category_label": "Dept - misc",
        "before_label": "Dept - Misc",
        "after_label": "dept Misc",
        "before_parent_section": "TGA",
        "after_parent_section": "TGA",
        "support_before_reports": 5,
        "support_after_reports": 5,
        "announcement_match_count": 0,
        "status": "UNRECONCILED",
    }
    announcement_rows, errors = audit.reconcile_announcements(
        _snapshot(report_dates=(report_date,)), [transition]
    )
    assert announcement_rows == []
    assert not errors
    assert transition["status"] == "RECONCILED_SOURCE_EQUIVALENCE"


def test_unmatched_structural_transition_fails_reconciliation() -> None:
    report_date = "2023-02-14"
    transition: dict[str, object] = {
        "report_date": report_date,
        "source_available_date_new_york": "2023-02-15",
        "table_id": "II",
        "side": "deposit",
        "transition_type": "stable_parent_section_change",
        "normalized_category_label": "Unannounced structural row",
        "before_label": "Unannounced structural row",
        "after_label": "Unannounced structural row",
        "before_parent_section": "Before",
        "after_parent_section": "After",
        "support_before_reports": 5,
        "support_after_reports": 5,
        "announcement_match_count": 0,
        "status": "UNRECONCILED",
    }
    _, errors = audit.reconcile_announcements(
        _snapshot(report_dates=(report_date,)), [transition]
    )
    assert errors == [
        "unannounced stable transition 2023-02-14 II/deposit "
        "stable_parent_section_change Unannounced structural row"
    ]
    assert transition["status"] == "UNRECONCILED"


def test_csv_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    rows = [{"b": "2", "a": "1"}]
    first = audit._write_csv_artifact(tmp_path / "first.csv.gz", rows, ("a", "b"))
    second = audit._write_csv_artifact(tmp_path / "second.csv.gz", rows, ("a", "b"))
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    assert first["file_sha256"] == second["file_sha256"]
    assert (tmp_path / "first.csv.gz").read_bytes() == (
        tmp_path / "second.csv.gz"
    ).read_bytes()


def test_full_audit_artifacts_are_deterministic_and_gates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    (source_dir / "source_build_report.json").write_text("{}\n", encoding="utf-8")
    snapshot = audit.SourceSnapshot(
        source_dir=source_dir,
        manifest={},
        build_report={},
        rows=(),
        table_i_rows=(),
        announcements=(),
    )

    def transition_rows() -> list[dict[str, object]]:
        return [
            {
                "report_date": "2023-02-14",
                "source_available_date_new_york": "2023-02-15",
                "table_id": "II",
                "side": "withdrawal",
                "transition_type": "stable_label_change",
                "normalized_category_label": "Dept misc",
                "before_label": "Dept - Misc",
                "after_label": "dept Misc",
                "before_parent_section": "TGA",
                "after_parent_section": "TGA",
                "support_before_reports": 5,
                "support_after_reports": 5,
                "announcement_match_count": 0,
                "status": "RECONCILED_SOURCE_EQUIVALENCE",
            }
        ]

    monkeypatch.setattr(audit, "load_snapshot", lambda _: snapshot)
    monkeypatch.setattr(audit, "verify_snapshot_binding", lambda _: (True, []))
    monkeypatch.setattr(audit, "audit_numeric_roundtrip", lambda _: ([], [], 0))
    monkeypatch.setattr(audit, "audit_accounting", lambda _: ([], []))
    monkeypatch.setattr(audit, "audit_duplicate_labels", lambda _: ([], []))
    monkeypatch.setattr(
        audit, "detect_schema_transitions", lambda _: (transition_rows(), [])
    )
    monkeypatch.setattr(
        audit, "reconcile_announcements", lambda _snapshot, _rows: ([], [])
    )
    clean_rerun_calls = 0

    def clean_rerun(*_args: object, **_kwargs: object) -> tuple[bool, list[str]]:
        nonlocal clean_rerun_calls
        clean_rerun_calls += 1
        return True, []

    monkeypatch.setattr(audit, "verify_clean_rerun", clean_rerun)

    outputs = [tmp_path / "audit-1", tmp_path / "audit-2"]
    reports = [
        audit.run_audit(
            audit.AuditConfig(
                source_dir=str(source_dir),
                output_dir=str(output),
                verify_clean_rerun=True,
            )
        )
        for output in outputs
    ]
    assert all(report["all_source_quality_gates_pass"] for report in reports)
    assert clean_rerun_calls == 2
    assert sorted(path.name for path in outputs[0].iterdir()) == sorted(
        path.name for path in outputs[1].iterdir()
    )
    for first in outputs[0].iterdir():
        assert first.read_bytes() == (outputs[1] / first.name).read_bytes()

    monkeypatch.setattr(
        audit,
        "verify_snapshot_binding",
        lambda _: (False, ["tampered source binding"]),
    )
    failed = audit.run_audit(
        audit.AuditConfig(
            source_dir=str(source_dir),
            output_dir=str(tmp_path / "audit-failed"),
            verify_clean_rerun=True,
        )
    )
    assert failed["decision"] == "SOURCE_QUALITY_FAIL"
    assert failed["all_source_quality_gates_pass"] is False
    assert "tampered source binding" in failed["failures"]
    assert "clean rerun blocked by failed source binding" in failed["failures"]
    assert clean_rerun_calls == 2

    monkeypatch.setattr(audit, "verify_snapshot_binding", lambda _: (True, []))
    skipped = audit.run_audit(
        audit.AuditConfig(
            source_dir=str(source_dir),
            output_dir=str(tmp_path / "audit-skipped"),
            verify_clean_rerun=False,
        )
    )
    assert skipped["all_source_quality_gates_pass"] is False
    assert skipped["failures"] == ["clean rerun was skipped"]
    assert clean_rerun_calls == 2

    monkeypatch.setattr(
        audit,
        "run_audit",
        lambda _config: {"all_source_quality_gates_pass": False},
    )
    assert audit.main([]) == 2


def test_clean_rerun_refuses_unverified_binding_without_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    def forbidden_build(_config: object) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError("network-capable build must not start")

    monkeypatch.setattr(audit.source, "build_source", forbidden_build)
    passed, errors = audit.verify_clean_rerun(_snapshot(), max_workers=1)
    assert passed is False
    assert errors == ["clean rerun requires a verified frozen source binding"]
    assert attempted is False


def test_direct_cli_help_is_importable_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "training/audit_daily_treasury_fiscal_flow_source.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Audit a frozen Daily Treasury Statement" in completed.stdout
