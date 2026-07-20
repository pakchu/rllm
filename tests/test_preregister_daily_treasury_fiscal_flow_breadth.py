from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest

from training import preregister_daily_treasury_fiscal_flow_breadth as prereg


def _cfg(tmp_path: Path, name: str = "dffb-prereg.json") -> prereg.Config:
    return prereg.Config(preregistration_output=str(tmp_path / name))


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    return payload


def _write_tampered(
    path: Path, artifact: dict[str, Any], mutator: Callable[[dict[str, Any]], None]
) -> None:
    drift = deepcopy(artifact)
    mutator(drift)
    path.write_text(json.dumps(_rehash(drift), sort_keys=True) + "\n", encoding="utf-8")


def test_writes_exact_singleton_outcome_boundary_and_no_incidence(
    tmp_path: Path,
) -> None:
    artifact = prereg.write_preregistration(_cfg(tmp_path))

    assert artifact == json.loads((tmp_path / "dffb-prereg.json").read_text())
    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["policy_id"] == "DFFB-601"
    assert artifact["policy"]["singleton"] is True
    assert artifact["outcomes_opened"] is False
    assert artifact["incidence_or_support_results"] is None
    assert artifact["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert artifact["outcome_boundary"]["source_value_rows_read"] == 0
    assert artifact["outcome_boundary"]["schema_transition_rows_read"] == 0
    assert artifact["outcome_boundary"]["comparator_clock_rows_read"] == 0
    assert artifact["outcome_boundary"]["auction_source_rows_read"] == 0
    assert artifact["outcome_boundary"]["signal_incidence_rows_derived"] == 0
    assert artifact["outcome_boundary"]["market_rows_loaded"] == 0
    assert artifact["outcome_boundary"]["return_or_pnl_fields_read"] == 0
    assert artifact["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in artifact.items() if key != "manifest_hash"}
    )
    assert artifact["policy_hash"] == (
        "14ed526851127c1fdc86f2795b4c3007e9f38f00bff4305f07a0b99e1b2dff4e"
    )
    assert artifact["policy_hash"] == prereg.canonical_hash(artifact["policy"])
    assert artifact["preregistration_document"] == {
        "path": str(prereg.PREREGISTRATION_DOCUMENT),
        "sha256": "5b6fe09d9e27c01c084d1e86fbaddfcfcbadcde531623f466ac6d5796621decb",
    }
    assert artifact["access_ledger"]["event_count"] == 37
    assert artifact["access_ledger"]["ledger_hash"] == (
        "adec4555879e35680340f1036c1a80f6b2e6f303f6710a76d60a1d36c872f738"
    )
    assert prereg.load_preregistration(tmp_path / "dffb-prereg.json") == artifact


def test_policy_freezes_strict_prior_exclusions_null_support_and_novelty(
    tmp_path: Path,
) -> None:
    policy = prereg.write_preregistration(_cfg(tmp_path))["policy"]

    source = policy["source_universe"]
    assert source["allowed_table_sides"] == [
        ["II", "deposit"],
        ["II", "withdrawal"],
        ["IIIA", "issue"],
        ["IIIA", "redemption"],
    ]
    assert source["required_row_kind"] == "detail"
    assert source["sign_transform"] == (
        "none; no absolute value and no side-dependent multiplication"
    )
    assert source["prohibited_value_fields"] == [
        "month_to_date_amount_usd_millions",
        "fiscal_year_to_date_amount_usd_millions",
    ]
    assert source["excluded_prefixes"] == [
        "total ",
        "sub-total ",
        "subtotal ",
        "net change",
        "change in balance",
    ]
    assert source["excluded_table_ii_bridge_prefixes"] == [
        "public debt cash issues",
        "public debt cash redemp",
    ]
    assert source["exclusion_fields"] == [
        "raw_category_label",
        "normalized_category_label",
    ]
    assert "transfers from tga (table v)" in source["excluded_exact_labels"]

    assert policy["missingness"] == {
        "birth": "first causal report printing the category identity",
        "first_appearance_current": "not prior-known and not rankable",
        "absent_after_birth": 0,
        "printed_null": "non-computable; never coerce to zero",
        "death_handling": "no future-detected death; trailing print-frequency eligibility removes stale identities causally",
    }
    assert policy["category_rank"]["prior_report_dates"] == 60
    assert policy["category_rank"]["minimum_prior_non_null_prints"] == 12
    assert policy["category_rank"]["current_excluded"] is True
    assert (
        policy["category_rank"]["null_in_current_or_window"]
        == "category non-computable"
    )
    assert policy["impulse_rank"]["prior_computable_impulses"] == 126
    assert policy["impulse_rank"]["current_excluded"] is True
    assert policy["impulse_rank"]["noncomputable_reports_skipped_from_history"] is True
    assert policy["event"] == {
        "long": "cash_rank126>=0.75 and debt_rank126>=0.75",
        "short": "cash_rank126<=0.25 and debt_rank126<=0.25",
        "otherwise": "none",
        "threshold_search": "forbidden",
    }
    assert policy["support_gates"]["train_total_minimum"] == 24
    assert policy["support_gates"]["selection_total_minimum"] == 12
    assert policy["support_gates"]["all_novelty_and_exposure_gates"] is True
    assert policy["comparators"]["decision_date_jaccard_maximum"] == 0.30
    assert (
        policy["comparators"]["dffb_within_one_us_business_day_fraction_maximum"]
        == 0.50
    )
    assert (
        policy["comparators"]["signed_occupied_exposure_absolute_pearson_maximum"]
        == 0.40
    )
    assert "no repair" in policy["support_gates"]["failure_action"]


def test_source_audit_comparator_hashes_and_headers_are_bound_exactly(
    tmp_path: Path,
) -> None:
    artifact = prereg.write_preregistration(_cfg(tmp_path))
    source = artifact["source_binding"]
    comparators = artifact["comparator_binding"]

    assert source["source_decision"] == {
        "path": str(prereg.SOURCE_DECISION),
        "sha256": prereg.SOURCE_DECISION_SHA256,
    }
    assert source["source_rows"] == {
        "path": str(prereg.SOURCE_ROWS),
        "sha256": prereg.SOURCE_ROWS_SHA256,
        "header": prereg.SOURCE_ROWS_HEADER,
    }
    assert source["operating_cash_rows"]["header"] == prereg.OPERATING_CASH_HEADER
    assert source["schema_transitions"]["sha256"] == prereg.SCHEMA_TRANSITIONS_SHA256
    assert source["schema_transitions"]["header"] == prereg.SCHEMA_TRANSITIONS_HEADER
    assert source["audit_manifest"]["sha256"] == prereg.AUDIT_MANIFEST_SHA256
    assert source["audit_report"]["sha256"] == prereg.AUDIT_REPORT_SHA256

    assert (
        comparators["flcc"]["preregistration"]["sha256"]
        == prereg.FLCC_PREREGISTRATION_SHA256
    )
    assert comparators["flcc"]["clock"]["sha256"] == prereg.FLCC_CLOCK_SHA256
    assert comparators["flcc"]["clock"]["header"] == prereg.FLCC_CLOCK_HEADER
    assert (
        comparators["flcc"]["clock"]["allowed_columns"] == prereg.FLCC_ALLOWED_COLUMNS
    )
    assert (
        comparators["tadi"]["preregistration"]["sha256"]
        == prereg.TADI_PREREGISTRATION_SHA256
    )
    assert comparators["tadi"]["clock"]["sha256"] == prereg.TADI_CLOCK_SHA256
    assert comparators["tadi"]["clock"]["header"] == prereg.TADI_CLOCK_HEADER
    assert (
        comparators["tadi"]["clock"]["allowed_columns"] == prereg.TADI_ALLOWED_COLUMNS
    )
    calendar = comparators["official_auction_settlement_calendar"]
    assert calendar["manifest"]["sha256"] == prereg.AUCTION_MANIFEST_SHA256
    assert calendar["normalized_panel"]["header"] == prereg.AUCTION_PANEL_HEADER
    assert (
        calendar["normalized_panel"]["allowed_columns"]
        == prereg.AUCTION_PANEL_ALLOWED_COLUMNS
    )
    assert calendar["raw_allowed_fields"] == prereg.AUCTION_RAW_ALLOWED_FIELDS


def test_hash_and_header_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prereg, "SOURCE_ROWS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="normalized source rows SHA drift"):
        prereg.write_preregistration(_cfg(tmp_path, "hash-drift.json"))

    monkeypatch.undo()
    real_read_header = prereg._read_gzip_header

    def drift_source_header(path: str | Path) -> list[str]:
        header = real_read_header(path)
        if prereg._repository_path(path) == prereg._repository_path(prereg.SOURCE_ROWS):
            return [*header, "future_return"]
        return header

    monkeypatch.setattr(prereg, "_read_gzip_header", drift_source_header)
    with pytest.raises(RuntimeError, match="normalized source rows header drift"):
        prereg.write_preregistration(_cfg(tmp_path, "header-drift.json"))


def test_bound_input_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "source-decision-link.md"
    link.symlink_to(prereg._repository_path(prereg.SOURCE_DECISION))
    monkeypatch.setattr(prereg, "SOURCE_DECISION", link)

    with pytest.raises(RuntimeError, match="bound input is a symlink"):
        prereg.write_preregistration(_cfg(tmp_path))


def test_access_ledger_and_import_boundary_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = prereg.AccessLedger()
    with pytest.raises(RuntimeError, match="not allowlisted"):
        ledger.record(prereg.SOURCE_ROWS, "value_rows")

    prohibited_source = tmp_path / "preregister_with_network.py"
    prohibited_source.write_text("import socket\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "PREREGISTRATION_SOURCE", prohibited_source)
    with pytest.raises(RuntimeError, match="imports prohibited clients"):
        prereg.write_preregistration(_cfg(tmp_path))


def test_comparator_preregistration_schema_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(
        prereg._repository_path(prereg.FLCC_PREREGISTRATION).read_text()
    )
    payload["measured_results"] = {"trades": 1}
    _rehash(payload)
    path = tmp_path / "flcc-prereg-with-results.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "FLCC_PREREGISTRATION", path)
    monkeypatch.setattr(prereg, "FLCC_PREREGISTRATION_SHA256", prereg.sha256_file(path))

    with pytest.raises(RuntimeError, match="FLCC preregistration schema drift"):
        prereg.write_preregistration(_cfg(tmp_path))


def test_invalid_source_audit_authorization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_report = deepcopy(
        json.loads(prereg._repository_path(prereg.AUDIT_REPORT).read_text())
    )
    audit_report["next_stage_authorized"] = "OUTCOME_EVALUATION"
    tampered_report = tmp_path / "source_quality_audit_report.json"
    tampered_report.write_text(
        json.dumps(audit_report, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(prereg, "AUDIT_REPORT", tampered_report)
    monkeypatch.setattr(
        prereg, "AUDIT_REPORT_SHA256", prereg.sha256_file(tampered_report)
    )

    with pytest.raises(RuntimeError, match="audit report next_stage_authorized drift"):
        prereg.write_preregistration(_cfg(tmp_path))


def test_source_hash_and_header_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prereg, "SOURCE_ROWS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="normalized source rows SHA drift"):
        prereg.write_preregistration(_cfg(tmp_path, "hash-drift.json"))

    monkeypatch.setattr(
        prereg, "SOURCE_ROWS_SHA256", prereg.sha256_file(prereg.SOURCE_ROWS)
    )
    monkeypatch.setattr(
        prereg, "SOURCE_ROWS_HEADER", ["record_date", "unexpected_column"]
    )
    with pytest.raises(RuntimeError, match="normalized source rows header drift"):
        prereg.write_preregistration(_cfg(tmp_path, "header-drift.json"))


def test_preregistration_reads_only_gzip_headers_not_value_or_comparator_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_gzip_open = gzip.open
    line_reads: list[str] = []

    class HeaderOnlyHandle:
        def __init__(self, handle: Any, name: str) -> None:
            self._handle = handle
            self._name = name

        def __enter__(self) -> "HeaderOnlyHandle":
            self._handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._handle.__exit__(*args)

        def readline(self, *args: Any, **kwargs: Any) -> str:
            if self._name in line_reads:
                raise AssertionError(f"read more than one gzip line from {self._name}")
            line_reads.append(self._name)
            return self._handle.readline(*args, **kwargs)

        def read(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError(f"read gzip body from {self._name}")

        def readlines(self, *args: Any, **kwargs: Any) -> list[str]:
            raise AssertionError(f"read gzip rows from {self._name}")

        def __iter__(self) -> Any:
            raise AssertionError(f"iterated gzip rows from {self._name}")

    def header_only_open(filename: Any, *args: Any, **kwargs: Any) -> HeaderOnlyHandle:
        return HeaderOnlyHandle(
            real_gzip_open(filename, *args, **kwargs), str(filename)
        )

    monkeypatch.setattr(prereg.gzip, "open", header_only_open)
    artifact = prereg.write_preregistration(_cfg(tmp_path))

    assert artifact["outcome_boundary"]["source_value_rows_read"] == 0
    assert artifact["outcome_boundary"]["comparator_clock_rows_read"] == 0
    assert set(Path(name).name for name in line_reads) == {
        prereg.SOURCE_ROWS.name,
        prereg.OPERATING_CASH_ROWS.name,
        prereg.SCHEMA_TRANSITIONS.name,
        prereg.FLCC_CLOCK.name,
        prereg.TADI_CLOCK.name,
        prereg.AUCTION_PANEL.name,
    }


def test_output_alias_immutability_and_cwd_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="protected source"):
        prereg.write_preregistration(
            prereg.Config(preregistration_output=str(prereg.SOURCE_MANIFEST))
        )
    with pytest.raises(ValueError, match="must be JSON"):
        prereg.write_preregistration(
            prereg.Config(preregistration_output=str(tmp_path / "artifact.txt"))
        )

    output = tmp_path / "nested" / "artifact.json"
    cfg = prereg.Config(preregistration_output=str(output))
    monkeypatch.chdir(tmp_path)
    artifact = prereg.write_preregistration(cfg)
    assert json.loads(output.read_text()) == artifact
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistration(cfg)


def test_load_rejects_policy_outcome_binding_document_and_config_tamper(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    artifact = prereg.write_preregistration(cfg)
    path = tmp_path / "dffb-prereg.json"

    _write_tampered(
        path, artifact, lambda value: value["policy"]["event"].update(otherwise="LONG")
    )
    with pytest.raises(RuntimeError, match="policy drift"):
        prereg.load_preregistration(path)

    _write_tampered(path, artifact, lambda value: value.update(outcomes_opened=True))
    with pytest.raises(RuntimeError, match="opened outcomes"):
        prereg.load_preregistration(path)

    _write_tampered(
        path,
        artifact,
        lambda value: value["source_binding"]["source_rows"].update(sha256="0" * 64),
    )
    with pytest.raises(RuntimeError, match="frozen source binding drift"):
        prereg.load_preregistration(path)

    _write_tampered(
        path,
        artifact,
        lambda value: value["comparator_binding"]["flcc"]["clock"].update(header=[]),
    )
    with pytest.raises(RuntimeError, match="comparator binding drift"):
        prereg.load_preregistration(path)

    _write_tampered(
        path,
        artifact,
        lambda value: value["preregistration_document"].update(sha256="0" * 64),
    )
    with pytest.raises(RuntimeError, match="document binding drift"):
        prereg.load_preregistration(path)

    _write_tampered(
        path,
        artifact,
        lambda value: value["config"].update(
            preregistration_output=str(tmp_path / "other.json")
        ),
    )
    with pytest.raises(RuntimeError, match="output-path binding drift"):
        prereg.load_preregistration(path)

    _write_tampered(
        path,
        artifact,
        lambda value: value["outcome_boundary"].update(market_rows_loaded=1),
    )
    with pytest.raises(RuntimeError, match="outcome boundary drift"):
        prereg.load_preregistration(path)

    broken = deepcopy(artifact)
    broken["policy_id"] = "DFFB-602"
    path.write_text(json.dumps(broken, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        prereg.load_preregistration(path)


def test_deterministic_byte_output_for_same_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    prereg.write_preregistration(cfg)
    first = (tmp_path / "dffb-prereg.json").read_bytes()
    (tmp_path / "dffb-prereg.json").unlink()

    prereg.write_preregistration(cfg)
    second = (tmp_path / "dffb-prereg.json").read_bytes()

    assert first == second


def test_committed_preregistration_artifact_is_exact() -> None:
    artifact = prereg.load_preregistration(prereg.DEFAULT_OUTPUT)

    assert prereg.sha256_file(prereg.DEFAULT_OUTPUT) == (
        "9370ead97eb0cf4ad4ffd271cf691b2d2de08a5099b942f7d3348352485ce6d6"
    )
    assert artifact["manifest_hash"] == (
        "67c98b014efc5c46c8096677eb6f6d77651e79001896759d915f86d35f6bbc4f"
    )
    assert artifact["policy_hash"] == (
        "14ed526851127c1fdc86f2795b4c3007e9f38f00bff4305f07a0b99e1b2dff4e"
    )
    assert artifact["outcomes_opened"] is False
    assert artifact["incidence_or_support_results"] is None


def test_direct_cli_writes_artifact_and_prints_same_json(tmp_path: Path) -> None:
    output = tmp_path / "cli-prereg.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(prereg._repository_path(prereg.PREREGISTRATION_SOURCE)),
            "--preregistration-output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output.read_text())
    assert stdout_payload == file_payload
    assert file_payload["policy_id"] == "DFFB-601"
    assert file_payload["outcomes_opened"] is False
    assert prereg.load_preregistration(output) == file_payload
