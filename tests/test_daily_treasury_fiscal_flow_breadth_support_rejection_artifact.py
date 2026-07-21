from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT = Path("results/daily_treasury_fiscal_flow_breadth_support_2026-07-21.json")
PRIMARY = Path(
    "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"
)
CONTROLS = Path(
    "results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz"
)
REPORT = Path("docs/daily-treasury-fiscal-flow-breadth-support-rejection-2026-07-21.md")

EXPECTED_HASHES = {
    RESULT: "a5bf3b15f40f05d876b7603eaa3104cfa21a867fa3dd1aa4681b6b0875c8f549",
    PRIMARY: "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978",
    CONTROLS: "416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312",
    REPORT: "e82c1dc74b66c49fdb4d1fa847aaa364d8139b7c0760601bfcfb7f606e345e16",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result() -> dict[str, Any]:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _clock(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def test_dffb_support_rejection_artifacts_are_hash_frozen() -> None:
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected

    result = _result()
    manifest_hash = result.pop("manifest_hash")
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == manifest_hash
    assert manifest_hash == (
        "05f2de6ab8982671d4adcf44ca7e77a25fd5aa9b0a33e840cce0a34efe2ab36c"
    )


def test_dffb_clocks_match_the_frozen_artifact() -> None:
    result = _result()
    artifacts = result["artifacts"]
    primary_header, primary_rows = _clock(PRIMARY)
    control_header, control_rows = _clock(CONTROLS)

    assert primary_header == artifacts["primary_clock"]["columns"]
    assert control_header == artifacts["control_clocks"]["columns"]
    assert len(primary_rows) == artifacts["primary_clock"]["rows"] == 112
    assert len(control_rows) == artifacts["control_clocks"]["rows"] == 1502
    assert Counter(row["clock"] for row in control_rows) == Counter(
        artifacts["control_clocks"]["clock_counts"]
    )
    assert _sha256(PRIMARY) == artifacts["primary_clock"]["sha256"]
    assert _sha256(CONTROLS) == artifacts["control_clocks"]["sha256"]


def test_dffb_is_rejected_only_by_the_frozen_source_only_novelty_gate() -> None:
    result = _result()
    assert result["support_gates"]["passed"] is True
    assert all(
        control["passed"] is True
        for control in result["control_support_gates"].values()
    )
    assert result["signed_occupied_exposure_gates"]["passed"] is True
    assert result["novelty_gates"]["passed"] is False
    assert result["all_source_only_gates_pass"] is False

    novelty = result["novelty_gates"]["comparators"]
    failed = {name for name, metrics in novelty.items() if not metrics["passed"]}
    assert failed == {
        "dts_total_net_cash",
        "flcc:union",
        "official_auction_settlement_calendar",
    }
    assert novelty["dts_total_net_cash"]["within_one_us_business_day_count"] == 102
    assert novelty["flcc:union"]["within_one_us_business_day_count"] == 58
    assert (
        novelty["official_auction_settlement_calendar"][
            "within_one_us_business_day_count"
        ]
        == 80
    )
    assert all(metrics["decision_date_jaccard"] <= 0.30 for metrics in novelty.values())


def test_dffb_rejection_keeps_the_outcome_boundary_closed() -> None:
    result = _result()
    assert result["performance_values_opened"] is False
    assert result["next_action"] == "reject DFFB-601 without opening outcomes"
    assert result["stopping_rule"].startswith("reject permanently without outcomes")
    boundary = result["outcome_boundary"]
    for field in (
        "database_calls",
        "funding_rows_loaded",
        "funding_values_read",
        "market_rows_loaded",
        "market_values_read",
        "network_calls",
        "return_or_pnl_fields_read",
        "return_rows_loaded",
        "schema_transition_rows_read",
        "subprocess_calls",
    ):
        assert boundary[field] == 0

    text = REPORT.read_text(encoding="utf-8")
    assert "permanently rejected before any BTC OHLC" in text
    assert "performance_values_opened = false" in text
    assert "may not proceed to outcomes" in text
