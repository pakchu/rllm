from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_wbtc_stablecoin_finalized_confirmation_relay_support as support


REPORT = Path(
    "results/wbtc_stablecoin_finalized_confirmation_relay_"
    "support_2026-07-23.json"
)
CLOCK = Path(
    "data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/"
    "wscf72_support_clocks_2021_2023.csv.gz"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_artifacts_are_hash_bound_terminal_rejection() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert _sha256(REPORT) == (
        "add1f54034953d1040fdf5b34d794865fde84d05675c8b7f7f8e4e8c7918f2bd"
    )
    assert _sha256(CLOCK) == (
        "86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3"
    )
    assert payload["manifest_hash"] == (
        "1a7ec88467779e461217af1430f79f21fdeb127ba7f29abd1a836a36c99b1faf"
    )
    assert payload["decision"] == "REJECT_SOURCE"
    assert payload["next_action"] == "retire candidate without BTC outcomes or repair"
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False


def test_rejection_preserves_exact_failed_gates_and_support_statistics() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["failed_checks"] == [
        "maximum_consecutive_same_side",
        "novelty:sealed_prior_stablecoin_bundle:AMTR-48:cross_minter",
        "novelty:sealed_prior_stablecoin_bundle:SDDR-12:primary",
        "novelty:sealed_prior_stablecoin_bundle:SQFD-6:no_participation",
        "novelty:sealed_prior_stablecoin_bundle:SQFD-6:no_usdt_lag",
        "novelty:sealed_prior_stablecoin_bundle:SQFD-6:primary",
        "novelty:sealed_prior_stablecoin_bundle:UCBR-12:primary",
    ]
    primary = payload["primary_support"]
    assert primary["total_trades"] == 193
    assert primary["all_year_counts"] == {"2021": 72, "2022": 69, "2023": 52}
    assert primary["train"]["trades"] == 141
    assert primary["train"]["side_counts"] == {"long": 87, "short": 54}
    assert primary["train"]["maximum_consecutive_same_side"] == 20
    assert primary["selection"]["trades"] == 52
    assert primary["selection"]["side_counts"] == {"long": 28, "short": 24}
    assert payload["controls"]["usdc_only_confirmation"]["trades"] == 192
    assert payload["controls"]["usdt_only_confirmation"]["trades"] == 10


def test_clock_schema_counts_and_sealed_boundary_are_source_only() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == support.CLOCK_COLUMNS
    assert len(rows) == 2681
    assert {row["control"] for row in rows} == set(support.CONTROL_ORDER)
    assert sum(row["control"] == "primary" for row in rows) == 193
    boundary = payload["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["post_2023_contract_event_value_rows_loaded"] == 0
    assert boundary["sealed_non_timestamp_fields_decoded"] == 0
    assert payload["source_audit"]["stablecoin"][
        "boundary_sentinel_timestamp_rows_scanned"
    ] == 1
