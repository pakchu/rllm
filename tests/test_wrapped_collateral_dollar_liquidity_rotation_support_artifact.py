from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from training import build_wrapped_collateral_dollar_liquidity_rotation_support as support


CLOCK = Path(
    "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
    "wcdr2016_support_clocks_2021_2023.csv.gz"
)
REPORT = Path(
    "results/wrapped_collateral_dollar_liquidity_rotation_"
    "support_2026-07-23.json"
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time(value: str) -> datetime:
    return support.parse_time(value)


def test_support_report_and_clock_are_hash_bound_and_outcome_blind() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    frozen_hash = report.pop("manifest_hash")
    assert _canonical_hash(report) == frozen_hash
    assert frozen_hash == (
        "0a28128c820c1f5baf73c7653901056d3803e9bc0cce54b29a03afc7051ef600"
    )
    assert report["clock"]["path"] == str(CLOCK)
    assert _sha256(CLOCK) == report["clock"]["sha256"]
    assert report["clock"]["rows"] == 818
    assert report["evaluator_source"]["sha256"] == (
        "3cbc2c5c06629775b240337dcdbaaa92c91e4410da0702787d1fba4f0f2d53c3"
    )
    assert report["source_audit"]["usdc"] == {
        "boundary_sentinel_timestamp_rows_scanned": 1,
        "eligible_rows": 265583,
        "first_available_at": "2020-01-01T04:44:32Z",
        "last_available_at": "2023-12-31T23:48:35Z",
        "physical_rows_read": 266360,
        "post_2023_contract_event_value_rows_loaded": 0,
        "sealed_from": "2024-01-01T00:00:00Z",
        "unique_identities": 265583,
    }
    assert report["source_support_passed"] is False
    assert report["advance_to_strict_evaluator_freeze"] is False
    assert report["decision"] == "retire_WCDR_2016_without_repair"
    assert report["outcome_boundary"] == {
        "outcomes_opened": False,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "post_2023_contract_event_rows_read": 0,
        "network_calls": 0,
        "subprocess_calls": 0,
    }


def test_support_clock_preserves_causal_nonoverlap_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    previous_exit: dict[str, datetime] = {}
    exact_clocks: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    rows = 0
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == report["clock"]["columns"]
        for row in reader:
            rows += 1
            control = row["control"]
            counts[control] += 1
            decision = _time(row["decision_time"])
            cutoff = _time(row["source_cutoff"])
            entry = _time(row["entry_time"])
            exit_time = _time(row["exit_time"])
            expected_staleness = (
                timedelta(days=7, hours=6)
                if control == "stale_7d"
                else timedelta(hours=6)
            )
            assert decision - cutoff == expected_staleness
            assert entry == decision + timedelta(minutes=5)
            assert exit_time == entry + timedelta(days=7)
            assert int(row["side"]) in {-1, 1}
            assert entry >= previous_exit.get(control, entry)
            previous_exit[control] = exit_time
            start, end = map(_time, support.WINDOWS[row["window"]])
            assert start <= entry and exit_time <= end
            exact_clocks[control].append((row["entry_time"], row["exit_time"]))

    assert rows == report["clock"]["rows"]
    assert dict(sorted(counts.items())) == report["clock"]["control_counts"]
    assert exact_clocks["direction_flip"] == exact_clocks["primary"]
    assert exact_clocks["deterministic_random_side"] == exact_clocks["primary"]


def test_rejection_is_exactly_the_frozen_source_support_failure() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["primary_support"]["train"]["trades"] == 44
    assert report["primary_support"]["train"]["year_counts"] == {
        "2021": 14,
        "2022": 30,
    }
    assert report["primary_support"]["selection"]["trades"] == 34
    assert report["primary_support"]["selection"]["side_counts"] == {
        "long": 6,
        "short": 28,
    }
    failed = {name for name, passed in report["support_checks"].items() if not passed}
    assert failed == {
        "train_total_minimum",
        "each_train_year_minimum",
        "each_train_half_year_minimum",
        "maximum_consecutive_same_side",
    }
