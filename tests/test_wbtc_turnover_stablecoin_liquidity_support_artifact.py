from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from training import build_wbtc_turnover_stablecoin_liquidity_support as support


CLOCK = Path(
    "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
    "wtsl168_support_clocks_2021_2023.csv.gz"
)
REPORT = Path(
    "results/wbtc_turnover_stablecoin_liquidity_support_2026-07-23.json"
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


def test_support_report_is_hash_bound_and_rejects_before_outcomes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    frozen_hash = report.pop("manifest_hash")
    assert _canonical_hash(report) == frozen_hash
    assert frozen_hash == (
        "b53de47d743f7f61240e59ac3149c0a37467f6bb8ce580c9c3c2bc84341b7e9e"
    )
    assert _sha256(REPORT) == (
        "1415b8e2a40f2aff908bfec1d1faa9621445c3fe87b41c43fd95a991725b23bd"
    )
    assert _sha256(CLOCK) == (
        "df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d"
    )
    assert report["evaluator_source"]["sha256"] == (
        "c527e0d8b6e64657e9e6a49f0f13a53acd589c26677d1a790f8e69d2faf4e57e"
    )
    assert report["source_support_passed"] is False
    assert report["advance_to_strict_evaluator_freeze"] is False
    assert report["decision"] == (
        "retire_WTSL_168_SOURCE_SEEN_without_outcomes"
    )
    assert report["outcome_boundary"]["outcomes_opened"] is False
    assert report["outcome_boundary"]["btc_market_rows_read"] == 0
    assert report["outcome_boundary"]["funding_rows_read"] == 0
    assert report["outcome_boundary"]["future_return_rows_read"] == 0


def test_clock_preserves_causal_execution_and_nonoverlap() -> None:
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
            expected_lag = {
                "stale_24h": timedelta(hours=30),
                "stale_48h": timedelta(hours=54),
            }.get(control, timedelta(hours=6))
            assert decision - cutoff == expected_lag
            assert entry == decision + timedelta(minutes=10)
            assert exit_time == entry + timedelta(hours=24)
            assert int(row["side"]) in {-1, 1}
            assert entry >= previous_exit.get(control, entry)
            previous_exit[control] = exit_time
            start, end = map(_time, support.WINDOWS[row["window"]])
            assert start <= entry and exit_time <= end
            exact_clocks[control].append((row["entry_time"], row["exit_time"]))

    assert rows == report["clock"]["rows"] == 2643
    assert dict(sorted(counts.items())) == report["clock"]["control_counts"]
    assert exact_clocks["direction_flip"] == exact_clocks["primary"]
    assert exact_clocks["deterministic_random_side"] == exact_clocks["primary"]


def test_rejection_matches_frozen_reproduction_and_structure_failures() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    support_stats = report["primary_support"]
    assert support_stats["total_trades"] == 167
    assert support_stats["all_side_counts"] == {"long": 132, "short": 35}
    assert support_stats["all_year_counts"] == {
        "2021": 123,
        "2022": 29,
        "2023": 15,
    }
    assert support_stats["selection"]["side_counts"] == {
        "long": 4,
        "short": 11,
    }
    failed = {name for name, passed in report["support_checks"].items() if not passed}
    assert failed == {
        "disclosed_selection_sides_reproduced",
        "disclosed_sides_reproduced",
        "disclosed_total_reproduced",
        "disclosed_years_reproduced",
        "maximum_consecutive_same_side",
        "maximum_month_share",
        "maximum_quarter_share",
        "selection_each_side_minimum",
        "selection_total_minimum",
    }
    assert report["control_report"]["no_black_funds_veto"]["trades"] == 240
