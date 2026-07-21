from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


REPORT = Path(
    "results/cross_domain_liquidity_transmission_relay_support_2026-07-21.json"
)
CLOCK = Path(
    "results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz"
)
REPORT_SHA256 = "ae56177d73836f9d232842ef72d05f385b066f741371defdbc15e909a5775e93"
CLOCK_SHA256 = "aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1"
MANIFEST_HASH = "cacc812a248263d688766d6a366a96a9aeb8531375638685399ea57a7c5adcfb"
EVALUATOR_SHA256 = "30b2c85e406fdd7ef54fb97035390ccad58b97dbb23e4347271ce7037c7e3bdc"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(REPORT.read_text(encoding="utf-8")))


def test_cdltr_support_rejection_is_hash_locked_and_outcome_blind() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256
    assert hashlib.sha256(CLOCK.read_bytes()).hexdigest() == CLOCK_SHA256
    report = _report()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["manifest_hash"] == _canonical_hash(core)
    assert report["protocol_version"].endswith("_v3")
    assert report["evaluator_source"]["sha256"] == EVALUATOR_SHA256
    assert report["clock_artifact"]["sha256"] == CLOCK_SHA256
    assert report["source_only"] is True
    assert report["market_outcomes_opened"] is False
    assert report["performance_values_opened"] is False
    assert report["verdict"] == {
        "failed_stages": [
            "support",
            "control_support_calendar_and_containment",
            "novelty",
        ],
        "passed": False,
        "repair_allowed_under_candidate_identity": False,
        "status": "REJECT",
        "strict_economic_train_authorized": False,
    }
    boundary = report["outcome_boundary"]
    assert boundary["btc_market_rows_loaded"] == 0
    assert boundary["funding_rows_loaded"] == 0
    assert boundary["return_rows_loaded"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_rows_read"] == 0
    assert boundary["network_calls"] == 0
    assert boundary["subprocess_calls"] == 0


def test_cdltr_primary_clock_reproduces_frozen_support_counts() -> None:
    report = _report()
    clocks = pd.read_csv(CLOCK)
    assert list(clocks.columns) == report["clock_artifact"]["columns"]
    assert len(clocks) == report["clock_artifact"]["rows"] == 677
    primary = clocks.loc[clocks["clock"].eq("primary")].copy()
    assert len(primary) == report["clock_artifact"]["primary_rows"] == 74
    assert int(primary["side"].eq(1).sum()) == 63
    assert int(primary["side"].eq(-1).sum()) == 11
    support = report["support"]
    assert support["counts"] == {
        "2021": 17,
        "2021H1": 5,
        "2021H2": 12,
        "2022": 25,
        "2022H1": 10,
        "2022H2": 15,
        "2023H1": 17,
        "2023H2": 15,
        "selection": 32,
        "train": 42,
    }
    assert support["side_counts"] == {
        "selection": {"long": 30, "short": 2},
        "train": {"long": 33, "short": 9},
    }
    assert [name for name, passed in support["checks"].items() if not passed] == [
        "each_train_half_year_minimum",
        "each_train_year_minimum",
        "maximum_weekday_share",
        "selection_each_side_minimum",
        "train_each_side_minimum",
        "train_total_minimum",
    ]
    for _, group in clocks.groupby("clock", sort=False):
        ordered = group.sort_values("entry_time_utc")
        entries = pd.to_datetime(ordered["entry_time_utc"], utc=True).reset_index(
            drop=True
        )
        exits = pd.to_datetime(ordered["exit_time_utc"], utc=True).reset_index(
            drop=True
        )
        assert (exits - entries).eq(pd.Timedelta("72h")).all()
        assert (entries.iloc[1:].to_numpy() >= exits.iloc[:-1].to_numpy()).all()


def test_cdltr_controls_and_novelty_fail_exactly_as_frozen() -> None:
    report = _report()
    controls = report["control_support_calendar_and_containment"]["controls"]
    assert {name: stage["rows"] for name, stage in controls.items()} == {
        "deterministic_random_side": 74,
        "direction_flip": 74,
        "macro_only": 142,
        "network_only": 191,
        "one_network_report_delay": 74,
        "reverse_order": 48,
    }
    assert [name for name, stage in controls.items() if stage["passed"]] == [
        "network_only"
    ]
    assert all(
        all(stage["calendar_and_containment_checks"].values())
        for stage in controls.values()
    )

    comparators = report["novelty"]["comparators"]
    failed = {name for name, stage in comparators.items() if not stage["passed"]}
    assert failed == {
        "CVTR-1",
        "prior_microstructure:mfic_fast",
        "prior_microstructure:mfic_slow",
        "prior_microstructure:mfic_union",
    }
    near = {
        name: comparators[name]["decision_date_overlap"][
            "candidate_dates_within_one_utc_day_fraction"
        ]
        for name in failed
    }
    assert near == {
        "CVTR-1": 44 / 74,
        "prior_microstructure:mfic_fast": 57 / 74,
        "prior_microstructure:mfic_slow": 58 / 74,
        "prior_microstructure:mfic_union": 63 / 74,
    }
    assert all(
        stage["checks"]["decision_date_jaccard"] for stage in comparators.values()
    )
