from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from training import build_btcdom_leverage_polarity_decomposition_support as support
from training import preregister_btcdom_leverage_polarity_decomposition as dlpd


CLOCK = Path("data/btcdom_leverage_polarity_decomposition_clocks_2022_2023.csv.gz")
RESULT = Path(
    "results/btcdom_leverage_polarity_decomposition_support_2026-07-20.json"
)
CLOCK_SHA256 = "b33990f1629465caa837aa1f6f74430054b7185b68ece47b8c7540f9c11bf0fb"
RESULT_SHA256 = "1107694d5ff304aabaabbb962e9aeeaa64075001e494a0432dba3261ceace4f6"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_support_artifacts_are_hash_bound_and_outcome_blind() -> None:
    assert _sha(CLOCK) == CLOCK_SHA256
    assert _sha(RESULT) == RESULT_SHA256
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["candidate"] == dlpd.POLICY_ID
    assert payload["builder_sha256"] == _sha(support.BUILDER)
    assert payload["clock_sha256"] == CLOCK_SHA256
    assert payload["preregistration_sha256"] == support.PREREGISTRATION_SHA256
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] == []
    assert payload["btc_execution_rows_loaded"] == 0
    assert payload["funding_rows_loaded"] == 0
    assert payload["post_2023_source_rows_loaded"] == 0
    assert payload["support_passed"] is True
    assert payload["support_failures"] == []
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == dlpd.canonical_hash(core)


def test_primary_support_and_novelty_match_frozen_verdict() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["support"]["2022"]["events"] == 237
    assert payload["support"]["2022"]["long"] == 122
    assert payload["support"]["2022"]["short"] == 115
    assert payload["support"]["2023"]["events"] == 184
    assert payload["support"]["2023"]["long"] == 122
    assert payload["support"]["2023"]["short"] == 62
    assert all(payload["support_checks"].values())
    assert all(metrics["exact_jaccard"] == 0.0 for metrics in payload["novelty"].values())
    assert max(
        metrics["max_bidirectional_near_share"]
        for metrics in payload["novelty"].values()
    ) == payload["novelty"]["PSR-30/6"]["max_bidirectional_near_share"]


def test_clock_schema_and_calendar_containment() -> None:
    frame = pd.read_csv(
        CLOCK,
        parse_dates=[
            "source_hour_start",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )
    assert frame.columns.tolist() == list(dlpd.EVENT_COLUMNS)
    assert len(frame) == 3_832
    assert set(frame["control"]) == set(dlpd.CONTROLS)
    assert set(frame["split"].astype(str)) == {"2022", "2023"}
    assert frame["side"].isin((-1, 1)).all()
    assert (frame["feature_available_time"] <= frame["entry_time"]).all()
    assert (
        frame["exit_time"] - frame["entry_time"] == pd.Timedelta(hours=12)
    ).all()
    for year in (2022, 2023):
        subset = frame[frame["split"].astype(str).eq(str(year))]
        assert subset["entry_time"].min() >= pd.Timestamp(f"{year}-01-01")
        assert subset["exit_time"].max() <= pd.Timestamp(f"{year + 1}-01-01")
