from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BUILDER = Path("training/freeze_chain_activity_comparator_clock.py")
CLOCK = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_2026-07-21.csv.gz"
)
MANIFEST = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_manifest_2026-07-21.json"
)
EXPECTED_HASHES = {
    BUILDER: "5eb852e956cdf51d544a2ac1c567639d014aa0f41d9d499b361dfcf21b0d81e7",
    CLOCK: "e50cc154e23950a381aa456180970140882083734128bd7f902257738633f320",
    MANIFEST: "899704a0e998d818fd09735ca90af3c82aecfce94a288eec2bbc77c0c3df8441",
}
EXPECTED_SCHEDULE_HASHES = {
    "fit_2021": "cf19964f7e1ee900871a0af75aa52b4fd34daf37a17f7436bfac3ac296595995",
    "fit_2022": "aa2266685e04a64cf5e3e8aa3a3edd01a210457b700fd3b9c4832fe74248b32c",
    "select_2023": "1012d9b9b2d8ec4e71a1180044ae3fff8a8576bf577f879348a52376fc115211",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_chain_comparator_clock_artifacts_are_hash_frozen() -> None:
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_chain_comparator_clock_matches_every_frozen_schedule() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == _canonical_hash(core)
    assert manifest["schedule_hashes"] == EXPECTED_SCHEDULE_HASHES
    assert manifest["clock"] == {
        "path": str(CLOCK),
        "sha256": EXPECTED_HASHES[CLOCK],
        "rows": 66,
        "counts": {"fit_2021": 21, "fit_2022": 24, "select_2023": 21},
        "columns": ["window", "decision_time", "entry_time", "exit_time", "side"],
    }


def test_chain_comparator_clock_contains_no_outcome_fields() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle, keep_default_na=False)
    assert tuple(frame.columns) == (
        "window",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    )
    assert len(frame) == 66
    assert bool(frame["side"].astype(int).isin((-1, 1)).all())
    assert not {
        "return",
        "pnl",
        "equity",
        "cagr",
        "mdd",
        "funding",
        "open",
        "high",
        "low",
        "close",
    }.intersection(frame.columns)


def test_chain_comparator_export_did_not_run_the_outcome_engine() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["outcome_boundary"] == {
        "equity_cagr_mdd_computed": False,
        "funding_rows_loaded": 0,
        "high_low_open_columns_loaded": 0,
        "post_2023_rows_loaded": 0,
        "signal_price_return_features_derived": [
            "price_ret_24h",
            "price_ret_72h",
        ],
        "trade_return_or_pnl_computed": False,
        "trade_simulator_called": False,
    }
