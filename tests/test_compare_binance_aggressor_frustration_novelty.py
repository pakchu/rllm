from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from training import compare_binance_aggressor_frustration_novelty as novelty


def _clock(dates: list[str], sides: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"signal_date": dates, "side": sides})


def test_one_to_one_pairs_are_chronological_and_non_reusing() -> None:
    pairs = novelty.one_to_one_pairs(
        [0, 10, 20],
        [4, 7, 25],
        tolerance_ns=5,
    )
    assert pairs == [(0, 4), (10, 7), (20, 25)]


def test_time_overlap_is_conservative_and_side_is_diagnostic() -> None:
    bafr = _clock(["2023-01-01 00:00:00"], [1])
    prior = _clock(["2023-01-01 00:00:00"], [-1])
    result = novelty.compare_clock(
        bafr,
        prior,
        name="opposite-side-same-time",
        coverage_start="2023-01-01",
        coverage_end="2024-01-01",
    )
    assert result["time_matches"] == 1
    assert result["same_side_matches"] == 0
    assert result["time_jaccard"] == 1.0
    assert result["bafr_time_containment"] == 1.0
    assert result["prior_time_containment_diagnostic"] == 1.0
    assert result["passes"] is False


def test_comparison_uses_declared_common_coverage() -> None:
    bafr = _clock(
        ["2022-12-31 23:55:00", "2023-01-02 00:00:00"],
        [1, -1],
    )
    prior = _clock(["2023-01-02 00:05:00"], [-1])
    result = novelty.compare_clock(
        bafr,
        prior,
        name="coverage",
        coverage_start="2023-01-01",
        coverage_end="2024-01-01",
        tolerance_bars=1,
    )
    assert result["bafr_events"] == 1
    assert result["prior_events"] == 1
    assert result["time_matches"] == 1


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            _clock(
                ["2023-01-01 00:05:00", "2023-01-01 00:00:00"],
                [1, -1],
            ),
            "strictly increasing",
        ),
        (
            _clock(
                ["2023-01-01 00:00:00", "2023-01-01 00:00:00"],
                [1, -1],
            ),
            "duplicate signal timestamps",
        ),
    ],
)
def test_clock_normalization_fails_closed_on_order_and_timestamp_conflict(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        novelty.normalize_clock(frame)


def test_bafr_loader_requires_exact_identity_and_hashes(tmp_path: Path) -> None:
    clock_path = tmp_path / "clock.csv"
    _clock(["2023-01-01 00:00:00"], [1]).to_csv(clock_path, index=False)
    clock_hash = hashlib.sha256(clock_path.read_bytes()).hexdigest()
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(
            {
                "candidate": "BAFR-24F",
                "stage": "outcome_blind_support",
                "outcomes_opened": False,
                "passed": True,
                "next_stage": "outcome_blind_novelty_gate",
                "source": {
                    "market_columns_loaded": ["date"],
                    "price_or_outcome_columns_loaded": [],
                },
                "clock": {"sha256": clock_hash, "rows": 1},
            }
        )
    )
    support_hash = hashlib.sha256(support_path.read_bytes()).hexdigest()
    cfg = novelty.NoveltyConfig(
        bafr_support=str(support_path),
        bafr_clock=str(clock_path),
    )
    clock, metadata = novelty.load_bafr(
        cfg,
        expected_support_sha256=support_hash,
        expected_clock_sha256=clock_hash,
    )
    assert len(clock) == 1
    assert metadata["clock_file_sha256"] == clock_hash

    support = json.loads(support_path.read_text())
    support["source"]["market_columns_loaded"] = ["date", "close"]
    support_path.write_text(json.dumps(support))
    changed_hash = hashlib.sha256(support_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="non-timestamp"):
        novelty.load_bafr(
            cfg,
            expected_support_sha256=changed_hash,
            expected_clock_sha256=clock_hash,
        )


def test_prior_bundle_loader_accepts_only_hash_bound_clock_fields(tmp_path: Path) -> None:
    comparators: dict[str, dict[str, object]] = {}
    for index, name in enumerate(novelty.REQUIRED_COMPARATORS):
        clock = _clock([f"2023-01-{index + 1:02d} 00:00:00"], [1])
        comparators[name] = {
            "family": name,
            "coverage_start_inclusive": "2023-01-01 00:00:00",
            "coverage_end_exclusive": "2024-01-01 00:00:00",
            "clock_rows": 1,
            "clock_sha256": novelty.clock_hash(clock),
            "events": clock.to_dict("records"),
        }
    payload = {
        "stage": "prior_comparator_clock_freeze",
        "protocol": {
            "bafr_source_loaded": False,
            "bafr_clock_loaded": False,
            "bafr_support_loaded": False,
            "bafr_outcomes_opened": False,
            "post_entry_outcomes_computed": False,
            "output_fields": ["signal_date", "side"],
        },
        "sources": {},
        "comparators": comparators,
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload))
    bundle_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, metadata = novelty.load_prior_bundle(
        path, expected_sha256=bundle_hash
    )
    assert tuple(loaded) == novelty.REQUIRED_COMPARATORS
    assert metadata["sha256"] == bundle_hash

    payload["protocol"]["bafr_clock_loaded"] = True
    path.write_text(json.dumps(payload))
    changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="touched BAFR"):
        novelty.load_prior_bundle(path, expected_sha256=changed_hash)
