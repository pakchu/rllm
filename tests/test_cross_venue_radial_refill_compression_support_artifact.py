from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from training import qualify_cross_venue_radial_refill_compression as qualify


SUPPORT_PATH = Path(
    "results/cross_venue_radial_refill_compression_support_2026-07-17.json"
)
CLOCK_PATH = Path(
    "results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"
)


def _load() -> tuple[dict, dict]:
    return json.loads(SUPPORT_PATH.read_text()), json.loads(CLOCK_PATH.read_text())


def test_support_freeze_opens_no_outcome_and_passes_every_gate() -> None:
    support, clock = _load()
    assert support["protocol"]["outcomes_opened_for_crrc72"] is False
    assert support["protocol"]["price_funding_return_or_equity_loaded"] is False
    assert support["protocol"]["support_rejected"] is False
    assert support["all_support_gates_pass"] is True
    assert support["passes_outcome_blind_independence"] is True
    assert clock["outcomes_opened"] is False
    assert clock["price_funding_return_or_equity_loaded"] is False


def test_selected_support_clock_is_frozen_exactly() -> None:
    support, clock = _load()
    assert support["raw_candidates"] == {
        "long": 117,
        "short": 91,
        "conflicts_flattened": 0,
    }
    assert support["support"]["nonoverlap_total"] == 156
    assert support["support"]["by_quarter"] == {
        "q1": 32,
        "q2": 25,
        "q3": 47,
        "q4": 52,
    }
    assert support["support"]["h1"] == 57
    assert support["support"]["h2"] == 99
    assert support["support"]["longs"] == 91
    assert support["support"]["shorts"] == 65
    assert support["support_selection"]["selected_cell"] == {
        "q_add": 0.85,
        "q_withdraw": 0.75,
        "q_net": 0.55,
        "q_flicker": 0.85,
        "events": 156,
    }
    assert clock["event_count"] == 156
    assert clock["side_counts"] == {"long": 91, "short": 65}
    assert clock["quarter_counts"] == support["support"]["by_quarter"]


def test_event_clock_hash_entry_delay_hold_and_quarter_containment() -> None:
    support, clock = _load()
    canonical = [
        {
            "signal_position": int(row["signal_position"]),
            "entry_position": int(row["entry_position"]),
            "exit_position": int(row["exit_position"]),
            "side": int(row["side"]),
            "hold_bars": int(row["hold_bars"]),
        }
        for row in clock["events"]
    ]
    digest = qualify.canonical_hash(canonical)
    assert digest == clock["event_clock_sha256"]
    assert digest == support["event_clock_sha256"]
    for row in clock["events"]:
        assert row["entry_position"] == row["signal_position"] + 2
        assert row["exit_position"] == row["entry_position"] + 72
        assert row["hold_bars"] == 72
        signal_quarter = pd.Timestamp(row["signal_date"]).quarter
        assert pd.Timestamp(row["entry_date"]).quarter == signal_quarter
        assert pd.Timestamp(row["exit_date"]).quarter == signal_quarter


def test_prior_clocks_and_overlap_metrics_are_replayed_exactly() -> None:
    support, _ = _load()
    assert support["prior_clock_counts"] == {
        "pdf10": 591,
        "cclh": 167,
        "rlwc144": 0,
        "near_pressure": 238,
    }
    expected = {
        "pdf10": (3, 46, 0.018125153776757157),
        "cclh": (0, 7, 0.09313998884551032),
        "rlwc144": (0, 0, 0.0),
        "near_pressure": (1, 24, 0.13274738381586607),
    }
    for name, (exact, tolerant, position_jaccard) in expected.items():
        row = support["outcome_blind_independence"][name]
        assert row["exact_entry_matches"] == exact
        assert row["tolerant_matches"] == tolerant
        assert row["position_time_jaccard"] == position_jaccard
        assert row["passes_frozen_overlap_gates"] is True
    assert support["outcome_blind_independence"]["rlwc144"]["evidentiary"] is False


def test_frozen_artifact_and_source_hashes_match_files() -> None:
    support, _ = _load()
    frozen = support["frozen_artifacts"]
    for key in ("preregistration", "near_pressure_manifest"):
        path = Path(frozen[key]["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[key]["sha256"]
    qualifier_path = Path(frozen["qualifier_source"])
    assert hashlib.sha256(qualifier_path.read_bytes()).hexdigest() == frozen[
        "qualifier_source_sha256"
    ]
    assert support["source"]["shell_sha256"] == (
        "ead931ec8ce2bbd73c946b8660e16d7750ce73051e60ce4989467a7c5bc68342"
    )
    assert support["source"]["credibility_sha256"] == (
        "45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429"
    )
    assert support["source"]["post_2023_rows_loaded"] is False
    assert support["source"]["outcome_columns_loaded"] is False
