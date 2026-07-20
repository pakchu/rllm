from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RESULT = Path(
    "results/regional_fiat_cross_rate_stress_v2_support_2026-07-20.json"
)
CLOCK = Path(
    "results/regional_fiat_cross_rate_stress_v2_clocks_2026-07-20.csv"
)
EXPECTED_RESULT_SHA256 = (
    "0b18ef3ab0a8b057e8966dcd5f358ea30320069ea5ba4ffdda3519526e5b0986"
)
EXPECTED_CLOCK_SHA256 = (
    "180181a7f95308a6fe5bac3d829dbb49c8e1e6aae8e84e69e6558146bee32413"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rfxs2_support_rejection_is_frozen_and_outcome_blind() -> None:
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    assert _sha256(CLOCK) == EXPECTED_CLOCK_SHA256
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert report["candidate"] == "RFXS2-576"
    assert report["decision"] == "REJECT"
    assert report["all_source_support_gates_pass"] is False
    assert report["next_stage_authorized"] is None
    assert report["clock_sha256"] == EXPECTED_CLOCK_SHA256
    assert report["protocol"] == {
        "execution_ohlc_opened": False,
        "funding_opened": False,
        "future_return_opened": False,
        "outcomes_opened": False,
        "pnl_cagr_mdd_opened": False,
        "post_2023_source_opened": False,
        "signed_exposure_gate_uses_absolute_pearson_magnitude": True,
        "source_only": True,
    }


def test_rfxs2_only_fails_the_frozen_train_return_shadow_gate() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert report["train"]["all_gates_pass"] is True
    assert report["selection"]["all_gates_pass"] is True
    novelty = report["novelty"]
    assert novelty["all_gates_pass"] is False
    assert novelty["values"][
        "train_abs_spearman_common_z_vs_btc_return_z"
    ] == 0.517801780144972
    assert novelty["values"][
        "selection_abs_spearman_common_z_vs_btc_return_z"
    ] == 0.39003536324443583
    assert novelty["gates"] == {
        "fqpr_abs_signed_exposure_correlation_at_most_0_40": True,
        "fqpr_exact_entry_jaccard_at_most_0_20": True,
        "sddr_abs_signed_exposure_correlation_at_most_0_40": True,
        "sddr_exact_entry_jaccard_at_most_0_10": True,
        "selection_abs_spearman_at_most_0_50": True,
        "train_abs_spearman_at_most_0_50": False,
    }


def test_rfxs2_primary_clock_matches_frozen_support_counts() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    clocks = pd.read_csv(CLOCK)
    raw_primary = clocks.loc[clocks["clock_name"].eq("primary")].copy()
    primary = raw_primary.loc[raw_primary["reserved"]].copy()
    train = primary.loc[primary["accepted_split"].eq("train")]
    selection = primary.loc[primary["accepted_split"].eq("selection")]
    assert len(raw_primary) == report["controls"]["primary"]["raw_candidates"] == 200
    assert len(primary) == 181
    assert int(raw_primary["suppressed_by_overlap"].sum()) == 19
    assert len(train) == report["train"]["accepted_events"] == 106
    assert len(selection) == report["selection"]["accepted_events"] == 75
    assert int(train["side"].eq(1).sum()) == report["train"]["long_events"] == 45
    assert int(train["side"].eq(-1).sum()) == report["train"]["short_events"] == 61
    assert int(selection["side"].eq(1).sum()) == 36
    assert int(selection["side"].eq(-1).sum()) == 39

    ordered = primary.sort_values("entry_time")
    entries = pd.to_datetime(ordered["entry_time"], utc=True).reset_index(drop=True)
    exits = pd.to_datetime(ordered["exit_time"], utc=True).reset_index(drop=True)
    assert (entries.iloc[1:].to_numpy() >= exits.iloc[:-1].to_numpy()).all()
