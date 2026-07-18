from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import preregister_cross_sectional_leadership_diffusion as cld


SUPPORT = Path("results/cross_sectional_leadership_diffusion_support_2026-07-18.json")
CLOCK = Path("results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json")


def test_support_artifact_is_outcome_blind_and_frozen() -> None:
    payload = json.loads(SUPPORT.read_text())
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert cld.canonical_hash(body) == payload["manifest_hash"]
    assert payload["all_support_gates_pass"] is True
    assert payload["protocol"]["evidence_boundary"]["post_entry_outcomes_opened"] is False
    assert payload["support_selection"]["post_entry_outcomes_used"] is False
    assert payload["frozen_sources"][
        "post_entry_return_funding_pnl_or_equity_loaded"
    ] is False
    assert payload["frozen_sources"]["post_entry_outcome_columns_computed"] == []
    assert payload["preregistration_source_sha256"] == cld.sha256(
        cld.PREREGISTRATION_SOURCE
    )


def test_selected_support_cell_matches_outcome_blind_rule() -> None:
    payload = json.loads(SUPPORT.read_text())
    selected = payload["support_selection"]["selected_cell"]
    assert {
        key: selected[key]
        for key in (
            "move_quantile",
            "prior_hhi_quantile",
            "maximum_hhi_ratio",
            "minimum_participation",
            "minimum_flow_alignment",
            "turnover_quantile",
            "leader_decline_quantile",
        )
    } == {
        "move_quantile": 0.6,
        "prior_hhi_quantile": 0.6,
        "maximum_hhi_ratio": 0.9,
        "minimum_participation": 5 / 6,
        "minimum_flow_alignment": 4 / 6,
        "turnover_quantile": 0.5,
        "leader_decline_quantile": 0.6,
    }
    assert selected["support"] == {
        "nonoverlap_total": 106,
        "by_quarter": {"q1": 15, "q2": 22, "q3": 29, "q4": 40},
        "h1": 37,
        "h2": 69,
        "longs": 41,
        "shorts": 65,
        "long_share": 41 / 106,
        "short_share": 65 / 106,
        "maximum_quarter_share": 40 / 106,
        "passes": True,
    }
    assert cld.select_support_cell(payload["support_selection"]["cells"]) == selected


def test_clock_is_causal_nonoverlapping_and_canonical() -> None:
    payload = json.loads(CLOCK.read_text())
    events = payload["events"]
    assert payload["post_entry_outcomes_opened"] is False
    assert payload["entry_or_later_ohlc_loaded"] is False
    assert payload["event_count"] == 106
    assert payload["event_clock_sha256"] == cld.canonical_hash(events)
    assert payload["event_clock_sha256"] == (
        "dcbed47f339ff8f602008ed4cdad482f2b9fcc73dc522ac3411014ca1420396e"
    )
    frame = pd.DataFrame(events)
    for column in ("signal_date", "feature_boundary", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    assert (frame["signal_date"] < frame["feature_boundary"]).all()
    assert (frame["feature_boundary"] < frame["entry_date"]).all()
    assert (frame["entry_date"] < frame["exit_date"]).all()
    assert frame["entry_position"].eq(frame["signal_position"] + 2).all()
    assert frame["exit_position"].eq(frame["entry_position"] + 72).all()
    assert frame["entry_date"].dt.quarter.eq(frame["exit_date"].dt.quarter).all()
    assert (
        frame["entry_date"].iloc[1:].reset_index(drop=True)
        >= frame["exit_date"].iloc[:-1].reset_index(drop=True)
    ).all()


def test_prior_clock_overlap_stays_below_preregistered_limits() -> None:
    payload = json.loads(SUPPORT.read_text())
    for metrics in payload["outcome_blind_independence"].values():
        assert metrics["jaccard"] <= cld.Config.maximum_prior_jaccard
        assert (
            metrics["new_clock_containment"]
            <= cld.Config.maximum_new_clock_containment
        )
