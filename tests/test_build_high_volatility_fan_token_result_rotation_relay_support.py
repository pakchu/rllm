import numpy as np
import pandas as pd

from training import build_high_volatility_fan_token_result_rotation_relay_support as support


def event(event_id="1", tracked_score="2", other_score="1", tracked_winner=True, other_winner=False):
    return {"id": event_id, "date": "2023-08-01T19:00Z", "status": {"type": {"completed": True}}, "competitions": [{"competitors": [{"team": {"id": "83"}, "score": tracked_score, "winner": tracked_winner}, {"team": {"id": "86"}, "score": other_score, "winner": other_winner}]}]}


def test_parse_match_maps_win_and_loss_and_skips_draw():
    assert support.parse_match(event())["side"] == 1
    assert support.parse_match(event(tracked_score="0", other_score="1", tracked_winner=False, other_winner=True))["side"] == -1
    assert support.parse_match(event(tracked_score="1", other_score="1", tracked_winner=False, other_winner=False)) is None


def test_tracked_head_to_head_is_ineligible():
    value = event(); value["competitions"][0]["competitors"][1]["team"]["id"] = "94"
    assert support.parse_match(value) is None


def test_simultaneous_opposite_results_are_ineligible():
    win = event("1"); loss = event("2", "0", "1", False, True); loss["competitions"][0]["competitors"][0]["team"]["id"] = "94"
    groups = support.build_match_groups([{"document": {"events": [win, loss]}}])
    assert groups.empty


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(list(range(60)) + [100.0]); ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[59]); assert ranks.iloc[60] == 1.0


def test_primary_clock_uses_frozen_volatility_gate_and_hold():
    features = pd.DataFrame({"decision_time": pd.to_datetime(["2023-08-01T22:00Z", "2023-08-03T22:00Z"]), "result_side": [1, -1], "match_count": [1, 1], "tracked_team_count": [1, 1], "btc_realized_variation": [0.1, 0.2], "btc_variation_rank": [0.7, 0.8]})
    clock = support.build_clock(features)
    assert clock.side.tolist() == [1, -1]
    assert (pd.to_datetime(clock.exit_time, utc=True) - pd.to_datetime(clock.entry_time, utc=True)).eq(pd.Timedelta(hours=12)).all()
