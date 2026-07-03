import json
from training.build_event_signal_preference_dpo import _pairs_for_signal, EventSignalPreferenceCfg


def _row(side, net, util):
    return {
        "date": "2025-01-01", "signal_pos": 1, "side": side,
        "candidate": {"hold_bars": 144},
        "reward": {"net_return_pct": net, "utility": util},
        "state_tokens": {"candidate_side": side.lower()},
        "feature_snapshot": {"trend_96": 0.1},
    }


def test_pairs_rank_best_action_against_no_trade_and_loser():
    cfg = EventSignalPreferenceCfg(train_candidates="", eval_candidates="", train_output="", eval_output="", summary_output="", max_pairs_per_signal=2)
    rows = [_row("LONG", 1.5, 1.0), _row("SHORT", -0.5, -0.5)]
    pairs = _pairs_for_signal(rows, cfg)
    assert pairs
    chosen = json.loads(pairs[0]["chosen"])
    assert chosen["gate"] == "TRADE"
    assert chosen["side"] == "LONG"
    assert any(json.loads(p["rejected"])["gate"] == "NO_TRADE" for p in pairs)
