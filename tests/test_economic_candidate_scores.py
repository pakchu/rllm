from training.economic_candidate_scores import fit_action_score_prior, rerank_score_rows


def test_prior_rerank_can_remove_unconditional_action_bias():
    prior_rows = [
        {"candidates": [{"action_key": "A", "score": 10.0}, {"action_key": "B", "score": 1.0}]},
        {"candidates": [{"action_key": "A", "score": 10.0}, {"action_key": "B", "score": 1.0}]},
    ]
    prior = fit_action_score_prior(prior_rows)
    rows = [
        {
            "date": "d",
            "signal_pos": 1,
            "chosen": {"gate": "TRADE", "side": "SHORT", "hold_bars": 72},
            "candidates": [
                {"action_key": "A", "action": {"gate": "TRADE", "side": "LONG", "hold_bars": 72}, "score": 10.1},
                {"action_key": "B", "action": {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}, "score": 1.5},
            ],
        }
    ]
    preds = rerank_score_rows(rows, prior, prior_scale=1.0)
    assert preds[0]["prediction"]["side"] == "SHORT"
