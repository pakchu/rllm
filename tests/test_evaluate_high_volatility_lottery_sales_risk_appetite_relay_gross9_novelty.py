from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_lottery_sales_risk_appetite_relay_gross9_novelty as novelty


def test_hvlsra_novelty_evaluator_is_outcome_blind_and_frozen():
    source = Path(novelty.__file__).read_text()
    assert novelty.POLICY == "HVLSRA-24"
    assert novelty.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }
    assert '"outcomes_opened": False' in source
    assert "bars_binance" not in source
    assert "funding_rates_binance" not in source


def test_hvlsra_novelty_binds_predecessors():
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    assert novelty.sha(novelty.CLOCK) == novelty.CLOCK_SHA
    assert novelty.load(novelty.PREREG)["novelty_gates"]["occupied_5m_jaccard_max"] == 0.25


def test_pair_enforces_all_registered_limits(monkeypatch):
    monkeypatch.setattr(
        novelty.metric,
        "evaluate_pair",
        lambda _a, _b: {
            "metrics": {
                "exact_entry_jaccard": 0.1,
                "one_to_one_6h_max_matched_share": 0.35,
                "occupied_5m_bar_jaccard": 0.25,
                "absolute_signed_exposure_pearson": 0.35,
            }
        },
    )
    result = novelty.pair(pd.DataFrame(), pd.DataFrame())
    assert result["passed"]
    assert all(result["checks"].values())
