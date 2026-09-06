from pathlib import Path

from training import evaluate_high_volatility_sample_entropy_collapse_continuation_gross9_novelty as novelty


def test_evaluator_is_outcome_blind_and_hash_bound() -> None:
    assert novelty.POLICY == "HVSENC-8"
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    assert novelty.sha(novelty.CLOCK) == novelty.CLOCK_SHA
    source = Path(novelty.__file__).read_text()
    assert '"outcomes_opened": False' in source
    assert "bars_binance" not in source
    assert "funding_rates_binance" not in source


def test_limits_match_frozen_preregistration() -> None:
    assert novelty.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }
