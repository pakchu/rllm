import numpy as np
import pandas as pd

from training.search_crossvenue_microstructure_consensus_alpha import (
    CrossVenueConsensusConfig,
    LOCAL_FEATURES,
    MICRO_FEATURES,
    _executed_signal_dates,
    _rule_masks,
    _selection_score,
)


def _stats(cagr: float = 5.0, ratio: float = 1.0, trades: int = 100) -> dict:
    return {
        "cagr_pct": cagr,
        "ratio": ratio,
        "trades": trades,
        "longs": trades // 2,
        "shorts": trades // 2,
        "strict_mdd_pct": 5.0,
        "p_value_mean_return_approx": 0.05,
    }


def test_rule_masks_use_one_local_and_one_micro_feature_without_rex():
    assert all("rex" not in name.lower() for name in (*LOCAL_FEATURES, *MICRO_FEATURES))
    frame = pd.DataFrame({"local": [-2.0, 2.0, -2.0, 2.0], "micro": [-2.0, 2.0, 2.0, -2.0]})
    base = {
        "local_feature": "local",
        "micro_feature": "micro",
        "local_lower": -1.0,
        "local_upper": 1.0,
        "micro_lower": -1.0,
        "micro_upper": 1.0,
    }
    agreement_long, agreement_short = _rule_masks(frame, np.ones(4, dtype=bool), {**base, "relation": "agreement"})
    disagreement_long, disagreement_short = _rule_masks(
        frame, np.ones(4, dtype=bool), {**base, "relation": "disagreement"}
    )
    assert agreement_long.tolist() == [False, True, False, False]
    assert agreement_short.tolist() == [True, False, False, False]
    assert disagreement_long.tolist() == [False, False, False, True]
    assert disagreement_short.tolist() == [False, False, True, False]


def test_selection_score_rejects_negative_fit_or_unstable_half():
    cfg = CrossVenueConsensusConfig(input_csv="x", output="y", manifest_output="z")
    stable = {
        "fit_2020_2022": _stats(trades=100),
        "select_2023": _stats(ratio=2.0, trades=60),
        "select_2023_h1": _stats(ratio=1.0, trades=30),
        "select_2023_h2": _stats(ratio=1.5, trades=30),
    }
    assert _selection_score(stable, cfg) > -1e11
    negative_fit = {key: dict(value) for key, value in stable.items()}
    negative_fit["fit_2020_2022"]["cagr_pct"] = -1.0
    assert _selection_score(negative_fit, cfg) <= -1e11
    unstable_half = {key: dict(value) for key, value in stable.items()}
    unstable_half["select_2023_h2"]["cagr_pct"] = -0.1
    assert _selection_score(unstable_half, cfg) <= -1e11


def test_executed_signal_dates_are_nonoverlapping_and_split_contained():
    dates = pd.Series(pd.date_range("2024-01-01", "2024-12-31 23:55", freq="5min"))
    n = len(dates)
    market = pd.DataFrame({"open": np.full(n, 100.0), "high": np.full(n, 100.0), "low": np.full(n, 100.0)})
    long_active = np.zeros(n, dtype=bool)
    long_active[[0, 1, 4, n - 2]] = True
    executed = _executed_signal_dates(
        market,
        dates,
        long_active,
        np.zeros(n, dtype=bool),
        window="test_2024",
        hold_bars=2,
        stride_bars=1,
    )
    assert pd.Timestamp("2024-01-01 00:00:00") in executed
    assert pd.Timestamp("2024-01-01 00:05:00") not in executed
    assert pd.Timestamp("2024-01-01 00:20:00") in executed
    assert dates.iloc[-2] not in executed
