import numpy as np

from training.search_funding_premium_independent_gate_alpha import (
    IndependentGateConfig,
    _apply_gate,
    _gate_mask,
    _selection_score,
)


def _stats(cagr: float = 5.0, ratio: float = 1.0, trades: int = 100) -> dict:
    return {
        "cagr_pct": cagr,
        "ratio": ratio,
        "trades": trades,
        "strict_mdd_pct": 5.0,
    }


def test_gate_modes_and_component_targeting():
    values = np.asarray([-2.0, -0.5, 0.5, 2.0])
    base = {"lower": -1.0, "upper": 1.0}
    assert _gate_mask(values, {**base, "gate_mode": "lower"}).tolist() == [True, False, False, False]
    assert _gate_mask(values, {**base, "gate_mode": "upper"}).tolist() == [False, False, False, True]
    assert _gate_mask(values, {**base, "gate_mode": "central"}).tolist() == [False, True, True, False]
    assert _gate_mask(values, {**base, "gate_mode": "outer"}).tolist() == [True, False, False, True]

    funding = np.asarray([True, True, False, False])
    premium = np.asarray([False, False, True, True])
    gate = np.asarray([True, False, False, True])
    assert _apply_gate(funding, premium, gate, "all").tolist() == [True, False, False, True]
    assert _apply_gate(funding, premium, gate, "funding").tolist() == [True, False, True, True]
    assert _apply_gate(funding, premium, gate, "premium").tolist() == [True, True, False, True]


def test_selection_score_requires_positive_fit_and_both_halves():
    cfg = IndependentGateConfig(input_csv="x", funding_csv="f", premium_csv="p", output="y", manifest_output="m")
    stable = {
        "fit_2020_2022": _stats(trades=100),
        "select_2023": _stats(ratio=2.0, trades=40),
        "select_2023_h1": _stats(ratio=1.0, trades=20),
        "select_2023_h2": _stats(ratio=1.2, trades=20),
    }
    assert _selection_score(stable, cfg) > -1e11
    bad = {key: dict(value) for key, value in stable.items()}
    bad["select_2023_h2"]["cagr_pct"] = -1.0
    assert _selection_score(bad, cfg) <= -1e11
