from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.search_causal_online_expert_alpha import _metric, _run_selector, _score


def _event(expert: str, signal: int, exit_pos: int, realized: float, adverse: float = 0.0) -> dict:
    return {
        "expert": expert,
        "side": "long" if expert.startswith("long_") else "short",
        "signal_pos": signal,
        "entry_pos": signal + 1,
        "exit_pos": exit_pos,
        "ret": np.asarray([realized], dtype=float),
        "adv": np.asarray([-adverse], dtype=float),
        "realized_return": realized,
        "max_adverse": adverse,
    }


def test_score_penalizes_adverse_excursion() -> None:
    history = [(0.02, 0.01), (0.02, 0.03)]
    raw = _score(history, lookback=10, method="normalized_mean", mae_penalty=0.0)
    penalized = _score(history, lookback=10, method="adverse_utility", mae_penalty=1.0)
    assert raw == 1.0
    assert penalized == pytest.approx(0.0)


def test_selector_does_not_learn_event_before_exit(monkeypatch) -> None:
    import training.search_causal_online_expert_alpha as module

    monkeypatch.setattr(
        module,
        "ALPHAS",
        {
            "long_a": {"side": "long"},
            "short_b": {"side": "short"},
        },
    )
    events = [
        _event("long_a", 0, 10, 0.10),
        _event("short_b", 0, 2, 0.01),
        _event("long_a", 5, 6, -0.10),
        _event("short_b", 5, 6, 0.01),
        _event("long_a", 11, 12, 0.01),
        _event("short_b", 11, 12, 0.01),
    ]
    spec = {"lookback": 10, "method": "normalized_mean", "mae_penalty": 0.0, "top_k": 1, "threshold": 0.0}
    accepted = _run_selector(events, spec, min_history=1)
    # At t=5 only short_b's t=0 event has matured. long_a's +10% must not leak.
    assert [(row["signal_pos"], row["expert"]) for row in accepted] == [(5, "short_b"), (11, "short_b")]


def test_metric_annualizes_full_idle_window_and_counts_adverse_mdd() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", "2024-12-31 23:55", freq="5min"))
    event = _event("long_a", 100, 101, 0.10, adverse=0.20)
    stats = _metric([event], dates, "2024-01-01", "2025-01-01")
    assert 9.9 < stats["return_pct"] < 10.1
    assert 9.9 < stats["cagr_pct"] < 10.2
    assert stats["strict_mdd_pct"] == pytest.approx(20.0)
    assert stats["trades"] == 1
