import numpy as np
import pandas as pd
import pytest

from training.search_funding_premium_external_state_gate_alpha import _manifest_core_hash, _validate_manifest
from training.search_ordinal_complexity_bidirectional_alpha import (
    OrdinalComplexityConfig,
    _active_masks,
    _completed_hourly_close,
    _ordinal_pattern_codes,
    _ordinal_statistics,
    _select_top,
    _signal_specs,
)


def _cfg(tmp_path) -> OrdinalComplexityConfig:
    return OrdinalComplexityConfig(
        input_csv="market.csv",
        output=str(tmp_path / "out.json"),
        manifest_output=str(tmp_path / "manifest.json"),
        min_fit_observations=4,
    )


def test_completed_hourly_close_rejects_partial_hour_and_delays_exposure():
    dates = pd.date_range("2022-01-01", periods=30, freq="5min")
    market = pd.DataFrame({"date": dates, "close": np.arange(100.0, 130.0)})

    hourly = _completed_hourly_close(market)

    assert hourly.loc[0, "effective_time"] == pd.Timestamp("2022-01-01 01:00:00")
    assert hourly.loc[0, "close"] == 111.0
    assert hourly.loc[1, "close"] == 123.0
    assert np.isnan(hourly.loc[2, "close"])


def test_ordinal_pattern_codes_preserve_direction_without_using_magnitude():
    increasing_code, increasing_direction = _ordinal_pattern_codes(np.array([1.0, 2.0, 3.0]), 3)
    decreasing_code, decreasing_direction = _ordinal_pattern_codes(np.array([3.0, 2.0, 1.0]), 3)

    assert increasing_code[-1] != decreasing_code[-1]
    assert increasing_direction[-1] == 1.0
    assert decreasing_direction[-1] == -1.0


def test_ordinal_statistics_exclude_current_pattern_from_history():
    codes = np.array([0, 0, 0, 0, 1], dtype=int)

    entropy, surprise, transition_surprise = _ordinal_statistics(codes, states=2, window=4)

    assert np.isnan(entropy[3])
    assert entropy[4] == pytest.approx(0.0)
    assert surprise[4] > 1.0, "the unseen current state must remain surprising"
    assert np.isfinite(transition_surprise[4])


def test_active_masks_reverse_only_the_direction_mapping():
    features = pd.DataFrame(
        {
            "oc_o3_w168_entropy": [0.1, 0.1, 0.9, np.nan],
            "oc_direction_3": [1.0, -1.0, 1.0, -1.0],
        }
    )
    base = {
        "feature": "oc_o3_w168_entropy",
        "op": "le",
        "threshold": 0.2,
        "order": 3,
        "direction_threshold": 0.5,
    }

    continuation = _active_masks(features, {**base, "direction_mode": "continuation"})
    reversal = _active_masks(features, {**base, "direction_mode": "reversal"})

    assert continuation[0].tolist() == [True, False, False, False]
    assert continuation[1].tolist() == [False, True, False, False]
    assert reversal[0].tolist() == continuation[1].tolist()
    assert reversal[1].tolist() == continuation[0].tolist()


def test_signal_family_is_bounded_to_64_masks(tmp_path):
    n = 20
    features = pd.DataFrame({
        f"oc_o{order}_w{window}_{suffix}": np.linspace(0.0, 1.0, n)
        for order in (3, 4)
        for window in (168, 720)
        for suffix in ("entropy", "pattern_surprise", "transition_surprise")
    })

    specs = _signal_specs(features, np.ones(n, dtype=bool), _cfg(tmp_path))

    assert len(specs) == 64
    assert {spec["direction_mode"] for spec in specs} == {"continuation", "reversal"}
    assert {spec["tail"] for spec in specs} == {0.2, 0.3}


def test_selection_is_deterministic_and_manifest_mutation_is_rejected():
    def row(rule, score, hold):
        return {
            "rule": rule,
            "selection_score": score,
            "selection_stats": {"select_2023": {"ratio": score, "return_pct": score}},
            "hold_bars": hold,
            "order": 3,
            "entropy_window_hours": 168,
        }

    rows = [row("low_entropy_continuation", 3.0, 144), row("low_entropy_continuation", 2.0, 288), row("low_entropy_continuation", 1.0, 576), row("high_entropy_reversal", 2.5, 144)]
    first = _select_top(rows, top_n=4, top_per_rule=2)
    second = _select_top(list(reversed(rows)), top_n=4, top_per_rule=2)
    assert first == second
    assert len(first) == 3

    core = {"protocol": {}, "selected": []}
    manifest = {"as_of": "now", "sha256": _manifest_core_hash(core), **core}
    _validate_manifest(manifest)
    manifest["selected"].append({"future": "leak"})
    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        _validate_manifest(manifest)
