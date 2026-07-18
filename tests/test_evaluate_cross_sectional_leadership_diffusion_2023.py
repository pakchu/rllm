from __future__ import annotations

import pandas as pd
import pytest

from training import evaluate_cross_sectional_leadership_diffusion_2023 as ev


def test_preoutcome_artifacts_verify_without_loading_execution_data() -> None:
    support, primary, controls = ev.verify_preoutcome_artifacts()
    assert support["all_support_gates_pass"] is True
    assert len(primary) == 106
    assert {name: len(frame) for name, frame in controls.items()} == ev.CONTROL_COUNTS


def test_primary_clock_is_causal_and_nonoverlapping() -> None:
    _, primary, _ = ev.verify_preoutcome_artifacts()
    checked = ev.validate_clock(primary, expected_count=106, primary=True)
    assert checked["entry_position"].eq(checked["signal_position"] + 2).all()
    assert checked["exit_position"].eq(checked["entry_position"] + 72).all()


def test_clock_transforms_are_fixed() -> None:
    frame = pd.DataFrame(
        {
            "signal_position": [0, 100],
            "entry_position": [2, 102],
            "exit_position": [74, 174],
            "signal_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "feature_boundary": pd.to_datetime(
                ["2023-01-01 00:05", "2023-01-02 00:05"]
            ),
            "entry_date": pd.to_datetime(
                ["2023-01-01 00:10", "2023-01-02 00:10"]
            ),
            "exit_date": pd.to_datetime(
                ["2023-01-01 06:10", "2023-01-02 06:10"]
            ),
            "side": [1, -1],
            "hold_bars": [72, 72],
            "quarter": ["q1", "q1"],
            "branch": ["x", "x"],
        }
    )
    assert ev.transform_clock(frame, "direction_flip")["side"].tolist() == [-1, 1]
    assert ev.transform_clock(frame, "long_only")["side"].tolist() == [1]
    assert ev.transform_clock(frame, "short_only")["side"].tolist() == [-1]
    delayed = ev.transform_clock(frame, "delay_five_minutes")
    assert delayed["entry_position"].tolist() == [3, 103]
    assert delayed["exit_position"].tolist() == [75, 175]


def _stats(*, ret: float = 10.0, ratio: float = 4.0, mdd: float = 2.5, trades: int = 100):
    return {
        "absolute_return_pct": ret,
        "cagr_pct": ret,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
    }


def test_selection_checks_bind_target_and_mechanism_controls() -> None:
    primary = {name: _stats() for name in ev.WINDOWS}
    controls = {name: _stats(ratio=1.0) for name in ev.CONTROL_COUNTS}
    checks = ev.selection_checks(
        primary,
        _stats(),
        _stats(),
        _stats(),
        _stats(),
        _stats(ret=-10.0, ratio=-1.0),
        controls,
        {"p_value_one_sided": 0.05},
    )
    assert all(checks.values())
    primary["2023"] = _stats(ratio=2.99)
    failed = ev.selection_checks(
        primary,
        _stats(),
        _stats(),
        _stats(),
        _stats(),
        _stats(ret=-10.0, ratio=-1.0),
        controls,
        {"p_value_one_sided": 0.05},
    )
    assert failed["annual_cagr_to_strict_mdd_at_least_3"] is False


def test_load_bundle_requires_evaluator_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "EVALUATION_FREEZE", ev.Path("/tmp/missing-cld-freeze.json"))
    with pytest.raises(RuntimeError, match="freeze is missing"):
        ev.load_bundle_2023()
