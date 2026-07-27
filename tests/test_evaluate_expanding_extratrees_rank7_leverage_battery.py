from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import (
    evaluate_expanding_extratrees_rank7_leverage_battery as battery,
)
from training.search_inventory_purge_reclaim_alpha import (
    Config,
    ExecutionEngine,
    _schedule_hash,
)


def _metrics(
    *,
    absolute: float = 10.0,
    cagr: float = 10.0,
    mdd: float = 3.0,
    ratio: float = 3.5,
    trades: int = 30,
) -> dict[str, float | int]:
    return {
        "absolute_return_pct": absolute,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
        "longs": trades,
        "shorts": 0,
        "mean_net_bps": 10.0,
        "mean_gross_bps": 20.0,
        "win_rate": 0.6,
    }


def _selection_cell(
    leverage: float,
    *,
    passes: bool = True,
) -> dict[str, object]:
    base = {window: _metrics() for window in battery.prereg.SELECTION_WINDOWS}
    stress = {window: _metrics() for window in battery.prereg.SELECTION_WINDOWS}
    if not passes:
        base["2024"] = _metrics(ratio=2.99)
    cell: dict[str, object] = {
        "leverage": leverage,
        "base": base,
        "stress": stress,
        "schedule_hashes": {},
    }
    passed, reasons = battery.selection_cell_passes(cell)
    cell["passes"] = passed
    cell["failure_reasons"] = reasons
    return cell


def test_selection_is_highest_passing_fixed_cell_and_cannot_see_future() -> None:
    cells = [
        _selection_cell(leverage, passes=leverage <= 1.25)
        for leverage in battery.prereg.LEVERAGE_GRID
    ]
    assert battery.select_leverage(cells) == 1.25

    contaminated = _selection_cell(0.5)
    contaminated["base"]["future"] = _metrics()  # type: ignore[index]
    with pytest.raises(ValueError, match="non-selection"):
        battery.selection_cell_passes(contaminated)


def test_selection_fails_closed_on_nonfinite_and_stress() -> None:
    cell = _selection_cell(0.5)
    cell["base"]["2023"]["cagr_pct"] = np.nan  # type: ignore[index]
    with pytest.raises(RuntimeError, match="nonfinite"):
        battery.selection_cell_passes(cell)

    cell = _selection_cell(0.5)
    cell["stress"]["selection"] = _metrics(absolute=0.0)  # type: ignore[index]
    passed, reasons = battery.selection_cell_passes(cell)
    assert not passed
    assert "selection:stress_nonpositive" in reasons


def _significance(p: float = 0.01) -> dict[str, object]:
    return {
        "weekly_cluster_sign_flip": {"p_value_one_sided": p},
        "stationary_trade_bootstrap": {"one_sided_p_value": p},
    }


def test_report_gates_and_user_target_are_independent() -> None:
    base = {
        window: _metrics(cagr=55.0, mdd=12.0, ratio=4.5)
        for window in battery.prereg.REPORT_ONLY_WINDOWS
    }
    base["2026h1"] = _metrics(
        cagr=55.0,
        mdd=12.0,
        ratio=4.5,
        trades=6,
    )
    base["future"] = _metrics(
        cagr=55.0,
        mdd=12.0,
        ratio=4.5,
        trades=18,
    )
    base["all"] = _metrics(
        cagr=55.0,
        mdd=12.0,
        ratio=4.5,
        trades=42,
    )
    stress = {
        window: _metrics(cagr=40.0, mdd=14.0, ratio=2.8)
        for window in battery.prereg.REPORT_ONLY_WINDOWS
    }
    significance = {"future": _significance(), "all": _significance()}

    assert battery.report_only_gates(base, stress, significance) == (True, [])
    assert battery.user_target_hit(base) == (True, [])

    base["future"] = _metrics(cagr=49.99, mdd=10.0, ratio=4.9)
    assert battery.report_only_gates(base, stress, significance) == (True, [])
    target, reasons = battery.user_target_hit(base)
    assert not target
    assert reasons == ["future:cagr_lt_50"]


def test_schedule_clock_is_invariant_to_account_leverage() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 101.0, 102.0, 102.0, 101.0, 100.0, 100.0],
            "high": [101.0, 101.0, 102.0, 103.0, 103.0, 102.0, 101.0, 101.0],
            "low": [99.0, 99.0, 100.0, 101.0, 101.0, 100.0, 99.0, 99.0],
            "close": [100.0, 101.0, 102.0, 102.0, 101.0, 100.0, 100.0, 100.0],
        }
    )
    funding = pd.DataFrame(
        {
            "date": [dates[2], dates[5]],
            "funding_rate": [0.001, 0.001],
        }
    )
    base_cfg = Config(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
    )
    trades = []
    for leverage in (0.5, 1.5):
        engine = ExecutionEngine(
            market,
            funding,
            replace(base_cfg, leverage=leverage),
        )
        trade = engine.trade_at(0, 1, 4, 1_000_000, 1_000_000)
        assert trade is not None
        trades.append(trade)
    assert _schedule_hash([trades[0]]) == _schedule_hash([trades[1]])
    assert trades[0].price_factor != trades[1].price_factor
    assert trades[0].funding_factor != trades[1].funding_factor


def test_preregistration_dependency_is_exact() -> None:
    payload = battery.validate_preregistration()
    assert payload["manifest_hash"] == (
        "01fa88ba5e1398c06ea192749c81a15e516982688761c570f160d6e416a16659"
    )


def test_preregistration_dependency_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sha256_file = battery.sha256_file

    def drift_one_dependency(path: str) -> str:
        if str(path).endswith("training/search_inventory_purge_reclaim_alpha.py"):
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(battery, "sha256_file", drift_one_dependency)
    with pytest.raises(RuntimeError, match="frozen dependency drifted"):
        battery.validate_preregistration()


def test_execution_state_requires_committed_untracked_clean_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git_output(*args: str) -> str:
        calls.append(args)
        if args[0] == "status":
            return ""
        return "abc123"

    monkeypatch.setattr(battery, "_git_output", fake_git_output)
    monkeypatch.setattr(battery, "sha256_file", lambda _path: "runner-hash")
    state = battery.validate_execution_state()

    assert calls[0] == ("status", "--porcelain", "--untracked-files=all")
    assert state == {
        "git_head": "abc123",
        "origin_main": "abc123",
        "runner_sha256": "runner-hash",
    }

    monkeypatch.setattr(
        battery,
        "_git_output",
        lambda *args: "?? runner.py" if args[0] == "status" else "abc123",
    )
    with pytest.raises(RuntimeError, match="clean"):
        battery.validate_execution_state()
