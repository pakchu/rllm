from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import audit_expanding_extratrees_rank7_hardened_strict as audit
from training.search_inventory_purge_reclaim_alpha import Config, ExecutionEngine


def _market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(4, 100.0),
            "high": np.full(4, 101.0),
            "low": np.full(4, 99.0),
            "close": np.full(4, 100.0),
        }
    )


def test_harden_trade_drops_boundary_credit_and_keeps_debit() -> None:
    cfg = replace(
        Config(
            input_csv="", metrics_csv="", funding_csv="", output="", manifest_output=""
        ),
        leverage=0.5,
    )
    funding = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-01 00:05:00"),
                pd.Timestamp("2024-01-01 00:10:00"),
            ],
            "funding_rate": [0.01, -0.02],
        }
    )
    engine = ExecutionEngine(_market(), funding, cfg)
    trade = engine.trade_at(0, 1, 1, 1_000_000, 1_000_000)
    assert trade is not None
    hardened = audit.harden_trade(trade, engine)
    assert hardened.funding_factor == pytest.approx(1.0 - cfg.leverage * 0.01)
    assert hardened.funding_debit_factor == pytest.approx(1.0 - cfg.leverage * 0.01)


def _stats(*, ratio: float = 3.0, trades: int = 20) -> dict[str, float]:
    return {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": 10.0,
        "trades": trades,
    }


def _significance(p: float = 0.10) -> dict[str, object]:
    return {
        "weekly_cluster_sign_flip": {"p_value_one_sided": p},
        "stationary_trade_bootstrap": {"one_sided_p_value": p},
    }


def test_hardened_pass_requires_all_windows_stress_and_significance() -> None:
    stats = {
        "2023": _stats(),
        "2024": _stats(),
        "2025": _stats(),
        "2026h1": _stats(trades=6),
        "future": _stats(trades=18),
        "all": _stats(trades=42),
    }
    stress = {"future": _stats(), "all": _stats()}
    significance = {"future": _significance(), "all": _significance()}
    passed, reasons = audit.hardened_pass(stats, stress, significance)
    assert passed
    assert reasons == []

    passed, reasons = audit.hardened_pass(
        stats,
        {**stress, "future": {**stress["future"], "absolute_return_pct": 0.0}},
        significance,
    )
    assert not passed
    assert "future:10bp_stress_nonpositive" in reasons

    failed_significance = {
        **significance,
        "all": _significance(0.10001),
    }
    passed, reasons = audit.hardened_pass(stats, stress, failed_significance)
    assert not passed
    assert "all:weekly_cluster_p_gt_0_10" in reasons
    assert "all:bootstrap_p_gt_0_10" in reasons


def test_frozen_dependency_hashes_are_current() -> None:
    frozen, rank7 = audit.verify_static_dependencies()
    assert frozen["selected_positions_hash"] == audit.EXPECTED_SELECTED_POSITIONS_HASH
    assert rank7["rank_position"] == 7
    assert audit.TREES == 300
    assert tuple(audit.SEEDS) == (7, 71, 715, 2026, 71515)
