"""Metric helpers for backtesting/evaluation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from utils import log_returns, min_sharpe, sharpe_ratio_log


def max_drawdown_pct(values: Sequence[float]) -> float:
    """Return max drawdown in percentage (positive number)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(-drawdowns.min() * 100.0)


def summarize_metrics(
    equity: Sequence[float],
    underlying: Sequence[float],
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Summarize core metrics for strategy vs benchmark."""
    strategy_log = log_returns(equity)
    sharpe = sharpe_ratio_log(
        strategy_log, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )
    return {
        "cumulative_return_pct": float((equity[-1] / equity[0] - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "sharpe_ratio": float(sharpe),
        "min_sharpe": float(
            min_sharpe(
                equity=equity,
                underlying=underlying,
                rf=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        ),
    }

