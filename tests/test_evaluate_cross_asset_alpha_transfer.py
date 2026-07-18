import json

import numpy as np
import pandas as pd

from training import evaluate_cross_asset_alpha_transfer as evaluator


def _frame(rows: int = 180) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + 0.08 * index + 2.0 * np.sin(index / 9.0)
    open_ = close * (1.0 + 0.001 * np.sin(index / 4.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2010-01-01", periods=rows),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000.0 + 1_000.0 * index,
        }
    )


def test_yahoo_parser_adjusts_ohlc_and_uses_exchange_date() -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1609770600, 1609857000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0],
                                "high": [103.0, 104.0],
                                "low": [99.0, 101.0],
                                "close": [102.0, 103.0],
                                "volume": [10.0, 11.0],
                            }
                        ],
                        "adjclose": [{"adjclose": [51.0, 103.0]}],
                    },
                }
            ],
        }
    }
    frame, meta = evaluator.parse_yahoo_payload(json.dumps(payload).encode(), "TEST")
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2021-01-04", "2021-01-05"]
    assert frame["open"].tolist() == [50.0, 102.0]
    assert frame["close"].tolist() == [51.0, 103.0]
    assert meta["rows"] == 2


def test_rex_features_do_not_change_when_only_future_rows_change() -> None:
    frame = _frame()
    base = evaluator.rex_signal_bank(frame)
    changed = frame.copy()
    changed.loc[150:, ["open", "high", "low", "close", "volume"]] *= 3.0
    future = evaluator.rex_signal_bank(changed)
    for policy in base:
        np.testing.assert_allclose(base[policy][0][:150], future[policy][0][:150], equal_nan=True)
        np.testing.assert_array_equal(base[policy][1][:150], future[policy][1][:150])


def test_barrier_features_do_not_change_when_only_future_rows_change() -> None:
    frame = _frame()
    score, side = evaluator.barrier_signal(frame)
    changed = frame.copy()
    changed.loc[150:, ["open", "high", "low", "close"]] *= 2.0
    changed_score, changed_side = evaluator.barrier_signal(changed)
    np.testing.assert_allclose(score[:150], changed_score[:150], equal_nan=True)
    np.testing.assert_array_equal(side[:150], changed_side[:150])


def test_build_trades_enters_next_open_and_skips_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "SPLITS",
        {"eval": ("2022-01-01", "2023-01-01")},
    )
    dates = pd.Series(pd.bdate_range("2022-01-03", periods=10))
    active = np.array([True, True, True, False, True, False, False, False, False, False])
    side = np.ones(10, dtype=np.int8)
    trades = evaluator.build_trades(active, side, dates, split="eval", hold_sessions=2)
    assert [(t.signal_index, t.entry_index, t.exit_index) for t in trades] == [(0, 1, 3), (4, 5, 7)]


def test_strict_mdd_uses_intrahold_extremes(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "SPLITS", {"eval": ("2022-01-01", "2023-01-01")})
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=5),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 120.0, 100.0, 100.0, 100.0],
            "low": [100.0, 80.0, 100.0, 100.0, 100.0],
            "close": [100.0] * 5,
            "volume": [1.0] * 5,
        }
    )
    stats = evaluator.simulate(
        frame,
        [evaluator.Trade(signal_index=0, entry_index=1, exit_index=2, side=1)],
        split="eval",
        cost_bps_per_side=0.0,
    )
    assert stats["absolute_return_pct"] == 0.0
    assert abs(stats["strict_mdd_pct"] - (1.0 - 80.0 / 120.0) * 100.0) < 1e-9


def test_transfer_gate_requires_all_frozen_checks() -> None:
    passing = {
        "windows": {
            "eval": {
                "base_5bp": {
                    "absolute_return_pct": 10.0,
                    "cagr_to_strict_mdd": 3.1,
                    "strict_mdd_pct": 10.0,
                    "trades": 20,
                    "positive_calendar_year_share": 0.6,
                },
                "stress_10bp": {"absolute_return_pct": 1.0},
                "direction_flip_5bp": {"cagr_to_strict_mdd": -1.0},
            }
        }
    }
    result = evaluator.transfer_gate({symbol: passing for symbol in evaluator.INSTRUMENTS})
    assert result["all_three_passed"] is True
    failing = json.loads(json.dumps(passing))
    failing["windows"]["eval"]["base_5bp"]["trades"] = 19
    matrix = {symbol: passing for symbol in evaluator.INSTRUMENTS}
    matrix["GLD"] = failing
    assert evaluator.transfer_gate(matrix)["all_three_passed"] is False
