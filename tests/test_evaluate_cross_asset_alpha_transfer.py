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
    assert frame["source_valid"].tolist() == [True, True]
    assert meta["rows"] == 2


def test_yahoo_parser_discards_long_null_prefix_and_quarantines_later_gap() -> None:
    timestamps = [1609770600 + 86400 * index for index in range(35)]
    values = [100.0] * 3 + [None] * 20 + [100.0] * 7 + [None] + [100.0] * 4
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "UTC"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": values,
                                "high": values,
                                "low": values,
                                "close": values,
                                "volume": values,
                            }
                        ],
                        "adjclose": [{"adjclose": values}],
                    },
                }
            ],
        }
    }
    frame, meta = evaluator.parse_yahoo_payload(json.dumps(payload).encode(), "GAPPED")
    assert meta["discarded_unusable_prefix_rows"] == 23
    assert len(frame) == 12
    assert frame["source_valid"].sum() == 11
    assert meta["invalid_rows_quarantined"] == 1


def test_yahoo_parser_rejects_duplicate_sessions() -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "UTC"},
                    "timestamp": [1609459200, 1609459200],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 100.0],
                                "high": [101.0, 101.0],
                                "low": [99.0, 99.0],
                                "close": [100.0, 100.0],
                                "volume": [10.0, 10.0],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.0, 100.0]}],
                    },
                }
            ],
        }
    }
    with np.testing.assert_raises_regex(RuntimeError, "duplicate sessions"):
        evaluator.parse_yahoo_payload(json.dumps(payload).encode(), "DUP")


def test_yahoo_parser_rejects_long_null_block_that_is_not_a_prefix() -> None:
    timestamps = [1609459200 + 86400 * index for index in range(60)]
    values = [100.0] * 30 + [None] * 20 + [100.0] * 10
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "UTC"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {"open": values, "high": values, "low": values, "close": values, "volume": values}
                        ],
                        "adjclose": [{"adjclose": values}],
                    },
                }
            ],
        }
    }
    with np.testing.assert_raises_regex(RuntimeError, "not an unusable listing prefix"):
        evaluator.parse_yahoo_payload(json.dumps(payload).encode(), "LATE_GAP")


def test_reference_calendar_inserts_explicit_invalid_rows() -> None:
    frame = _frame(3)
    missing = frame["date"].iloc[1]
    sparse = frame.drop(index=1).reset_index(drop=True)
    sparse["source_valid"] = True
    aligned, meta = evaluator.align_to_reference_calendar(sparse, pd.DatetimeIndex(frame["date"]))
    assert aligned["date"].tolist() == frame["date"].tolist()
    assert aligned["source_valid"].tolist() == [True, False, True]
    assert pd.isna(aligned.loc[1, "close"])
    assert meta["calendar_missing_rows_inserted"] == 1
    assert aligned.loc[1, "date"] == missing


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


def test_build_trades_skips_any_signal_whose_hold_crosses_a_source_gap(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "SPLITS", {"eval": ("2022-01-01", "2023-01-01")})
    dates = pd.Series(pd.bdate_range("2022-01-03", periods=8))
    active = np.array([True, False, False, False, True, False, False, False])
    side = np.ones(8, dtype=np.int8)
    source_valid = np.array([True, True, False, True, True, True, True, True])
    trades = evaluator.build_trades(
        active,
        side,
        dates,
        split="eval",
        hold_sessions=2,
        source_valid=source_valid,
    )
    assert [(t.signal_index, t.entry_index, t.exit_index) for t in trades] == [(4, 5, 7)]


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
        "raw_signal_counts": {"train": 40, "test": 20, "eval": 20},
        "windows": {
            "eval": {
                "base_5bp": {
                    "absolute_return_pct": 10.0,
                    "cagr_to_strict_mdd": 3.1,
                    "strict_mdd_pct": 10.0,
                    "trades": 20,
                    "years_with_trade": 3,
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


def test_source_metadata_does_not_depend_on_cache_or_download_mode(tmp_path, monkeypatch) -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "UTC"},
                    "timestamp": [1609459200, 1609545600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [101.0, 102.0],
                                "low": [99.0, 100.0],
                                "close": [100.0, 101.0],
                                "volume": [10.0, 11.0],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.0, 101.0]}],
                    },
                }
            ],
        }
    }
    raw = json.dumps(payload).encode()
    monkeypatch.setattr(evaluator, "download_payload", lambda *args, **kwargs: (raw, "download"))
    _, downloaded = evaluator.load_market("TEST", str(tmp_path))
    monkeypatch.setattr(evaluator, "download_payload", lambda *args, **kwargs: (raw, "cache"))
    _, cached = evaluator.load_market("TEST", str(tmp_path))
    assert downloaded == cached
    assert "load_mode" not in downloaded
