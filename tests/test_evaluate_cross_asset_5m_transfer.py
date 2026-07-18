from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cross_asset_5m_transfer as evaluator


def _patch_splits(monkeypatch: pytest.MonkeyPatch, **splits: tuple[str, str]) -> None:
    current = dict(getattr(evaluator, "SPLITS", {}))
    current.update(splits)
    monkeypatch.setattr(evaluator, "SPLITS", current)


def _bars(rows: int = 12, *, start: str = "2025-07-01 09:30:00+00:00") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="5min")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1_000.0),
            "source_valid": np.full(rows, True),
        }
    )


def _trade_fields(trade: object) -> dict[str, int]:
    if hasattr(trade, "__dataclass_fields__"):
        row = asdict(trade)
    elif isinstance(trade, dict):
        row = trade
    else:
        row = {
            name: getattr(trade, name)
            for name in ("signal_index", "entry_index", "exit_index", "side")
        }
    return {key: int(row[key]) for key in ("signal_index", "entry_index", "exit_index", "side")}


def test_build_trades_enters_next_bar_and_skips_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2025-07-01 09:30:00+00:00", periods=22, freq="5min")
    _patch_splits(
        monkeypatch,
        eval=(dates[0].isoformat(), (dates[-1] + pd.Timedelta(minutes=5)).isoformat()),
    )
    active = np.zeros(len(dates), dtype=bool)
    active[[0, 1, 6, 18]] = True
    side = np.ones(len(dates), dtype=np.int8)
    side[6] = -1

    trades = evaluator.build_trades(active, side, pd.Series(dates), split="eval", hold_bars=4)

    assert [_trade_fields(trade) for trade in trades] == [
        {"signal_index": 0, "entry_index": 1, "exit_index": 5, "side": 1},
        {"signal_index": 6, "entry_index": 7, "exit_index": 11, "side": -1},
    ]


def test_build_trades_requires_exit_inside_split(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2025-07-01 09:30:00+00:00", periods=8, freq="5min")
    _patch_splits(monkeypatch, eval=(dates[0].isoformat(), dates[5].isoformat()))
    active = np.zeros(len(dates), dtype=bool)
    active[1] = True
    side = np.ones(len(dates), dtype=np.int8)

    trades = evaluator.build_trades(active, side, pd.Series(dates), split="eval", hold_bars=4)

    assert trades == []


def test_simulate_strict_mdd_includes_costs_and_held_high_low_but_not_exit_bar_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _bars(8)
    _patch_splits(
        monkeypatch,
        eval=(
            frame["timestamp_utc"].iloc[0].isoformat(),
            (frame["timestamp_utc"].iloc[-1] + pd.Timedelta(minutes=5)).isoformat(),
        ),
    )
    frame.loc[1, "high"] = 120.0
    frame.loc[1, "low"] = 90.0
    frame.loc[5, "open"] = 110.0
    frame.loc[5, "low"] = 1.0  # the scheduled exit bar's low occurs after exit-open liquidation.
    trade = evaluator.Trade(signal_index=0, entry_index=1, exit_index=5, side=1)

    metrics = evaluator.simulate(frame, [trade], split="eval", cost_bps_per_side=5.0)

    expected_exit_equity = (1.0 - 0.0005) * 1.10 * (1.0 - 0.0005)
    assert metrics["absolute_return_pct"] == pytest.approx((expected_exit_equity - 1.0) * 100.0)
    assert metrics["strict_mdd_pct"] == pytest.approx(25.0)


def test_simulate_strict_mdd_charges_entry_and_exit_costs_when_price_is_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _bars(8)
    _patch_splits(
        monkeypatch,
        eval=(
            frame["timestamp_utc"].iloc[0].isoformat(),
            (frame["timestamp_utc"].iloc[-1] + pd.Timedelta(minutes=5)).isoformat(),
        ),
    )
    trade = evaluator.Trade(signal_index=0, entry_index=1, exit_index=5, side=1)

    metrics = evaluator.simulate(frame, [trade], split="eval", cost_bps_per_side=5.0)

    expected_exit_equity = (1.0 - 0.0005) * (1.0 - 0.0005)
    assert metrics["absolute_return_pct"] == pytest.approx((expected_exit_equity - 1.0) * 100.0)
    assert metrics["strict_mdd_pct"] == pytest.approx((1.0 - expected_exit_equity) * 100.0)


def test_simulate_uses_full_wall_clock_split_for_cagr(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _bars(8, start="2025-01-02 09:30:00+00:00")
    _patch_splits(monkeypatch, eval=("2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"))
    frame.loc[5, "open"] = 110.0
    trade = evaluator.Trade(signal_index=0, entry_index=1, exit_index=5, side=1)

    metrics = evaluator.simulate(frame, [trade], split="eval", cost_bps_per_side=0.0)

    years = (
        pd.Timestamp("2026-01-01T00:00:00+00:00")
        - pd.Timestamp("2025-01-01T00:00:00+00:00")
    ).total_seconds() / (365.2425 * 86400.0)
    assert metrics["absolute_return_pct"] == pytest.approx(10.0)
    assert metrics["cagr_pct"] == pytest.approx((1.10 ** (1.0 / years) - 1.0) * 100.0)


def test_fit_threshold_uses_only_positive_train_rows_and_is_suffix_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dates = pd.date_range("2024-09-01", periods=104, freq="D")
    eval_dates = pd.date_range("2025-07-01", periods=3, freq="D")
    _patch_splits(
        monkeypatch,
        train=("2024-09-01", "2024-12-31"),
        eval=("2025-07-01", "2025-07-10"),
    )
    train_strength = np.r_[np.nan, -5.0, 0.0, np.arange(1.0, 102.0)]
    strength = np.r_[train_strength, [1_000.0, 2_000.0, 3_000.0]]
    dates = pd.Series(train_dates.append(eval_dates))

    threshold, support = evaluator.fit_threshold(strength, dates, quantile=0.5)
    extended_threshold, extended_support = evaluator.fit_threshold(
        np.r_[strength, [10_000.0, 20_000.0]],
        pd.Series(dates.tolist() + [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]),
        quantile=0.5,
    )

    assert (threshold, support) == (51.0, 101)
    assert (extended_threshold, extended_support) == (threshold, support)


def test_build_trades_is_prefix_invariant_to_rows_after_split_end(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2025-07-01 09:30:00+00:00", periods=9, freq="5min")
    end = dates[-1] + pd.Timedelta(minutes=5)
    _patch_splits(monkeypatch, eval=(dates[0].isoformat(), end.isoformat()))
    active = np.zeros(len(dates), dtype=bool)
    active[[0, 6]] = True
    side = np.ones(len(dates), dtype=np.int8)
    baseline = evaluator.build_trades(active, side, pd.Series(dates), split="eval", hold_bars=3)

    extended_dates = dates.append(pd.date_range(end, periods=4, freq="5min"))
    extended_active = np.r_[active, [True, True, True, True]]
    extended_side = np.r_[side, [-1, -1, -1, -1]].astype(np.int8)
    extended = evaluator.build_trades(
        extended_active,
        extended_side,
        pd.Series(extended_dates),
        split="eval",
        hold_bars=3,
    )

    assert [_trade_fields(trade) for trade in extended] == [_trade_fields(trade) for trade in baseline]


def test_weekly_cluster_signflip_is_seed_deterministic() -> None:
    returns = [0.01, 0.02, -0.005, 0.03, -0.01]
    entry_times = ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-19", "2026-01-20"]

    first = evaluator.weekly_cluster_signflip(returns, entry_times, permutations=512, seed=7)
    second = evaluator.weekly_cluster_signflip(returns, entry_times, permutations=512, seed=7)

    assert first == second
    assert first["cluster_count"] == 3


def test_result_hash_ignores_its_own_field_and_uses_canonical_json() -> None:
    payload = {
        "schema_version": 1,
        "research_id": "CAT-XA-5M-1",
        "nested": {"b": [2, 1], "a": "same"},
        "result_hash": "0" * 64,
    }
    expected_payload = dict(payload)
    expected_payload.pop("result_hash")
    expected = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    assert evaluator.canonical_result_hash(payload) == expected
    mutated = dict(payload, result_hash="f" * 64)
    assert evaluator.canonical_result_hash(mutated) == expected


def test_committed_five_minute_evaluation_is_self_hashed_and_rejected() -> None:
    path = Path("results/cross_asset_5m_transfer_evaluation_2026-07-19.json")
    report = json.loads(path.read_text())

    assert report["result_hash"] == evaluator.canonical_result_hash(report)
    assert report["result_hash"] == "55a4ffa0fd71d8467c2d70f5ef5f3006f4cc5a4427ea927870a294062ca4f895"
    assert report["decision"]["pass"] is False
    assert report["decision"]["transferred_policies"] == []
    assert all(row["all_assets_pass"] is False for row in report["policies"].values())
    assert {
        symbol: row["pass"] for symbol, row in report["prefix_invariance"].items()
    } == {"QQQ": True, "069500": True, "GLD": True}

    reclaim = report["policies"]["rex_htf_pullback_reclaim_5m"]["assets"]
    assert reclaim["QQQ"]["splits"]["eval"]["base"]["absolute_return_pct"] == pytest.approx(
        2.368489858036682
    )
    assert reclaim["069500"]["splits"]["eval"]["base"]["strict_mdd_pct"] == pytest.approx(
        25.442510471394677
    )
