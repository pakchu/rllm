from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_bitmex_insurance_fund_absorption as ifar


def _small_cfg(**changes: object) -> ifar.Config:
    values: dict[str, object] = {
        "fund_lookback_days": 5,
        "minimum_prior_loss_days": 2,
        "price_lookback_days": 5,
        "minimum_prior_price_days": 3,
        "eligibility_start": "2020-01-01",
        "minimum_total": 1,
        "minimum_train_2020h2_2021": 0,
        "minimum_train_2020h2": 0,
        "minimum_train_2021": 0,
        "minimum_test_2022": 0,
        "minimum_each_test_half": 0,
        "minimum_each_eligible_quarter": 0,
        "minimum_side_share": 0.0,
        "maximum_quarter_share": 1.0,
    }
    values.update(changes)
    return replace(ifar.Config(), **values)


def _panel(rows: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = pd.date_range("2019-12-20 12:00", periods=rows, freq="1D")
    balance = 1_000_000 * np.cumprod(
        np.where(np.arange(rows) % 2, 0.99, 1.01)
    )
    price = 10_000 * np.cumprod(
        np.where(np.arange(rows) % 2, 0.98, 1.02)
    )
    insurance = pd.DataFrame(
        {
            "snapshot_time": snapshots,
            "wallet_balance_satoshi": balance,
        }
    )
    prices = pd.DataFrame(
        {"snapshot_time": snapshots, "snapshot_price": price}
    )
    return insurance, prices


def test_market_reader_parses_only_completed_daily_snapshot_close(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv"
    source.write_text(
        "date,open,high,low,close,quote_asset_volume\n"
        "2020-01-01 11:50:00,malformed\n"
        "2020-01-01 11:55:00,100,101,99,100.5,10\n"
        "2020-01-01 12:00:00,malformed\n"
    )
    frame = ifar.read_daily_snapshot_prices(source)
    assert frame.to_dict(orient="records") == [
        {
            "snapshot_time": pd.Timestamp("2020-01-01 12:00"),
            "snapshot_price": 100.5,
        }
    ]
    assert frame.attrs["selected_non_date_rows_parsed"] == 1
    assert frame.attrs["outside_snapshot_non_date_rows_parsed"] == 0


def test_prior_positive_quantile_excludes_zero_current_and_future() -> None:
    values = pd.Series([0.0, 1.0, 3.0, 0.0, 5.0, 7.0, 100.0])
    first = ifar.prior_positive_quantile(
        values,
        lookback=5,
        minimum_positive=2,
        quantile=0.5,
    )
    assert np.isnan(first.loc[2])
    assert first.loc[3] == 2.0
    changed = values.copy()
    changed.loc[6] = 1_000_000.0
    second = ifar.prior_positive_quantile(
        changed,
        lookback=5,
        minimum_positive=2,
        quantile=0.5,
    )
    pd.testing.assert_series_equal(first.loc[:6], second.loc[:6])


def test_signal_uses_full_day_embargo_and_reverses_completed_move() -> None:
    insurance, prices = _panel()
    cfg = _small_cfg()
    panel = ifar.build_signal_panel(insurance, prices, cfg)
    candidates = panel.loc[panel["candidate"]]
    assert not candidates.empty
    assert (
        candidates["decision_time"]
        == candidates["snapshot_time"] + pd.Timedelta(days=1)
    ).all()
    assert (
        candidates["entry_time"]
        == candidates["snapshot_time"] + pd.Timedelta(days=1, minutes=5)
    ).all()
    assert (
        candidates["exit_time"]
        == candidates["entry_time"] + pd.Timedelta(days=1)
    ).all()
    assert (
        candidates["side"]
        == -np.sign(candidates["pre_snapshot_return"]).astype(np.int8)
    ).all()


def test_signal_requires_net_fund_loss_and_both_prior_thresholds() -> None:
    insurance, prices = _panel()
    cfg = _small_cfg()
    panel = ifar.build_signal_panel(insurance, prices, cfg)
    candidate_index = int(panel.index[panel["candidate"]][0])

    no_loss = insurance.copy()
    no_loss.loc[candidate_index, "wallet_balance_satoshi"] = (
        no_loss.loc[candidate_index - 1, "wallet_balance_satoshi"] * 1.01
    )
    changed = ifar.build_signal_panel(no_loss, prices, cfg)
    assert not bool(changed.loc[candidate_index, "candidate"])

    small_move = prices.copy()
    small_move.loc[candidate_index, "snapshot_price"] = (
        small_move.loc[candidate_index - 1, "snapshot_price"] * 1.00001
    )
    changed = ifar.build_signal_panel(insurance, small_move, cfg)
    assert not bool(changed.loc[candidate_index, "candidate"])


def test_signal_is_prefix_invariant() -> None:
    insurance, prices = _panel(50)
    cfg = _small_cfg()
    first = ifar.build_signal_panel(insurance, prices, cfg)
    changed_insurance = insurance.copy()
    changed_prices = prices.copy()
    changed_insurance.loc[40:, "wallet_balance_satoshi"] *= 100
    changed_prices.loc[40:, "snapshot_price"] *= 100
    second = ifar.build_signal_panel(changed_insurance, changed_prices, cfg)
    pd.testing.assert_frame_equal(first.loc[:39], second.loc[:39])


def test_support_summary_enforces_ten_quarters_and_side_balance() -> None:
    dates: list[pd.Timestamp] = []
    for quarter in pd.period_range("2020Q3", "2022Q4", freq="Q"):
        start = quarter.start_time + pd.Timedelta(days=5, hours=12, minutes=5)
        dates.extend(start + pd.to_timedelta(np.arange(6), unit="D"))
    schedule = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(dates),
            "side": np.where(np.arange(len(dates)) % 2, 1, -1),
        }
    )
    cfg = replace(
        ifar.Config(),
        minimum_train_2020h2=6,
        minimum_train_2021=20,
        minimum_test_2022=20,
        minimum_each_test_half=8,
    )
    summary = ifar.support_summary(schedule, cfg)
    assert summary["counts"] == {
        "total_2020h2_2022": 60,
        "train_2020h2_2021": 36,
        "train_2020h2": 12,
        "train_2021": 24,
        "test_2022": 24,
        "test_2022_h1": 12,
        "test_2022_h2": 12,
    }
    assert summary["passed"] is True
    missing_quarter = schedule[
        schedule["entry_time"].dt.to_period("Q").astype(str).ne("2021Q2")
    ].reset_index(drop=True)
    assert (
        ifar.support_summary(missing_quarter, cfg)["checks"][
            "each_eligible_quarter"
        ]
        is False
    )
    one_sided = schedule.copy()
    one_sided["side"] = 1
    assert ifar.support_summary(one_sided, cfg)["passed"] is False


def test_event_hash_binds_clock_side_config_protocol_and_source() -> None:
    events = [
        {
            "snapshot_time": "2022-01-01 12:00:00",
            "decision_time": "2022-01-02 12:00:00",
            "entry_time": "2022-01-02 12:05:00",
            "exit_time": "2022-01-03 12:05:00",
            "side": 1,
        }
    ]
    kwargs = {
        "cfg": ifar.Config(),
        "protocol_hash": "protocol",
        "source_manifest_hash": "manifest",
        "source_sha256": "source",
    }
    baseline = ifar.event_clock_hash(events, **kwargs)
    changed = [{**events[0], "side": -1}]
    assert ifar.event_clock_hash(changed, **kwargs) != baseline
    assert (
        ifar.event_clock_hash(
            events, **{**kwargs, "cfg": replace(ifar.Config(), hold_bars=289)}
        )
        != baseline
    )
    assert (
        ifar.event_clock_hash(
            events, **{**kwargs, "protocol_hash": "other"}
        )
        != baseline
    )
    assert (
        ifar.event_clock_hash(
            events, **{**kwargs, "source_sha256": "other"}
        )
        != baseline
    )


def test_protocol_and_default_configuration_are_frozen() -> None:
    cfg = ifar.Config()
    ifar._validate_config(cfg)
    payload = ifar.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["source"]["funding_loaded"] is False
    assert payload["source"]["post_decision_execution_or_outcome_bars_loaded"] is False
    assert payload["source"]["insurance_source_sha256"] == (
        "pending_outcome_blind_download"
    )
    assert payload["clock"]["decision"] == "snapshot timestamp + 1 full calendar day"
    assert payload["feature"]["threshold_grid"] is False
    assert payload["later_evaluation_contract"]["sealed_sequential"] == [
        "2023",
        "2024",
        "2025",
        "2026_ytd",
    ]
    with pytest.raises(ValueError, match="configuration is frozen"):
        ifar._validate_config(replace(cfg, hold_bars=289))
