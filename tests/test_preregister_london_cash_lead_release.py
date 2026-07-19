from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_london_cash_lead_release as lclr


def _small_cfg(**changes: object) -> lclr.Config:
    values: dict[str, object] = {
        "lookback_windows": 8,
        "minimum_prior_windows": 4,
        "minimum_total": 1,
        "minimum_train_2020_2021": 0,
        "minimum_each_train_year": 0,
        "minimum_test_2022": 0,
        "minimum_each_test_half": 0,
        "minimum_side_share": 0.0,
        "maximum_quarter_share": 1.0,
    }
    values.update(changes)
    return replace(lclr.Config(), **values)


def _source_day(
    day: str,
    *,
    coinbase: bool,
    complete: bool = True,
) -> pd.DataFrame:
    local = pd.date_range(
        f"{day} 15:00", periods=12, freq="5min", tz="Europe/London"
    )
    utc = local.tz_convert("UTC").tz_localize(None)
    base = np.linspace(100.0, 101.1, 12)
    frame = pd.DataFrame(
        {
            "date": utc,
            "local_date": day,
            "local_slot": np.arange(12),
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.1,
        }
    )
    if coinbase:
        frame["volume"] = 10.0
        frame["source_complete"] = 1
        if not complete:
            frame.loc[5, "source_complete"] = 0
            frame.loc[5, ["open", "high", "low", "close", "volume"]] = np.nan
    else:
        frame["quote_asset_volume"] = 100_000.0
    return frame


def _signal_panel(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    decision = dates + pd.Timedelta(hours=16)
    panel = pd.DataFrame(
        {
            "window_date": dates,
            "decision_time": decision,
            "entry_time": decision + pd.Timedelta(minutes=5),
            "exit_time": decision + pd.Timedelta(hours=2, minutes=5),
            "source_complete": True,
            "cash_return": 0.002,
            "perp_return": 0.001,
            "relative_return": 0.001,
            "cash_efficiency": 0.50,
            "final_cash_return": 0.0002,
            "cash_quote_share": 0.10,
        }
    )
    return panel


def test_london_clock_is_dst_aware() -> None:
    winter = lclr._london_timestamp(pd.Timestamp("2022-01-10 15:00"))
    summer = lclr._london_timestamp(pd.Timestamp("2022-07-11 14:00"))
    assert (winter.hour, winter.minute) == (15, 0)
    assert (summer.hour, summer.minute) == (15, 0)
    assert winter.utcoffset() == pd.Timedelta(0)
    assert summer.utcoffset() == pd.Timedelta(hours=1)


def test_reader_parses_non_date_fields_only_inside_weekday_window(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "date,open,high,low,close,volume,source_complete\n"
        "2022-01-10 14:00:00,malformed\n"
        "2022-01-10 15:00:00,100,101,99,100.5,10,1\n"
        "2022-01-10 16:00:00,malformed\n"
        "2022-01-15 15:00:00,malformed\n"
    )
    frame = lclr.read_source_window(
        source,
        numeric_columns=("open", "high", "low", "close", "volume"),
        complete_column="source_complete",
    )
    assert len(frame) == 1
    assert frame.iloc[0]["local_date"] == "2022-01-10"
    assert frame.attrs["selected_non_date_rows_parsed"] == 1
    assert frame.attrs["outside_window_non_date_rows_parsed"] == 0


def test_window_panel_maps_winter_and_summer_to_correct_utc_decisions() -> None:
    coinbase = pd.concat(
        [
            _source_day("2022-01-10", coinbase=True),
            _source_day("2022-07-11", coinbase=True),
        ],
        ignore_index=True,
    )
    binance = pd.concat(
        [
            _source_day("2022-01-10", coinbase=False),
            _source_day("2022-07-11", coinbase=False),
        ],
        ignore_index=True,
    )
    panel = lclr.build_window_panel(coinbase, binance, _small_cfg())
    assert panel["decision_time"].tolist() == [
        pd.Timestamp("2022-01-10 16:00"),
        pd.Timestamp("2022-07-11 15:00"),
    ]
    assert panel["entry_time"].tolist() == [
        pd.Timestamp("2022-01-10 16:05"),
        pd.Timestamp("2022-07-11 15:05"),
    ]


def test_missing_coinbase_partition_invalidates_whole_window() -> None:
    coinbase = _source_day("2022-01-10", coinbase=True, complete=False)
    binance = _source_day("2022-01-10", coinbase=False)
    panel = lclr.build_window_panel(coinbase, binance, _small_cfg())
    assert panel.loc[0, "source_complete"] == np.bool_(False)
    assert np.isnan(panel.loc[0, "cash_return"])


@pytest.mark.parametrize("missing_source", ["coinbase", "binance"])
def test_physically_missing_partition_invalidates_only_its_window(
    missing_source: str,
) -> None:
    coinbase = pd.concat(
        [
            _source_day("2022-01-10", coinbase=True),
            _source_day("2022-01-11", coinbase=True),
        ],
        ignore_index=True,
    )
    binance = pd.concat(
        [
            _source_day("2022-01-10", coinbase=False),
            _source_day("2022-01-11", coinbase=False),
        ],
        ignore_index=True,
    )
    target = coinbase if missing_source == "coinbase" else binance
    missing = (target["local_date"] == "2022-01-10") & (
        target["local_slot"] == 5
    )
    target.drop(target.index[missing], inplace=True)
    panel = lclr.build_window_panel(coinbase, binance, _small_cfg())
    assert panel["source_complete"].tolist() == [False, True]
    assert np.isnan(panel.loc[0, "cash_return"])
    assert np.isfinite(panel.loc[1, "cash_return"])


def test_prior_quantile_excludes_current_and_future_rows() -> None:
    cfg = _small_cfg()
    values = pd.Series(np.arange(20, dtype=float))
    first = lclr.prior_quantile(values, quantile=0.5, cfg=cfg)
    changed = values.copy()
    changed.loc[15:] = 1_000_000.0
    second = lclr.prior_quantile(changed, quantile=0.5, cfg=cfg)
    pd.testing.assert_series_equal(first.loc[:15], second.loc[:15])
    assert first.loc[4] == np.median([0.0, 1.0, 2.0, 3.0])


def test_prior_quantile_uses_fixed_source_window_and_ignores_nan() -> None:
    cfg = _small_cfg(lookback_windows=4, minimum_prior_windows=3)
    values = pd.Series([0.0, np.nan, 2.0, 3.0, 4.0, 5.0])
    result = lclr.prior_quantile(values, quantile=0.5, cfg=cfg)
    assert result.loc[4] == 2.0
    assert result.loc[5] == 3.0


def test_signal_requires_cash_lead_and_two_optional_votes() -> None:
    panel = _signal_panel()
    row = 63
    panel.loc[row, ["cash_return", "perp_return", "relative_return"]] = [
        0.004,
        0.001,
        0.003,
    ]
    panel.loc[row, ["cash_efficiency", "cash_quote_share", "final_cash_return"]] = [
        0.8,
        0.2,
        0.001,
    ]
    cfg = _small_cfg(lookback_windows=70, minimum_prior_windows=63)
    signal = lclr.build_signal(panel, cfg)
    assert bool(signal.loc[row, "candidate"])
    assert signal.loc[row, "side"] == 1
    assert signal.loc[row, "optional_votes"] == 4

    no_lead = panel.copy()
    no_lead.loc[row, ["perp_return", "relative_return"]] = [0.005, -0.001]
    assert not bool(
        lclr.build_signal(no_lead, cfg).loc[row, "candidate"]
    )
    flat_perp = panel.copy()
    flat_perp.loc[row, ["perp_return", "relative_return"]] = [0.0, 0.004]
    assert not bool(lclr.build_signal(flat_perp, cfg).loc[row, "candidate"])


def test_signal_is_prefix_invariant() -> None:
    panel = _signal_panel(100)
    cfg = _small_cfg()
    first = lclr.build_signal(panel, cfg)
    changed = panel.copy()
    changed.loc[90:, ["relative_return", "cash_efficiency", "cash_quote_share"]] = 999.0
    second = lclr.build_signal(changed, cfg)
    pd.testing.assert_frame_equal(first.loc[:89], second.loc[:89])


def test_support_summary_enforces_period_distribution_and_side_balance() -> None:
    def spread(year: int, count: int) -> pd.DatetimeIndex:
        grid = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
        positions = np.linspace(0, len(grid) - 1, count, dtype=int)
        return grid[positions]

    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                spread(2020, 50),
                spread(2021, 60),
                spread(2022, 70),
            ]
        )
    ).sort_values()
    schedule = pd.DataFrame(
        {
            "window_date": dates,
            "side": np.where(np.arange(len(dates)) % 2, 1, -1),
        }
    )
    summary = lclr.support_summary(schedule, lclr.Config())
    assert summary["counts"] == {
        "total_2020_2022": 180,
        "train_2020_2021": 110,
        "train_2020": 50,
        "train_2021": 60,
        "test_2022": 70,
        "test_2022_h1": 35,
        "test_2022_h2": 35,
    }
    assert summary["passed"] is True

    one_sided = schedule.copy()
    one_sided["side"] = 1
    assert lclr.support_summary(one_sided, lclr.Config())["passed"] is False


def test_event_records_bind_side_clock_and_votes() -> None:
    schedule = pd.DataFrame(
        {
            "window_date": [pd.Timestamp("2022-01-10")],
            "decision_time": [pd.Timestamp("2022-01-10 16:00")],
            "entry_time": [pd.Timestamp("2022-01-10 16:05")],
            "exit_time": [pd.Timestamp("2022-01-10 18:05")],
            "side": [1],
            "optional_votes": [3],
            "displacement_vote": [True],
            "coherence_vote": [True],
            "participation_vote": [False],
            "backload_vote": [True],
        }
    )
    events = lclr.event_records(schedule)
    protocol_hash = lclr.canonical_hash({"frozen": True})
    first = lclr.event_clock_hash(
        events, cfg=lclr.Config(), protocol_hash=protocol_hash
    )
    changed = schedule.copy()
    changed.loc[0, "side"] = -1
    assert (
        lclr.event_clock_hash(
            lclr.event_records(changed),
            cfg=lclr.Config(),
            protocol_hash=protocol_hash,
        )
        != first
    )
    changed = schedule.copy()
    changed.loc[0, "backload_vote"] = False
    assert (
        lclr.event_clock_hash(
            lclr.event_records(changed),
            cfg=lclr.Config(),
            protocol_hash=protocol_hash,
        )
        != first
    )
    assert (
        lclr.event_clock_hash(
            events,
            cfg=replace(lclr.Config(), hold_bars=25),
            protocol_hash=protocol_hash,
        )
        != first
    )
    assert (
        lclr.event_clock_hash(
            events,
            cfg=lclr.Config(),
            protocol_hash="different",
        )
        != first
    )


def test_protocol_and_default_configuration_are_frozen() -> None:
    cfg = lclr.Config()
    lclr._validate_config(cfg)
    payload = lclr.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["source"]["funding_loaded"] is False
    assert payload["source"]["post_window_execution_or_outcome_bars_loaded"] is False
    assert payload["clock"]["timezone"] == "Europe/London"
    assert payload["later_evaluation_contract"]["sealed_sequential"] == [
        "2023",
        "2024",
        "2025",
        "2026_ytd",
    ]
    with pytest.raises(ValueError, match="configuration is frozen"):
        lclr._validate_config(replace(cfg, optional_votes_required=1))
