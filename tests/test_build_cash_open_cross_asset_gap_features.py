from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cash_open_cross_asset_gap_features as source


def _market(
    *,
    start: str = "2022-01-03",
    periods: int = 220,
    phase: float = 0.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    position = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(
        0.001 * position + 0.02 * np.sin(position / 9.0 + phase)
    )
    open_price = close * (
        1.0 + 0.003 * np.sin(position / 5.0 + phase)
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.006,
            "low": np.minimum(open_price, close) * 0.994,
            "close": close,
            "volume": 1_000_000.0
            + 20_000.0 * np.cos(position / 7.0 + phase),
            "cash_dividend": 0.0,
            "split_ratio": 1.0,
            "open_valid": True,
            "history_valid": True,
        }
    )


def _markets(periods: int = 220) -> dict[str, pd.DataFrame]:
    return {
        "QQQ": _market(periods=periods),
        "GLD": _market(periods=periods, phase=0.7),
    }


def _payload(frame: pd.DataFrame) -> bytes:
    local = pd.DatetimeIndex(frame["date"]).tz_localize(
        source.NEW_YORK_TIMEZONE
    ) + pd.Timedelta(hours=9, minutes=30)
    timestamps = (
        local.tz_convert("UTC").view("i8") // 1_000_000_000
    ).tolist()
    quote = {
        name: frame[name].astype(object).where(
            pd.notna(frame[name]), None
        ).tolist()
        for name in ("open", "high", "low", "close", "volume")
    }
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "exchangeTimezoneName": (
                                source.NEW_YORK_TIMEZONE
                            )
                        },
                        "timestamp": timestamps,
                        "events": {},
                        "indicators": {"quote": [quote]},
                    }
                ],
            }
        }
    ).encode()


def _valid_row(
    frame: pd.DataFrame,
    session: pd.Timestamp,
) -> pd.Series:
    row = frame.loc[frame["session_date"] == session]
    assert len(row) == 1
    assert bool(row["feature_valid"].iloc[0])
    return row.iloc[0]


def test_safe_schema_contains_only_open_and_strictly_prior_features() -> None:
    frame = source.build_safe_feature_frame(
        _markets(),
        cutoff="2023-01-01",
    )
    assert tuple(frame.columns) == source.OUTPUT_COLUMNS
    assert len(source.FEATURE_COLUMNS) == 31
    assert not set(source.FORBIDDEN_OUTPUT_COLUMNS).intersection(
        frame.columns
    )
    assert not {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }.intersection(frame.columns)
    assert pd.to_datetime(frame["session_date"]).max() < pd.Timestamp(
        "2023-01-01"
    )


def test_current_high_low_close_and_volume_cannot_change_current_features() -> None:
    markets = _markets()
    session = markets["QQQ"]["date"].iloc[120]
    baseline = source.build_safe_feature_frame(
        markets, cutoff="2023-01-01"
    )
    changed = {name: frame.copy() for name, frame in markets.items()}
    for symbol in source.SYMBOLS:
        index = changed[symbol].index[
            changed[symbol]["date"] == session
        ][0]
        changed[symbol].loc[index, ["high", "low", "close", "volume"]] = [
            np.nan,
            np.nan,
            np.nan,
            -1.0,
        ]
    altered = source.build_safe_feature_frame(
        changed, cutoff="2023-01-01"
    )
    left = _valid_row(baseline, session)
    right = _valid_row(altered, session)
    assert np.array_equal(
        left.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
        right.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
    )


def test_raw_parser_current_close_cannot_change_current_open_features() -> None:
    raw_markets = _markets()
    session = raw_markets["QQQ"]["date"].iloc[-1]
    parsed = {
        symbol: source.parse_research_payload(
            _payload(raw_markets[symbol]), symbol
        )[0]
        for symbol in source.SYMBOLS
    }
    changed = {name: frame.copy() for name, frame in raw_markets.items()}
    changed["QQQ"].loc[
        changed["QQQ"].index[-1],
        ["high", "low", "close", "volume"],
    ] = [np.nan, np.nan, np.nan, -1.0]
    reparsed = {
        "QQQ": source.parse_research_payload(
            _payload(changed["QQQ"]), "QQQ"
        )[0],
        "GLD": parsed["GLD"].copy(),
    }
    baseline = source.build_safe_feature_frame(
        parsed, cutoff="2023-01-01"
    )
    altered = source.build_safe_feature_frame(
        reparsed, cutoff="2023-01-01"
    )
    left = _valid_row(baseline, session)
    right = _valid_row(altered, session)
    assert np.array_equal(
        left.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
        right.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
    )
    assert bool(right["feature_valid"])


@pytest.mark.parametrize(
    ("open_multiplier", "dividend", "split_ratio"),
    [(0.98, 2.0, 1.0), (1.0, 0.0, 2.0)],
)
def test_current_gap_uses_dividend_but_does_not_double_apply_split(
    open_multiplier: float,
    dividend: float,
    split_ratio: float,
) -> None:
    markets = _markets()
    position = 120
    session = markets["QQQ"]["date"].iloc[position]
    prior_close = float(markets["QQQ"]["close"].iloc[position - 1])
    markets["QQQ"].loc[position, "open"] = prior_close * open_multiplier
    markets["QQQ"].loc[position, "cash_dividend"] = dividend
    markets["QQQ"].loc[position, "split_ratio"] = split_ratio
    output = source.build_safe_feature_frame(
        markets, cutoff="2023-01-01"
    )
    expected = np.log(
        (
            prior_close * open_multiplier
            + dividend
        )
        / prior_close
    )
    assert float(
        _valid_row(output, session)["gap_open_qqq"]
    ) == pytest.approx(expected)


def test_current_open_changes_only_the_current_gap_feature_block() -> None:
    markets = _markets()
    session = markets["QQQ"]["date"].iloc[100]
    baseline = source.build_safe_feature_frame(
        markets, cutoff="2023-01-01"
    )
    changed = {name: frame.copy() for name, frame in markets.items()}
    index = changed["QQQ"].index[changed["QQQ"]["date"] == session][0]
    changed["QQQ"].loc[index, "open"] *= 1.01
    altered = source.build_safe_feature_frame(
        changed, cutoff="2023-01-01"
    )
    left = _valid_row(baseline, session)
    right = _valid_row(altered, session)
    changed_columns = {
        name
        for name in source.FEATURE_COLUMNS
        if not np.isclose(float(left[name]), float(right[name]))
    }
    assert changed_columns == {
        "gap_open_qqq",
        "gap_risk_rotation",
        "gap_joint_liquidity",
        "gap_abs_total",
    }


def test_future_suffix_is_truncated_before_feature_calculation() -> None:
    markets = _markets(periods=300)
    cutoff = pd.Timestamp("2022-10-01")
    baseline = source.build_safe_feature_frame(markets, cutoff=cutoff)
    changed = {name: frame.copy() for name, frame in markets.items()}
    for symbol in source.SYMBOLS:
        future = changed[symbol]["date"] >= cutoff
        changed[symbol].loc[
            future, ["open", "high", "low", "close", "volume"]
        ] = 1e100
    altered = source.build_safe_feature_frame(changed, cutoff=cutoff)
    assert baseline.equals(altered)


def test_future_uniform_price_adjustment_leaves_ratio_features_unchanged() -> None:
    markets = _markets()
    session = markets["QQQ"]["date"].iloc[120]
    baseline = source.build_safe_feature_frame(
        markets, cutoff="2023-01-01"
    )
    adjusted = {name: frame.copy() for name, frame in markets.items()}
    for symbol in source.SYMBOLS:
        prefix = adjusted[symbol]["date"] <= session
        adjusted[symbol].loc[
            prefix, ["open", "high", "low", "close"]
        ] *= 0.5
    transformed = source.build_safe_feature_frame(
        adjusted, cutoff="2023-01-01"
    )
    left = _valid_row(baseline, session)
    right = _valid_row(transformed, session)
    assert np.allclose(
        left.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
        right.loc[list(source.FEATURE_COLUMNS)].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    )


def test_robust_volume_z_uses_previous_observation_and_prior_reference() -> None:
    values = pd.Series(np.arange(62, dtype=float))
    observed = source._robust_previous_observation_z(values)
    reference = values.iloc[:60]
    expected = (
        values.iloc[60] - reference.quantile(0.50)
    ) / (
        (
            reference.quantile(0.75)
            - reference.quantile(0.25)
        )
        / 1.349
    )
    assert np.isclose(observed.iloc[61], expected)


def test_new_york_signal_and_entry_times_respect_dst() -> None:
    dates = pd.DatetimeIndex(["2024-01-08", "2024-07-08"])
    signal = source._safe_session_time(
        dates, source.FEATURE_AVAILABLE_LOCAL_TIME
    )
    entry = source._safe_session_time(dates, source.ENTRY_LOCAL_TIME)
    assert signal.tolist() == [
        pd.Timestamp("2024-01-08 14:35"),
        pd.Timestamp("2024-07-08 13:35"),
    ]
    assert entry.tolist() == [
        pd.Timestamp("2024-01-08 14:40"),
        pd.Timestamp("2024-07-08 13:40"),
    ]


def test_build_is_byte_deterministic_and_outcome_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets = _markets()

    def fake_load(
        path: str | Path,
        *,
        symbol: str,
        enforce_hash: bool,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        del path, enforce_hash
        return markets[symbol].copy(), {
            "path": f"/frozen/{symbol}.json",
            "sha256": symbol.lower().ljust(64, "0"),
            "bytes": 1,
            "provider": "fixture",
            "provider_url": "fixture",
            "exchange_timezone": source.NEW_YORK_TIMEZONE,
            "first_session": str(markets[symbol]["date"].iloc[0]),
            "last_session": str(markets[symbol]["date"].iloc[-1]),
            "open_valid_rows": len(markets[symbol]),
            "history_valid_rows": len(markets[symbol]),
            "dividend_events": 0,
            "split_events": 0,
            "current_open_depends_on_same_day_close": False,
            "adjusted_close_read": False,
        }

    monkeypatch.setattr(source, "_load_cache", fake_load)
    cfg = source.BuildConfig(
        qqq_cache="unused",
        gld_cache="unused",
        output=str(tmp_path / "features.csv.gz"),
        manifest=str(tmp_path / "manifest.json"),
        cutoff="2023-01-01",
        enforce_frozen_inputs=False,
    )
    first = source.build(cfg)
    first_output = Path(cfg.output).read_bytes()
    first_manifest = Path(cfg.manifest).read_bytes()
    second = source.build(cfg)
    assert first == second
    assert Path(cfg.output).read_bytes() == first_output
    assert Path(cfg.manifest).read_bytes() == first_manifest
    contract = first["source_contract"]
    assert contract["btc_or_portfolio_values_read"] is False
    assert contract["post_entry_outcomes_opened"] is False
    assert first["output"]["feature_count"] == 31


def test_frozen_cutoff_and_source_order_fail_closed() -> None:
    with pytest.raises(ValueError, match="universe/order"):
        source.build_safe_feature_frame(
            {"GLD": _market(), "QQQ": _market()},
            cutoff="2023-01-01",
        )
    with pytest.raises(ValueError, match="cutoff is frozen"):
        source.build(source.BuildConfig(cutoff="2026-01-01"))
