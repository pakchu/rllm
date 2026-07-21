# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportReturnType=false, reportCallIssue=false
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from training import cchr_comparator_clock_common as clocks
from training import export_cchr_dtv_pure_clocks as dtv


UTC = timezone.utc


def _market(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    idx = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "close": 20_000.0 + idx * 0.7 + 30.0 * np.sin(idx / 37.0),
            "quote_asset_volume": 1_000.0 + (idx % 17.0) * 3.0,
            "taker_buy_quote": 520.0 + (idx % 11.0) * 2.0,
            "open_interest": 100.0 + idx * 0.01,
            "open_interest_value": 2_000_000.0 + idx * 29.0,
            "spot_close": 19_950.0 + idx * 0.65 + 25.0 * np.cos(idx / 41.0),
            "spot_volume": 50.0 + (idx % 13.0),
            "spot_rows": 5,
            "sum_open_interest": 10_000.0 + idx * 1.3 + 20.0 * np.sin(idx / 53.0),
            "sum_open_interest_value": 210_000_000.0 + idx * 410.0,
            "count_long_short_ratio": 1.1 + 0.05 * np.sin(idx / 29.0),
            "positioning_source_time": dates - pd.Timedelta(minutes=5),
        }
    )


def _reference_prior_z(values: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    minimum = max(288, window // 2)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (numeric - mean) / std


def test_source_bindings_are_hash_bound_and_allowlisted() -> None:
    assert dtv.source_bindings() == {
        "market": {
            "path": dtv.MARKET_PATH,
            "sha256": "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
            "columns": (
                "date",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
                "open_interest",
                "open_interest_value",
            ),
        },
        "spot": {
            "path": dtv.SPOT_PATH,
            "sha256": "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617",
            "columns": ("date", "spot_close", "spot_volume", "spot_rows"),
        },
        "metrics": {
            "path": dtv.METRICS_PATH,
            "sha256": "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106",
            "columns": (
                "create_time",
                "sum_open_interest",
                "sum_open_interest_value",
                "count_long_short_ratio",
            ),
        },
    }


def test_candidate_map_is_exact_dtv_prereg_family_subset() -> None:
    members = dtv.comparator_candidate_map()
    expected_ids = tuple(
        sorted(
            f"dtv:memory={memory}:accept={accept}:q={q:.2f}:hold={hold}"
            for memory in (72, 288)
            for accept in (72, 288)
            for q in (0.90, 0.95)
            for hold in (72, 144, 288)
        )
    )
    assert tuple(members) == expected_ids
    assert len(members) == 24
    assert dtv.candidate_ids() == expected_ids
    assert members["dtv:memory=72:accept=288:q=0.95:hold=144"] == {
        "family": "dtv",
        "parameters": {"memory": 72, "acceptance_horizon": 288, "q": 0.95},
        "hold_bars": 144,
        "component_weight": None,
    }
    assert dtv.candidate_map_hash() == clocks.candidate_map_hash(members)


def test_attach_delayed_metrics_reproduces_one_complete_bar_delay() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=4, freq="5min", tz="UTC"),
            "close": [1, 2, 3, 4],
        }
    )
    metrics = pd.DataFrame(
        {
            "create_time": pd.date_range(
                "2021-01-01", periods=4, freq="5min", tz="UTC"
            ),
            "sum_open_interest": [10.0, 20.0, 30.0, 40.0],
            "sum_open_interest_value": [100.0, 200.0, 300.0, 400.0],
            "count_long_short_ratio": [1.1, 1.2, 1.3, 1.4],
        }
    )
    joined = dtv.attach_delayed_metrics(market, metrics, tolerance="5min", delay_bars=1)
    assert np.isnan(joined.loc[0, "sum_open_interest"])
    assert list(joined.loc[1:, "sum_open_interest"]) == [10.0, 20.0, 30.0]
    assert list(joined["positioning_source_time"].astype(str)) == [
        "NaT",
        "2021-01-01 00:00:00",
        "2021-01-01 00:05:00",
        "2021-01-01 00:10:00",
    ]

    stale_market = market.drop(index=2).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete 5-minute grid"):
        dtv.attach_delayed_metrics(stale_market, metrics)


def test_attach_delayed_metrics_enforces_bounded_post_shift_staleness() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=5, freq="5min", tz="UTC"),
            "close": np.arange(5, dtype=float),
        }
    )
    metrics = pd.DataFrame(
        {
            "create_time": [pd.Timestamp("2021-01-01T00:00:00Z")],
            "sum_open_interest": [10.0],
            "sum_open_interest_value": [100.0],
            "count_long_short_ratio": [1.1],
        }
    )
    joined = dtv.attach_delayed_metrics(market, metrics, tolerance="5min", delay_bars=1)
    available = joined["positioning_available"].eq(1.0)
    ages = (
        joined.loc[available, "date"] - joined.loc[available, "positioning_source_time"]
    )
    assert ages.max() == pd.Timedelta("10min")
    assert joined["positioning_available"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_build_features_primary_matches_reference_fragments() -> None:
    market = _market("2021-01-01", 2_700)
    features = dtv.build_features(market, memory=72, acceptance_horizon=72)

    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0))
    oi = np.log(pd.to_numeric(market["sum_open_interest"], errors="coerce"))
    oi_value = np.log(pd.to_numeric(market["sum_open_interest_value"], errors="coerce"))
    global_ratio = np.log(
        pd.to_numeric(market["count_long_short_ratio"], errors="coerce")
    )
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_flow = (
        ((2.0 * taker_buy / quote.replace(0.0, np.nan)) - 1.0)
        .rolling(12, min_periods=12)
        .mean()
    )
    owner = np.tanh(
        0.5 * _reference_prior_z(global_ratio, 288)
        + 0.5 * _reference_prior_z(taker_flow, 288)
    )
    owner_change = owner - owner.shift(12)
    new_debt = (oi - oi.shift(1)).clip(lower=0.0)
    transfer_velocity = (
        (new_debt * owner_change).ewm(halflife=72, min_periods=72, adjust=False).mean()
    )
    complete_spot = pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
    spot_notional_24h = (
        (
            pd.to_numeric(market["spot_close"], errors="coerce")
            * pd.to_numeric(market["spot_volume"], errors="coerce")
        )
        .where(complete_spot)
        .rolling(288, min_periods=276)
        .sum()
    )
    cash_gap = _reference_prior_z(
        oi_value
        - oi_value.shift(288)
        - np.log(spot_notional_24h / spot_notional_24h.shift(288)),
        2016,
    )
    acceptance_z = _reference_prior_z(
        np.sign(transfer_velocity) * (log_price - log_price.shift(72)), 288
    )
    transfer_intensity_z = _reference_prior_z(transfer_velocity.abs(), 2016)
    score = (
        transfer_intensity_z.clip(lower=0.0)
        * cash_gap.clip(lower=0.0)
        * acceptance_z.clip(lower=0.0)
    )

    for column, expected in {
        "owner": owner,
        "owner_change": owner_change,
        "new_debt": new_debt,
        "transfer_velocity": transfer_velocity,
        "cash_gap": cash_gap,
        "acceptance_z": acceptance_z,
        "transfer_intensity_z": transfer_intensity_z,
        "score": score,
    }.items():
        pd.testing.assert_series_equal(
            cast(pd.Series, features[column]),
            expected.rename(column),
            check_names=False,
        )
    with pytest.raises(ValueError, match="primary"):
        dtv.build_features(market, memory=72, acceptance_horizon=72, variant="no_cash")


def test_fit_threshold_uses_only_positive_pre_2023_fit_scores() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-10-14T23:55:00Z",
                "2020-10-15T00:00:00Z",
                "2021-01-01T00:00:00Z",
                "2022-12-31T23:55:00Z",
                "2023-01-01T00:00:00Z",
                "2023-06-01T00:00:00Z",
            ],
            utc=True,
        )
    )
    score = pd.Series([100.0, -5.0, 1.0, 3.0, 1_000.0, 2_000.0])
    assert dtv.fit_threshold(score, dates, 0.5, min_positive=1) == 2.0


def test_timing_hold_and_causal_origin_use_completed_signal_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = pd.DataFrame(
        {"date": pd.date_range("2023-02-01", periods=3, freq="5min", tz="UTC")}
    )
    features = pd.DataFrame(
        {"score": [0.0, 2.0, 0.0], "transfer_velocity": [0.0, -4.0, 0.0]}
    )
    monkeypatch.setattr(dtv, "build_features", lambda *_args, **_kwargs: features)
    monkeypatch.setattr(dtv, "fit_threshold", lambda *_args, **_kwargs: 1.0)

    [candidate] = dtv.raw_clock_candidates_for_member(
        market,
        memory=72,
        acceptance_horizon=288,
        quantile=0.95,
        hold=144,
        min_positive=1,
    )
    assert candidate.candidate_id == "dtv:memory=72:accept=288:q=0.95:hold=144"
    assert candidate.signal_time == datetime(2023, 2, 1, 0, 5, tzinfo=UTC)
    assert candidate.causal_origins == (candidate.signal_time,)
    assert candidate.decision_time == datetime(2023, 2, 1, 0, 10, tzinfo=UTC)
    assert candidate.entry_time == candidate.decision_time
    assert candidate.exit_time == datetime(2023, 2, 1, 12, 10, tzinfo=UTC)
    assert candidate.side == 1


def test_containment_then_independent_nonoverlap_per_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = datetime(2023, 2, 1, tzinfo=UTC)

    def cand(member: str, entry: datetime, hold: int) -> clocks.ClockCandidate:
        signal = entry - timedelta(minutes=5)
        return clocks.ClockCandidate(
            candidate_id=member,
            causal_origins=(signal,),
            signal_time=signal,
            decision_time=entry,
            entry_time=entry,
            exit_time=entry + timedelta(minutes=5 * hold),
            side=1,
        )

    raw = [
        cand("dtv:memory=72:accept=72:q=0.90:hold=72", base, 2),
        cand("dtv:memory=72:accept=72:q=0.90:hold=72", base + timedelta(minutes=5), 2),
        cand("dtv:memory=72:accept=72:q=0.90:hold=72", base + timedelta(minutes=10), 2),
        cand("dtv:memory=72:accept=72:q=0.90:hold=144", base + timedelta(minutes=5), 2),
        cand(
            "dtv:memory=72:accept=72:q=0.90:hold=144",
            datetime(2023, 12, 31, 23, 55, tzinfo=UTC),
            2,
        ),
    ]
    monkeypatch.setattr(dtv, "raw_clock_candidates", lambda *_args, **_kwargs: raw)
    frame = dtv.build_clock_frame(pd.DataFrame())
    assert list(frame["candidate_id"]) == [
        "dtv:memory=72:accept=72:q=0.90:hold=144",
        "dtv:memory=72:accept=72:q=0.90:hold=72",
        "dtv:memory=72:accept=72:q=0.90:hold=72",
    ]
    assert list(frame["entry_time"]) == [
        "2023-02-01T00:05:00Z",
        "2023-02-01T00:00:00Z",
        "2023-02-01T00:10:00Z",
    ]


def test_module_has_no_legacy_search_or_evaluation_imports() -> None:
    source = Path(dtv.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    forbidden = [
        module
        for module in imported_modules
        if module.startswith("training.search_")
        or module.startswith("training.evaluate_")
    ]
    assert forbidden == []
