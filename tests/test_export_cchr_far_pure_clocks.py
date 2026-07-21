# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportReturnType=false, reportOperatorIssue=false, reportInvalidTypeForm=false, reportIndexIssue=false
from __future__ import annotations

import ast
from datetime import timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from training import export_cchr_far_pure_clocks as far


UTC = timezone.utc


def _market_frame(periods: int, *, start: str = "2021-01-01T00:00:00Z") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "close": np.full(periods, 100.0),
            "quote_asset_volume": np.full(periods, 100.0),
            "taker_buy_quote": np.full(periods, 60.0),
            "open_interest": np.linspace(1000.0, 1000.0 + periods, periods),
            "open_interest_value": np.linspace(100000.0, 100000.0 + periods, periods),
            "sum_open_interest": np.linspace(1000.0, 1000.0 + periods, periods),
            "count_long_short_ratio": np.full(periods, 1.2),
        }
    )


def _metrics_frame(
    periods: int, *, start: str = "2021-01-01T00:00:00Z"
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "create_time": dates,
            "sum_open_interest": np.linspace(1000.0, 1000.0 + periods, periods),
            "count_long_short_ratio": np.full(periods, 1.2),
        }
    )


def test_exact_far_candidate_map_subset_and_member_ids() -> None:
    members = far.far_candidate_map()
    assert len(members) == 12
    assert list(members) == sorted(members)
    assert set(members) == {
        f"far:age={age}:half_life={half_life}:q=0.90:hold={hold}"
        for age in (1, 3, 6)
        for half_life in (288, 864)
        for hold in (72, 144)
    }
    for candidate_id, spec in members.items():
        assert spec["family"] == "far"
        assert spec["component_weight"] is None
        assert candidate_id.endswith(f"hold={spec['hold_bars']}")
        assert spec["parameters"]["q"] == 0.90


def test_hash_bound_loader_contracts_are_allowlisted() -> None:
    assert far.MARKET_COLUMNS == (
        "date",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
        "open_interest",
        "open_interest_value",
    )
    assert far.METRICS_COLUMNS == (
        "create_time",
        "sum_open_interest",
        "count_long_short_ratio",
    )
    assert far.FUNDING_COLUMNS == ("date", "funding_time", "funding_rate")
    assert (
        far.MARKET_SHA256
        == "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
    )
    assert (
        far.METRICS_SHA256
        == "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
    )
    assert (
        far.FUNDING_SHA256
        == "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
    )


def test_attach_delayed_metrics_requires_exact_one_complete_bar_no_staleness() -> None:
    market = _market_frame(4).loc[:, list(far.MARKET_COLUMNS)]
    metrics = _metrics_frame(4)
    joined = far._attach_delayed_metrics(market, metrics)
    assert pd.isna(joined.loc[0, "sum_open_interest"])
    assert joined.loc[1, "sum_open_interest"] == metrics.loc[0, "sum_open_interest"]
    assert joined.loc[1, "positioning_source_time"] == market.loc[0, "date"]

    stale_metrics = metrics.drop(index=1).reset_index(drop=True)
    bounded = far._attach_delayed_metrics(market, stale_metrics)
    assert bounded["positioning_available"].tolist() == [0.0, 1.0, 0.0, 1.0]
    available = bounded["positioning_available"].eq(1.0)
    ages = (
        bounded.loc[available, "date"]
        - bounded.loc[available, "positioning_source_time"]
    )
    assert ages.eq(pd.Timedelta("5min")).all()


def test_funding_settlement_timing_uses_exact_time_ceil_without_future_rows() -> None:
    dates = pd.Series(
        pd.date_range("2021-01-01T00:00:00Z", periods=4, freq="5min", tz="UTC")
    )
    funding = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2021-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T00:00:00Z"),
            ],
            "funding_time": [
                int(pd.Timestamp("2021-01-01T00:02:00Z").timestamp() * 1000),
                int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000),
            ],
            "funding_rate": [0.001, 0.999],
        }
    )
    event_rate = far._funding_event_rate(dates, funding)
    assert np.isnan(event_rate[0])
    assert event_rate[1] == pytest.approx(0.001)
    assert np.isnan(event_rate[2:]).all()


def test_feature_formula_decay_aging_and_causal_origin_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market_frame(8)
    dates = cast(pd.Series, market["date"])
    event_rate = np.full(len(market), np.nan)
    event_rate[[2, 4]] = [0.10, 0.20]

    owner = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    owner_change = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, np.nan, -1.0, np.nan, np.nan]
    )
    monkeypatch.setattr(far, "owner_state", lambda _market: (owner, owner_change))
    monkeypatch.setattr(
        far,
        "prior_z",
        lambda values, window=2016: pd.to_numeric(values, errors="coerce"),
    )

    features_fast = far.build_features(
        market,
        dates,
        event_rate,
        min_age_settlements=1,
        half_life_bars=1,
    )
    features_slow = far.build_features(
        market,
        dates,
        event_rate,
        min_age_settlements=1,
        half_life_bars=1_000_000,
    )

    assert features_fast.causal_origin.iloc[5] == pd.Timestamp("2021-01-01T00:10:00Z")
    assert features_fast.score.iloc[5] < 0
    assert abs(features_slow.score.iloc[5]) > abs(features_fast.score.iloc[5])

    too_young = far.build_features(
        market,
        dates,
        event_rate,
        min_age_settlements=3,
        half_life_bars=1,
    )
    assert pd.isna(too_young.causal_origin.iloc[5])
    assert too_young.score.fillna(0.0).abs().sum() == 0.0


def test_fit_range_is_fixed_to_2020_10_15_through_2023_01_01() -> None:
    dates = pd.Series(
        [
            pd.Timestamp("2020-10-14T23:55:00Z"),
            pd.Timestamp("2020-10-15T00:00:00Z"),
            pd.Timestamp("2022-12-31T23:55:00Z"),
            pd.Timestamp("2023-01-01T00:00:00Z"),
        ]
    )
    score = pd.Series([100.0, 1.0, 3.0, 1000.0])
    assert far.fit_abs_threshold(score, dates, min_observations=2) == pytest.approx(2.8)
    with pytest.raises(ValueError, match="insufficient"):
        far.fit_abs_threshold(score, dates, min_observations=3)


def test_signal_timing_hold_containment_and_per_id_nonoverlap() -> None:
    features = far.FarFeatureBank(
        dates=pd.Series(
            pd.to_datetime(
                [
                    "2023-01-01T00:00:00Z",
                    "2023-01-01T00:05:00Z",
                    "2023-01-01T00:10:00Z",
                    "2023-01-01T00:15:00Z",
                    "2023-01-01T00:20:00Z",
                ],
                utc=True,
            )
        ),
        score=pd.Series([0.0, 2.0, 3.0, 0.0, -2.0]),
        causal_origin=pd.Series(
            [
                pd.NaT,
                pd.Timestamp("2023-01-01T00:00:00Z"),
                pd.Timestamp("2023-01-01T00:00:00Z"),
                pd.NaT,
                pd.Timestamp("2023-01-01T00:15:00Z"),
            ]
        ),
    )
    candidates = far.raw_candidates_for_member(
        features,
        candidate_id="far:synthetic",
        threshold=1.0,
        hold_bars=4,
    )
    assert len(candidates) == 2
    first = candidates[0]
    assert first.signal_time == pd.Timestamp("2023-01-01T00:05:00Z").to_pydatetime()
    assert first.decision_time == pd.Timestamp("2023-01-01T00:10:00Z").to_pydatetime()
    assert first.entry_time == first.decision_time
    assert first.exit_time == pd.Timestamp("2023-01-01T00:30:00Z")
    assert first.causal_origins == (
        pd.Timestamp("2023-01-01T00:00:00Z").to_pydatetime(),
    )

    frame = far.schedule_candidates(candidates)
    assert list(frame["entry_time"]) == ["2023-01-01T00:10:00Z"]
    assert list(frame["side"]) == [1]


def test_build_far_clock_frame_uses_common_containment_then_per_id_nonoverlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.Series(
        pd.date_range("2023-01-01T00:00:00Z", periods=5, freq="5min", tz="UTC")
    )
    market = pd.DataFrame(index=dates.index)
    event_rate = np.full(len(dates), np.nan)

    def fake_build_features(
        _market: pd.DataFrame,
        input_dates: pd.Series,
        _event_rate: np.ndarray,
        *,
        min_age_settlements: int,
        half_life_bars: int,
    ) -> far.FarFeatureBank:
        del min_age_settlements, half_life_bars
        return far.FarFeatureBank(
            dates=input_dates,
            score=pd.Series([0.0, 2.0, 0.0, -2.0, 0.0]),
            causal_origin=pd.Series([pd.NaT] * len(input_dates)),
        )

    monkeypatch.setattr(far, "build_features", fake_build_features)
    monkeypatch.setattr(far, "fit_abs_threshold", lambda *args, **kwargs: 1.0)
    frame = far.build_far_clock_frame(market, dates, event_rate, min_fit_observations=1)
    assert set(frame["candidate_id"]) == set(far.far_candidate_map())
    assert len(frame) == 12
    assert frame.groupby("candidate_id").size().eq(1).all()


def test_module_has_no_legacy_imports_or_outcome_result_json_reads() -> None:
    source = Path(far.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert "training.search_funding_age_rollover_transfer_alpha" not in imported_modules
    assert all("search_positioning" not in module for module in imported_modules)
    assert all("strict_bar_backtest" not in module for module in imported_modules)
    assert "funding_age_rollover_transfer_alpha_scan_2026-07-13.json" not in source
