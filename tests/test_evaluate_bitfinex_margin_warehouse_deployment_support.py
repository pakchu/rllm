from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_bitfinex_margin_warehouse_deployment_support as support
from training import preregister_bitfinex_margin_warehouse_deployment as prereg


SOURCE_ACCESS_SEAL_SHA256 = (
    "ab82e7456d6710b30d8654066e79f4eb14f58d29e5ec744421692fe4518dd72b"
)


def test_support_evaluator_accepts_exact_frozen_preregistration() -> None:
    payload = support.validate_preregistration()
    assert payload["candidate_family"] == prereg.CANDIDATE_FAMILY
    assert payload["support_gates"] == prereg.SUPPORT_GATES


def test_source_access_seal_binds_evaluator_before_feature_access() -> None:
    path = support.SOURCE_ACCESS_SEAL
    assert support.sha256_file(path) == SOURCE_ACCESS_SEAL_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    unhashed = dict(payload)
    manifest_hash = unhashed.pop("manifest_hash")
    assert manifest_hash == support.canonical_hash(unhashed)
    assert payload["bindings"]["evaluator_source"]["sha256"] == (
        "6983246760bfd49a02e059b8dbebea9ca445778f8c700a6d488ed4bf5fffb630"
    )
    assert payload["bindings"]["protocol_document"]["sha256"] == (
        "e117472e600643e4b082e447814d9f486716494cdaa92be32f1723e05cd459d3"
    )
    assert payload["feature_values_inspected_before_seal"] is False
    assert payload["market_outcomes_opened_before_seal"] is False


def test_strict_prior_robust_zscore_excludes_current_observation() -> None:
    values = pd.Series([0.0, 2.0, 4.0, 100.0])
    result = support.strict_prior_robust_zscore(
        values,
        window=3,
        minimum=3,
        mad_scale=1.4826,
        block_rows=2,
    )
    assert result.iloc[:3].isna().all()
    expected = (100.0 - 2.0) / (1.4826 * 2.0)
    assert result.iloc[3] == pytest.approx(expected)


def test_missing_physical_hour_stays_inside_fixed_rolling_window() -> None:
    values = pd.Series([0.0, 1.0, np.nan, 3.0, 10.0])
    result = support.strict_prior_robust_zscore(
        values,
        window=3,
        minimum=3,
        mad_scale=1.4826,
    )
    assert math.isnan(result.iloc[4])


def test_exact_lag_never_uses_nearest_or_filled_value() -> None:
    times = pd.Series(
        pd.to_datetime(
            ["2021-01-01 00:05", "2021-01-01 01:05", "2021-01-01 03:05"],
            utc=True,
        )
    )
    values = pd.Series([10.0, 20.0, 40.0])
    lagged = support.exact_lag(values, times, hours=1)
    assert math.isnan(lagged.iloc[0])
    assert lagged.iloc[1] == 10.0
    assert math.isnan(lagged.iloc[2])


def test_onset_requires_exact_preceding_hour() -> None:
    times = pd.Series(
        pd.to_datetime(
            ["2021-01-01 00:05", "2021-01-01 01:05", "2021-01-01 03:05"],
            utc=True,
        )
    )
    onset = support.exact_hourly_onset(pd.Series([False, True, True]), times)
    assert onset.tolist() == [False, True, False]

    complete_grid = pd.Series(pd.date_range("2021-01-01", periods=3, freq="h", tz="UTC"))
    onset_after_missing = support.exact_hourly_onset(
        pd.Series([False, False, True]),
        complete_grid,
        source_complete=pd.Series([True, False, True]),
    )
    assert onset_after_missing.tolist() == [False, False, False]


def test_hourly_grid_keeps_latest_duplicate_and_inserts_missing_slot() -> None:
    times = pd.to_datetime(
        [
            "2020-01-01 00:05:00",
            "2020-01-01 01:02:00",
            "2020-01-01 01:05:00",
            "2020-01-01 03:05:00",
        ],
        utc=True,
    )
    source = pd.DataFrame(
        {
            "symbol": "fUSD",
            "observation_time": times,
            "available_at": pd.Series(times).dt.floor("h")
            + pd.Timedelta(minutes=15),
            "funding_amount": [100.0, 101.0, 102.0, 103.0],
        }
    )
    grid = support.normalize_symbol_hourly_grid(source)
    first = grid.iloc[:4].reset_index(drop=True)
    assert first.loc[1, "observation_time"] == pd.Timestamp(
        "2020-01-01 01:05:00", tz="UTC"
    )
    assert first.loc[1, "funding_amount"] == 102.0
    assert pd.isna(first.loc[2, "observation_time"])
    assert pd.isna(first.loc[2, "funding_amount"])


def test_source_header_is_exact_allowlist() -> None:
    support.validate_source_header(
        support.EXPECTED_SOURCE_COLUMNS,
        list(support.EXPECTED_SOURCE_COLUMNS),
    )
    with pytest.raises(ValueError, match="exactly allowlisted"):
        support.validate_source_header(
            (*support.EXPECTED_SOURCE_COLUMNS, "future_return"),
            list(support.EXPECTED_SOURCE_COLUMNS),
        )


def _candidate_rows(entries: list[str]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame(
        {
            "candidate": prereg.CANDIDATE_FAMILY,
            "variant_id": prereg.VARIANTS[0].variant_id,
            "control": "primary",
            "split": "",
            "symbol": "fUSD",
            "side": 1,
            "observation_time": entry - pd.Timedelta(minutes=20),
            "source_available_at": entry - pd.Timedelta(minutes=5),
            "decision_available_at": entry - pd.Timedelta(minutes=5),
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=12),
        }
    )


def test_schedule_is_split_contained_and_globally_nonoverlapping() -> None:
    candidates = _candidate_rows(
        [
            "2021-01-01 00:20",
            "2021-01-01 02:20",
            "2021-01-01 12:20",
            "2022-12-31 12:00",
            "2022-12-31 18:20",
        ]
    )
    scheduled = support._schedule_split(
        candidates,
        split="train",
        start=support._timestamp("2021-01-01T00:00:00Z"),
        end=support._timestamp("2023-01-01T00:00:00Z"),
    )
    assert list(scheduled["entry_time"]) == list(
        pd.to_datetime(["2021-01-01 00:20", "2021-01-01 12:20"], utc=True)
    )
    assert bool(scheduled["split"].eq("train").all())


def test_two_symbol_simultaneous_onsets_are_abstentions(monkeypatch) -> None:
    times = pd.to_datetime(
        [
            "2021-01-01 00:05",
            "2021-01-01 01:05",
            "2021-01-01 12:05",
            "2021-01-01 13:05",
        ],
        utc=True,
    )
    source = pd.concat(
        [
            pd.DataFrame({"symbol": symbol, "observation_time": times})
            for symbol in prereg.FROZEN_POLICY.symbols
        ],
        ignore_index=True,
    )

    def fake_features(
        symbol_source: pd.DataFrame,
        variant: prereg.Variant,
        *,
        block_rows: int = 256,
    ) -> pd.DataFrame:
        del variant, block_rows
        symbol = str(symbol_source["symbol"].iloc[0])
        observation = times + (
            pd.Timedelta(minutes=3) if symbol == "fBTC" else pd.Timedelta(0)
        )
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "source_hour": pd.Series(times).dt.floor("h"),
                "observation_time": observation,
                "available_at": pd.Series(times).dt.floor("h")
                + pd.Timedelta(minutes=20 if symbol == "fBTC" else 15),
            }
        )
        shared = pd.Series([True, False, False, symbol == "fUSD"])
        for control in support.CONTROL_ORDER:
            frame[control] = shared if control == "primary" else False
        return frame

    monkeypatch.setattr(support, "build_symbol_features", fake_features)
    clocks = support.build_variant_clocks(source, prereg.VARIANTS[0])
    primary = clocks[clocks["control"].eq("primary")]
    assert len(primary) == 1
    assert primary.iloc[0]["symbol"] == "fUSD"
    assert primary.iloc[0]["entry_time"] == pd.Timestamp(
        "2021-01-01 13:20", tz="UTC"
    )


def test_support_summary_reports_all_frozen_concentration_axes() -> None:
    events = _candidate_rows(
        [
            "2021-01-01 00:20",
            "2021-01-08 00:20",
            "2021-02-01 00:20",
            "2023-07-01 00:20",
        ]
    )
    events.loc[1, "side"] = -1
    events.loc[3, "side"] = -1
    summary = support.support_summary(events)
    assert summary["events"] == 4
    assert summary["long_share"] == 0.5
    assert summary["short_share"] == 0.5
    assert summary["year_counts"] == {"2021": 3, "2023": 1}
    assert summary["half_counts"] == {"2021-H1": 3, "2023-H2": 1}
    assert summary["maximum_month_share"] == 0.5
    assert summary["maximum_rolling_14day_share"] == 0.5


def test_novelty_is_common_window_bounded_and_bidirectional() -> None:
    candidate = pd.DatetimeIndex(
        pd.to_datetime(
            ["2020-12-31 23:20", "2021-01-01 00:20", "2021-01-02 00:20"],
            utc=True,
        )
    )
    comparator = pd.DatetimeIndex(
        pd.to_datetime(
            ["2021-01-01 00:20", "2021-01-02 01:20", "2024-01-01 00:20"],
            utc=True,
        )
    )
    metrics = support.novelty_metrics(
        candidate,
        comparator,
        start=support._timestamp("2021-01-01T00:00:00Z"),
        end=support._timestamp("2024-01-01T00:00:00Z"),
        hours=1,
        minimum_candidate=2,
        minimum_comparator=2,
    )
    assert metrics["candidate_events"] == 2
    assert metrics["comparator_events"] == 2
    assert metrics["sufficient_common_support"] is True
    assert metrics["exact_jaccard"] == pytest.approx(1 / 3)
    assert metrics["candidate_near_share"] == 1.0
    assert metrics["comparator_near_share"] == 1.0


def test_sqfd_comparator_loader_uses_only_frozen_2023_prefix(monkeypatch) -> None:
    freeze = json.loads(support.COMPARATOR_FREEZE.read_text(encoding="utf-8"))
    expected_hashes = {
        str(item["path"]): str(item["sha256"]) for item in freeze["comparators"]
    }
    expected_hashes[str(support.SQFD_PREFIX)] = support.SQFD_PREFIX_SHA256
    opened: list[Path] = []

    def fake_hash(path: str | Path) -> str:
        return expected_hashes[str(path)]

    def fake_read_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
        opened.append(Path(path))
        if kwargs.get("nrows") == 0:
            return pd.DataFrame(columns=pd.Index(["control", "entry_time"]))
        return pd.DataFrame(
            {
                "control": ["primary"],
                "entry_time": ["2023-09-02T01:05:00Z"],
            }
        )

    monkeypatch.setattr(support, "sha256_file", fake_hash)
    monkeypatch.setattr(support.pd, "read_csv", fake_read_csv)
    clocks, _ = support.load_comparator_clocks(freeze)
    assert len(clocks["SQFD-6"]) == 1
    assert support.repository_path(support.SQFD_PREFIX) in opened
    assert support.repository_path(
        "data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz"
    ) not in opened


def test_support_checks_apply_named_subsets_and_union_concentration() -> None:
    train = {
        "events": 60,
        "year_counts": {"2021": 30, "2022": 30},
    }
    selection = {
        "events": 30,
        "half_counts": {"2023-H1": 15, "2023-H2": 15},
    }
    combined = {
        "long_share": 0.5,
        "short_share": 0.5,
        "maximum_month_share": 0.19,
        "maximum_weekday_share": 0.24,
        "maximum_rolling_14day_share": 0.19,
    }
    checks, failures = support.support_checks(train, selection, combined)
    assert failures == []
    assert all(checks.values())

    selection["half_counts"] = {"2023-H1": 20, "2023-H2": 10}
    combined["maximum_weekday_share"] = 0.26
    _, failures = support.support_checks(train, selection, combined)
    assert failures == ["minimum_2023_h2_events", "maximum_weekday_share"]


def test_clock_encoding_is_reproducible_and_contains_no_outcome_fields() -> None:
    clocks = _candidate_rows(["2021-01-01 00:20"])
    assert support._clock_bytes(clocks) == support._clock_bytes(clocks)
    forbidden = {"price", "return", "pnl", "cagr", "drawdown"}
    assert forbidden.isdisjoint({str(column).lower() for column in clocks.columns})
    source_text = Path(support.EVALUATOR_SOURCE).read_text(encoding="utf-8")
    assert "BTCUSDT_5m" not in source_text
    assert "strict_bar_backtest" not in source_text


def test_frozen_support_artifacts_pass_without_opening_outcomes() -> None:
    result_path = Path(
        "results/bitfinex_margin_warehouse_deployment_support_2026-07-20.json"
    )
    clock_path = Path(
        "data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz"
    )
    assert support.sha256_file(result_path) == (
        "c857e070f4cb157a005f4a95bee0bff9c7b30daf97128832a690a68d05bfb79c"
    )
    assert support.sha256_file(clock_path) == (
        "02b4fcc462a5a48be7673649f4cf4b2f9bb210baca4294eed1696d479820cccc"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    unhashed = dict(payload)
    manifest_hash = unhashed.pop("manifest_hash")
    assert manifest_hash == support.canonical_hash(unhashed)
    assert payload["clock_artifact"]["rows"] == 6_583
    assert payload["family_support_passed"] is True
    assert payload["passing_variants"] == [
        variant.variant_id for variant in prereg.VARIANTS
    ]
    assert payload["outcome_boundary"]["outcomes_opened"] is False
    assert payload["outcome_boundary"]["btc_market_rows_read"] == 0
    assert payload["outcome_boundary"]["post_2023_rows_read"] == 0


def test_robust_zscore_requires_positive_mad() -> None:
    values = pd.Series(np.ones(5))
    result = support.strict_prior_robust_zscore(
        values,
        window=3,
        minimum=3,
        mad_scale=1.4826,
    )
    assert result.isna().all()
