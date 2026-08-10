import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    build_high_volatility_volume_bucket_toxicity_reversal_support as support,
)
from training import (
    preregister_high_volatility_volume_bucket_toxicity_reversal as prereg,
)


def _bars(
    start: pd.Timestamp,
    periods: int,
    *,
    quote: float | np.ndarray = 1.0,
    buy_share: float = 0.75,
) -> pd.DataFrame:
    minute = np.arange(periods)
    opens = 100.0 + (minute % 13) * 0.01
    closes = opens * np.exp(np.where(minute % 2 == 0, 0.001, -0.001))
    quotes = np.broadcast_to(quote, periods).astype(float).copy()
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
            "quote_asset_volume": quotes,
            "taker_buy_quote": quotes * buy_share,
        }
    )


def _pair_rows(count: int = 2, start: str = "2024-07-01T00:00:00Z") -> pd.DataFrame:
    starts = pd.date_range(start, periods=count, freq="1h")
    rows = pd.DataFrame(
        {
            "bucket_id": np.arange(count),
            "bucket_start_time": starts,
            "bucket_final_bar_time": starts + pd.Timedelta(minutes=2),
            "feature_available_time": starts + pd.Timedelta(minutes=3),
            "source_valid": [True] * count,
            "bucket_valid": [True] * count,
            "variation_valid": [True] * count,
            "target_prior_minute_count": [1440] * count,
            "bucket_minute_count": [3] * count,
            "target_quote_volume": [60.0] * count,
            "bucket_quote_volume": [75.0] * count,
            "bucket_overshoot_quote_volume": [15.0] * count,
            "bucket_signed_flow": [25.0] * count,
            "bucket_imbalance": [1 / 3] * count,
            "btc_variation": np.arange(1, count + 1, dtype=float),
            "terminal_failure_reason": [""] * count,
        }
    )
    return rows.loc[:, support.PAIR_COLUMNS]


def _features(
    availability: list[str] | None = None,
    eligible: list[bool] | None = None,
) -> pd.DataFrame:
    times = pd.to_datetime(
        availability or ["2024-07-01T00:03:00Z", "2024-07-01T01:03:00Z"],
        utc=True,
    )
    count = len(times)
    active = eligible or [False, True][:count]
    pair = _pair_rows(count)
    pair["feature_available_time"] = times
    pair["bucket_final_bar_time"] = times - pd.Timedelta(minutes=1)
    pair["bucket_start_time"] = times - pd.Timedelta(minutes=3)
    frame = pair.copy()
    frame["toxicity"] = 0.5
    frame["toxicity_rank"] = [0.90 if value else 0.10 for value in active]
    frame["direction_consensus"] = True
    frame["consensus_sign"] = 1
    frame["variation_rank"] = 0.70
    frame["eligible_state"] = active
    frame["source_valid_onset"] = False
    return frame.loc[:, support.FEATURE_COLUMNS]


def test_preregistration_is_bound_to_exact_committed_artifact() -> None:
    assert support.PREREG_SHA == (
        "bd558bc00d6e8ac84a4e57c49832e374e6b2e96fceb8995a47d07b738d3fbee9"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])


def test_query_reads_only_frozen_source_fields() -> None:
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith(
        "SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote FROM bars_binance"
    )
    assert "symbol='BTCUSDT'" in normalized and "interval='1m'" in normalized
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.FIRST_BUCKET_START == pd.Timestamp("2023-04-02T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    for forbidden in ("funding", "gross9", "execution", "pnl"):
        assert forbidden not in normalized.lower()


def test_prepare_source_strictly_validates_schema_timestamps_ohlc_and_flow() -> None:
    raw = _bars(pd.Timestamp("2024-01-01T00:00:00Z"), 4)
    assert support.prepare_source(raw).row_valid.tolist() == [True] * 4

    invalid = raw.copy()
    invalid.loc[0, "high"] = invalid.loc[0, "low"] - 1
    invalid.loc[1, "quote_asset_volume"] = -1
    invalid.loc[2, "taker_buy_quote"] = invalid.loc[2, "quote_asset_volume"] + 1
    invalid.loc[3, "close"] = np.inf
    assert support.prepare_source(invalid).row_valid.tolist() == [False] * 4

    with pytest.raises(RuntimeError, match="schema drift"):
        support.prepare_source(raw.drop(columns="taker_buy_quote"))
    with pytest.raises(RuntimeError, match="duplicate source timestamps"):
        support.prepare_source(pd.concat([raw, raw.iloc[[1]]], ignore_index=True))


def test_bucket_target_is_frozen_at_start_and_whole_minute_is_not_split() -> None:
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    quotes = np.ones(1450)
    quotes[1440:] = 25.0
    raw = _bars(start, 1450, quote=quotes)
    pair = support.build_pair_panel(
        raw,
        first_bucket_start=start + pd.Timedelta(days=1),
        end=start + pd.Timedelta(days=1, minutes=10),
    )
    first = pair.iloc[0]
    assert first.target_quote_volume == 60.0
    assert first.bucket_minute_count == 3
    assert first.bucket_quote_volume == 75.0
    assert first.bucket_overshoot_quote_volume == 15.0
    assert first.bucket_final_bar_time == pd.Timestamp("2023-04-02T00:02:00Z")
    assert first.feature_available_time == pd.Timestamp("2023-04-02T00:03:00Z")
    assert first.bucket_start_time == pd.Timestamp("2023-04-02T00:00:00Z")
    assert pair.iloc[1].bucket_start_time == first.feature_available_time
    assert first.target_quote_volume != first.bucket_quote_volume


def test_bucket_variation_ends_at_exact_completion_availability() -> None:
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    quotes = np.ones(1445)
    quotes[1440:] = 30.0
    raw = _bars(start, 1445, quote=quotes)
    pair = support.build_pair_panel(
        raw,
        first_bucket_start=start + pd.Timedelta(days=1),
        end=start + pd.Timedelta(days=1, minutes=5),
    )
    first = pair.iloc[0]
    exact = raw.iloc[2:1442]
    expected = float(np.square(np.log(exact.close / exact.open)).sum())
    assert first.feature_available_time == pd.Timestamp("2023-04-02T00:02:00Z")
    assert first.btc_variation == pytest.approx(expected)
    assert first.source_valid


def test_maximum_360_minute_failure_is_terminal_and_fails_closed() -> None:
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    quotes = np.full(1800, 24.0)
    quotes[1440:] = 3.0
    pair = support.build_pair_panel(
        _bars(start, 1800, quote=quotes),
        first_bucket_start=start + pd.Timedelta(days=1),
        end=start + pd.Timedelta(days=1, minutes=360),
    )
    assert len(pair) == 1
    failure = pair.iloc[0]
    assert failure.target_quote_volume == 1440.0
    assert failure.bucket_minute_count == 360
    assert failure.bucket_quote_volume == 1080.0
    assert not failure.source_valid
    assert failure.terminal_failure_reason == "maximum_duration_target_not_reached"


def test_missing_or_invalid_constituent_minute_terminally_ends_construction() -> None:
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    quotes = np.ones(1450)
    quotes[1440:] = 20.0
    raw = _bars(start, 1450, quote=quotes).drop(index=1441).reset_index(drop=True)
    pair = support.build_pair_panel(
        raw,
        first_bucket_start=start + pd.Timedelta(days=1),
        end=start + pd.Timedelta(days=1, minutes=10),
    )
    assert len(pair) == 1
    assert pair.iloc[0].bucket_minute_count == 1
    assert (
        pair.iloc[0].terminal_failure_reason == "invalid_or_missing_constituent_minute"
    )


def test_strict_prior_720_480_midrank_excludes_current_skips_invalid_and_caps_history() -> (
    None
):
    values = pd.Series([*map(float, range(721)), np.nan, 720.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:480].isna().all()
    assert ranks.iloc[480] == 1.0
    assert ranks.iloc[720] == 1.0
    assert math.isnan(ranks.iloc[721])
    assert ranks.iloc[722] == pytest.approx((719 + 0.5) / 720)


def test_features_use_24_bucket_toxicity_three_bucket_consensus_and_valid_ranks() -> (
    None
):
    pair = _pair_rows(505)
    pair["bucket_imbalance"] = np.linspace(0.01, 0.50, len(pair))
    pair["bucket_signed_flow"] = pair.bucket_imbalance * pair.bucket_quote_volume
    features = support.build_features(pair)
    assert features.toxicity.iloc[:23].isna().all()
    assert features.toxicity.iloc[23] == pytest.approx(
        pair.bucket_imbalance.iloc[:24].abs().mean()
    )
    assert features.direction_consensus.iloc[1] == False
    assert features.direction_consensus.iloc[2] == True
    assert features.variation_rank.iloc[479] != features.variation_rank.iloc[479]
    assert features.variation_rank.iloc[480] == 1.0
    # Toxicity first exists at bucket 23, so 480 strict priors first exist at 503.
    assert math.isnan(features.toxicity_rank.iloc[502])
    assert features.toxicity_rank.iloc[503] == 1.0


def test_source_valid_onset_requires_immediately_prior_valid_completed_bucket() -> None:
    pair = _pair_rows(506)
    pair["bucket_imbalance"] = np.linspace(0.01, 0.50, len(pair))
    pair["bucket_signed_flow"] = pair.bucket_imbalance * pair.bucket_quote_volume
    features = support.build_features(pair)
    features.loc[502, "toxicity_rank"] = 0.1
    features.loc[503, ["toxicity_rank", "variation_rank"]] = [0.9, 0.7]
    eligible, onset, _, _ = support.active_and_side(features)
    assert not eligible.iloc[502] and eligible.iloc[503]
    assert onset.iloc[503]
    features.loc[502, "source_valid"] = False
    assert not support.active_and_side(features)[1].iloc[503]
    features.loc[502, "source_valid"] = True
    features.loc[503, "bucket_id"] = 999
    assert not support.active_and_side(features)[1].iloc[503]


def test_clock_ceils_availability_adds_five_minutes_reserves_globally_and_skips_splits() -> (
    None
):
    availability = [
        "2023-12-31T18:00:00Z",
        "2023-12-31T20:01:00Z",  # crossing: skipped
        "2024-01-01T00:01:00Z",
        "2024-01-01T00:03:00Z",  # accepted 00:10
        "2024-01-01T01:00:00Z",
        "2024-01-01T01:01:00Z",  # reservation suppresses
        "2024-01-01T06:00:00Z",
        "2024-01-01T06:05:00Z",  # equal exit accepted
    ]
    features = _features(
        availability, [False, True, False, True, False, True, False, True]
    )
    clock = support.build_clock(features)
    assert clock.bucket_id.tolist() == [3, 7]
    assert (
        clock.decision_time.tolist()
        == pd.to_datetime(["2024-01-01T00:05:00Z", "2024-01-01T06:05:00Z"]).tolist()
    )
    assert (
        clock.entry_time.tolist()
        == pd.to_datetime(["2024-01-01T00:10:00Z", "2024-01-01T06:10:00Z"]).tolist()
    )
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert set(clock.split) == {"test"}


def test_all_controls_are_isolated_and_nonpromotable_geometries() -> None:
    features = _features()
    assert support.active_and_side(features)[0].tolist() == [False, True]

    features.loc[1, "variation_rank"] = 0.1
    assert support.active_and_side(features)[0].tolist() == [False, False]
    assert support.active_and_side(features, "no_variation_gate")[0].tolist() == [
        False,
        True,
    ]
    features.loc[1, ["variation_rank", "toxicity_rank"]] = [0.7, 0.1]
    assert support.active_and_side(features, "no_toxicity_tail")[0].tolist() == [
        True,
        True,
    ]

    features.loc[0, "variation_rank"] = 0.1
    features.loc[1, ["toxicity_rank", "direction_consensus", "consensus_sign"]] = [
        0.9,
        False,
        0,
    ]
    features.loc[1, "bucket_imbalance"] = -0.2
    _, onset, side, _ = support.active_and_side(
        features, "single_last_bucket_direction"
    )
    assert onset.tolist() == [False, True] and side.iloc[1] == 1

    features.loc[1, ["direction_consensus", "consensus_sign"]] = [True, -1]
    assert support.active_and_side(features, "direction_flip")[2].iloc[1] == -1
    assert support.active_and_side(features, "forced_long")[2].iloc[1] == 1

    stale = _features(
        ["2024-07-01T00:03:00Z", "2024-07-01T01:03:00Z", "2024-07-01T02:03:00Z"],
        [False, True, False],
    )
    stale.loc[0, ["toxicity_rank", "variation_rank", "consensus_sign"]] = [0.9, 0.7, -1]
    stale.loc[1, ["toxicity_rank", "variation_rank"]] = [0.1, 0.1]
    eligible, _, side, used = support.active_and_side(
        stale, "one_bucket_stale_features"
    )
    assert eligible.tolist() == [False, True, False]
    assert side.iloc[1] == 1 and used.loc[1, "bucket_id"] == 0


def test_support_stats_enforce_event_side_and_month_gates() -> None:
    clock = pd.DataFrame(
        {
            "split": ["train"] * 10,
            "side": [1] * 8 + [-1] * 2,
            "entry_time": pd.to_datetime(
                [f"2023-{month:02d}-01T00:00:00Z" for month in range(7, 12)] * 2
            ),
        }
    )
    stats = support.support_stats(clock, "train")
    assert stats == {
        "events": 10,
        "longs": 8,
        "shorts": 2,
        "minority_side_share": 0.2,
        "max_month_share": 0.2,
    }
    assert support.support_stats(clock, "test")["events"] == 0


def test_deterministic_immutable_writer_allows_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    frame = _features()
    first = support.deterministic_csv_gzip(frame)
    assert first == support.deterministic_csv_gzip(frame.copy())
    assert gzip.decompress(first).startswith(b"bucket_id,bucket_start_time")
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, first)
    with pytest.raises(RuntimeError, match="immutable HVVBTR artifact"):
        support.write_immutable(path, first + b"drift")


def test_run_writes_pair_feature_clock_controls_manifest_and_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "PAIR_PANEL", source_dir / "pair.csv.gz")
    monkeypatch.setattr(support, "FEATURE_PANEL", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(support, "load_source", lambda: pd.DataFrame())
    pair = _pair_rows()
    features = _features()
    monkeypatch.setattr(support, "build_pair_panel", lambda _bars: pair)
    monkeypatch.setattr(support, "build_features", lambda _pair: features)

    result = support.run()
    assert support.run() == result
    manifest = json.loads((source_dir / "manifest.json").read_text())
    written = json.loads((tmp_path / "result.json").read_text())
    assert written == result
    assert manifest["pair_panel"]["path"] == str(source_dir / "pair.csv.gz")
    assert manifest["feature_panel"]["path"] == str(source_dir / "features.csv.gz")
    assert manifest["execution_prices_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["ranking"] == {
        "lookback_valid_buckets": 720,
        "minimum_prior_valid_buckets": 480,
        "current_excluded": True,
        "ties": "midrank",
    }
    assert result["reservation"] == {
        "scope": "global",
        "hours": 6,
        "interval": "half_open",
        "equal_open_after_exit_allowed": True,
        "split_crossing_action": "skip",
    }
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all(not item["promotion_authorized"] for item in result["controls"].values())
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
