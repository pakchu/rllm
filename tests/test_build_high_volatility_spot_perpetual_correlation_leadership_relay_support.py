import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_spot_perpetual_correlation_leadership_relay_support as subject
from training import preregister_high_volatility_spot_perpetual_correlation_leadership_relay as prereg


def return_paths(periods: int = 360) -> tuple[np.ndarray, np.ndarray]:
    minute = np.arange(periods, dtype=float)
    spot = 0.0003 + 0.0012 * np.sin(minute / 7.0) + 0.0004 * np.cos(minute / 17.0)
    perpetual = np.empty(periods)
    perpetual[0] = spot[0]
    perpetual[1:] = spot[:-1] + 0.00002 * np.sin(minute[1:] / 5.0)
    return spot, perpetual


def bars(start: pd.Timestamp, periods: int = 360) -> pd.DataFrame:
    spot, perpetual = return_paths(periods)
    frames = []
    for venue, path, offset in (("spot", spot, 0.0), ("perpetual", perpetual, 1.0)):
        minute = np.arange(periods)
        opens = 100.0 + offset + minute * 0.001
        closes = opens * np.exp(path)
        frames.append(pd.DataFrame({
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "venue": venue,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "venue"]).reset_index(drop=True)


def features() -> pd.DataFrame:
    decisions = pd.to_datetime([
        "2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z",
        "2024-07-01T02:00:00Z", "2024-07-01T03:00:00Z",
    ])
    rows = []
    for decision in decisions:
        row = {column: 0.0 for column in subject.FEATURE_COLUMNS}
        row.update({
            "decision_time": decision, "feature_available_time": decision,
            "source_valid": True, "minute_count": 720,
            "spot_leads_perpetual": 0.4, "perpetual_leads_spot": 0.1,
            "leadership_advantage": 0.3, "same_minute_correlation": 0.2,
            "perpetual_variation": 0.01, "perpetual_final_hour_return": 0.01,
            "spot_final_hour_return": 0.02, "direction_side": 1,
            "leadership_rank": 0.9, "same_minute_correlation_rank": 0.9,
            "perpetual_variation_rank": 0.7,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=subject.FEATURE_COLUMNS)


def test_preregistration_and_source_only_query_are_bound() -> None:
    assert subject.PREREG_SHA == "780443153cd5587dee3faaf99354710edfaba7e789296d9dde9d12a69068d756"
    assert subject.sha(prereg.DEFAULT_OUTPUT) == subject.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    normalized = " ".join(subject.QUERY.split())
    assert "FROM bars_binance_spot" in normalized
    assert "FROM bars_binance" in normalized
    assert "UNION ALL" in normalized
    assert all(word not in normalized.lower() for word in ("funding", "pnl", "gross9"))


def test_exact_directed_correlations_variation_and_direction() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    pair = subject.boundary_pair(subject.prepare_source(bars(start)), start + pd.Timedelta(hours=6))
    spot, perpetual = return_paths()
    spot_lead = np.corrcoef(spot[:-1], perpetual[1:])[0, 1]
    perpetual_lead = np.corrcoef(perpetual[:-1], spot[1:])[0, 1]
    assert pair["source_valid"] is True
    assert pair["minute_count"] == 720
    assert pair["spot_leads_perpetual"] == pytest.approx(spot_lead)
    assert pair["perpetual_leads_spot"] == pytest.approx(perpetual_lead)
    assert pair["leadership_advantage"] == pytest.approx(spot_lead - perpetual_lead)
    assert pair["same_minute_correlation"] == pytest.approx(np.corrcoef(spot, perpetual)[0, 1])
    assert pair["perpetual_variation"] == pytest.approx(np.sqrt(np.square(perpetual).sum()))
    assert pair["perpetual_final_hour_return"] == pytest.approx(perpetual[-60:].sum())
    assert pair["spot_final_hour_return"] == pytest.approx(spot[-60:].sum())
    assert pair["direction_side"] == 1


def test_source_rules_fail_closed() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    raw = bars(start)
    missing = raw.drop(raw[raw.venue.eq("spot")].index[10])
    assert subject.boundary_pair(subject.prepare_source(missing), start + pd.Timedelta(hours=6))["source_valid"] is False
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate source key"):
        subject.prepare_source(duplicate)


def test_strict_prior_midrank_excludes_current() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = subject.strict_prior_midrank(values)
    assert ranks.iloc[:1440].isna().all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_primary_uses_fresh_source_valid_onset_and_controls_are_diagnostic() -> None:
    frame = features()
    frame["leadership_rank"] = [0.7, 0.8, 0.9, 0.7]
    eligible, onset, side, _ = subject.active_and_side(frame)
    assert eligible.tolist() == [False, True, True, False]
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]
    frame["leadership_rank"] = 0.7
    assert subject.active_and_side(frame, "no_leadership_tail")[0].all()
    frame["perpetual_variation_rank"] = 0.6
    assert not subject.active_and_side(frame, "no_leadership_tail")[0].any()
    frame["leadership_rank"] = 0.9
    assert subject.active_and_side(frame, "no_variation_gate")[0].all()
    frame["perpetual_variation_rank"] = 0.7
    frame["same_minute_correlation_rank"] = [0.79, 0.8, 0.8, 0.8]
    assert subject.active_and_side(frame, "same_minute_correlation")[0].tolist() == [False, True, True, True]
    assert subject.active_and_side(frame, "direction_flip")[2].tolist() == [-1, -1, -1, -1]
    assert subject.active_and_side(frame, "forced_long")[2].tolist() == [1, 1, 1, 1]


def test_clock_uses_eight_hour_global_reservation() -> None:
    frame = features()
    frame["leadership_rank"] = [0.7, 0.8, 0.7, 0.8]
    clock = subject.build_clock(frame)
    assert len(clock) == 1
    assert clock.entry_time.iloc[0] == pd.Timestamp("2024-07-01T01:05:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2024-07-01T09:05:00Z")
