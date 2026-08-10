import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_alt_variation_dispersion_compression_relay_support as support
from training import preregister_high_volatility_alt_variation_dispersion_compression_relay as prereg


def _bars(start: pd.Timestamp, periods: int = 360) -> pd.DataFrame:
    frames = []
    signs = {
        "BTCUSDT": 1, "ADAUSDT": 1, "BNBUSDT": 1, "DOGEUSDT": 1,
        "ETHUSDT": 1, "SOLUSDT": -1, "XRPUSDT": -1,
    }
    for symbol_index, symbol in enumerate(support.SYMBOLS):
        minute = np.arange(periods)
        scale = 0.0005 * (symbol_index + 1)
        opens = 100.0 + symbol_index + minute * 0.001
        closes = opens * np.exp(signs[symbol] * scale)
        frames.append(pd.DataFrame({
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "symbol": symbol,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)


def _features(decisions: list[str] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(decisions or [
        "2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z",
        "2024-07-01T02:00:00Z", "2024-07-01T03:00:00Z",
    ])
    rows = []
    for decision in times:
        row = {column: 0.0 for column in support.FEATURE_COLUMNS}
        row.update({
            "decision_time": decision, "feature_available_time": decision,
            "source_valid": True, "minute_count": 2520,
            "btc_variation": 0.01, "alt_log_variation_dispersion": 0.1,
            "btc_final_hour_return": 0.01, "alt_positive_count": 4,
            "alt_negative_count": 2, "alt_majority_side": 1,
            "dispersion_rank": 0.1, "btc_variation_rank": 0.7,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=support.FEATURE_COLUMNS)


def test_preregistration_binding_and_source_only_query_contract() -> None:
    assert support.PREREG_SHA == "1c11a52271a3c3d7829880851d7012f451350cc00fe0a1e5269a2bccb0feb23b"
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith("SELECT ts,symbol,open,high,low,close FROM bars_binance")
    assert "interval='1m'" in normalized
    assert all(f"'{symbol}'" in normalized for symbol in support.SYMBOLS)
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    for forbidden in ("funding", "execution", "pnl", "gross9"):
        assert forbidden not in support.QUERY.lower()


def test_exact_aligned_grid_variation_dispersion_and_final_hour_formulas() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    raw = _bars(start)
    pair = support.boundary_pair(
        support.prepare_source(raw), start + pd.Timedelta(hours=6)
    )
    assert pair["source_valid"] is True
    assert pair["minute_count"] == 360 * 7
    variations = []
    for symbol in support.SYMBOLS:
        symbol_rows = raw[raw.symbol.eq(symbol)]
        expected_variation = np.square(
            np.log(symbol_rows.close.to_numpy() / symbol_rows.open.to_numpy())
        ).sum()
        key = "btc_variation" if symbol == "BTCUSDT" else f"{symbol.lower()}_variation"
        assert pair[key] == pytest.approx(expected_variation)
        if symbol != "BTCUSDT":
            variations.append(expected_variation)
        final = symbol_rows.iloc[-60:]
        expected_return = math.log(final.close.iloc[-1] / final.open.iloc[0])
        return_key = "btc_final_hour_return" if symbol == "BTCUSDT" else f"{symbol.lower()}_final_hour_return"
        assert pair[return_key] == pytest.approx(expected_return)
    assert pair["alt_log_variation_dispersion"] == pytest.approx(
        np.std(np.log(variations), ddof=0)
    )
    assert pair["alt_positive_count"] == 4
    assert pair["alt_negative_count"] == 2
    assert pair["alt_majority_side"] == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incoherent", "zero_variation"])
def test_source_and_exact_grid_fail_closed(mutation: str) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    raw = _bars(start)
    if mutation == "missing":
        raw = raw.drop(raw[(raw.symbol == "ADAUSDT")].index[10])
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate source key"):
            support.prepare_source(raw)
        return
    elif mutation == "incoherent":
        index = raw[raw.symbol.eq("BNBUSDT")].index[20]
        raw.loc[index, "high"] = raw.loc[index, "low"] - 1
    else:
        indexes = raw[raw.symbol.eq("DOGEUSDT")].index
        raw.loc[indexes, "close"] = raw.loc[indexes, "open"]
        raw.loc[indexes, "high"] = raw.loc[indexes, "open"]
        raw.loc[indexes, "low"] = raw.loc[indexes, "open"]
    pair = support.boundary_pair(support.prepare_source(raw), start + pd.Timedelta(hours=6))
    assert pair["source_valid"] is False


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_history() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:1440].isna().all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_feature_ranks_use_only_source_valid_decisions() -> None:
    rows = []
    start = pd.Timestamp("2023-04-02T00:00:00Z")
    for index in range(1442):
        row = _features([str(start + pd.Timedelta(hours=index))]).iloc[0].to_dict()
        row["alt_log_variation_dispersion"] = 1 + index
        row["btc_variation"] = 2 + index
        if index == 100:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    assert np.isnan(features.dispersion_rank.iloc[1440])
    assert features.dispersion_rank.iloc[1441] == 1.0
    assert features.btc_variation_rank.iloc[1441] == 1.0


def test_majority_confirmation_and_source_valid_onset() -> None:
    features = _features()
    features["dispersion_rank"] = [0.2, 0.1, 0.1, 0.2]
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, True, False]
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]

    features.loc[1, ["alt_positive_count", "alt_negative_count", "alt_majority_side"]] = [3, 3, 0]
    assert not bool(support.active_and_side(features)[0].iloc[1])
    features.loc[1, ["alt_positive_count", "alt_negative_count", "alt_majority_side"]] = [4, 2, 1]
    features.loc[1, "btc_final_hour_return"] = -0.01
    assert not bool(support.active_and_side(features)[0].iloc[1])
    features.loc[0, "source_valid"] = False
    features.loc[1, "btc_final_hour_return"] = 0.01
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_all_controls_are_diagnostic_and_isolated() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features.loc[:, ["dispersion_rank", "btc_variation_rank"]] = [0.2, 0.6]
    assert not support.active_and_side(features)[0].any()
    assert support.active_and_side(features, "no_dispersion_compression")[0].tolist() == [False, False]
    assert support.active_and_side(features, "no_btc_variation_gate")[0].tolist() == [False, False]
    features.loc[:, "dispersion_rank"] = 0.1
    assert support.active_and_side(features, "no_btc_variation_gate")[0].all()
    features.loc[:, "btc_variation_rank"] = 0.7
    features.loc[:, "alt_majority_side"] = 0
    assert support.active_and_side(features, "btc_direction_only")[0].all()
    features.loc[:, "alt_majority_side"] = 1
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1]
    features.loc[0, "btc_final_hour_return"] = -0.01
    _, _, _, stale = support.active_and_side(features, "one_hour_stale_features")
    assert stale.loc[1, "btc_final_hour_return"] == -0.01
    assert stale.loc[1, "feature_available_time"] == features.loc[0, "decision_time"]


def test_clock_uses_global_half_open_reservation_and_skips_split_crossing() -> None:
    features = _features([
        "2023-12-31T22:00:00Z", "2023-12-31T23:00:00Z",
        "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
        "2024-01-01T06:00:00Z", "2024-01-01T07:00:00Z",
        "2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z",
    ])
    features["dispersion_rank"] = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]
    clock = support.build_clock(features)
    assert clock.decision_time.tolist() == pd.to_datetime([
        "2024-01-01T01:00:00Z", "2024-01-01T07:00:00Z"
    ]).tolist()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert set(clock.split) == {"test"}


def test_deterministic_immutable_artifacts(tmp_path: Path) -> None:
    frame = _features().iloc[:2]
    first = support.deterministic_csv_gzip(frame)
    assert first == support.deterministic_csv_gzip(frame.copy())
    assert gzip.decompress(first).startswith(b"decision_time,feature_available_time")
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, first)
    with pytest.raises(RuntimeError, match="immutable HVAVDCR artifact"):
        support.write_immutable(path, first + b"drift")
