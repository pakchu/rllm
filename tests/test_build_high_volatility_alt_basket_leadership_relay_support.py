import builtins
import gzip
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_alt_basket_leadership_relay_support as support
from training import preregister_high_volatility_alt_basket_leadership_relay as prereg


def _return_paths(periods: int = 360) -> dict[str, np.ndarray]:
    minute = np.arange(periods, dtype=float)
    alt = 0.0012 * np.sin(minute / 7.0) + 0.0004 * np.cos(minute / 17.0)
    btc = 0.0009 * np.sin((minute - 1.0) / 7.0) + 0.0002 * np.cos(minute / 11.0)
    paths = {"BTCUSDT": btc}
    for index, symbol in enumerate(support.ALTS):
        paths[symbol] = alt + (index - 2.5) * 0.00001
    return paths


def _bars(start: pd.Timestamp, periods: int = 360) -> pd.DataFrame:
    frames = []
    paths = _return_paths(periods)
    for symbol_index, symbol in enumerate(support.SYMBOLS):
        minute = np.arange(periods)
        opens = 100.0 + symbol_index + minute * 0.001
        closes = opens * np.exp(paths[symbol])
        frames.append(pd.DataFrame({
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "symbol": symbol, "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999, "close": closes,
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
            "alt_leads_btc": 0.4, "btc_leads_alt": 0.1,
            "leadership_advantage": 0.3, "contemporaneous_correlation": 0.2,
            "btc_variation": 0.01, "btc_final_hour_return": 0.01,
            "alt_final_hour_return": 0.02, "direction_side": 1,
            "leadership_rank": 0.9, "contemporaneous_correlation_rank": 0.9,
            "btc_variation_rank": 0.7,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=support.FEATURE_COLUMNS)


def test_preregistration_binding_source_only_query_and_lazy_sqlalchemy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert support.PREREG_SHA == "285d84214e03743e6120380016fbaf35cee4978fedb3d209359b4e91efd7cc76"
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith("SELECT ts,symbol,open,high,low,close FROM bars_binance")
    assert "interval='1m'" in normalized
    assert all(f"'{symbol}'" in normalized for symbol in support.SYMBOLS)
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    assert all(word not in support.QUERY.lower() for word in ("funding", "execution", "pnl", "gross9"))

    sys.modules.pop(support.__name__, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise AssertionError("SQLAlchemy imported at module load")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module(support.__name__)


def test_exact_aligned_leadership_variation_and_final_hour_sum_formulas() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    pair = support.boundary_pair(
        support.prepare_source(_bars(start)), start + pd.Timedelta(hours=6)
    )
    paths = _return_paths()
    btc = paths["BTCUSDT"]
    alt = np.median(np.column_stack([paths[symbol] for symbol in support.ALTS]), axis=1)
    alt_leads = np.corrcoef(alt[:-1], btc[1:])[0, 1]
    btc_leads = np.corrcoef(btc[:-1], alt[1:])[0, 1]
    assert pair["source_valid"] is True
    assert pair["minute_count"] == 360 * 7
    assert pair["alt_leads_btc"] == pytest.approx(alt_leads)
    assert pair["btc_leads_alt"] == pytest.approx(btc_leads)
    assert pair["leadership_advantage"] == pytest.approx(alt_leads - btc_leads)
    assert pair["contemporaneous_correlation"] == pytest.approx(np.corrcoef(alt, btc)[0, 1])
    assert pair["btc_variation"] == pytest.approx(np.square(btc).sum())
    assert pair["btc_final_hour_return"] == pytest.approx(btc[-60:].sum())
    assert pair["alt_final_hour_return"] == pytest.approx(alt[-60:].sum())
    expected_side = (
        int(np.sign(btc[-60:].sum()))
        if np.sign(btc[-60:].sum()) == np.sign(alt[-60:].sum())
        else 0
    )
    assert pair["direction_side"] == expected_side


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incoherent", "zero_variance"])
def test_source_and_correlation_variance_rules_fail_closed(mutation: str) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    raw = _bars(start)
    if mutation == "missing":
        raw = raw.drop(raw[raw.symbol.eq("ADAUSDT")].index[10])
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate source key"):
            support.prepare_source(raw)
        return
    elif mutation == "incoherent":
        index = raw[raw.symbol.eq("BNBUSDT")].index[20]
        raw.loc[index, "high"] = raw.loc[index, "low"] - 1
    else:
        indexes = raw[raw.symbol.eq("BTCUSDT")].index
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


def test_all_three_feature_ranks_use_only_source_valid_decisions() -> None:
    rows = []
    start = pd.Timestamp("2023-04-02T00:00:00Z")
    for index in range(1442):
        row = _features([str(start + pd.Timedelta(hours=index))]).iloc[0].to_dict()
        row["leadership_advantage"] = index + 1.0
        row["contemporaneous_correlation"] = index + 2.0
        row["btc_variation"] = index + 3.0
        if index == 100:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    for column in (
        "leadership_rank", "contemporaneous_correlation_rank", "btc_variation_rank"
    ):
        assert np.isnan(features[column].iloc[1440])
        assert features[column].iloc[1441] == 1.0


def test_primary_gates_strict_sign_and_source_valid_onset() -> None:
    features = _features()
    features["leadership_rank"] = [0.7, 0.8, 0.9, 0.7]
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, True, False]
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]

    features.loc[1, "direction_side"] = 0
    assert not bool(support.active_and_side(features)[0].iloc[1])
    features.loc[1, "direction_side"] = -1
    assert bool(support.active_and_side(features)[0].iloc[1])
    assert support.active_and_side(features)[2].iloc[1] == -1
    features.loc[0, "source_valid"] = False
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_all_controls_are_diagnostic_isolated_and_contemporaneous_has_own_rank() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features.loc[:, "leadership_rank"] = 0.7
    assert not support.active_and_side(features)[0].any()
    assert support.active_and_side(features, "no_leadership_gate")[0].all()
    features.loc[:, "btc_variation_rank"] = 0.6
    assert not support.active_and_side(features, "no_leadership_gate")[0].any()
    features.loc[:, "leadership_rank"] = 0.9
    assert support.active_and_side(features, "no_btc_variation_gate")[0].all()
    features.loc[:, "btc_variation_rank"] = 0.7
    features.loc[:, "contemporaneous_correlation_rank"] = [0.79, 0.80]
    assert support.active_and_side(features, "contemporaneous_correlation")[0].tolist() == [False, True]
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1]
    features.loc[0, "leadership_advantage"] = -0.4
    _, _, _, stale = support.active_and_side(features, "one_hour_stale_features")
    assert stale.loc[1, "leadership_advantage"] == -0.4
    assert stale.loc[1, "feature_available_time"] == features.loc[0, "decision_time"]


def test_clock_uses_global_half_open_reservation_and_skips_split_crossing() -> None:
    features = _features([
        "2023-12-31T22:00:00Z", "2023-12-31T23:00:00Z",
        "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
        "2024-01-01T06:00:00Z", "2024-01-01T07:00:00Z",
        "2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z",
    ])
    features["leadership_rank"] = [0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8]
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
    with pytest.raises(RuntimeError, match="immutable HVABLR artifact"):
        support.write_immutable(path, first + b"drift")
