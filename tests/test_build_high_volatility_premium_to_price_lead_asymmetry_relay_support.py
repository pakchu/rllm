import json
from pathlib import Path

import numpy as np
import pandas as pd

from training import build_high_volatility_premium_to_price_lead_asymmetry_relay_support as support


def _ohlc_from_returns(start: pd.Timestamp, returns: np.ndarray) -> pd.DataFrame:
    open_ = np.full(len(returns), 100.0)
    close = open_ * np.exp(returns)
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=len(returns), freq="1min"),
        "open": open_,
        "high": np.maximum(open_, close) * 1.001,
        "low": np.minimum(open_, close) * 0.999,
        "close": close,
        "duplicate_count": 1,
    })


def _premium_ohlc(start: pd.Timestamp, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=len(closes), freq="1min"),
        "open": closes,
        "high": closes + 0.01,
        "low": closes - 0.01,
        "close": closes,
        "duplicate_count": 1,
    })


def _feature_frame() -> pd.DataFrame:
    decisions = pd.to_datetime([
        "2024-07-01T00:00:00Z", "2024-07-01T08:00:00Z",
        "2024-07-01T16:00:00Z", "2024-07-02T00:00:00Z",
    ])
    return pd.DataFrame({
        "decision_time": decisions,
        "feature_available_time": decisions,
        "source_valid": [True] * 4,
        "premium_lead": [0.4] * 4,
        "price_lead": [0.1] * 4,
        "lead_advantage": [0.3] * 4,
        "premium_displacement": [0.2, -0.2, 0.2, -0.2],
        "abs_premium_displacement": [0.2] * 4,
        "btc_return": [0.03, -0.04, 0.05, -0.06],
        "direction_alignment": [True] * 4,
        "btc_realized_variation": [0.2] * 4,
        "lead_advantage_rank": [0.75] * 4,
        "premium_displacement_rank": [0.60] * 4,
        "variation_rank": [0.65] * 4,
    })


def test_average_tied_spearman_is_exact_and_rejects_constant_rank() -> None:
    left = np.array([1.0, 1.0, 3.0, 4.0])
    right = np.array([4.0, 3.0, 2.0, 1.0])
    expected = np.corrcoef(
        pd.Series(left).rank(method="average"),
        pd.Series(right).rank(method="average"),
    )[0, 1]
    assert np.isclose(support.average_tied_spearman(left, right), expected)
    assert np.isnan(support.average_tied_spearman(np.ones(4), right))


def test_strict_prior_midrank_uses_270_finite_prior_values_and_excludes_current() -> None:
    values = pd.Series([*range(270), np.nan, 270.0, 270.0])
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[:180]).all()
    assert np.isnan(ranks.iloc[270])
    assert ranks.iloc[271] == 1.0
    # The second 270 sees exactly the prior finite window [1, ..., 269, 270].
    assert ranks.iloc[272] == (269 + 0.5) / 270


def test_exact_aligned_block_cross_lags_and_24h_variation() -> None:
    rng = np.random.default_rng(8)
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    returns = np.zeros(1440)
    block_returns = rng.normal(0, 0.002, 480)
    block_returns[-1] = 0.004
    returns[-480:] = block_returns

    changes = np.empty(479)
    changes[:478] = block_returns[2:]
    changes[478] = -0.0007
    premium_close = np.concatenate(([0.0], np.cumsum(changes)))
    btc = support.prepare_ohlc(_ohlc_from_returns(start, returns), positive=True)
    premium_raw = _premium_ohlc(start + pd.Timedelta(hours=16), premium_close)
    premium = support.prepare_ohlc(premium_raw, positive=False)

    feature = support.boundary_features(btc, premium, decision)
    expected_price_lead = support.average_tied_spearman(block_returns[1:479], changes[1:])
    assert feature["source_valid"] is True
    assert np.isclose(feature["premium_lead"], 1.0)
    assert np.isclose(feature["price_lead"], expected_price_lead)
    assert np.isclose(feature["lead_advantage"], 1.0 - expected_price_lead)
    assert np.isclose(feature["premium_displacement"], changes.sum())
    assert np.isclose(feature["btc_return"], block_returns[-1])
    assert np.isclose(feature["btc_realized_variation"], np.sqrt(np.square(returns).sum()))
    assert feature["direction_alignment"] == (changes.sum() > 0)


def test_exact_alignment_and_signed_ohlc_coherence_are_mandatory() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    signed = _premium_ohlc(start, np.array([-0.2, -0.3]))
    prepared = support.prepare_ohlc(signed, positive=False)
    assert prepared.source_valid.tolist() == [True, True]
    signed.loc[1, "high"] = -0.4
    assert support.prepare_ohlc(signed, positive=False).source_valid.tolist() == [True, False]

    decision = start + pd.Timedelta(hours=24)
    btc = support.prepare_ohlc(_ohlc_from_returns(start, np.full(1440, 0.001)), positive=True)
    premium = _premium_ohlc(start + pd.Timedelta(hours=16), np.linspace(-1, 1, 480))
    premium = premium.drop(index=237).reset_index(drop=True)
    invalid = support.boundary_features(
        btc, support.prepare_ohlc(premium, positive=False), decision
    )
    assert invalid["source_valid"] is False
    assert np.isnan(invalid["premium_lead"])


def test_negative_premium_lead_remains_source_valid_but_is_not_eligible() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    block_returns = np.linspace(-0.003, 0.004, 480)
    returns = np.zeros(1440)
    returns[-480:] = block_returns
    changes = np.r_[-block_returns[2:], 0.001]
    premium_close = np.r_[0.0, np.cumsum(changes)]
    btc = support.prepare_ohlc(_ohlc_from_returns(start, returns), positive=True)
    premium = support.prepare_ohlc(
        _premium_ohlc(start + pd.Timedelta(hours=16), premium_close), positive=False
    )
    feature = support.boundary_features(btc, premium, decision)
    assert feature["source_valid"] is True
    assert np.isclose(feature["premium_lead"], -1.0)

    features = _feature_frame().iloc[[0]].copy()
    features.loc[:, "premium_lead"] = feature["premium_lead"]
    assert not bool(support.active_and_side(features)[0].iloc[0])


def test_primary_gates_side_and_frozen_controls() -> None:
    features = _feature_frame()
    features.loc[1, "lead_advantage_rank"] = 0.74
    features.loc[2, "premium_displacement_rank"] = 0.59
    features.loc[3, "variation_rank"] = 0.64
    primary, side, _ = support.active_and_side(features)
    assert primary.tolist() == [True, False, False, False]
    assert side.tolist() == [1, -1, 1, -1]
    assert support.active_and_side(
        features, "no_lead_advantage_gate"
    )[0].tolist() == [True, True, False, False]
    assert support.active_and_side(
        features, "no_premium_displacement_tail"
    )[0].tolist() == [True, False, True, False]
    assert support.active_and_side(
        features, "no_volatility_gate"
    )[0].tolist() == [True, False, False, True]
    assert support.active_and_side(features, "direction_flip")[1].tolist() == [-1, 1, -1, 1]
    assert support.CONTROLS == (
        "no_lead_advantage_gate", "no_premium_displacement_tail",
        "no_volatility_gate", "one_block_stale_premium", "direction_flip",
    )


def test_one_block_stale_premium_uses_exact_prior_block_but_current_btc() -> None:
    features = _feature_frame()
    features.loc[1, "premium_displacement"] = 0.2
    features.loc[1, "direction_alignment"] = False
    active, side, used = support.active_and_side(features, "one_block_stale_premium")
    assert active.tolist() == [False, False, True, False]
    assert used.loc[1, "premium_displacement"] == features.loc[0, "premium_displacement"]
    assert side.tolist() == [1, -1, 1, -1]

    features.loc[1, "btc_return"] = 0.04
    active, _, _ = support.active_and_side(features, "one_block_stale_premium")
    assert active.tolist() == [False, True, True, False]
    features.loc[2, "decision_time"] += pd.Timedelta(minutes=1)
    assert not bool(
        support.active_and_side(features, "one_block_stale_premium")[0].iloc[2]
    )


def test_clock_has_exact_entry_hold_split_and_half_open_reservation() -> None:
    clock = support.build_clock(_feature_frame())
    assert len(clock) == 4
    assert clock.side.tolist() == [1, -1, 1, -1]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=8)).all()
    # Every next open equals the prior exit and is therefore admitted.
    assert (
        clock.entry_time.iloc[1:].reset_index(drop=True)
        == clock.exit_time.iloc[:-1].reset_index(drop=True)
    ).all()
    assert set(clock.split) == {"test"}


def test_source_queries_are_frozen_source_only() -> None:
    queries = support.BTC_QUERY + support.PREMIUM_QUERY
    assert "FROM bars_binance\n" in support.BTC_QUERY
    assert "FROM bars_binance_premium\n" in support.PREMIUM_QUERY
    assert "funding_rate" not in queries.lower()
    assert "pnl" not in queries.lower()
    assert "gross9" not in queries.lower()
    assert support.PREREG_SHA == "f00247a72273e10336e9a769400dc43c9609d51882bcda151e2e3906091ea644"
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")


def test_run_writes_split_reservation_control_and_support_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "sources"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "FEATURES", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(support, "load_sources", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(support, "build_features", lambda _btc, _premium: _feature_frame())

    result = support.run()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert result["policy_id"] == "HVPPLA-8"
    assert result["reservation"] == {
        "scope": "global", "interval": "half_open",
        "equal_open_after_exit_allowed": True, "split_crossing_action": "skip",
    }
    assert set(result["support"]) == set(support.SPLITS)
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert manifest["tables"] == ["bars_binance", "bars_binance_premium"]
    assert manifest["funding_values_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
