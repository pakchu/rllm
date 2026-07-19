from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import preregister_eth_btc_liquidation_relay as prereg


BTC_SOURCE = Path(
    "data/binance_coinm_liquidation_snapshot_btc_2023_2024/"
    "BTCUSD_PERP_liquidation_5m_2023-06-25_2024-10-14.csv.gz"
)
BTC_MANIFEST = Path("results/binance_coinm_liquidation_snapshot_btc_2023_2024_manifest.json")
BTC_SOURCE_SHA256 = "a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3"
BTC_MANIFEST_SHA256 = "5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e"
ETH_SOURCE = Path(
    "data/binance_coinm_liquidation_snapshot_eth_2023_2024/"
    "ETHUSD_PERP_liquidation_5m_2023-06-25_2024-10-14.csv.gz"
)
ETH_MANIFEST = Path("results/binance_coinm_liquidation_snapshot_eth_2023_2024_manifest.json")
ETH_SOURCE_SHA256 = "8d17ab3d5f9592f5254fef2e649065233be1777b8976983b4af38c77a8cc5bff"
ETH_MANIFEST_SHA256 = "c515731a9029d1786c8650f5106923d4cfbe8c35ed7a947f5420a16154601f5d"
CLBR_CLOCKS = Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz")
ICLA_CLOCKS = Path("data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_constant(name: str) -> Any:
    constants = getattr(prereg, "constants", None)
    if isinstance(constants, dict) and name in constants:
        return constants[name]
    if constants is not None and hasattr(constants, name):
        return getattr(constants, name)
    return getattr(prereg, name)


def _sources(rows: int = 8_500) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-06-25", periods=rows, freq="5min")
    btc = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "source_valid": np.ones(rows, dtype=bool),
            "event_count": np.ones(rows, dtype=int),
            "short_liquidation_usd": np.full(rows, 50.0),
            "long_liquidation_usd": np.full(rows, 50.0),
            "total_liquidation_usd": np.full(rows, 100.0),
            "signed_liquidation_usd": np.zeros(rows),
            "liquidation_imbalance": np.zeros(rows),
        }
    )
    eth = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "source_valid": np.ones(rows, dtype=bool),
            "event_count": np.ones(rows, dtype=int),
            "short_liquidation_event_count": np.ones(rows, dtype=int),
            "long_liquidation_event_count": np.zeros(rows, dtype=int),
            "short_liquidation_contracts": np.full(rows, 50.0),
            "long_liquidation_contracts": np.full(rows, 50.0),
            "total_liquidation_contracts": np.full(rows, 100.0),
            "signed_liquidation_contracts": np.zeros(rows),
            "liquidation_imbalance": np.zeros(rows),
        }
    )
    return btc, eth


def _inject_eth_relay_event(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    end_index: int,
    *,
    eth_imbalance: float = 0.75,
    btc_total_per_bar: float = 40.0,
    eth_event_count: int = 3,
) -> None:
    start = end_index - 11
    btc.loc[start:end_index, "total_liquidation_usd"] = btc_total_per_bar
    btc.loc[start:end_index, "short_liquidation_usd"] = btc_total_per_bar / 2.0
    btc.loc[start:end_index, "long_liquidation_usd"] = btc_total_per_bar / 2.0
    eth.loc[start:end_index, "event_count"] = 0
    eth.loc[start : start + eth_event_count - 1, "event_count"] = 1
    eth.loc[start:end_index, "total_liquidation_contracts"] = 200.0
    eth.loc[start:end_index, "signed_liquidation_contracts"] = 200.0 * eth_imbalance
    eth.loc[start:end_index, "liquidation_imbalance"] = eth_imbalance
    eth.loc[start:end_index, "short_liquidation_contracts"] = 100.0 * (1.0 + eth_imbalance)
    eth.loc[start:end_index, "long_liquidation_contracts"] = 100.0 * (1.0 - eth_imbalance)


def _tiny_clock(entry_minutes: list[int], *, candidate: str = "BASE") -> pd.DataFrame:
    start = pd.Timestamp("2023-06-25")
    return pd.DataFrame(
        {
            "candidate": candidate,
            "entry_time": [start + pd.Timedelta(minutes=m) for m in entry_minutes],
            "planned_exit_time": [start + pd.Timedelta(minutes=m + 30) for m in entry_minutes],
        }
    )


def test_config_binds_frozen_coinm_sources_and_support_constants() -> None:
    assert _sha256(BTC_SOURCE) == BTC_SOURCE_SHA256
    assert _sha256(BTC_MANIFEST) == BTC_MANIFEST_SHA256
    assert _sha256(ETH_SOURCE) == ETH_SOURCE_SHA256
    assert _sha256(ETH_MANIFEST) == ETH_MANIFEST_SHA256

    cfg = prereg.Config()
    assert Path(cfg.btc_source_path) == BTC_SOURCE
    assert Path(cfg.btc_manifest_path) == BTC_MANIFEST
    assert Path(cfg.eth_source_path) == ETH_SOURCE
    assert Path(cfg.eth_manifest_path) == ETH_MANIFEST
    assert cfg.expected_btc_source_sha256 == BTC_SOURCE_SHA256
    assert cfg.expected_btc_manifest_sha256 == BTC_MANIFEST_SHA256
    assert cfg.expected_eth_source_sha256 == ETH_SOURCE_SHA256
    assert cfg.expected_eth_manifest_sha256 == ETH_MANIFEST_SHA256

    assert _get_constant("CANDIDATE") == "EBLR-60/30"
    assert _get_constant("WAVE_BARS") == 12
    assert _get_constant("LOOKBACK_DAYS") == 28
    assert _get_constant("MIN_POSITIVE_WINDOWS") == 300
    assert _get_constant("BTC_QUIET_SEVERITY_MAX") == 0.50
    assert _get_constant("ETH_SEVERITY_MIN") == 1.0
    assert _get_constant("ETH_MIN_EVENT_COUNT") == 3
    assert _get_constant("ETH_ABS_IMBALANCE_MIN") == 0.70
    assert _get_constant("HOLD_BARS") == 6
    assert _get_constant("MAX_ENTRY_JACCARD") == 0.10
    assert _get_constant("SPLITS") == {
        "train": ("2023-06-25", "2023-10-15"),
        "test": ("2023-10-15", "2024-04-15"),
        "eval": ("2024-04-15", "2024-10-15"),
    }
    assert _get_constant("SUPPORT_MIN_TOTAL") == {"train": 20, "test": 50, "eval": 50}
    assert _get_constant("SUPPORT_MIN_PER_SIDE") == {"train": 6, "test": 12, "eval": 12}
    assert _get_constant("SUPPORT_MAX_MONTH_SHARE") == {
        "train": 0.40,
        "test": 0.30,
        "eval": 0.30,
    }


def test_load_sources_verifies_hashes_and_never_loads_market_outcomes() -> None:
    btc, eth = prereg.load_sources(prereg.Config())
    assert set(btc.columns).isdisjoint({"open", "high", "low", "close", "return", "pnl"})
    assert set(eth.columns).isdisjoint({"open", "high", "low", "close", "return", "pnl"})
    assert btc["date"].min() == pd.Timestamp("2023-06-25")
    assert eth["date"].min() == pd.Timestamp("2023-06-25")

    with pytest.raises(ValueError, match="BTC.*sha256|sha256.*BTC"):
        prereg.load_sources(replace(prereg.Config(), expected_btc_source_sha256="0" * 64))
    with pytest.raises(ValueError, match="ETH.*sha256|sha256.*ETH"):
        prereg.load_sources(replace(prereg.Config(), expected_eth_source_sha256="0" * 64))


def test_thresholds_are_strictly_prior_and_prefix_independent() -> None:
    btc, eth = _sources()
    index = 8_100
    _inject_eth_relay_event(btc, eth, index)
    state = prereg.derive_relay_state(btc, eth)
    assert bool(state.loc[index, "is_candidate"])
    assert state.loc[index, "eth_prior_q95"] == pytest.approx(1_200.0)
    assert state.loc[index, "btc_prior_q95"] == pytest.approx(1_200.0)

    current_mutation_btc = btc.copy()
    current_mutation_eth = eth.copy()
    current_mutation_btc.loc[index, "total_liquidation_usd"] *= 1_000.0
    current_mutation_eth.loc[index, "total_liquidation_contracts"] *= 1_000.0
    current_mutation_eth.loc[index, "signed_liquidation_contracts"] *= 1_000.0
    replay = prereg.derive_relay_state(current_mutation_btc, current_mutation_eth)
    assert replay.loc[index, "eth_prior_q95"] == state.loc[index, "eth_prior_q95"]
    assert replay.loc[index, "btc_prior_q95"] == state.loc[index, "btc_prior_q95"]

    future_mutation_btc = btc.copy()
    future_mutation_eth = eth.copy()
    future_mutation_btc.loc[index + 1 :, "total_liquidation_usd"] = 1e12
    future_mutation_eth.loc[index + 1 :, "total_liquidation_contracts"] = 1e12
    future_replay = prereg.derive_relay_state(future_mutation_btc, future_mutation_eth)
    pd.testing.assert_series_equal(
        state.loc[:index, "eth_prior_q95"],
        future_replay.loc[:index, "eth_prior_q95"],
    )
    pd.testing.assert_series_equal(
        state.loc[:index, "btc_prior_q95"],
        future_replay.loc[:index, "btc_prior_q95"],
    )


def test_thresholds_require_a_full_28_calendar_day_reference() -> None:
    btc, eth = _sources()
    state = prereg.derive_relay_state(btc, eth)

    assert pd.isna(state.loc[prereg.LOOKBACK_BARS - 1, "eth_prior_q95"])
    assert pd.isna(state.loc[prereg.LOOKBACK_BARS - 1, "btc_prior_q95"])
    assert state.loc[prereg.LOOKBACK_BARS, "eth_prior_q95"] == pytest.approx(1_200.0)
    assert state.loc[prereg.LOOKBACK_BARS, "btc_prior_q95"] == pytest.approx(1_200.0)


def test_missing_bar_in_either_source_invalidates_whole_12_bar_state() -> None:
    btc, eth = _sources()
    index = 8_100
    _inject_eth_relay_event(btc, eth, index)
    btc.loc[index - 4, "source_valid"] = False
    state = prereg.derive_relay_state(btc, eth)
    assert not bool(state.loc[index, "wave_source_valid"])
    assert not bool(state.loc[index, "is_candidate"])

    btc.loc[index - 4, "source_valid"] = True
    eth.loc[index - 7, "source_valid"] = False
    state = prereg.derive_relay_state(btc, eth)
    assert not bool(state.loc[index, "wave_source_valid"])
    assert not bool(state.loc[index, "is_candidate"])


def test_direction_follows_eth_forced_flow_and_btc_quiet_gate_rejects() -> None:
    btc, eth = _sources()
    long_index = 8_100
    short_index = 8_140
    loud_btc_index = 8_180
    _inject_eth_relay_event(btc, eth, long_index, eth_imbalance=0.75)
    _inject_eth_relay_event(btc, eth, short_index, eth_imbalance=-0.75)
    _inject_eth_relay_event(btc, eth, loud_btc_index, btc_total_per_bar=70.0)
    state = prereg.derive_relay_state(btc, eth)

    assert bool(state.loc[long_index, "is_candidate"])
    assert int(state.loc[long_index, "direction"]) == 1
    assert bool(state.loc[short_index, "is_candidate"])
    assert int(state.loc[short_index, "direction"]) == -1
    assert state.loc[loud_btc_index, "btc_quiet_severity"] > 0.50
    assert not bool(state.loc[loud_btc_index, "is_candidate"])


def test_latency_hold_and_nonoverlap_are_enforced_per_split(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2023-06-25", periods=80, freq="5min")
    state = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "wave_source_valid": np.ones(len(dates), dtype=bool),
            "is_candidate": np.zeros(len(dates), dtype=bool),
            "direction": np.ones(len(dates), dtype=int),
            "eth_event_count_60m": np.full(len(dates), 3),
            "eth_wave_total": np.full(len(dates), 2_400.0),
            "eth_prior_q95": np.full(len(dates), 1_200.0),
            "eth_severity": np.full(len(dates), 2.0),
            "eth_wave_imbalance": np.full(len(dates), 0.75),
            "btc_wave_total": np.full(len(dates), 480.0),
            "btc_prior_q95": np.full(len(dates), 1_200.0),
            "btc_quiet_severity": np.full(len(dates), 0.40),
        }
    )
    state.loc[[20, 21, 26, 27], "is_candidate"] = True
    state.loc[26, "direction"] = -1
    monkeypatch.setattr(prereg, "SPLITS", {"train": ("2023-06-25", "2023-06-26")})

    clocks = prereg.build_clocks(state)
    assert clocks["last_bar_open_time"].tolist() == [dates[20], dates[26]]
    assert clocks["feature_available_time"].tolist() == [
        dates[20] + pd.Timedelta(minutes=5, seconds=1),
        dates[26] + pd.Timedelta(minutes=5, seconds=1),
    ]
    assert clocks["entry_time"].tolist() == [
        dates[20] + pd.Timedelta(minutes=10),
        dates[26] + pd.Timedelta(minutes=10),
    ]
    assert bool(
        clocks["planned_exit_time"].sub(clocks["entry_time"]).eq(pd.Timedelta(minutes=30)).all()
    )
    assert clocks["entry_time"].iloc[1] >= clocks["planned_exit_time"].iloc[0]
    assert clocks["direction"].tolist() == [1, -1]


def test_build_result_contract_records_support_orthogonality_and_no_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = _tiny_clock([0, 30, 60, 90, 120, 150, 180, 210], candidate="EBLR-60/30")
    clocks["split"] = ["train"] * 2 + ["test"] * 3 + ["eval"] * 3
    clocks["direction"] = [1, -1, 1, -1, 1, -1, 1, -1]
    clbr_path = tmp_path / "clbr.csv.gz"
    icla_path = tmp_path / "icla.csv.gz"
    _tiny_clock([999, 2_997], candidate="CLBR-24").to_csv(
        clbr_path, index=False, compression="gzip"
    )
    _tiny_clock([999, 1_998], candidate="ICLA-60").to_csv(icla_path, index=False, compression="gzip")

    monkeypatch.setattr(prereg, "load_sources", lambda _cfg: _sources(rows=80))
    monkeypatch.setattr(prereg, "derive_relay_state", lambda _btc, _eth: pd.DataFrame())
    monkeypatch.setattr(prereg, "build_clocks", lambda _state: clocks)
    monkeypatch.setattr(prereg, "SUPPORT_MIN_TOTAL", {"train": 2, "test": 3, "eval": 3})
    monkeypatch.setattr(prereg, "SUPPORT_MIN_PER_SIDE", {"train": 1, "test": 1, "eval": 1})
    monkeypatch.setattr(prereg, "SUPPORT_MAX_MONTH_SHARE", {"train": 1.0, "test": 1.0, "eval": 1.0})

    cfg = replace(
        prereg.Config(),
        output_clock_path=str(tmp_path / "eblr.csv.gz"),
        output_result_path=str(tmp_path / "result.json"),
        clbr_clock_path=str(clbr_path),
        icla_clock_path=str(icla_path),
    )
    result = prereg.build(cfg)

    assert result["protocol"]["candidate"] == "EBLR-60/30"
    assert result["protocol"]["candidate_count"] == 1
    assert result["protocol"]["outcomes_opened"] is False
    assert result["protocol"]["market_prices_opened"] is False
    assert result["protocol"]["return_labels_constructed"] is False
    assert result["sources"]["btc"]["sha256"] == BTC_SOURCE_SHA256
    assert result["sources"]["eth"]["sha256"] == ETH_SOURCE_SHA256
    assert result["support"]["passes"] is True
    assert result["support"]["train"]["minimum_required"] == 2
    assert result["support"]["test"]["minimum_required"] == 3
    assert result["support"]["eval"]["minimum_required"] == 3
    assert set(result["clock_overlap"]) == {"CLBR", "ICLA", "maximum_entry_jaccard_allowed", "passes"}
    assert result["clock_overlap"]["maximum_entry_jaccard_allowed"] == 0.10
    assert result["clock_overlap"]["CLBR"]["path"] == str(clbr_path)
    assert result["clock_overlap"]["ICLA"]["path"] == str(icla_path)
    assert result["clock_overlap"]["CLBR"]["entry_jaccard"] <= 0.10
    assert result["clock_overlap"]["ICLA"]["entry_jaccard"] <= 0.10
    assert result["clock_overlap"]["passes"] is True

    written = json.loads(Path(cfg.output_result_path).read_text())
    core = {key: value for key, value in written.items() if key != "manifest_hash"}
    assert written["manifest_hash"] == prereg.canonical_hash(core)
    emitted = pd.read_csv(cfg.output_clock_path, compression="gzip")
    assert set(emitted.columns).isdisjoint({"open", "high", "low", "close", "return", "pnl"})
