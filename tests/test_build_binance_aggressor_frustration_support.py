from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.aggressor_frustration import BAR_COLUMNS
from training import build_binance_aggressor_frustration_support as support


def _write_fixture(tmp_path: Path, *, corrupt_score: bool = False) -> support.SupportConfig:
    dates = pd.date_range("2020-01-01", periods=2_200, freq="5min")
    quote = np.full(len(dates), 1_000.0)
    buy_frustrated = np.linspace(100.0, 300.0, len(dates))
    sell_frustrated = np.linspace(300.0, 100.0, len(dates))
    score = (sell_frustrated - buy_frustrated) / quote
    if corrupt_score:
        score[100] += 0.1
    features = pd.DataFrame(
        {
            "date": dates,
            "first_transact_time_ms": dates.astype("int64") // 1_000_000,
            "last_transact_time_ms": dates.astype("int64") // 1_000_000 + 1,
            "first_agg_trade_id": np.arange(len(dates)) * 2,
            "last_agg_trade_id": np.arange(len(dates)) * 2 + 1,
            "agg_trade_count": np.full(len(dates), 2),
            "classified_tick_count": np.full(len(dates), 2),
            "unavailable_tick_count": np.zeros(len(dates)),
            "state_reset_count": np.zeros(len(dates)),
            "quote_notional": quote,
            "classified_quote_notional": quote,
            "buy_quote_notional": np.full(len(dates), 500.0),
            "sell_quote_notional": np.full(len(dates), 500.0),
            "signed_quote_notional": np.zeros(len(dates)),
            "up_tick_notional": np.full(len(dates), 500.0),
            "down_tick_notional": np.full(len(dates), 500.0),
            "carried_zero_up_notional": np.zeros(len(dates)),
            "carried_zero_down_notional": np.zeros(len(dates)),
            "strict_buy_frustrated_notional": buy_frustrated,
            "strict_sell_frustrated_notional": sell_frustrated,
            "carried_buy_frustrated_notional": np.zeros(len(dates)),
            "carried_sell_frustrated_notional": np.zeros(len(dates)),
            "buy_frustrated_notional": buy_frustrated,
            "sell_frustrated_notional": sell_frustrated,
            "frustrated_notional_share": (buy_frustrated + sell_frustrated) / quote,
            "frustration_score": score,
            "tick_notional_imbalance": np.zeros(len(dates)),
        }
    ).loc[:, BAR_COLUMNS]
    features_path = tmp_path / "features.csv.gz"
    features.to_csv(features_path, index=False, compression="gzip", float_format="%.12g")
    feature_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()
    manifest = {
        "protocol": {"outcomes_opened": False},
        "combined_sha256": feature_hash,
        "rows": len(features),
        "columns": list(features.columns),
        "months": [
            {
                "month": "2020-01",
                "requested_dates": ["2020-01-01"],
                "warmup": {"date": "2019-12-31", "status": "unavailable"},
                "output": str(features_path),
                "output_sha256": feature_hash,
                "rows": len(features),
                "archives": [
                    {
                        "date": "2020-01-01",
                        "archive_sha256": "0" * 64,
                        "first_agg_trade_id": 0,
                        "last_agg_trade_id": int(features["last_agg_trade_id"].iloc[-1]),
                        "state_reset_count": 0,
                        "state_in": {
                            "previous_price": None,
                            "last_nonzero_tick": 0,
                            "previous_agg_trade_id": None,
                        },
                        "state_out": {
                            "previous_price": 100.0,
                            "last_nonzero_tick": 1,
                            "previous_agg_trade_id": int(
                                features["last_agg_trade_id"].iloc[-1]
                            ),
                        },
                    }
                ]
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    market_path = tmp_path / "market.csv.gz"
    pd.DataFrame({"date": dates, "open": np.arange(len(dates))}).to_csv(
        market_path, index=False, compression="gzip"
    )
    return support.SupportConfig(
        features=str(features_path),
        feature_manifest=str(manifest_path),
        market=str(market_path),
        clock_output=str(tmp_path / "clock.csv"),
        result_output=str(tmp_path / "support.json"),
    )


def test_prior_clean_quantile_counts_clean_observations_not_calendar_rows() -> None:
    values = pd.Series([1.0, 100.0, 2.0, 3.0, 4.0])
    clean = pd.Series([True, False, True, True, True])
    threshold = support.prior_clean_quantile(
        values, clean, quantile=0.5, window=3, min_periods=2
    )
    assert np.isnan(threshold.iloc[1])
    assert threshold.iloc[3] == 1.5
    assert threshold.iloc[4] == 2.0


def test_nonoverlap_schedule_allows_reentry_at_prior_exit() -> None:
    rows = 2_100
    dates = pd.date_range("2020-01-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "frustration_score": np.zeros(rows),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )
    # With the fixed 2,016-observation warmup, place signals 24 bars apart so
    # their entries are exactly one prior exit apart.
    frame.loc[2_016, "frustration_score"] = 10.0
    frame.loc[2_040, "frustration_score"] = -10.0
    clock, _ = support.build_schedule(frame)
    assert len(clock) == 2
    assert clock.loc[1, "entry_position"] == clock.loc[0, "exit_position"]


def test_run_support_parses_no_market_outcome_columns(tmp_path: Path) -> None:
    result = support.run_support(_write_fixture(tmp_path))
    assert result["outcomes_opened"] is False
    assert result["source"]["market_columns_loaded"] == ["date"]
    assert result["source"]["price_or_outcome_columns_loaded"] == []
    assert result["source_checks"]["frustration_score_identity"] is True
    assert result["passed"] is False


def test_corrupt_frustration_score_fails_before_clock_build(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frustration_score_identity"):
        support.run_support(_write_fixture(tmp_path, corrupt_score=True))


def test_broken_manifest_state_chain_fails_closed(tmp_path: Path) -> None:
    cfg = _write_fixture(tmp_path)
    path = Path(cfg.feature_manifest)
    manifest = json.loads(path.read_text())
    manifest["months"][0]["archives"][0]["state_in"]["previous_agg_trade_id"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="month_warmup_chain"):
        support.run_support(cfg)


def test_support_metrics_enforce_temporal_and_side_breadth() -> None:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2024):
        dates = pd.date_range(f"{year}-01-01", periods=70, freq="5D")
        for index, timestamp in enumerate(dates):
            rows.append(
                {
                    "entry_date": str(timestamp),
                    "side": 1 if index % 2 == 0 else -1,
                }
            )
    metrics = support.support_metrics(pd.DataFrame(rows))
    assert metrics["total"] == 280
    assert metrics["passes"] is True
