from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from training import build_oi_enriched_cache as mod


def _write_market(path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01 00:00", periods=4, freq="5min"),
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.0, 1.1, 1.2, 1.3],
            "low": [1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_csv(path, index=False)


def _oi_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01 00:00", periods=4, freq="5min"),
            "open_interest": [100.0, 101.0, 102.0, 103.0],
            "open_interest_value": [1000.0, 1010.0, 1020.0, 1030.0],
            "cmc_circulating_supply": [19000.0, 19000.0, 19000.0, 19000.0],
        }
    )


def _patch_oi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_load_oi", lambda cfg, start, end: _oi_frame())


def _metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "symbol": "BTCUSDT",
        "count_toptrader_long_short_ratio": 1.1,
        "sum_toptrader_long_short_ratio": 1.2,
        "count_long_short_ratio": 1.3,
        "sum_taker_long_short_vol_ratio": 1.4,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_default_preserves_oi_only_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_oi(monkeypatch)
    input_csv = tmp_path / "market.csv"
    output_csv = tmp_path / "out.csv"
    _write_market(input_csv)

    report = mod.run(mod.OiEnrichConfig(input_csv=str(input_csv), output_csv=str(output_csv)))
    out = pd.read_csv(output_csv)

    assert "positioning_available" not in out.columns
    assert "positioning_metrics" not in report
    assert out["open_interest_available"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out["open_interest"].tolist() == [100.0, 101.0, 102.0, 103.0]


def test_default_oi_only_still_accepts_non_five_minute_market_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_oi(monkeypatch)
    input_csv = tmp_path / "market.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01 00:01:00", "2024-01-01 00:06:00"]),
            "close": [1.0, 1.1],
        }
    ).to_csv(input_csv, index=False)
    mod.run(mod.OiEnrichConfig(input_csv=str(input_csv), output_csv=str(output_csv)))
    assert len(pd.read_csv(output_csv)) == 2


def test_metrics_are_shifted_by_exactly_one_completed_source_bar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_oi(monkeypatch)
    input_csv = tmp_path / "market.csv"
    metrics_csv = tmp_path / "metrics.csv"
    output_csv = tmp_path / "out.csv"
    _write_market(input_csv)
    _metrics(
        [
            {"create_time": "2023-12-31 23:55:00", "count_toptrader_long_short_ratio": 10.0, "sum_toptrader_long_short_ratio": 20.0, "count_long_short_ratio": 30.0, "sum_taker_long_short_vol_ratio": 40.0},
            {"create_time": "2024-01-01 00:00:00", "count_toptrader_long_short_ratio": 11.0, "sum_toptrader_long_short_ratio": 21.0, "count_long_short_ratio": 31.0, "sum_taker_long_short_vol_ratio": 41.0},
            {"create_time": "2024-01-01 00:05:00", "count_toptrader_long_short_ratio": 12.0, "sum_toptrader_long_short_ratio": 22.0, "count_long_short_ratio": 32.0, "sum_taker_long_short_vol_ratio": 42.0},
            {"create_time": "2024-01-01 00:10:00", "count_toptrader_long_short_ratio": 13.0, "sum_toptrader_long_short_ratio": 23.0, "count_long_short_ratio": 33.0, "sum_taker_long_short_vol_ratio": 43.0},
        ]
    ).to_csv(metrics_csv, index=False)

    report = mod.run(
        mod.OiEnrichConfig(
            input_csv=str(input_csv), output_csv=str(output_csv), metrics_csv=str(metrics_csv)
        )
    )
    out = pd.read_csv(output_csv, parse_dates=["date", "positioning_source_time"])

    assert out["positioning_available"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out["positioning_age_minutes"].tolist() == [5.0, 5.0, 5.0, 5.0]
    assert out["count_toptrader_long_short_ratio"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert out["sum_toptrader_long_short_ratio"].tolist() == [20.0, 21.0, 22.0, 23.0]
    assert out["count_long_short_ratio"].tolist() == [30.0, 31.0, 32.0, 33.0]
    assert out["sum_taker_long_short_vol_ratio"].tolist() == [40.0, 41.0, 42.0, 43.0]
    assert out["positioning_source_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2023-12-31 23:55:00",
        "2024-01-01 00:00:00",
        "2024-01-01 00:05:00",
        "2024-01-01 00:10:00",
    ]
    assert report["positioning_metrics"]["metrics_sha256"] == hashlib.sha256(metrics_csv.read_bytes()).hexdigest()
    assert report["positioning_leakage_guard"]["exact_one_completed_source_bar_delay"] is True
    assert report["positioning_leakage_guard"]["ratio_forward_fill"] is False


def test_metrics_gaps_and_missing_ratios_are_unavailable_not_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_oi(monkeypatch)
    input_csv = tmp_path / "market.csv"
    metrics_csv = tmp_path / "metrics.csv"
    output_csv = tmp_path / "out.csv"
    _write_market(input_csv)
    _metrics(
        [
            {"create_time": "2023-12-31 23:55:00", "count_long_short_ratio": 9.9},
            {"create_time": "2024-01-01 00:05:00", "count_long_short_ratio": None},
            {"create_time": "2024-01-01 00:10:00", "count_long_short_ratio": 3.0},
        ]
    ).to_csv(metrics_csv, index=False)

    mod.run(
        mod.OiEnrichConfig(
            input_csv=str(input_csv), output_csv=str(output_csv), metrics_csv=str(metrics_csv)
        )
    )
    out = pd.read_csv(output_csv)

    assert out["positioning_available"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert out["positioning_gap"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out["positioning_missing_required"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert pd.isna(out.loc[1, "count_long_short_ratio"])
    assert pd.isna(out.loc[2, "count_long_short_ratio"])
    assert out.loc[3, "count_long_short_ratio"] == 3.0
    # OI keeps its legacy ffill/asof semantics independently from ratio columns.
    assert out["open_interest"].tolist() == [100.0, 101.0, 102.0, 103.0]


def test_metrics_validation_rejects_bad_symbol_duplicate_and_nonpositive(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _metrics(
        [
            {"create_time": "2024-01-01 00:00:00"},
            {"create_time": "2024-01-01 00:00:00"},
        ]
    ).to_csv(metrics_csv, index=False)
    cfg = mod.OiEnrichConfig(input_csv="unused", output_csv="unused", metrics_csv=str(metrics_csv))
    with pytest.raises(ValueError, match="duplicate timestamps"):
        mod._load_positioning_metrics(cfg)

    _metrics([{"create_time": "2024-01-01 00:00:00", "symbol": "ETHUSDT"}]).to_csv(metrics_csv, index=False)
    with pytest.raises(ValueError, match="symbol set"):
        mod._load_positioning_metrics(cfg)

    _metrics([{"create_time": "2024-01-01 00:00:00", "count_long_short_ratio": -1.0}]).to_csv(metrics_csv, index=False)
    with pytest.raises(ValueError, match="nonnegative finite"):
        mod._load_positioning_metrics(cfg)

    _metrics([{"create_time": "2024-01-01 00:00:00", "count_long_short_ratio": "bad"}]).to_csv(metrics_csv, index=False)
    with pytest.raises(ValueError, match="malformed numeric"):
        mod._load_positioning_metrics(cfg)

    _metrics([{"create_time": "2024-01-01 00:00:00", "count_long_short_ratio": 0.0}]).to_csv(metrics_csv, index=False)
    zero_metrics, zero_report = mod._load_positioning_metrics(cfg)
    assert pd.isna(zero_metrics.loc[0, "count_long_short_ratio"])
    assert zero_report["metrics_zero_as_missing"]["count_long_short_ratio"] == 1

    with pytest.raises(ValueError, match="5-minute grid"):
        mod._validate_five_minute_timestamps(
            pd.Series(pd.to_datetime(["2024-01-01 00:01:00"])),
            label="market date",
        )


def test_metrics_off_grid_rows_are_dropped_as_gaps_not_rounded(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _metrics(
        [
            {"create_time": "2024-01-01 00:00:00"},
            {"create_time": "2024-01-01 00:05:01"},
        ]
    ).to_csv(metrics_csv, index=False)
    metrics, report = mod._load_positioning_metrics(
        mod.OiEnrichConfig(
            input_csv="unused", output_csv="unused", metrics_csv=str(metrics_csv)
        )
    )
    assert metrics["create_time"].tolist() == [pd.Timestamp("2024-01-01 00:00:00")]
    assert report["metrics_off_grid_rows_dropped"] == 1
    assert report["metrics_off_grid_examples"] == ["2024-01-01 00:05:01"]
