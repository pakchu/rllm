"""Strict standalone BTC alpha from causally available Deribit DVOL candles."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import WINDOWS, _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


FIT_START = "2021-04-15"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"


@dataclass(frozen=True)
class DvolSearchConfig:
    input_csv: str
    dvol_csv: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    dvol_tolerance: str = "65min"
    min_select_trades: int = 24
    min_half_trades: int = 8


def attach_dvol(market: pd.DataFrame, dvol: pd.DataFrame, *, tolerance: str) -> pd.DataFrame:
    left = market.copy()
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="raise").dt.tz_convert(None)
    left = left.sort_values("date").reset_index(drop=True)
    right = dvol.copy()
    right["close_time"] = pd.to_datetime(right["close_time"], utc=True, errors="raise").dt.tz_convert(None)
    right = right.rename(columns={"open": "dvol_open", "high": "dvol_high", "low": "dvol_low", "close": "dvol_close"})
    joined = pd.merge_asof(
        left,
        right[["close_time", "dvol_open", "dvol_high", "dvol_low", "dvol_close"]].sort_values("close_time"),
        left_on="date",
        right_on="close_time",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    valid = joined["close_time"].notna()
    if (joined.loc[valid, "close_time"] > joined.loc[valid, "date"]).any():
        raise RuntimeError("DVOL candle was visible before close_time")
    joined["dvol_available"] = joined["dvol_close"].notna().astype(float)
    return joined


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(24, window // 2)).mean()
    std = series.rolling(window, min_periods=max(24, window // 2)).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def build_dvol_features(market: pd.DataFrame) -> pd.DataFrame:
    dvol = pd.to_numeric(market["dvol_close"], errors="coerce")
    columns: dict[str, pd.Series] = {}
    for window in (2016, 8640, 25920):
        columns[f"dvol_z{window}"] = _rolling_z(dvol, window)
        columns[f"dvol_logchg{window}"] = np.log(dvol / dvol.shift(window))
    columns["dvol_available"] = market["dvol_available"].astype(float)
    return pd.DataFrame(columns, index=market.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _load(cfg: DvolSearchConfig, *, cutoff: str | None = None) -> pd.DataFrame:
    if cutoff is None:
        market = pd.read_csv(cfg.input_csv, compression="infer")
        dvol = pd.read_csv(cfg.dvol_csv, compression="infer")
    else:
        market = _read_before(cfg.input_csv, "date", cutoff)
        dvol = _read_before(cfg.dvol_csv, "close_time", cutoff)
    return attach_dvol(market, dvol, tolerance=cfg.dvol_tolerance)


def _mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _tail_masks(values: np.ndarray, lower: float, upper: float, side: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    long_signal = finite & (values >= upper) & (side in {"both", "long"})
    short_signal = finite & (values <= lower) & (side in {"both", "short"})
    return long_signal, short_signal


def _signal_hash(long_signal: np.ndarray, short_signal: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(long_signal, np.uint8).tobytes() + np.asarray(short_signal, np.uint8).tobytes()).hexdigest()


def run(cfg: DvolSearchConfig) -> dict[str, Any]:
    phase1_market = _load(cfg, cutoff=SELECTION_END)
    phase1_dates = pd.to_datetime(phase1_market["date"])
    phase1_features = build_dvol_features(phase1_market)
    feature_hash = _feature_hash(phase1_features, phase1_dates)
    fit_mask = _mask(phase1_dates, FIT_START, FIT_END)
    specs: list[dict[str, Any]] = []
    for feature in ("dvol_logchg2016", "dvol_z8640", "dvol_logchg8640", "dvol_logchg25920", "dvol_z25920"):
        values = phase1_features[feature].to_numpy(float)
        reference = values[fit_mask & np.isfinite(values)]
        for tail in (0.05, 0.10, 0.15, 0.20, 0.25):
            specs.append({"feature": feature, "tail": tail, "lower": float(np.quantile(reference, tail)), "upper": float(np.quantile(reference, 1.0 - tail))})
    extremes = {
        hold: (
            _future_extreme(phase1_market["low"].to_numpy(float), hold, "min"),
            _future_extreme(phase1_market["high"].to_numpy(float), hold, "max"),
        )
        for hold in (144, 288, 576)
    }
    candidates = []
    signal_cache: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        values = phase1_features[spec["feature"]].to_numpy(float)
        for side in ("both", "long", "short"):
            long_signal, short_signal = _tail_masks(values, spec["lower"], spec["upper"], side)
            signal_cache[(spec["feature"], spec["tail"], side)] = (long_signal, short_signal)
            for hold, stride in itertools.product((144, 288, 576), (6, 12, 24)):
                kwargs = dict(hold_bars=hold, stride_bars=stride, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, extremes=extremes[hold])
                selection = _simulate_no_stop(phase1_market, phase1_dates, long_signal, short_signal, window="select2023", **kwargs)
                halves = [_simulate_no_stop(phase1_market, phase1_dates, long_signal, short_signal, window=name, **kwargs) for name in ("select2023_h1", "select2023_h2")]
                if selection["trades"] < cfg.min_select_trades or min(half["trades"] for half in halves) < cfg.min_half_trades:
                    continue
                candidates.append({**spec, "side": side, "hold_bars": hold, "stride_bars": stride, "select2023": selection, "select2023_halves": halves, "selection_score": {"positive_halves": sum(half["return_pct"] > 0 for half in halves), "min_half_ratio": min(half["ratio"] for half in halves), "full_ratio": selection["ratio"]}})
    candidates.sort(key=lambda row: (row["selection_score"]["positive_halves"], row["selection_score"]["min_half_ratio"], row["selection_score"]["full_ratio"], row["select2023"]["return_pct"]), reverse=True)
    selected = candidates[: cfg.top_n]
    manifest_policies = [{key: row[key] for key in ("feature", "tail", "lower", "upper", "side", "hold_bars", "stride_bars")} for row in selected]
    manifest_json = json.dumps(manifest_policies, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "phase": "pre_future_manifest", "feature_hash": feature_hash, "manifest_hash": manifest_hash, "policies": manifest_policies, "phase1_signal_hashes": [_signal_hash(*signal_cache[(row["feature"], row["tail"], row["side"])]) for row in selected]}
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    full_market = _load(cfg)
    full_dates = pd.to_datetime(full_market["date"])
    full_features = build_dvol_features(full_market)
    prefix_dates = full_dates.iloc[: len(phase1_dates)].reset_index(drop=True)
    if not prefix_dates.equals(phase1_dates.reset_index(drop=True)) or _feature_hash(full_features.iloc[: len(phase1_features)], prefix_dates) != feature_hash:
        raise RuntimeError("DVOL pre-2024 feature prefix changed")
    extremes_full = {hold: (_future_extreme(full_market["low"].to_numpy(float), hold, "min"), _future_extreme(full_market["high"].to_numpy(float), hold, "max")) for hold in (144, 288, 576)}
    for row in selected:
        values = full_features[row["feature"]].to_numpy(float)
        long_signal, short_signal = _tail_masks(values, row["lower"], row["upper"], row["side"])
        if _signal_hash(long_signal[: len(phase1_market)], short_signal[: len(phase1_market)]) != manifest["phase1_signal_hashes"][selected.index(row)]:
            raise RuntimeError("DVOL phase1 signal prefix changed")
        for window in ("test2024", "eval2025", "ytd2026"):
            row[window] = _simulate_no_stop(full_market, full_dates, long_signal, short_signal, window=window, hold_bars=row["hold_bars"], stride_bars=row["stride_bars"], leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, extremes=extremes_full[row["hold_bars"]])
        row["passes_alpha_target"] = row["test2024"]["ratio"] >= 3 and row["eval2025"]["ratio"] >= 3 and row["test2024"]["trades"] >= 16 and row["eval2025"]["trades"] >= 16
        row["passes_live_target"] = row["passes_alpha_target"] and row["ytd2026"]["ratio"] >= 5 and row["ytd2026"]["trades"] >= 8 and row["ytd2026"]["return_pct"] > 0
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": {"fit": (FIT_START, FIT_END), "selection": WINDOWS["select2023"], "manifest_written_before_future": True, "dvol_availability": "hourly candle close_time backward-asof only", "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown"}, "tested": len(candidates), "manifest_hash": manifest_hash, "selected": selected, "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]], "live_qualifiers": [row for row in selected if row["passes_live_target"]]}
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> DvolSearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--dvol-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--top-n", type=int, default=DvolSearchConfig.top_n)
    parser.add_argument("--leverage", type=float, default=DvolSearchConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=DvolSearchConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=DvolSearchConfig.slippage_rate)
    parser.add_argument("--dvol-tolerance", default=DvolSearchConfig.dvol_tolerance)
    parser.add_argument("--min-select-trades", type=int, default=DvolSearchConfig.min_select_trades)
    parser.add_argument("--min-half-trades", type=int, default=DvolSearchConfig.min_half_trades)
    return DvolSearchConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"tested": report["tested"], "hash": report["manifest_hash"], "alpha": len(report["alpha_qualifiers"]), "live": len(report["live_qualifiers"]), "selected": report["selected"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
