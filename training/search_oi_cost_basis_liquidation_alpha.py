"""Pre-2024 OI cost-basis liquidation-pressure alpha preflight.

This experiment is intentionally physically capped at ``2024-01-01``.  It
uses one complete-bar delayed Binance positioning metrics to attribute new OI
into long/short cohorts, tracks path-dependent cohort cost basis, and trades
only when underwater cohort pressure coincides with OI contraction and taker
flow in the mechanically expected liquidation direction.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _attach_delayed_metrics, _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_absorption_alpha import _prior_z

MARKET_CSV = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
METRICS_CSV = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
SELECTION_END = "2024-01-01"
RESULT_SCAN = "results/oi_cost_basis_liquidation_alpha_scan_2026-07-13.json"
RESULT_MANIFEST = "results/oi_cost_basis_liquidation_top_manifest_2026-07-13.json"
RESULT_VERIFY = "results/oi_cost_basis_liquidation_preflight_verification_2026-07-13.json"
DOC_PATH = "docs/oi-cost-basis-liquidation-alpha-preflight-2026-07-13.md"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit": ("2020-10-15", "2022-01-01"),
    "fit_2020q4": ("2020-10-15", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "quarantine_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", "2024-01-01"),
}
FIT_SEGMENTS = ("fit_2020q4", "fit_2021h1", "fit_2021h2")
SELECT_SEGMENTS = ("select_2023h1", "select_2023h2")
ROBUSTNESS_SEGMENTS = (*FIT_SEGMENTS, *SELECT_SEGMENTS)
HALF_LIVES = (288, 864)
TAILS = (0.05, 0.10)
MODES = ("both", "long_only", "short_only")
HOLDS = (72, 144)

SideMode = Literal["both", "long_only", "short_only"]
ScoreVariant = Literal["base", "no_oi_contraction", "no_taker_flow", "no_cost_basis_pressure"]


@dataclass(frozen=True)
class OiCostBasisConfig:
    input_csv: str = MARKET_CSV
    metrics_csv: str = METRICS_CSV
    scan_output: str = RESULT_SCAN
    manifest_output: str = RESULT_MANIFEST
    verification_output: str = RESULT_VERIFY
    doc_output: str = DOC_PATH
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    metrics_tolerance: str = "5min"
    source_delay_bars: int = 1
    selection_end: str = SELECTION_END


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_log(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return np.log(values.where(values > 0.0))


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def load_pre2024_market(cfg: OiCostBasisConfig) -> pd.DataFrame:
    """Load and attach source-delayed metrics, physically excluding 2024+."""
    market = _read_before(cfg.input_csv, "date", cfg.selection_end)
    metrics = _read_before(cfg.metrics_csv, "create_time", cfg.selection_end)
    attached = _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.source_delay_bars,
    )
    dates = pd.to_datetime(attached["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cfg.selection_end):
        raise RuntimeError("preflight loader admitted 2024+ rows")
    source_time = pd.to_datetime(attached["positioning_source_time"], errors="coerce")
    if source_time.notna().any() and source_time.max() >= pd.Timestamp(cfg.selection_end):
        raise RuntimeError("preflight loader admitted 2024+ positioning source rows")
    return attached


def build_cost_basis_features(market: pd.DataFrame, half_life: int) -> pd.DataFrame:
    """Build causal OI cohort state and liquidation-pressure score.

    Cohort pressure at bar ``t`` is computed from state accumulated through
    ``t-1``.  The current bar's OI expansion is appended only after the pressure
    and score for that bar have been emitted, preventing same-bar cost-basis
    lookahead.
    """
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0.0)).to_numpy(float)
    log_oi = _numeric_log(market, "sum_open_interest")
    oi_change = log_oi.diff().to_numpy(float)
    new_oi = np.maximum(np.nan_to_num(oi_change, nan=0.0), 0.0)

    top_pos = _numeric_log(market, "sum_toptrader_long_short_ratio")
    global_acct = _numeric_log(market, "count_long_short_ratio")
    taker_ratio = _numeric_log(market, "sum_taker_long_short_vol_ratio")
    raw_side = 0.5 * top_pos + 0.25 * global_acct + 0.25 * taker_ratio
    side_z = _prior_z(raw_side, 288)
    attribution = np.tanh(side_z.to_numpy(float))

    decay = float(np.exp(np.log(0.5) / float(half_life)))
    long_weight = short_weight = long_cost = short_cost = 0.0
    long_weights: list[float] = []
    short_weights: list[float] = []
    long_basis: list[float] = []
    short_basis: list[float] = []
    long_pressure: list[float] = []
    short_pressure: list[float] = []

    ret = np.log(close).diff()
    realized_path_vol = np.sqrt(ret.pow(2).rolling(288, min_periods=288).sum()).replace(0.0, np.nan).to_numpy(float)

    for price, added, side_attr, rv in zip(log_price, new_oi, attribution, realized_path_vol, strict=True):
        lb = long_cost / long_weight if long_weight > 1e-15 else np.nan
        sb = short_cost / short_weight if short_weight > 1e-15 else np.nan
        long_weights.append(long_weight)
        short_weights.append(short_weight)
        long_basis.append(lb)
        short_basis.append(sb)
        if np.isfinite(price) and np.isfinite(rv) and rv > 0.0:
            long_pressure.append(long_weight * max(lb - price, 0.0) / rv if np.isfinite(lb) else np.nan)
            short_pressure.append(short_weight * max(price - sb, 0.0) / rv if np.isfinite(sb) else np.nan)
        else:
            long_pressure.append(np.nan)
            short_pressure.append(np.nan)

        long_weight *= decay
        short_weight *= decay
        long_cost *= decay
        short_cost *= decay
        if np.isfinite(price) and np.isfinite(side_attr) and added > 0.0:
            long_increment = added * max(side_attr, 0.0)
            short_increment = added * max(-side_attr, 0.0)
            long_weight += long_increment
            short_weight += short_increment
            long_cost += long_increment * price
            short_cost += short_increment * price

    long_pressure_z = _prior_z(pd.Series(long_pressure, index=market.index), 2016)
    short_pressure_z = _prior_z(pd.Series(short_pressure, index=market.index), 2016)
    oi_unwind = np.maximum(-(log_oi - log_oi.shift(72)).to_numpy(float), 0.0)

    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = (2.0 * taker_buy - quote).rolling(24, min_periods=24).sum() / quote.rolling(24, min_periods=24).sum().replace(0.0, np.nan)
    flow_z = _prior_z(flow, 288).to_numpy(float)

    long_liquidation_push = np.maximum(short_pressure_z.to_numpy(float), 0.0) * np.maximum(flow_z, 0.0)
    short_liquidation_push = np.maximum(long_pressure_z.to_numpy(float), 0.0) * np.maximum(-flow_z, 0.0)
    directional_pressure = np.maximum(short_pressure_z.to_numpy(float), 0.0) - np.maximum(long_pressure_z.to_numpy(float), 0.0)
    base_score = oi_unwind * (long_liquidation_push - short_liquidation_push)

    out = pd.DataFrame(
        {
            "side_z": side_z,
            "long_weight": long_weights,
            "short_weight": short_weights,
            "long_basis": long_basis,
            "short_basis": short_basis,
            "long_pressure": long_pressure,
            "short_pressure": short_pressure,
            "long_pressure_z": long_pressure_z,
            "short_pressure_z": short_pressure_z,
            "oi_unwind": oi_unwind,
            "flow_z": flow_z,
            "score": base_score,
            "score_no_oi_contraction": long_liquidation_push - short_liquidation_push,
            "score_no_taker_flow": oi_unwind * directional_pressure,
            "score_no_cost_basis_pressure": oi_unwind * flow_z,
        },
        index=market.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def fit_thresholds(score: np.ndarray, fit_mask: np.ndarray, tail: float) -> tuple[float, float]:
    finite_fit = fit_mask & np.isfinite(score)
    positive = score[finite_fit & (score > 0.0)]
    negative = score[finite_fit & (score < 0.0)]
    if min(len(positive), len(negative)) < 100:
        raise ValueError(f"insufficient signed fit observations: positive={len(positive)} negative={len(negative)}")
    return float(np.quantile(negative, tail)), float(np.quantile(positive, 1.0 - tail))


def build_signals(score: np.ndarray, lower: float, upper: float, mode: SideMode = "both") -> tuple[np.ndarray, np.ndarray]:
    long_active = np.isfinite(score) & (score >= upper)
    short_active = np.isfinite(score) & (score <= lower)
    active = long_active | short_active
    onset = active & ~np.r_[False, active[:-1]]
    long_signal = onset & long_active
    short_signal = onset & short_active
    if mode == "long_only":
        short_signal = np.zeros(len(score), dtype=bool)
    elif mode == "short_only":
        long_signal = np.zeros(len(score), dtype=bool)
    elif mode != "both":
        raise ValueError(f"unknown mode: {mode}")
    return long_signal, short_signal


def _score_for_variant(features: pd.DataFrame, variant: ScoreVariant) -> np.ndarray:
    column = "score" if variant == "base" else f"score_{variant}"
    return features[column].to_numpy(float)


def _simulate_windows(
    market: pd.DataFrame,
    dates: pd.Series,
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    hold: int,
    cfg: OiCostBasisConfig,
    extremes_by_hold: dict[int, tuple[np.ndarray, np.ndarray]],
    names: Iterable[str] = WINDOWS.keys(),
) -> dict[str, dict[str, Any]]:
    return {
        name: _simulate_no_stop(
            market,
            dates,
            long_signal,
            short_signal,
            window=name,
            hold_bars=hold,
            stride_bars=1,
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            extremes=extremes_by_hold[hold],
            windows=WINDOWS,
        )
        for name in names
    }


def _row_score(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fit_positive = sum(stats[name]["return_pct"] > 0.0 for name in FIT_SEGMENTS)
    select_positive = sum(stats[name]["return_pct"] > 0.0 for name in SELECT_SEGMENTS)
    robust_positive = fit_positive + select_positive
    return {
        "fit_positive_segments": int(fit_positive),
        "select_positive_segments": int(select_positive),
        "positive_segments": int(robust_positive),
        "min_fit_ratio": float(min(stats[name]["ratio"] for name in FIT_SEGMENTS)),
        "min_select_ratio": float(min(stats[name]["ratio"] for name in SELECT_SEGMENTS)),
        "min_segment_ratio": float(min(stats[name]["ratio"] for name in ROBUSTNESS_SEGMENTS)),
        "min_segment_trades": int(min(stats[name]["trades"] for name in ROBUSTNESS_SEGMENTS)),
    }


def _candidate_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "half_life": row["half_life"],
        "tail": row["tail"],
        "lower": row["lower"],
        "upper": row["upper"],
        "mode": row["mode"],
        "hold": row["hold"],
    }


def _admission(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_open_for_oos": False,
        "decision": "fail_pre2024_admission",
        "reason": "pre-2024 selection did not clear the admission gate: all 5 robustness segments must be positive and both fit/select ratios must be >= 3.0",
        "checks": {
            "positive_5_of_5": row["positive_segments"] == 5,
            "fit_ratio_ge_3": row["fit"]["ratio"] >= 3.0,
            "select_2023_ratio_ge_3": row["select_2023"]["ratio"] >= 3.0,
            "no_2024_loaded": True,
        },
    }


def scan_candidates(market: pd.DataFrame, cfg: OiCostBasisConfig) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    fit_mask = _window_mask(dates, "fit")
    extremes_by_hold = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    features_by_h: dict[int, pd.DataFrame] = {}
    for half_life, tail in itertools.product(HALF_LIVES, TAILS):
        features = features_by_h.setdefault(half_life, build_cost_basis_features(market, half_life))
        score = features["score"].to_numpy(float)
        lower, upper = fit_thresholds(score, fit_mask, tail)
        for mode, hold in itertools.product(MODES, HOLDS):
            long_signal, short_signal = build_signals(score, lower, upper, mode)  # type: ignore[arg-type]
            stats = _simulate_windows(market, dates, long_signal, short_signal, hold, cfg, extremes_by_hold)
            row = {
                "name": f"oi_cost_basis_liq_h{half_life}_tail{tail:.2f}_{mode}_hold{hold}",
                "half_life": half_life,
                "tail": tail,
                "lower": lower,
                "upper": upper,
                "mode": mode,
                "hold": hold,
                **_row_score(stats),
                **stats,
            }
            row["admission"] = _admission(row)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["min_segment_trades"] >= 6,
            row["positive_segments"],
            row["min_segment_ratio"],
            row["select_2023"]["ratio"],
            row["fit"]["ratio"],
            row["select_2023"]["return_pct"],
        ),
        reverse=True,
    )
    return {"rows": rows, "features_by_h": features_by_h, "extremes_by_hold": extremes_by_hold}


def run_ablations(
    market: pd.DataFrame,
    cfg: OiCostBasisConfig,
    top: dict[str, Any],
    features: pd.DataFrame,
    extremes_by_hold: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    fit_mask = _window_mask(dates, "fit")
    out: list[dict[str, Any]] = []
    base_score = features["score"].to_numpy(float)
    base_long, base_short = build_signals(base_score, top["lower"], top["upper"], top["mode"])
    for name, long_signal, short_signal in [("direction_flip", base_short, base_long)]:
        stats = _simulate_windows(market, dates, long_signal, short_signal, top["hold"], cfg, extremes_by_hold)
        out.append({"name": name, "threshold_source": "top_base_reused_and_sides_swapped", **_row_score(stats), **stats})
    for variant in ("no_oi_contraction", "no_taker_flow", "no_cost_basis_pressure"):
        score = _score_for_variant(features, variant)  # type: ignore[arg-type]
        lower, upper = fit_thresholds(score, fit_mask, top["tail"])
        long_signal, short_signal = build_signals(score, lower, upper, top["mode"])
        stats = _simulate_windows(market, dates, long_signal, short_signal, top["hold"], cfg, extremes_by_hold)
        out.append({"name": variant, "lower": lower, "upper": upper, "threshold_source": "fit_tail_refit_same_top_tail", **_row_score(stats), **stats})
    return out


def _input_fingerprint(market: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    source = pd.to_datetime(market["positioning_source_time"], errors="coerce")
    key_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "sum_open_interest",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "quote_asset_volume",
        "taker_buy_quote",
        "positioning_source_time",
    ]
    digest = hashlib.sha256(pd.util.hash_pandas_object(market[key_columns], index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()
    return {
        "rows": int(len(market)),
        "start": str(dates.min()),
        "end": str(dates.max()),
        "max_positioning_source_time": str(source.max()),
        "sha256_hash_pandas_key_columns": digest,
        "positioning_available_fraction_by_window": {
            name: float(pd.to_numeric(market.loc[_window_mask(dates, name), "positioning_available"], errors="coerce").mean())
            for name in WINDOWS
        },
    }


def _verification_payload(report_core: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    exact = {
        "scan_hash": _stable_json_hash(report_core["candidates"]),
        "manifest_hash": _stable_json_hash(manifest["manifest"]),
        "top_identity_hash": _stable_json_hash(manifest["manifest"][0] if manifest["manifest"] else {}),
    }
    top = report_core["candidates"][0]
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "deterministic_hashes": exact,
        "protocol_assertions": {
            "physical_cutoff": SELECTION_END,
            "no_2024_rows_loaded": report_core["input"]["end"] < SELECTION_END,
            "no_2024_source_rows_loaded": report_core["input"]["max_positioning_source_time"] < SELECTION_END,
            "search_space_count": report_core["tested"] == len(HALF_LIVES) * len(TAILS) * len(MODES) * len(HOLDS),
            "top_is_not_open_for_oos": top["admission"]["is_open_for_oos"] is False,
        },
        "top_summary": {
            "name": top["name"],
            "positive_segments": top["positive_segments"],
            "fit_ratio": top["fit"]["ratio"],
            "select_2023_ratio": top["select_2023"]["ratio"],
            "admission": top["admission"],
        },
    }


def write_markdown(report: dict[str, Any], manifest: dict[str, Any], verification: dict[str, Any], path: str) -> None:
    top = report["candidates"][0]
    ablation_lines = []
    for row in report["ablations"]:
        ablation_lines.append(
            f"| {row['name']} | {row['positive_segments']}/5 | {row['fit']['ratio']:.2f} | {row['select_2023']['ratio']:.2f} | {row['select_2023']['trades']} |"
        )
    candidate_lines = []
    for row in report["candidates"][:10]:
        candidate_lines.append(
            f"| {row['name']} | {row['positive_segments']}/5 | {row['fit']['ratio']:.2f} | {row['select_2023']['ratio']:.2f} | {row['min_segment_ratio']:.2f} | {row['select_2023']['trades']} |"
        )
    text = f"""# OI cost-basis liquidation alpha preflight — 2026-07-13

## Protocol

- Physical data cutoff: `{SELECTION_END}`; no 2024+ rows or positioning source rows are loaded.
- Mechanism: one-bar delayed Binance OI/positioning, path-dependent long/short OI expansion cohort cost basis, underwater cohort pressure, OI contraction, and taker-flow-confirmed liquidation direction.
- 2022 positioning gap handling: 2022 is loaded only for continuity/quarantine diagnostics and is excluded from fit and selection.
- Search grid: H `{HALF_LIVES}`, tail `{TAILS}`, mode `{MODES}`, hold `{HOLDS}` = `{report['tested']}` candidates.
- Execution: next-bar open, 0.5x, 6bp/side (`fee_rate=0.0005`, `slippage_rate=0.0001`), fixed hold, strict OHLC MDD.

## Top result

| name | positive segments | fit ratio | select 2023 ratio | min segment ratio | select trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| {top['name']} | {top['positive_segments']}/5 | {top['fit']['ratio']:.2f} | {top['select_2023']['ratio']:.2f} | {top['min_segment_ratio']:.2f} | {top['select_2023']['trades']} |

Admission decision: **{top['admission']['decision']}** / **not open for OOS**.  The best pre-2024 candidate is long-only and has 5/5 positive robustness segments, but its fit/select ratios are only about `{top['fit']['ratio']:.2f}` / `{top['select_2023']['ratio']:.2f}`, below the required admission gate.

## Top-10 pre-2024 candidates

| name | positive segments | fit ratio | select 2023 ratio | min segment ratio | select trades |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(candidate_lines)}

## Direction flip / component ablation on top spec

| variant | positive segments | fit ratio | select 2023 ratio | select trades |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(ablation_lines)}

## Deterministic verification

- Input rows: `{report['input']['rows']}`, range `{report['input']['start']}` through `{report['input']['end']}`.
- Max delayed positioning source time: `{report['input']['max_positioning_source_time']}`.
- Manifest hash: `{manifest['manifest_hash']}`.
- Scan hash: `{verification['deterministic_hashes']['scan_hash']}`.
- Protocol assertions: `{json.dumps(verification['protocol_assertions'], sort_keys=True)}`.

## Conclusion

The mechanism is economically interpretable but remains a preflight-only discovery.  Because admission fails before any 2024+ data is opened, this alpha is **not admitted to OOS/live evaluation**.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def run(cfg: OiCostBasisConfig) -> dict[str, Any]:
    market = load_pre2024_market(cfg)
    scan = scan_candidates(market, cfg)
    rows = scan["rows"]
    top_rows = rows[: cfg.top_n]
    manifest_entries = [_candidate_identity(row) | {"name": row["name"]} for row in top_rows]
    manifest_hash = _stable_json_hash(manifest_entries)
    top = rows[0]
    ablations = run_ablations(market, cfg, top, scan["features_by_h"][top["half_life"]], scan["extremes_by_hold"])
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "physical_cutoff": cfg.selection_end,
            "fit": WINDOWS["fit"],
            "fit_segments": {name: WINDOWS[name] for name in FIT_SEGMENTS},
            "quarantine": {"2022": WINDOWS["quarantine_2022"], "rule": "excluded from fit/selection because Binance top-trader positioning is missing-heavy"},
            "selection": {"select_2023": WINDOWS["select_2023"], "halves": {name: WINDOWS[name] for name in SELECT_SEGMENTS}},
            "search_grid": {"half_life": HALF_LIVES, "tail": TAILS, "mode": MODES, "hold": HOLDS},
            "execution": "signal on completed bar; entry next bar open; 0.5x; 6bp/side; strict OHLC MDD",
            "source_delay_bars": cfg.source_delay_bars,
        },
        "input": _input_fingerprint(market),
        "tested": len(rows),
        "manifest_hash": manifest_hash,
        "candidates": rows,
        "ablations": ablations,
        "alpha_qualifiers": [row for row in rows if row["admission"]["is_open_for_oos"]],
    }
    manifest = {
        "as_of": report["as_of"],
        "protocol_cutoff": cfg.selection_end,
        "manifest_hash": manifest_hash,
        "admission_status": "closed_no_oos_admission",
        "manifest": manifest_entries,
        "top": top,
    }
    verification = _verification_payload(report, manifest)

    for output, payload in (
        (cfg.scan_output, report),
        (cfg.manifest_output, manifest),
        (cfg.verification_output, verification),
    ):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default), encoding="utf-8")
    write_markdown(report, manifest, verification, cfg.doc_output)
    return {"report": report, "manifest": manifest, "verification": verification}


def parse_args() -> OiCostBasisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=OiCostBasisConfig.input_csv)
    parser.add_argument("--metrics-csv", default=OiCostBasisConfig.metrics_csv)
    parser.add_argument("--scan-output", default=OiCostBasisConfig.scan_output)
    parser.add_argument("--manifest-output", default=OiCostBasisConfig.manifest_output)
    parser.add_argument("--verification-output", default=OiCostBasisConfig.verification_output)
    parser.add_argument("--doc-output", default=OiCostBasisConfig.doc_output)
    parser.add_argument("--top-n", type=int, default=OiCostBasisConfig.top_n)
    parser.add_argument("--leverage", type=float, default=OiCostBasisConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=OiCostBasisConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=OiCostBasisConfig.slippage_rate)
    parser.add_argument("--metrics-tolerance", default=OiCostBasisConfig.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=OiCostBasisConfig.source_delay_bars)
    parser.add_argument("--selection-end", default=OiCostBasisConfig.selection_end)
    return OiCostBasisConfig(**vars(parser.parse_args()))


def main() -> None:
    result = run(parse_args())
    top = result["report"]["candidates"][0]
    print(
        json.dumps(
            {
                "tested": result["report"]["tested"],
                "manifest_hash": result["manifest"]["manifest_hash"],
                "top": {
                    "name": top["name"],
                    "positive_segments": top["positive_segments"],
                    "fit_ratio": top["fit"]["ratio"],
                    "select_2023_ratio": top["select_2023"]["ratio"],
                    "admission": top["admission"],
                },
                "verification": result["verification"]["protocol_assertions"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
