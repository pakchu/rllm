"""Search a non-REX alpha from cross-venue and microstructure consensus.

The selection phase physically excludes every row from 2024 onward:

* feature thresholds are fitted on 2020-2022;
* rule/execution variants are ranked on 2023 and both 2023 half-years;
* a deterministic Top-10 manifest is written before 2024+ data is loaded;
* the frozen manifest is replayed on 2024, 2025 and 2026 YTD.

Every rule combines exactly one Korean-local/FX feature with exactly one
Binance microstructure feature.  No REX feature, event, prediction or model is
an input to candidate construction.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.long_regime_combo_scan import LongComboScanConfig, _load_market
from training.search_jump_variation_bidirectional_alpha import features as build_jump_features
from training.search_kimchi_leadlag_bidirectional_alpha import features as build_kimchi_features
from training.search_liquidity_recovery_bidirectional_alpha import features as build_liquidity_features
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_volume_clock_bidirectional_alpha import features as build_volume_clock_features


WINDOWS = {
    "fit_2020_2022": ("2020-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}

LOCAL_FEATURES = (
    "kl_local_impulse_48",
    "kl_local_impulse_144",
    "kl_local_impulse_288",
    "kl_kimchi_btc_gap_48",
    "kl_kimchi_btc_gap_144",
)

MICRO_FEATURES = (
    "vc_flow_speed_0p25",
    "vc_imbalance_0p25",
    "vc_flow_speed_0p5",
    "lr_flow_recovery",
    "lr_signed_eff_72",
    "lr_flow_72",
    "jv_signed_jump_72",
    "jv_flow_recovery",
)


@dataclass(frozen=True)
class CrossVenueConsensusConfig(LongComboScanConfig):
    manifest_output: str = ""
    exclude_from: str = "2026-06-02"
    hold_bars: str = "24,48,72,144,288"
    stride_bars: str = "12,24"
    quantiles: str = "0.20,0.30"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    top_n: int = 10
    top_per_pair: int = 2
    min_fit_trades: int = 60
    min_select_trades: int = 24
    min_half_trades: int = 8


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _build_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build only the pre-registered non-REX feature families."""

    features = pd.DataFrame(index=market.index)
    features = build_kimchi_features(market, features)
    features = build_volume_clock_features(market, features)
    features = build_jump_features(market, features)
    features = build_liquidity_features(market, features)
    forbidden = sorted(name for name in features.columns if "rex" in str(name).lower())
    if forbidden:
        raise RuntimeError(f"REX-derived feature entered independent search: {forbidden[:5]}")
    return features


def _availability_mask(market: pd.DataFrame) -> np.ndarray:
    required = ("kimchi_available", "usdkrw_available")
    missing = [name for name in required if name not in market.columns]
    if missing:
        raise ValueError(f"market data lacks cross-venue availability columns: {missing}")
    return (
        (pd.to_numeric(market["kimchi_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5)
        & (pd.to_numeric(market["usdkrw_available"], errors="coerce").fillna(0.0).to_numpy(float) > 0.5)
    )


def _fit_quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float:
    reference = values[mask & np.isfinite(values)]
    if len(reference) < 20_000:
        raise ValueError(f"insufficient fit observations: {len(reference)}")
    return float(np.quantile(reference, float(quantile)))


def _condition(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    return finite & ((values >= threshold) if op == "ge" else (values <= threshold))


def _rule_masks(
    features: pd.DataFrame,
    available: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    local = features[spec["local_feature"]].to_numpy(float)
    micro = features[spec["micro_feature"]].to_numpy(float)
    long_active = available & _condition(local, "ge", spec["local_upper"])
    short_active = available & _condition(local, "le", spec["local_lower"])
    if spec["relation"] == "agreement":
        long_active &= _condition(micro, "ge", spec["micro_upper"])
        short_active &= _condition(micro, "le", spec["micro_lower"])
    elif spec["relation"] == "disagreement":
        long_active &= _condition(micro, "le", spec["micro_lower"])
        short_active &= _condition(micro, "ge", spec["micro_upper"])
    else:
        raise ValueError(f"unknown relation: {spec['relation']}")
    return long_active, short_active


def _candidate_specs(
    features: pd.DataFrame,
    fit_mask: np.ndarray,
    available: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, Any]]:
    reference = fit_mask & available
    specs: list[dict[str, Any]] = []
    for local_feature, micro_feature, relation, local_tail, micro_tail in itertools.product(
        LOCAL_FEATURES,
        MICRO_FEATURES,
        ("agreement", "disagreement"),
        quantiles,
        quantiles,
    ):
        local_values = features[local_feature].to_numpy(float)
        micro_values = features[micro_feature].to_numpy(float)
        spec = {
            "pair_family": f"{local_feature}__{micro_feature}__{relation}",
            "local_feature": local_feature,
            "micro_feature": micro_feature,
            "relation": relation,
            "local_tail": float(local_tail),
            "micro_tail": float(micro_tail),
            "local_lower": _fit_quantile(local_values, reference, local_tail),
            "local_upper": _fit_quantile(local_values, reference, 1.0 - local_tail),
            "micro_lower": _fit_quantile(micro_values, reference, micro_tail),
            "micro_upper": _fit_quantile(micro_values, reference, 1.0 - micro_tail),
        }
        specs.append(spec)
    return specs


def _rank_correlation(
    features: pd.DataFrame,
    available: np.ndarray,
    period: np.ndarray,
    spec: dict[str, Any],
) -> float:
    local = features[spec["local_feature"]].to_numpy(float)
    micro = features[spec["micro_feature"]].to_numpy(float)
    keep = available & period & np.isfinite(local) & np.isfinite(micro)
    if int(keep.sum()) < 100:
        return 0.0
    local_rank = pd.Series(local[keep]).rank(method="average").to_numpy(float)
    micro_rank = pd.Series(micro[keep]).rank(method="average").to_numpy(float)
    corr = np.corrcoef(local_rank, micro_rank)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    cfg: CrossVenueConsensusConfig,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
    extremes: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        long_active,
        short_active,
        window=window,
        hold_bars=hold_bars,
        stride_bars=stride_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=WINDOWS,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: CrossVenueConsensusConfig) -> float:
    fit = stats["fit_2020_2022"]
    select = stats["select_2023"]
    h1 = stats["select_2023_h1"]
    h2 = stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or select["trades"] < cfg.min_select_trades:
        return -1e12
    if h1["trades"] < cfg.min_half_trades or h2["trades"] < cfg.min_half_trades:
        return -1e12
    if min(fit["longs"], fit["shorts"]) < 8 or min(select["longs"], select["shorts"]) < 5:
        return -1e12
    if min(h1["longs"], h1["shorts"], h2["longs"], h2["shorts"]) < 2:
        return -1e12
    if min(fit["cagr_pct"], select["cagr_pct"], h1["cagr_pct"], h2["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 30.0 or select["strict_mdd_pct"] > 20.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], select["ratio"], h1["ratio"], h2["ratio"]], dtype=float)
    robust_ratio = float(np.min(ratios))
    central_ratio = float(np.median(ratios))
    trade_support = min(1.0, select["trades"] / 80.0)
    significance_bonus = max(0.0, 0.10 - float(select["p_value_mean_return_approx"]))
    return robust_ratio + 0.25 * central_ratio + 0.20 * trade_support + significance_bonus


def _activation_hash(long_active: np.ndarray, short_active: np.ndarray) -> str:
    packed = np.packbits(np.stack([long_active, short_active], axis=0), axis=None)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _select_manifest(cfg: CrossVenueConsensusConfig) -> dict[str, Any]:
    selection_cfg = replace(cfg, exclude_from="2024-01-01")
    market = _load_market(selection_cfg)
    if not market.empty and pd.Timestamp(market["date"].max()) >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("selection phase contains post-2023 market data")
    dates = pd.to_datetime(market["date"])
    features = _build_features(market)
    available = _availability_mask(market)
    fit_mask = _window_mask(dates, "fit_2020_2022")
    specs = _candidate_specs(features, fit_mask, available, _parse_csv(cfg.quantiles, float))
    holds = _parse_csv(cfg.hold_bars, int)
    strides = _parse_csv(cfg.stride_bars, int)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    seen_masks: set[str] = set()
    for spec in specs:
        long_active, short_active = _rule_masks(features, available, spec)
        mask_hash = _activation_hash(long_active, short_active)
        if mask_hash in seen_masks:
            continue
        seen_masks.add(mask_hash)
        correlation = _rank_correlation(features, available, fit_mask, spec)
        for hold_bars, stride_bars in itertools.product(holds, strides):
            stats = {
                window: _simulate(
                    market,
                    dates,
                    long_active,
                    short_active,
                    cfg,
                    window=window,
                    hold_bars=hold_bars,
                    stride_bars=stride_bars,
                    extremes=extremes[hold_bars],
                )
                for window in ("fit_2020_2022", "select_2023", "select_2023_h1", "select_2023_h2")
            }
            score = _selection_score(stats, cfg)
            if score <= -1e11:
                continue
            rows.append(
                {
                    **spec,
                    "hold_bars": int(hold_bars),
                    "stride_bars": int(stride_bars),
                    "feature_rank_correlation_fit": correlation,
                    "activation_hash": mask_hash,
                    "selection_score": score,
                    "selection_stats": stats,
                }
            )
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023"]["ratio"],
            row["selection_stats"]["select_2023"]["return_pct"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for row in rows:
        family = row["pair_family"]
        if family_counts.get(family, 0) >= cfg.top_per_pair:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= cfg.top_n:
            break
    manifest_core = {
        "protocol": {
            "fit": WINDOWS["fit_2020_2022"],
            "selection": WINDOWS["select_2023"],
            "selection_halves": [WINDOWS["select_2023_h1"], WINDOWS["select_2023_h2"]],
            "future_rows_physically_excluded_before_manifest": True,
            "features": "one cross-venue local feature plus one Binance microstructure feature; no REX input",
            "entry": "next 5m open",
            "exit": "fixed hold, no TP/SL",
            "cost": "6bp/side",
            "mdd": "strict favorable-high-water then adverse-extreme path",
        },
        "search_space": {
            "raw_specs": len(specs),
            "effective_unique_masks": len(seen_masks),
            "eligible_execution_variants": len(rows),
            "top_n": cfg.top_n,
            "top_per_pair": cfg.top_per_pair,
        },
        "selected": selected,
    }
    canonical = json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        **manifest_core,
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _executed_signal_dates(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
) -> set[pd.Timestamp]:
    period = _window_mask(dates, window)
    candidates = np.arange(0, len(market) - hold_bars - 2, stride_bars, dtype=np.int64)
    candidates = candidates[period[candidates] & (long_active[candidates] | short_active[candidates])]
    executed: set[pd.Timestamp] = set()
    next_position = 0
    for position in candidates:
        if position < next_position:
            continue
        side_conflict = bool(long_active[position]) == bool(short_active[position])
        exit_position = int(position) + 1 + int(hold_bars)
        if side_conflict or exit_position >= len(market) or not period[exit_position]:
            continue
        executed.add(pd.Timestamp(dates.iloc[int(position)]))
        next_position = exit_position + 1
    return executed


def _load_reference_dates(paths: list[str]) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("date"):
                dates.add(pd.Timestamp(row["date"]).tz_localize(None))
    return dates


def _jaccard(left: set[pd.Timestamp], right: set[pd.Timestamp]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _replay_manifest(
    cfg: CrossVenueConsensusConfig,
    manifest: dict[str, Any],
    rex_reference_jsonl: list[str],
) -> dict[str, Any]:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    features = _build_features(market)
    available = _availability_mask(market)
    holds = sorted({int(row["hold_bars"]) for row in manifest["selected"]})
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    reference_dates = _load_reference_dates(rex_reference_jsonl)
    replayed: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec_keys = (
            "pair_family",
            "local_feature",
            "micro_feature",
            "relation",
            "local_tail",
            "micro_tail",
            "local_lower",
            "local_upper",
            "micro_lower",
            "micro_upper",
        )
        spec = {key: frozen[key] for key in spec_keys}
        long_active, short_active = _rule_masks(features, available, spec)
        hold_bars = int(frozen["hold_bars"])
        stride_bars = int(frozen["stride_bars"])
        stats = {
            window: _simulate(
                market,
                dates,
                long_active,
                short_active,
                cfg,
                window=window,
                hold_bars=hold_bars,
                stride_bars=stride_bars,
                extremes=extremes[hold_bars],
            )
            for window in WINDOWS
        }
        for key in ("fit_2020_2022", "select_2023", "select_2023_h1", "select_2023_h2"):
            if stats[key] != frozen["selection_stats"][key]:
                raise RuntimeError(f"frozen selection replay drift for rank {rank} window {key}")
        overlap = {}
        for window in ("test_2024", "eval_2025", "holdout_2026"):
            candidate_dates = _executed_signal_dates(
                market,
                dates,
                long_active,
                short_active,
                window=window,
                hold_bars=hold_bars,
                stride_bars=stride_bars,
            )
            start, end = WINDOWS[window]
            reference_window = {date for date in reference_dates if pd.Timestamp(start) <= date < pd.Timestamp(end)}
            overlap[window] = {
                "candidate_signals": len(candidate_dates),
                "rex_reference_signals": len(reference_window),
                "exact_intersection": len(candidate_dates & reference_window),
                "jaccard": _jaccard(candidate_dates, reference_window),
            }
        test = stats["test_2024"]
        evaluation = stats["eval_2025"]
        holdout = stats["holdout_2026"]
        combined = stats["oos_2024_2026"]
        enough = (
            test["trades"] >= 20
            and evaluation["trades"] >= 20
            and holdout["trades"] >= 12
            and min(test["longs"], test["shorts"], evaluation["longs"], evaluation["shorts"]) >= 4
        )
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5
        passes_live_grade = (
            passes_alpha_pool
            and holdout["ratio"] >= 3.0
            and combined["ratio"] >= 3.0
            and combined["p_value_mean_return_approx"] <= 0.10
            and max(item["jaccard"] for item in overlap.values()) < 0.05
        )
        replayed.append(
            {
                "manifest_rank": rank,
                **frozen,
                "feature_rank_correlation_select": _rank_correlation(
                    features,
                    available,
                    _window_mask(dates, "select_2023"),
                    spec,
                ),
                "stats": stats,
                "rex_activation_overlap": overlap,
                "passes_alpha_pool": passes_alpha_pool,
                "passes_live_grade": passes_live_grade,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": str(Path(cfg.manifest_output)),
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "independence_guard": {
            "rex_features_used": False,
            "rex_events_used_for_selection": False,
            "rex_reference_used_only_for_post_selection_overlap_audit": bool(reference_dates),
            "max_allowed_exact_signal_jaccard": 0.05,
        },
        "selected": replayed,
        "alpha_pool_qualifiers": [row for row in replayed if row["passes_alpha_pool"]],
        "live_grade": [row for row in replayed if row["passes_live_grade"]],
    }


def run(cfg: CrossVenueConsensusConfig, rex_reference_jsonl: list[str]) -> dict[str, Any]:
    if not cfg.manifest_output:
        raise ValueError("manifest_output is required")
    manifest = _select_manifest(cfg)
    report = _replay_manifest(cfg, manifest, rex_reference_jsonl)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> tuple[CrossVenueConsensusConfig, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--exclude-from", default="2026-06-02")
    parser.add_argument("--hold-bars", default="24,48,72,144,288")
    parser.add_argument("--stride-bars", default="12,24")
    parser.add_argument("--quantiles", default="0.20,0.30")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-per-pair", type=int, default=2)
    parser.add_argument("--rex-reference-jsonl", action="append", default=[])
    args = parser.parse_args()
    references = list(args.rex_reference_jsonl)
    delattr(args, "rex_reference_jsonl")
    return CrossVenueConsensusConfig(**vars(args)), references


def main() -> None:
    cfg, references = parse_args()
    report = run(cfg, references)
    print(
        json.dumps(
            {
                "manifest": report["manifest"],
                "manifest_sha256": report["manifest_sha256"],
                "selected": len(report["selected"]),
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "live_grade": len(report["live_grade"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
