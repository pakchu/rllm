"""Search a causal price/order-flow path-signature alpha.

The feature is the normalized level-2 signed area of a trailing path whose
    increments are standardized BTC log returns and taker-order-flow changes.
The area distinguishes price-leading-flow loops from flow-leading-price loops;
fixed orientation mappings are selected before 2024 and replayed unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_causal_online_expert_alpha import (
    ALPHAS,
    OnlineExpertConfig,
    _build_expert_events,
    _global_nonoverlap,
    _load_bundle as _load_expert_bundle,
    _metric,
)
from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _validate_manifest,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before
from training.search_premium_intrabar_shape_alpha import (
    QUARTER_WINDOWS,
    WINDOWS,
    _fmt,
    _path_hash,
    _stats,
    _window_mask,
)
from training.search_spot_perp_absorption_alpha import (
    SELECTION_END,
    _jaccard,
    _make_event,
    _merge_with_priority,
    _prior_z,
)


PATH_WINDOWS = (24, 72, 144)
POLICY_FAMILIES = ("flow_led_continuation", "price_led_crowding_fade")
STEP_NORMALIZER = 576
SIGNAL_STRIDE = 12


@dataclass(frozen=True)
class PathSignatureConfig:
    input_csv: str
    funding_csv: str
    premium_csv: str
    output: str
    manifest_output: str
    docs_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    top_n: int = 6
    refresh_manifest: bool = False


def _source_hashes(cfg: PathSignatureConfig) -> dict[str, str]:
    return {
        str(Path(path)): _file_sha256(path)
        for path in (cfg.input_csv, cfg.funding_csv, cfg.premium_csv)
    }


def _load_market(cfg: PathSignatureConfig, *, cutoff: str) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = _read_before(cfg.input_csv, "date", cutoff)
    prefix_hashes = {"market": _frame_hash(raw)}
    market = raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    required = {
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"path-signature market data lacks columns: {missing}")
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("path-signature market was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("path-signature search requires a complete 5-minute grid")
    return market, prefix_hashes


def rolling_signed_area(dx: pd.Series, dy: pd.Series, window: int) -> pd.Series:
    """Translation-invariant level-2 signed area over trailing increments.

    Positive area means the x path tends to move before the y path under the
    orientation ``x_prev * dy - y_prev * dx``.  The implementation uses only
    prefix sums through the current completed increment.
    """

    x_step = pd.to_numeric(dx, errors="coerce").fillna(0.0)
    y_step = pd.to_numeric(dy, errors="coerce").fillna(0.0)
    x = x_step.cumsum()
    y = y_step.cumsum()
    x_prev = x.shift(1, fill_value=0.0)
    y_prev = y.shift(1, fill_value=0.0)
    cross = x_prev * y_step - y_prev * x_step
    base_x = x.shift(int(window)).fillna(0.0)
    base_y = y.shift(int(window)).fillna(0.0)
    local_cross = (
        cross.rolling(int(window), min_periods=int(window)).sum()
        - base_x * (y - base_y)
        + base_y * (x - base_x)
    )
    return 0.5 * local_cross


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    log_return = np.log(close.where(close > 0.0)).diff()
    imbalance = ((2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)).clip(-1.0, 1.0)
    flow_change = imbalance.diff()
    price_step = _prior_z(log_return, STEP_NORMALIZER).clip(-8.0, 8.0)
    flow_step = _prior_z(flow_change, STEP_NORMALIZER).clip(-8.0, 8.0)
    out: dict[str, pd.Series] = {
        "ps_price_step": price_step,
        "ps_flow_step": flow_step,
        "ps_taker_imbalance": imbalance,
    }
    for path_window in PATH_WINDOWS:
        raw_area = rolling_signed_area(price_step, flow_step, path_window)
        price_length = price_step.abs().rolling(path_window, min_periods=path_window).sum()
        flow_length = flow_step.abs().rolling(path_window, min_periods=path_window).sum()
        scale = (price_length * flow_length).replace(0.0, np.nan)
        area = (2.0 * raw_area / scale).clip(-2.0, 2.0)
        price_displacement = price_step.rolling(path_window, min_periods=path_window).sum() / price_length.replace(0.0, np.nan)
        flow_displacement = flow_step.rolling(path_window, min_periods=path_window).sum() / flow_length.replace(0.0, np.nan)
        out[f"ps_area_{path_window}"] = area
        out[f"ps_price_direction_{path_window}"] = price_displacement.clip(-1.0, 1.0)
        out[f"ps_flow_direction_{path_window}"] = flow_displacement.clip(-1.0, 1.0)
    return pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _fit_quantile(values: pd.Series, fit_mask: np.ndarray, quantile: float) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    reference = array[fit_mask & np.isfinite(array)]
    if len(reference) < 20_000:
        raise ValueError(f"insufficient fit observations for path-signature quantile: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _policy_specs(features: pd.DataFrame, fit_mask: np.ndarray) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path_window in PATH_WINDOWS:
        area = features[f"ps_area_{path_window}"]
        flow = features[f"ps_flow_direction_{path_window}"]
        thresholds = {
            "area_lower": _fit_quantile(area, fit_mask, 0.20),
            "area_upper": _fit_quantile(area, fit_mask, 0.80),
            "flow_lower": _fit_quantile(flow, fit_mask, 0.30),
            "flow_upper": _fit_quantile(flow, fit_mask, 0.70),
        }
        for family in POLICY_FAMILIES:
            specs.append(
                {
                    "path_window": path_window,
                    "family": family,
                    "hold": path_window,
                    "stride": SIGNAL_STRIDE,
                    **thresholds,
                }
            )
    return specs


def _signals(
    features: pd.DataFrame,
    spec: dict[str, Any],
    *,
    flip: bool = False,
    flow_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    path_window = int(spec["path_window"])
    area = pd.to_numeric(features[f"ps_area_{path_window}"], errors="coerce").to_numpy(float)
    flow = pd.to_numeric(features[f"ps_flow_direction_{path_window}"], errors="coerce").to_numpy(float)
    stride_bars = int(spec["stride"])
    sampled = np.arange(len(features), dtype=np.int64) % stride_bars == 0
    finite = np.isfinite(area) & np.isfinite(flow)
    family = str(spec["family"])
    if family == "flow_led_continuation":
        area_gate = np.ones(len(features), dtype=bool) if flow_only else area <= float(spec["area_lower"])
        long_active = finite & area_gate & (flow >= float(spec["flow_upper"]))
        short_active = finite & area_gate & (flow <= float(spec["flow_lower"]))
    elif family == "price_led_crowding_fade":
        area_gate = np.ones(len(features), dtype=bool) if flow_only else area >= float(spec["area_upper"])
        long_active = finite & area_gate & (flow <= float(spec["flow_lower"]))
        short_active = finite & area_gate & (flow >= float(spec["flow_upper"]))
    else:
        raise KeyError(family)
    previous_long = np.zeros(len(features), dtype=bool)
    previous_short = np.zeros(len(features), dtype=bool)
    previous_long[stride_bars:] = long_active[:-stride_bars]
    previous_short[stride_bars:] = short_active[:-stride_bars]
    long_active = sampled & long_active & ~previous_long
    short_active = sampled & short_active & ~previous_short
    active = long_active | short_active
    side = np.zeros(len(features), dtype=np.int8)
    side[long_active] = 1
    side[short_active] = -1
    if flip:
        side = -side
    side[~active] = 0
    return active, side


def _build_events(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: dict[str, Any],
    cfg: PathSignatureConfig,
    *,
    cost_rate: float,
    flip: bool = False,
    flow_only: bool = False,
) -> list[dict[str, Any]]:
    active, sides = _signals(features, spec, flip=flip, flow_only=flow_only)
    dummy_z = np.zeros(len(market), dtype=float)
    events: list[dict[str, Any]] = []
    next_allowed = 0
    for pos in np.flatnonzero(active):
        if int(pos) < next_allowed:
            continue
        event = _make_event(
            market,
            dummy_z,
            int(pos),
            int(sides[pos]),
            0,
            max_hold=int(spec["hold"]),
            dynamic_exit=False,
            exit_abs_z=0.0,
            cost_rate=cost_rate,
            leverage=float(cfg.leverage),
            name="path_signature_flip" if flip else "path_signature",
        )
        if event is None:
            continue
        events.append(event)
        next_allowed = int(event["exit_pos"])
    return events


def _selection_score(stats: dict[str, Any]) -> float:
    fit, full = stats["fit"], stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if (
        fit["return_pct"] <= 0.0
        or fit["trades"] < 30
        or full["return_pct"] <= 0.0
        or full["ratio"] < 1.0
        or full["trades"] < 12
        or min(h1["return_pct"], h2["return_pct"]) <= 0.0
        or min(h1["trades"], h2["trades"]) < 4
    ):
        return -1e12
    return float(min(full["ratio"], h1["ratio"], h2["ratio"]) + 0.1 * fit["ratio"] + 0.01 * min(full["trades"], 100))


def _select_top(rows: list[dict[str, Any]], cfg: PathSignatureConfig) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (-row["selection_score"], json.dumps(row["spec"], sort_keys=True)))
    return rows[: int(cfg.top_n)]


def _select_manifest(cfg: PathSignatureConfig) -> dict[str, Any]:
    market, prefix_hashes = _load_market(cfg, cutoff=SELECTION_END)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    fit_mask = _window_mask(dates, "fit")
    specs = _policy_specs(features, fit_mask)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        events = _build_events(market, features, spec, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
        flipped = _build_events(market, features, spec, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate), flip=True)
        flow_only = _build_events(market, features, spec, cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate), flow_only=True)
        stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        score = _selection_score(stats)
        diagnostics.append(
            {
                "spec": spec,
                "selection_score": score,
                "stats": stats,
                "direction_flipped": _stats(flipped, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2")),
                "flow_only_ablation": _stats(flow_only, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2")),
            }
        )
        if score <= -1e11:
            continue
        path_hash = _path_hash(events, dates, "select_2023")
        if path_hash in seen:
            continue
        seen.add(path_hash)
        rows.append({"spec": spec, "selection_score": score, "selection_stats": stats, "selection_path_hash": path_hash})
    selected = _select_top(rows, cfg)
    core = {
        "protocol": {
            "hypothesis": "level-2 price/order-flow signed area separates price-leading exhaustion from flow-leading continuation",
            "feature": "standardized completed 5m log-return and taker-imbalance-change increments; normalized trailing signed area",
            "normalization": "step moments end at t-1; current completed increments enter only the current path endpoint; area/flow thresholds use fit only",
            "selection": {name: WINDOWS[name] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")},
            "all_market_rows_physically_excluded_before_manifest": True,
            "later_metrics_included": False,
            "search_cap": f"{len(specs)} fixed signed-area policies; no threshold/hold grid",
            "thresholds": "fit-only q20/q80 signed area and q30/q70 flow direction",
            "entry_exit": "first hourly state-entry observation; next 5m open; hold equals path window; global non-overlap",
            "preflight_revision": "initial pre-2024 hourly state re-entry produced zero eligible paths and excessive repeat costs; first-entry-only sampling was fixed before any 2024+ replay",
            "cost": "6bp/side base, 10bp/side stress, 0.5x",
            "mdd": "strict entry cost plus intrabar adverse OHLC and realized high-water",
            "marginal_rule": "must improve deterministic six-sleeve union on combined return and CAGR/MDD",
            "status_ceiling": "shadow research",
        },
        "source_prefix_hashes": prefix_hashes,
        "feature_hash": _feature_hash(features, dates),
        "search_space": {"raw_specs": len(specs), "eligible_unique_paths": len(rows), "top_n": int(cfg.top_n)},
        "preflight_diagnostics": diagnostics,
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _expert_config(cfg: PathSignatureConfig) -> OnlineExpertConfig:
    return OnlineExpertConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="",
        manifest_output="",
        docs_output="",
        exclude_from=cfg.exclude_from,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        stress_fee_rate=cfg.stress_fee_rate,
    )


def _correlation_audit(features: pd.DataFrame, expert_features: pd.DataFrame, dates: pd.Series) -> dict[str, float]:
    fit = _window_mask(dates, "fit")
    references = [name for name in ("taker_imbalance", "trend_96", "range_vol", "volume_zscore") if name in expert_features]
    audit: dict[str, float] = {}
    for path_window in PATH_WINDOWS:
        source = features.loc[fit, f"ps_area_{path_window}"]
        correlations = [abs(float(source.corr(expert_features.loc[fit, name], method="spearman"))) for name in references]
        audit[f"ps_area_{path_window}_max_abs_spearman_vs_core"] = max((value for value in correlations if np.isfinite(value)), default=0.0)
    return audit


def _replay(cfg: PathSignatureConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    prefix_market, prefix_hashes = _load_market(cfg, cutoff=SELECTION_END)
    prefix_dates = pd.to_datetime(prefix_market["date"])
    prefix_features = build_features(prefix_market)
    if prefix_hashes != manifest["source_prefix_hashes"] or _feature_hash(prefix_features, prefix_dates) != manifest["feature_hash"]:
        raise RuntimeError("pre-2024 path-signature reconstruction drift")
    market, _ = _load_market(cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), dates.loc[prefix].reset_index(drop=True)) != manifest["feature_hash"]:
        raise RuntimeError("full replay path-signature prefix drift")

    expert_cfg = _expert_config(cfg)
    expert_market, expert_features, _ = _load_expert_bundle(expert_cfg, cutoff=cfg.exclude_from)
    if not pd.to_datetime(expert_market["date"]).equals(dates):
        raise RuntimeError("path-signature and expert baseline grids differ")
    base_events = _build_expert_events(expert_market, expert_features, expert_cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
    base_by_expert = {name: [event for event in base_events if event["expert"] == name] for name in ALPHAS}
    base_union = _global_nonoverlap(base_events)
    eval_windows = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    baseline_stats = _stats(base_union, dates, eval_windows)
    correlation_audit = _correlation_audit(features, expert_features, dates)
    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate, slippage_rate=0.0001)
    stress_expert_cfg = _expert_config(stress_cfg)
    stress_base_events = _build_expert_events(expert_market, expert_features, stress_expert_cfg, cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate))
    stress_base_union = _global_nonoverlap(stress_base_events)

    rows: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        events = _build_events(market, features, frozen["spec"], cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate))
        selection_stats = _stats(events, dates, ("fit", "select_2023", "select_2023_h1", "select_2023_h2"))
        if selection_stats != frozen["selection_stats"] or _path_hash(events, dates, "select_2023") != frozen["selection_path_hash"]:
            raise RuntimeError(f"pre-2024 path-signature policy drift at rank {rank}")
        stats = _stats(events, dates, WINDOWS)
        flipped = _build_events(market, features, frozen["spec"], cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate), flip=True)
        flow_only = _build_events(market, features, frozen["spec"], cfg, cost_rate=float(cfg.fee_rate + cfg.slippage_rate), flow_only=True)
        stress_events = _build_events(market, features, frozen["spec"], stress_cfg, cost_rate=float(stress_cfg.fee_rate + stress_cfg.slippage_rate))
        combined = _merge_with_priority(base_union, events)
        stress_combined = _merge_with_priority(stress_base_union, stress_events)
        combined_stats = _stats(combined, dates, eval_windows)
        stress_stats = _stats(stress_events, dates, eval_windows)
        flow_only_stats = _stats(flow_only, dates, eval_windows)
        quarterly = {name: _metric(events, dates, start, end) for name, (start, end) in QUARTER_WINDOWS.items()}
        quarter_summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        jaccards = {name: _jaccard(events, source, dates) for name, source in base_by_expert.items()}
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, all_oos = stats["holdout_2026"], stats["oos_2024_2026"]
        standalone = (
            test["return_pct"] > 0.0
            and test["ratio"] >= 3.0
            and test["trades"] >= 20
            and min(test["long_trades"], test["short_trades"]) >= 5
            and test["strict_mdd_pct"] <= 15.0
            and evaluation["return_pct"] > 0.0
            and evaluation["ratio"] >= 3.0
            and evaluation["trades"] >= 20
            and min(evaluation["long_trades"], evaluation["short_trades"]) >= 5
            and evaluation["strict_mdd_pct"] <= 15.0
            and holdout["return_pct"] > 0.0
            and holdout["trades"] >= 12
            and min(holdout["long_trades"], holdout["short_trades"]) >= 3
            and holdout["ratio"] >= 1.5
            and holdout["strict_mdd_pct"] <= 15.0
            and all_oos["return_pct"] > 0.0
        )
        base_all, merged_all = baseline_stats["oos_2024_2026"], combined_stats["oos_2024_2026"]
        marginal = merged_all["return_pct"] > base_all["return_pct"] and merged_all["ratio"] > base_all["ratio"]
        stress_ok = min(stress_stats[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        flow_only_all = flow_only_stats["oos_2024_2026"]
        area_adds_value = (
            all_oos["ratio"] >= flow_only_all["ratio"] * 1.25
            or min(flow_only_stats["test_2024"]["return_pct"], flow_only_stats["eval_2025"]["return_pct"]) <= 0.0
        )
        feature_correlation = correlation_audit[f"ps_area_{int(frozen['spec']['path_window'])}_max_abs_spearman_vs_core"]
        low_correlation = feature_correlation <= 0.25
        bonferroni = min(1.0, all_oos["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        qualifies = standalone and marginal and stress_ok and area_adds_value and low_correlation and max(jaccards.values(), default=0.0) <= 0.10 and bonferroni <= 0.05
        rows.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "direction_flipped": _stats(flipped, dates, eval_windows),
                "flow_only_ablation": flow_only_stats,
                "stress_10bp_each_side": stress_stats,
                "combined_with_six_sleeve_union": combined_stats,
                "stress_combined_with_six_sleeve_union": _stats(stress_combined, dates, eval_windows),
                "quarterly_stats": quarterly,
                "quarterly_summary": quarter_summary,
                "signal_jaccard_vs_fixed_experts": jaccards,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_standalone_gate": bool(standalone),
                "adds_value_vs_six_sleeve_union": bool(marginal),
                "passes_cost_stress": bool(stress_ok),
                "signed_area_adds_value_vs_flow_only": bool(area_adds_value),
                "passes_low_feature_correlation": bool(low_correlation),
                "passes_alpha_pool": bool(qualifies),
                "passes_live_grade": False,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_file_hashes_after_manifest_freeze": _source_hashes(cfg),
        "feature_correlation_audit": correlation_audit,
        "six_sleeve_union_baseline": baseline_stats,
        "selected": rows,
        "alpha_pool_qualifiers": [row for row in rows if row["passes_alpha_pool"]],
        "live_grade": [],
    }


def _write_doc(cfg: PathSignatureConfig, report: dict[str, Any]) -> None:
    manifest = json.loads(Path(cfg.manifest_output).read_text())
    search = manifest["search_space"]
    if report.get("preflight_only"):
        lines = [
            "# Price/order-flow path-signature alpha preflight (2026-07-13)",
            "",
            "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
            "",
            "| policy | fit | 2023 | 2023H1 | 2023H2 | flipped 2023 | flow-only 2023 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in report["preflight_diagnostics"]:
            stats = row["stats"]
            lines.append(
                f"| `{row['spec']['family']} w{row['spec']['path_window']}` | {_fmt(stats['fit'])} | {_fmt(stats['select_2023'])} | {_fmt(stats['select_2023_h1'])} | {_fmt(stats['select_2023_h2'])} | {_fmt(row['direction_flipped']['select_2023'])} | {_fmt(row['flow_only_ablation']['select_2023'])} |"
            )
        lines += [
            "",
            "## Verdict",
            "",
            f"- Eligible policies: {search['eligible_unique_paths']} of {search['raw_specs']}; OOS opened: **no**.",
            "- Every original policy lost in fit or 2023/H1/H2. Direction flips also lacked two-half stability, so this is not a simple sign error.",
            "- Signed-area filtering usually reduced the loss versus flow-only, but did not create a positive executable edge after 6bp/side.",
            "- Reject these exact fixed quantile/direction/hold mappings without spending the 2024-2026 replay budget.",
            "- Continuous signed-area and flow-direction fields remain research context only; they are not an alpha.",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python -m training.search_path_signature_leadlag_alpha --input-csv {cfg.input_csv} --funding-csv {cfg.funding_csv} --premium-csv {cfg.premium_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
            "```",
        ]
        path = Path(cfg.docs_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return
    lines = [
        "# Price/order-flow path-signature alpha search (2026-07-13)",
        "",
        "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "| rank | policy | 2024 | 2025 | 2026 | combined | +union combined | alpha |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["selected"]:
        stats = row["stats"]
        merged = row["combined_with_six_sleeve_union"]["oos_2024_2026"]
        lines.append(
            f"| {row['manifest_rank']} | `{row['spec']}` | {_fmt(stats['test_2024'])} | {_fmt(stats['eval_2025'])} | {_fmt(stats['holdout_2026'])} | {_fmt(stats['oos_2024_2026'])} | {_fmt(merged)} | {'yes' if row['passes_alpha_pool'] else 'no'} |"
        )
    baseline = report["six_sleeve_union_baseline"]["oos_2024_2026"]
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Pre-2024 admission: {search['eligible_unique_paths']} unique paths from {search['raw_specs']} fixed policies; alpha-pool qualifiers: {len(report['alpha_pool_qualifiers'])}.",
        f"- Deterministic six-sleeve union baseline: `{_fmt(baseline)}`.",
        "- The signed area uses only completed price and taker-flow increments, prior-fitted normalization, and next-open execution.",
        "- A standalone pass is insufficient without flow-only ablation improvement, low correlation, positive marginal contribution, and cost-stress survival.",
        "- 2024-2026 are replay evidence and cannot promote a policy directly to live grade.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python -m training.search_path_signature_leadlag_alpha --input-csv {cfg.input_csv} --funding-csv {cfg.funding_csv} --premium-csv {cfg.premium_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
        "```",
    ]
    path = Path(cfg.docs_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(cfg: PathSignatureConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    if not manifest["selected"]:
        report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "manifest": cfg.manifest_output,
            "manifest_sha256": manifest["sha256"],
            "protocol": manifest["protocol"],
            "source_prefix_hashes": manifest["source_prefix_hashes"],
            "preflight_only": True,
            "oos_opened": False,
            "preflight_diagnostics": manifest["preflight_diagnostics"],
            "selected": [],
            "alpha_pool_qualifiers": [],
            "live_grade": [],
        }
    else:
        report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    _write_doc(cfg, report)
    return report


def parse_args() -> PathSignatureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docs-output", required=True)
    parser.add_argument("--exclude-from", default=PathSignatureConfig.exclude_from)
    parser.add_argument("--refresh-manifest", action="store_true")
    return PathSignatureConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest": report["manifest"], "qualifiers": len(report["alpha_pool_qualifiers"]), "top": report["selected"][:3]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
