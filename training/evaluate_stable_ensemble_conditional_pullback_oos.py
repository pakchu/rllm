"""Evaluate the frozen conditional-pullback alpha on preregistered 2024+ windows.

The evaluator validates the externally pinned manifest and reconstructs the
complete pre-2024 activation/schedules before calling the future-data builder.
No candidate parameter is searchable in this module.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.search_market_braid_alpha as market_braid
import training.search_nested_barrier_witness_alpha as nested_barrier
from training.audit_confirmed_pullback_squeeze_live_parity import (
    _execution_config,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.audit_stable_ensemble_conditional_pullback_alpha import evaluate_model
from training.audit_weak_feature_responsibility_stability import (
    CANDIDATE_SPEC,
    FIT_END,
    FIT_START,
    OTHER_COLUMNS,
    PA_COLUMNS,
    _activation_hash,
    _array_hash,
    _resolve_existing,
    causal_shift,
    recent_side,
)
from training.freeze_stable_ensemble_conditional_pullback_alpha import (
    AUDIT_FILE_SHA256,
    AUDIT_RESULT,
    FUTURE_WINDOWS,
    MANIFEST_FIELDS,
    OOS_GATE,
    SELECTION_FILE_SHA256,
    SELECTION_RESULT,
    manifest_hash,
    validate_manifest,
)
from training.long_component_tp_union_scan import _component_mask
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_liveparity_state_feature_interactions import (
    feature_matrix as state_feature_matrix,
    immutable_anchors,
    net_target,
    slim,
    state_bank,
)
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_stable_ensemble_conditional_pullback_alpha import (
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    SEEDS,
    WIDTH_FEATURE,
    Config as SearchConfig,
    conditional_activation,
    deterministic_forest_predict,
    routed_schedule,
)

EXPECTED_MANIFEST_HASH = "ebf5e4602ac1cfd18d4c98a8955839f88df0ad358ded0d37ae911cf0c4aa20be"
SELECTION_END = "2024-01-01"
DEFAULT_MANIFEST = "results/stable_ensemble_conditional_pullback_manifest_2026-07-15.json"
DEFAULT_OUTPUT = "results/stable_ensemble_conditional_pullback_oos_2026-07-15.json"
DEFAULT_DOCS = "docs/stable-ensemble-conditional-pullback-oos-2026-07-15.md"
TREES_PER_SEED = 2_000


@dataclass(frozen=True)
class Config(SearchConfig):
    manifest: str = DEFAULT_MANIFEST
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = "2026-06-02"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_manifest(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest_hash(manifest) != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("manifest differs from the evaluator-pinned hash")
    if manifest["manifest_hash"] != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("manifest self-hash differs from the evaluator pin")
    if tuple(key for key in manifest if key in MANIFEST_FIELDS) != MANIFEST_FIELDS:
        raise RuntimeError("manifest payload shape changed")
    if _sha256(SELECTION_RESULT) != SELECTION_FILE_SHA256:
        raise RuntimeError("selection artifact changed after freeze")
    if _sha256(AUDIT_RESULT) != AUDIT_FILE_SHA256:
        raise RuntimeError("audit artifact changed after freeze")
    if manifest["future_windows"] != {key: list(value) for key, value in FUTURE_WINDOWS.items()}:
        raise RuntimeError("future windows differ from the frozen manifest")
    if manifest["oos_gate"] != OOS_GATE:
        raise RuntimeError("OOS gate differs from the frozen manifest")
    return manifest


def validate_oos_horizon(cfg: Config, manifest: dict[str, Any]) -> None:
    frozen_ends = [window[1] for window in manifest["future_windows"].values()]
    expected = max(frozen_ends)
    if cfg.exclude_from != expected:
        raise RuntimeError(
            f"exclude_from must match the frozen OOS horizon: {cfg.exclude_from} != {expected}"
        )


def fit_frozen_ensemble(context: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Fit frozen forests on pre-2023 examples and retain them for future prediction."""
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    anchors = immutable_anchors(context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"]))
    positions = np.flatnonzero(anchors & context["fit"])
    targets = np.asarray([net_target(engine, int(pos), 576, cfg) for pos in positions], dtype=float)
    finite = np.isfinite(targets)
    positions, targets = positions[finite], targets[finite]
    anchor_positions = np.flatnonzero(anchors)
    models: list[RandomForestRegressor] = []
    train_predictions: list[np.ndarray] = []
    anchor_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        model = RandomForestRegressor(
            n_estimators=TREES_PER_SEED,
            max_depth=int(CANDIDATE_SPEC["max_depth"]),
            min_samples_leaf=int(CANDIDATE_SPEC["min_samples_leaf"]),
            max_features=float(CANDIDATE_SPEC["max_features"]),
            random_state=int(seed),
            n_jobs=-1,
        ).fit(context["matrix"][positions], targets)
        train_predictions.append(deterministic_forest_predict(model, context["matrix"][positions]))
        anchor_predictions.append(
            deterministic_forest_predict(model, context["matrix"][anchor_positions])
        )
        models.append(model)
    return {
        "models": models,
        "engine": engine,
        "train_positions": positions,
        "anchor_positions": anchor_positions,
        "train_predictions": np.mean(np.stack(train_predictions), axis=0),
        "anchor_predictions": np.mean(np.stack(anchor_predictions), axis=0),
    }


def _replay_pre_oos(cfg: Config, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove the frozen prefix before any future builder can be called."""
    context = build_design_pre2024(cfg)
    if context["source_hashes"] != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-OOS source hashes changed")
    if context["feature_prefix_hash"] != manifest["feature_prefix_hash"]:
        raise RuntimeError("pre-OOS feature hash changed")
    fitted = fit_frozen_ensemble(context, cfg)
    replay = evaluate_model(
        context,
        fitted,
        cfg,
        label="frozen_pre_oos_replay",
        trees=len(SEEDS) * TREES_PER_SEED,
        seeds=list(SEEDS),
    )
    if replay["activation_hash"] != manifest["selected_activation_hash"]:
        raise RuntimeError("pre-OOS activation did not replay the freeze")
    if replay["schedule_hashes"] != manifest["selected_schedule_hashes"]:
        raise RuntimeError("pre-OOS schedules did not replay the freeze")
    if replay["stats"] != manifest["selected_stats"]:
        raise RuntimeError("pre-OOS statistics did not replay the freeze")
    for key, frozen in manifest["candidate_spec"]["thresholds"].items():
        if not np.isclose(float(replay["thresholds"][key.replace("_threshold", "")]), float(frozen), rtol=0.0, atol=1e-15):
            raise RuntimeError(f"pre-OOS {key} did not replay the freeze")
    return context, fitted | {"pre_replay": replay}


def build_design_pre2024(cfg: Config) -> dict[str, Any]:
    """Late import wrapper makes the pre-OOS/future ordering testable."""
    from training.audit_weak_feature_responsibility_stability import build_design

    return build_design(cfg)


def _load_full_nested(cutoff: str) -> tuple[pd.DataFrame, pd.Series]:
    path = str(_resolve_existing("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"))
    market = _read_before(path, "date", cutoff)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("nested source was not physically truncated")
    return market, dates


def _load_full_braid(cfg: Config, cutoff: str) -> tuple[pd.DataFrame, pd.Series]:
    market_path = str(_resolve_existing("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"))
    spot_path = str(_resolve_existing(cfg.spot_premium_csv))
    market = _read_before(market_path, "date", cutoff)
    auxiliary = _read_before(spot_path, "date", cutoff)
    for frame in (market, auxiliary):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    auxiliary = auxiliary.sort_values("date").drop_duplicates("date", keep="last")
    columns = ["date", "spot_close", "spot_rows", "premium_index_1m_close", "premium_rows"]
    market = market.merge(auxiliary[columns], on="date", how="left", validate="one_to_one").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("braid source was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("market-braid future source is not a complete 5-minute grid")
    return market, dates


def build_full_design(cfg: Config) -> dict[str, Any]:
    """Rebuild the frozen causal feature graph through the preregistered cutoff."""
    resolved = replace(
        cfg,
        input_csv=str(_resolve_existing(cfg.input_csv)),
        funding_csv=str(_resolve_existing(cfg.funding_csv)),
        premium_csv=str(_resolve_existing(cfg.premium_csv)),
    )
    market, raw_features, funding, source_hashes = _load_bundle(
        resolved,
        cutoff=cfg.exclude_from,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    features = live_decision_features(raw_features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    funding_leg = decisions & _component_mask(features, "funding10_trend70")
    premium_leg = decisions & _component_mask(features, "premium20_mom90")
    base = funding_leg | premium_leg
    bank = state_bank(market, dates)
    base &= (bank["kalman"] >= 0) & (bank["bocpd"] >= 0) & (bank["semimarkov"] >= 0)

    nested_market, nested_dates = _load_full_nested(cfg.exclude_from)
    if not np.array_equal(dates.to_numpy(), nested_dates.to_numpy()):
        raise RuntimeError("nested-barrier future grid mismatch")
    barrier_bank = nested_barrier.build_barrier_bank(nested_market)
    long_signal, short_signal, info = nested_barrier.coalesced_barrier_signals(
        nested_market,
        barrier_bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )
    nested_side = causal_shift(long_signal.astype(np.int8) - short_signal.astype(np.int8))
    nested24, nested_age = recent_side(nested_side, 288)
    nested48, _ = recent_side(nested_side, 576)

    braid_market, braid_dates = _load_full_braid(cfg, cfg.exclude_from)
    if not np.array_equal(dates.to_numpy(), braid_dates.to_numpy()):
        raise RuntimeError("market-braid future grid mismatch")
    braid_state = market_braid.build_bar_state(braid_market)
    events = market_braid.market_braid_events(
        braid_state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=144,
        topology_mode="relative_order",
    )
    braid_side = causal_shift(events.signal_side.to_numpy(np.int8))
    braid24, braid_age = recent_side(braid_side, 288)
    braid48, _ = recent_side(braid_side, 576)

    state = state_feature_matrix(bank, funding_leg, premium_leg)
    raw = np.column_stack(
        [state, *[pd.to_numeric(features[column], errors="coerce").to_numpy(float) for column in PA_COLUMNS + OTHER_COLUMNS]]
    )
    weak = np.column_stack(
        [
            causal_shift(info["high_work_ratio"], np.nan),
            causal_shift(info["low_work_ratio"], np.nan),
            causal_shift(info["high_coalescence"]),
            causal_shift(info["low_coalescence"]),
            nested24,
            nested48,
            np.minimum(nested_age, 576),
            braid24,
            braid48,
            np.minimum(braid_age, 576),
        ]
    )
    unfilled = np.column_stack([raw, weak])
    fit = ((dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END))).to_numpy(bool)
    median = np.nanmedian(unfilled[fit], axis=0)
    matrix = np.clip(np.where(np.isfinite(unfilled), unfilled, median), -20.0, 20.0)
    prefix = (dates < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    prefix_frame = pd.DataFrame(matrix[prefix], columns=FEATURE_COLUMNS).assign(
        date=dates[prefix].to_numpy()
    )
    return {
        "market": market,
        "funding": funding,
        "dates": dates,
        "features": features,
        "matrix": matrix,
        "fit": fit,
        "base": base,
        "funding_leg": funding_leg,
        "premium_leg": premium_leg,
        "source_hashes": source_hashes,
        "feature_prefix_hash": _frame_hash(prefix_frame),
        "feature_full_hash": _frame_hash(
            pd.DataFrame(matrix, columns=FEATURE_COLUMNS).assign(date=dates.to_numpy())
        ),
        "base_prefix_hash": _activation_hash(base[prefix], dates[prefix]),
        "anchor_prefix_hash": _activation_hash(
            immutable_anchors(base, int(CANDIDATE_SPEC["anchor_cooldown_bars"]))[prefix],
            dates[prefix],
        ),
        "nested_prefix_hash": _array_hash(nested_side[prefix]),
        "braid_prefix_hash": _array_hash(braid_side[prefix]),
    }


def predict_full_ensemble(
    full_context: dict[str, Any],
    pre_fitted: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    anchors = immutable_anchors(full_context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"]))
    anchor_positions = np.flatnonzero(anchors)
    predictions = [
        deterministic_forest_predict(model, full_context["matrix"][anchor_positions])
        for model in pre_fitted["models"]
    ]
    train_positions = np.asarray(pre_fitted["train_positions"], dtype=np.int64)
    if not np.array_equal(
        train_positions,
        np.flatnonzero(anchors & full_context["fit"])[: len(train_positions)],
    ):
        raise RuntimeError("full design changed frozen fit positions")
    return {
        "engine": ExecutionEngine(
            full_context["market"],
            full_context["funding"],
            _execution_config(cfg, cfg.leverage),
        ),
        "train_positions": train_positions,
        "anchor_positions": anchor_positions,
        "train_predictions": pre_fitted["train_predictions"],
        "anchor_predictions": np.mean(np.stack(predictions), axis=0),
    }


def frozen_activation(
    context: dict[str, Any],
    model: dict[str, Any],
    manifest: dict[str, Any],
) -> np.ndarray:
    positions = np.asarray(model["anchor_positions"], dtype=np.int64)
    funding = np.asarray(context["funding_leg"], dtype=bool)[positions]
    matrix = np.asarray(context["matrix"], dtype=float)
    thresholds = manifest["candidate_spec"]["thresholds"]
    return conditional_activation(
        size=len(context["market"]),
        anchor_positions=positions,
        anchor_predictions=model["anchor_predictions"],
        anchor_is_funding=funding,
        anchor_width=matrix[positions, FEATURE_COLUMNS.index(WIDTH_FEATURE)],
        anchor_pullback=matrix[positions, FEATURE_COLUMNS.index(PULLBACK_FEATURE)],
        funding_threshold=float(thresholds["funding_threshold"]),
        premium_threshold=float(thresholds["premium_threshold"]),
        width_threshold=float(thresholds["width_threshold"]),
        pullback_threshold=float(thresholds["pullback_threshold"]),
    )


def passes_oos_gate(stats: dict[str, dict[str, Any]]) -> bool:
    for window, gate in OOS_GATE.items():
        row = stats[window]
        if gate["positive"] and row["absolute_return_pct"] <= 0.0:
            return False
        if row["cagr_to_strict_mdd"] < gate["min_ratio"]:
            return False
        if row["strict_mdd_pct"] > gate["max_mdd"]:
            return False
        if row["trades"] < gate["min_trades"]:
            return False
    return True


def _window_record(
    context: dict[str, Any],
    model: dict[str, Any],
    active: np.ndarray,
    cfg: Config,
    *,
    start: str,
    end: str,
) -> tuple[list[Trade], dict[str, Any], dict[str, Any]]:
    trades = routed_schedule(context, {"engine": model["engine"], "active": active}, start=start, end=end)
    stats = slim(equity_stats(trades, start=start, end=end, cfg=_execution_config(cfg, cfg.leverage)))
    stress = slim(
        equity_stats(
            trades,
            start=start,
            end=end,
            cfg=_execution_config(cfg, cfg.leverage),
            cost_rate=0.0010,
        )
    )
    return trades, stats, stress


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stable ensemble conditional-pullback OOS — 2026-07-15",
        "",
        f"**{report['verdict']}**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "| Window | 6 bp/side result | 10 bp/side stress | Gate pass |",
        "|---|---:|---:|---:|",
    ]
    for name in FUTURE_WINDOWS:
        lines.append(
            f"| {name} | {_metric(report['stats_6bp'][name])} | "
            f"{_metric(report['stats_10bp'][name])} | {report['window_gate_passes'][name]} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- Pinned manifest `{report['manifest_hash']}` was validated before the future builder ran.",
        "- Pre-2024 feature, activation, and all schedule hashes replayed exactly.",
        "- Models were fit only on frozen 2020-07-01..2022-12-31 examples; 2023 was selection; 2024+ was not used for thresholds, exits, or model choice.",
        "- Execution uses next-open entry, 6 bp/notional/side, realized funding, stop-before-take, split-contained exits, wall-clock CAGR, and strict path MDD.",
        "- This is candidate-level implementation-clean OOS. It is not globally epistemically pristine because the broader repository previously researched related feature families on later periods.",
        "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    manifest = validate_frozen_manifest(cfg)
    validate_oos_horizon(cfg, manifest)
    pre_context, pre_fitted = _replay_pre_oos(cfg, manifest)
    pre_replay = pre_fitted["pre_replay"]
    pre_integrity = {
        "base_prefix_hash": pre_context["base_activation_hash"],
        "anchor_prefix_hash": _activation_hash(
            immutable_anchors(
                pre_context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"])
            ),
            pre_context["dates"],
        ),
        "nested_prefix_hash": pre_context["nested_side_hash"],
        "braid_prefix_hash": pre_context["braid_side_hash"],
    }
    pre_fitted.pop("pre_replay")
    pre_fitted.pop("engine")
    del pre_context
    gc.collect()

    # This is intentionally the first call allowed to open 2024+.
    full_context = build_full_design(cfg)
    if full_context["feature_prefix_hash"] != manifest["feature_prefix_hash"]:
        raise RuntimeError("full causal graph changed its frozen feature prefix")
    for key, expected in pre_integrity.items():
        if full_context[key] != expected:
            raise RuntimeError(f"full causal graph changed frozen {key}")
    full_model = predict_full_ensemble(full_context, pre_fitted, cfg)
    active = frozen_activation(full_context, full_model, manifest)
    prefix = (full_context["dates"] < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    if _activation_hash(active[prefix], full_context["dates"][prefix]) != manifest["selected_activation_hash"]:
        raise RuntimeError("full replay changed the frozen activation prefix")
    for name, frozen_hash in manifest["selected_schedule_hashes"].items():
        if name not in {
            "train",
            "train_2020h2",
            "train_2021",
            "train_2022",
            "select_2023",
            "select_2023_h1",
            "select_2023_h2",
            "pre_2024",
        }:
            continue
        windows = {
            "train": ("2020-07-01", "2023-01-01"),
            "train_2020h2": ("2020-07-01", "2021-01-01"),
            "train_2021": ("2021-01-01", "2022-01-01"),
            "train_2022": ("2022-01-01", "2023-01-01"),
            "select_2023": ("2023-01-01", "2024-01-01"),
            "select_2023_h1": ("2023-01-01", "2023-07-01"),
            "select_2023_h2": ("2023-07-01", "2024-01-01"),
            "pre_2024": ("2020-07-01", "2024-01-01"),
        }
        trades = routed_schedule(
            full_context,
            {"engine": full_model["engine"], "active": active},
            start=windows[name][0],
            end=windows[name][1],
        )
        if _schedule_hash(trades) != frozen_hash:
            raise RuntimeError(f"full replay changed frozen {name} schedule")

    stats_6bp: dict[str, dict[str, Any]] = {}
    stats_10bp: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[str, str] = {}
    source_counts: dict[str, dict[str, int]] = {}
    window_gate_passes: dict[str, bool] = {}
    for name, (start, end) in FUTURE_WINDOWS.items():
        trades, stats, stress = _window_record(
            full_context, full_model, active, cfg, start=start, end=end
        )
        stats_6bp[name] = stats
        stats_10bp[name] = stress
        schedule_hashes[name] = _schedule_hash(trades)
        funding_count = sum(bool(full_context["funding_leg"][trade.signal_position]) for trade in trades)
        source_counts[name] = {"funding": funding_count, "premium": len(trades) - funding_count}
        gate = OOS_GATE[name]
        window_gate_passes[name] = bool(
            stats["absolute_return_pct"] > 0.0
            and stats["cagr_to_strict_mdd"] >= gate["min_ratio"]
            and stats["strict_mdd_pct"] <= gate["max_mdd"]
            and stats["trades"] >= gate["min_trades"]
        )
    passed = passes_oos_gate(stats_6bp)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "ALPHA_QUALIFIED_FROZEN_OOS" if passed else "REJECTED_FROZEN_OOS",
        "future_opened": True,
        "candidate_level_oos_clean": True,
        "globally_epistemically_pristine": False,
        "manifest": cfg.manifest,
        "manifest_hash": manifest["manifest_hash"],
        "config": asdict(cfg),
        "integrity": {
            "pre_replay_activation_hash": pre_replay["activation_hash"],
            "pre_replay_schedule_hashes": pre_replay["schedule_hashes"],
            "full_feature_prefix_hash": full_context["feature_prefix_hash"],
            "full_feature_hash": full_context["feature_full_hash"],
            "full_active_prefix_hash": _activation_hash(active[prefix], full_context["dates"][prefix]),
            "pre_builder_prefix_hashes": pre_integrity,
            "full_builder_prefix_hashes": {
                key: full_context[key] for key in pre_integrity
            },
        },
        "stats_6bp": stats_6bp,
        "stats_10bp": stats_10bp,
        "window_gate_passes": window_gate_passes,
        "schedule_hashes": schedule_hashes,
        "source_counts": source_counts,
        "passes_oos_gate": passed,
    }
    target = Path(cfg.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    _write_docs(cfg.docs_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--spot-premium-csv", default=Config.spot_premium_csv)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    return parser.parse_args()


def main() -> None:
    report = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "manifest_hash": report["manifest_hash"],
                "passes_oos_gate": report["passes_oos_gate"],
                "stats_6bp": report["stats_6bp"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
