#!/usr/bin/env python3
"""Executable COGR-12 cash-open cross-asset gap relay evaluator.

``selection`` physically limits every COGR feature, target, funding row, and
candidate path to the prefix ending at 2024-01-01. ``eval`` opens only the
frozen passing 2023H2 top-1 and limits the candidate prefix to 2025-01-01.
Gross9 may build its checksum-bound pre-2025 legacy context in either phase,
but COGR accounting is always generated through the canonical funding-aware
subaccount and shared portfolio-array pipeline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

import training.audit_gross9_alt_flow_geometry_inference_marginal as accounting
import training.audit_gross9_residual_recovery_extratrees_marginal as gross9_runtime
import training.portfolio_opt_added_alpha_update as portfolio
from training.audit_gross9_oi_pullback_marginal import _sha256


PREREGISTRATION = Path(
    "results/cash_open_cross_asset_gap_relay_marginal_preregistration_2026-07-28.json"
)
SELECTION_OUTPUT = Path(
    "results/gross9_cash_open_cross_asset_gap_relay_selection_2023h2_2026-07-28.json"
)
EVAL_OUTPUT = Path(
    "results/gross9_cash_open_cross_asset_gap_relay_eval_2024_2026-07-28.json"
)
DOCS_SELECTION_OUTPUT = Path(
    "docs/gross9-cash-open-cross-asset-gap-relay-selection-2023h2-2026-07-28.md"
)
DOCS_EVAL_OUTPUT = Path(
    "docs/gross9-cash-open-cross-asset-gap-relay-eval-2024-2026-07-28.md"
)
AS_OF = "2026-07-28"
CANDIDATE_SLEEVE = "cogr"

FEATURE_COLUMNS = (
    "gap_open_qqq",
    "prior_close_return_1d_qqq",
    "prior_close_return_5d_qqq",
    "prior_close_return_20d_qqq",
    "prior_intraday_return_qqq",
    "prior_range_qqq",
    "prior_realized_vol_5d_qqq",
    "prior_realized_vol_20d_qqq",
    "prior_volume_z60_qqq",
    "gap_open_gld",
    "prior_close_return_1d_gld",
    "prior_close_return_5d_gld",
    "prior_close_return_20d_gld",
    "prior_intraday_return_gld",
    "prior_range_gld",
    "prior_realized_vol_5d_gld",
    "prior_realized_vol_20d_gld",
    "prior_volume_z60_gld",
    "gap_risk_rotation",
    "gap_joint_liquidity",
    "gap_abs_total",
    "gap_direction_agreement",
    "prior_return_spread_1d",
    "prior_return_spread_5d",
    "prior_return_spread_20d",
    "prior_return_corr_20d",
    "prior_return_beta_20d",
    "prior_vol_ratio_20d",
    "prior_volume_z_spread",
    "weekday_sin",
    "weekday_cos",
)
CURRENT_GAP_BLOCK = (
    "gap_open_qqq",
    "gap_open_gld",
    "gap_risk_rotation",
    "gap_joint_liquidity",
    "gap_abs_total",
    "gap_direction_agreement",
)
TARGET_COLUMNS = (
    "long_exact_net_return",
    "long_strict_adverse_excursion",
    "short_exact_net_return",
    "short_strict_adverse_excursion",
)
COORDINATION_MODES = (
    "unrestricted",
    "gross9_flat_at_signal",
    "gross9_drawdown_ge_5pct",
)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (7, 71, 715)
HOLD_BARS = 144
ENTRY_DELAY_BARS = 1
UNIT_LEVERAGE = 0.5
NORMAL_COST = 0.0006
STRESS_COST = 0.001
FIT_START = pd.Timestamp("2020-10-15")
MINIMUM_FIT_ROWS = 500
BASELINE_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
BASELINE_GROSS = 9.0
FEATURE_CONTROL_NAMES = (
    "qqq_only",
    "gld_only",
    "prior_only_no_current_open",
    "one_session_stale_current_open",
    "weekday_only",
)
FIXED_CONTROL_NAMES = (
    "exact_side_flip",
    "constant_long",
    "deterministic_random_side",
    "one_us_session_delayed_entry",
)
CONTROL_NAMES = FEATURE_CONTROL_NAMES + FIXED_CONTROL_NAMES
FOLDS = (
    {
        "name": "calibration_2023h1",
        "fit_end_exclusive": "2023-01-01",
        "prediction_start": "2023-01-01",
        "prediction_end_exclusive": "2023-07-01",
        "threshold_source": None,
        "candidate_entries_allowed": False,
        "outcomes_used_for_ranking": False,
    },
    {
        "name": "selection_2023h2",
        "fit_end_exclusive": "2023-07-01",
        "prediction_start": "2023-07-01",
        "prediction_end_exclusive": "2024-01-01",
        "threshold_source": "calibration_2023h1",
        "candidate_entries_allowed": True,
        "outcomes_used_for_ranking": True,
    },
    {
        "name": "eval_2024",
        "fit_end_exclusive": "2024-01-01",
        "prediction_start": "2024-01-01",
        "prediction_end_exclusive": "2025-01-01",
        "threshold_source": "selection_2023h2",
        "candidate_entries_allowed": True,
        "outcomes_used_for_ranking": False,
        "exact_frozen_top1_only": True,
    },
)
FOLD_BY_NAME = {str(row["name"]): row for row in FOLDS}
PHASE_FOLDS = {
    "selection": ("calibration_2023h1", "selection_2023h2"),
    "eval": (
        "calibration_2023h1",
        "selection_2023h2",
        "eval_2024",
    ),
}
PHASE_CUTOFFS = {
    "selection": pd.Timestamp("2024-01-01"),
    "eval": pd.Timestamp("2025-01-01"),
}
PHASE_SPLITS = {
    "selection": "train",
    "eval": "test2024",
}
WINDOWS = {
    "selection_2023h2": (
        pd.Timestamp("2023-07-01"),
        pd.Timestamp("2024-01-01"),
    ),
    "eval_2024": (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-01-01"),
    ),
}
EVAL_HALF_WINDOWS = {
    "first_calendar_half": (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-07-01"),
    ),
    "second_calendar_half": (
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2025-01-01"),
    ),
}
INPUT_KEYS = (
    "safe_features",
    "safe_feature_manifest",
    "safe_feature_builder",
    "market",
    "market_with_oi",
    "funding",
    "premium_1h",
    "gross9_pre2025_anchor",
    "rank7_capacity_evidence",
    "gross9_context_builder",
    "gross9_runtime",
    "gross9_portfolio_engine",
    "funding_bar_path",
    "execution_engine",
    "cogr_evaluator",
)


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    selection_output: str = str(SELECTION_OUTPUT)
    eval_output: str = str(EVAL_OUTPUT)
    docs_selection_output: str = str(DOCS_SELECTION_OUTPUT)
    docs_eval_output: str = str(DOCS_EVAL_OUTPUT)
    safe_features: str = (
        "data/cash_open_cross_asset_gap_relay_pre2025/"
        "qqq_gld_cash_open_safe_features_pre2025.csv.gz"
    )
    safe_feature_manifest: str = (
        "data/cash_open_cross_asset_gap_relay_pre2025/build_manifest.json"
    )
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    gross9_pre2025_anchor: str = (
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    )
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Index,)):
        return json_ready(value.tolist())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("COGR artifacts cannot contain NaN or infinity")
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def configured_inputs(
    cfg: Config,
    *,
    evaluator_path: str | Path | None = None,
) -> dict[str, str]:
    evaluator = Path(evaluator_path or __file__)
    return {
        "safe_features": cfg.safe_features,
        "safe_feature_manifest": cfg.safe_feature_manifest,
        "safe_feature_builder": "training/build_cash_open_cross_asset_gap_features.py",
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "funding": cfg.funding_csv,
        "premium_1h": cfg.premium_csv,
        "gross9_pre2025_anchor": cfg.gross9_pre2025_anchor,
        "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
        "gross9_context_builder": (
            "training/audit_gross9_fixed_candidate_state_substitution.py"
        ),
        "gross9_runtime": (
            "training/audit_gross9_residual_recovery_extratrees_marginal.py"
        ),
        "gross9_portfolio_engine": "training/portfolio_opt_added_alpha_update.py",
        "funding_bar_path": (
            "training/audit_rank7_fresh_kimchi_fixed_portfolio.py"
        ),
        "execution_engine": "training/search_inventory_purge_reclaim_alpha.py",
        "cogr_evaluator": str(evaluator),
    }


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(
            f"COGR preregistration semantic drift at {label}: "
            f"{observed!r} != {expected!r}"
        )


def validate_preregistration_semantics(payload: Mapping[str, Any]) -> None:
    _require_equal("name", payload.get("name"), "cash_open_cross_asset_gap_relay_gross9_marginal_battery")
    _require_equal("short_name", payload.get("short_name"), "COGR-12")
    _require_equal("created_on", payload.get("created_on"), AS_OF)
    _require_equal(
        "physical_selection_cutoff",
        payload.get("physical_selection_cutoff"),
        "2024-01-01",
    )

    source = payload["source_contract"]
    _require_equal("source.feature_count", source.get("feature_count"), 31)
    _require_equal(
        "source.feature_available_local_time",
        source.get("feature_available_local_time"),
        "09:35 America/New_York",
    )
    _require_equal(
        "source.entry_local_time",
        source.get("entry_local_time"),
        "09:40 America/New_York",
    )
    _require_equal(
        "source.selection_cutoff_exclusive",
        source.get("selection_cutoff_exclusive"),
        "2024-01-01",
    )
    _require_equal(
        "source.source_artifact_cutoff_exclusive",
        source.get("source_artifact_cutoff_exclusive"),
        "2025-01-01",
    )
    _require_equal(
        "source.safe_feature_path",
        source.get("safe_feature_path"),
        (
            "data/cash_open_cross_asset_gap_relay_pre2025/"
            "qqq_gld_cash_open_safe_features_pre2025.csv.gz"
        ),
    )
    _require_equal(
        "source.safe_manifest_path",
        source.get("safe_manifest_path"),
        "data/cash_open_cross_asset_gap_relay_pre2025/build_manifest.json",
    )
    feature = payload["feature_contract"]
    _require_equal("feature.columns", tuple(feature["columns"]), FEATURE_COLUMNS)
    _require_equal(
        "feature.current_gap_block",
        tuple(feature["current_gap_block"]),
        CURRENT_GAP_BLOCK,
    )
    _require_equal(
        "feature.training_imputation",
        feature.get("training_imputation"),
        "none; every admitted fit and prediction row must be finite",
    )
    _require_equal(
        "feature.scaling",
        feature.get("scaling"),
        "none; ExtraTrees receives the frozen float64 values directly",
    )

    learner = payload["learner_contract"]
    expected_learner = {
        "estimator": "sklearn.ensemble.ExtraTreesRegressor",
        "sklearn_version": "1.7.2",
        "n_estimators": 256,
        "max_depth": 4,
        "min_samples_leaf": 32,
        "max_features": 0.75,
        "bootstrap": False,
        "n_jobs": 1,
        "fit_start_inclusive": "2020-10-15",
        "minimum_fit_rows": 500,
        "risk_lambda": 0.5,
        "seed_uncertainty_lambda": 0.5,
        "minimum_score": 0.0,
        "score_quantile": 0.75,
        "unit_leverage": UNIT_LEVERAGE,
        "normal_cost_per_notional_per_side": NORMAL_COST,
        "candidate_stress_cost_per_notional_per_side": STRESS_COST,
        "entry_delay_bars": ENTRY_DELAY_BARS,
        "hold_bars": HOLD_BARS,
        "tp": None,
        "sl": None,
    }
    for key, expected in expected_learner.items():
        _require_equal(f"learner.{key}", learner.get(key), expected)
    _require_equal("learner.seeds", tuple(learner["seeds"]), SEEDS)
    _require_equal(
        "learner.target_columns",
        tuple(learner["target_columns"]),
        TARGET_COLUMNS,
    )
    _require_equal("runtime.sklearn_version", sklearn.__version__, learner["sklearn_version"])

    _require_equal("expanding_folds", payload["expanding_folds"], list(FOLDS))
    universe = payload["candidate_universe"]
    _require_equal(
        "candidate.coordination_modes",
        tuple(row["name"] for row in universe["coordination_modes"]),
        COORDINATION_MODES,
    )
    _require_equal(
        "candidate.tie_break",
        tuple(universe["coordination_tie_break_order"]),
        COORDINATION_MODES,
    )
    _require_equal(
        "candidate.weights",
        tuple(float(value) for value in universe["candidate_weight_grid"]),
        WEIGHTS,
    )
    _require_equal("candidate.policy_cells", universe.get("policy_cells"), 3)
    _require_equal("candidate.portfolio_cells", universe.get("portfolio_cells"), 12)

    controls = payload["controls"]
    _require_equal(
        "controls.feature_names",
        tuple(row["name"] for row in controls["feature_model_controls"]),
        FEATURE_CONTROL_NAMES,
    )
    _require_equal(
        "controls.fixed_names",
        tuple(row["name"] for row in controls["fixed_schedule_controls"]),
        FIXED_CONTROL_NAMES,
    )

    selection = payload["selection_contract"]
    _require_equal(
        "selection.windows",
        (
            selection["calibration_window"]["start_inclusive"],
            selection["calibration_window"]["end_exclusive"],
            selection["selection_window"]["start_inclusive"],
            selection["selection_window"]["end_exclusive"],
            selection["eval_window"]["start_inclusive"],
            selection["eval_window"]["end_exclusive"],
        ),
        (
            "2023-01-01",
            "2023-07-01",
            "2023-07-01",
            "2024-01-01",
            "2024-01-01",
            "2025-01-01",
        ),
    )
    _require_equal(
        "selection.standalone_requirements",
        selection["selection_2023h2_standalone_requirements"],
        {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 1.5,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_trades": 25,
            "minimum_long_share": 0.2,
            "minimum_short_share": 0.2,
            "maximum_month_share": 0.35,
            "maximum_weekday_share": 0.35,
            "candidate_10bp_stress_absolute_return_positive": True,
        },
    )
    _require_equal(
        "selection.portfolio_requirements",
        selection["selection_2023h2_portfolio_requirements"],
        {
            "maximum_strict_mdd_pct": 20.0,
            "absolute_return_retention_floor_vs_unscaled_gross9": 0.97,
            "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9": 0.05,
            "strict_mdd_reduction_vs_unscaled_gross9": True,
            "maximum_exact_entry_jaccard_vs_any_gross9_sleeve": 0.25,
            "candidate_10bp_stress_portfolio_absolute_return_positive": True,
        },
    )
    _require_equal(
        "selection.mechanism_requirements",
        selection["selection_2023h2_mechanism_requirements"],
        {"primary_cagr_mdd_margin_over_best_of_all_nine_controls": 0.1},
    )
    _require_equal("selection.top1_only", selection.get("top1_only"), True)

    evaluation = payload["eval_2024_veto_contract"]
    _require_equal(
        "eval.exact_top1",
        evaluation.get("exact_frozen_2023h2_top1_only"),
        True,
    )
    _require_equal(
        "eval.standalone_requirements",
        evaluation["standalone_requirements"],
        {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 1.5,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_trades": 50,
            "minimum_long_share": 0.2,
            "minimum_short_share": 0.2,
            "maximum_month_share": 0.2,
            "maximum_weekday_share": 0.3,
            "first_calendar_half_absolute_return_positive": True,
            "second_calendar_half_absolute_return_positive": True,
            "candidate_10bp_stress_absolute_return_positive": True,
        },
    )
    _require_equal(
        "eval.portfolio_requirements",
        evaluation["portfolio_requirements"],
        {
            "maximum_strict_mdd_pct": 20.0,
            "absolute_return_retention_floor_vs_unscaled_gross9": 0.97,
            "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9": 0.05,
            "strict_mdd_reduction_vs_unscaled_gross9": True,
            "maximum_exact_entry_jaccard_vs_any_gross9_sleeve": 0.25,
            "candidate_10bp_stress_portfolio_absolute_return_positive": True,
        },
    )
    _require_equal(
        "eval.mechanism_requirements",
        evaluation["mechanism_requirements"],
        {"primary_cagr_mdd_margin_over_best_of_all_nine_controls": 0.0},
    )
    statistical = evaluation["statistical_requirement"]
    for key, expected in {
        "simulations": 10_000,
        "seed": 20260728,
        "minimum_active_weeks_each_test": 26,
        "maximum_p_value_each_test": 0.1,
    }.items():
        _require_equal(f"eval.statistical.{key}", statistical.get(key), expected)

    same_gross = payload["same_gross_formula"]
    _require_equal(
        "same_gross.baseline_weights",
        {str(key): float(value) for key, value in same_gross["baseline_weights"].items()},
        BASELINE_WEIGHTS,
    )
    _require_equal(
        "same_gross.baseline_configured_gross_units",
        float(same_gross["baseline_configured_gross_units"]),
        BASELINE_GROSS,
    )
    phase = payload["phase_contract"]
    _require_equal(
        "phase.selection_command",
        phase.get("selection_command"),
        "evaluate_cash_open_cross_asset_gap_relay_marginal.py selection",
    )
    _require_equal(
        "phase.eval_command",
        phase.get("eval_command"),
        "evaluate_cash_open_cross_asset_gap_relay_marginal.py eval",
    )
    _require_equal(
        "phase.selection_reads_through",
        phase.get("selection_reads_candidate_labels_through"),
        "2023-12-31T23:59:59.999999999Z",
    )
    _require_equal(
        "phase.eval_reads_through",
        phase.get("eval_reads_candidate_labels_through"),
        "2024-12-31T23:59:59.999999999Z",
    )
    _require_equal(
        "phase.eval_requires_freeze",
        phase.get("eval_requires_passing_selection_freeze"),
        True,
    )
    _require_equal(
        "phase.selection_ranks_on",
        phase.get("selection_ranks_on"),
        "2023H2 only",
    )
    _require_equal("phase.eval_ranks_on", phase.get("eval_ranks_on"), None)
    _require_equal(
        "phase.selection_output",
        phase.get("selection_output"),
        str(SELECTION_OUTPUT),
    )
    _require_equal(
        "phase.eval_output",
        phase.get("eval_output"),
        str(EVAL_OUTPUT),
    )
    provenance = payload["input_provenance"]
    _require_equal("input_provenance.keys", tuple(provenance), INPUT_KEYS)


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preregistration_semantics(payload)
    return payload


def validate_inputs(
    cfg: Config,
    preregistration: Mapping[str, Any],
    *,
    evaluator_path: str | Path | None = None,
    configured_override: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    evaluator = Path(evaluator_path or __file__).resolve(strict=True)
    configured = dict(
        configured_override
        if configured_override is not None
        else configured_inputs(cfg, evaluator_path=evaluator)
    )
    if tuple(configured) != INPUT_KEYS:
        raise RuntimeError("configured COGR input order drifted")
    provenance = preregistration["input_provenance"]
    if tuple(provenance) != INPUT_KEYS:
        raise RuntimeError("preregistered COGR input order drifted")
    if Path(configured["cogr_evaluator"]).resolve(strict=True) != evaluator:
        raise RuntimeError("configured cogr_evaluator is not Path(__file__)")

    records: dict[str, dict[str, Any]] = {}
    for name in INPUT_KEYS:
        configured_path = Path(configured[name]).resolve(strict=True)
        frozen_path = Path(provenance[name]["path"]).resolve(strict=True)
        if configured_path != frozen_path:
            raise RuntimeError(f"input path drifted for {name}")
        observed = _sha256(configured_path)
        expected = str(provenance[name]["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"input hash drifted for {name}: {observed} != {expected}"
            )
        records[name] = {
            "path": str(provenance[name]["path"]),
            "sha256": observed,
            "validated_against_preregistration": True,
        }
    return records


def _utc_naive(value: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True)).tz_convert(None)


def _open_text(path: str | Path):
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8", newline="")
    return source.open("rt", encoding="utf-8", newline="")


def read_csv_prefix(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp,
    date_columns: Sequence[str],
) -> pd.DataFrame:
    """Stop before parsing numeric fields from the first row at/after cutoff."""
    boundary = _utc_naive(cutoff)
    rows: list[list[str]] = []
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = list(next(reader))
        except StopIteration as exc:
            raise RuntimeError(f"empty CSV: {path}") from exc
        date_column = next((name for name in date_columns if name in header), None)
        if date_column is None:
            raise RuntimeError(
                f"{path} lacks every date column in {tuple(date_columns)}"
            )
        date_position = header.index(date_column)
        previous: pd.Timestamp | None = None
        for raw in reader:
            if len(raw) != len(header):
                raise RuntimeError(f"malformed CSV row in {path}")
            timestamp = _utc_naive(raw[date_position])
            if previous is not None and timestamp < previous:
                raise RuntimeError(f"source is not time ordered: {path}")
            previous = timestamp
            if timestamp >= boundary:
                break
            rows.append(raw)
    if not rows:
        raise RuntimeError(f"no rows before {boundary} in {path}")
    return pd.DataFrame(rows, columns=header)


def read_market_prefix(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    frame = read_csv_prefix(path, cutoff=cutoff, date_columns=("date",))
    required = ("date", "open", "high", "low", "close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"market prefix missing columns: {missing}")
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    for name in required[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    dates = pd.DatetimeIndex(frame["date"])
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise RuntimeError("candidate market prefix clock is not unique/increasing")
    if dates[-1] >= _utc_naive(cutoff):
        raise RuntimeError("candidate market prefix crossed the phase cutoff")
    values = frame.loc[:, required[1:]].to_numpy(np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("candidate market OHLC is invalid")
    return frame.reset_index(drop=True)


def _bool_values(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    if not normalized.isin(mapping).all():
        raise RuntimeError("feature_valid contains a non-boolean value")
    return normalized.map(mapping).to_numpy(bool)


def load_safe_features(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = (
        read_csv_prefix(
            path,
            cutoff=cutoff,
            date_columns=("session_date", "date"),
        )
        if cutoff is not None
        else pd.read_csv(path)
    )
    date_column = "session_date" if "session_date" in frame.columns else "date"
    frame["session_date"] = (
        pd.to_datetime(frame[date_column], errors="raise")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if "feature_valid" not in frame.columns:
        raise RuntimeError("safe features missing feature_valid")
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"safe features missing columns: {missing}")
    for name in FEATURE_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame = (
        frame.sort_values("session_date")
        .drop_duplicates("session_date", keep="last")
        .reset_index(drop=True)
    )
    finite = np.isfinite(frame.loc[:, FEATURE_COLUMNS].to_numpy(float)).all(axis=1)
    valid = _bool_values(frame["feature_valid"]) & finite
    output = frame.loc[valid, ["session_date", *FEATURE_COLUMNS]].reset_index(
        drop=True
    )
    if cutoff is not None and (
        output["session_date"] >= pd.Timestamp(cutoff).tz_localize(None)
    ).any():
        raise RuntimeError("safe feature prefix crossed the phase cutoff")
    return output


def read_funding_prefix(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    with _open_text(path) as handle:
        try:
            source_header = tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise RuntimeError(f"empty funding CSV: {path}") from exc
    expected_header = (
        "date",
        "symbol",
        "funding_rate",
        "funding_time",
        "mark_price",
    )
    if source_header != expected_header:
        raise RuntimeError(
            "funding source schema drifted: "
            f"{source_header!r} != {expected_header!r}"
        )
    frame = accounting.read_funding_prefix(path, cutoff=cutoff)
    if tuple(frame.columns) != ("date", "funding_rate"):
        raise RuntimeError("funding prefix schema drifted")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    frame["funding_rate"] = pd.to_numeric(
        frame["funding_rate"], errors="raise"
    )
    dates = pd.DatetimeIndex(frame["date"])
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise RuntimeError("funding prefix clock is not unique/increasing")
    if dates[-1] >= _utc_naive(cutoff):
        raise RuntimeError("funding prefix crossed the phase cutoff")
    if not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
        raise RuntimeError("funding prefix contains non-finite rates")
    return frame.reset_index(drop=True)


def feature_times_utc(session_dates: Sequence[Any]) -> pd.DatetimeIndex:
    zone = ZoneInfo("America/New_York")
    dates = pd.DatetimeIndex(pd.to_datetime(session_dates)).tz_localize(None).normalize()
    output = []
    for date in dates:
        local = (date + pd.Timedelta(hours=9, minutes=35)).tz_localize(zone)
        output.append(local.tz_convert("UTC").tz_localize(None))
    return pd.DatetimeIndex(output)


def _market_dates(market: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    )


def align_feature_sessions_to_market(
    features: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    dates = _market_dates(market)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise RuntimeError("market clock must be unique and increasing")
    if len(dates) < 3:
        raise RuntimeError("market clock is too short")
    all_signal_times = feature_times_utc(features["session_date"])
    in_clock = (all_signal_times >= dates[0]) & (all_signal_times <= dates[-1])
    out = features.loc[in_clock].copy().reset_index(drop=True)
    signal_times = all_signal_times[in_clock]
    lookup = {timestamp: position for position, timestamp in enumerate(dates)}
    zone = ZoneInfo("America/New_York")
    rows: list[tuple[int, int, int, int]] = []
    for session_date, signal_time in zip(
        pd.DatetimeIndex(out["session_date"]),
        signal_times,
        strict=True,
    ):
        coordination_time = signal_time - pd.Timedelta(minutes=5)
        entry_time = signal_time + pd.Timedelta(minutes=5)
        exit_time = entry_time + pd.Timedelta(hours=12)
        signal = lookup.get(signal_time, -1)
        coordination = lookup.get(coordination_time, -1)
        entry = lookup.get(entry_time, -1)
        if min(signal, coordination, entry) < 0:
            raise RuntimeError(
                "missing exact 09:30/09:35/09:40 Binance bar-open alignment "
                f"for {session_date.date()}"
            )
        if coordination != signal - 1 or entry != signal + 1:
            raise RuntimeError("COGR latency clock is not one exact five-minute bar")
        exit_position = lookup.get(exit_time, -1)
        if exit_position < 0 and exit_time <= dates[-1]:
            raise RuntimeError(
                f"missing exact +12h exit timestamp for {session_date.date()}"
            )
        if exit_position >= 0 and exit_position != entry + HOLD_BARS:
            raise RuntimeError("COGR exit is not exactly entry+144 bars")
        local_signal = signal_time.tz_localize("UTC").tz_convert(zone)
        local_entry = entry_time.tz_localize("UTC").tz_convert(zone)
        if (
            local_signal.date() != session_date.date()
            or (local_signal.hour, local_signal.minute) != (9, 35)
            or local_entry.date() != session_date.date()
            or (local_entry.hour, local_entry.minute) != (9, 40)
        ):
            raise RuntimeError("COGR NY signal/entry clock drifted")
        rows.append((signal, coordination, entry, exit_position))
    if not rows:
        raise RuntimeError("no safe feature sessions overlap the candidate market")
    positions = np.asarray(rows, dtype=np.int64)
    out["feature_time_utc"] = signal_times
    out["signal_position"] = positions[:, 0]
    out["coordination_position"] = positions[:, 1]
    out["entry_position"] = positions[:, 2]
    out["exit_position"] = positions[:, 3]
    return out


def accounting_config(
    cfg: Config | None = None,
    *,
    cutoff: str | pd.Timestamp,
    cost_rate: float,
) -> accounting.Config:
    return accounting.Config(
        market_csv="" if cfg is None else cfg.market_csv,
        market_with_oi_csv="" if cfg is None else cfg.market_with_oi_csv,
        funding_csv="" if cfg is None else cfg.funding_csv,
        premium_csv="" if cfg is None else cfg.premium_csv,
        gross9_pre2025_anchor="" if cfg is None else cfg.gross9_pre2025_anchor,
        rank7_capacity_evidence="" if cfg is None else cfg.rank7_capacity_evidence,
        cutoff=str(pd.Timestamp(cutoff).date()),
        cost_rate=float(cost_rate),
        stress_cost_rate=STRESS_COST,
        leverage=UNIT_LEVERAGE,
    )


def exact_targets(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    signals: np.ndarray,
    *,
    cost_rate: float = NORMAL_COST,
    cfg: Config | None = None,
    cutoff: str | pd.Timestamp | None = None,
) -> np.ndarray:
    boundary = cutoff or (
        _market_dates(market)[-1] + pd.Timedelta(minutes=5)
    )
    exact_funding = funding
    if len(exact_funding) == 0:
        exact_funding = pd.DataFrame(
            {
                "date": [_market_dates(market)[0] - pd.Timedelta(days=1)],
                "funding_rate": [0.0],
            }
        )
    targets, _ = accounting.build_targets(
        market,
        exact_funding,
        np.asarray(signals, dtype=np.int64),
        accounting_config(cfg, cutoff=boundary, cost_rate=cost_rate),
    )
    return np.asarray(targets, dtype=np.float64)


def contained(
    row: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    market_dates: pd.DatetimeIndex,
) -> bool:
    positions = (
        int(row["coordination_position"]),
        int(row["signal_position"]),
        int(row["entry_position"]),
        int(row["exit_position"]),
    )
    if min(positions) < 0 or max(positions) >= len(market_dates):
        return False
    return all(start <= market_dates[position] < end for position in positions)


def fit_mask(
    aligned: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    fit_end_exclusive: str,
) -> np.ndarray:
    end = pd.Timestamp(fit_end_exclusive)
    finite = np.isfinite(
        aligned.loc[:, FEATURE_COLUMNS].to_numpy(np.float64)
    ).all(axis=1)
    return np.asarray(
        [
            finite[index]
            and pd.Timestamp(row["feature_time_utc"]) >= FIT_START
            and contained(row, FIT_START, end, market_dates)
            for index, (_, row) in enumerate(aligned.iterrows())
        ],
        dtype=bool,
    )


def prediction_mask(
    aligned: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    start: str,
    end: str,
) -> np.ndarray:
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    finite = np.isfinite(
        aligned.loc[:, FEATURE_COLUMNS].to_numpy(np.float64)
    ).all(axis=1)
    return np.asarray(
        [
            finite[index] and contained(row, lower, upper, market_dates)
            for index, (_, row) in enumerate(aligned.iterrows())
        ],
        dtype=bool,
    )


def model_hash(model: ExtraTreesRegressor) -> str:
    return hashlib.sha256(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def fit_predict(
    matrix: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    preregistration: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    learner = preregistration["learner_contract"]
    minimum = int(learner.get("minimum_fit_rows", MINIMUM_FIT_ROWS))
    if int(fit.sum()) < minimum:
        raise RuntimeError(f"insufficient fit rows: {int(fit.sum())}")
    if not predict.any():
        raise RuntimeError("empty prediction fold")
    x_fit = np.asarray(matrix[fit], dtype=np.float64)
    y_fit = np.asarray(targets[fit], dtype=np.float64)
    x_predict = np.asarray(matrix[predict], dtype=np.float64)
    if (
        not np.isfinite(x_fit).all()
        or not np.isfinite(y_fit).all()
        or not np.isfinite(x_predict).all()
    ):
        raise RuntimeError("COGR admits no non-finite model rows")
    predictions: list[np.ndarray] = []
    hashes: list[str] = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=int(learner["n_estimators"]),
            max_depth=int(learner["max_depth"]),
            min_samples_leaf=int(learner["min_samples_leaf"]),
            max_features=float(learner["max_features"]),
            bootstrap=False,
            random_state=int(seed),
            n_jobs=1,
        ).fit(x_fit, y_fit)
        predictions.append(model.predict(x_predict))
        hashes.append(model_hash(model))
    stacked = np.stack(predictions, axis=0)
    return stacked, {
        "fit_rows": int(fit.sum()),
        "predict_rows": int(predict.sum()),
        "model_hashes": hashes,
        "prediction_hash": array_hash(stacked),
    }


def adjusted_scores(
    predictions: np.ndarray,
    risk_lambda: float = 0.5,
    uncertainty_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    long_utility = values[:, :, 0] - risk_lambda * values[:, :, 1]
    short_utility = values[:, :, 2] - risk_lambda * values[:, :, 3]
    long_score = long_utility.mean(axis=0) - uncertainty_lambda * long_utility.std(
        axis=0, ddof=0
    )
    short_score = short_utility.mean(
        axis=0
    ) - uncertainty_lambda * short_utility.std(axis=0, ddof=0)
    choose_long = long_score >= short_score
    return (
        np.where(choose_long, long_score, short_score),
        np.where(choose_long, 1, -1).astype(np.int8),
        long_score,
        short_score,
    )


def feature_subset(frame: pd.DataFrame, control: str | None) -> pd.DataFrame:
    if control is None:
        return frame.loc[:, FEATURE_COLUMNS]
    if control == "qqq_only":
        columns = [
            name for name in FEATURE_COLUMNS if name.endswith("_qqq")
        ] + ["weekday_sin", "weekday_cos"]
    elif control == "gld_only":
        columns = [
            name for name in FEATURE_COLUMNS if name.endswith("_gld")
        ] + ["weekday_sin", "weekday_cos"]
    elif control == "prior_only_no_current_open":
        columns = [
            name for name in FEATURE_COLUMNS if name not in CURRENT_GAP_BLOCK
        ]
    elif control == "one_session_stale_current_open":
        stale = frame.loc[:, FEATURE_COLUMNS].copy()
        stale.loc[:, CURRENT_GAP_BLOCK] = stale.loc[:, CURRENT_GAP_BLOCK].shift(1)
        return stale
    elif control == "weekday_only":
        columns = ["weekday_sin", "weekday_cos"]
    else:
        raise KeyError(control)
    return frame.loc[:, columns]


def phase_fold_names(
    *,
    phase: str | None,
    allowed_folds: Sequence[str] | None,
) -> tuple[str, ...]:
    if phase is None and allowed_folds is None:
        raise ValueError("fold_predictions requires phase or allowed_folds")
    if phase is not None:
        if phase not in PHASE_FOLDS:
            raise ValueError(f"unknown COGR phase: {phase}")
        expected = PHASE_FOLDS[phase]
        if allowed_folds is not None and tuple(allowed_folds) != expected:
            raise RuntimeError(f"{phase} fold universe drifted")
        names = expected
    else:
        names = tuple(str(name) for name in allowed_folds or ())
    if not names or any(name not in FOLD_BY_NAME for name in names):
        raise RuntimeError("invalid COGR fold list")
    for name in names:
        source = FOLD_BY_NAME[name]["threshold_source"]
        if source is not None and str(source) not in names:
            raise RuntimeError(f"threshold source {source} was not computed")
    return names


def prepare_phase_targets(
    aligned: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    phase: str | None = None,
    allowed_folds: Sequence[str] | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    names = phase_fold_names(phase=phase, allowed_folds=allowed_folds)
    cutoff = max(
        pd.Timestamp(FOLD_BY_NAME[name]["prediction_end_exclusive"])
        for name in names
    )
    market_dates = _market_dates(market)
    if market_dates[-1] >= cutoff:
        raise RuntimeError("candidate market contains rows at/after phase cutoff")
    complete = (
        (aligned["exit_position"].to_numpy(np.int64) >= 0)
        & (pd.to_datetime(aligned["feature_time_utc"]) < cutoff).to_numpy()
    )
    work = aligned.loc[complete].copy()
    work["_source_row"] = np.flatnonzero(complete)
    work = work.reset_index(drop=True)
    signals = work["signal_position"].to_numpy(np.int64)
    targets = exact_targets(
        market,
        funding,
        signals,
        cost_rate=NORMAL_COST,
        cfg=cfg,
        cutoff=cutoff,
    )
    if len(targets) != len(work):
        raise RuntimeError("target row count drifted")
    return {
        "aligned": work,
        "targets": targets,
        "allowed_folds": names,
        "cutoff": cutoff,
        "target_hash": array_hash(targets),
        "target_rows": int(len(targets)),
        "target_last_signal": (
            str(market_dates[signals[-1]]) if len(signals) else None
        ),
    }


def fold_predictions(
    aligned: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    preregistration: Mapping[str, Any],
    *,
    phase: str | None = None,
    allowed_folds: Sequence[str] | None = None,
    control: str | None = None,
    prepared: Mapping[str, Any] | None = None,
    cfg: Config | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    names = phase_fold_names(phase=phase, allowed_folds=allowed_folds)
    phase_data = (
        dict(prepared)
        if prepared is not None
        else prepare_phase_targets(
            aligned,
            market,
            funding,
            phase=phase,
            allowed_folds=allowed_folds,
            cfg=cfg,
        )
    )
    if tuple(phase_data["allowed_folds"]) != names:
        raise RuntimeError("prepared target fold universe drifted")
    work = phase_data["aligned"]
    targets = np.asarray(phase_data["targets"], dtype=np.float64)
    matrix = feature_subset(work, control).to_numpy(np.float64)
    dates = _market_dates(market)
    finite_targets = np.isfinite(targets).all(axis=1)
    finite_matrix = np.isfinite(matrix).all(axis=1)
    output: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "variant": control or "primary",
        "allowed_folds": list(names),
        "target_hash": str(phase_data["target_hash"]),
        "target_rows": int(phase_data["target_rows"]),
        "target_last_signal": phase_data["target_last_signal"],
        "folds": {},
    }
    for name in names:
        fold = FOLD_BY_NAME[name]
        fit = (
            fit_mask(work, dates, str(fold["fit_end_exclusive"]))
            & finite_targets
            & finite_matrix
        )
        predict = (
            prediction_mask(
                work,
                dates,
                str(fold["prediction_start"]),
                str(fold["prediction_end_exclusive"]),
            )
            & finite_matrix
        )
        predictions, fold_meta = fit_predict(
            matrix, targets, fit, predict, preregistration
        )
        score, side, long_score, short_score = adjusted_scores(predictions)
        local_rows = np.flatnonzero(predict)
        source_rows = work.iloc[local_rows]["_source_row"].to_numpy(np.int64)
        output[name] = {
            "rows": source_rows,
            "score": score,
            "side": side,
            "long_score": long_score,
            "short_score": short_score,
        }
        fold_meta.update(
            {
                "fit_end_exclusive": fold["fit_end_exclusive"],
                "prediction_start": fold["prediction_start"],
                "prediction_end_exclusive": fold["prediction_end_exclusive"],
                "score_hash": array_hash(score),
                "side_hash": array_hash(side),
            }
        )
        metadata["folds"][name] = fold_meta
    for name in names:
        source = FOLD_BY_NAME[name]["threshold_source"]
        if source is None:
            continue
        prior = np.asarray(output[str(source)]["score"], dtype=np.float64)
        finite = prior[np.isfinite(prior)]
        if not len(finite):
            raise RuntimeError("empty prior score fold")
        raw_q75 = float(np.quantile(finite, 0.75))
        threshold = max(0.0, raw_q75)
        output[name]["threshold"] = threshold
        metadata["folds"][name]["threshold"] = {
            "source": source,
            "raw_q75": raw_q75,
            "threshold": threshold,
            "outcomes_used": False,
        }
    return output, metadata


def prediction_variants(
    aligned: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    preregistration: Mapping[str, Any],
    *,
    phase: str,
    cfg: Config | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    prepared = prepare_phase_targets(
        aligned, market, funding, phase=phase, cfg=cfg
    )
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for control in (None, *FEATURE_CONTROL_NAMES):
        name = control or "primary"
        predictions[name], metadata[name] = fold_predictions(
            aligned,
            market,
            funding,
            preregistration,
            phase=phase,
            control=control,
            prepared=prepared,
            cfg=cfg,
        )
    return predictions, {
        "phase": phase,
        "prepared_target_hash": prepared["target_hash"],
        "prepared_target_rows": prepared["target_rows"],
        "variants": metadata,
    }


def gross9_state_at_signal(
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    base_events: list[dict[str, Any]],
    baseline_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flat, drawdown, metadata = gross9_runtime.gross9_state(
        market, masks, base_events, baseline_weights
    )
    return np.asarray(flat, bool), np.asarray(drawdown, float), metadata


def mode_gate(
    mode: str,
    coordination_positions: np.ndarray,
    flat: np.ndarray,
    drawdown: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(coordination_positions, dtype=np.int64)
    if len(positions) and (
        positions.min() < 0 or positions.max() >= len(flat)
    ):
        raise RuntimeError("coordination position is outside Gross9 state arrays")
    if mode == "unrestricted":
        return np.ones(len(positions), dtype=bool)
    if mode == "gross9_flat_at_signal":
        return np.asarray(flat[positions], dtype=bool)
    if mode == "gross9_drawdown_ge_5pct":
        return np.asarray(drawdown[positions] >= 0.05, dtype=bool)
    raise KeyError(mode)


def accepted_schedule(
    aligned: pd.DataFrame,
    fold_rows: Mapping[str, Any],
    mode: str,
    flat: np.ndarray,
    drawdown: np.ndarray,
) -> list[dict[str, Any]]:
    rows = np.asarray(fold_rows["rows"], dtype=np.int64)
    score = np.asarray(fold_rows["score"], dtype=np.float64)
    side = np.asarray(fold_rows["side"], dtype=np.int8)
    if not (len(rows) == len(score) == len(side)):
        raise RuntimeError("prediction row/score/side lengths drifted")
    coordinates = aligned.iloc[rows]["coordination_position"].to_numpy(np.int64)
    gate = mode_gate(mode, coordinates, flat, drawdown)
    active = (
        np.isfinite(score)
        & (score >= float(fold_rows["threshold"]))
        & gate
    )
    output: list[dict[str, Any]] = []
    next_signal_allowed = -1
    for source_row, value, direction, admitted in zip(
        rows, score, side, active, strict=True
    ):
        if not admitted:
            continue
        row = aligned.iloc[int(source_row)]
        signal = int(row["signal_position"])
        exit_position = int(row["exit_position"])
        if exit_position < 0 or signal <= next_signal_allowed:
            continue
        output.append(
            {
                "source_row": int(source_row),
                "session_date": str(pd.Timestamp(row["session_date"]).date()),
                "signal_position": signal,
                "coordination_position": int(row["coordination_position"]),
                "entry_position": int(row["entry_position"]),
                "exit_position": exit_position,
                "side": int(direction),
                "score": float(value),
            }
        )
        next_signal_allowed = exit_position
    return output


def deterministic_random_side(session_date: str) -> int:
    bit = hashlib.sha256(f"COGR-12|{session_date}".encode("utf-8")).digest()[-1] & 1
    return 1 if bit else -1


def delayed_control(
    primary_schedule: Sequence[Mapping[str, Any]],
    aligned: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    sessions = list(pd.DatetimeIndex(aligned["session_date"]).date)
    session_index = {date: index for index, date in enumerate(sessions)}
    output: list[dict[str, Any]] = []
    next_signal_allowed = -1
    for item in primary_schedule:
        original = pd.Timestamp(item["session_date"]).date()
        index = session_index.get(original, -1)
        if index < 0 or index + 1 >= len(aligned):
            continue
        row = aligned.iloc[index + 1]
        delayed = {
            **dict(item),
            "source_row": int(index + 1),
            "session_date": str(pd.Timestamp(row["session_date"]).date()),
            "signal_position": int(row["signal_position"]),
            "coordination_position": int(row["coordination_position"]),
            "entry_position": int(row["entry_position"]),
            "exit_position": int(row["exit_position"]),
        }
        if int(delayed["signal_position"]) <= next_signal_allowed:
            continue
        if contained(delayed, start, end, market_dates):
            output.append(delayed)
            next_signal_allowed = int(delayed["exit_position"])
    for left, right in zip(output, output[1:]):
        if int(right["signal_position"]) <= int(left["exit_position"]):
            raise RuntimeError("delayed control overlap survived suppression")
    return output


def _install_candidate_sleeve() -> None:
    if CANDIDATE_SLEEVE not in portfolio.SLEEVES:
        portfolio.SLEEVES = (*portfolio.SLEEVES, CANDIDATE_SLEEVE)
    portfolio.FAMILIES[CANDIDATE_SLEEVE] = "cash_open_cross_asset_gap_relay"


def split_mask_for_phase(
    market: pd.DataFrame,
    *,
    phase: str,
) -> dict[str, np.ndarray]:
    split = PHASE_SPLITS[phase]
    start, end = portfolio.SPLIT_BOUNDS[split]
    dates = _market_dates(market)
    mask = np.asarray(
        (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)),
        dtype=bool,
    )
    if not mask.any():
        raise RuntimeError(f"candidate {split} mask is empty")
    return {split: mask}


def canonical_schedule(
    specifications: Sequence[Mapping[str, Any]],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cfg: Config,
    *,
    cutoff: pd.Timestamp,
) -> tuple[list[Any], dict[str, Any]]:
    _install_candidate_sleeve()
    long_active = np.zeros(len(market), dtype=bool)
    short_active = np.zeros(len(market), dtype=bool)
    for item in specifications:
        signal = int(item["signal_position"])
        side = int(item["side"])
        if signal < 0 or signal >= len(market) or side not in (-1, 1):
            raise RuntimeError("invalid COGR schedule specification")
        target = long_active if side > 0 else short_active
        other = short_active if side > 0 else long_active
        if other[signal]:
            raise RuntimeError("both COGR sides active at one signal")
        target[signal] = True
    schedules, metadata = accounting.build_schedules(
        market,
        funding,
        masks,
        {
            CANDIDATE_SLEEVE: {
                "long_active": long_active,
                "short_active": short_active,
            }
        },
        accounting_config(cfg, cutoff=cutoff, cost_rate=NORMAL_COST),
    )
    split = next(iter(masks))
    return schedules[CANDIDATE_SLEEVE][split], metadata[CANDIDATE_SLEEVE][split]


def _specifications_from_trades(
    trades: Sequence[Any],
    primary_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_signal = {
        int(item["signal_position"]): dict(item) for item in primary_specs
    }
    output = []
    for trade in trades:
        signal = int(trade.signal_position)
        if signal not in by_signal:
            raise RuntimeError("canonical trade lost its accepted source signal")
        output.append({**by_signal[signal], "side": int(trade.side)})
    return output


def schedules_for_mode(
    mode: str,
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aligned: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    flat: np.ndarray,
    drawdown: np.ndarray,
    cfg: Config,
    *,
    phase: str,
) -> tuple[list[Any], dict[str, list[Any]], dict[str, Any]]:
    fold_name = PHASE_FOLDS[phase][-1]
    masks = split_mask_for_phase(market, phase=phase)
    cutoff = PHASE_CUTOFFS[phase]
    primary_specs = accepted_schedule(
        aligned, predictions["primary"][fold_name], mode, flat, drawdown
    )
    primary, primary_meta = canonical_schedule(
        primary_specs, market, funding, masks, cfg, cutoff=cutoff
    )
    fixed_base = _specifications_from_trades(primary, primary_specs)
    controls: dict[str, list[Any]] = {}
    schedule_meta: dict[str, Any] = {"primary": primary_meta, "controls": {}}

    for name in FEATURE_CONTROL_NAMES:
        specs = accepted_schedule(
            aligned, predictions[name][fold_name], mode, flat, drawdown
        )
        controls[name], schedule_meta["controls"][name] = canonical_schedule(
            specs, market, funding, masks, cfg, cutoff=cutoff
        )

    fixed_specs = {
        "exact_side_flip": [
            {**item, "side": -int(item["side"])} for item in fixed_base
        ],
        "constant_long": [{**item, "side": 1} for item in fixed_base],
        "deterministic_random_side": [
            {
                **item,
                "side": deterministic_random_side(str(item["session_date"])),
            }
            for item in fixed_base
        ],
    }
    start, end = WINDOWS[fold_name]
    fixed_specs["one_us_session_delayed_entry"] = delayed_control(
        fixed_base, aligned, _market_dates(market), start, end
    )
    for name in FIXED_CONTROL_NAMES:
        controls[name], schedule_meta["controls"][name] = canonical_schedule(
            fixed_specs[name], market, funding, masks, cfg, cutoff=cutoff
        )
    if tuple(controls) != CONTROL_NAMES:
        raise RuntimeError("all nine controls must be produced in frozen order")
    return primary, controls, schedule_meta


def candidate_arrays(
    trades: Sequence[Any],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: Config,
    *,
    phase: str,
    cost_rate: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _install_candidate_sleeve()
    masks = split_mask_for_phase(market, phase=phase)
    split = PHASE_SPLITS[phase]
    schedules = {CANDIDATE_SLEEVE: {split: list(trades)}}
    events: list[dict[str, Any]] = []
    counts = accounting.append_candidate_paths(
        events,
        market,
        funding,
        schedules,
        accounting_config(
            cfg,
            cutoff=PHASE_CUTOFFS[phase],
            cost_rate=cost_rate,
        ),
        cost_rate=cost_rate,
    )
    arrays = portfolio.split_arrays(events, market, masks)
    observed_entries = arrays[split]["entry_positions"][CANDIDATE_SLEEVE]
    expected_entries = np.asarray(
        sorted(int(trade.entry_position) for trade in trades), dtype=np.int64
    )
    if not np.array_equal(observed_entries, expected_entries):
        raise RuntimeError("candidate array entries differ from canonical schedule")
    return arrays, {"counts": counts, "event_count": len(events)}


def slice_array_data(
    data: dict[str, Any],
    market: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    return accounting.slice_array_data(
        data, market, start=str(start), end=str(end)
    )


def merge_array_data(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    _install_candidate_sleeve()
    left_dates = pd.DatetimeIndex(baseline["dates"])
    right_dates = pd.DatetimeIndex(candidate["dates"])
    if not left_dates.equals(right_dates):
        raise RuntimeError("Gross9 and COGR array clocks differ")
    output = dict(baseline)
    for key in ("R", "A", "U", "L", "H"):
        left = np.asarray(baseline[key], dtype=np.float64)
        right = np.asarray(candidate[key], dtype=np.float64)
        if left.shape != right.shape:
            raise RuntimeError(f"Gross9/COGR {key} shape mismatch")
        output[key] = left + right
    output["counts"] = np.asarray(baseline["counts"], dtype=np.int64) + np.asarray(
        candidate["counts"], dtype=np.int64
    )
    output["wins"] = np.asarray(baseline["wins"], dtype=np.int64) + np.asarray(
        candidate["wins"], dtype=np.int64
    )
    output["dates"] = left_dates
    output["entry_positions"] = {
        sleeve: np.asarray(
            sorted(
                set(
                    map(
                        int,
                        np.asarray(
                            baseline["entry_positions"].get(
                                sleeve, np.empty(0, dtype=np.int64)
                            ),
                            dtype=np.int64,
                        ),
                    )
                )
                | set(
                    map(
                        int,
                        np.asarray(
                            candidate["entry_positions"].get(
                                sleeve, np.empty(0, dtype=np.int64)
                            ),
                            dtype=np.int64,
                        ),
                    )
                )
            ),
            dtype=np.int64,
        )
        for sleeve in portfolio.SLEEVES
    }
    return output


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / (365.25 * 86_400.0)


def strict_metric(
    data: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    return portfolio.strict_metric(
        dict(data), years_between(start, end), dict(weights)
    )


def same_gross_weights(
    baseline_weights: Mapping[str, float],
    candidate_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    weight = float(candidate_weight)
    combined = {str(name): float(value) for name, value in baseline_weights.items()}
    combined[CANDIDATE_SLEEVE] = weight
    comparator = {
        str(name): float(value) * (BASELINE_GROSS + weight) / BASELINE_GROSS
        for name, value in baseline_weights.items()
    }
    return combined, comparator


def weighted_bar_returns(
    data: Mapping[str, Any],
    weights: Mapping[str, float],
) -> np.ndarray:
    vector = np.asarray(
        [float(weights.get(name, 0.0)) for name in portfolio.SLEEVES],
        dtype=np.float64,
    )
    return vector @ np.asarray(data["R"], dtype=np.float64)


def portfolio_metrics(
    normal_data: Mapping[str, Any],
    stress_data: Mapping[str, Any],
    baseline_data: Mapping[str, Any],
    baseline_weights: Mapping[str, float],
    candidate_weight: float,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(
        baseline_weights, candidate_weight
    )
    return {
        "combined": strict_metric(
            normal_data, combined_weights, start=start, end=end
        ),
        "same_gross_comparator": strict_metric(
            normal_data, comparator_weights, start=start, end=end
        ),
        "unscaled_gross9": strict_metric(
            baseline_data, baseline_weights, start=start, end=end
        ),
        "stressed_combined": strict_metric(
            stress_data, combined_weights, start=start, end=end
        ),
        "combined_weights": combined_weights,
        "comparator_weights": comparator_weights,
    }


def trade_net_return(trade: Any, *, cost_rate: float = NORMAL_COST) -> float:
    factor = 1.0 - UNIT_LEVERAGE * float(cost_rate)
    terminal = (
        factor
        * float(trade.price_factor)
        * float(trade.funding_factor)
        * factor
    )
    if not math.isfinite(terminal) or terminal <= 0.0:
        raise RuntimeError("invalid COGR trade terminal factor")
    return terminal - 1.0


def concentration(trades: Sequence[Any]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "max_month_share": 0.0,
            "max_weekday_share": 0.0,
        }
    entries = pd.DatetimeIndex(
        [pd.Timestamp(trade.entry_date) for trade in trades]
    )
    sides = np.asarray([int(trade.side) for trade in trades], dtype=np.int8)
    months = pd.Series(entries.month).value_counts(normalize=True)
    weekdays = pd.Series(entries.weekday).value_counts(normalize=True)
    return {
        "trades": int(len(trades)),
        "long_share": float((sides > 0).mean()),
        "short_share": float((sides < 0).mean()),
        "max_month_share": float(months.max()),
        "max_weekday_share": float(weekdays.max()),
    }


def entry_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def entry_jaccards(
    base_arrays: Mapping[str, Mapping[str, Any]],
    candidate_trades: Sequence[Any],
    market: pd.DataFrame,
    baseline_weights: Mapping[str, float],
    *,
    window_name: str,
) -> dict[str, float]:
    split = (
        "train" if window_name == "selection_2023h2" else "test2024"
    )
    start, end = WINDOWS[window_name]
    dates = _market_dates(market)
    candidate_entries = {
        int(trade.entry_position)
        for trade in candidate_trades
        if start <= dates[int(trade.entry_position)] < end
    }
    output: dict[str, float] = {}
    for sleeve, weight in baseline_weights.items():
        if float(weight) <= 0.0:
            continue
        raw = np.asarray(
            base_arrays[split]["entry_positions"][sleeve], dtype=np.int64
        )
        entries = {
            int(position)
            for position in raw
            if start <= dates[int(position)] < end
        }
        output[sleeve] = entry_jaccard(candidate_entries, entries)
    if tuple(output) != tuple(baseline_weights):
        raise RuntimeError("entry Jaccard did not compare every Gross9 sleeve")
    return output


def weekly_effects(
    trades: Sequence[Any],
    *,
    cost_rate: float = NORMAL_COST,
) -> np.ndarray:
    buckets: dict[tuple[int, int], float] = {}
    for trade in trades:
        entry = pd.Timestamp(
            trade.entry_date
            if hasattr(trade, "entry_date")
            else trade["entry_time"]
        )
        value = (
            trade_net_return(trade, cost_rate=cost_rate)
            if hasattr(trade, "price_factor")
            else float(trade["return"])
        )
        year, week, _ = entry.isocalendar()
        key = (int(year), int(week))
        buckets[key] = buckets.get(key, 0.0) + math.log1p(value)
    return np.asarray([buckets[key] for key in sorted(buckets)], dtype=np.float64)


def paired_weekly_effects(
    treatment_returns: np.ndarray,
    comparator_returns: np.ndarray,
    dates: Sequence[Any],
) -> np.ndarray:
    treatment = np.asarray(treatment_returns, dtype=np.float64)
    comparator = np.asarray(comparator_returns, dtype=np.float64)
    if (
        treatment.shape != comparator.shape
        or len(treatment) != len(dates)
        or (treatment <= -1.0).any()
        or (comparator <= -1.0).any()
    ):
        raise RuntimeError("invalid paired barwise return arrays")
    timestamps = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).tz_convert(None)
    differences = np.log1p(treatment) - np.log1p(comparator)
    buckets: dict[tuple[int, int], float] = {}
    for timestamp, difference in zip(timestamps, differences, strict=True):
        year, week, _ = timestamp.isocalendar()
        key = (int(year), int(week))
        buckets[key] = buckets.get(key, 0.0) + float(difference)
    return np.asarray(
        [
            buckets[key]
            for key in sorted(buckets)
            if abs(buckets[key]) > 0.0
        ],
        dtype=np.float64,
    )


def sign_flip_pvalue(
    effects: np.ndarray,
    *,
    simulations: int = 10_000,
    seed: int = 20260728,
) -> float:
    values = np.asarray(effects, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    observed = float(values.sum())
    generator = np.random.default_rng(seed)
    signs = generator.choice(
        np.asarray([-1.0, 1.0]),
        size=(int(simulations), len(values)),
    )
    simulated = signs @ values
    return float(
        (1 + np.count_nonzero(simulated >= observed))
        / (int(simulations) + 1)
    )


def bootstrap_lower_mean(
    effects: np.ndarray,
    *,
    simulations: int = 10_000,
    seed: int = 20260729,
    q: float = 0.10,
) -> float | None:
    values = np.asarray(effects, dtype=np.float64)
    if len(values) == 0:
        return None
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        values, size=(int(simulations), len(values)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, q, method="linear"))


def portfolio_statistics(
    trades: Sequence[Any],
    normal_data: Mapping[str, Any],
    combined_weights: Mapping[str, float],
    comparator_weights: Mapping[str, float],
    statistical_requirement: Mapping[str, Any],
) -> dict[str, Any]:
    simulations = int(statistical_requirement["simulations"])
    seed = int(statistical_requirement["seed"])
    standalone = weekly_effects(trades)
    portfolio_effects = paired_weekly_effects(
        weighted_bar_returns(normal_data, combined_weights),
        weighted_bar_returns(normal_data, comparator_weights),
        normal_data["dates"],
    )
    return {
        "standalone_active_weeks": int(len(standalone)),
        "standalone_sign_flip_p": sign_flip_pvalue(
            standalone, simulations=simulations, seed=seed
        ),
        "portfolio_active_weeks": int(len(portfolio_effects)),
        "portfolio_sign_flip_p": sign_flip_pvalue(
            portfolio_effects, simulations=simulations, seed=seed
        ),
        "portfolio_bootstrap_90pct_lower_mean_log_excess": bootstrap_lower_mean(
            portfolio_effects,
            simulations=simulations,
            seed=20260729,
            q=0.10,
        ),
        "standalone_effect_hash": array_hash(standalone),
        "portfolio_effect_hash": array_hash(portfolio_effects),
    }


def apply_statistical_checks(
    checks: dict[str, bool],
    statistics: Mapping[str, Any],
    requirement: Mapping[str, Any],
) -> None:
    minimum = int(requirement["minimum_active_weeks_each_test"])
    maximum_p = float(requirement["maximum_p_value_each_test"])
    lower = statistics["portfolio_bootstrap_90pct_lower_mean_log_excess"]
    checks.update(
        {
            "standalone_active_weeks": int(
                statistics["standalone_active_weeks"]
            )
            >= minimum,
            "standalone_sign_flip_p": float(
                statistics["standalone_sign_flip_p"]
            )
            <= maximum_p,
            "portfolio_active_weeks": int(statistics["portfolio_active_weeks"])
            >= minimum,
            "portfolio_sign_flip_p": float(
                statistics["portfolio_sign_flip_p"]
            )
            <= maximum_p,
            "portfolio_bootstrap_lower_positive": (
                lower is not None and float(lower) > 0.0
            ),
        }
    )


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = {name: index for index, name in enumerate(COORDINATION_MODES)}
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["passes"]),
            -float(row["same_gross_cagr_mdd_improvement"]),
            -float(row["strict_mdd_reduction_vs_unscaled_gross9"]),
            -float(row["standalone"]["cagr_to_strict_mdd"]),
            -float(row["stressed_standalone"]["absolute_return_pct"]),
            float(row["candidate_weight"]),
            mode_order[str(row["coordination_mode"])],
        ),
    )


def row_for_cell(
    mode: str,
    weight: float,
    *,
    normal_data: Mapping[str, Any],
    stress_data: Mapping[str, Any],
    baseline_data: Mapping[str, Any],
    baseline_weights: Mapping[str, float],
    primary_trades: Sequence[Any],
    control_metrics: Mapping[str, dict[str, Any]],
    jaccards: Mapping[str, float],
    window_name: str,
    preregistration: Mapping[str, Any],
    half_standalone: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if tuple(control_metrics) != CONTROL_NAMES:
        raise RuntimeError("best-control comparison must contain all nine controls")
    if tuple(jaccards) != tuple(baseline_weights):
        raise RuntimeError("Jaccard comparison must contain every Gross9 sleeve")
    start, end = WINDOWS[window_name]
    portfolio_rows = portfolio_metrics(
        normal_data,
        stress_data,
        baseline_data,
        baseline_weights,
        weight,
        start=start,
        end=end,
    )
    standalone = strict_metric(
        normal_data,
        {CANDIDATE_SLEEVE: 1.0},
        start=start,
        end=end,
    )
    stressed_standalone = strict_metric(
        stress_data,
        {CANDIDATE_SLEEVE: 1.0},
        start=start,
        end=end,
    )
    combined = portfolio_rows["combined"]
    comparator = portfolio_rows["same_gross_comparator"]
    baseline = portfolio_rows["unscaled_gross9"]
    stressed_combined = portfolio_rows["stressed_combined"]
    concentration_row = concentration(primary_trades)
    best_control = max(
        float(control_metrics[name]["cagr_to_strict_mdd"])
        for name in CONTROL_NAMES
    )
    same_gross_improvement = float(
        combined["cagr_to_strict_mdd"]
        - comparator["cagr_to_strict_mdd"]
    )
    mdd_reduction = float(
        baseline["strict_mdd_pct"] - combined["strict_mdd_pct"]
    )
    max_jaccard = max(map(float, jaccards.values()), default=0.0)
    if window_name == "selection_2023h2":
        standalone_requirement = preregistration["selection_contract"][
            "selection_2023h2_standalone_requirements"
        ]
        portfolio_requirement = preregistration["selection_contract"][
            "selection_2023h2_portfolio_requirements"
        ]
        mechanism_requirement = preregistration["selection_contract"][
            "selection_2023h2_mechanism_requirements"
        ]
    else:
        standalone_requirement = preregistration["eval_2024_veto_contract"][
            "standalone_requirements"
        ]
        portfolio_requirement = preregistration["eval_2024_veto_contract"][
            "portfolio_requirements"
        ]
        mechanism_requirement = preregistration["eval_2024_veto_contract"][
            "mechanism_requirements"
        ]
    baseline_return = float(baseline["absolute_return_pct"])
    retention = (
        float(combined["absolute_return_pct"]) / baseline_return
        if abs(baseline_return) > 1e-12
        else float("-inf")
    )
    checks: dict[str, bool] = {
        "standalone_absolute_return_positive": float(
            standalone["absolute_return_pct"]
        )
        > 0.0,
        "standalone_cagr_mdd": float(standalone["cagr_to_strict_mdd"])
        >= float(standalone_requirement["minimum_cagr_to_strict_mdd"]),
        "standalone_mdd": float(standalone["strict_mdd_pct"])
        <= float(standalone_requirement["maximum_strict_mdd_pct"]),
        "minimum_trades": concentration_row["trades"]
        >= int(standalone_requirement["minimum_trades"]),
        "long_share": concentration_row["long_share"]
        >= float(standalone_requirement["minimum_long_share"]),
        "short_share": concentration_row["short_share"]
        >= float(standalone_requirement["minimum_short_share"]),
        "month_concentration": concentration_row["max_month_share"]
        <= float(standalone_requirement["maximum_month_share"]),
        "weekday_concentration": concentration_row["max_weekday_share"]
        <= float(standalone_requirement["maximum_weekday_share"]),
        "stress_standalone_positive": float(
            stressed_standalone["absolute_return_pct"]
        )
        > 0.0,
        "portfolio_mdd": float(combined["strict_mdd_pct"])
        <= float(portfolio_requirement["maximum_strict_mdd_pct"]),
        "return_retention": retention
        >= float(
            portfolio_requirement[
                "absolute_return_retention_floor_vs_unscaled_gross9"
            ]
        ),
        "same_gross_improvement": same_gross_improvement
        >= float(
            portfolio_requirement[
                "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9"
            ]
        ),
        "mdd_reduction": mdd_reduction > 0.0,
        "entry_jaccard": max_jaccard
        <= float(
            portfolio_requirement[
                "maximum_exact_entry_jaccard_vs_any_gross9_sleeve"
            ]
        ),
        "stress_portfolio_positive": float(
            stressed_combined["absolute_return_pct"]
        )
        > 0.0,
        "all_control_margin": float(standalone["cagr_to_strict_mdd"])
        - best_control
        >= float(
            mechanism_requirement[
                "primary_cagr_mdd_margin_over_best_of_all_nine_controls"
            ]
        ),
    }
    statistics: dict[str, Any] | None = None
    if window_name == "eval_2024":
        if tuple(half_standalone or ()) != tuple(EVAL_HALF_WINDOWS):
            raise RuntimeError("eval requires exact first/second-half standalone metrics")
        checks.update(
            {
                "first_calendar_half_positive": float(
                    half_standalone["first_calendar_half"][
                        "absolute_return_pct"
                    ]
                )
                > 0.0,
                "second_calendar_half_positive": float(
                    half_standalone["second_calendar_half"][
                        "absolute_return_pct"
                    ]
                )
                > 0.0,
            }
        )
        requirement = preregistration["eval_2024_veto_contract"][
            "statistical_requirement"
        ]
        statistics = portfolio_statistics(
            primary_trades,
            normal_data,
            portfolio_rows["combined_weights"],
            portfolio_rows["comparator_weights"],
            requirement,
        )
        apply_statistical_checks(checks, statistics, requirement)
    return {
        "coordination_mode": mode,
        "candidate_weight": float(weight),
        "gross": BASELINE_GROSS + float(weight),
        "passes": bool(all(checks.values())),
        "checks": checks,
        "standalone": standalone,
        "stressed_standalone": stressed_standalone,
        "portfolio": combined,
        "same_gross_comparator": comparator,
        "unscaled_gross9": baseline,
        "stressed_portfolio": stressed_combined,
        "concentration": concentration_row,
        "controls": dict(control_metrics),
        "best_control_cagr_mdd": best_control,
        "entry_jaccards": dict(jaccards),
        "max_entry_jaccard": max_jaccard,
        "same_gross_cagr_mdd_improvement": same_gross_improvement,
        "absolute_return_retention_vs_unscaled_gross9": retention,
        "strict_mdd_reduction_vs_unscaled_gross9": mdd_reduction,
        "half_standalone": dict(half_standalone or {}),
        "statistics": statistics,
    }


def _atomic_canonical_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(destination)


def render(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# COGR-12 {payload['phase']} report",
        "",
        "Metric format: absolute return / full-calendar CAGR / strict MDD / "
        "CAGR-MDD / trades.",
        "",
        f"- as of: `{payload['as_of']}`",
        f"- decision: **{payload['decision']}**",
        f"- tested cells: `{payload.get('tested_cells', 1)}`",
        "",
    ]
    if payload.get("frozen_top1"):
        top = payload["frozen_top1"]
        lines.extend(
            [
                "## Frozen top1",
                "",
                f"- mode: `{top['coordination_mode']}`",
                f"- candidate weight: `{top['candidate_weight']}`",
                f"- pass: `{top['passes']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Integrity",
            "",
            f"- preregistration SHA-256: `{payload['preregistration_sha256']}`",
            f"- result hash: `{payload['result_hash']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    payload: dict[str, Any],
    json_path: str | Path,
    docs_path: str | Path,
) -> dict[str, Any]:
    ready = finalize_payload(payload)
    _atomic_canonical_json(json_path, ready)
    destination = Path(docs_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(render(ready), encoding="utf-8")
    temporary.replace(destination)
    return ready


def finalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = json_ready(payload)
    ready.pop("result_hash", None)
    ready["result_hash"] = json_hash(
        ready
    )
    return ready


def verify_result_hash(payload: Mapping[str, Any]) -> None:
    observed = payload.get("result_hash")
    expected = json_hash(
        {key: value for key, value in payload.items() if key != "result_hash"}
    )
    if observed != expected:
        raise RuntimeError("selection artifact result_hash recomputation failed")


def verify_selection_artifact(
    selection: Mapping[str, Any],
    *,
    preregistration_sha256: str,
    expected_input_identity: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    verify_result_hash(selection)
    if selection.get("as_of") != AS_OF:
        raise RuntimeError("selection artifact as_of drifted")
    if selection.get("phase") != "selection":
        raise RuntimeError("eval requires a selection-phase artifact")
    if selection.get("preregistration_sha256") != preregistration_sha256:
        raise RuntimeError("selection artifact preregistration SHA drifted")
    if selection.get("config") != json_ready(expected_config):
        raise RuntimeError("selection artifact config drifted")
    if selection.get("input_identity") != json_ready(expected_input_identity):
        raise RuntimeError("selection artifact input identity drifted")
    if tuple(selection.get("input_identity", {})) != INPUT_KEYS:
        raise RuntimeError("selection artifact input identity is incomplete")
    if "gross9_source_meta" in selection or "gross9_observable_state" in selection:
        raise RuntimeError("selection artifact exposed post-selection Gross9 metadata")
    _require_equal(
        "selection.disclosure",
        selection.get("gross9_selection_disclosure"),
        {
            "candidate_metric_window": "2023H2",
            "future_candidate_data_opened": False,
            "future_gross9_metadata_exposed": False,
        },
    )
    model_meta = selection.get("model_meta")
    if not isinstance(model_meta, Mapping) or model_meta.get("phase") != "selection":
        raise RuntimeError("selection artifact model metadata drifted")
    variants = model_meta.get("variants")
    expected_variants = {"primary", *FEATURE_CONTROL_NAMES}
    if not isinstance(variants, Mapping) or set(variants) != expected_variants:
        raise RuntimeError("selection artifact model variants drifted")
    for name, metadata in variants.items():
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("variant") != name
            or tuple(metadata.get("allowed_folds", ()))
            != PHASE_FOLDS["selection"]
        ):
            raise RuntimeError(f"selection artifact model fold drifted for {name}")
    schedule_meta = selection.get("schedule_meta")
    if not isinstance(schedule_meta, Mapping) or set(schedule_meta) != set(
        COORDINATION_MODES
    ):
        raise RuntimeError("selection artifact schedule metadata drifted")
    if selection.get("decision") != "freeze_top1_for_eval":
        raise RuntimeError("selection artifact did not freeze a passing top1")
    if selection.get("eval_opened") is not False:
        raise RuntimeError("selection artifact improperly opened eval")
    if int(selection.get("tested_cells", -1)) != len(COORDINATION_MODES) * len(
        WEIGHTS
    ):
        raise RuntimeError("selection artifact did not test the frozen 12 cells")
    if selection.get("rank2_opened", False) or selection.get("frozen_rank2"):
        raise RuntimeError("rank2 is forbidden")
    top = selection.get("frozen_top1")
    if not isinstance(top, Mapping) or top.get("passes") is not True:
        raise RuntimeError("eval fails closed without a passing frozen top1")
    mode = str(top.get("coordination_mode"))
    weight = float(top.get("candidate_weight"))
    if mode not in COORDINATION_MODES or weight not in WEIGHTS:
        raise RuntimeError("frozen top1 is outside the preregistered universe")
    rows = selection.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != len(COORDINATION_MODES) * len(WEIGHTS)
    ):
        raise RuntimeError("selection artifact must contain exactly 12 ranked rows")
    required_row_keys = {
        "coordination_mode",
        "candidate_weight",
        "passes",
        "checks",
        "standalone",
        "stressed_standalone",
        "portfolio",
        "stressed_portfolio",
        "same_gross_comparator",
        "unscaled_gross9",
        "controls",
        "entry_jaccards",
    }
    expected_cells = {
        (mode, float(weight))
        for mode in COORDINATION_MODES
        for weight in WEIGHTS
    }
    observed_cells: set[tuple[str, float]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not required_row_keys.issubset(row):
            raise RuntimeError("selection artifact contains an incomplete ranked row")
        cell = (
            str(row.get("coordination_mode")),
            float(row.get("candidate_weight")),
        )
        observed_cells.add(cell)
        checks = row.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or any(not isinstance(value, bool) for value in checks.values())
            or bool(row.get("passes")) != all(checks.values())
        ):
            raise RuntimeError("selection artifact row checks drifted")
    if observed_cells != expected_cells or len(observed_cells) != len(rows):
        raise RuntimeError("selection artifact cell universe drifted")
    passing = [row for row in rows if row.get("passes") is True]
    if not passing or top != passing[0]:
        raise RuntimeError("frozen top1 does not match the ranked passing top1")
    return dict(top)


def verify_reproduced_selection(
    selection: Mapping[str, Any],
    reproduced: Mapping[str, Any],
) -> None:
    verify_result_hash(selection)
    verify_result_hash(reproduced)
    if canonical_json(selection) != canonical_json(reproduced):
        raise RuntimeError(
            "selection artifact does not match deterministic selection replay"
        )


def _validate_shared_market_prefix(
    candidate_market: pd.DataFrame,
    gross9_market: pd.DataFrame,
) -> None:
    if len(candidate_market) > len(gross9_market):
        raise RuntimeError("candidate market is longer than Gross9 market")
    gross_prefix = gross9_market.iloc[: len(candidate_market)]
    if not _market_dates(candidate_market).equals(_market_dates(gross_prefix)):
        raise RuntimeError("COGR and Gross9 market clocks differ")
    for column in ("open", "high", "low", "close"):
        if not np.array_equal(
            pd.to_numeric(candidate_market[column], errors="raise").to_numpy(float),
            pd.to_numeric(gross_prefix[column], errors="raise").to_numpy(float),
        ):
            raise RuntimeError(f"COGR and Gross9 {column} prefixes differ")


def build_context(
    cfg: Config,
    preregistration: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in PHASE_CUTOFFS:
        raise ValueError(phase)
    cutoff = PHASE_CUTOFFS[phase]
    features = load_safe_features(cfg.safe_features, cutoff=cutoff)
    candidate_market = read_market_prefix(cfg.market_csv, cutoff=cutoff)
    funding = read_funding_prefix(cfg.funding_csv, cutoff=cutoff)
    aligned = align_feature_sessions_to_market(features, candidate_market)
    if (
        pd.to_datetime(aligned["session_date"]).max()
        >= cutoff
    ):
        raise RuntimeError("aligned feature frame crossed the phase cutoff")

    base_cfg = gross9_runtime.Config(
        market_csv=cfg.market_csv,
        market_with_oi_csv=cfg.market_with_oi_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        gross9_pre2025_anchor=cfg.gross9_pre2025_anchor,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cutoff="2025-01-01",
        cost_rate=NORMAL_COST,
        stress_cost_rate=STRESS_COST,
        leverage=UNIT_LEVERAGE,
    )
    gross9_market, masks, base_events, gross9_meta = (
        gross9_runtime.build_gross9_context(base_cfg)
    )
    _validate_shared_market_prefix(candidate_market, gross9_market)
    _install_candidate_sleeve()
    base_arrays = portfolio.split_arrays(base_events, gross9_market, masks)
    anchor = json.loads(
        Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8")
    )
    baseline_weights = {
        str(name): float(weight) for name, weight in anchor["weights"].items()
    }
    if baseline_weights != BASELINE_WEIGHTS:
        raise RuntimeError("authoritative Gross9 weights drifted")
    preregistered_weights = {
        str(name): float(weight)
        for name, weight in preregistration["same_gross_formula"][
            "baseline_weights"
        ].items()
    }
    if baseline_weights != preregistered_weights:
        raise RuntimeError("anchor and preregistered Gross9 weights differ")
    gross9_runtime.validate_baseline(base_arrays, anchor)
    flat, drawdown, state_meta = gross9_state_at_signal(
        gross9_market, masks, base_events, baseline_weights
    )
    return {
        "phase": phase,
        "cutoff": cutoff,
        "aligned": aligned,
        "candidate_market": candidate_market,
        "funding": funding,
        "gross9_market": gross9_market,
        "masks": masks,
        "base_events": base_events,
        "base_arrays": base_arrays,
        "baseline_weights": baseline_weights,
        "gross9_meta": gross9_meta,
        "state_meta": state_meta,
        "flat": flat,
        "drawdown": drawdown,
        "authoritative_anchor_validated": True,
    }


def _window_arrays(
    context: Mapping[str, Any],
    candidate_split_arrays: Mapping[str, Mapping[str, Any]],
    stress_split_arrays: Mapping[str, Mapping[str, Any]],
    *,
    window_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split = (
        "train" if window_name == "selection_2023h2" else "test2024"
    )
    start, end = WINDOWS[window_name]
    baseline = slice_array_data(
        context["base_arrays"][split],
        context["gross9_market"],
        start=start,
        end=end,
    )
    candidate = slice_array_data(
        candidate_split_arrays[split],
        context["candidate_market"],
        start=start,
        end=end,
    )
    stressed_candidate = slice_array_data(
        stress_split_arrays[split],
        context["candidate_market"],
        start=start,
        end=end,
    )
    return (
        merge_array_data(baseline, candidate),
        merge_array_data(baseline, stressed_candidate),
        baseline,
    )


def _control_metrics(
    controls: Mapping[str, Sequence[Any]],
    context: Mapping[str, Any],
    cfg: Config,
    *,
    phase: str,
    window_name: str,
) -> dict[str, dict[str, Any]]:
    start, end = WINDOWS[window_name]
    split = PHASE_SPLITS[phase]
    output: dict[str, dict[str, Any]] = {}
    for name in CONTROL_NAMES:
        arrays, _ = candidate_arrays(
            controls[name],
            context["candidate_market"],
            context["funding"],
            cfg,
            phase=phase,
            cost_rate=NORMAL_COST,
        )
        sliced = slice_array_data(
            arrays[split],
            context["candidate_market"],
            start=start,
            end=end,
        )
        output[name] = strict_metric(
            sliced,
            {CANDIDATE_SLEEVE: 1.0},
            start=start,
            end=end,
        )
    if tuple(output) != CONTROL_NAMES:
        raise RuntimeError("control metrics omitted a frozen control")
    return output


def _half_standalone_metrics(
    candidate_split_arrays: Mapping[str, Mapping[str, Any]],
    candidate_market: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, (start, end) in EVAL_HALF_WINDOWS.items():
        data = slice_array_data(
            candidate_split_arrays["test2024"],
            candidate_market,
            start=start,
            end=end,
        )
        output[name] = strict_metric(
            data,
            {CANDIDATE_SLEEVE: 1.0},
            start=start,
            end=end,
        )
    return output


def _mode_rows(
    context: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    preregistration: Mapping[str, Any],
    cfg: Config,
    *,
    phase: str,
    modes: Sequence[str],
    weights: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_name = PHASE_FOLDS[phase][-1]
    rows: list[dict[str, Any]] = []
    mode_metadata: dict[str, Any] = {}
    for mode in modes:
        primary, controls, schedule_meta = schedules_for_mode(
            mode,
            predictions,
            context["aligned"],
            context["candidate_market"],
            context["funding"],
            context["flat"],
            context["drawdown"],
            cfg,
            phase=phase,
        )
        normal_arrays, normal_meta = candidate_arrays(
            primary,
            context["candidate_market"],
            context["funding"],
            cfg,
            phase=phase,
            cost_rate=NORMAL_COST,
        )
        stress_arrays, stress_meta = candidate_arrays(
            primary,
            context["candidate_market"],
            context["funding"],
            cfg,
            phase=phase,
            cost_rate=STRESS_COST,
        )
        split = PHASE_SPLITS[phase]
        if not np.array_equal(
            normal_arrays[split]["entry_positions"][CANDIDATE_SLEEVE],
            stress_arrays[split]["entry_positions"][CANDIDATE_SLEEVE],
        ):
            raise RuntimeError("10bp stress replay changed candidate entries")
        normal_data, stress_data, baseline_data = _window_arrays(
            context,
            normal_arrays,
            stress_arrays,
            window_name=window_name,
        )
        controls_metrics = _control_metrics(
            controls,
            context,
            cfg,
            phase=phase,
            window_name=window_name,
        )
        jaccards = entry_jaccards(
            context["base_arrays"],
            primary,
            context["gross9_market"],
            context["baseline_weights"],
            window_name=window_name,
        )
        halves = (
            _half_standalone_metrics(
                normal_arrays, context["candidate_market"]
            )
            if phase == "eval"
            else None
        )
        for weight in weights:
            rows.append(
                row_for_cell(
                    mode,
                    float(weight),
                    normal_data=normal_data,
                    stress_data=stress_data,
                    baseline_data=baseline_data,
                    baseline_weights=context["baseline_weights"],
                    primary_trades=primary,
                    control_metrics=controls_metrics,
                    jaccards=jaccards,
                    window_name=window_name,
                    preregistration=preregistration,
                    half_standalone=halves,
                )
            )
        mode_metadata[mode] = {
            "schedule": schedule_meta,
            "normal_arrays": normal_meta,
            "stress_arrays": stress_meta,
            "entry_jaccards": jaccards,
        }
    return rows, mode_metadata


def build_selection_payload(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    preregistration_sha = _sha256(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    context = build_context(cfg, preregistration, phase="selection")
    predictions, model_meta = prediction_variants(
        context["aligned"],
        context["candidate_market"],
        context["funding"],
        preregistration,
        phase="selection",
        cfg=cfg,
    )
    rows, schedule_meta = _mode_rows(
        context,
        predictions,
        preregistration,
        cfg,
        phase="selection",
        modes=COORDINATION_MODES,
        weights=WEIGHTS,
    )
    ranked = rank_rows(rows)
    passing = [row for row in ranked if row["passes"]]
    payload = {
        "as_of": AS_OF,
        "phase": "selection",
        "config": asdict(cfg),
        "preregistration_sha256": preregistration_sha,
        "input_identity": input_identity,
        "authoritative_gross9_anchor_validated": context[
            "authoritative_anchor_validated"
        ],
        "gross9_selection_disclosure": {
            "candidate_metric_window": "2023H2",
            "future_candidate_data_opened": False,
            "future_gross9_metadata_exposed": False,
        },
        "model_meta": model_meta,
        "schedule_meta": schedule_meta,
        "tested_cells": len(ranked),
        "rows": ranked,
        "frozen_top1": passing[0] if passing else None,
        "decision": (
            "freeze_top1_for_eval"
            if passing
            else "reject_no_passing_2023h2_cell"
        ),
        "eval_opened": False,
        "rank2_opened": False,
    }
    if "gross9_source_meta" in payload or "gross9_observable_state" in payload:
        raise RuntimeError("selection payload exposed forbidden Gross9 metadata")
    return payload


def run_selection(cfg: Config) -> dict[str, Any]:
    return _write_outputs(
        build_selection_payload(cfg),
        cfg.selection_output,
        cfg.docs_selection_output,
    )


def run_eval(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    preregistration_sha = _sha256(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    selection = json.loads(
        Path(cfg.selection_output).read_text(encoding="utf-8")
    )
    top = verify_selection_artifact(
        selection,
        preregistration_sha256=preregistration_sha,
        expected_input_identity=input_identity,
        expected_config=asdict(cfg),
    )
    reproduced_selection = finalize_payload(build_selection_payload(cfg))
    verify_reproduced_selection(selection, reproduced_selection)
    mode = str(top["coordination_mode"])
    weight = float(top["candidate_weight"])

    context = build_context(cfg, preregistration, phase="eval")
    predictions, model_meta = prediction_variants(
        context["aligned"],
        context["candidate_market"],
        context["funding"],
        preregistration,
        phase="eval",
        cfg=cfg,
    )
    rows, schedule_meta = _mode_rows(
        context,
        predictions,
        preregistration,
        cfg,
        phase="eval",
        modes=(mode,),
        weights=(weight,),
    )
    if len(rows) != 1:
        raise RuntimeError("eval must produce exactly one frozen top1 row")
    row = rows[0]
    payload = {
        "as_of": AS_OF,
        "phase": "eval",
        "config": asdict(cfg),
        "selection_freeze_hash": selection["result_hash"],
        "preregistration_sha256": preregistration_sha,
        "input_identity": input_identity,
        "authoritative_gross9_anchor_validated": context[
            "authoritative_anchor_validated"
        ],
        "gross9_source_meta": context["gross9_meta"],
        "gross9_observable_state": context["state_meta"],
        "model_meta": model_meta,
        "schedule_meta": schedule_meta,
        "tested_cells": 1,
        "frozen_top1": row,
        "decision": (
            "pass_2024_top1"
            if row["passes"]
            else "terminal_veto_2024_top1"
        ),
        "reranked": False,
        "rank2_opened": False,
    }
    return _write_outputs(payload, cfg.eval_output, cfg.docs_eval_output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="COGR-12 evaluator")
    parser.add_argument("phase", choices=("selection", "eval"))
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--selection-output", default=Config.selection_output)
    parser.add_argument("--eval-output", default=Config.eval_output)
    parser.add_argument(
        "--docs-selection-output", default=Config.docs_selection_output
    )
    parser.add_argument("--docs-eval-output", default=Config.docs_eval_output)
    arguments = parser.parse_args(argv)
    cfg = Config(
        preregistration=arguments.preregistration,
        selection_output=arguments.selection_output,
        eval_output=arguments.eval_output,
        docs_selection_output=arguments.docs_selection_output,
        docs_eval_output=arguments.docs_eval_output,
    )
    payload = (
        run_selection(cfg) if arguments.phase == "selection" else run_eval(cfg)
    )
    print(canonical_json(payload))


if __name__ == "__main__":
    main()
