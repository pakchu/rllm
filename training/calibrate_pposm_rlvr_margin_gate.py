"""Train-only margin calibration and separate OOS application for PPOSM.

``calibrate`` reads only pre-2024 margins and the frozen pre-2024 schedule.
``apply`` first validates the frozen threshold artifact, then (and only then)
opens OOS margins and applies the threshold as a veto-only schedule subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from training import build_pposm_sft_rlvr_data as builder
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.search_inventory_purge_reclaim_alpha import Trade, equity_stats


DEFAULT_TRAIN_MARGINS = Path("results/pposm_rlvr_train_pre2024_margins.jsonl")
DEFAULT_OOS_MARGINS = Path("results/pposm_rlvr_oos_2024_2026_margins.jsonl")
DEFAULT_THRESHOLD_ARTIFACT = Path("results/pposm_rlvr_margin_threshold.json")
DEFAULT_APPLY_OUTPUT = Path("results/pposm_rlvr_margin_gate_oos.json")

QUANTILES = tuple(index / 10.0 for index in range(10))
PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2020H2", "2020-07-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
)
OOS_WINDOWS = tuple(
    (window, start, end)
    for split, window, start, end in builder.SPLIT_WINDOWS
    if split == "oos"
)
COSTS = {"base_6bp": 0.0006, "stress_10bp": 0.0010}


@dataclass(frozen=True)
class CalibrationConfig:
    manifest: Path = builder.DEFAULT_MANIFEST
    train_margins: Path = DEFAULT_TRAIN_MARGINS
    threshold_artifact: Path = DEFAULT_THRESHOLD_ARTIFACT


@dataclass(frozen=True)
class ApplyConfig:
    manifest: Path = builder.DEFAULT_MANIFEST
    threshold_artifact: Path = DEFAULT_THRESHOLD_ARTIFACT
    oos_margins: Path = DEFAULT_OOS_MARGINS
    output: Path = DEFAULT_APPLY_OUTPUT
    signflip_permutations: int = 100_000
    signflip_seed: int = 20_260_819


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no margin rows loaded from {path}")
    return rows


def _identity(row: dict[str, Any], index: int) -> str:
    value = row.get("identity")
    if not isinstance(value, str):
        metadata = row.get("metadata")
        value = metadata.get("identity") if isinstance(metadata, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"margin row {index} has no identity")
    return value


def identity_align_margins(
    rows: Sequence[dict[str, Any]], expected_identities: Sequence[str]
) -> list[float]:
    """Align by exact identity and reject missing, extra, or duplicate rows."""

    expected = list(expected_identities)
    if len(expected) != len(set(expected)):
        raise ValueError("expected schedule identities are not unique")
    observed: dict[str, float] = {}
    for index, row in enumerate(rows):
        identity = _identity(row, index)
        if identity in observed:
            raise ValueError(f"duplicate margin identity: {identity}")
        try:
            margin = float(row["margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"margin row {index} has no numeric margin") from exc
        if not math.isfinite(margin):
            raise ValueError(f"margin row {index} is non-finite")
        observed[identity] = margin
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"margin identities do not equal frozen schedule: missing={missing}, extra={extra}"
        )
    return [observed[identity] for identity in expected]


def threshold_candidates(margins: Sequence[float]) -> list[float]:
    values = np.asarray(list(margins), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite training margins are required")
    candidates = [float("-inf")]
    for quantile in QUANTILES:
        value = float(np.quantile(values, quantile))
        if value not in candidates:
            candidates.append(value)
    return candidates


def _period_trades(trades: Sequence[Trade], start: str, end: str) -> list[Trade]:
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    return [
        trade
        for trade in trades
        if lower <= pd.Timestamp(trade.entry_date) < upper
    ]


def threshold_metrics(
    trades: Sequence[Trade], *, strategy_cfg: Any
) -> dict[str, Any]:
    periods = {
        name: equity_stats(
            _period_trades(trades, start, end),
            start=start,
            end=end,
            cfg=strategy_cfg,
            cost_rate=COSTS["base_6bp"],
        )
        for name, start, end in PERIODS
    }
    combined = equity_stats(
        trades,
        start="2020-07-01",
        end="2024-01-01",
        cfg=strategy_cfg,
        cost_rate=COSTS["base_6bp"],
    )
    eligible = (
        combined["trades"] >= 60
        and all(
            period["trades"] >= 5 and period["absolute_return_pct"] > 0.0
            for period in periods.values()
        )
    )
    return {"combined_pre2024": combined, "periods": periods, "eligible": eligible}


def select_threshold(
    margins: Sequence[float],
    trades: Sequence[Trade],
    *,
    strategy_cfg: Any,
    metric_function: Callable[[Sequence[Trade]], dict[str, Any]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if len(margins) != len(trades):
        raise ValueError("training margins and frozen trades must have equal lengths")
    evaluator = metric_function or (
        lambda selected: threshold_metrics(selected, strategy_cfg=strategy_cfg)
    )
    reports: list[dict[str, Any]] = []
    for threshold in threshold_candidates(margins):
        selected = [
            trade
            for trade, margin in zip(trades, margins, strict=True)
            if margin >= threshold
        ]
        metrics = evaluator(selected)
        reports.append(
            {
                "threshold": _encode_threshold(threshold),
                "selected_trade_count": len(selected),
                **metrics,
            }
        )
    eligible = [report for report in reports if bool(report.get("eligible"))]
    if not eligible:
        raise RuntimeError("no train-only threshold satisfies the trade/period gates")

    def rank(report: dict[str, Any]) -> tuple[float, float, float]:
        combined = report["combined_pre2024"]
        threshold = _decode_threshold(report["threshold"])
        return (
            float(combined["cagr_to_strict_mdd"]),
            float(combined["absolute_return_pct"]),
            -threshold,
        )

    chosen = max(eligible, key=rank)
    return _decode_threshold(chosen["threshold"]), reports


def _encode_threshold(value: float) -> float | str:
    return "-inf" if value == float("-inf") else float(value)


def _decode_threshold(value: Any) -> float:
    if value == "-inf":
        return float("-inf")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("artifact threshold must be finite or '-inf'")
    return threshold


def _artifact_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return sha256_bytes(canonical_json(core).encode())


def _write_frozen_artifact(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def load_threshold_artifact(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("threshold artifact is not an object")
    if artifact.get("protocol") != "pposm_rlvr_train_only_margin_threshold_v1":
        raise ValueError("unexpected threshold artifact protocol")
    if artifact.get("artifact_sha256") != _artifact_hash(artifact):
        raise ValueError("threshold artifact hash mismatch")
    _decode_threshold(artifact.get("threshold"))
    return artifact


def calibrate(cfg: CalibrationConfig) -> dict[str, Any]:
    manifest, strategy_cfg = builder.load_frozen_manifest(cfg.manifest)
    _, _, schedules = builder.replay_frozen_schedules(manifest, strategy_cfg)
    train_trades = tuple(schedules["pre_2024"])
    expected = [builder.trade_identity("pre_2024", trade) for trade in train_trades]
    margin_bytes = cfg.train_margins.read_bytes()
    margin_rows = _read_jsonl(cfg.train_margins)
    for index, row in enumerate(margin_rows):
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("window") not in (None, "pre_2024"):
            raise ValueError(f"non-training margin row {index} supplied to calibrate")
    margins = identity_align_margins(margin_rows, expected)
    threshold, candidate_reports = select_threshold(
        margins, train_trades, strategy_cfg=strategy_cfg
    )
    artifact: dict[str, Any] = {
        "protocol": "pposm_rlvr_train_only_margin_threshold_v1",
        "manifest_freeze_hash": manifest["freeze_hash"],
        "train_margin_sha256": sha256_bytes(margin_bytes),
        "train_identity_sha256": sha256_bytes("\n".join(expected).encode()),
        "training_window": {"start": "2020-07-01", "end_exclusive": "2024-01-01"},
        "quantiles": list(QUANTILES),
        "selection_gates": {
            "combined_min_trades": 60,
            "each_period_min_trades": 5,
            "each_period_absolute_return_positive": True,
        },
        "ranking": [
            "combined_pre2024_cagr_to_strict_mdd_desc",
            "combined_pre2024_absolute_return_desc",
            "threshold_asc",
        ],
        "threshold": _encode_threshold(threshold),
        "candidates": candidate_reports,
        "invariants": {
            "threshold_inputs": "pre2024_only",
            "oos_opened": False,
            "oos_can_rank_or_repair": False,
        },
    }
    artifact["artifact_sha256"] = _artifact_hash(artifact)
    _write_frozen_artifact(cfg.threshold_artifact, artifact)
    return artifact


def _net_returns(trades: Sequence[Trade], *, leverage: float, cost: float) -> list[float]:
    one_side = 1.0 - float(leverage) * float(cost)
    return [
        float(one_side * trade.price_factor * trade.funding_factor * one_side - 1.0)
        for trade in trades
    ]


def _strict_report(
    trades: Sequence[Trade],
    *,
    start: str,
    end: str,
    strategy_cfg: Any,
    cfg: ApplyConfig,
) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for name, cost in COSTS.items():
        stats = equity_stats(
            trades, start=start, end=end, cfg=strategy_cfg, cost_rate=cost
        )
        stats["one_sided_utc_week_sign_flip"] = weekly_cluster_sign_flip(
            _net_returns(trades, leverage=strategy_cfg.leverage, cost=cost),
            [trade.entry_date for trade in trades],
            permutations=cfg.signflip_permutations,
            seed=cfg.signflip_seed,
        )
        costs[name] = stats
    return {"selected_trade_count": len(trades), "costs": costs}


def apply(cfg: ApplyConfig) -> dict[str, Any]:
    if cfg.signflip_permutations < 1:
        raise ValueError("signflip_permutations must be positive")

    # This must precede opening cfg.oos_margins.  A missing or altered train
    # artifact therefore prevents any OOS data access.
    artifact = load_threshold_artifact(cfg.threshold_artifact)
    threshold = _decode_threshold(artifact["threshold"])
    manifest, strategy_cfg = builder.load_frozen_manifest(cfg.manifest)
    if artifact["manifest_freeze_hash"] != manifest["freeze_hash"]:
        raise ValueError("threshold artifact belongs to a different frozen manifest")
    _, _, schedules = builder.replay_frozen_schedules(manifest, strategy_cfg)
    baseline = tuple(
        trade for window, _, _ in OOS_WINDOWS for trade in schedules[window]
    )
    expected = [
        builder.trade_identity(window, trade)
        for window, _, _ in OOS_WINDOWS
        for trade in schedules[window]
    ]
    margins = identity_align_margins(_read_jsonl(cfg.oos_margins), expected)
    selected = tuple(
        trade
        for trade, margin in zip(baseline, margins, strict=True)
        if margin >= threshold
    )
    baseline_objects = {id(trade) for trade in baseline}
    if any(id(trade) not in baseline_objects for trade in selected):
        raise RuntimeError("margin gate introduced a replacement OOS trade")

    reports: dict[str, Any] = {}
    offset = 0
    for window, start, end in OOS_WINDOWS:
        window_baseline = tuple(schedules[window])
        window_margins = margins[offset : offset + len(window_baseline)]
        window_selected = tuple(
            trade
            for trade, margin in zip(window_baseline, window_margins, strict=True)
            if margin >= threshold
        )
        reports[window] = _strict_report(
            window_selected,
            start=start,
            end=end,
            strategy_cfg=strategy_cfg,
            cfg=cfg,
        )
        reports[window]["baseline_trade_count"] = len(window_baseline)
        reports[window]["vetoed_trade_count"] = len(window_baseline) - len(window_selected)
        offset += len(window_baseline)
    reports["combined_2024_2026_06_02"] = _strict_report(
        selected,
        start="2024-01-01",
        end="2026-06-02",
        strategy_cfg=strategy_cfg,
        cfg=cfg,
    )
    reports["combined_2024_2026_06_02"]["baseline_trade_count"] = len(baseline)
    reports["combined_2024_2026_06_02"]["vetoed_trade_count"] = len(baseline) - len(selected)
    output = {
        "protocol": "pposm_rlvr_frozen_margin_veto_oos_v1",
        "threshold_artifact_sha256": artifact["artifact_sha256"],
        "threshold": artifact["threshold"],
        "manifest_freeze_hash": manifest["freeze_hash"],
        "windows": reports,
        "invariants": {
            "threshold_artifact_validated_before_oos_read": True,
            "identity_alignment": "exact_set_then_frozen_schedule_order",
            "action": "veto_only",
            "replacement_allowed": False,
            "oos_influenced_threshold": False,
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> tuple[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("calibrate")
    train.add_argument("--manifest", type=Path, default=builder.DEFAULT_MANIFEST)
    train.add_argument("--train-margins", type=Path, default=DEFAULT_TRAIN_MARGINS)
    train.add_argument(
        "--threshold-artifact", type=Path, default=DEFAULT_THRESHOLD_ARTIFACT
    )
    oos = subparsers.add_parser("apply")
    oos.add_argument("--manifest", type=Path, default=builder.DEFAULT_MANIFEST)
    oos.add_argument(
        "--threshold-artifact", type=Path, default=DEFAULT_THRESHOLD_ARTIFACT
    )
    oos.add_argument("--oos-margins", type=Path, default=DEFAULT_OOS_MARGINS)
    oos.add_argument("--output", type=Path, default=DEFAULT_APPLY_OUTPUT)
    oos.add_argument("--signflip-permutations", type=int, default=100_000)
    oos.add_argument("--signflip-seed", type=int, default=20_260_819)
    args = parser.parse_args()
    values = vars(args)
    mode = values.pop("mode")
    return mode, values


def main() -> None:
    mode, values = parse_args()
    result = (
        calibrate(CalibrationConfig(**values))
        if mode == "calibrate"
        else apply(ApplyConfig(**values))
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
