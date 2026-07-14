"""Frozen pre-2024 return evaluator for CSPR-12.

The evaluator must be committed and recorded in the evaluator-freeze manifest
before :func:`run_evaluation` can load any execution price.  It replays the
frozen primary clock exactly, constructs only the preregistered controls, and
opens 2020-2023 outcomes.  Calendar 2024 onward remains sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cash_sponsored_perp_rejection as cspr
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    simulate_schedule,
)


PREREGISTRATION_COMMIT = "b45fca0eeecc942df3f37b9f057697a117871cc1"
SUPPORT_COMMIT = "f91661530bbe740dc7b6182ed321dadb3f4dc600"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cash_sponsored_perp_rejection.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "6dff5875aa98c9c74a8c27c0488890ed3d0aeb76175009e4721f7dd8336b36a1"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cash-sponsored-perp-rejection-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "bdfcb522e5e2974d6cb4201befe68c67622b05645014c239534833c3785c7ab2"
)
SUPPORT_DOCUMENT = Path(
    "docs/cash-sponsored-perp-rejection-support-freeze-2026-07-14.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "1ced9133793196df33d8fd22d027bcd640596de38b69f1a955262a17f78d5f07"
)
EVALUATOR_DOCUMENT = Path(
    "docs/cash-sponsored-perp-rejection-evaluator-freeze-2026-07-14.md"
)
EVALUATOR_DOCUMENT_SHA256 = (
    "3cfffedced431240d6fdfcf75e8b4a1b08b6fc3c393abc827a63207070f93c2d"
)
PREREGISTRATION_RESULT = Path(
    "results/cash_sponsored_perp_rejection_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "6b3d67a2a1ee4feccbab8368ae038929ad22ef9ebe99055d89a00abcb0ca038a"
)
EVENT_CLOCK = Path(
    "results/cash_sponsored_perp_rejection_clock_2026-07-14.csv"
)
EVENT_CLOCK_SHA256 = (
    "353e14cc09b79960938802c9882ba36527e9f4c4819f8a492312fdefffdf1c0f"
)
EVENT_CLOCK_MANIFEST = Path(
    "results/cash_sponsored_perp_rejection_clock_manifest_2026-07-14.json"
)
EVENT_CLOCK_MANIFEST_SHA256 = (
    "37132d281942ac8f0e72edf239358ac74b9622db6401581461dca4ae4494caa2"
)
CAUSAL_LOADER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
CAUSAL_LOADER_SOURCE_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
TRADE_STATS_SOURCE = Path("training/strict_bar_backtest.py")
TRADE_STATS_SOURCE_SHA256 = (
    "3e95ad320d8869755afa1f4907d2d478200a3ebfc015e4eaeace0be0b15f9682"
)
PERP_MANIFEST = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
PERP_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
SPOT_MANIFEST = Path(
    "data/binance_spot_kline_microstructure_btc_2020_2023/build_manifest.json"
)
SPOT_MANIFEST_SHA256 = (
    "69fbce64b4860eecbf1ce414ea719b5c4001852016fe439e61240e050b39b57b"
)
EVALUATION_SOURCE = Path(
    "training/evaluate_cash_sponsored_perp_rejection.py"
)
EVALUATION_FREEZE = Path(
    "results/cash_sponsored_perp_rejection_evaluator_freeze_2026-07-14.json"
)

SELECTED_QUANTILE = 0.50
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "signal_delay_1bar",
    "no_centroid",
    "no_perp_event_confirmation",
    "spot_only",
    "perp_only",
    "role_swap",
    "spot_lag_1h",
    "spot_lag_24h",
)
QUALIFICATION_CONTROLS = POLICY_NAMES[1:]
CLOCK_COLUMNS = (
    "signal_position",
    "entry_position",
    "exit_position",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "branch",
    "hold_bars",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cash_sponsored_perp_rejection_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_evaluation_config(cfg: EvaluationConfig) -> None:
    expected = {
        "leverage": 0.5,
        "fee_rate": 0.0005,
        "slippage_rate": 0.0001,
        "cluster_permutations": 100_000,
        "cluster_seed": 20_260_714,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"CSPR evaluation config is frozen: {changed}")


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("CSPR evaluator freeze manifest is missing")
    freeze = json.loads(EVALUATION_FREEZE.read_text())
    if freeze.get("outcomes_opened_for_cspr12") is not False:
        raise ValueError("CSPR evaluator was not frozen before outcomes")
    if freeze.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("CSPR evaluator freeze path changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("CSPR evaluator differs from pre-outcome freeze")
    commit = str(freeze.get("evaluation_source_commit", ""))
    if len(commit) != 40:
        raise ValueError("CSPR evaluator source commit is not full length")
    if freeze.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("CSPR evaluator freeze preregistration changed")
    if freeze.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("CSPR evaluator freeze support commit changed")
    if freeze.get("support_result_sha256") != PREREGISTRATION_RESULT_SHA256:
        raise ValueError("CSPR evaluator freeze support hash changed")
    if freeze.get("event_clock_sha256") != EVENT_CLOCK_SHA256:
        raise ValueError("CSPR evaluator freeze event clock changed")
    if freeze.get("opened_windows") != []:
        raise ValueError("CSPR evaluator freeze already opened a window")
    expected_sealed = [*WINDOWS, "test2024", "eval2025", "ytd2026"]
    if freeze.get("sealed_windows") != expected_sealed:
        raise ValueError("CSPR evaluator freeze sealed windows changed")
    return freeze


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (EVALUATOR_DOCUMENT, EVALUATOR_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (EVENT_CLOCK, EVENT_CLOCK_SHA256),
        (EVENT_CLOCK_MANIFEST, EVENT_CLOCK_MANIFEST_SHA256),
        (CAUSAL_LOADER_SOURCE, CAUSAL_LOADER_SOURCE_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (PERP_MANIFEST, PERP_MANIFEST_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
        (SPOT_MANIFEST, SPOT_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen CSPR dependency changed: {path}")

    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("CSPR support artifact opened outcomes")
    if protocol.get("support_only") is not True:
        raise ValueError("CSPR support artifact is not support-only")
    if result.get("support_decision") != "pass":
        raise ValueError("CSPR support gate did not pass")
    if result.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CSPR selected quantile changed")
    if result.get("config") != asdict(cspr.Config()):
        raise ValueError("CSPR signal config differs from frozen support")
    selected = result.get("selected_support", {})
    if selected.get("passes_support") is not True:
        raise ValueError("CSPR selected support row is not passing")
    if selected.get("support", {}).get("nonoverlap_total") != 850:
        raise ValueError("CSPR selected event count changed")

    manifest = json.loads(EVENT_CLOCK_MANIFEST.read_text())
    frozen_protocol = manifest.get("protocol", {})
    clock = manifest.get("clock", {})
    if frozen_protocol.get("outcomes_opened") is not False:
        raise ValueError("CSPR event clock opened outcomes")
    if frozen_protocol.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("CSPR event clock preregistration changed")
    if frozen_protocol.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CSPR event clock quantile changed")
    if clock.get("sha256") != EVENT_CLOCK_SHA256 or clock.get("rows") != 850:
        raise ValueError("CSPR event clock record changed")
    if manifest.get("support", {}).get("sha256") != PREREGISTRATION_RESULT_SHA256:
        raise ValueError("CSPR event clock support hash changed")
    return result


def _canonical_clock_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(CLOCK_COLUMNS).difference(schedule.columns)
    if missing:
        raise ValueError(f"CSPR clock is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for row in schedule[list(CLOCK_COLUMNS)].itertuples(index=False):
        records.append(
            {
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "hold_bars": int(row.hold_bars),
            }
        )
    return records


def verify_signal_replay(
    frame: pd.DataFrame,
    cfg: cspr.Config,
    preregistration: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    signal, controls = cspr.classify_events(
        frame,
        cfg,
        quantile=SELECTED_QUANTILE,
    )
    replayed = cspr.nonoverlapping_schedule(signal, frame)
    frozen = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock_records(replayed) != _canonical_clock_records(frozen):
        raise ValueError("CSPR primary event-clock replay differs from freeze")
    if not replayed["entry_position"].eq(replayed["signal_position"] + 1).all():
        raise ValueError("CSPR replay is not next-open entry")
    if not replayed["hold_bars"].eq(cfg.hold_bars).all():
        raise ValueError("CSPR replay hold changed")
    if not replayed["exit_position"].eq(
        replayed["entry_position"] + cfg.hold_bars
    ).all():
        raise ValueError("CSPR replay exit changed")
    support = cspr._support(replayed, frame, cfg)
    if support != preregistration["selected_support"]["support"]:
        raise ValueError("CSPR selected support replay differs from freeze")
    if int(controls["primary"].sum()) != preregistration["selected_support"][
        "raw_primary"
    ]:
        raise ValueError("CSPR raw primary clock differs from freeze")
    clock_manifest = json.loads(EVENT_CLOCK_MANIFEST.read_text())
    if source != clock_manifest.get("source"):
        raise ValueError("CSPR source replay differs from clock freeze")
    return controls, replayed.reset_index(drop=True)


def _schedule_from_control(
    frame: pd.DataFrame,
    cfg: cspr.Config,
    *,
    mask: pd.Series,
    side: pd.Series,
    quarantine: pd.Series,
    branch: str,
) -> pd.DataFrame:
    signal = pd.DataFrame(
        {
            "side": np.where(mask.fillna(False), side.fillna(0.0), 0.0).astype(
                np.int8
            ),
            "branch": np.where(mask.fillna(False), branch, "none"),
            "hold_bars": np.where(mask.fillna(False), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )
    schedule_frame = frame.assign(quarantined=quarantine.fillna(True).astype(bool))
    return cspr.nonoverlapping_schedule(signal, schedule_frame).reindex(
        columns=CLOCK_COLUMNS
    ).reset_index(drop=True)


def build_control_schedules(
    frame: pd.DataFrame,
    controls: dict[str, pd.Series],
    primary_schedule: pd.DataFrame,
    cfg: cspr.Config,
) -> dict[str, pd.DataFrame]:
    """Build only preregistered controls on their causally required sources."""
    values = cspr._directions(frame)
    spot_side = values["side"]
    perp_side = pd.Series(
        np.sign(values["perp_return"]), index=frame.index, dtype=float
    )
    joint_clean = frame["quarantined"].astype(bool)
    spot_quarantine = frame["spot_quarantined"].astype(bool)
    perp_quarantine = frame["perp_quarantined"].astype(bool)
    frozen_primary = primary_schedule.reindex(columns=CLOCK_COLUMNS).reset_index(
        drop=True
    )
    schedules: dict[str, pd.DataFrame] = {"primary": frozen_primary}

    direction_flip = frozen_primary.copy()
    direction_flip["side"] = -direction_flip["side"].astype(np.int8)
    direction_flip["branch"] = "direction_flip"
    schedules["direction_flip"] = direction_flip.reset_index(drop=True)

    specifications = {
        "signal_delay_1bar": (
            spot_side.shift(1),
            joint_clean.shift(1, fill_value=True),
        ),
        "no_centroid": (spot_side, joint_clean),
        "no_perp_event_confirmation": (spot_side, joint_clean),
        "spot_only": (spot_side, spot_quarantine),
        "perp_only": (perp_side, perp_quarantine),
        "role_swap": (perp_side, joint_clean),
        "spot_lag_1h": (spot_side.shift(12), perp_quarantine),
        "spot_lag_24h": (spot_side.shift(288), perp_quarantine),
    }
    for name, (side, quarantine) in specifications.items():
        schedules[name] = _schedule_from_control(
            frame,
            cfg,
            mask=controls[name],
            side=side,
            quarantine=quarantine,
            branch=name,
        )
    if tuple(schedules) != POLICY_NAMES:
        raise ValueError("CSPR evaluator control set differs from preregistration")
    return schedules


def slice_schedule(
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("CSPR split start must precede end")
    if schedule.empty:
        return schedule.copy()
    signal_date = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry_date = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_date = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        signal_date.ge(start_timestamp)
        & signal_date.lt(end_timestamp)
        & entry_date.ge(start_timestamp)
        & entry_date.lt(end_timestamp)
        & exit_date.ge(start_timestamp)
        & exit_date.lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def evaluate_policy(
    frame: pd.DataFrame,
    global_schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    schedule = slice_schedule(global_schedule, start=start, end=end)
    metrics = simulate_schedule(
        frame,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    metrics["global_clock_count"] = int(len(global_schedule))
    metrics["split_clock_count"] = int(len(schedule))
    return metrics


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train"]["primary"]
    select = windows["select2023"]["primary"]
    h1 = windows["select2023_h1"]["primary"]
    h2 = windows["select2023_h2"]["primary"]
    failures: list[str] = []

    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")

    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 30:
            failures.append(f"{name}: fewer than 30 trades")
    if select["trade_count"] < 80:
        failures.append("select2023: fewer than 80 trades")

    primary_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    control_min_ratios: dict[str, float] = {}
    for control in QUALIFICATION_CONTROLS:
        control_min = min(
            windows["train"][control]["cagr_to_strict_mdd"],
            windows["select2023"][control]["cagr_to_strict_mdd"],
        )
        control_min_ratios[control] = float(control_min)
        if primary_min_ratio <= control_min:
            failures.append(
                "primary: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "primary_min_train_select_ratio": float(primary_min_ratio),
        "control_min_train_select_ratios": control_min_ratios,
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    _validate_evaluation_config(cfg)
    evaluator_freeze = verify_evaluation_freeze()
    preregistration = verify_preregistration()
    signal_cfg = cspr.Config()
    frame, source = cspr.load_causal_frame(signal_cfg)
    controls, primary_schedule = verify_signal_replay(
        frame,
        signal_cfg,
        preregistration,
        source,
    )
    schedules = build_control_schedules(
        frame,
        controls,
        primary_schedule,
        signal_cfg,
    )

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        windows[window_name] = {
            policy: evaluate_policy(
                frame,
                schedules[policy],
                start=start,
                end=end,
                cfg=cfg,
            )
            for policy in POLICY_NAMES
        }

    qualification = _qualification(windows)
    return {
        "protocol": {
            "name": "CSPR-12 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "evaluation_source_commit": evaluator_freeze[
                "evaluation_source_commit"
            ],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_manifest_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened_for_cspr12": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "primary_clock_replayed_from_freeze": True,
            "control_clocks_reserved_before_return_slicing": True,
            "entry": "next 5m open after completed Spot and USD-M bars",
            "exit": "scheduled open 12 completed 5m bars after entry",
            "cost_multiplier": "(1-0.0003)*(1+0.5*r)*(1-0.0003)",
            "strict_mdd": (
                "held path, favorable extreme first then adverse; exit-bar "
                "high/low excluded"
            ),
            "cagr": "full wall-clock split including idle cash",
            "control_actions": {
                "direction_flip": "negative primary side on exact primary clock",
                "signal_delay_1bar": "primary side and signal shifted one 5m bar",
                "no_centroid": "Spot taker-flow side on no-centroid clock",
                "no_perp_event_confirmation": (
                    "Spot taker-flow side without USD-M event-count confirmation"
                ),
                "spot_only": "Spot taker-flow side using Spot-only quarantine",
                "perp_only": "USD-M price-return side using USD-M-only quarantine",
                "role_swap": "USD-M price-return side on role-swapped clock",
                "spot_lag_1h": "Spot taker-flow side stale by 12 bars",
                "spot_lag_24h": "Spot taker-flow side stale by 288 bars",
            },
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "global_clock_counts": {
            policy: int(len(schedule)) for policy, schedule in schedules.items()
        },
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "cspr12" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen pre-2024 gate"
                if qualification["qualifies"]
                else "CSPR-12 failed at least one frozen pre-2024 gate"
            ),
        },
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trade_count",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    args = parser.parse_args()
    cfg = EvaluationConfig(output=args.output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "primary": {
                    name: _headline(policies["primary"])
                    for name, policies in result["windows"].items()
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
