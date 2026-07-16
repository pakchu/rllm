"""One-shot strict 2023-2024 selector for frozen PFCR-2.

The exact support clock, evaluator source, tests, configuration, and strict
simulator dependency must be committed and frozen before this module loads any
post-entry market or funding outcomes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import select_leave_one_out_residual_exhaustion_pre2025 as lore
from training.build_post_funding_crowding_release_episode_v2_support import (
    assert_clock_contract,
)
from training.preregister_post_funding_crowding_release_episode_v2 import canonical_hash


START = pd.Timestamp("2023-01-01 00:00:00")
MID = pd.Timestamp("2024-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")

EVALUATION_SOURCE = Path(
    "training/evaluate_post_funding_crowding_release_episode_v2_2023_2024.py"
)
TEST_PATH = Path(
    "tests/test_evaluate_post_funding_crowding_release_episode_v2_2023_2024.py"
)
EVALUATION_FREEZE = Path(
    "results/post_funding_crowding_release_episode_v2_evaluator_freeze_2026-07-17.json"
)
PREREGISTRATION = Path(
    "results/post_funding_crowding_release_episode_v2_preregistration_2026-07-17.json"
)
SUPPORT_MANIFEST = Path(
    "results/post_funding_crowding_release_episode_v2_support_2026-07-17.json"
)
CLOCK_PATH = Path(
    "data/post_funding_crowding_release_episode_v2_clock_2023_2024.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "results/post_funding_crowding_release_episode_v2_selection_2023_2024_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/post-funding-crowding-release-episode-v2-selection-2023-2024-2026-07-17.md"
)

PREREGISTRATION_SHA256 = "14af65aa684033d85210a6d28d98571b00cf0b07d2dcdbe6206a5de7a864f59b"
SUPPORT_MANIFEST_SHA256 = "d09c3bc68efa541da00ab994ffac55f64cee1152c148361252e0206b53ffe083"
EXPECTED_SUPPORT_MANIFEST_HASH = "86d91d5fced3270c818f448f511413fab76473fba55d7f5eee0133d50f43930e"
CLOCK_SHA256 = "ebeb32ccaf1bc096c95f5c848ed34c6964d5be828555a8024a42a8f826586fbc"
STRICT_SIMULATOR_SOURCE_SHA256 = "5235514371ff89632f8378949398648550eefeec5d70fa83ab115fe1a1a9cbb4"
PFCR2_SUPPORT_SOURCE_SHA256 = "8f987d41e63c06e07ca5380cdcf439524045deadbe51d98f9a851e514ba5f2cd"
PFCR2_PREREGISTRATION_SOURCE_SHA256 = "6e1de0ec3a7dc6fd397c5c51231fa4c82afa8740f2a03d23ea45cb0c56eca2c3"
LORE_SOURCE_MANIFEST_SHA256 = "b3f5841f10b3e44ee47fb5d69c7acc6a2df0975596cdc0fd4019925f49b6eb66"
LORE_SOURCE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"


@dataclass(frozen=True)
class EvaluationConfig:
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    entry_delay_minutes: int = 5
    fake_settlement_shift_hours: int = 4
    cluster_signflip_samples: int = 20_000
    cluster_signflip_seed: int = 20_260_717


CONFIG = EvaluationConfig()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _body_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(
        {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    )


def verify_support_and_clock() -> tuple[dict[str, Any], pd.DataFrame]:
    dependencies = (
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT_MANIFEST, SUPPORT_MANIFEST_SHA256),
        (CLOCK_PATH, CLOCK_SHA256),
        (Path(lore.SELECTOR_PATH), STRICT_SIMULATOR_SOURCE_SHA256),
        (
            Path("training/build_post_funding_crowding_release_episode_v2_support.py"),
            PFCR2_SUPPORT_SOURCE_SHA256,
        ),
        (
            Path("training/preregister_post_funding_crowding_release_episode_v2.py"),
            PFCR2_PREREGISTRATION_SOURCE_SHA256,
        ),
        (Path(lore.SOURCE_MANIFEST), LORE_SOURCE_MANIFEST_SHA256),
    )
    for path, expected in dependencies:
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen PFCR-2 dependency changed: {path}")
    support = json.loads(SUPPORT_MANIFEST.read_text())
    if support.get("manifest_hash") != EXPECTED_SUPPORT_MANIFEST_HASH:
        raise RuntimeError("PFCR-2 support manifest identity changed")
    if _body_hash(support) != EXPECTED_SUPPORT_MANIFEST_HASH:
        raise RuntimeError("PFCR-2 support manifest body changed")
    if support.get("post_entry_returns_calculated") is not False:
        raise RuntimeError("PFCR-2 support artifact already opened outcomes")
    if support.get("support", {}).get("passes_support") is not True:
        raise RuntimeError("PFCR-2 support gate did not pass")
    if support.get("clock_sha256") != CLOCK_SHA256:
        raise RuntimeError("PFCR-2 support-to-clock binding changed")
    if lore.EXPECTED_SOURCE_MANIFEST_HASH != LORE_SOURCE_MANIFEST_HASH:
        raise RuntimeError("strict simulator source-manifest identity changed")
    clock = pd.read_csv(CLOCK_PATH)
    assert_clock_contract(clock)
    for column in ("settlement_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if len(clock) != 82:
        raise RuntimeError("PFCR-2 frozen clock row count changed")
    return support, clock


def execution_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    out["signal_time"] = pd.to_datetime(out["settlement_time"], errors="raise")
    out["choice"] = "crowding_release"
    out["gross_scale"] = 1.0
    out["predicted_edge"] = out["current_funding_spread"] - out["prior_spread_q90"]
    out["confidence_threshold"] = out["prior_spread_q90"]
    columns = (
        "policy_id",
        "settlement_time",
        "signal_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
        "long_symbol",
        "short_symbol",
        "long_weight",
        "short_weight_abs",
        "long_beta",
        "short_beta",
        "choice",
        "gross_scale",
        "predicted_edge",
        "confidence_threshold",
        "current_funding_spread",
        "prior_spread_q90",
    )
    out = out.loc[:, columns].sort_values("entry_time").reset_index(drop=True)
    if not (out["signal_time"] < out["feature_available_time"]).all():
        raise RuntimeError("PFCR-2 settlement did not precede feature availability")
    if not (out["feature_available_time"] < out["entry_time"]).all():
        raise RuntimeError("PFCR-2 feature availability crossed entry")
    if not np.allclose(out["long_weight"] + out["short_weight_abs"], 1.0):
        raise RuntimeError("PFCR-2 execution clock lost gross-one sizing")
    exposure = out["long_weight"] * out["long_beta"] - out[
        "short_weight_abs"
    ] * out["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("PFCR-2 execution clock lost beta neutrality")
    return out


def transform_clock(clock: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = clock.copy()
    if kind == "direction_flip":
        out[["long_symbol", "short_symbol"]] = out[
            ["short_symbol", "long_symbol"]
        ].to_numpy()
        out[["long_weight", "short_weight_abs"]] = out[
            ["short_weight_abs", "long_weight"]
        ].to_numpy()
        out[["long_beta", "short_beta"]] = out[
            ["short_beta", "long_beta"]
        ].to_numpy()
        out["choice"] = "direction_flip"
    elif kind == "delay_five_minutes":
        delta = pd.Timedelta(minutes=CONFIG.entry_delay_minutes)
        out["entry_time"] = pd.to_datetime(out["entry_time"]) + delta
        out["exit_time"] = pd.to_datetime(out["exit_time"]) + delta
    elif kind == "fake_settlement_plus_four_hours":
        delta = pd.Timedelta(hours=CONFIG.fake_settlement_shift_hours)
        for column in (
            "settlement_time",
            "signal_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ):
            out[column] = pd.to_datetime(out[column]) + delta
    else:
        raise ValueError(kind)
    return out


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.exists():
        raise RuntimeError("PFCR-2 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("PFCR-2 evaluator freeze hash mismatch")
    checks = {
        "outcomes_opened": False,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
        "test_path": str(TEST_PATH),
        "test_sha256": _sha256(TEST_PATH),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_sha256": CLOCK_SHA256,
        "strict_simulator_source_sha256": STRICT_SIMULATOR_SOURCE_SHA256,
        "lore_source_manifest_sha256": LORE_SOURCE_MANIFEST_SHA256,
        "lore_source_manifest_hash": LORE_SOURCE_MANIFEST_HASH,
        "evaluation_config": asdict(CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"PFCR-2 evaluator freeze changed: {key}")
    return payload


def _clean_repository_head() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before PFCR-2 outcomes open")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_bundle() -> lore.MarketBundle:
    verify_evaluation_freeze()
    return lore.load_bundle()


def _simulate(
    bundle: lore.MarketBundle,
    clock: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cost_bp: float,
) -> dict[str, Any]:
    return lore.simulate(
        bundle,
        clock,
        start=str(start.date()),
        end=str(end.date()),
        cost_bp=cost_bp,
    )


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def selection_checks(
    primary: dict[str, dict[str, Any]],
    stress: dict[str, Any],
    delay: dict[str, Any],
    opposite: dict[str, Any],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    combined = primary["combined_2023_2024"]
    return {
        "2023_absolute_return_positive": primary["2023"]["absolute_return_pct"] > 0.0,
        "2024_absolute_return_positive": primary["2024"]["absolute_return_pct"] > 0.0,
        "2023_cagr_to_strict_mdd_at_least_1_5": (
            primary["2023"]["cagr_to_strict_mdd"] >= 1.5
        ),
        "2024_cagr_to_strict_mdd_at_least_1_5": (
            primary["2024"]["cagr_to_strict_mdd"] >= 1.5
        ),
        "combined_cagr_to_strict_mdd_at_least_3": (
            combined["cagr_to_strict_mdd"] >= 3.0
        ),
        "combined_strict_mdd_at_most_15": combined["strict_mdd_pct"] <= 15.0,
        "combined_trades_at_least_60": combined["trades"] >= 60,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "entry_delay_plus_5m_absolute_return_positive": delay["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": opposite["cagr_pct"] < combined["cagr_pct"],
        "weekly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
    }


def evaluate(bundle: lore.MarketBundle, clock: pd.DataFrame) -> dict[str, Any]:
    windows = {
        "2023": (START, MID),
        "2024": (MID, END),
        "combined_2023_2024": (START, END),
    }
    primary_raw = {
        name: _simulate(
            bundle,
            clock,
            start,
            end,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
        for name, (start, end) in windows.items()
    }
    stress_raw = _simulate(
        bundle,
        clock,
        START,
        END,
        cost_bp=CONFIG.stress_cost_bp_per_notional_side,
    )
    delayed_raw = _simulate(
        bundle,
        transform_clock(clock, "delay_five_minutes"),
        START,
        END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    opposite_raw = _simulate(
        bundle,
        transform_clock(clock, "direction_flip"),
        START,
        END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    fake_raw = _simulate(
        bundle,
        transform_clock(clock, "fake_settlement_plus_four_hours"),
        START,
        END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    signflip = lore.weekly_cluster_signflip(
        primary_raw["combined_2023_2024"]["trade_rows"],
        seed=CONFIG.cluster_signflip_seed,
        samples=CONFIG.cluster_signflip_samples,
    )
    primary = {name: _slim(stats) for name, stats in primary_raw.items()}
    stress, delayed, opposite, fake = map(
        _slim, (stress_raw, delayed_raw, opposite_raw, fake_raw)
    )
    checks = selection_checks(primary, stress, delayed, opposite, signflip)
    return {
        "primary": primary,
        "ten_bp_notional_side_cost_stress": stress,
        "entry_delay_plus_5m": delayed,
        "direction_flip": opposite,
        "fake_settlement_plus_four_hours": fake,
        "weekly_cluster_signflip": signflip,
        "selection_gates": checks,
        "passes_2023_2024_selection": all(checks.values()),
        "executed_trade_rows": primary_raw["combined_2023_2024"]["trade_rows"],
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# PFCR-2 strict 2023–2024 one-shot selection — 2026-07-17",
        "",
        f"- Decision: **{result['decision']}**",
        "- 2025 and 2026 remain sealed.",
        "",
        "| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("2023", "2023"),
        ("2024", "2024"),
        ("combined_2023_2024", "2023–2024"),
    ):
        stats = evaluation["primary"][key]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | "
            f"{stats['trades']} |"
        )
    lines.extend(
        [
            "",
            "| Control, 2023–2024 | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("ten_bp_notional_side_cost_stress", "10 bp/side"),
        ("entry_delay_plus_5m", "Entry/exit +5m"),
        ("direction_flip", "Direction flip"),
        ("fake_settlement_plus_four_hours", "Fake settlement +4h"),
    ):
        stats = evaluation[key]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | "
            f"{stats['trades']} |"
        )
    failed = [name for name, passed in evaluation["selection_gates"].items() if not passed]
    lines.extend(
        [
            "",
            f"- Weekly-cluster sign-flip p: `{evaluation['weekly_cluster_signflip']['raw_p_value']:.6f}`",
            f"- Failed gates: `{failed}`",
            "- CAGR includes the complete declared calendar, including idle time.",
            "- Strict MDD uses global/pre-entry HWM, favorable-before-adverse two-leg OHLC, "
            "entry/hypothetical-liquidation/exit costs, and exact held-interval funding.",
            "- No threshold, sign, cooldown, hold, pair, or beta repair is permitted after this opening.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    output_path = Path(output)
    if output_path.exists():
        raise RuntimeError("refusing to overwrite one-shot PFCR-2 2023-2024 outcomes")
    support, frozen_clock = verify_support_and_clock()
    evaluator_freeze = verify_evaluation_freeze()
    opening_commit = _clean_repository_head()
    bundle = load_bundle()
    clock = execution_clock(frozen_clock)
    evaluation = evaluate(bundle, clock)
    passed = evaluation["passes_2023_2024_selection"]
    core: dict[str, Any] = {
        "protocol_version": "pfcr_v2_selection_2023_2024_2026-07-17",
        "outcomes_opened": True,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "opened_windows": ["2023", "2024"],
        "sealed_windows": ["2025", "2026"],
        "opening_commit": opening_commit,
        "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
        "evaluation_source_sha256": evaluator_freeze["evaluation_source_sha256"],
        "evaluator_freeze_sha256": _sha256(EVALUATION_FREEZE),
        "support_manifest_hash": support["manifest_hash"],
        "clock_sha256": CLOCK_SHA256,
        "strict_simulator_source_sha256": STRICT_SIMULATOR_SOURCE_SHA256,
        "config": asdict(CONFIG),
        "evaluation": evaluation,
        "decision": (
            "passed_2023_2024_pending_2025"
            if passed
            else "rejected_before_2025_no_outcome_repair"
        ),
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(result))
    return result


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "primary": result["evaluation"]["primary"],
                "controls": {
                    key: result["evaluation"][key]
                    for key in (
                        "ten_bp_notional_side_cost_stress",
                        "entry_delay_plus_5m",
                        "direction_flip",
                        "fake_settlement_plus_four_hours",
                    )
                },
                "weekly_cluster_signflip": result["evaluation"][
                    "weekly_cluster_signflip"
                ],
                "selection_gates": result["evaluation"]["selection_gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
