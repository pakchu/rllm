"""Strict v2 evaluator on the repaired COIN-M quarterly source."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate_coinm_roll_migration_pre2024 import (
    EvaluationConfig as BaseEvaluationConfig,
    FULL_WINDOWS,
    WINDOWS,
    _build_trades,
    _schedule_hash,
    _stable_artifact_hash,
    _stats,
    _timestamp,
    _window_trades,
    _write_json_exclusive,
    delayed_schedule,
    selection_gates,
    winner_sort_key,
)
from training.preregister_coinm_roll_migration_alpha import (
    CANDIDATES,
    Candidate,
    build_signal_state,
    candidate_clock,
    canonical_hash,
    load_source,
    windowed_support_summary,
)


SUPPORT_COMMIT = "422d0ef36c60ac38b4a0ea035e356b7c4be5e2fc"
STATIC_INPUT_SHA256 = {
    "training/preregister_coinm_roll_migration_alpha_v2.py": (
        "48cc6b551c5973ad7697875a57ccddb0accf97d81a5d7dffc57d70190fbc3dd8"
    ),
    "docs/coinm-roll-migration-v2-preregistration-2026-07-19.md": (
        "6f196f07742cd64e481577706afabb00a8ab48a52fecfe3e6a3c0c43377f78a4"
    ),
    "results/coinm_roll_migration_support_v2_2026-07-19.json": (
        "9d080c925c3aecb6a501c5ead589d614f53af8c7070a2cc5c9c982ecc4f51b2f"
    ),
    "training/evaluate_coinm_roll_migration_pre2024.py": (
        "b206c5cfd076777557bcb044791a61af157498a0d36360582fbc219786522ca0"
    ),
    "training/evaluate_metaorder_fragmentation_impact_curvature.py": (
        "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
    ),
}
SOURCE_SHA256 = "d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b"
MANIFEST_SHA256 = "29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af"
SUPPORT_RESULT = Path("results/coinm_roll_migration_support_v2_2026-07-19.json")
V1_ABORTED_SELECTION = Path(
    "results/coinm_roll_migration_pre2024_selection_2026-07-19.json"
)
EVALUATOR_SOURCE = Path("training/evaluate_coinm_roll_migration_pre2024_v2.py")
EVALUATOR_FREEZE = Path(
    "results/coinm_roll_migration_evaluator_freeze_v2_2026-07-19.json"
)


@dataclass(frozen=True)
class EvaluationConfig(BaseEvaluationConfig):
    source_csv: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/"
        "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
    )
    manifest_json: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/build_manifest.json"
    )
    output: str = "results/coinm_roll_migration_pre2024_selection_v2_2026-07-19.json"
    freeze_output: str = str(EVALUATOR_FREEZE)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_canonical_config(cfg: EvaluationConfig) -> None:
    if asdict(cfg) != asdict(EvaluationConfig()):
        raise ValueError("all v2 evaluator paths and protocol parameters are frozen")


def _support_stable_payload(support: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in support.items()
        if key not in {"schema_version", "created_at", "support_freeze_hash"}
    }


def _verify_static_dependencies() -> dict[str, Any]:
    if V1_ABORTED_SELECTION.exists():
        raise ValueError("v1 aborted evaluation unexpectedly has a selection artifact")
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen v2 dependency changed: {path}")
    support = _read_json(SUPPORT_RESULT)
    if support.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("v2 support stage opened outcomes")
    if support.get("protocol", {}).get("v1_selection_artifact_created") is not False:
        raise ValueError("v1 failure unexpectedly produced a selection artifact")
    if support.get("support_freeze_hash") != canonical_hash(
        _support_stable_payload(support)
    ):
        raise ValueError("v2 support freeze hash changed")
    if support.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("v2 support source identity changed")
    if support.get("source_manifest", {}).get("sha256") != MANIFEST_SHA256:
        raise ValueError("v2 support manifest identity changed")
    if _sha256(support["source"]["path"]) != SOURCE_SHA256:
        raise ValueError("v2 source bytes changed after support freeze")
    if _sha256(support["source_manifest"]["path"]) != MANIFEST_SHA256:
        raise ValueError("v2 manifest bytes changed after support freeze")
    expected_candidates = [asdict(candidate) for candidate in CANDIDATES]
    if [item.get("candidate") for item in support["candidates"]] != expected_candidates:
        raise ValueError("candidate definitions changed after v2 support freeze")
    if sum(item.get("passes_support") is True for item in support["candidates"]) != 1:
        raise ValueError("unexpected v2 supported-candidate count")
    return support


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_config(cfg)
    support = _verify_static_dependencies()
    if Path(cfg.output).exists():
        raise ValueError("v2 selection result already exists; evaluator cannot be frozen")
    if Path(cfg.freeze_output).exists():
        raise ValueError("v2 evaluator freeze already exists and cannot be replaced")
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
        raise ValueError("v2 outcome source identity changed")
    if _sha256(cfg.manifest_json) != MANIFEST_SHA256:
        raise ValueError("v2 outcome source manifest changed")
    core = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "support_commit": SUPPORT_COMMIT,
        "support_freeze_hash": support["support_freeze_hash"],
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "fixed_v1_evaluator_logic_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_coinm_roll_migration_pre2024.py"
        ],
        "config": asdict(cfg),
        "source_sha256": SOURCE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": [
            "fit",
            "select_2023",
            "test_2024",
            "eval_2025",
            "holdout_2026",
        ],
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    core["freeze_hash"] = _stable_artifact_hash(core)
    _write_json_exclusive(cfg.freeze_output, core)
    return core


def verify_evaluator_freeze(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_config(cfg)
    freeze = _read_json(cfg.freeze_output)
    if freeze.get("freeze_hash") != _stable_artifact_hash(freeze):
        raise ValueError("v2 evaluator freeze hash changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("v2 evaluator source changed after freeze")
    if freeze.get("config") != asdict(cfg):
        raise ValueError("v2 evaluation config changed after freeze")
    if freeze.get("opened_windows") != [] or freeze.get("mutable_parameters") != []:
        raise ValueError("v2 evaluator freeze is not sealed")
    if freeze.get("candidate_returns_computed_before_freeze") is not False:
        raise ValueError("v2 candidate returns were computed before freeze")
    if freeze.get("simulation_run") is not False:
        raise ValueError("v2 evaluator freeze ran a simulation")
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
        raise ValueError("frozen v2 outcome source changed")
    if _sha256(cfg.manifest_json) != MANIFEST_SHA256:
        raise ValueError("frozen v2 outcome manifest changed")
    return freeze


def _supported_candidates(support: dict[str, Any]) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    return [
        by_name[item["candidate"]["name"]]
        for item in support["candidates"]
        if item.get("passes_support") is True
    ]


def _event_clock_hash(
    source: pd.DataFrame,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
) -> str:
    rows = [
        {
            "signal_bar_open": str(source.iloc[position]["signal_bar_open_utc"]),
            "side": int(side[position]),
            "symbol": str(source.iloc[position][f"{candidate.traded_leg}_symbol"]),
        }
        for position in np.flatnonzero(active)
    ]
    return canonical_hash(rows)


def _rebuild_event_clocks(
    cfg: EvaluationConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = load_source(cfg.source_csv)
    state = build_signal_state(source)
    clocks = {
        candidate.name: candidate_clock(source, state, candidate)
        for candidate in CANDIDATES
    }
    return source, clocks


def _verify_rebuilt_support(
    source: pd.DataFrame,
    clocks: dict[str, tuple[np.ndarray, np.ndarray]],
    support: dict[str, Any],
) -> None:
    support_by_name = {
        item["candidate"]["name"]: item for item in support["candidates"]
    }
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.name]
        frozen = support_by_name[candidate.name]
        if _event_clock_hash(source, active, side, candidate) != frozen["clock_hash"]:
            raise ValueError(f"v2 frozen event clock changed: {candidate.name}")
        rebuilt_support, schedule_hashes = windowed_support_summary(
            source, active, side, candidate
        )
        if rebuilt_support != frozen["support"]:
            raise ValueError(f"v2 frozen support changed: {candidate.name}")
        if schedule_hashes != frozen["schedule_hashes"]:
            raise ValueError(f"v2 frozen schedules changed: {candidate.name}")


def _load_outcomes(cfg: EvaluationConfig) -> pd.DataFrame:
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
        raise ValueError("v2 outcome source identity changed")
    columns = [
        "signal_bar_open_utc",
        "front_symbol",
        "next_symbol",
        "front_open",
        "front_high",
        "front_low",
        "front_close",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
    ]
    outcome = pd.read_csv(
        cfg.source_csv,
        compression="infer",
        usecols=lambda column: column in columns,
    )
    outcome["signal_bar_open_utc"] = (
        pd.to_datetime(outcome["signal_bar_open_utc"], utc=True, errors="raise")
        .dt.tz_convert(None)
    )
    dates = outcome["signal_bar_open_utc"]
    if (
        outcome.empty
        or dates.duplicated().any()
        or dates.iloc[0] != pd.Timestamp("2020-07-01")
        or dates.iloc[-1] != pd.Timestamp("2023-12-31 23:50")
    ):
        raise ValueError("v2 outcome source is not the exact sealed interval")
    expected = pd.Series(
        pd.date_range(dates.iloc[0], dates.iloc[-1], freq="5min"),
        name="signal_bar_open_utc",
    )
    if not dates.equals(expected):
        raise ValueError("v2 outcome source is not a complete five-minute grid")
    for leg in ("front", "next"):
        price_columns = [f"{leg}_{field}" for field in ("open", "high", "low", "close")]
        prices = outcome[price_columns].apply(pd.to_numeric, errors="coerce")
        finite_count = np.isfinite(prices.to_numpy(float)).sum(axis=1)
        if ((finite_count != 0) & (finite_count != 4)).any():
            raise ValueError(f"partial {leg} OHLC row in v2 outcome source")
        complete = finite_count == 4
        if (prices.loc[complete] <= 0.0).any().any():
            raise ValueError(f"non-positive {leg} v2 outcome price")
        if (
            prices.loc[complete, f"{leg}_high"]
            < prices.loc[complete, [f"{leg}_open", f"{leg}_close"]].max(axis=1)
        ).any() or (
            prices.loc[complete, f"{leg}_low"]
            > prices.loc[complete, [f"{leg}_open", f"{leg}_close"]].min(axis=1)
        ).any() or (
            prices.loc[complete, f"{leg}_high"]
            < prices.loc[complete, f"{leg}_low"]
        ).any():
            raise ValueError(f"{leg} v2 outcome source violates OHLC invariants")
        outcome.loc[:, price_columns] = prices
    return outcome


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_config(cfg)
    if Path(cfg.output).exists():
        raise ValueError("v2 selection result already exists and cannot be replaced")
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg)
    source, clocks = _rebuild_event_clocks(cfg)
    _verify_rebuilt_support(source, clocks, support)
    outcome = _load_outcomes(cfg)
    if not outcome["signal_bar_open_utc"].equals(source["signal_bar_open_utc"]):
        raise ValueError("v2 signal and outcome clocks differ")
    positions = {
        _timestamp(timestamp): position
        for position, timestamp in enumerate(outcome["signal_bar_open_utc"])
    }
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    support_by_name = {
        item["candidate"]["name"]: item for item in support["candidates"]
    }
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.name]
        windows: dict[str, dict[str, Any]] = {}
        for name in WINDOWS:
            trades, _ = _window_trades(
                source, outcome, positions, active, side, candidate, name
            )
            windows[name] = _stats(
                trades, name, cfg, cluster=name in FULL_WINDOWS
            )

        stress: dict[str, dict[str, Any]] = {}
        direction_flip: dict[str, dict[str, Any]] = {}
        delay_1h: dict[str, dict[str, Any]] = {}
        delay_24h: dict[str, dict[str, Any]] = {}
        control_schedule_hashes = {"delay_1h": {}, "delay_24h": {}}
        for name in FULL_WINDOWS:
            base_trades, base_schedule = _window_trades(
                source, outcome, positions, active, side, candidate, name
            )
            one_hour_schedule = delayed_schedule(
                base_schedule,
                source,
                candidate,
                bars=cfg.delay_1h_bars,
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
            )
            day_schedule = delayed_schedule(
                base_schedule,
                source,
                candidate,
                bars=cfg.delay_24h_bars,
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
            )
            flip_trades = _build_trades(
                outcome, positions, base_schedule, candidate, flip=True
            )
            one_hour_trades = _build_trades(
                outcome, positions, one_hour_schedule, candidate
            )
            day_trades = _build_trades(
                outcome, positions, day_schedule, candidate
            )
            control_schedule_hashes["delay_1h"][name] = _schedule_hash(
                one_hour_schedule
            )
            control_schedule_hashes["delay_24h"][name] = _schedule_hash(day_schedule)
            stress[name] = _stats(
                base_trades,
                name,
                cfg,
                cost_rate=cfg.stress_cost_rate_per_side,
            )
            direction_flip[name] = _stats(flip_trades, name, cfg)
            delay_1h[name] = _stats(one_hour_trades, name, cfg)
            delay_24h[name] = _stats(day_trades, name, cfg)

        gates = selection_gates(
            windows, stress, direction_flip, delay_1h, delay_24h
        )
        frozen = support_by_name[candidate.name]
        row = {
            "candidate": asdict(candidate),
            "name": candidate.name,
            "clock_hash": frozen["clock_hash"],
            "windows": windows,
            "stress_10bp_per_side": stress,
            "direction_flip": direction_flip,
            "delay_1h": delay_1h,
            "delay_24h": delay_24h,
            "control_schedule_hashes": control_schedule_hashes,
            "gates": gates,
            "passes_selection": bool(all(gates.values())),
        }
        rows.append(row)
        if row["passes_selection"]:
            eligible.append(row)

    eligible.sort(key=winner_sort_key)
    winner = eligible[0] if eligible else None
    stable = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "support_commit": SUPPORT_COMMIT,
            "support_freeze_hash": support["support_freeze_hash"],
            "evaluator_freeze_hash": freeze["freeze_hash"],
            "fixed_v1_evaluator_logic_sha256": STATIC_INPUT_SHA256[
                "training/evaluate_coinm_roll_migration_pre2024.py"
            ],
            "opened_windows": ["fit", "select_2023"],
            "sealed_windows": ["test_2024", "eval_2025", "holdout_2026"],
            "full_calendar_cagr": True,
            "strict_mdd": (
                "global/pre-entry HWM plus entry-cost/favorable-before-adverse held "
                "five-minute path, hypothetical exit cost, and realized two-sided cost"
            ),
            "inverse_ledger": (
                "exact coin PnL converted to USD at each mark; fractional fixed-USD-face "
                "research contracts; no delivery-futures funding"
            ),
            "production_rounding_modeled": False,
            "btc_collateral_beta_modeled": False,
            "post_selection_parameter_repair_allowed": False,
        },
        "config": asdict(cfg),
        "candidates_evaluated": len(rows),
        "candidates_passing": len(eligible),
        "winner": winner,
        "advance_to_2024_test": winner is not None,
        "candidates": rows,
    }
    report = {**stable, "result_hash": _stable_artifact_hash(stable)}
    _write_json_exclusive(cfg.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    freeze_only = parser.parse_args().freeze_only
    cfg = EvaluationConfig()
    report = freeze_evaluator(cfg) if freeze_only else evaluate(cfg)
    keys = (
        ["freeze_hash"]
        if freeze_only
        else ["candidates_evaluated", "candidates_passing", "advance_to_2024_test"]
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
