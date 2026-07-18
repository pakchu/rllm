"""Freeze v2 support on the repaired COIN-M quarterly strip.

All event definitions, thresholds, holds, and support gates are imported from
the already-frozen v1 preregistration.  This wrapper changes only the sealed
source identity and artifact path after the v1 evaluator exposed a missing
outcome path and aborted before producing any return statistic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.preregister_coinm_roll_migration_alpha import (
    CANDIDATES,
    DELIVERY_BUFFER_HOURS,
    MAX_FRONT_DTE_HOURS,
    MIN_FRONT_SHARE,
    MIN_NEXT_SHARE,
    ROBUST_MIN_PERIODS,
    ROBUST_WINDOW_BARS,
    build_signal_state,
    candidate_clock,
    canonical_hash,
    load_source,
    support_gates,
    windowed_support_summary,
    write_exclusive,
)


BASE_PREREGISTRATION = Path("training/preregister_coinm_roll_migration_alpha.py")
BASE_PREREGISTRATION_SHA256 = (
    "e8872482d6382e51eb0d80400c662ac1c4bd5626a8653b14b6489d8b2df8b3a3"
)
EXPECTED_SOURCE_SHA256 = (
    "d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b"
)
EXPECTED_MANIFEST_SHA256 = (
    "29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af"
)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/"
        "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
    )
    manifest_json: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/build_manifest.json"
    )
    output: str = "results/coinm_roll_migration_support_v2_2026-07-19.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_path() -> str:
    source = Path(__file__).resolve()
    return str(source.relative_to(source.parents[1]))


def verify_fixed_logic() -> dict[str, str]:
    observed = sha256_file(BASE_PREREGISTRATION)
    if observed != BASE_PREREGISTRATION_SHA256:
        raise ValueError("v1 frozen preregistration logic changed")
    return {"path": str(BASE_PREREGISTRATION), "sha256": observed}


def verify_source_seal(cfg: Config) -> dict[str, Any]:
    source_hash = sha256_file(cfg.input_csv)
    manifest_hash = sha256_file(cfg.manifest_json)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("repaired quarterly source SHA mismatch")
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("repaired quarterly manifest SHA mismatch")
    manifest = json.loads(Path(cfg.manifest_json).read_text())
    if manifest.get("output_sha256") != source_hash:
        raise ValueError("repaired manifest does not bind source output")
    if manifest.get("rows") != 368_351 or manifest.get("valid_rows") != 368_180:
        raise ValueError("repaired manifest row counts changed")
    if manifest.get("invalid_reasons") != {"ok": 368_180, "next_row_missing": 171}:
        raise ValueError("repaired source validity profile changed")
    overlap = manifest.get("monthly_overlap_diagnostics", {})
    if overlap.get("conflict_rows") != 2:
        raise ValueError("repaired overlap revision count changed")
    if overlap.get("conflict_sha256") != (
        "b73a21bb4d4e7024edb71cb2f11a6ae26762b0d1a0bc548a00a07d4ffaf58028"
    ):
        raise ValueError("repaired overlap revision identity changed")
    return {
        "path": cfg.manifest_json,
        "sha256": manifest_hash,
        "output_sha256": source_hash,
        "monthly_rows_added": int(manifest["monthly_rows_added"]),
        "monthly_overlap_diagnostics": overlap,
    }


def build_report(cfg: Config) -> dict[str, Any]:
    if cfg != Config():
        raise ValueError("v2 support paths are frozen")
    fixed_logic = verify_fixed_logic()
    manifest_seal = verify_source_seal(cfg)
    source = load_source(cfg.input_csv)
    state = build_signal_state(source)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        active, side = candidate_clock(source, state, candidate)
        event_clock = [
            {
                "signal_bar_open": str(source.iloc[position]["signal_bar_open_utc"]),
                "side": int(side[position]),
                "symbol": str(source.iloc[position][f"{candidate.traded_leg}_symbol"]),
            }
            for position in np.flatnonzero(active)
        ]
        support, schedule_hashes = windowed_support_summary(
            source, active, side, candidate
        )
        gates = support_gates(support)
        candidates.append(
            {
                "candidate": asdict(candidate),
                "raw_events": int(active.sum()),
                "clock_hash": canonical_hash(event_clock),
                "schedule_hashes": schedule_hashes,
                "support": support,
                "gates": gates,
                "passes_support": bool(all(gates.values())),
            }
        )
    stable = {
        "protocol": {
            "outcomes_opened": False,
            "candidate_return_statistics_opened_for_v2": False,
            "v1_evaluator_aborted_before_result": True,
            "v1_selection_artifact_created": False,
            "source_end_exclusive": "2024-01-01 00:00:00",
            "normalization": (
                "strictly-prior 7d rolling median/IQR, min 80%, reset by front/next pair"
            ),
            "volume_semantics": (
                "raw COIN-M contract counts; signed pressure is taker imbalance "
                "times sqrt(contracts)"
            ),
            "signal_clock": "completed five-minute bar; entry no earlier than next bar open",
            "liquidity_floor": (
                "current combined contracts >= strictly-prior pair-local 25th percentile"
            ),
            "delivery_rule": (
                f"front DTE <={MAX_FRONT_DTE_HOURS / 24:.0f}d and both legs remain "
                f">={DELIVERY_BUFFER_HOURS:.0f}h from delivery after fixed exit"
            ),
            "candidate_count": len(CANDIDATES),
            "direction_repair_allowed": False,
            "threshold_repair_after_support_allowed": False,
            "threshold_repair_after_returns_allowed": False,
            "2024_plus_opened": False,
        },
        "thresholds": {
            "robust_window_bars": ROBUST_WINDOW_BARS,
            "robust_min_periods": ROBUST_MIN_PERIODS,
            "next_led": {
                "hold_minutes": 60,
                "next_share_min": MIN_NEXT_SHARE,
                "z_next_share_min": 1.0,
                "z_abs_next_pressure_min": 2.0,
                "directional_next_bar_return_min": 0.0005,
                "directional_front_bar_return_min": -0.0001,
            },
            "front_rejection": {
                "hold_minutes": 30,
                "front_share_min": MIN_FRONT_SHARE,
                "z_front_share_min": 0.75,
                "z_abs_front_pressure_min": 1.25,
                "z_abs_next_pressure_max": 0.75,
                "directional_front_bar_return_max": -0.0002,
                "directional_next_bar_return_max": 0.0,
            },
        },
        "source": {"path": cfg.input_csv, "sha256": sha256_file(cfg.input_csv)},
        "source_manifest": manifest_seal,
        "fixed_v1_logic": fixed_logic,
        "implementation": {
            "path": implementation_path(),
            "sha256": sha256_file(__file__),
        },
        "candidates": candidates,
    }
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stable,
        "support_freeze_hash": canonical_hash(stable),
    }


def main() -> None:
    cfg = Config()
    report = build_report(cfg)
    write_exclusive(cfg.output, report)
    print(
        json.dumps(
            {
                "candidates": len(report["candidates"]),
                "support_passes": sum(
                    item["passes_support"] for item in report["candidates"]
                ),
                "raw_events": {
                    item["candidate"]["name"]: item["raw_events"]
                    for item in report["candidates"]
                },
                "support_freeze_hash": report["support_freeze_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
