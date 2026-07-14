"""Reproduce and freeze the selected UMFR-36 support clock without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import preregister_um_forced_flow_reversion as umfr

EXPECTED_SUPPORT_SHA256 = (
    "22a6cabe015020fa427a660d611917590a95d1212426be1d391476ddce77d3ba"
)
PREREGISTRATION_COMMIT = "035e2ee"
SUPPORT_COMMIT = "43322e4"
RETURN_TOKENS = {
    "return",
    "future",
    "forward",
    "pnl",
    "profit",
    "cagr",
    "mdd",
    "funding",
    "high",
    "low",
    "open",
    "close",
}


@dataclass(frozen=True)
class FreezeConfig:
    support: str = "results/um_forced_flow_reversion_support_2026-07-14.json"
    clock: str = "results/um_forced_flow_reversion_clock_2026-07-14.csv"
    manifest: str = "results/um_forced_flow_reversion_clock_manifest_2026-07-14.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_outcome_column(columns: list[str] | tuple[str, ...]) -> bool:
    return any(
        token in column.lower().split("_")
        for column in columns
        for token in RETURN_TOKENS
    )


def _validate_schedule(
    schedule: pd.DataFrame,
    support_result: dict[str, Any],
    *,
    raw_primary: int,
    cfg: umfr.Config,
) -> dict[str, Any]:
    selected = support_result.get("selected_support")
    if support_result.get("support_decision") != "pass" or not isinstance(
        selected, dict
    ):
        raise ValueError("UMFR support did not select a passing clock")
    selected_quantile = support_result.get("selected_quantile")
    if selected_quantile != selected.get("quantile"):
        raise ValueError("selected UMFR quantile is inconsistent")
    if selected_quantile not in umfr.SUPPORT_QUANTILES:
        raise ValueError("selected UMFR quantile is outside the frozen grid")
    if raw_primary != selected.get("raw_primary"):
        raise ValueError("reproduced raw UMFR count differs from support freeze")
    if tuple(schedule.columns) != umfr.SCHEDULE_COLUMNS:
        raise ValueError("UMFR clock columns differ from frozen schema")
    if _has_outcome_column(tuple(schedule.columns)):
        raise ValueError("UMFR support clock unexpectedly contains outcome columns")
    if not schedule.empty:
        signal_position = schedule["signal_position"].to_numpy(int)
        entry_position = schedule["entry_position"].to_numpy(int)
        exit_position = schedule["exit_position"].to_numpy(int)
        if not (entry_position == signal_position + 1).all():
            raise ValueError("UMFR entry is not the next five-minute open")
        if not (exit_position == entry_position + cfg.hold_bars).all():
            raise ValueError("UMFR exit does not match the fixed hold")
        if not (entry_position[1:] >= exit_position[:-1]).all():
            raise ValueError("UMFR clock contains overlapping holds")
        if not schedule["side"].isin((-1, 1)).all():
            raise ValueError("UMFR clock contains a non-directional action")
        if not schedule["branch"].eq("umfr36").all():
            raise ValueError("UMFR clock contains a non-primary branch")
        if not schedule["hold_bars"].eq(cfg.hold_bars).all():
            raise ValueError("UMFR clock contains a mutable hold")
        if pd.to_datetime(schedule["exit_date"]).max() >= umfr.SELECTION_END:
            raise ValueError("UMFR support clock opens the sealed interval")
    calculated = umfr._support(schedule, cfg)
    if calculated != selected.get("support"):
        raise ValueError("reproduced UMFR schedule differs from support freeze")
    return calculated


def run_freeze(freeze_cfg: FreezeConfig) -> dict[str, Any]:
    support_path = Path(freeze_cfg.support)
    if _sha256(support_path) != EXPECTED_SUPPORT_SHA256:
        raise ValueError("UMFR support artifact differs from its frozen SHA-256")
    support_result = json.loads(support_path.read_text())
    protocol = support_result.get("protocol", {})
    if protocol.get("support_only") is not True:
        raise ValueError("UMFR support artifact is not support-only")
    if protocol.get("umfr_outcomes_opened") is not False:
        raise ValueError("UMFR support artifact is not outcome-blind")
    cfg = umfr.Config()
    if support_result.get("config") != asdict(cfg):
        raise ValueError("UMFR support config differs from preregistration defaults")
    quantile = support_result.get("selected_quantile")
    if quantile not in umfr.SUPPORT_QUANTILES:
        raise ValueError("UMFR selected quantile is outside the frozen grid")

    frame, source = umfr.load_causal_frame()
    signal, controls, _, _ = umfr.classify_events(frame, cfg, quantile=quantile)
    schedule = umfr.nonoverlapping_schedule(signal, frame)
    calculated = _validate_schedule(
        schedule, support_result, raw_primary=int(controls["primary"].sum()), cfg=cfg
    )
    clock_path = Path(freeze_cfg.clock)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(clock_path, index=False, lineterminator="\n")
    selected = support_result["selected_support"]
    manifest = {
        "protocol": {
            "name": "UMFR-36 — USD-M Forced Flow Reversion",
            "stage": "support_clock_freeze",
            "outcomes_opened": False,
            "outcome_columns_present": False,
            "selected_quantile": quantile,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "selection_end_exclusive": str(umfr.SELECTION_END),
        },
        "clock": {
            "path": str(clock_path),
            "sha256": _sha256(clock_path),
            "rows": int(len(schedule)),
            "first_signal_date": str(schedule["signal_date"].iloc[0]),
            "last_signal_date": str(schedule["signal_date"].iloc[-1]),
            "long_events": int(schedule["side"].gt(0).sum()),
            "short_events": int(schedule["side"].lt(0).sum()),
            "columns": list(schedule.columns),
        },
        "support": {
            "path": str(support_path),
            "sha256": EXPECTED_SUPPORT_SHA256,
            "raw_primary": selected["raw_primary"],
            **calculated,
        },
        "source": source,
    }
    manifest_path = Path(freeze_cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", default=FreezeConfig.support)
    parser.add_argument("--clock", default=FreezeConfig.clock)
    parser.add_argument("--manifest", default=FreezeConfig.manifest)
    manifest = run_freeze(FreezeConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "outcomes_opened": manifest["protocol"]["outcomes_opened"],
                "selected_quantile": manifest["protocol"]["selected_quantile"],
                **manifest["clock"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
