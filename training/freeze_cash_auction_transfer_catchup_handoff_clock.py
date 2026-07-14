"""Reproduce and freeze the selected CATCH-12 support clock without outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import preregister_cash_auction_transfer_catchup_handoff as catch


EXPECTED_SUPPORT_SHA256 = (
    "454c54cb234a34b51fca12a810332039d7b21e4395f47f4e6b8ad6375370be02"
)
PREREGISTRATION_COMMIT = "5d1270bf0d3c453ecfc3a5e02a193db8db59f1e5"
SUPPORT_COMMIT = "49bf3f7b17c39b9c2abc56ef64e1a6fa0bad4279"
RETURN_TOKENS = {"return", "future", "forward", "pnl", "profit", "cagr", "mdd"}


@dataclass(frozen=True)
class FreezeConfig:
    support: str = "results/cash_auction_transfer_catchup_handoff_support_2026-07-14.json"
    clock: str = "results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv"
    manifest: str = (
        "results/cash_auction_transfer_catchup_handoff_clock_manifest_2026-07-14.json"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_return_column(columns: list[str] | tuple[str, ...]) -> bool:
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
    cfg: catch.Config,
) -> dict[str, Any]:
    selected = support_result.get("selected_support")
    if support_result.get("support_decision") != "pass" or not isinstance(
        selected, dict
    ):
        raise ValueError("CATCH support did not select a passing clock")
    if support_result.get("selected_quantile") != selected.get("quantile"):
        raise ValueError("selected CATCH quantile is inconsistent")
    if raw_primary != selected.get("raw_primary"):
        raise ValueError("reproduced raw CATCH count differs from support freeze")
    calculated = catch._support(schedule, cfg)
    if calculated != selected.get("support"):
        raise ValueError("reproduced CATCH schedule differs from support freeze")
    if tuple(schedule.columns) != catch.SCHEDULE_COLUMNS:
        raise ValueError("CATCH clock columns differ from frozen schema")
    if _has_return_column(tuple(schedule.columns)):
        raise ValueError("CATCH support clock unexpectedly contains outcome columns")
    if not schedule.empty:
        signal_position = schedule["signal_position"].to_numpy(int)
        entry_position = schedule["entry_position"].to_numpy(int)
        exit_position = schedule["exit_position"].to_numpy(int)
        if not (entry_position == signal_position + 1).all():
            raise ValueError("CATCH entry is not the next five-minute open")
        if not (exit_position == entry_position + cfg.hold_bars).all():
            raise ValueError("CATCH exit does not match the fixed hold")
        if not (entry_position[1:] >= exit_position[:-1]).all():
            raise ValueError("CATCH clock contains overlapping holds")
        if not schedule["side"].isin((-1, 1)).all():
            raise ValueError("CATCH clock contains a non-directional action")
        if pd.to_datetime(schedule["exit_date"]).max() >= catch.SELECTION_END:
            raise ValueError("CATCH support clock opens the sealed interval")
    return calculated


def run_freeze(freeze_cfg: FreezeConfig) -> dict[str, Any]:
    support_path = Path(freeze_cfg.support)
    if _sha256(support_path) != EXPECTED_SUPPORT_SHA256:
        raise ValueError("CATCH support artifact differs from its frozen SHA-256")
    support_result = json.loads(support_path.read_text())
    if support_result.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("CATCH support artifact is not outcome-blind")
    cfg = catch.Config()
    if support_result.get("config") != asdict(cfg):
        raise ValueError("CATCH support config differs from preregistration defaults")
    quantile = support_result.get("selected_quantile")
    if quantile not in catch.SUPPORT_QUANTILES:
        raise ValueError("CATCH selected quantile is outside the frozen grid")

    frame, source = catch.load_causal_frame(cfg)
    signal, controls, _, _ = catch.classify_events(frame, cfg, quantile=quantile)
    schedule = catch.nonoverlapping_schedule(signal, frame)
    calculated = _validate_schedule(
        schedule,
        support_result,
        raw_primary=int(controls["primary"].sum()),
        cfg=cfg,
    )
    clock_path = Path(freeze_cfg.clock)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(clock_path, index=False, lineterminator="\n")
    clock_sha = _sha256(clock_path)
    selected = support_result["selected_support"]
    manifest = {
        "protocol": {
            "name": "CATCH-12 — Cash Auction Transfer & Catch-up Handoff",
            "stage": "support_clock_freeze",
            "outcomes_opened": False,
            "return_columns_present": False,
            "selected_quantile": quantile,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "selection_end_exclusive": str(catch.SELECTION_END),
        },
        "clock": {
            "path": str(clock_path),
            "sha256": clock_sha,
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
