"""Post-hoc MFIC v1 gross/cost/branch failure decomposition.

This module cannot promote or retune MFIC.  It only decomposes already-opened
2020-2023 outcomes and keeps all 2024+ windows sealed.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    EvaluationConfig,
    WINDOWS,
    _sha256,
    _verify_preregistration,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    CANDIDATES,
    Config as SignalConfig,
    compute_mfic,
    load_causal_frame,
    nonoverlapping_schedule,
)


SELECTION_RESULT = Path(
    "results/metaorder_fragmentation_impact_curvature_selection_2026-07-14.json"
)
SELECTION_RESULT_SHA256 = (
    "2e33ac7e76c8212dcd0b3f919c6bb912647251cfcfb71264fe7702175af47de3"
)
OUTPUT = Path(
    "results/metaorder_fragmentation_impact_curvature_failure_diagnostic_2026-07-14.json"
)


def _verify_rejected_selection() -> dict[str, Any]:
    _verify_preregistration()
    if _sha256(SELECTION_RESULT) != SELECTION_RESULT_SHA256:
        raise ValueError("MFIC selection result changed after rejection was recorded")
    result = json.loads(SELECTION_RESULT.read_text())
    if result.get("selection", {}).get("rejected") is not True:
        raise ValueError("MFIC selection result is not a frozen rejection")
    if result.get("protocol", {}).get("sealed_windows") != [
        "test2024",
        "eval2025",
        "ytd2026",
    ]:
        raise ValueError("MFIC selection result did not preserve sealed OOS windows")
    return result


def build_trade_ledger(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    opens = frame["open"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for trade in schedule.itertuples(index=False):
        entry_position = int(trade.entry_position)
        exit_position = int(trade.exit_position)
        side = int(trade.side)
        underlying_raw = side * (
            opens[exit_position] / opens[entry_position] - 1.0
        )
        account_gross = cfg.leverage * underlying_raw
        account_net = (
            (1.0 - per_side_cost)
            * (1.0 + account_gross)
            * (1.0 - per_side_cost)
            - 1.0
        )
        inverted_account_net = (
            (1.0 - per_side_cost)
            * (1.0 - account_gross)
            * (1.0 - per_side_cost)
            - 1.0
        )
        rows.append(
            {
                "entry_date": str(frame["date"].iloc[entry_position]),
                "branch": str(trade.branch),
                "side": side,
                "underlying_raw_return": float(underlying_raw),
                "account_gross_return": float(account_gross),
                "account_net_return": float(account_net),
                "posthoc_inverted_account_net_return": float(inverted_account_net),
            }
        )
    return pd.DataFrame(rows)


def summarize_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "trade_count": 0,
            "mean_underlying_raw_bps": 0.0,
            "mean_account_gross_bps": 0.0,
            "mean_account_net_bps": 0.0,
            "mean_posthoc_inverted_account_net_bps": 0.0,
            "account_gross_win_rate": 0.0,
        }
    return {
        "trade_count": int(len(ledger)),
        "mean_underlying_raw_bps": float(
            ledger["underlying_raw_return"].mean() * 10_000.0
        ),
        "mean_account_gross_bps": float(
            ledger["account_gross_return"].mean() * 10_000.0
        ),
        "mean_account_net_bps": float(
            ledger["account_net_return"].mean() * 10_000.0
        ),
        "mean_posthoc_inverted_account_net_bps": float(
            ledger["posthoc_inverted_account_net_return"].mean() * 10_000.0
        ),
        "account_gross_win_rate": float(
            ledger["account_gross_return"].gt(0.0).mean()
        ),
    }


def run_diagnostic() -> dict[str, Any]:
    selection = _verify_rejected_selection()
    signal_cfg = SignalConfig()
    execution_cfg = EvaluationConfig()
    frame, source = load_causal_frame(signal_cfg)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        signal = compute_mfic(frame, candidate, signal_cfg)
        windows: dict[str, Any] = {}
        for window in ("train", "select2023"):
            start, end = WINDOWS[window]
            schedule = nonoverlapping_schedule(
                signal, frame, start=start, end=end
            )
            ledger = build_trade_ledger(frame, schedule, execution_cfg)
            groups: list[dict[str, Any]] = []
            for (branch, side), group in ledger.groupby(
                ["branch", "side"], sort=True, observed=True
            ):
                groups.append(
                    {
                        "branch": str(branch),
                        "side": int(side),
                        **summarize_ledger(group),
                    }
                )
            windows[window] = {
                "all": summarize_ledger(ledger),
                "by_branch_and_side": groups,
            }
        candidates.append({"candidate": asdict(candidate), "windows": windows})

    per_side_cost = (
        execution_cfg.fee_rate + execution_cfg.slippage_rate
    ) * execution_cfg.leverage
    flat_round_trip_cost_bps = (
        1.0 - (1.0 - per_side_cost) ** 2
    ) * 10_000.0
    return {
        "protocol": {
            "name": "MFIC v1 post-hoc failure decomposition",
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "outcomes_already_opened": [
                "train",
                "select2023",
                "select2023_h1",
                "select2023_h2",
            ],
            "sealed_windows_still_unopened": [
                "test2024",
                "eval2025",
                "ytd2026",
            ],
            "may_promote_or_retune_mfic": False,
            "inverted_direction_is_diagnostic_only": True,
        },
        "execution_config": asdict(execution_cfg),
        "flat_round_trip_account_cost_bps": float(flat_round_trip_cost_bps),
        "source": source,
        "selection_verdict": selection["selection"],
        "candidates": candidates,
    }


def main() -> None:
    result = run_diagnostic()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "flat_round_trip_account_cost_bps": result[
                    "flat_round_trip_account_cost_bps"
                ],
                "candidates": result["candidates"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
