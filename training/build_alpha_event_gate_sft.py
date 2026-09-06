"""Build temporal event-level TAKE/SKIP data from frozen Gross9-passed clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import evaluate_high_volatility_spot_adverse_underwater_duration_relay_economics as econ
from training.build_alpha_formula_gate_sft import _compact, _formula, _read


STAGES = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}


def _latest(results: Path, slug: str, suffix: str) -> Path | None:
    rows = sorted(results.glob(f"{slug}_{suffix}_*.json"))
    return rows[-1] if rows else None


def _clock_path(support: dict[str, Any]) -> Path | None:
    clock = support.get("clock", {})
    raw = clock.get("path") if isinstance(clock, dict) else None
    return Path(str(raw)) if raw else None


def discover(results: Path) -> list[dict[str, Any]]:
    candidates = []
    for prereg_path in sorted(results.glob("high_volatility_*_preregistration_*.json")):
        slug = prereg_path.name.split("_preregistration_")[0]
        support_path = _latest(results, slug, "support")
        gross_path = _latest(results, slug, "gross9_novelty")
        if support_path is None or gross_path is None:
            continue
        try:
            prereg, support, gross = _read(prereg_path), _read(support_path), _read(gross_path)
        except Exception:
            continue
        if support.get("support_passed") is not True or gross.get("advance_to_economic_outcomes") is not True:
            continue
        clock = _clock_path(support)
        if clock is None or not clock.is_file():
            continue
        train_path = _latest(results, slug, "train_economics")
        train_pass = False
        if train_path is not None:
            train_obj = _read(train_path)
            train_pass = train_obj.get("passed", train_obj.get("advance_to_next_stage")) is True
        candidates.append(
            {
                "slug": slug,
                "policy_id": str(prereg.get("policy_id") or slug),
                "formula": _formula(prereg),
                "clock": clock,
                "research_train_pass": train_pass,
                "preregistration": prereg_path,
                "support": support_path,
                "gross9": gross_path,
            }
        )
    return candidates


def _event_context(row: pd.Series) -> dict[str, Any]:
    excluded = {"candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side"}
    values: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        low = str(key).lower()
        if low.startswith("future_") or any(token in low for token in ("postentry", "net_return", "funding_cash", "pnl")):
            continue
        if isinstance(value, (np.integer, int)):
            values[key] = int(value)
        elif isinstance(value, (np.floating, float)) and np.isfinite(value):
            values[key] = round(float(value), 8)
        elif isinstance(value, str) and len(value) <= 120:
            values[key] = value
    return values


def _prompt(candidate: dict[str, Any], event: pd.Series) -> str:
    return "\n".join(
        [
            "You are the event-level RLLM gate for a frozen BTC alpha.",
            "The alpha side and hold are immutable. Use only formula and signal-time event fields.",
            "Return exactly one token: TRADE when expected edge after 6bp-per-side cost and funding is positive; otherwise NO_TRADE.",
            f"policy_id: {candidate['policy_id']}",
            f"entry_time: {event['entry_time']}",
            f"exit_time: {event['exit_time']}",
            f"frozen_side: {int(event['side'])}",
            f"frozen_formula: {_compact(candidate['formula'], 4200)}",
            f"signal_time_event: {_compact(_event_context(event), 1800)}",
        ]
    )


def _load_clock(path: Path, stage: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "control" in frame.columns:
        frame = frame.loc[frame["control"].eq("primary")].copy()
    for key in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        if key in frame:
            frame[key] = pd.to_datetime(frame[key], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    if "split" in frame:
        frame = frame.loc[frame["split"].eq(stage)]
    return frame.loc[frame["entry_time"].ge(start) & frame["exit_time"].le(end)].sort_values("entry_time").reset_index(drop=True)


def _validate_funding_for_labels(funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    required = ["date", "funding_rate", "mark_price"]
    if list(funding.columns) != required or funding.empty:
        raise RuntimeError("funding label source schema/emptiness drift")
    if funding["date"].duplicated().any() or not funding["date"].is_monotonic_increasing:
        raise RuntimeError("funding label clock order drift")
    values = funding[["funding_rate", "mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or funding["mark_price"].le(0).any():
        raise RuntimeError("funding label values invalid")
    # Binance actual settlement timestamps can carry millisecond jitter.  The
    # frozen simulator posts them to floor(5m), so boundary completeness must
    # be checked on that same bucket without altering the raw event timestamp.
    if funding["date"].iloc[0].floor("5min") != start or funding["date"].iloc[-1].floor("5min") < end - pd.Timedelta(hours=8):
        raise RuntimeError("funding label boundary incomplete")
    if len(funding) > 1 and funding["date"].diff().iloc[1:].max() > pd.Timedelta(hours=8, minutes=1):
        raise RuntimeError("funding label gap exceeds tolerance")


def build_stage(candidates: list[dict[str, Any]], stage: str, output: Path) -> dict[str, Any]:
    start, end = STAGES[stage]
    market, funding, source = econ.load_sources(stage, start, end)
    econ.engine.validate_market(market, start, end)
    _validate_funding_for_labels(funding, start, end)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for candidate in candidates:
        try:
            clock = _load_clock(candidate["clock"], stage, start, end)
            if clock.empty:
                skipped["empty_clock"] += 1
                continue
            result = econ.engine.simulate(clock[["entry_time", "exit_time", "side"]], market, funding, start, end, 0.0006)
            by_key = {(r["entry_time"], r["exit_time"], int(r["side"])): r for r in result["trade_rows"]}
            for _, event in clock.iterrows():
                trade = by_key[(event["entry_time"], event["exit_time"], int(event["side"]))]
                net = float(trade["net_factor"] - 1.0)
                rows.append(
                    {
                        "task": "alpha_event_gate",
                        "prompt": _prompt(candidate, event),
                        "target": "TRADE" if net > 0.0 else "NO_TRADE",
                        "policy_id": candidate["policy_id"],
                        "slug": candidate["slug"],
                        "stage": stage,
                        "entry_time": str(event["entry_time"]),
                        "exit_time": str(event["exit_time"]),
                        "side": int(event["side"]),
                        "research_train_pass": bool(candidate["research_train_pass"]),
                        "metadata": {
                            "clock": str(candidate["clock"]),
                            "gross_return": float(trade["gross_return"]),
                            "net_return": net,
                            "funding_cash_over_pre_equity": float(trade["funding_cash_over_pre_equity"]),
                            "leakage_guard": "future event return/funding appear only in metadata and target, never prompt",
                        },
                    }
                )
        except Exception:
            skipped["candidate_error"] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    return {
        "stage": stage,
        "rows": len(rows),
        "targets": dict(Counter(row["target"] for row in rows)),
        "research_train_pass_rows": sum(row["research_train_pass"] for row in rows),
        "policies": len({row["policy_id"] for row in rows}),
        "skipped": dict(skipped),
        "source": source,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def build(results: Path, output_dir: Path, summary_output: Path) -> dict[str, Any]:
    candidates = discover(results)
    reports = {}
    for stage in STAGES:
        reports[stage] = build_stage(candidates, stage, output_dir / f"rllm_alpha_event_gate_{stage}_2026-08-19.jsonl")
    report = {"candidates": len(candidates), "stages": reports, "leakage_guard": {"prompts_contain_future_event_metadata": False, "train_stage_only_for_sft": True, "oos_stages_never_used_for_training": True}}
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--summary-output", type=Path, required=True)
    print(json.dumps(build(**vars(p.parse_args())), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
