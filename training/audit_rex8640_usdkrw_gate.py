"""Replay the pre-2025 REX8640-width / USDKRW gate with corrected strict MDD."""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars


WIDTH_THRESHOLD = 0.2836633876944003
USDKRW_THRESHOLD = 0.2603593471820541
DEFAULT_GATES_JSON = json.dumps(
    [
        {"feature": "rex_8640_range_width_pct", "op": ">=", "threshold": WIDTH_THRESHOLD},
        {"feature": "usdkrw_zscore", "op": "<=", "threshold": USDKRW_THRESHOLD},
    ],
    separators=(",", ":"),
)
WINDOWS = {
    "train2021_2023": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "selection2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "eval2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "holdout2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-02")),
    "eval2025_2026": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-02")),
}


@dataclass(frozen=True)
class AuditConfig:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    gates_json: str = DEFAULT_GATES_JSON
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    hold_bars: int = 144


def load_rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_gates(raw: str) -> list[dict[str, Any]]:
    gates = json.loads(raw)
    if not isinstance(gates, list) or not gates:
        raise ValueError("gates_json must be a non-empty JSON list")
    return [
        {"feature": str(gate["feature"]), "op": str(gate["op"]), "threshold": float(gate["threshold"])}
        for gate in gates
    ]


def gate_match(row: dict[str, Any], gates: list[dict[str, Any]] | None = None) -> bool:
    snapshot = row.get("feature_snapshot") or {}
    gates = gates or parse_gates(DEFAULT_GATES_JSON)
    for gate in gates:
        try:
            value = float(snapshot[gate["feature"]])
        except (KeyError, TypeError, ValueError):
            return False
        if gate["op"] == ">=" and value < gate["threshold"]:
            return False
        if gate["op"] == "<=" and value > gate["threshold"]:
            return False
        if gate["op"] not in {">=", "<="}:
            raise ValueError(f"unsupported gate op: {gate['op']}")
    return True


def prediction_rows(
    rows: list[dict[str, Any]],
    market: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    side_mode: str,
    hold_bars: int,
    gates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    predictions: list[dict[str, Any]] = []
    for row in rows:
        date = pd.Timestamp(str(row["date"]))
        side = str((row.get("action") or {}).get("side", "")).upper()
        signal_pos = int(row["signal_pos"])
        exit_pos = signal_pos + 1 + int(hold_bars)
        if not (start <= date < end) or not gate_match(row, gates):
            continue
        if side_mode != "both" and side != side_mode.upper():
            continue
        if side not in {"LONG", "SHORT"} or exit_pos >= len(market) or pd.Timestamp(dates.iloc[exit_pos]) >= end:
            continue
        predictions.append(
            {
                "date": str(row["date"]),
                "signal_pos": signal_pos,
                "prediction": {"gate": "TRADE", "side": side, "hold_bars": int(hold_bars)},
            }
        )
    return predictions


def score(
    rows: list[dict[str, Any]],
    market: pd.DataFrame,
    cfg: AuditConfig,
    window: tuple[pd.Timestamp, pd.Timestamp],
    side_mode: str,
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    start, end = window
    selected = prediction_rows(
        rows,
        market,
        start=start,
        end=end,
        side_mode=side_mode,
        hold_bars=cfg.hold_bars,
        gates=gates,
    )
    if not selected:
        return {"return_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "ratio": 0.0, "trades": 0, "p_value": 1.0}
    report = strict_backtest_actions(
        selected,
        market,
        EconomicActionBacktestConfig(
            annualization_start=str(start),
            annualization_end=str(end),
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            entry_delay_bars=1,
            max_hold_bars=cfg.hold_bars,
        ),
    )
    sim = report["sim"]
    stats = report["trade_stats"]
    return {
        "return_pct": float(sim["ret_pct"]),
        "cagr_pct": float(sim["cagr_pct"]),
        "strict_mdd_pct": float(sim["strict_mdd_pct"]),
        "ratio": float(sim["cagr_to_strict_mdd"]),
        "trades": int(sim["trade_entries"]),
        "p_value": float(stats["p_value_mean_ret_approx"]),
    }


def run(cfg: AuditConfig) -> dict[str, Any]:
    market = load_market_bars(cfg.market_csv)
    gates = parse_gates(cfg.gates_json)
    sets = {
        "train2021_2023": load_rows(cfg.train_jsonl),
        "selection2024": load_rows(cfg.test_jsonl),
    }
    eval_rows = load_rows(cfg.eval_jsonl)
    sets.update({"eval2025": eval_rows, "holdout2026": eval_rows, "eval2025_2026": eval_rows})
    results = {
        side: {name: score(sets[name], market, cfg, window, side, gates) for name, window in WINDOWS.items()}
        for side in ("both", "long", "short")
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "fixed_gates": gates,
        "protocol": {
            "provenance": "gate selected on 2021-2023 train plus 2024 selection; 2025-2026 replay only",
            "selection2024_is_not_oos_evidence": True,
            "eval2025_and_holdout2026_not_used_by_this_replay": True,
            "cagr": "full configured calendar window including idle time",
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
        },
        "results": results,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in AuditConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    ns.leverage = float(ns.leverage)
    ns.fee_rate = float(ns.fee_rate)
    ns.slippage_rate = float(ns.slippage_rate)
    ns.hold_bars = int(ns.hold_bars)
    return AuditConfig(**vars(ns))


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
