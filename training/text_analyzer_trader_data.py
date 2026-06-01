"""Build text-only analyzer/trader SFT data from executable path labels.

This module separates two LLM jobs:

1. analyzer: turn past-only market context into a compact symbolic market
   summary. Its target text is deterministic and uses no future labels.
2. trader: consume only the analyzer summary and emit TRADE/NO_TRADE plus
   LONG/SHORT when executable path labels say a trade is worth taking.

The intent is to exploit LLM strengths (symbolic compression, regime language,
reasoning over discrete evidence) instead of forcing the model to regress raw
numbers directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from preprocessing.timeframe import make_window
from training.path_outcome_dataset import PathOutcomeConfig, make_path_outcome_record, summarize_records
from training.vlm_trading_data import (
    _candle_pattern_label,
    _engineered_prompt_features,
    _location_label,
    _oscillator_label,
    _risk_state_label,
    _symbolic_market_labels,
    _trend_alignment_label,
    _volume_state_label,
)

AnalyzerTask = Literal["analyzer"]
TraderTask = Literal["trader"]


@dataclass(frozen=True)
class TextPipelineConfig:
    window_size: int = 96
    hold_bars: int = 144
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 1.0
    mae_penalty: float = 1.0
    mfe_bonus: float = 0.0
    min_net_return: float = 0.001
    min_utility: float = 0.0
    max_mae: float = 0.015
    stride_bars: int = 12


def load_market_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"market csv lacks required columns: {sorted(missing)}")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def _pct(value: float) -> str:
    return f"{100.0 * float(value):+.2f}%"


def _feature_line(name: str, value: str) -> str:
    return f"- {name}: {value}"


def build_analyzer_input(market: pd.DataFrame, signal_pos: int, *, window_size: int) -> str:
    """Prompt for analyzer fine-tuning; uses only bars <= signal_pos."""
    window = make_window(market, t=int(signal_pos), w=int(window_size))
    last = window.iloc[-1]
    first = window.iloc[0]
    ret_window = 0.0 if float(first["close"]) == 0.0 else float(last["close"] / first["close"] - 1.0)
    ret_12 = 0.0
    if len(window) > 12 and float(window.iloc[-13]["close"]) != 0.0:
        ret_12 = float(last["close"] / window.iloc[-13]["close"] - 1.0)
    high = float(window["high"].max())
    low = float(window["low"].min())
    close = float(last["close"])
    range_pos = 0.5 if high <= low else (close - low) / (high - low)
    lines = [
        "You are the analyzer stage for a BTCUSDT futures trading bot.",
        "Use only this past market window. Convert it into a compact symbolic summary.",
        "Do not recommend a trade; only describe regime, risk, location, and evidence.",
        "",
        f"Window bars: {len(window)} ending at {pd.to_datetime(last['date'])}",
        _feature_line("window return", _pct(ret_window)),
        _feature_line("recent 12-bar return", _pct(ret_12)),
        _feature_line("window high-low range", _pct((high - low) / close if close else 0.0)),
        _feature_line("close location in window", f"{range_pos:.2f}"),
        _feature_line("last candle body", _pct((float(last['close']) - float(last['open'])) / float(last['close']) if float(last['close']) else 0.0)),
        _feature_line("last candle range", _pct((float(last['high']) - float(last['low'])) / float(last['close']) if float(last['close']) else 0.0)),
        "",
        "Return JSON with keys: regime, trend_alignment, location, oscillator, volume_state, candle_pattern, risk_state, evidence.",
    ]
    return "\n".join(lines)


def build_analyzer_summary(
    market: pd.DataFrame,
    signal_pos: int,
    *,
    window_size: int,
    feature_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Deterministic past-only teacher summary for analyzer SFT targets."""
    features = feature_frame if feature_frame is not None else build_market_feature_frame(market, window_size=window_size)
    feature_row = features.iloc[int(signal_pos)]
    window = make_window(market, t=int(signal_pos), w=int(window_size))
    regime, vol_level, momentum, trend_strength, window_vol = _symbolic_market_labels(window)
    numeric, symbolic, tags = _engineered_prompt_features(window, feature_row)
    numeric_map = {k: float(v) for k, v in numeric}
    symbolic_map = {k: str(v) for k, v in symbolic}
    summary = {
        "regime": regime,
        "volatility_level": vol_level,
        "momentum": momentum,
        "trend_strength": trend_strength,
        "trend_alignment": _trend_alignment_label(feature_row),
        "location": _location_label(feature_row),
        "oscillator": _oscillator_label(feature_row),
        "volume_state": _volume_state_label(feature_row),
        "candle_pattern": _candle_pattern_label(feature_row),
        "risk_state": _risk_state_label(feature_row),
        "context_tags": list(tags),
        "evidence": {
            "window_volatility_pct": round(float(window_vol) * 100.0, 3),
            "momentum_1h_pct": round(float(feature_row.get("trend_12", 0.0)) * 100.0, 3),
            "momentum_2h_pct": round(float(feature_row.get("trend_24", 0.0)) * 100.0, 3),
            "momentum_8h_pct": round(float(feature_row.get("trend_96", 0.0)) * 100.0, 3),
            "range_position": round(float(feature_row.get("range_pos", 0.0)), 3),
            "window_drawdown_pct": round(float(feature_row.get("window_drawdown", 0.0)) * 100.0, 3),
            "volume_zscore": round(float(feature_row.get("volume_zscore", 0.0)), 3),
        },
        "symbolic_features": symbolic_map,
        "numeric_feature_names": sorted(numeric_map),
    }
    return summary


def analyzer_summary_to_text(summary: dict[str, Any]) -> str:
    """Stable compact JSON string for feeding the trader stage."""
    return json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_trader_input(analyzer_summary_text: str, *, hold_bars: int, entry_delay_bars: int) -> str:
    return "\n".join(
        [
            "You are the trader stage for a BTCUSDT futures trading bot.",
            "You receive only the analyzer's past-only symbolic market summary.",
            "Decide whether an executable trade is justified after fees/slippage and adverse excursion risk.",
            f"Execution: enter after {int(entry_delay_bars)} bar(s), hold {int(hold_bars)} bar(s), then flat.",
            "Output exactly one JSON object with keys gate and side.",
            "gate must be TRADE or NO_TRADE. side must be LONG, SHORT, or NONE.",
            "",
            f"Analyzer summary: {analyzer_summary_text}",
        ]
    )


def build_trader_target(path_record: dict[str, Any]) -> dict[str, str]:
    gate = str(path_record["trade_gate_label"])
    side = str(path_record["trade_side_label"]) if gate == "TRADE" else "NONE"
    return {"gate": gate, "side": side}


def _iter_signal_positions(market: pd.DataFrame, cfg: TextPipelineConfig, start_date: str | None, end_date: str | None):
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    path_cfg = PathOutcomeConfig(
        hold_bars=cfg.hold_bars,
        entry_delay_bars=cfg.entry_delay_bars,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        leverage=cfg.leverage,
        mae_penalty=cfg.mae_penalty,
        mfe_bonus=cfg.mfe_bonus,
        hold_margin=cfg.min_utility,
        min_net_return=cfg.min_net_return,
        max_mae=cfg.max_mae,
    )
    last_signal_pos = len(market) - max(1, int(cfg.entry_delay_bars)) - max(1, int(cfg.hold_bars)) - 1
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_signal_pos) + 1, max(1, int(cfg.stride_bars))):
        ts = pd.to_datetime(market.iloc[pos]["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        path_record = make_path_outcome_record(market, pos, path_cfg)
        if path_record is None:
            continue
        yield pos, path_record


def build_text_pipeline_records(
    market: pd.DataFrame,
    cfg: TextPipelineConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analyzer_records: list[dict[str, Any]] = []
    trader_records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    feature_frame = build_market_feature_frame(market, window_size=cfg.window_size)
    for pos, path_record in _iter_signal_positions(market, cfg, start_date, end_date):
        summary = build_analyzer_summary(market, pos, window_size=cfg.window_size, feature_frame=feature_frame)
        summary_text = analyzer_summary_to_text(summary)
        analyzer_records.append(
            {
                "task": "analyzer",
                "date": path_record["date"],
                "signal_pos": int(pos),
                "prompt": build_analyzer_input(market, pos, window_size=cfg.window_size),
                "target": summary_text,
                "leakage_guard": {"target_uses_future_path": False, "input_bars_end_at_signal_pos": int(pos)},
            }
        )
        trader_records.append(
            {
                "task": "trader",
                "date": path_record["date"],
                "signal_pos": int(pos),
                "prompt": build_trader_input(summary_text, hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars),
                "target": json.dumps(build_trader_target(path_record), sort_keys=True, separators=(",", ":")),
                "trade_gate_label": path_record["trade_gate_label"],
                "trade_side_label": path_record["trade_side_label"],
                "best_net_return": float(path_record["best_net_return"]),
                "best_mae": float(path_record["best_mae"]),
                "best_utility": float(path_record["best_utility"]),
                "leakage_guard": {"prompt_uses_analyzer_summary_only": True, "label_uses_future_path": True},
            }
        )
        path_records.append(path_record)
        if max_records is not None and len(trader_records) >= int(max_records):
            break
    return analyzer_records, trader_records, path_records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build text-only analyzer/trader SFT data")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--analyzer-output", default="data/text_analyzer_sft.jsonl")
    p.add_argument("--trader-output", default="data/text_trader_sft.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--window-size", type=int, default=96)
    p.add_argument("--hold-bars", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--min-net-return", type=float, default=0.001)
    p.add_argument("--min-utility", type=float, default=0.0)
    p.add_argument("--max-mae", type=float, default=0.015)
    p.add_argument("--stride-bars", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TextPipelineConfig(
        window_size=args.window_size,
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
        mae_penalty=args.mae_penalty,
        mfe_bonus=args.mfe_bonus,
        min_net_return=args.min_net_return,
        min_utility=args.min_utility,
        max_mae=args.max_mae,
        stride_bars=args.stride_bars,
    )
    market = load_market_frame(args.market_csv)
    analyzer_rows, trader_rows, path_rows = build_text_pipeline_records(
        market,
        cfg,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        max_records=args.max_records or None,
    )
    write_jsonl(args.analyzer_output, analyzer_rows)
    write_jsonl(args.trader_output, trader_rows)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "market_csv": str(Path(args.market_csv).resolve()),
        "outputs": {"analyzer": args.analyzer_output, "trader": args.trader_output},
        "config": asdict(cfg),
        "records": {"analyzer": len(analyzer_rows), "trader": len(trader_rows)},
        "path_label_summary": summarize_records(path_rows),
        "leakage_guard": {
            "analyzer_target_uses_future_path": False,
            "trader_label_uses_future_path": True,
            "trader_prompt_uses_only_analyzer_summary": True,
            "prompt_bars_end_at_or_before_signal_pos": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "next_return_scalar_not_used": True,
        },
    }
    summary_output = args.summary_output or f"{args.trader_output}.summary.json"
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
