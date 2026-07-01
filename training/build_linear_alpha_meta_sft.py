"""Build single-LLM meta-controller SFT rows from frozen alpha predictions.

Prompts contain only signal-time alpha/state context.  Targets use future trade
outcomes only as supervised labels, so these rows are for training/validation,
not for live inference.  The model's job is deliberately narrow: decide whether
to TAKE or SKIP a pre-existing weak alpha signal and choose a coarse size bucket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LinearAlphaMetaSftConfig:
    predictions_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    window_size: int = 144
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_hold_bars: int = 576
    take_full_ret_pct: float = 0.35
    take_small_ret_pct: float = 0.0
    include_no_trade_fraction: float = 0.03
    seed: int = 17


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_market(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _prediction(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction")
    return pred if isinstance(pred, dict) else {}


def _side_signal(side: str) -> int:
    if side == "LONG":
        return 1
    if side == "SHORT":
        return -1
    return 0


def _trade_return_pct(
    *,
    market: pd.DataFrame,
    signal_pos: int,
    side: str,
    hold_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
) -> float | None:
    signal = _side_signal(side)
    if signal == 0:
        return None
    entry_pos = int(signal_pos) + int(entry_delay_bars)
    exit_pos = entry_pos + max(1, int(hold_bars))
    if entry_pos >= len(market) - 1 or exit_pos >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    eq = 1.0
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    eq *= max(0.0, 1.0 - cost)
    for j in range(entry_pos, exit_pos):
        open_j = float(opens[j])
        if open_j <= 0.0:
            continue
        if signal > 0:
            close_ret = (float(opens[j + 1]) - open_j) / open_j
        else:
            close_ret = (open_j - float(opens[j + 1])) / open_j
        eq *= max(0.0, 1.0 + float(leverage) * close_ret)
        if eq <= 0.0:
            break
    eq *= max(0.0, 1.0 - cost)
    return (eq - 1.0) * 100.0


def _label(ret_pct: float | None, cfg: LinearAlphaMetaSftConfig) -> dict[str, str]:
    if ret_pct is None:
        return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "missing_future_bars"}
    if ret_pct >= float(cfg.take_full_ret_pct):
        return {"decision": "TAKE", "size_bucket": "FULL", "risk_reason": "realized_edge_strong"}
    if ret_pct > float(cfg.take_small_ret_pct):
        return {"decision": "TAKE", "size_bucket": "SMALL", "risk_reason": "realized_edge_positive_but_thin"}
    return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "realized_edge_negative_or_costly"}


def _bucket(value: float, cuts: tuple[float, float]) -> str:
    if value <= cuts[0]:
        return "low"
    if value >= cuts[1]:
        return "high"
    return "mid"


def _feature_value(features: pd.DataFrame, pos: int, key: str) -> float:
    if key not in features.columns or pos < 0 or pos >= len(features):
        return 0.0
    value = features[key].iloc[pos]
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if np.isfinite(out) else 0.0


def _state_card(row: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
    pos = int(row.get("signal_pos", -1) or -1)
    pred = _prediction(row)
    keys = [
        "trend_12", "trend_96", "range_pos", "range_vol", "window_drawdown",
        "rsi_norm", "bb_z", "return_zscore_48", "volume_zscore", "taker_buy_ratio",
        "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change",
        "usdkrw_zscore", "usdkrw_momentum", "rex_144_range_pos", "rex_2016_range_pos",
        "rex_8640_cur_to_max_pct", "rex_8640_cur_to_min_pct",
    ]
    numeric = {k: round(_feature_value(features, pos, k), 6) for k in keys if k in features.columns}
    tokens = {
        "alpha_group": str(row.get("group", "unknown")),
        "alpha_variant": str(row.get("variant", "unknown")),
        "alpha_side": str(pred.get("side", "NONE")),
        "alpha_hold_bars": int(pred.get("hold_bars", 0) or 0),
        "alpha_score": round(float(row.get("score", 0.0) or 0.0), 8),
        "range_pos_bucket": _bucket(numeric.get("range_pos", 0.0), (0.25, 0.75)),
        "dxy_bucket": _bucket(numeric.get("dxy_zscore", 0.0), (-1.0, 1.0)),
        "kimchi_bucket": _bucket(numeric.get("kimchi_premium_zscore", 0.0), (-1.0, 1.0)),
        "drawdown_bucket": _bucket(numeric.get("window_drawdown", 0.0), (-0.08, -0.02)),
    }
    return {"tokens": tokens, "numeric": numeric}


def _prompt(row: dict[str, Any], card: dict[str, Any]) -> str:
    pred = _prediction(row)
    lines = [
        "You are a single LLM meta-controller for a BTCUSDT futures bot.",
        "A frozen weak alpha already proposed a candidate trade. Do not invent a new side.",
        "Use only signal-time state. Decide whether to TAKE or SKIP, and choose size_bucket in {FULL, SMALL, NONE}.",
        f"date: {row.get('date')}",
        f"candidate_side: {pred.get('side', 'NONE')}",
        f"candidate_hold_bars: {pred.get('hold_bars', 0)}",
        "state_tokens:",
    ]
    for key, value in sorted(card["tokens"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("numeric_state:")
    for key, value in sorted(card["numeric"].items()):
        lines.append(f"- {key}: {value:+.6f}")
    lines.append('Return compact JSON with keys: decision, size_bucket, risk_reason.')
    return "\n".join(lines)


def _convert(rows: list[dict[str, Any]], market: pd.DataFrame, features: pd.DataFrame, cfg: LinearAlphaMetaSftConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed))
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = _prediction(row)
        gate = str(pred.get("gate", "NO_TRADE"))
        if gate != "TRADE" and float(rng.random()) > float(cfg.include_no_trade_fraction):
            continue
        ret_pct = None
        if gate == "TRADE":
            ret_pct = _trade_return_pct(
                market=market,
                signal_pos=int(row.get("signal_pos", -1) or -1),
                side=str(pred.get("side", "NONE")),
                hold_bars=min(int(pred.get("hold_bars", 0) or 0), int(cfg.max_hold_bars)),
                leverage=float(cfg.leverage),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                entry_delay_bars=int(cfg.entry_delay_bars),
            )
        target = _label(ret_pct, cfg) if gate == "TRADE" else {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "alpha_did_not_trigger"}
        card = _state_card(row, features)
        prompt = _prompt(row, card)
        target_text = json.dumps(target, ensure_ascii=False, sort_keys=True)
        out.append({
            "task": "linear_alpha_meta_controller_sft",
            "prompt": prompt,
            "target": target_text,
            "messages": [
                {"role": "system", "content": "You are a conservative no-leak BTCUSDT futures meta-controller."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target_text},
            ],
            "metadata": {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "alpha_group": row.get("group"),
                "candidate_side": pred.get("side", "NONE"),
                "candidate_gate": gate,
                "realized_trade_ret_pct": ret_pct,
                "target_decision": target["decision"],
                "target_size_bucket": target["size_bucket"],
                "leakage_guard": "prompt is signal-time only; target uses future outcome for supervised label only",
            },
        })
    return out


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]], cfg: LinearAlphaMetaSftConfig) -> dict[str, Any]:
    decisions = Counter(row["metadata"]["target_decision"] for row in rows)
    sizes = Counter(row["metadata"]["target_size_bucket"] for row in rows)
    sides = Counter(row["metadata"]["candidate_side"] for row in rows)
    rets = [float(row["metadata"]["realized_trade_ret_pct"]) for row in rows if row["metadata"]["realized_trade_ret_pct"] is not None]
    chars = [len(row["prompt"]) + len(row["target"]) for row in rows]
    return {
        "config": asdict(cfg),
        "rows": len(rows),
        "target_decisions": dict(sorted(decisions.items())),
        "target_size_buckets": dict(sorted(sizes.items())),
        "candidate_sides": dict(sorted(sides.items())),
        "trade_stats": _trade_stats([x / 100.0 for x in rets]),
        "prompt_target_chars": {
            "min": min(chars) if chars else 0,
            "max": max(chars) if chars else 0,
            "mean": sum(chars) / max(1, len(chars)),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
        "leakage_guard": {
            "prompt_uses_signal_time_features_only": True,
            "future_outcome_used_only_for_sft_target": True,
            "not_a_live_inference_path": True,
        },
    }


def run(cfg: LinearAlphaMetaSftConfig) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    rows = _read_jsonl(cfg.predictions_jsonl)
    feature_frame = build_market_feature_frame(market, window_size=int(cfg.window_size))
    out_rows = _convert(rows, market, feature_frame, cfg)
    _write_jsonl(cfg.output_jsonl, out_rows)
    summary = _summary(out_rows, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build linear-alpha meta-controller SFT rows")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--window-size", type=int, default=LinearAlphaMetaSftConfig.window_size)
    parser.add_argument("--leverage", type=float, default=LinearAlphaMetaSftConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=LinearAlphaMetaSftConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=LinearAlphaMetaSftConfig.slippage_rate)
    parser.add_argument("--entry-delay-bars", type=int, default=LinearAlphaMetaSftConfig.entry_delay_bars)
    parser.add_argument("--max-hold-bars", type=int, default=LinearAlphaMetaSftConfig.max_hold_bars)
    parser.add_argument("--take-full-ret-pct", type=float, default=LinearAlphaMetaSftConfig.take_full_ret_pct)
    parser.add_argument("--take-small-ret-pct", type=float, default=LinearAlphaMetaSftConfig.take_small_ret_pct)
    parser.add_argument("--include-no-trade-fraction", type=float, default=LinearAlphaMetaSftConfig.include_no_trade_fraction)
    parser.add_argument("--seed", type=int, default=LinearAlphaMetaSftConfig.seed)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(LinearAlphaMetaSftConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
