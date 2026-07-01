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
    target_schema: str = "decision_size_reason"
    prompt_style: str = "default"
    label_mode: str = "realized_return"
    path_full_max_adverse_pct: float = 0.35
    path_small_max_adverse_pct: float = 0.70
    path_min_mfe_mae_ratio: float = 1.25
    path_min_full_mfe_pct: float = 0.55
    path_min_small_mfe_pct: float = 0.20
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


def _trade_path_stats(
    *,
    market: pd.DataFrame,
    signal_pos: int,
    side: str,
    hold_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
) -> dict[str, float] | None:
    signal = _side_signal(side)
    if signal == 0:
        return None
    entry_pos = int(signal_pos) + int(entry_delay_bars)
    exit_pos = entry_pos + max(1, int(hold_bars))
    if entry_pos >= len(market) - 1 or exit_pos >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    entry_price = float(opens[entry_pos])
    if entry_price <= 0.0:
        return None

    eq = 1.0
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    eq *= max(0.0, 1.0 - cost)
    max_favorable_pct = 0.0
    max_adverse_pct = 0.0
    for j in range(entry_pos, exit_pos):
        open_j = float(opens[j])
        if open_j <= 0.0:
            continue
        if signal > 0:
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            favorable = (float(highs[j]) - entry_price) / entry_price
            adverse = (entry_price - float(lows[j])) / entry_price
        else:
            close_ret = (open_j - float(opens[j + 1])) / open_j
            favorable = (entry_price - float(lows[j])) / entry_price
            adverse = (float(highs[j]) - entry_price) / entry_price
        max_favorable_pct = max(max_favorable_pct, 100.0 * float(leverage) * float(favorable))
        max_adverse_pct = max(max_adverse_pct, 100.0 * float(leverage) * float(adverse))
        eq *= max(0.0, 1.0 + float(leverage) * close_ret)
        if eq <= 0.0:
            break
    eq *= max(0.0, 1.0 - cost)
    ret_pct = (eq - 1.0) * 100.0
    return {
        "realized_return_pct": float(ret_pct),
        "max_favorable_pct": float(max_favorable_pct),
        "max_adverse_pct": float(max_adverse_pct),
        "mfe_mae_ratio": float(max_favorable_pct / max(1e-9, max_adverse_pct)),
    }


def _trade_return_pct(**kwargs: Any) -> float | None:
    stats = _trade_path_stats(**kwargs)
    return None if stats is None else float(stats["realized_return_pct"])


def _label(path_stats: dict[str, float] | None, cfg: LinearAlphaMetaSftConfig) -> dict[str, str]:
    if path_stats is None:
        return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "missing_future_bars"}
    ret_pct = float(path_stats["realized_return_pct"])
    if cfg.label_mode == "realized_return":
        if ret_pct >= float(cfg.take_full_ret_pct):
            return {"decision": "TAKE", "size_bucket": "FULL", "risk_reason": "realized_edge_strong"}
        if ret_pct > float(cfg.take_small_ret_pct):
            return {"decision": "TAKE", "size_bucket": "SMALL", "risk_reason": "realized_edge_positive_but_thin"}
        return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "realized_edge_negative_or_costly"}
    if cfg.label_mode != "path_quality":
        raise ValueError("label_mode must be realized_return|path_quality")

    adverse = float(path_stats["max_adverse_pct"])
    favorable = float(path_stats["max_favorable_pct"])
    ratio = float(path_stats["mfe_mae_ratio"])
    if (
        ret_pct >= float(cfg.take_full_ret_pct)
        and adverse <= float(cfg.path_full_max_adverse_pct)
        and favorable >= float(cfg.path_min_full_mfe_pct)
        and ratio >= float(cfg.path_min_mfe_mae_ratio)
    ):
        return {"decision": "TAKE", "size_bucket": "FULL", "risk_reason": "path_quality_strong_low_adverse"}
    if (
        ret_pct > float(cfg.take_small_ret_pct)
        and adverse <= float(cfg.path_small_max_adverse_pct)
        and favorable >= float(cfg.path_min_small_mfe_pct)
        and ratio >= 1.0
    ):
        return {"decision": "TAKE", "size_bucket": "SMALL", "risk_reason": "path_quality_positive_controlled_adverse"}
    return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "path_quality_failed_return_or_adverse"}


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
        "rex_576_range_width_pct", "rex_576_range_pos",
        "rex_576_cur_to_max_pct", "rex_576_cur_to_min_pct",
        "rex_2016_range_width_pct", "rex_2016_range_pos",
        "rex_2016_cur_to_max_pct", "rex_2016_cur_to_min_pct",
        "rex_8640_range_width_pct", "rex_8640_range_pos",
        "rex_8640_cur_to_max_pct", "rex_8640_cur_to_min_pct",
        "htf_4h_return_4", "htf_4h_range_pos", "htf_4h_drawdown_4",
        "htf_1d_return_4", "htf_1d_range_pos", "htf_1d_drawdown_4",
        "htf_1w_return_4", "htf_1w_range_pos", "htf_1w_drawdown_4",
    ]
    numeric = {k: round(_feature_value(features, pos, k), 6) for k in keys if k in features.columns}
    side_signal = _side_signal(str(pred.get("side", "NONE")))
    numeric["candidate_side_signal"] = float(side_signal)
    if side_signal != 0:
        for key in ("trend_12", "trend_96", "range_pos", "rsi_norm", "bb_z", "return_zscore_48", "taker_buy_ratio", "rex_144_range_pos", "rex_576_range_pos", "rex_2016_range_pos", "rex_8640_range_pos", "htf_4h_return_4", "htf_1d_return_4", "htf_1w_return_4"):
            if key in numeric:
                numeric[f"side_{key}"] = round(float(side_signal) * float(numeric[key]), 6)
        numeric["side_room_to_adverse_extreme_pct"] = round(
            float(numeric.get("rex_8640_cur_to_min_pct", 0.0) if side_signal > 0 else numeric.get("rex_8640_cur_to_max_pct", 0.0)),
            6,
        )
        numeric["side_room_to_favorable_extreme_pct"] = round(
            float(numeric.get("rex_8640_cur_to_max_pct", 0.0) if side_signal > 0 else numeric.get("rex_8640_cur_to_min_pct", 0.0)),
            6,
        )
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
        "range_vol_bucket": _bucket(numeric.get("range_vol", 0.0), (0.02, 0.08)),
        "side_range_pos_bucket": _bucket(numeric.get("side_range_pos", 0.0), (-0.5, 0.5)),
    }
    return {"tokens": tokens, "numeric": numeric}


def _prompt(row: dict[str, Any], card: dict[str, Any], cfg: LinearAlphaMetaSftConfig) -> str:
    pred = _prediction(row)
    if cfg.prompt_style not in {"default", "conservative", "stable_pa"}:
        raise ValueError("prompt_style must be default|conservative|stable_pa")
    tokens = dict(card["tokens"])
    numeric = dict(card["numeric"])
    if cfg.prompt_style == "stable_pa":
        token_keep = {
            "alpha_group", "alpha_variant", "alpha_side", "alpha_hold_bars",
            "range_pos_bucket", "range_vol_bucket", "side_range_pos_bucket",
            "dxy_bucket", "kimchi_bucket", "drawdown_bucket",
        }
        numeric_keep = {
            "trend_96", "range_vol", "window_drawdown", "range_pos",
            "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change",
            "usdkrw_zscore", "usdkrw_momentum",
            "rex_2016_range_pos", "rex_2016_cur_to_max_pct", "rex_2016_cur_to_min_pct",
            "rex_8640_range_pos", "rex_8640_cur_to_max_pct", "rex_8640_cur_to_min_pct",
            "side_trend_96", "side_range_pos", "side_rex_2016_range_pos", "side_rex_8640_range_pos",
            "side_room_to_adverse_extreme_pct", "side_room_to_favorable_extreme_pct",
            "htf_4h_return_4", "htf_1d_return_4", "htf_1w_return_4",
        }
        tokens = {k: v for k, v in tokens.items() if k in token_keep}
        numeric = {k: v for k, v in numeric.items() if k in numeric_keep}
    lines = [
        "You are a single LLM meta-controller for a BTCUSDT futures bot.",
        "A frozen weak alpha already proposed a candidate trade. Do not invent a new side.",
        "Use only signal-time state. Decide whether to TAKE or SKIP, and choose size_bucket in {FULL, SMALL, NONE}.",
        f"date: {row.get('date')}",
        f"candidate_side: {pred.get('side', 'NONE')}",
        f"candidate_hold_bars: {pred.get('hold_bars', 0)}",
    ]
    if cfg.prompt_style == "conservative":
        lines.extend([
            "Default to SKIP/NONE unless the signal-time evidence clearly supports the frozen alpha.",
            "Use TAKE/FULL only for unusually strong evidence; use TAKE/SMALL for marginal positive evidence.",
        ])
    elif cfg.prompt_style == "stable_pa":
        lines.extend([
            "Use a compact stable price-action card: rolling-extrema, range/volatility, trend, and macro bucket context.",
            "Ignore missing nuance; prefer robust TAKE only when multiple stable clues align with the frozen alpha side.",
        ])
    lines.append("state_tokens:")
    for key, value in sorted(tokens.items()):
        lines.append(f"- {key}: {value}")
    lines.append("numeric_state:")
    for key, value in sorted(numeric.items()):
        lines.append(f"- {key}: {value:+.6f}")
    if cfg.target_schema == "decision":
        lines.append('Return compact JSON with exactly key: decision.')
    elif cfg.target_schema == "decision_size":
        lines.append('Return compact JSON with exactly keys: decision, size_bucket.')
    elif cfg.target_schema == "decision_size_reason":
        lines.append('Return compact JSON with keys: decision, size_bucket, risk_reason.')
    else:
        raise ValueError("target_schema must be decision|decision_size|decision_size_reason")
    return "\n".join(lines)


def _convert(rows: list[dict[str, Any]], market: pd.DataFrame, features: pd.DataFrame, cfg: LinearAlphaMetaSftConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed))
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = _prediction(row)
        gate = str(pred.get("gate", "NO_TRADE"))
        if gate != "TRADE" and float(rng.random()) > float(cfg.include_no_trade_fraction):
            continue
        path_stats = None
        if gate == "TRADE":
            path_stats = _trade_path_stats(
                market=market,
                signal_pos=int(row.get("signal_pos", -1) or -1),
                side=str(pred.get("side", "NONE")),
                hold_bars=min(int(pred.get("hold_bars", 0) or 0), int(cfg.max_hold_bars)),
                leverage=float(cfg.leverage),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                entry_delay_bars=int(cfg.entry_delay_bars),
            )
        target = _label(path_stats, cfg) if gate == "TRADE" else {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "alpha_did_not_trigger"}
        card = _state_card(row, features)
        prompt = _prompt(row, card, cfg)
        if cfg.target_schema == "decision":
            target_for_text = {"decision": target["decision"]}
        elif cfg.target_schema == "decision_size":
            target_for_text = {"decision": target["decision"], "size_bucket": target["size_bucket"]}
        elif cfg.target_schema == "decision_size_reason":
            target_for_text = target
        else:
            raise ValueError("target_schema must be decision|decision_size|decision_size_reason")
        target_text = json.dumps(target_for_text, ensure_ascii=False, sort_keys=True)
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
                "realized_trade_ret_pct": None if path_stats is None else path_stats["realized_return_pct"],
                "trade_path_stats": path_stats,
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
    adverse = [float(row["metadata"]["trade_path_stats"]["max_adverse_pct"]) for row in rows if row["metadata"].get("trade_path_stats") is not None]
    favorable = [float(row["metadata"]["trade_path_stats"]["max_favorable_pct"]) for row in rows if row["metadata"].get("trade_path_stats") is not None]
    chars = [len(row["prompt"]) + len(row["target"]) for row in rows]
    return {
        "config": asdict(cfg),
        "rows": len(rows),
        "target_decisions": dict(sorted(decisions.items())),
        "target_size_buckets": dict(sorted(sizes.items())),
        "candidate_sides": dict(sorted(sides.items())),
        "trade_stats": _trade_stats([x / 100.0 for x in rets]),
        "path_stats_pct": {
            "max_adverse_mean": float(np.mean(adverse)) if adverse else 0.0,
            "max_adverse_p90": float(np.quantile(adverse, 0.90)) if adverse else 0.0,
            "max_favorable_mean": float(np.mean(favorable)) if favorable else 0.0,
            "max_favorable_p90": float(np.quantile(favorable, 0.90)) if favorable else 0.0,
        },
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
    parser.add_argument("--target-schema", choices=["decision", "decision_size", "decision_size_reason"], default=LinearAlphaMetaSftConfig.target_schema)
    parser.add_argument("--prompt-style", choices=["default", "conservative", "stable_pa"], default=LinearAlphaMetaSftConfig.prompt_style)
    parser.add_argument("--label-mode", choices=["realized_return", "path_quality"], default=LinearAlphaMetaSftConfig.label_mode)
    parser.add_argument("--path-full-max-adverse-pct", type=float, default=LinearAlphaMetaSftConfig.path_full_max_adverse_pct)
    parser.add_argument("--path-small-max-adverse-pct", type=float, default=LinearAlphaMetaSftConfig.path_small_max_adverse_pct)
    parser.add_argument("--path-min-mfe-mae-ratio", type=float, default=LinearAlphaMetaSftConfig.path_min_mfe_mae_ratio)
    parser.add_argument("--path-min-full-mfe-pct", type=float, default=LinearAlphaMetaSftConfig.path_min_full_mfe_pct)
    parser.add_argument("--path-min-small-mfe-pct", type=float, default=LinearAlphaMetaSftConfig.path_min_small_mfe_pct)
    parser.add_argument("--seed", type=int, default=LinearAlphaMetaSftConfig.seed)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(LinearAlphaMetaSftConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
