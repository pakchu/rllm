"""Export compact candidate-level binary edge rows for LLM scoring.

The previous same-signal A/B/C task asks the model to choose between LONG,
SHORT, and NO_TRADE at once. This exporter instead asks whether one concrete
candidate setup has enough edge. The downstream policy can score all candidates
and keep risk/execution logic outside the LLM.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventCandidateBinaryEdgeCfg:
    train_candidates_jsonl: str
    eval_candidates_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    min_edge_utility: float = 0.003
    numeric_keys: str = (
        "action_strength,trend_24,trend_96,range_pos,bb_z,volume_zscore,taker_imbalance,"
        "dxy_zscore,kimchi_premium_zscore,kimchi_premium_change,usdkrw_momentum,"
        "htf_4h_return_4,htf_1d_return_1,htf_1w_return_4,"
        "pa_ext_36_range_pos,pa_ext_36_to_max_high_pct,pa_ext_36_to_min_low_pct,pa_ext_36_max_high_age_frac,pa_ext_36_min_low_age_frac,"
        "pa_ext_72_range_pos,pa_ext_72_to_max_high_pct,pa_ext_72_to_min_low_pct,pa_ext_72_max_high_age_frac,pa_ext_72_min_low_age_frac,"
        "pa_ext_144_range_pos,pa_ext_144_to_max_high_pct,pa_ext_144_to_min_low_pct,pa_ext_144_max_high_age_frac,pa_ext_144_min_low_age_frac,"
        "rex_36_range_width_pct,rex_36_range_pos,rex_36_cur_to_max_pct,rex_36_cur_to_min_pct,"
        "rex_144_range_width_pct,rex_144_range_pos,rex_144_cur_to_max_pct,rex_144_cur_to_min_pct,"
        "rex_576_range_width_pct,rex_576_range_pos,rex_576_cur_to_max_pct,rex_576_cur_to_min_pct,"
        "rex_2016_range_width_pct,rex_2016_range_pos,rex_2016_cur_to_max_pct,rex_2016_cur_to_min_pct,"
        "rex_8640_range_width_pct,rex_8640_range_pos,rex_8640_cur_to_max_pct,rex_8640_cur_to_min_pct"
    )


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("utility", reward.get("rank_utility", reward.get("net_return_pct", 0.0))) or 0.0)


def _reward_summary(row: dict[str, Any]) -> dict[str, float]:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return {
        "utility": _utility(row),
        "net_return_pct": float(reward.get("net_return_pct", reward.get("net_return", 0.0)) or 0.0),
        "mae": float(reward.get("mae", reward.get("mae_pct", 0.0)) or 0.0),
        "mfe": float(reward.get("mfe", reward.get("mfe_pct", 0.0)) or 0.0),
    }


def _fmt_float(v: Any) -> str | None:
    try:
        return f"{float(v):+.5f}"
    except Exception:
        return None


def _prompt(row: dict[str, Any], cfg: EventCandidateBinaryEdgeCfg) -> str:
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    tokens = row.get("state_tokens") if isinstance(row.get("state_tokens"), dict) else {}
    snap = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
    numeric_keys = [k.strip() for k in str(cfg.numeric_keys).split(",") if k.strip()]
    lines = [
        "Task: decide whether this single BTCUSDT futures setup has positive edge.",
        "Use only signal-time context. Do not infer or use future reward.",
        "Answer with exactly one letter: A or B.",
        "A = EDGE_TRADE",
        "B = NO_EDGE",
        f"Date: {row.get('date')}",
        f"Signal position: {row.get('signal_pos')}",
        f"Candidate side: {str(cand.get('side', row.get('side', 'UNKNOWN'))).upper()}",
        f"Candidate hold_bars: {int(cand.get('hold_bars', cand.get('horizon', 0)) or 0)}",
        f"Candidate family: {cand.get('family', 'unknown')}",
        "State buckets:",
    ]
    keep_prefixes = ("tok:rex_", "tok:pa_ext_")
    keep_keys = {
        "family", "hold_bucket", "range_location", "side", "side_trend_24", "side_trend_96",
        "trend_24", "trend_96", "htf_4h", "htf_1d", "htf_1w", "taker_flow",
        "volume_state", "bb_pressure", "drawdown_state", "dxy_pressure", "kimchi_level",
        "kimchi_change", "usdkrw_pressure",
    }
    for key in sorted(tokens):
        if key in keep_keys or str(key).startswith(keep_prefixes):
            lines.append(f"- {key}: {tokens[key]}")
    lines.append("Numeric price-action/context evidence:")
    for key in numeric_keys:
        if key in snap:
            val = _fmt_float(snap[key])
            if val is not None:
                lines.append(f"- {key}: {val}")
    return "\n".join(lines)


def _rows(rows: list[dict[str, Any]], cfg: EventCandidateBinaryEdgeCfg) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        utility = _utility(row)
        target = "A" if utility >= float(cfg.min_edge_utility) else "B"
        out.append(
            {
                "task": "event_candidate_binary_edge",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "side": str(cand.get("side", row.get("side", "UNKNOWN"))).upper(),
                "hold_bars": int(cand.get("hold_bars", cand.get("horizon", 0)) or 0),
                "prompt": _prompt(row, cfg),
                "target": target,
                "label": "EDGE_TRADE" if target == "A" else "NO_EDGE",
                "choice_utility": {"A": utility, "B": 0.0},
                "candidate": cand,
                "reward_audit": _reward_summary(row),
                "source": {
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "side": str(cand.get("side", row.get("side", "UNKNOWN"))).upper(),
                    "candidate": cand,
                    "reward": row.get("reward"),
                },
                "leakage_guard": {
                    "prompt_uses_future_reward": False,
                    "target_uses_future_reward_for_training_only": True,
                    "candidate_level_binary_edge": True,
                },
            }
        )
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r["target"]) for r in rows)
    sides = Counter(str(r.get("side")) for r in rows)
    holds = Counter(str(r.get("hold_bars")) for r in rows)
    lens = [len(str(r.get("prompt", ""))) for r in rows]
    utils = [float((r.get("reward_audit") or {}).get("utility", 0.0)) for r in rows]
    return {
        "rows": len(rows),
        "target_counts": dict(sorted(counts.items())),
        "side_counts": dict(sorted(sides.items())),
        "hold_counts": dict(sorted(holds.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)),
        "prompt_chars": {"min": min(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens)), "max": max(lens) if lens else 0},
        "utility": {"min": min(utils) if utils else 0.0, "mean": sum(utils) / max(1, len(utils)), "max": max(utils) if utils else 0.0},
    }


def run(cfg: EventCandidateBinaryEdgeCfg) -> dict[str, Any]:
    train = _rows(_load(cfg.train_candidates_jsonl), cfg)
    eval_rows = _rows(_load(cfg.eval_candidates_jsonl), cfg)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report = {
        "config": asdict(cfg),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": _summary(train),
        "eval": _summary(eval_rows),
        "contract": "Candidate-level A/B binary edge classification; prompt is signal-time only; target is future-reward label for training only",
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-candidates-jsonl", required=True)
    p.add_argument("--eval-candidates-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--min-edge-utility", type=float, default=EventCandidateBinaryEdgeCfg.min_edge_utility)
    p.add_argument("--numeric-keys", default=EventCandidateBinaryEdgeCfg.numeric_keys)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateBinaryEdgeCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
