"""Export regime/family-matched A/B option pairs for LLM relative setup judgment."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegimePairwiseOptionCfg:
    train_candidates_jsonl: str
    eval_candidates_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    max_pairs_per_group: int = 8
    min_utility_gap: float = 0.006
    seed: int = 45
    prompt_mode: str = "verbose"
    numeric_keys: str = (
        "action_strength,trend_24,trend_96,range_pos,bb_z,volume_zscore,taker_imbalance,"
        "htf_4h_return_4,htf_1d_return_1,htf_1w_return_4,"
        "pa_ext_36_range_pos,pa_ext_36_to_max_high_pct,pa_ext_36_to_min_low_pct,pa_ext_36_max_high_age_frac,pa_ext_36_min_low_age_frac,"
        "pa_ext_72_range_pos,pa_ext_72_to_max_high_pct,pa_ext_72_to_min_low_pct,pa_ext_72_max_high_age_frac,pa_ext_72_min_low_age_frac,"
        "pa_ext_144_range_pos,pa_ext_144_to_max_high_pct,pa_ext_144_to_min_low_pct,pa_ext_144_max_high_age_frac,pa_ext_144_min_low_age_frac,"
        "rex_36_range_width_pct,rex_36_range_pos,rex_36_cur_to_max_pct,rex_36_cur_to_min_pct,"
        "rex_144_range_width_pct,rex_144_range_pos,rex_144_cur_to_max_pct,rex_144_cur_to_min_pct,"
        "rex_576_range_width_pct,rex_576_range_pos,rex_576_cur_to_max_pct,rex_576_cur_to_min_pct,"
        "rex_2016_range_width_pct,rex_2016_range_pos,rex_2016_cur_to_max_pct,rex_2016_cur_to_min_pct"
    )


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("rank_utility", reward.get("net_return_pct", 0.0)) or 0.0)


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("candidate") if isinstance(row.get("candidate"), dict) else {}


def _group_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    cand = _candidate(row)
    side = str(cand.get("side", row.get("side", "UNKNOWN"))).upper()
    hold = int(cand.get("hold_bars", cand.get("horizon", 0)) or 0)
    family = str(cand.get("family", "unknown"))
    return (_month(row), side, hold, family)


def _fmt(v: Any) -> str | None:
    try:
        return f"{float(v):+.4f}"
    except Exception:
        return None


def _setup_block(label: str, row: dict[str, Any], keys: list[str]) -> list[str]:
    cand = _candidate(row)
    tokens = row.get("state_tokens") if isinstance(row.get("state_tokens"), dict) else {}
    snap = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
    lines = [
        f"Setup {label}:",
        f"- side: {str(cand.get('side', row.get('side', 'UNKNOWN'))).upper()}",
        f"- hold_bars: {int(cand.get('hold_bars', cand.get('horizon', 0)) or 0)}",
        f"- family: {cand.get('family', 'unknown')}",
    ]
    keep_tokens = [
        "range_location", "side_trend_24", "side_trend_96", "trend_24", "trend_96",
        "htf_4h", "htf_1d", "htf_1w", "taker_flow", "volume_state", "drawdown_state",
        "tok:rex_144_loc", "tok:rex_144_lower_gap", "tok:rex_144_upper_gap",
        "tok:rex_2016_loc", "tok:rex_2016_lower_gap", "tok:rex_2016_upper_gap",
        "tok:rex_576_loc", "tok:rex_8640_loc",
    ]
    for key in keep_tokens:
        if key in tokens:
            lines.append(f"- {key}: {tokens[key]}")
    nums = []
    for key in keys:
        if key in snap:
            val = _fmt(snap[key])
            if val is not None:
                nums.append(f"{key}={val}")
    # Keep block compact. The selected keys are already ordered by relevance.
    if nums:
        lines.append("- numeric: " + "; ".join(nums))
    return lines



def _bucket_signed(v: float, eps: float = 1e-9) -> str:
    if v > eps:
        return "A_HIGHER"
    if v < -eps:
        return "B_HIGHER"
    return "SIMILAR"


def _bucket_abs_gap(v: float) -> str:
    av = abs(float(v))
    if av >= 0.05:
        return "large"
    if av >= 0.015:
        return "medium"
    if av >= 0.004:
        return "small"
    return "tiny"


def _compact_pair_prompt(a_row: dict[str, Any], b_row: dict[str, Any], keys: list[str]) -> str:
    a_cand = _candidate(a_row); b_cand = _candidate(b_row)
    a_tokens = a_row.get("state_tokens") if isinstance(a_row.get("state_tokens"), dict) else {}
    b_tokens = b_row.get("state_tokens") if isinstance(b_row.get("state_tokens"), dict) else {}
    a_snap = a_row.get("feature_snapshot") if isinstance(a_row.get("feature_snapshot"), dict) else {}
    b_snap = b_row.get("feature_snapshot") if isinstance(b_row.get("feature_snapshot"), dict) else {}
    lines = [
        "Task: choose which same-regime BTCUSDT setup has better forward path quality.",
        "Answer exactly one letter: A or B.",
        f"Shared context: side={str(a_cand.get('side', a_row.get('side', 'UNKNOWN'))).upper()}, hold_bars={int(a_cand.get('hold_bars', 0) or 0)}, family={a_cand.get('family', 'unknown')}, month={_month(a_row)}",
        "Compare semantic state:",
    ]
    compare_tokens = [
        "range_location", "drawdown_state", "trend_24", "trend_96", "htf_4h", "htf_1d", "htf_1w",
        "taker_flow", "volume_state", "tok:rex_144_loc", "tok:rex_2016_loc", "tok:rex_144_lower_gap", "tok:rex_144_upper_gap",
    ]
    for key in compare_tokens:
        av = a_tokens.get(key)
        bv = b_tokens.get(key)
        if av is not None or bv is not None:
            lines.append(f"- {key}: A={av if av is not None else 'na'} | B={bv if bv is not None else 'na'}")
    lines.append("Relative numeric evidence (A minus B, bucketed):")
    for key in keys:
        if key not in a_snap or key not in b_snap:
            continue
        try:
            diff = float(a_snap[key]) - float(b_snap[key])
        except Exception:
            continue
        lines.append(f"- {key}: {_bucket_signed(diff)} gap={_bucket_abs_gap(diff)}")
    return "\n".join(lines)



def _ultra_compact_pair_prompt(a_row: dict[str, Any], b_row: dict[str, Any], keys: list[str]) -> str:
    a_cand = _candidate(a_row)
    a_tokens = a_row.get("state_tokens") if isinstance(a_row.get("state_tokens"), dict) else {}
    b_tokens = b_row.get("state_tokens") if isinstance(b_row.get("state_tokens"), dict) else {}
    a_snap = a_row.get("feature_snapshot") if isinstance(a_row.get("feature_snapshot"), dict) else {}
    b_snap = b_row.get("feature_snapshot") if isinstance(b_row.get("feature_snapshot"), dict) else {}
    lines = [
        "Task: choose the better same-regime BTCUSDT price-action setup.",
        "Answer exactly A or B.",
        f"Shared: side={str(a_cand.get('side', a_row.get('side', 'UNKNOWN'))).upper()} hold={int(a_cand.get('hold_bars', 0) or 0)} family={a_cand.get('family', 'unknown')} month={_month(a_row)}",
        "State: "
        + "; ".join(
            f"{k}:A={a_tokens.get(k, 'na')},B={b_tokens.get(k, 'na')}"
            for k in ["range_location", "drawdown_state", "trend_24", "trend_96", "htf_4h", "htf_1d", "htf_1w", "taker_flow", "tok:rex_144_loc", "tok:rex_2016_loc"]
        ),
        "Relative PA evidence:",
    ]
    for key in keys:
        if key not in a_snap or key not in b_snap:
            continue
        try:
            diff = float(a_snap[key]) - float(b_snap[key])
        except Exception:
            continue
        lines.append(f"- {key}: {_bucket_signed(diff)} {_bucket_abs_gap(diff)}")
    return "\n".join(lines)


def _source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "candidate": row.get("candidate"),
        "side": str(_candidate(row).get("side", row.get("side", "UNKNOWN"))).upper(),
        "reward": row.get("reward"),
        "utility": _utility(row),
    }


def _row(win: dict[str, Any], lose: dict[str, Any], *, winner_label: str, keys: list[str], gap: float, prompt_mode: str) -> dict[str, Any]:
    a_row, b_row = (win, lose) if winner_label == "A" else (lose, win)
    if prompt_mode == "ultra_compact":
        prompt = _ultra_compact_pair_prompt(a_row, b_row, keys)
    elif prompt_mode == "compact":
        prompt = _compact_pair_prompt(a_row, b_row, keys)
    else:
        prompt_lines = [
            "Task: compare two BTCUSDT futures setups from the same month/regime family.",
            "Use only the signal-time price-action structure shown below.",
            "Choose the setup with better forward path quality. Answer exactly one letter: A or B.",
            "Both setups share side/hold/family context; focus on relative price-action location, extrema, trend, and risk.",
            "",
            *_setup_block("A", a_row, keys),
            "",
            *_setup_block("B", b_row, keys),
        ]
        prompt = "\n".join(prompt_lines)
    return {
        "task": "event_candidate_regime_pairwise_option",
        "date": win.get("date"),
        "month": _month(win),
        "signal_pos": win.get("signal_pos"),
        "group_key": list(_group_key(win)),
        "prompt": prompt,
        "target": winner_label,
        "utility_gap": gap,
        "choice_utility": {"A": _utility(a_row), "B": _utility(b_row)},
        "candidates": {"A": _source(a_row), "B": _source(b_row)},
        "leakage_guard": {
            "prompt_uses_future_reward": False,
            "target_uses_future_reward_for_training_only": True,
            "pairs_matched_by_month_side_hold_family": True,
            "winner_position_randomized": True,
            "prompt_mode": prompt_mode,
        },
    }


def _build(rows: list[dict[str, Any]], cfg: RegimePairwiseOptionCfg, *, seed_offset: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    rng = random.Random(int(cfg.seed) + int(seed_offset))
    keys = [x.strip() for x in str(cfg.numeric_keys).split(",") if x.strip()]
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        vals = sorted(groups[key], key=_utility, reverse=True)
        if len(vals) < 2:
            continue
        left = 0
        right = len(vals) - 1
        made = 0
        while left < right and made < int(cfg.max_pairs_per_group):
            win = vals[left]
            lose = vals[right]
            gap = _utility(win) - _utility(lose)
            if gap < float(cfg.min_utility_gap):
                break
            winner_label = "A" if rng.random() < 0.5 else "B"
            out.append(_row(win, lose, winner_label=winner_label, keys=keys, gap=gap, prompt_mode=str(cfg.prompt_mode)))
            made += 1
            left += 1
            right -= 1
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = Counter(str(r["target"]) for r in rows)
    groups = Counter("|".join(map(str, r.get("group_key", []))) for r in rows)
    lens = [len(str(r.get("prompt", ""))) for r in rows]
    gaps = [float(r.get("utility_gap", 0.0)) for r in rows]
    return {
        "rows": len(rows),
        "target_counts": dict(sorted(targets.items())),
        "groups": len(groups),
        "prompt_chars": {"min": min(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens)), "max": max(lens) if lens else 0},
        "utility_gap": {"min": min(gaps) if gaps else 0.0, "mean": sum(gaps) / max(1, len(gaps)), "max": max(gaps) if gaps else 0.0},
    }


def run(cfg: RegimePairwiseOptionCfg) -> dict[str, Any]:
    train = _build(_load(cfg.train_candidates_jsonl), cfg, seed_offset=0)
    eval_rows = _build(_load(cfg.eval_candidates_jsonl), cfg, seed_offset=10_000)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report = {
        "config": asdict(cfg),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": _summary(train),
        "eval": _summary(eval_rows),
        "contract": "A/B pairwise option; pairs matched by month, side, hold, and family; prompt is signal-time only",
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
    p.add_argument("--max-pairs-per-group", type=int, default=RegimePairwiseOptionCfg.max_pairs_per_group)
    p.add_argument("--min-utility-gap", type=float, default=RegimePairwiseOptionCfg.min_utility_gap)
    p.add_argument("--seed", type=int, default=RegimePairwiseOptionCfg.seed)
    p.add_argument("--prompt-mode", choices=["verbose", "compact", "ultra_compact"], default=RegimePairwiseOptionCfg.prompt_mode)
    p.add_argument("--numeric-keys", default=RegimePairwiseOptionCfg.numeric_keys)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RegimePairwiseOptionCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
