"""Export pairwise preference rows from episode survival candidates.

Each row compares two candidates at the same signal timestamp.  The prompt uses
only causal candidate/setup/macro/history descriptors; the chosen answer is the
candidate with higher future path utility for offline preference training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market


@dataclass(frozen=True)
class EpisodeSurvivalPairwiseCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output_dir: str
    min_utility_gap_pct: float = 0.35
    max_pairs_per_signal: int = 3
    max_rows_per_split: int = 50000
    seed: int = 42
    gzip_output: bool = True
    prompt_style: str = "json"
    augment_swaps: bool = False


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]], gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _prompt_parts(prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in str(prompt).splitlines():
        if line.startswith("setup_quality: "):
            out["setup_quality"] = json.loads(line.split(": ", 1)[1])
        elif line.startswith("macro_context: "):
            out["macro_context"] = json.loads(line.split(": ", 1)[1])
    return out


def _history_context(market: pd.DataFrame, pos: int) -> dict[str, Any]:
    close = market["close"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    pos = int(pos)
    out: dict[str, Any] = {}
    vols: dict[int, float] = {}
    rets_by_w: dict[int, float] = {}
    for w in (12, 48, 144, 576):
        start = max(0, pos - int(w) + 1)
        c0 = float(close[start]) if close[start] > 0 else float(close[pos])
        ret = float(close[pos] / c0 - 1.0) if c0 > 0 else 0.0
        path = close[start : pos + 1]
        rets = np.diff(np.log(np.maximum(path, 1e-12))) if len(path) > 1 else np.asarray([0.0])
        hi = float(np.max(high[start : pos + 1]))
        lo = float(np.min(low[start : pos + 1]))
        rng = max(1e-12, hi - lo)
        vol = float(np.std(rets))
        vols[w] = vol
        rets_by_w[w] = ret
        out[f"ret_{w}"] = round(ret, 5)
        out[f"vol_{w}"] = round(vol, 6)
        out[f"range_pos_{w}"] = round(float((close[pos] - lo) / rng), 4)
        out[f"drawdown_{w}"] = round(float(close[pos] / max(1e-12, hi) - 1.0), 5)
        out[f"range_bps_{w}"] = round(float(rng / max(1e-12, close[pos]) * 10_000.0), 2)

    # Causal compression / expansion descriptors.
    out["vol12_to_vol144"] = round(vols.get(12, 0.0) / max(1e-9, vols.get(144, 0.0)), 4)
    out["vol48_to_vol576"] = round(vols.get(48, 0.0) / max(1e-9, vols.get(576, 0.0)), 4)
    out["trend_alignment_score"] = int(np.sign(rets_by_w.get(48, 0.0)) + np.sign(rets_by_w.get(144, 0.0)) + np.sign(rets_by_w.get(576, 0.0)))
    out["trend_stack"] = "BULL" if out["trend_alignment_score"] >= 2 else "BEAR" if out["trend_alignment_score"] <= -2 else "MIXED"

    # Consecutive bars above/below local SMA are a causal regime-age proxy.
    for w in (48, 144):
        start = max(0, pos - w + 1)
        sma = float(np.mean(close[start : pos + 1]))
        sign = 1 if close[pos] >= sma else -1
        age = 0
        for j in range(pos, max(-1, pos - 288), -1):
            jj_start = max(0, j - w + 1)
            jj_sma = float(np.mean(close[jj_start : j + 1]))
            jj_sign = 1 if close[j] >= jj_sma else -1
            if jj_sign != sign:
                break
            age += 1
        out[f"sma{w}_side"] = "ABOVE" if sign > 0 else "BELOW"
        out[f"sma{w}_age"] = age

    # Prior realized adverse-risk proxies from completed bars only.
    for w in (12, 48, 144):
        start = max(0, pos - w + 1)
        o = np.maximum(open_[start : pos + 1], 1e-12)
        lower_adverse = np.maximum(0.0, (o - low[start : pos + 1]) / o)
        upper_adverse = np.maximum(0.0, (high[start : pos + 1] - o) / o)
        out[f"prior_long_mae_proxy_{w}"] = round(float(np.mean(lower_adverse) * 100.0), 4)
        out[f"prior_short_mae_proxy_{w}"] = round(float(np.mean(upper_adverse) * 100.0), 4)
        out[f"tail_risk_max_{w}"] = round(float(max(np.max(lower_adverse), np.max(upper_adverse)) * 100.0), 4)
    return out


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    parts = _prompt_parts(str(row.get("prompt", "")))
    return {
        "candidate": row.get("candidate") or {},
        "setup_quality": parts.get("setup_quality") or {},
        "macro_context": parts.get("macro_context") or {},
    }


def _utility(row: dict[str, Any]) -> float:
    return float((row.get("target_audit") or {}).get("utility_pct", 0.0) or 0.0)


def _target_audit(row: dict[str, Any]) -> dict[str, Any]:
    a = dict(row.get("target_audit") or {})
    return {k: a.get(k) for k in ("net_pct", "mae_pct", "mfe_pct", "mfe_to_mae", "utility_pct", "decision", "reason")}


def _competition_context(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ca = a.get("candidate") or {}
    cb = b.get("candidate") or {}
    sa = a.get("setup_quality") or {}
    sb = b.get("setup_quality") or {}
    def f(obj: dict[str, Any], key: str) -> float:
        try:
            return float(obj.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0
    return {
        "same_side": str(ca.get("side")) == str(cb.get("side")),
        "same_event_type": str(ca.get("event_type")) == str(cb.get("event_type")),
        "horizon_diff": int(ca.get("horizon", 0) or 0) - int(cb.get("horizon", 0) or 0),
        "risk_bps_diff_A_minus_B": round(f(sa, "risk_bps") - f(sb, "risk_bps"), 3),
        "range_bps_diff_A_minus_B": round(f(sa, "range_bps") - f(sb, "range_bps"), 3),
        "close_quality_diff_A_minus_B": round(f(sa, "close_quality") - f(sb, "close_quality"), 4),
        "wick_frac_diff_A_minus_B": round(f(sa, "wick_frac") - f(sb, "wick_frac"), 4),
    }


def _pair_prompt_json(date: str, history: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a BTCUSDT futures candidate ranker.",
            "Use only causal history/setup/context. Choose the candidate more likely to produce higher path-risk-adjusted utility.",
            "Return JSON with keys: choice, confidence, reason. choice must be A or B.",
            f"date: {date}",
            f"causal_history: {json.dumps(history, sort_keys=True, separators=(',', ':'))}",
            f"candidate_A: {json.dumps(a, sort_keys=True, separators=(',', ':'))}",
            f"candidate_B: {json.dumps(b, sort_keys=True, separators=(',', ':'))}",
            f"competition_context: {json.dumps(_competition_context(a, b), sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _zone(v: float) -> str:
    if v <= 0.2:
        return "BOTTOM_QUINTILE"
    if v <= 0.4:
        return "LOWER_RANGE"
    if v <= 0.6:
        return "MIDDLE_RANGE"
    if v <= 0.8:
        return "UPPER_RANGE"
    return "TOP_QUINTILE"


def _move(v: float) -> str:
    bps = float(v) * 10_000.0
    if bps <= -250:
        return "SHARP_DOWN"
    if bps <= -80:
        return "DOWN"
    if bps < 80:
        return "FLAT"
    if bps < 250:
        return "UP"
    return "SHARP_UP"


def _ratio(v: float, lo: float = 0.75, hi: float = 1.35) -> str:
    if v < lo:
        return "COMPRESSED"
    if v > hi:
        return "EXPANDED"
    return "NORMAL"


def _z(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "UNKNOWN"
    if x <= -1.0:
        return "LOW"
    if x >= 1.0:
        return "HIGH"
    return "NEUTRAL"


def _market_clauses(history: dict[str, Any]) -> list[str]:
    clauses = [
        f"trend_stack={history.get('trend_stack')} alignment={history.get('trend_alignment_score')}",
        f"volatility short_vs_medium={_ratio(float(history.get('vol12_to_vol144', 1.0) or 1.0))} medium_vs_long={_ratio(float(history.get('vol48_to_vol576', 1.0) or 1.0))}",
    ]
    for w in (12, 48, 144, 576):
        clauses.append(
            " ".join(
                [
                    f"lookback_{w}:",
                    f"return={_move(float(history.get(f'ret_{w}', 0.0) or 0.0))}",
                    f"price_zone={_zone(float(history.get(f'range_pos_{w}', 0.5) or 0.5))}",
                    f"drawdown={_move(float(history.get(f'drawdown_{w}', 0.0) or 0.0))}",
                ]
            )
        )
    clauses.append(f"sma48={history.get('sma48_side')} age_bucket={'LONG' if int(history.get('sma48_age', 0) or 0) >= 48 else 'SHORT'}")
    clauses.append(f"sma144={history.get('sma144_side')} age_bucket={'LONG' if int(history.get('sma144_age', 0) or 0) >= 144 else 'SHORT'}")
    return clauses


def _candidate_clauses(label: str, view: dict[str, Any]) -> list[str]:
    cand = view.get("candidate") or {}
    setup = view.get("setup_quality") or {}
    macro = view.get("macro_context") or {}
    return [
        f"{label}: side={cand.get('side')} event_type={cand.get('event_type')} episode={cand.get('episode')} horizon={cand.get('horizon')}",
        f"{label}: setup close={setup.get('close_quality_bucket')} risk={setup.get('risk_bucket')} range={setup.get('range_bucket')} wick={setup.get('wick_bucket')} body={setup.get('body_bucket')}",
        f"{label}: macro kimchi={_z(macro.get('kimchi_z'))} dxy={_z(macro.get('dxy_z'))} usdkrw={_z(macro.get('usdkrw_z'))} kimchi_change={'UP' if float(macro.get('kimchi_chg', 0.0) or 0.0) > 0 else 'DOWN_OR_FLAT'}",
    ]


def _competition_clauses(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    ctx = _competition_context(a, b)
    horizon_diff = int(ctx.get("horizon_diff", 0) or 0)
    if horizon_diff > 0:
        horizon = "A_LONGER"
    elif horizon_diff < 0:
        horizon = "B_LONGER"
    else:
        horizon = "SAME"
    return [
        f"competition: same_side={ctx.get('same_side')} same_event_type={ctx.get('same_event_type')} horizon={horizon}",
        f"competition: A_minus_B risk={_move(float(ctx.get('risk_bps_diff_A_minus_B', 0.0) or 0.0) / 10_000.0)} close_quality={_move(float(ctx.get('close_quality_diff_A_minus_B', 0.0) or 0.0))} wick={_move(float(ctx.get('wick_frac_diff_A_minus_B', 0.0) or 0.0))}",
    ]


def _pair_prompt_clauses(date: str, history: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    lines = [
        "You are a BTCUSDT futures candidate ranker.",
        "Read the causal price-action clauses and choose the candidate with higher future path-risk-adjusted utility.",
        "Do not use future returns. Return JSON with keys: choice, confidence, reason. choice must be A or B.",
        f"date: {date}",
        "market_regime:",
    ]
    lines.extend(f"- {clause}" for clause in _market_clauses(history))
    lines.append("candidates:")
    lines.extend(f"- {clause}" for clause in _candidate_clauses("A", a))
    lines.extend(f"- {clause}" for clause in _candidate_clauses("B", b))
    lines.append("comparison:")
    lines.extend(f"- {clause}" for clause in _competition_clauses(a, b))
    return "\n".join(lines)


def _pair_prompt(date: str, history: dict[str, Any], a: dict[str, Any], b: dict[str, Any], style: str) -> str:
    if style == "json":
        return _pair_prompt_json(date, history, a, b)
    if style == "clauses":
        return _pair_prompt_clauses(date, history, a, b)
    raise ValueError(f"unknown prompt_style={style!r}")


def _group(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    g: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[int(r.get("signal_pos", -1))].append(r)
    return g


def _build_split(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: EpisodeSurvivalPairwiseCfg, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for pos, group in _group(rows).items():
        if len(group) < 2 or pos < 0 or pos >= len(market):
            continue
        ordered = sorted(group, key=_utility, reverse=True)
        best = ordered[0]
        best_u = _utility(best)
        pairs = 0
        for loser in ordered[1:]:
            gap = best_u - _utility(loser)
            if gap < float(cfg.min_utility_gap_pct):
                continue
            best_view = _candidate_view(best)
            loser_view = _candidate_view(loser)
            date = str(best.get("date") or market.iloc[pos]["date"])
            history = _history_context(market, pos)

            def append_pair(a: dict[str, Any], b: dict[str, Any], choice: str, swap_index: int) -> None:
                out.append(
                    {
                        "task": "episode_survival_pairwise_preference",
                        "date": date,
                        "signal_pos": pos,
                        "prompt": _pair_prompt(date, history, a, b, str(cfg.prompt_style)),
                        "target": json.dumps({"choice": choice, "confidence": "HIGH", "reason": "higher_future_path_utility"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                        "chosen_candidate": best.get("candidate"),
                        "rejected_candidate": loser.get("candidate"),
                        "chosen_audit": _target_audit(best),
                        "rejected_audit": _target_audit(loser),
                        "utility_gap_pct": round(gap, 6),
                        "pair_augmentation": {
                            "swap_augmented": bool(cfg.augment_swaps),
                            "swap_index": swap_index,
                        },
                        "leakage_guard": {
                            "prompt_uses_future_path": False,
                            "chosen_rejected_use_future_path_for_training_only": True,
                            "candidates_share_same_signal_timestamp": True,
                        },
                    }
                )

            if bool(cfg.augment_swaps):
                append_pair(best_view, loser_view, "A", 0)
                append_pair(loser_view, best_view, "B", 1)
            else:
                flip = rng.random() < 0.5
                a, b = (loser_view, best_view) if flip else (best_view, loser_view)
                choice = "B" if flip else "A"
                append_pair(a, b, choice, 0)
            pairs += 1
            if pairs >= int(cfg.max_pairs_per_signal):
                break
    rng.shuffle(out)
    out = out[: int(cfg.max_rows_per_split)]
    out.sort(key=lambda r: (str(r["date"]), int(r["signal_pos"]), str(r["target"])))
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    choices = Counter(json.loads(r["target"])["choice"] for r in rows)
    gaps = [float(r.get("utility_gap_pct", 0.0) or 0.0) for r in rows]
    chosen_sides = Counter(str((r.get("chosen_candidate") or {}).get("side")) for r in rows)
    return {
        "rows": len(rows),
        "choice_counts": dict(choices),
        "chosen_sides": dict(chosen_sides),
        "mean_utility_gap_pct": float(np.mean(gaps)) if gaps else 0.0,
        "median_utility_gap_pct": float(np.median(gaps)) if gaps else 0.0,
    }


def run(cfg: EpisodeSurvivalPairwiseCfg) -> dict[str, Any]:
    rng = random.Random(int(cfg.seed))
    market = _load_market(cfg.market_csv)
    loaded = {"train": _load(cfg.train_jsonl), "test": _load(cfg.test_jsonl), "eval": _load(cfg.eval_jsonl)}
    out_dir = Path(cfg.output_dir)
    suffix = ".jsonl.gz" if cfg.gzip_output else ".jsonl"
    report = {"config": asdict(cfg), "splits": {}}
    for split, rows in loaded.items():
        pairs = _build_split(rows, market, cfg, rng)
        path = out_dir / f"episode_survival_pairwise_{split}{suffix}"
        _write(path, pairs, bool(cfg.gzip_output))
        report["splits"][split] = {**_summary(pairs), "source_rows": len(rows), "output": str(path)}
    sp = out_dir / "episode_survival_pairwise_summary.json"
    sp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-utility-gap-pct", type=float, default=EpisodeSurvivalPairwiseCfg.min_utility_gap_pct)
    p.add_argument("--max-pairs-per-signal", type=int, default=EpisodeSurvivalPairwiseCfg.max_pairs_per_signal)
    p.add_argument("--max-rows-per-split", type=int, default=EpisodeSurvivalPairwiseCfg.max_rows_per_split)
    p.add_argument("--seed", type=int, default=EpisodeSurvivalPairwiseCfg.seed)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.add_argument("--prompt-style", choices=["json", "clauses"], default=EpisodeSurvivalPairwiseCfg.prompt_style)
    p.add_argument("--augment-swaps", action="store_true", default=EpisodeSurvivalPairwiseCfg.augment_swaps)
    p.set_defaults(gzip_output=EpisodeSurvivalPairwiseCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EpisodeSurvivalPairwiseCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
