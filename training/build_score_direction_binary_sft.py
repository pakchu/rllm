"""Build compact binary SFT rows for trusting high vs low pre-fold family scores.

This is a smaller, more LLM-shaped view of score-direction regime labels.  It
excludes ABSTAIN rows and converts numeric market snapshots into qualitative
buckets so the model can reason over regime descriptors instead of memorizing a
large numeric JSON blob.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoreDirectionBinarySFTConfig:
    input_jsonl: str
    output_jsonl: str
    split_name: str = "all"


FEATURES = {
    "long_range_width": "rex_8640_range_width_pct_last",
    "mid_range_width": "rex_2016_range_width_pct_last",
    "short_range_width": "rex_144_range_width_pct_last",
    "long_range_position": "rex_8640_range_pos_last",
    "mid_range_position": "rex_2016_range_pos_last",
    "bb_location": "bb_z_last",
    "bb_location_30d": "bb_z_mean",
    "trend_4h": "htf_4h_return_4_last",
    "trend_1d": "htf_1d_return_4_last",
    "trend_1w": "htf_1w_return_4_last",
    "weekly_4w": "weekly_return_4w_last",
    "drawdown_1w": "htf_1w_drawdown_4_last",
    "kimchi_z": "kimchi_premium_zscore_last",
    "dxy_z": "dxy_zscore_last",
    "usdkrw_z": "usdkrw_zscore_last",
}


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _label(row: dict[str, Any]) -> str:
    obj = row.get("label", row.get("target"))
    if isinstance(obj, str):
        obj = json.loads(obj)
    return str((obj or {}).get("direction_regime", ""))


def _num(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v != v or v in (float("inf"), float("-inf")) else v


def _bucket(name: str, value: float) -> str:
    if "range_width" in name:
        return "very_wide" if value >= 0.35 else "wide" if value >= 0.20 else "normal" if value >= 0.08 else "compressed"
    if "range_position" in name:
        return "upper" if value >= 0.67 else "middle" if value >= 0.33 else "lower"
    if name in {"bb_location", "bb_location_30d", "kimchi_z", "dxy_z", "usdkrw_z"}:
        return "high" if value >= 1.0 else "positive" if value >= 0.2 else "neutral" if value > -0.2 else "negative" if value > -1.0 else "low"
    if "drawdown" in name:
        return "deep" if value >= 0.08 else "moderate" if value >= 0.03 else "shallow"
    return "strong_up" if value >= 0.08 else "up" if value >= 0.015 else "flat" if value > -0.015 else "down" if value > -0.08 else "strong_down"


def _feature_summary(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for alias, key in FEATURES.items():
        v = _num(features.get(key))
        out[alias] = {"bucket": _bucket(alias, v), "value": round(v, 6)}
    return out


def _scoreboard_summary(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("scoreboard_summary") or {}
    fams = list(raw.get("families") or [])
    scores = [_num(x) for x in (raw.get("scores") or [])]
    pairs = [{"rank": i + 1, "family": fam, "score": round(scores[i], 6) if i < len(scores) else 0.0} for i, fam in enumerate(fams)]
    finite_scores = [s for s in scores if s > -1e8]
    return {
        "ranked_families": pairs,
        "score_spread": round((max(finite_scores) - min(finite_scores)) if finite_scores else 0.0, 6),
        "has_dead_options": any(s <= -1e8 for s in scores),
    }


def _prompt(row: dict[str, Any]) -> str:
    payload = {
        "fold": row.get("fold"),
        "market_regime_buckets": _feature_summary(row.get("features") or {}),
        "pre_fold_scoreboard": _scoreboard_summary(row),
    }
    return "\n".join([
        "Decide whether the coming fold should trust the high-ranked family scores or invert toward lower-ranked families.",
        "Use only pre-fold market-regime buckets and the pre-fold family scoreboard.",
        "Return exactly one JSON object: {\"trust_score_rank\": \"HIGH\"} or {\"trust_score_rank\": \"LOW\"}.",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ])


def build_rows(cfg: ScoreDirectionBinarySFTConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _load(cfg.input_jsonl):
        direction = _label(row)
        if direction not in {"HIGH_SCORE_WINS", "LOW_SCORE_WINS"}:
            continue
        target = {"trust_score_rank": "HIGH" if direction == "HIGH_SCORE_WINS" else "LOW"}
        rows.append({
            "task": "score_direction_binary",
            "split": cfg.split_name,
            "fold": row.get("fold"),
            "source_direction_regime": direction,
            "prompt": _prompt(row),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True),
            "completion": json.dumps(target, ensure_ascii=False, sort_keys=True),
            "leakage_guard": row.get("leakage_guard", {}),
        })
    return rows


def run(cfg: ScoreDirectionBinarySFTConfig) -> dict[str, Any]:
    rows = build_rows(cfg)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))
    counts: dict[str, int] = {}
    for row in rows:
        val = json.loads(row["target"])["trust_score_rank"]
        counts[val] = counts.get(val, 0) + 1
    return {"config": asdict(cfg), "rows": len(rows), "targets": counts, "output_jsonl": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--split-name", default="all")
    print(json.dumps(run(ScoreDirectionBinarySFTConfig(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
