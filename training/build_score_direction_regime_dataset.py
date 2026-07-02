"""Build fold-level regime records for whether high or low pre-fold scores win.

The family-choice pairwise target was unstable.  This dataset decomposes the
problem: first learn the fold regime that explains whether the target-fold
winner came from high-ranked or low-ranked pre-fold family scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.build_event_candidate_family_clean_pairwise_cards import CleanPairwiseFamilyCardConfig, choose_clean_target


FEATURE_COLUMNS = [
    "range_vol", "trend_12", "trend_24", "trend_96", "sma12_ratio", "sma24_ratio", "sma48_ratio",
    "rsi_norm", "mfi_norm", "bb_z", "range_pos", "window_drawdown",
    "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum",
    "weekly_return_1w", "weekly_return_4w", "weekly_range_pos", "weekly_drawdown_4",
    "htf_4h_return_4", "htf_4h_range_pos", "htf_4h_drawdown_4",
    "htf_1d_return_4", "htf_1d_range_pos", "htf_1d_drawdown_4",
    "htf_1w_return_4", "htf_1w_range_pos", "htf_1w_drawdown_4",
    "rex_144_range_pos", "rex_144_range_width_pct", "rex_576_range_pos", "rex_576_range_width_pct",
    "rex_2016_range_pos", "rex_2016_range_width_pct", "rex_8640_range_pos", "rex_8640_range_width_pct",
]


@dataclass(frozen=True)
class ScoreDirectionRegimeConfig:
    selector_report: str
    market_csv: str
    output_jsonl: str
    split_name: str = "all"
    fold_start: str = ""
    fold_end: str = ""
    max_options: int = 5
    lookback_bars: int = 8640  # about 30d of 5m bars
    high_quantile: float = 0.5
    min_diagnostic_trades: int = 12
    min_diagnostic_ratio: float = 0.25
    max_diagnostic_mdd_pct: float = 25.0


def _num(x: Any, ndigits: int = 6) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    return round(v, ndigits)


def _in_range(fold: dict[str, Any], cfg: ScoreDirectionRegimeConfig) -> bool:
    start = str((fold.get("fold") or {}).get("start", ""))
    if cfg.fold_start and start < cfg.fold_start:
        return False
    if cfg.fold_end and start >= cfg.fold_end:
        return False
    return True


def _load_market_features(path: str):
    import pandas as pd
    from preprocessing.market_features import build_market_feature_frame

    market = pd.read_csv(path)
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    features = build_market_feature_frame(market)
    features["date"] = market["date"]
    return features


def _snapshot(features, fold_start: str, lookback_bars: int) -> dict[str, Any]:
    import pandas as pd

    t = pd.Timestamp(fold_start)
    before = features[features["date"] < t]
    if before.empty:
        return {}
    tail = before.tail(max(1, int(lookback_bars)))
    last = tail.iloc[-1]
    out: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col not in tail.columns:
            continue
        s = tail[col].astype(float).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        out[f"{col}_last"] = _num(last.get(col, 0.0))
        out[f"{col}_mean"] = _num(s.mean())
        out[f"{col}_std"] = _num(s.std(ddof=0))
    return out


def _scoreboard_options(fold: dict[str, Any], max_options: int) -> list[dict[str, Any]]:
    return [{"id": chr(ord("A") + i), "family": row.get("family"), "pre_fold_score": _num(row.get("score"))} for i, row in enumerate((fold.get("pre_fold_scoreboard") or [])[: int(max_options)])]


def _quantile_threshold(values: list[float], q: float) -> float:
    """Return a deterministic linear-interpolated quantile threshold."""
    if not values:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _direction_label(target: dict[str, Any], options: list[dict[str, Any]], high_quantile: float) -> tuple[str, dict[str, Any]]:
    if target.get("family") == "ABSTAIN":
        return "ABSTAIN", {"reason": target.get("reason")}
    scores = sorted(_num(o.get("pre_fold_score")) for o in options if o.get("family") != "ABSTAIN")
    target_opt = next((o for o in options if o.get("family") == target.get("family")), None)
    if not scores or not target_opt:
        return "ABSTAIN", {"reason": "target_not_in_prefold_options"}
    threshold = _quantile_threshold(scores, high_quantile)
    target_score = _num(target_opt.get("pre_fold_score"))
    label = "HIGH_SCORE_WINS" if target_score >= threshold else "LOW_SCORE_WINS"
    return label, {"target_family": target.get("family"), "target_score": target_score, "score_threshold": threshold, "score_quantile": high_quantile, "scores": scores}


def _prompt(row: dict[str, Any]) -> str:
    payload = {
        "fold": row["fold"],
        "market_regime_features": row["features"],
        "scoreboard_summary": row["scoreboard_summary"],
    }
    return "\n".join([
        "Classify the next fold's family-score regime.",
        "Use only market features known before the fold and the pre-fold family scoreboard summary.",
        "Return exactly one JSON object: {\"direction_regime\": \"HIGH_SCORE_WINS\"} or {\"direction_regime\": \"LOW_SCORE_WINS\"} or {\"direction_regime\": \"ABSTAIN\"}.",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ])


def build_records(cfg: ScoreDirectionRegimeConfig) -> list[dict[str, Any]]:
    report = json.loads(Path(cfg.selector_report).read_text())
    features = _load_market_features(cfg.market_csv)
    target_cfg = CleanPairwiseFamilyCardConfig(
        selector_report=cfg.selector_report,
        output_jsonl="",
        max_options=cfg.max_options,
        min_diagnostic_trades=cfg.min_diagnostic_trades,
        min_diagnostic_ratio=cfg.min_diagnostic_ratio,
        max_diagnostic_mdd_pct=cfg.max_diagnostic_mdd_pct,
    )
    rows: list[dict[str, Any]] = []
    for fold in report.get("folds", []):
        if not _in_range(fold, cfg):
            continue
        options = _scoreboard_options(fold, cfg.max_options)
        target = choose_clean_target(fold, options + [{"id": "ABSTAIN", "family": "ABSTAIN", "pre_fold_score": 0.0}], target_cfg)
        label, label_meta = _direction_label(target, options, cfg.high_quantile)
        rec = {
            "split": cfg.split_name,
            "fold": fold.get("fold"),
            "features": _snapshot(features, (fold.get("fold") or {}).get("start", ""), cfg.lookback_bars),
            "scoreboard_summary": {
                "families": [o.get("family") for o in options],
                "scores": [o.get("pre_fold_score") for o in options],
                "max_score": max([o.get("pre_fold_score") for o in options], default=0.0),
                "min_score": min([o.get("pre_fold_score") for o in options], default=0.0),
            },
            "target": json.dumps({"direction_regime": label}, ensure_ascii=False, sort_keys=True),
            "label": {"direction_regime": label},
            "label_meta": label_meta,
            "leakage_guard": {"features_before_fold_start": True, "target_diagnostic_not_in_prompt": True},
        }
        rows.append({**rec, "prompt": _prompt(rec), "completion": rec["target"]})
    return rows


def run(cfg: ScoreDirectionRegimeConfig) -> dict[str, Any]:
    rows = build_records(cfg)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))
    counts: dict[str, int] = {}
    for r in rows:
        k = r["label"]["direction_regime"]
        counts[k] = counts.get(k, 0) + 1
    return {"config": asdict(cfg), "rows": len(rows), "targets": counts, "output_jsonl": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for field in ScoreDirectionRegimeConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        kwargs: dict[str, Any] = {}
        if field.default is MISSING:
            kwargs["required"] = True
        else:
            kwargs["default"] = field.default
        if isinstance(field.default, int):
            kwargs["type"] = int
        elif isinstance(field.default, float):
            kwargs["type"] = float
        p.add_argument(name, **kwargs)
    print(json.dumps(run(ScoreDirectionRegimeConfig(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
