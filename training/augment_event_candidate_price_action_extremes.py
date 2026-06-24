"""Augment event candidate ranker rows with leakage-safe price-action extreme features."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.price_action_extreme_feature_audit import build_extreme_bar_features


@dataclass(frozen=True)
class AugmentPriceActionExtremesCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    lookbacks: tuple[int, ...] = (36, 72, 144, 288, 576)
    tolerance: str = "5min"
    include_features: tuple[str, ...] = ()
    token_features: tuple[str, ...] = (
        "pa_ext_36_max_high_bar_spread_pct",
        "pa_ext_72_max_high_bar_spread_pct",
        "pa_ext_144_max_high_bar_spread_pct",
        "pa_ext_288_max_high_bar_spread_pct",
        "pa_ext_576_max_high_bar_spread_pct",
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _load_market_features(path: str, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    market = pd.read_csv(path, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    feats = build_extreme_bar_features(market, lookbacks)
    out = pd.concat([market[["date"]], feats], axis=1)
    return out.sort_values("date").reset_index(drop=True)


def _bucket(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    x = float(value)
    if x < 0.0025:
        return "low"
    if x < 0.006:
        return "medium"
    if x < 0.012:
        return "high"
    return "extreme"


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def augment_rows(rows: list[dict[str, Any]], feature_frame: pd.DataFrame, cfg: AugmentPriceActionExtremesCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"rows": 0, "matched_rows": 0, "added_feature_count": 0}
    key = pd.DataFrame({"_row_id": np.arange(len(rows)), "date": pd.to_datetime([r.get("date") for r in rows], errors="raise")})
    merged = pd.merge_asof(
        key.sort_values("date"),
        feature_frame.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(cfg.tolerance),
    ).sort_values("_row_id")
    feature_cols = [c for c in feature_frame.columns if c != "date"]
    if cfg.include_features:
        allowed = set(cfg.include_features)
        feature_cols = [c for c in feature_cols if c in allowed]
    out: list[dict[str, Any]] = []
    matched = 0
    for row, (_, feat_row) in zip(rows, merged.iterrows()):
        new_row = dict(row)
        snap = dict(new_row.get("feature_snapshot", {}) if isinstance(new_row.get("feature_snapshot"), dict) else {})
        toks = dict(new_row.get("state_tokens", {}) if isinstance(new_row.get("state_tokens"), dict) else {})
        any_match = False
        for col in feature_cols:
            val = feat_row.get(col)
            if pd.isna(val):
                continue
            snap[col] = float(val)
            any_match = True
        for col in cfg.token_features:
            val = feat_row.get(col)
            toks[f"tok:{col}"] = _bucket(float(val)) if val is not None and not pd.isna(val) else "missing"
        if any_match:
            matched += 1
        new_row["feature_snapshot"] = dict(sorted(snap.items()))
        new_row["state_tokens"] = dict(sorted(toks.items()))
        lg = dict(new_row.get("leakage_guard", {}) if isinstance(new_row.get("leakage_guard"), dict) else {})
        lg["price_action_extreme_features_backward_asof"] = True
        lg["price_action_extreme_features_use_candles_at_or_before_t"] = True
        new_row["leakage_guard"] = lg
        out.append(new_row)
    summary = {
        "rows": len(rows),
        "matched_rows": matched,
        "match_rate": matched / max(1, len(rows)),
        "added_feature_count": len(feature_cols),
        "token_features": list(cfg.token_features),
        "first_date": rows[0].get("date"),
        "last_date": rows[-1].get("date"),
    }
    return out, summary


def run(cfg: AugmentPriceActionExtremesCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    feature_frame = _load_market_features(cfg.market_csv, cfg.lookbacks)
    out, summary = augment_rows(rows, feature_frame, cfg)
    _write_jsonl(cfg.output_jsonl, out)
    report = {"config": asdict(cfg) | {"lookbacks": list(cfg.lookbacks), "include_features": list(cfg.include_features), "token_features": list(cfg.token_features)}, "summary": summary, "leakage_guard": {"market_feature_join_direction": "backward_asof", "features_use_candles_at_or_before_t": True, "reward_fields_unchanged": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment compressor ranker rows with price-action extreme-bar features")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--lookbacks", default="36,72,144,288,576")
    p.add_argument("--tolerance", default=AugmentPriceActionExtremesCfg.tolerance)
    p.add_argument("--include-features", default="")
    p.add_argument("--token-features", default=",".join(AugmentPriceActionExtremesCfg.token_features))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AugmentPriceActionExtremesCfg(
        input_jsonl=args.input_jsonl,
        market_csv=args.market_csv,
        output_jsonl=args.output_jsonl,
        summary_output=args.summary_output,
        lookbacks=_parse_ints(args.lookbacks),
        tolerance=args.tolerance,
        include_features=_parse_strings(args.include_features),
        token_features=_parse_strings(args.token_features),
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
