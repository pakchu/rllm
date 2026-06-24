"""Augment event candidate rows with rolling max/min versus current price features."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AugmentRollingExtremaCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    windows: tuple[int, ...] = (36, 72, 144, 288, 576, 2016, 4032, 8640)
    tolerance: str = "5min"
    token_windows: tuple[int, ...] = (144, 576, 2016, 8640)


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def build_rolling_extrema_features(market: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    df = market.copy()
    close = df["close"].astype(float) if "close" in df.columns else df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    out: dict[str, pd.Series] = {}
    for w_raw in windows:
        w = int(w_raw)
        prefix = f"rex_{w}"
        rmax = high.rolling(w, min_periods=w).max()
        rmin = low.rolling(w, min_periods=w).min()
        width = (rmax - rmin).replace(0.0, np.nan)
        out[f"{prefix}_cur_to_max_pct"] = (close / rmax.replace(0.0, np.nan) - 1.0) * 100.0
        out[f"{prefix}_cur_to_min_pct"] = (close / rmin.replace(0.0, np.nan) - 1.0) * 100.0
        out[f"{prefix}_max_to_cur_pct"] = (rmax / close.replace(0.0, np.nan) - 1.0) * 100.0
        out[f"{prefix}_min_to_cur_pct"] = (rmin / close.replace(0.0, np.nan) - 1.0) * 100.0
        out[f"{prefix}_range_pos"] = (close - rmin) / width
        out[f"{prefix}_range_width_pct"] = width / close.replace(0.0, np.nan) * 100.0
        out[f"{prefix}_upper_gap_over_width"] = (rmax - close) / width
        out[f"{prefix}_lower_gap_over_width"] = (close - rmin) / width
    feats = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df[["date"]], feats], axis=1)


def _load_feature_frame(path: str, windows: tuple[int, ...]) -> pd.DataFrame:
    market = pd.read_csv(path, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return build_rolling_extrema_features(market, windows).sort_values("date").reset_index(drop=True)


def _bucket_range_pos(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 0.15:
        return "near_min"
    if value < 0.35:
        return "lower"
    if value < 0.65:
        return "middle"
    if value < 0.85:
        return "upper"
    return "near_max"


def _bucket_gap(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    x = abs(float(value))
    if x < 0.5:
        return "touching"
    if x < 2.0:
        return "near"
    if x < 6.0:
        return "far"
    return "very_far"


def augment_rows(rows: list[dict[str, Any]], feature_frame: pd.DataFrame, cfg: AugmentRollingExtremaCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    out: list[dict[str, Any]] = []
    matched = 0
    for row, (_, feat_row) in zip(rows, merged.iterrows()):
        new_row = dict(row)
        snap = dict(new_row.get("feature_snapshot", {}) if isinstance(new_row.get("feature_snapshot"), dict) else {})
        toks = dict(new_row.get("state_tokens", {}) if isinstance(new_row.get("state_tokens"), dict) else {})
        side = str(new_row.get("side", "NONE")).upper()
        side_sign = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
        any_match = False
        for col in feature_cols:
            val = feat_row.get(col)
            if pd.isna(val):
                continue
            x = float(val)
            snap[col] = x
            # Explicit side interaction: near rolling max/min can mean different things for long/short.
            snap[f"{col}_x_side"] = x * side_sign
            any_match = True
        for w in cfg.token_windows:
            pos = feat_row.get(f"rex_{int(w)}_range_pos")
            up_gap = feat_row.get(f"rex_{int(w)}_max_to_cur_pct")
            low_gap = feat_row.get(f"rex_{int(w)}_cur_to_min_pct")
            toks[f"tok:rex_{int(w)}_loc"] = _bucket_range_pos(float(pos)) if pos is not None and not pd.isna(pos) else "missing"
            toks[f"tok:rex_{int(w)}_upper_gap"] = _bucket_gap(float(up_gap)) if up_gap is not None and not pd.isna(up_gap) else "missing"
            toks[f"tok:rex_{int(w)}_lower_gap"] = _bucket_gap(float(low_gap)) if low_gap is not None and not pd.isna(low_gap) else "missing"
        if any_match:
            matched += 1
        new_row["feature_snapshot"] = dict(sorted(snap.items()))
        new_row["state_tokens"] = dict(sorted(toks.items()))
        lg = dict(new_row.get("leakage_guard", {}) if isinstance(new_row.get("leakage_guard"), dict) else {})
        lg["rolling_extrema_features_backward_asof"] = True
        lg["rolling_extrema_features_use_candles_at_or_before_t"] = True
        new_row["leakage_guard"] = lg
        out.append(new_row)
    return out, {"rows": len(rows), "matched_rows": matched, "match_rate": matched / max(1, len(rows)), "added_feature_count": len(feature_cols) * 2, "first_date": rows[0].get("date"), "last_date": rows[-1].get("date")}


def run(cfg: AugmentRollingExtremaCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    frame = _load_feature_frame(cfg.market_csv, cfg.windows)
    out, summary = augment_rows(rows, frame, cfg)
    _write_jsonl(cfg.output_jsonl, out)
    report = {"config": asdict(cfg) | {"windows": list(cfg.windows), "token_windows": list(cfg.token_windows)}, "summary": summary, "leakage_guard": {"market_feature_join_direction": "backward_asof", "features_use_candles_at_or_before_t": True, "reward_fields_unchanged": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment event candidates with rolling max/min current-price features")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--windows", default="36,72,144,288,576,2016,4032,8640")
    p.add_argument("--token-windows", default="144,576,2016,8640")
    p.add_argument("--tolerance", default=AugmentRollingExtremaCfg.tolerance)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AugmentRollingExtremaCfg(
        input_jsonl=args.input_jsonl,
        market_csv=args.market_csv,
        output_jsonl=args.output_jsonl,
        summary_output=args.summary_output,
        windows=_parse_ints(args.windows),
        token_windows=_parse_ints(args.token_windows),
        tolerance=args.tolerance,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
