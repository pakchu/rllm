"""Augment event candidate ranker rows with past-only rolling market regime features."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AugmentMarketRegimeCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    windows: tuple[int, ...] = (288, 864, 2016, 4032, 8640)
    tolerance: str = "5min"
    token_windows: tuple[int, ...] = (2016, 4032, 8640)


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def build_market_regime_features(market: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    df = market.copy()
    close = df["close"].astype(float) if "close" in df.columns else df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    logret = np.log(open_.clip(lower=1e-12)).diff()
    out: dict[str, pd.Series] = {}
    for w in windows:
        prefix = f"mreg_{int(w)}"
        roll_high = high.rolling(int(w), min_periods=int(w)).max()
        roll_low = low.rolling(int(w), min_periods=int(w)).min()
        roll_peak = open_.rolling(int(w), min_periods=int(w)).max()
        roll_trough = open_.rolling(int(w), min_periods=int(w)).min()
        width = (roll_high - roll_low).replace(0.0, np.nan)
        out[f"{prefix}_ret_pct"] = (open_ / open_.shift(int(w)) - 1.0) * 100.0
        out[f"{prefix}_vol_proxy"] = logret.rolling(int(w), min_periods=int(w)).std() * np.sqrt(288.0)
        out[f"{prefix}_range_pos"] = (close - roll_low) / width
        out[f"{prefix}_drawdown_pct"] = (1.0 - open_ / roll_peak.replace(0.0, np.nan)) * 100.0
        out[f"{prefix}_runup_pct"] = (open_ / roll_trough.replace(0.0, np.nan) - 1.0) * 100.0
        out[f"{prefix}_range_width_pct"] = _safe_div(roll_high - roll_low, close) * 100.0
        out[f"{prefix}_trend_to_vol"] = out[f"{prefix}_ret_pct"] / (out[f"{prefix}_vol_proxy"].replace(0.0, np.nan) * 100.0)
    feats = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df[["date"]], feats], axis=1)


def _load_market_features(path: str, windows: tuple[int, ...]) -> pd.DataFrame:
    market = pd.read_csv(path, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return build_market_regime_features(market, windows).sort_values("date").reset_index(drop=True)


def _bucket_signed(value: float, *, small: float, large: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value <= -large:
        return "strong_down"
    if value <= -small:
        return "down"
    if value < small:
        return "flat"
    if value < large:
        return "up"
    return "strong_up"


def _bucket_range(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 0.2:
        return "lower"
    if value < 0.5:
        return "mid_lower"
    if value < 0.8:
        return "mid_upper"
    return "upper"


def augment_rows(rows: list[dict[str, Any]], feature_frame: pd.DataFrame, cfg: AugmentMarketRegimeCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        any_match = False
        for col in feature_cols:
            val = feat_row.get(col)
            if pd.isna(val):
                continue
            snap[col] = float(val)
            any_match = True
        for w in cfg.token_windows:
            ret = feat_row.get(f"mreg_{int(w)}_ret_pct")
            pos = feat_row.get(f"mreg_{int(w)}_range_pos")
            toks[f"tok:mreg_{int(w)}_ret"] = _bucket_signed(float(ret), small=3.0, large=10.0) if ret is not None and not pd.isna(ret) else "missing"
            toks[f"tok:mreg_{int(w)}_range"] = _bucket_range(float(pos)) if pos is not None and not pd.isna(pos) else "missing"
        if any_match:
            matched += 1
        new_row["feature_snapshot"] = dict(sorted(snap.items()))
        new_row["state_tokens"] = dict(sorted(toks.items()))
        lg = dict(new_row.get("leakage_guard", {}) if isinstance(new_row.get("leakage_guard"), dict) else {})
        lg["market_regime_features_backward_asof"] = True
        lg["market_regime_features_use_candles_at_or_before_t"] = True
        new_row["leakage_guard"] = lg
        out.append(new_row)
    summary = {
        "rows": len(rows),
        "matched_rows": matched,
        "match_rate": matched / max(1, len(rows)),
        "added_feature_count": len(feature_cols),
        "first_date": rows[0].get("date"),
        "last_date": rows[-1].get("date"),
    }
    return out, summary


def run(cfg: AugmentMarketRegimeCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    feature_frame = _load_market_features(cfg.market_csv, cfg.windows)
    out, summary = augment_rows(rows, feature_frame, cfg)
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "config": asdict(cfg) | {"windows": list(cfg.windows), "token_windows": list(cfg.token_windows)},
        "summary": summary,
        "leakage_guard": {"market_feature_join_direction": "backward_asof", "features_use_candles_at_or_before_t": True, "reward_fields_unchanged": True},
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment event candidate rows with rolling market regime features")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--windows", default="288,864,2016,4032,8640")
    p.add_argument("--tolerance", default=AugmentMarketRegimeCfg.tolerance)
    p.add_argument("--token-windows", default="2016,4032,8640")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AugmentMarketRegimeCfg(
        input_jsonl=args.input_jsonl,
        market_csv=args.market_csv,
        output_jsonl=args.output_jsonl,
        summary_output=args.summary_output,
        windows=_parse_ints(args.windows),
        tolerance=args.tolerance,
        token_windows=_parse_ints(args.token_windows),
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
