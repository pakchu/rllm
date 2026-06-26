"""Augment path-shape trader prompts with past-only price-action tokens."""
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


@dataclass(frozen=True)
class AugmentPathShapePACfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    windows: str = "36,144,576,2016"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _summary_from_prompt(prompt: str) -> tuple[dict[str, Any], str, str]:
    marker = "Past-only analyzer summary: "
    if marker not in prompt:
        return {}, prompt, ""
    prefix, rest = prompt.split(marker, 1)
    suffix = ""
    raw = rest
    if "\n\nAnalyzer path-shape output:" in rest:
        raw, suffix = rest.split("\n\nAnalyzer path-shape output:", 1)
        suffix = "\n\nAnalyzer path-shape output:" + suffix
    try:
        obj = json.loads(raw.strip())
        return (obj if isinstance(obj, dict) else {}), prefix + marker, suffix
    except Exception:
        return {}, prefix + marker, suffix


def _bucket_signed_pct(x: float) -> str:
    if x <= -5:
        return "<=-5pct"
    if x <= -2:
        return "-5..-2pct"
    if x <= -0.75:
        return "-2..-0.75pct"
    if x < 0.75:
        return "flat"
    if x < 2:
        return "0.75..2pct"
    if x < 5:
        return "2..5pct"
    return ">=5pct"


def _bucket_pos(x: float) -> str:
    if x < 0.15:
        return "BOTTOM"
    if x < 0.35:
        return "LOWER"
    if x < 0.65:
        return "MID"
    if x < 0.85:
        return "UPPER"
    return "TOP"


def _bucket_dist_pct(x: float) -> str:
    if x < 0.25:
        return "TOUCH"
    if x < 0.75:
        return "NEAR"
    if x < 2.0:
        return "MID"
    if x < 5.0:
        return "FAR"
    return "VERY_FAR"


def _bucket_age(x: float) -> str:
    if x < 0.20:
        return "RECENT"
    if x < 0.50:
        return "FRESH"
    if x < 0.80:
        return "OLD"
    return "STALE"


def _parse_windows(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def price_action_tokens(market: pd.DataFrame, signal_pos: int, windows: list[int]) -> tuple[list[str], dict[str, float]]:
    pos = int(signal_pos)
    close = market["close"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    volume = market["volume"].to_numpy(dtype=float) if "volume" in market.columns else np.ones(len(market), dtype=float)
    toks: list[str] = []
    nums: dict[str, float] = {}
    if pos <= 0 or pos >= len(market):
        return toks, nums
    cur = float(close[pos])
    for w in windows:
        start = max(0, pos - int(w) + 1)
        if pos - start + 1 < max(3, min(int(w), 12)):
            continue
        hs = high[start : pos + 1]
        ls = low[start : pos + 1]
        cs = close[start : pos + 1]
        mx = float(np.max(hs))
        mn = float(np.min(ls))
        width = max(1e-12, mx - mn)
        range_pos = (cur - mn) / width
        to_high = (mx / cur - 1.0) * 100.0 if cur > 0 else 0.0
        to_low = (cur / mn - 1.0) * 100.0 if mn > 0 else 0.0
        ret = (cur / float(cs[0]) - 1.0) * 100.0 if float(cs[0]) > 0 else 0.0
        range_pct = width / cur * 100.0 if cur > 0 else 0.0
        max_age = (len(hs) - 1 - int(np.argmax(hs))) / max(1, len(hs) - 1)
        min_age = (len(ls) - 1 - int(np.argmin(ls))) / max(1, len(ls) - 1)
        prefix = f"pa.w{int(w)}"
        vals = {"range_pos": range_pos, "to_high_pct": to_high, "to_low_pct": to_low, "return_pct": ret, "range_pct": range_pct, "max_age_frac": max_age, "min_age_frac": min_age}
        nums.update({f"{prefix}.{k}": float(v) for k, v in vals.items()})
        toks.extend([
            f"{prefix}.range_pos={_bucket_pos(range_pos)}",
            f"{prefix}.to_high={_bucket_dist_pct(to_high)}",
            f"{prefix}.to_low={_bucket_dist_pct(to_low)}",
            f"{prefix}.return={_bucket_signed_pct(ret)}",
            f"{prefix}.range={_bucket_dist_pct(range_pct)}",
            f"{prefix}.max_age={_bucket_age(max_age)}",
            f"{prefix}.min_age={_bucket_age(min_age)}",
        ])
        if len(volume[start : pos + 1]) >= 12:
            vs = volume[start : pos + 1]
            vz = (float(vs[-1]) - float(np.mean(vs))) / max(1e-9, float(np.std(vs)))
            nums[f"{prefix}.volume_z"] = vz
            toks.append(f"{prefix}.volume_z={_bucket_signed_pct(vz)}")
    return toks, nums


def augment_row(row: dict[str, Any], market: pd.DataFrame, windows: list[int]) -> dict[str, Any]:
    out = dict(row)
    summary, prefix, suffix = _summary_from_prompt(str(row.get("prompt", "")))
    toks, nums = price_action_tokens(market, int(row.get("signal_pos", -1)), windows)
    summary = dict(summary)
    summary["augmented_price_action_tokens"] = toks
    summary["augmented_price_action_features"] = nums
    out["prompt"] = prefix + json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + suffix
    out["augmentation"] = {"price_action_windows": windows, "past_only": True, "token_count": len(toks)}
    return out


def augment_file(cfg: AugmentPathShapePACfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.input_jsonl)
    market = pd.read_csv(cfg.market_csv, compression="infer")
    windows = _parse_windows(cfg.windows)
    out = [augment_row(r, market, windows) for r in rows]
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(cfg.output_jsonl).open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    token_counts = Counter(int(r.get("augmentation", {}).get("token_count", 0)) for r in out)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(out), "token_count_distribution": dict(sorted((str(k), v) for k, v in token_counts.items())), "leakage_guard": {"features_use_bars_at_or_before_signal_pos": True, "targets_unchanged": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment path-shape trader prompts with past-only price-action tokens")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--windows", default="36,144,576,2016")
    return p.parse_args()


def main() -> None:
    print(json.dumps(augment_file(AugmentPathShapePACfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
