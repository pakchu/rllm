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
    context_market_csv: str = ""
    micro_windows: str = "12,36,72"


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


def _bucket_z(x: float) -> str:
    if x <= -2.0:
        return "LOW_EXTREME"
    if x <= -1.0:
        return "LOW"
    if x < 1.0:
        return "NEUTRAL"
    if x < 2.0:
        return "HIGH"
    return "HIGH_EXTREME"


def _bucket_small_mom(x: float) -> str:
    if x <= -0.01:
        return "DOWN_EXTREME"
    if x <= -0.003:
        return "DOWN"
    if x < 0.003:
        return "FLAT"
    if x < 0.01:
        return "UP"
    return "UP_EXTREME"


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


def _bucket_count(x: int, *, low: int, high: int) -> str:
    if x <= low:
        return "LOW"
    if x >= high:
        return "HIGH"
    return "MID"


def _bucket_ratio(x: float) -> str:
    if x < 0.50:
        return "LOW"
    if x < 0.90:
        return "BELOW_ONE"
    if x <= 1.10:
        return "BALANCED"
    if x <= 2.0:
        return "ABOVE_ONE"
    return "HIGH"


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def micro_path_tokens(market: pd.DataFrame, signal_pos: int, windows: list[int]) -> tuple[list[str], dict[str, float]]:
    """Past-only recent trajectory descriptors for target/stop path learnability."""
    pos = int(signal_pos)
    close = market["close"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float) if "open" in market.columns else close
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    volume = market["volume"].to_numpy(dtype=float) if "volume" in market.columns else np.ones(len(market), dtype=float)
    toks: list[str] = []
    nums: dict[str, float] = {}
    if pos <= 1 or pos >= len(market):
        return toks, nums
    for w in windows:
        start = max(1, pos - int(w) + 1)
        if pos - start + 1 < max(3, min(int(w), 8)):
            continue
        cs = close[start : pos + 1]
        os = open_[start : pos + 1]
        hs = high[start : pos + 1]
        ls = low[start : pos + 1]
        vs = volume[start : pos + 1]
        prev = close[start - 1 : pos]
        rets = np.divide(cs, np.where(prev == 0, np.nan, prev)) - 1.0
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        signs = [_sign(float(x), eps=0.00005) for x in rets]
        up_count = sum(1 for x in signs if x > 0)
        down_count = sum(1 for x in signs if x < 0)
        flat_count = sum(1 for x in signs if x == 0)
        alternations = sum(1 for a, b in zip(signs, signs[1:]) if a != 0 and b != 0 and a != b)
        same_dir_runs = []
        cur_run = 0
        last = 0
        for sg in signs:
            if sg != 0 and sg == last:
                cur_run += 1
            elif sg != 0:
                cur_run = 1
                last = sg
            else:
                cur_run = 0
                last = 0
            same_dir_runs.append(cur_run)
        max_run = max(same_dir_runs or [0])
        bodies = np.abs(cs - os)
        ranges = np.maximum(1e-12, hs - ls)
        upper_wicks = hs - np.maximum(cs, os)
        lower_wicks = np.minimum(cs, os) - ls
        upper_rej = int(np.sum((upper_wicks / ranges) > 0.45))
        lower_rej = int(np.sum((lower_wicks / ranges) > 0.45))
        wide = int(np.sum((ranges / np.maximum(1e-12, cs)) > np.nanpercentile(ranges / np.maximum(1e-12, cs), 75))) if len(ranges) > 3 else 0
        body_eff = float(np.sum(cs - os) / max(1e-12, np.sum(ranges)))
        realized_vol = float(np.std(rets) * np.sqrt(max(1, len(rets))) * 100.0)
        path_ret = (float(cs[-1]) / float(cs[0]) - 1.0) * 100.0 if float(cs[0]) > 0 else 0.0
        mfe_long = (float(np.max(hs)) / float(cs[0]) - 1.0) * 100.0 if float(cs[0]) > 0 else 0.0
        mae_long = (float(np.min(ls)) / float(cs[0]) - 1.0) * 100.0 if float(cs[0]) > 0 else 0.0
        range_exp = (float(ranges[-1]) / max(1e-12, float(np.mean(ranges))))
        vol_exp = (float(vs[-1]) / max(1e-12, float(np.mean(vs))))
        prefix = f"micro.w{int(w)}"
        vals = {
            "return_pct": path_ret,
            "realized_vol_pct": realized_vol,
            "body_efficiency": body_eff,
            "alternations": float(alternations),
            "max_run": float(max_run),
            "upper_rejections": float(upper_rej),
            "lower_rejections": float(lower_rej),
            "range_expansion": range_exp,
            "volume_expansion": vol_exp,
            "mfe_long_pct": mfe_long,
            "mae_long_pct": mae_long,
        }
        nums.update({f"{prefix}.{k}": float(v) for k, v in vals.items()})
        toks.extend([
            f"{prefix}.return={_bucket_signed_pct(path_ret)}",
            f"{prefix}.vol={_bucket_dist_pct(realized_vol)}",
            f"{prefix}.body_eff={_bucket_z(body_eff * 3.0)}",
            f"{prefix}.up_count={_bucket_count(up_count, low=max(1, int(w)//5), high=max(2, int(w)//2))}",
            f"{prefix}.down_count={_bucket_count(down_count, low=max(1, int(w)//5), high=max(2, int(w)//2))}",
            f"{prefix}.flat_count={_bucket_count(flat_count, low=max(0, int(w)//8), high=max(2, int(w)//3))}",
            f"{prefix}.alternation={_bucket_count(alternations, low=max(1, int(w)//8), high=max(2, int(w)//3))}",
            f"{prefix}.max_run={_bucket_count(max_run, low=1, high=max(3, int(w)//4))}",
            f"{prefix}.upper_rej={_bucket_count(upper_rej, low=0, high=max(2, int(w)//5))}",
            f"{prefix}.lower_rej={_bucket_count(lower_rej, low=0, high=max(2, int(w)//5))}",
            f"{prefix}.range_exp={_bucket_ratio(range_exp)}",
            f"{prefix}.volume_exp={_bucket_ratio(vol_exp)}",
            f"{prefix}.mfe_long={_bucket_signed_pct(mfe_long)}",
            f"{prefix}.mae_long={_bucket_signed_pct(mae_long)}",
        ])
    return toks, nums


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


def _macro_by_date(context_market: pd.DataFrame | None, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if context_market is None or context_market.empty or "date" not in context_market.columns:
        return {}
    ctx = context_market.copy()
    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce")
    keep = [c for c in ["date", "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum", "dxy_available", "kimchi_available", "usdkrw_available"] if c in ctx.columns]
    ctx = ctx[keep].dropna(subset=["date"]).sort_values("date")
    req = pd.DataFrame({"_row": range(len(rows)), "date": pd.to_datetime([r.get("date") for r in rows], errors="coerce")}).sort_values("date")
    merged = pd.merge_asof(req, ctx, on="date", direction="backward")
    out: dict[str, dict[str, Any]] = {}
    for rec in merged.to_dict("records"):
        row_idx = int(rec["_row"])
        vals = {k: v for k, v in rec.items() if k not in {"_row", "date"} and pd.notna(v)}
        out[str(row_idx)] = vals
    return out


def macro_tokens(values: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    toks: list[str] = []
    nums: dict[str, float] = {}
    for name in ("dxy", "kimchi", "usdkrw"):
        avail_key = f"{name}_available" if name != "kimchi" else "kimchi_available"
        if avail_key in values:
            toks.append(f"macro.{name}.available={int(float(values.get(avail_key, 0.0) or 0.0) >= 0.5)}")
    mapping = {
        "dxy_zscore": ("macro.dxy.z", _bucket_z),
        "dxy_momentum": ("macro.dxy.mom", _bucket_small_mom),
        "kimchi_premium_zscore": ("macro.kimchi.z", _bucket_z),
        "kimchi_premium_change": ("macro.kimchi.change", _bucket_small_mom),
        "usdkrw_zscore": ("macro.usdkrw.z", _bucket_z),
        "usdkrw_momentum": ("macro.usdkrw.mom", _bucket_small_mom),
    }
    for src, (tok_name, bucket_fn) in mapping.items():
        if src not in values:
            continue
        val = float(values[src])
        nums[tok_name] = val
        toks.append(f"{tok_name}={bucket_fn(val)}")
    return toks, nums


def augment_row(row: dict[str, Any], market: pd.DataFrame, windows: list[int], macro_values: dict[str, Any] | None = None, micro_windows: list[int] | None = None) -> dict[str, Any]:
    out = dict(row)
    summary, prefix, suffix = _summary_from_prompt(str(row.get("prompt", "")))
    toks, nums = price_action_tokens(market, int(row.get("signal_pos", -1)), windows)
    mtoks, mnums = macro_tokens(macro_values or {})
    micro_toks, micro_nums = micro_path_tokens(market, int(row.get("signal_pos", -1)), micro_windows or [])
    summary = dict(summary)
    summary["augmented_price_action_tokens"] = toks
    summary["augmented_price_action_features"] = nums
    if mtoks or mnums:
        summary["augmented_macro_tokens"] = mtoks
        summary["augmented_macro_features"] = mnums
    if micro_toks or micro_nums:
        summary["augmented_micro_path_tokens"] = micro_toks
        summary["augmented_micro_path_features"] = micro_nums
    out["prompt"] = prefix + json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + suffix
    out["augmentation"] = {"price_action_windows": windows, "micro_windows": micro_windows or [], "past_only": True, "token_count": len(toks) + len(mtoks) + len(micro_toks), "price_action_token_count": len(toks), "macro_token_count": len(mtoks), "micro_path_token_count": len(micro_toks)}
    return out


def augment_file(cfg: AugmentPathShapePACfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.input_jsonl)
    market = pd.read_csv(cfg.market_csv, compression="infer")
    windows = _parse_windows(cfg.windows)
    micro_windows = _parse_windows(cfg.micro_windows)
    context_market = pd.read_csv(cfg.context_market_csv, compression="infer") if cfg.context_market_csv else None
    macro_lookup = _macro_by_date(context_market, rows)
    out = [augment_row(r, market, windows, macro_lookup.get(str(i), {}), micro_windows) for i, r in enumerate(rows)]
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(cfg.output_jsonl).open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    token_counts = Counter(int(r.get("augmentation", {}).get("token_count", 0)) for r in out)
    summaries = [json.loads(r["prompt"].split("Past-only analyzer summary: ", 1)[1].split("\n\nAnalyzer", 1)[0]) for r in out]
    macro_rows = sum(1 for x in summaries if "augmented_macro_tokens" in x)
    micro_rows = sum(1 for x in summaries if "augmented_micro_path_tokens" in x)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(out), "rows_with_macro": macro_rows, "rows_with_micro_path": micro_rows, "token_count_distribution": dict(sorted((str(k), v) for k, v in token_counts.items())), "leakage_guard": {"features_use_bars_at_or_before_signal_pos": True, "macro_context_join_direction": "backward_asof_by_date", "micro_path_uses_bars_at_or_before_signal_pos": True, "targets_unchanged": True}}
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
    p.add_argument("--context-market-csv", default="", help="Optional date-aligned macro/context market CSV for backward-asof DXY/kimchi/usdkrw tokens")
    p.add_argument("--micro-windows", default="12,36,72")
    return p.parse_args()


def main() -> None:
    print(json.dumps(augment_file(AugmentPathShapePACfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
