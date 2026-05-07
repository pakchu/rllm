"""Past-only regime filter search for hierarchical gate+side reports.

This script deliberately uses the same leakage-safe split contract as
``hierarchical_direct_split_search``: choose parameters on the test/validation
period, then report the untouched eval period.  Regime features are computed
from OHLCV bars at or before each prediction timestamp only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from training.hierarchical_direct_split_search import HierSimConfig, _load_rows, _pair_rows, _norm_cdf
from training.search_significant_cagr_mdd_pool import _pass_relaxed, _pass_strict


@dataclass(frozen=True)
class RegimeFilter:
    name: str
    vol_min: float | None = None
    vol_max: float | None = None
    abs_trend_min: float | None = None
    abs_trend_max: float | None = None
    drawdown_max: float | None = None
    range_min: float | None = None
    range_max: float | None = None
    align_mode: str = "any"  # any | trend_follow | mean_revert
    trend_col: str = "trend_144"


def _load_market_features(csv_path: str) -> dict[str, dict[str, float]]:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"CSV lacks date column: {csv_path}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ret = close.pct_change().fillna(0.0)
    for n in (12, 48, 144, 288):
        df[f"trend_{n}"] = close.pct_change(n).fillna(0.0)
        df[f"vol_{n}"] = ret.rolling(n, min_periods=max(4, n // 4)).std().fillna(0.0) * math.sqrt(n)
        roll_high = high.rolling(n, min_periods=max(4, n // 4)).max()
        roll_low = low.rolling(n, min_periods=max(4, n // 4)).min()
        denom = (roll_high - roll_low).replace(0.0, float("nan"))
        df[f"range_pos_{n}"] = ((close - roll_low) / denom).fillna(0.5).clip(0.0, 1.0)
        peak = close.rolling(n, min_periods=max(4, n // 4)).max()
        df[f"drawdown_{n}"] = (1.0 - close / peak.replace(0.0, float("nan"))).fillna(0.0).clip(lower=0.0)

    cols = [
        "trend_12", "trend_48", "trend_144", "trend_288",
        "vol_48", "vol_144", "vol_288",
        "range_pos_144", "range_pos_288",
        "drawdown_144", "drawdown_288",
    ]
    out: dict[str, dict[str, float]] = {}
    for row in df[["date", *cols]].itertuples(index=False):
        date = getattr(row, "date").strftime("%Y-%m-%d %H:%M:%S")
        out[date] = {c: float(getattr(row, c)) for c in cols}
    return out


def _attach_features(rows: list[dict[str, Any]], features: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        f = features.get(str(row["date"]))
        if f is None:
            continue
        merged = dict(row)
        merged["_features"] = f
        kept.append(merged)
    if not kept:
        raise ValueError("No prediction rows matched market feature dates")
    return kept


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


def _make_filters(test_rows: list[dict[str, Any]]) -> list[RegimeFilter]:
    feats = [r["_features"] for r in test_rows]
    q = {
        key: {p: _quantile([float(f[key]) for f in feats], p) for p in (0.2, 0.35, 0.5, 0.65, 0.8)}
        for key in ("vol_144", "vol_288", "drawdown_144", "range_pos_144")
    }
    filters: list[RegimeFilter] = [RegimeFilter("none")]
    for vol_col, trend_col in (("vol_144", "trend_144"), ("vol_288", "trend_288")):
        filters.extend(
            [
                RegimeFilter(f"{vol_col}_low80", vol_max=q[vol_col][0.8], trend_col=trend_col),
                RegimeFilter(f"{vol_col}_low65", vol_max=q[vol_col][0.65], trend_col=trend_col),
                RegimeFilter(f"{vol_col}_mid20_80", vol_min=q[vol_col][0.2], vol_max=q[vol_col][0.8], trend_col=trend_col),
                RegimeFilter(f"{vol_col}_high35", vol_min=q[vol_col][0.35], trend_col=trend_col),
            ]
        )
    for trend_col in ("trend_48", "trend_144", "trend_288"):
        for thresh in (0.0, 0.0025, 0.005, 0.01, 0.02):
            suffix = str(thresh).replace(".", "p")
            filters.append(RegimeFilter(f"tf_{trend_col}_{suffix}", abs_trend_min=thresh, align_mode="trend_follow", trend_col=trend_col))
            filters.append(RegimeFilter(f"mr_{trend_col}_{suffix}", abs_trend_min=thresh, align_mode="mean_revert", trend_col=trend_col))
    for dd_q in (0.5, 0.65, 0.8):
        filters.append(RegimeFilter(f"dd144_max_q{dd_q}", drawdown_max=q["drawdown_144"][dd_q]))
    filters.extend(
        [
            RegimeFilter("range_bottom65", range_max=q["range_pos_144"][0.65]),
            RegimeFilter("range_top35", range_min=q["range_pos_144"][0.35]),
            RegimeFilter("range_mid20_80", range_min=q["range_pos_144"][0.2], range_max=q["range_pos_144"][0.8]),
            RegimeFilter("lowvol_tf144", vol_max=q["vol_144"][0.65], abs_trend_min=0.0025, align_mode="trend_follow", trend_col="trend_144"),
            RegimeFilter("lowvol_mr144", vol_max=q["vol_144"][0.65], abs_trend_min=0.0025, align_mode="mean_revert", trend_col="trend_144"),
        ]
    )
    return filters


def _row_signal(row: dict[str, Any], cfg: HierSimConfig) -> int:
    if float(row["_gate_margin"]) < cfg.gate_margin_threshold:
        return 0
    if float(row["_side_margin"]) < cfg.side_margin_threshold:
        return 0
    return (-1 if cfg.inverse else 1) if float(row["_side_dir"]) >= 0.0 else (1 if cfg.inverse else -1)


def _regime_ok(row: dict[str, Any], cfg: HierSimConfig, filt: RegimeFilter) -> bool:
    if filt.name == "none":
        return True
    f = row["_features"]
    vol = float(f.get("vol_144", 0.0)) if "vol_144" in filt.name else float(f.get("vol_288", f.get("vol_144", 0.0)))
    if filt.vol_min is not None and vol < filt.vol_min:
        return False
    if filt.vol_max is not None and vol > filt.vol_max:
        return False
    dd = float(f.get("drawdown_144", 0.0))
    if filt.drawdown_max is not None and dd > filt.drawdown_max:
        return False
    rp = float(f.get("range_pos_144", 0.5))
    if filt.range_min is not None and rp < filt.range_min:
        return False
    if filt.range_max is not None and rp > filt.range_max:
        return False
    trend = float(f.get(filt.trend_col, 0.0))
    if filt.abs_trend_min is not None and abs(trend) < filt.abs_trend_min:
        return False
    if filt.abs_trend_max is not None and abs(trend) > filt.abs_trend_max:
        return False
    if filt.align_mode == "any":
        return True
    sig = _row_signal(row, cfg)
    if sig == 0:
        return True
    if trend == 0.0:
        return False
    aligned = (sig > 0 and trend > 0.0) or (sig < 0 and trend < 0.0)
    return aligned if filt.align_mode == "trend_follow" else not aligned


def _simulate_filtered(
    rows: list[dict[str, Any]],
    cfg: HierSimConfig,
    filt: RegimeFilter,
    *,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    eq = peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    step = max(1, int(cfg.hold_bars))
    cooldown_step = max(0, int(cfg.cooldown_bars))
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    i = 0
    while i < len(rows):
        row = rows[i]
        signal = _row_signal(row, cfg)
        if signal == 0 or not _regime_ok(row, cfg, filt):
            i += 1
            continue
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        eq *= max(0.0, 1.0 + float(signal) * float(row.get("next_return", 0.0)) * float(leverage))
        eq *= max(0.0, 1.0 - cost)
        trade_returns.append(eq / entry_eq - 1.0)
        peak = max(peak, eq)
        if peak > 0.0:
            max_dd = max(max_dd, 1.0 - eq / peak)
        i += step + cooldown_step

    n = len(trade_returns)
    mean = sum(trade_returns) / n if n else 0.0
    if n >= 2:
        var = sum((x - mean) ** 2 for x in trade_returns) / (n - 1)
        std = math.sqrt(max(0.0, var))
    else:
        std = 0.0
    se = std / math.sqrt(n) if n else 0.0
    t_like = mean / se if se > 0 else 0.0
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t_like))) if se > 0.0 else 1.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    effect_d = mean / std if std > 1e-12 else 0.0
    n_required = int(math.ceil(((1.959963984540054 + 0.8416212335729143) / abs(effect_d)) ** 2)) if abs(effect_d) > 1e-12 else None
    start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = float((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0 else -100.0
    mdd_pct = max_dd * 100.0
    ratio = cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf")
    return {
        "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": ratio,
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "return_application": "entry_forward_return_non_overlap_regime_filtered",
        },
        "trade_stats": {
            "n_trades": n,
            "mean_trade_ret_pct": mean * 100.0,
            "std_trade_ret_pct": std * 100.0,
            "t_stat_like": t_like,
            "p_value_mean_ret_approx": p_two,
            "ci95_mean_trade_ret_pct": [ci_low * 100.0, ci_high * 100.0],
            "effect_size_d": effect_d,
            "n_required_for_80pct_power_alpha5pct": n_required,
            "n_gap_to_power_rule": max(0, n_required - n) if n_required is not None else None,
        },
    }


def _rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    sim = row["test"]["sim"]
    stats = row["test"]["trade_stats"]
    return (
        1.0 if row["significance"].get("relaxed_pass") else 0.0,
        float(sim["cagr_to_strict_mdd"]),
        float(stats["ci95_mean_trade_ret_pct"][0]),
        float(sim["cagr_pct"]),
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    features = _load_market_features(args.market_csv)
    test_rows = _attach_features(_pair_rows(_load_rows(args.gate_test_file), _load_rows(args.side_test_file)), features)
    eval_rows = _attach_features(_pair_rows(_load_rows(args.gate_eval_file), _load_rows(args.side_eval_file)), features)
    filters = _make_filters(test_rows)
    if args.filter_names:
        wanted = {x.strip() for x in args.filter_names.split(",") if x.strip()}
        filters = [f for f in filters if f.name in wanted]
        if not filters:
            raise ValueError(f"No filters matched --filter-names={args.filter_names}")

    gate_margins = _parse_float_list(args.gate_margins, [0.0, 1.5, 2.0, 3.0, 4.0])
    side_margins = _parse_float_list(args.side_margins, [0.0, 0.5, 1.0, 3.0, 6.0])
    holds = _parse_int_list(args.hold_bars, [144, 288, 432])
    cooldowns = _parse_int_list(args.cooldown_bars, [0, 1, 12, 24])
    leverages = _parse_float_list(args.leverages, [1.0, 1.5, 2.0])

    top: list[dict[str, Any]] = []
    count = 0
    inverse_opts = [x.strip().lower() in {"1", "true", "yes", "y"} for x in args.inverse_opts.split(",") if x.strip()] if args.inverse_opts else [False, True]
    for lev in leverages:
        for inv in inverse_opts:
            for gm in gate_margins:
                for sm in side_margins:
                    for hb in holds:
                        for cd in cooldowns:
                            cfg = HierSimConfig(inv, gm, sm, hb, cd)
                            for filt in filters:
                                count += 1
                                test = _simulate_filtered(test_rows, cfg, filt, leverage=lev, fee_rate=args.fee_rate, slippage_rate=args.slippage_rate)
                                row = {"leverage": lev, "params": cfg.__dict__, "regime_filter": filt.__dict__, "test": test}
                                row["significance"] = {
                                    "relaxed_pass": _pass_relaxed(test, alpha=args.alpha, min_trades=args.min_trades),
                                    "strict_pass": _pass_strict(test, alpha=args.alpha, min_trades=args.min_trades),
                                }
                                if len(top) < args.keep_top or _rank(row) > _rank(top[-1]):
                                    top.append(row)
                                    top.sort(key=_rank, reverse=True)
                                    del top[args.keep_top :]
    for row in top:
        cfg = HierSimConfig(**row["params"])
        filt = RegimeFilter(**row["regime_filter"])
        ev = _simulate_filtered(eval_rows, cfg, filt, leverage=row["leverage"], fee_rate=args.fee_rate, slippage_rate=args.slippage_rate)
        row["eval"] = ev
        row["eval_significance"] = {
            "relaxed_pass": _pass_relaxed(ev, alpha=args.alpha, min_trades=args.min_trades),
            "strict_pass": _pass_strict(ev, alpha=args.alpha, min_trades=args.min_trades),
        }
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "files": vars(args),
        "num_candidates": count,
        "num_filters": len(filters),
        "top": top,
        "leakage_guard": {
            "test_end": str(test_rows[-1]["date"]),
            "eval_start": str(eval_rows[0]["date"]),
            "eval_strictly_after_test": datetime.fromisoformat(str(eval_rows[0]["date"])) > datetime.fromisoformat(str(test_rows[-1]["date"])),
            "features_are_past_only": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    if not raw:
        return default
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past-only regime-filtered hierarchical split search")
    p.add_argument("--gate-test-file", required=True)
    p.add_argument("--side-test-file", required=True)
    p.add_argument("--gate-eval-file", required=True)
    p.add_argument("--side-eval-file", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/hierarchical_regime_filter_search.json")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades", type=int, default=60)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--keep-top", type=int, default=40)
    p.add_argument("--gate-margins", default="", help="Comma-separated override")
    p.add_argument("--side-margins", default="", help="Comma-separated override")
    p.add_argument("--hold-bars", default="", help="Comma-separated override")
    p.add_argument("--cooldown-bars", default="", help="Comma-separated override")
    p.add_argument("--leverages", default="", help="Comma-separated override")
    p.add_argument("--inverse-opts", default="", help="Comma-separated booleans, e.g. false,true")
    p.add_argument("--filter-names", default="", help="Comma-separated exact regime filter names")
    return p.parse_args()


def main() -> None:
    out = run_search(parse_args())
    best = out["top"][0] if out["top"] else None
    print(json.dumps({
        "num_candidates": out["num_candidates"],
        "num_filters": out["num_filters"],
        "best_test": None if best is None else best["test"]["sim"],
        "best_eval": None if best is None else best.get("eval", {}).get("sim"),
        "best_params": None if best is None else best["params"],
        "best_filter": None if best is None else best["regime_filter"],
        "leakage_guard": out["leakage_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
