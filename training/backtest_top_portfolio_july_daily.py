"""Daily July backtest for the current top live portfolio config."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training.search_portfolio_gross20_step005_mdd20_with_dynamic as top

CONFIG = Path("configs/live/portfolio_gross610_dynamic_top1_2026-07-08.json")
OUT = Path("results/top_portfolio_gross610_daily_july_2026-07-10.json")
DOC = Path("docs/top-portfolio-gross610-daily-july-2026-07-10.md")


def _patch_july_prep() -> None:
    import pandas as pd
    from training.evaluate_oi_llm_selector import SelectorCfg, _load_market_with_oi, _feature_frame

    splits = {
        "train": ("2020-09-01", "2024-01-01"),
        "test2024": ("2024-01-01", "2025-01-01"),
        "eval2025": ("2025-01-01", "2026-01-01"),
        "ytd2026": ("2026-01-01", "2026-08-01"),
    }
    top.ep.SPLITS.clear()
    top.ep.SPLITS.update(splits)

    def patched_prep():
        scfg = SelectorCfg(
            market_csv="data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz",
            exclude_from="2026-08-01",
        )
        market = _load_market_with_oi(scfg)
        market["date"] = pd.to_datetime(market["date"])
        feat = _feature_frame(market, 144)
        dates = pd.to_datetime(market["date"])
        masks = {k: top.ep._split_mask(dates, a, b) for k, (a, b) in splits.items()}
        years = {k: max(1 / 365.25, (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / (365.25 * 24 * 3600)) for k, (a, b) in splits.items()}
        interest = top.ep.build_interest_features(market, feat)
        raw = top.ep._build_score_frame(market, feat, interest)
        score, _ = top.ep._score_variant(raw, masks["train"], "activity_flow_htf")
        feat["activity_flow_htf"] = score
        return market, feat, masks, years

    top.ep._prep = patched_prep


def _strict_path_metrics(r: np.ndarray, adv: np.ndarray) -> dict[str, float]:
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for rr, aa in zip(r, adv):
        mdd = max(mdd, 1.0 - (eq * max(0.0, 1.0 + float(aa))) / max(peak, 1e-12))
        eq *= max(0.0, 1.0 + float(rr))
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / max(peak, 1e-12))
    return {"return_pct": (eq - 1.0) * 100.0, "strict_mdd_pct": mdd * 100.0, "end_equity": eq}


def main() -> None:
    _patch_july_prep()
    cfg = json.loads(CONFIG.read_text())
    weights = {str(k): float(v) for k, v in cfg["weights"].items() if float(v) != 0.0}

    market, feat, masks, years, events, _ = top.vw.build_events()
    top.add_old_live_events(events, market, feat, masks)
    events.extend(top.dx.build_dynamic_sleeves(market, feat, masks, years))

    dates = pd.to_datetime(market["date"])
    n = len(market)
    r_total = np.zeros(n, dtype=float)
    adv_total = np.zeros(n, dtype=float)
    sleeve_r: dict[str, np.ndarray] = {s: np.zeros(n, dtype=float) for s in weights}
    sleeve_adv: dict[str, np.ndarray] = {s: np.zeros(n, dtype=float) for s in weights}
    entry_counts: dict[str, Counter[str]] = defaultdict(Counter)
    active_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    active_event_entry_dates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for e in events:
        sleeve = str(e.get("sleeve"))
        w = weights.get(sleeve, 0.0)
        if not w:
            continue
        rr = np.asarray(e["ret"], dtype=float) * w
        aa = np.asarray(e["adv"], dtype=float) * w
        r_total += rr
        adv_total += aa
        sleeve_r[sleeve] += rr
        sleeve_adv[sleeve] += aa
        p = int(e.get("pos", -1))
        entry_day = str(dates.iloc[p].date()) if 0 <= p < n else "unknown"
        if 0 <= p < n:
            day = str(dates.iloc[p].date())
            if day.startswith("2026-07"):
                entry_counts[day][sleeve] += 1
        nz = np.flatnonzero((np.abs(rr) > 1e-12) | (np.abs(aa) > 1e-12))
        if len(nz):
            for day in sorted(set(str(x.date()) for x in dates.iloc[nz] if str(x.date()).startswith("2026-07"))):
                active_event_counts[day][sleeve] += 1
                active_event_entry_dates[day].append({"sleeve": sleeve, "entry_date": entry_day})

    july_mask = (dates >= pd.Timestamp("2026-07-01")) & (dates < pd.Timestamp("2026-08-01"))
    july_idx = np.flatnonzero(july_mask.to_numpy())
    if len(july_idx) == 0:
        raise SystemExit("No July 2026 bars found")

    daily = []
    for day, idx in pd.Series(july_idx, index=dates.iloc[july_idx].dt.date).groupby(level=0):
        ii = idx.to_numpy(dtype=int)
        met = _strict_path_metrics(r_total[ii], adv_total[ii])
        sleeve_returns = {s: _strict_path_metrics(sleeve_r[s][ii], sleeve_adv[s][ii])["return_pct"] for s in weights}
        daily.append({
            "date": str(day),
            "return_pct": met["return_pct"],
            "strict_mdd_pct": met["strict_mdd_pct"],
            "end_equity": met["end_equity"],
            "new_entries": int(sum(entry_counts[str(day)].values())),
            "sleeve_new_entry_counts": dict(entry_counts[str(day)]),
            "active_events": int(sum(active_event_counts[str(day)].values())),
            "sleeve_active_event_counts": dict(active_event_counts[str(day)]),
            "active_event_entry_dates": active_event_entry_dates[str(day)],
            "sleeve_return_pct": sleeve_returns,
            "active_bars": int(np.count_nonzero(np.abs(r_total[ii]) > 1e-12)),
        })

    all_idx = july_idx
    total = _strict_path_metrics(r_total[all_idx], adv_total[all_idx])
    total["new_entries"] = int(sum(sum(c.values()) for c in entry_counts.values()))
    total["active_bars"] = int(np.count_nonzero(np.abs(r_total[all_idx]) > 1e-12))
    total["active_events"] = int(sum(sum(c.values()) for c in active_event_counts.values()))
    total["sleeve_new_entry_counts"] = dict(sum((Counter(c) for c in entry_counts.values()), Counter()))
    total["sleeve_active_event_counts"] = dict(sum((Counter(c) for c in active_event_counts.values()), Counter()))
    total["sleeve_return_pct"] = {s: _strict_path_metrics(sleeve_r[s][all_idx], sleeve_adv[s][all_idx])["return_pct"] for s in weights}

    report: dict[str, Any] = {
        "config": str(CONFIG),
        "portfolio_name": cfg.get("name"),
        "weights": weights,
        "gross_weight": float(sum(weights.values())),
        "cost_each_side": top.COST,
        "data_start": str(dates.min()),
        "data_end": str(dates.max()),
        "july_bar_start": str(dates.iloc[int(july_idx[0])]),
        "july_bar_end": str(dates.iloc[int(july_idx[-1])]),
        "total": total,
        "daily": daily,
        "notes": [
            "Uses the same event construction as search_portfolio_gross20_step005_mdd20_with_dynamic.py.",
            "Daily strict MDD includes in-day adverse excursion arrays from each weighted sleeve.",
            "new_entries are entry counts by entry date; active_events are positions/events contributing PnL or adverse excursion on that date.",
            "PnL is mark-to-path return contribution grouped by bar date, so July can show PnL from pre-July carryover positions.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    md = [
        "# Top portfolio gross6.10 July daily backtest (2026-07-10)",
        "",
        f"Config: `{CONFIG}`",
        f"Portfolio: `{cfg.get('name')}`",
        f"Weights: `{weights}` gross={sum(weights.values()):.2f}",
        f"Cost: `{top.COST}` per side",
        f"Data: `{report['data_start']}` .. `{report['data_end']}`; July bars `{report['july_bar_start']}` .. `{report['july_bar_end']}`",
        "",
        "## Total July window",
        "",
        f"- return: {total['return_pct']:.4f}%",
        f"- strict MDD: {total['strict_mdd_pct']:.4f}%",
        f"- new entries: {total['new_entries']}",
        f"- active events: {total['active_events']}",
        f"- active bars: {total['active_bars']}",
        f"- sleeve new entries: `{total['sleeve_new_entry_counts']}`",
        f"- sleeve active events: `{total['sleeve_active_event_counts']}`",
        f"- sleeve return pct: `{ {k: round(v,4) for k,v in total['sleeve_return_pct'].items()} }`",
        "",
        "## Daily",
        "",
        "| date | return % | strict MDD % | new entries | active events | active bars | sleeve active events | sleeve returns % |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in daily:
        md.append(
            f"| {row['date']} | {row['return_pct']:.4f} | {row['strict_mdd_pct']:.4f} | {row['new_entries']} | {row['active_events']} | {row['active_bars']} | `{row['sleeve_active_event_counts']}` | `{ {k: round(v,4) for k,v in row['sleeve_return_pct'].items()} }` |"
        )
    DOC.write_text("\n".join(md) + "\n")
    print(json.dumps({"output": str(OUT), "doc": str(DOC), "data_end": report["data_end"], "total": total, "daily": daily}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
