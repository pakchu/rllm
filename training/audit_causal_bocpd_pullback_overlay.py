from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.audit_confirmed_pullback_squeeze_live_parity import (
    PRE2024_WINDOWS,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_bocpd_state_gated_alpha import _bucket3, _model_output
from training.search_pullback_premium_overheat_state_machine_alpha import (
    Config,
    FROZEN_CHAMPION,
    _selection_score,
    _schedules_and_stats,
    build_state_masks,
    fit_state_thresholds,
    selection_passes,
    state_feature_frame,
)


OUTPUT = "results/causal_bocpd_pullback_overlay_pre2024_2026-07-15.json"
DOCS_OUTPUT = "docs/causal-bocpd-pullback-overlay-pre2024-2026-07-15.md"


def completed_hour_features(market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(market["date"])
    frame = market.set_index(dates).sort_index()
    quote = pd.to_numeric(frame["quote_asset_volume"], errors="coerce")
    buy = pd.to_numeric(frame["taker_buy_quote"], errors="coerce")
    rule = dict(rule="1h", closed="left", label="right")
    rows = frame["close"].resample(**rule).count()
    hour = pd.DataFrame(
        {
            "close": pd.to_numeric(frame["close"], errors="coerce").resample(**rule).last(),
            "quote": quote.resample(**rule).sum(min_count=12),
            "buy": buy.resample(**rule).sum(min_count=12),
            "rows": rows,
        }
    )
    hour = hour.loc[hour["rows"].eq(12)].copy()
    ret1 = np.log(hour["close"].where(hour["close"] > 0)).diff()
    flow = 2.0 * hour["buy"] / hour["quote"].replace(0.0, np.nan) - 1.0
    return pd.DataFrame(
        {"ret1": ret1, "flow24": flow.rolling(24, min_periods=24).mean()},
        index=hour.index,
    ).replace([np.inf, -np.inf], np.nan)


def exact_hour_map(dates: pd.Series, output: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "position": np.arange(len(dates), dtype=np.int64),
            }
        ),
        output.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("0min"),
    ).sort_values("position")


def build_bocpd_state(mapped: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    primary = mapped["primary"].to_numpy(float)
    short_mass = mapped["short_mass"].to_numpy(float)
    secondary = mapped["secondary"].to_numpy(float)
    value = (
        _bucket3(primary, thresholds["primary_low"], thresholds["primary_high"]) * 4
        + (short_mass >= thresholds["short_mass_high"]).astype(int) * 2
        + (secondary >= thresholds["secondary_high"]).astype(int)
    )
    finite = np.isfinite(primary) & np.isfinite(short_mass) & np.isfinite(secondary)
    return np.where(finite, value, -1)


def trade_net(trade, leverage: float = 0.5, cost: float = 0.0006) -> float:
    return float(
        (1.0 - leverage * cost) ** 2
        * trade.price_factor
        * trade.funding_factor
        - 1.0
    )


def _metric(row: dict) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / "
        f"{row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str, result: dict) -> None:
    top = result["rows"][0]
    base = result["base_selection_stats"]
    stats = top["stats"]
    lines = [
        "# Causal BOCPD pullback overlay audit — 2026-07-15",
        "",
        "## Verdict",
        "",
        "**Rejected.** Four of eight overlays satisfy the absolute pre-2024 gate, "
        "but none beats the frozen pullback premium-overheat comparator on the "
        "same lexicographic selection score. The best overlay therefore is not "
        "incremental alpha, and 2024+ was not opened for this family.",
        "",
        "## Causal contract",
        "",
        "- Hour `H` contains exactly `[H-1h,H)`; the unfinished `HH:00` 5-minute bar is excluded.",
        "- BOCPD output is mapped only to the exact hour boundary; no stale two-hour carry-forward.",
        "- Standardization, state thresholds, and state trade quality use 2020-07 through 2022-12 only.",
        "- Entry, realized funding, 6 bp/notional/side cost, split-contained exits, and strict MDD are inherited unchanged from the frozen state machine.",
        "",
        "Metric format: absolute return / CAGR / strict MDD / CAGR-MDD / trades.",
        "",
        "## Comparator versus best overlay",
        "",
        "| Policy | Train | 2023 selection | Pre-2024 | Score |",
        "|---|---:|---:|---:|---:|",
        f"| Frozen comparator | {_metric(base['train'])} | {_metric(base['select_2023'])} | {_metric(base['pre_2024'])} | `{result['base_score']}` |",
        f"| Best BOCPD overlay | {_metric(stats['train'])} | {_metric(stats['select_2023'])} | {_metric(stats['pre_2024'])} | `{top['score']}` |",
        "",
        "## Best overlay specification",
        "",
        "```json",
        json.dumps(top["spec"], indent=2),
        "```",
        "",
        f"Passing overlay cells: `{result['passing_cells']}/8`; comparator-beating cells: `{result['accepted_cells']}/8`.",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config, *, output: str = OUTPUT, docs_output: str = DOCS_OUTPUT) -> dict:
    market, raw, funding, hashes = _load_bundle(
        cfg, cutoff="2024-01-01", premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw)
    base_active, base_thresholds = _fit_active(features, dates, decisions)
    state_features = state_feature_frame(features)
    execution_thresholds = fit_state_thresholds(state_features, dates, base_active)
    capitulation, overheat = build_state_masks(
        state_features, execution_thresholds, FROZEN_CHAMPION["overheat"]
    )
    base_schedules, base_stats = _schedules_and_stats(
        market,
        funding,
        base_active,
        capitulation,
        overheat,
        cfg,
        overheat_action=FROZEN_CHAMPION["action"],
        windows=PRE2024_WINDOWS,
    )
    if not selection_passes(base_stats):
        raise RuntimeError("frozen base no longer passes pre-2024 selection")

    hourly = completed_hour_features(market)
    fit_hour = np.asarray(
        (hourly.index >= pd.Timestamp("2020-07-01")) & (hourly.index < pd.Timestamp("2023-01-01")),
        dtype=bool,
    )
    rows = []
    for hazard in (168, 336):
        hourly_output, metadata = _model_output(
            hourly,
            fit_hour,
            columns=("ret1", "flow24"),
            secondary_index=1,
            hazard_lambda=hazard,
        )
        mapped = exact_hour_map(dates, hourly_output)
        fit_output = hourly_output[
            (hourly_output["date"] >= pd.Timestamp("2020-07-01"))
            & (hourly_output["date"] < pd.Timestamp("2023-01-01"))
        ]
        for low_q, high_q in ((0.25, 0.75), (0.33, 0.67)):
            thresholds = {
                "primary_low": float(fit_output["primary"].quantile(low_q)),
                "primary_high": float(fit_output["primary"].quantile(high_q)),
                "short_mass_high": float(fit_output["short_mass"].quantile(0.75)),
                "secondary_high": float(fit_output["secondary"].quantile(0.50)),
            }
            states = build_bocpd_state(mapped, thresholds)
            quality = {}
            for trade in base_schedules["train"]:
                key = int(states[trade.signal_position])
                if key >= 0:
                    quality.setdefault(key, []).append(trade_net(trade))
            for min_count in (3, 5):
                allowed = sorted(
                    key
                    for key, returns in quality.items()
                    if len(returns) >= min_count and float(np.mean(returns)) >= 0.0
                )
                active = base_active & np.isin(states, allowed)
                _, stats = _schedules_and_stats(
                    market,
                    funding,
                    active,
                    capitulation,
                    overheat,
                    cfg,
                    overheat_action=FROZEN_CHAMPION["action"],
                    windows=PRE2024_WINDOWS,
                )
                rows.append(
                    {
                        "spec": {
                            "hazard_hours": hazard,
                            "primary_quantiles": [low_q, high_q],
                            "short_mass_quantile": 0.75,
                            "secondary_quantile": 0.50,
                            "min_state_trades": min_count,
                            "min_state_edge": 0.0,
                        },
                        "model": metadata,
                        "state_thresholds": thresholds,
                        "allowed_states": allowed,
                        "state_quality": {
                            str(key): {"n": len(quality[key]), "mean_net": float(np.mean(quality[key]))}
                            for key in allowed
                        },
                        "raw_active": int(active.sum()),
                        "selection_passed": selection_passes(stats),
                        "score": _selection_score(stats),
                        "stats": stats,
                    }
                )
    base_score = _selection_score(base_stats)
    for row in rows:
        row["beats_base"] = bool(tuple(row["score"]) > tuple(base_score))
        row["accepted"] = bool(row["selection_passed"] and row["beats_base"])
    rows.sort(
        key=lambda row: (row["accepted"], row["selection_passed"], *row["score"]),
        reverse=True,
    )
    result = {
        "protocol": {
            "cutoff": "2024-01-01",
            "max_market_time": str(dates.max()),
            "max_hour_label": str(hourly.index.max()),
            "base": "frozen pullback premium-overheat state machine",
            "overlay": "causal BOCPD return+24h taker-flow state veto",
            "hourly_observation": "[H-1h,H) labelled H; exact H map",
            "fit_and_state_quality": "2020-07-01 through 2022-12-31 only",
            "grid_cells": 8,
            "entry_cost_funding_mdd": "unchanged frozen strict-live execution contract",
            "oos_opened": False,
        },
        "hashes": hashes,
        "base_thresholds": base_thresholds,
        "execution_state_thresholds": execution_thresholds,
        "base_selection_stats": base_stats,
        "base_score": base_score,
        "passing_cells": sum(bool(row["selection_passed"]) for row in rows),
        "accepted_cells": sum(bool(row["accepted"]) for row in rows),
        "verdict": "reject_no_incremental_pre2024_alpha",
        "rows": rows,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_docs(docs_output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a live-causal BOCPD veto over the frozen pullback state machine"
    )
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        Config(
            input_csv=args.input_csv,
            funding_csv=args.funding_csv,
            premium_csv=args.premium_csv,
        ),
        output=args.output,
        docs_output=args.docs_output,
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "passing_cells": result["passing_cells"],
                "accepted_cells": result["accepted_cells"],
                "base_score": result["base_score"],
                "top": result["rows"][0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
