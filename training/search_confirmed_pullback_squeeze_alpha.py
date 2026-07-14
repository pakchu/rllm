"""Confirm the pullback-squeeze candidate with two broad overheat vetoes.

This is the second, frozen interaction stage.  It starts from the causal
pullback-squeeze rule and admits it only when Bollinger displacement is below
the train-event q80 threshold and one-day quote-volume z-score is below q90.
The thresholds are fitted on 2020H2-2022 candidate events only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.search_causal_online_expert_alpha import OnlineExpertConfig, _load_bundle
from training.search_specific_pullback_squeeze_alpha import (
    FINAL_END,
    FIT_END,
    FIT_START,
    WINDOWS,
    PullbackSqueezeConfig,
    _metrics,
    _moving_block_bootstrap,
    _slim,
    _slim_all,
    _window_mask,
    fit_rule_masks,
    simulate_mask,
)


@dataclass(frozen=True)
class ConfirmedPullbackSqueezeConfig(PullbackSqueezeConfig):
    output: str = "results/confirmed_pullback_squeeze_alpha_audit_2026-07-15.json"
    bb_quantile: float = 0.80
    quote_volume_quantile: float = 0.90


def fit_confirmation_masks(
    features: pd.DataFrame,
    dates: pd.Series,
    base_active: np.ndarray,
    *,
    fit_start: pd.Timestamp = FIT_START,
    fit_end: pd.Timestamp = FIT_END,
    bb_quantile: float = 0.80,
    quote_volume_quantile: float = 0.90,
) -> dict[str, Any]:
    """Fit broad train-event vetoes and return individual/combined masks."""

    required = {"bb_z", "quote_vol_z_1d"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing confirmation features: {sorted(missing)}")
    if len(features) != len(dates) or len(base_active) != len(features):
        raise ValueError("features, dates and base_active must have equal length")
    fit_events = base_active & _window_mask(dates, fit_start, fit_end)

    def fit(name: str, quantile: float) -> tuple[np.ndarray, float]:
        values = pd.to_numeric(features[name], errors="coerce").to_numpy(float)
        reference = values[fit_events & np.isfinite(values)]
        if len(reference) < 50:
            raise ValueError(f"insufficient candidate events for {name}: {len(reference)}")
        return values, float(np.quantile(reference, quantile))

    bb, bb_threshold = fit("bb_z", bb_quantile)
    quote_volume, quote_volume_threshold = fit("quote_vol_z_1d", quote_volume_quantile)
    bb_gate = np.isfinite(bb) & (bb <= bb_threshold)
    quote_volume_gate = np.isfinite(quote_volume) & (quote_volume <= quote_volume_threshold)
    return {
        "thresholds": {
            "bb_z_q": bb_threshold,
            "quote_vol_z_1d_q": quote_volume_threshold,
            "bb_quantile": float(bb_quantile),
            "quote_volume_quantile": float(quote_volume_quantile),
        },
        "bb_only": base_active & bb_gate,
        "quote_volume_only": base_active & quote_volume_gate,
        "active": base_active & bb_gate & quote_volume_gate,
    }


def run(cfg: ConfirmedPullbackSqueezeConfig) -> dict[str, Any]:
    loader_cfg = OnlineExpertConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output=cfg.output,
        manifest_output="",
        docs_output="",
        exclude_from=cfg.exclude_from,
        window_size=cfg.window_size,
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    market, features, source_prefix_hashes = _load_bundle(loader_cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    decision = np.zeros(len(market), dtype=bool)
    decision[np.arange(max(143, cfg.window_size - 1), len(market), cfg.stride_bars)] = True
    base = fit_rule_masks(
        features,
        dates,
        decision,
        range_quantile=cfg.range_quantile,
        overheat_quantile=cfg.overheat_quantile,
    )
    confirmation = fit_confirmation_masks(
        features,
        dates,
        base["active"],
        bb_quantile=cfg.bb_quantile,
        quote_volume_quantile=cfg.quote_volume_quantile,
    )
    active = confirmation["active"]
    metrics = _metrics(market, dates, active, cfg)

    stress = {
        "cost_8bp_side": _slim_all(_metrics(market, dates, active, cfg, slippage_rate=0.0003)),
        "cost_10bp_side": _slim_all(_metrics(market, dates, active, cfg, slippage_rate=0.0005)),
        "entry_lag_2_bars": _slim_all(_metrics(market, dates, active, cfg, entry_delay_bars=2)),
        "entry_lag_3_bars": _slim_all(_metrics(market, dates, active, cfg, entry_delay_bars=3)),
        "hold_432": _slim_all(_metrics(market, dates, active, cfg, hold_bars=432)),
        "hold_720": _slim_all(_metrics(market, dates, active, cfg, hold_bars=720)),
    }
    sensitivity = []
    for bb_quantile in (0.75, 0.80, 0.85):
        for quote_volume_quantile in (0.85, 0.90, 0.95):
            variant = fit_confirmation_masks(
                features,
                dates,
                base["active"],
                bb_quantile=bb_quantile,
                quote_volume_quantile=quote_volume_quantile,
            )
            sensitivity.append(
                {
                    "bb_quantile": bb_quantile,
                    "quote_volume_quantile": quote_volume_quantile,
                    "thresholds": variant["thresholds"],
                    "metrics": _slim_all(_metrics(market, dates, variant["active"], cfg)),
                }
            )

    filter_ablation = {
        "unconfirmed_pullback_squeeze": _slim_all(_metrics(market, dates, base["active"], cfg)),
        "bb_only": _slim_all(_metrics(market, dates, confirmation["bb_only"], cfg)),
        "quote_volume_only": _slim_all(_metrics(market, dates, confirmation["quote_volume_only"], cfg)),
        "both": _slim_all(metrics),
    }
    source_ablation = {
        "funding_only": _slim_all(_metrics(market, dates, active & base["funding_active"], cfg)),
        "premium_only": _slim_all(_metrics(market, dates, active & base["premium_active"], cfg)),
    }
    leverage_sweep = {
        f"leverage_{leverage:g}": _slim_all(
            _metrics(market, dates, active, replace(cfg, leverage=leverage))
        )
        for leverage in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00)
    }
    operating_cfg = replace(cfg, leverage=0.90)
    operating_point_stress = {
        "base_6bp_side": _slim_all(_metrics(market, dates, active, operating_cfg)),
        "cost_10bp_side": _slim_all(
            _metrics(market, dates, active, operating_cfg, slippage_rate=0.0005)
        ),
        "entry_lag_3_bars": _slim_all(
            _metrics(market, dates, active, operating_cfg, entry_delay_bars=3)
        ),
        "cost_10bp_and_lag_3": _slim_all(
            _metrics(
                market,
                dates,
                active,
                operating_cfg,
                entry_delay_bars=3,
                slippage_rate=0.0005,
            )
        ),
    }
    bootstrap = {
        name: _moving_block_bootstrap(
            metrics[name]["trade_returns"],
            samples=cfg.bootstrap_samples,
        )
        for name in ("train", "select2023", "test2024", "eval2025_2026", "oos2024_2026", "full")
    }
    quarterly: dict[str, dict[str, Any]] = {}
    for year in (2024, 2025, 2026):
        for quarter in range(1, 5):
            month = 1 + 3 * (quarter - 1)
            start = pd.Timestamp(year=year, month=month, day=1)
            end = pd.Timestamp(year=year + 1, month=1, day=1) if quarter == 4 else pd.Timestamp(year=year, month=month + 3, day=1)
            end = min(end, FINAL_END)
            if start < end:
                quarterly[f"{year}Q{quarter}"] = _slim(
                    simulate_mask(
                        market,
                        dates,
                        active,
                        start=start,
                        end=end,
                        hold_bars=cfg.hold_bars,
                        entry_delay_bars=cfg.entry_delay_bars,
                        leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate,
                        slippage_rate=cfg.slippage_rate,
                    )
                )

    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "base_and_confirmation_fit": "2020-07-01 through 2022-12-31 only",
            "selection": "2023 only",
            "frozen_family": "Top-20 before 2024 replay",
            "pre2024_family_rank": 9,
            "entry": "completed t signal, t+1 open",
            "hold_bars": cfg.hold_bars,
            "leverage": cfg.leverage,
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "strict_mdd": "pre-entry high water plus favorable-then-adverse intrabar marking",
            "period_exit_purge": True,
            "warning": "1,280 fourth-stage interactions, 29,133 prior interactions, and broader OOS exposure; retrospective shadow evidence",
        },
        "source_prefix_hashes": source_prefix_hashes,
        "rule": {
            "base_thresholds": base["base_thresholds"],
            "pullback_context_thresholds": base["context_thresholds"],
            "confirmation_thresholds": confirmation["thresholds"],
            "logic": "pullback_squeeze & bb_z<=train_event_q80 & quote_vol_z_1d<=train_event_q90",
            "availability_required": True,
        },
        "metrics": _slim_all(metrics),
        "stress": stress,
        "threshold_sensitivity": sensitivity,
        "filter_ablation": filter_ablation,
        "source_ablation": source_ablation,
        "leverage_sweep": leverage_sweep,
        "operating_point": {
            "leverage": 0.90,
            "reason": "highest tested leverage that keeps baseline and isolated 10bp/side full-period strict MDD below 15% while exceeding 50% CAGR",
            "stress": operating_point_stress,
        },
        "quarterly": quarterly,
        "moving_block_bootstrap": bootstrap,
        "multiplicity": {
            "first_interaction_stage": 29_133,
            "confirmation_stage": 1_280,
            "frozen_family": 20,
            "interpretation": "backtest p-values are descriptive after search; fresh forward evidence is required",
        },
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=ConfirmedPullbackSqueezeConfig.input_csv)
    parser.add_argument("--funding-csv", default=ConfirmedPullbackSqueezeConfig.funding_csv)
    parser.add_argument("--premium-csv", default=ConfirmedPullbackSqueezeConfig.premium_csv)
    parser.add_argument("--output", default=ConfirmedPullbackSqueezeConfig.output)
    parser.add_argument("--exclude-from", default=ConfirmedPullbackSqueezeConfig.exclude_from)
    parser.add_argument("--window-size", type=int, default=ConfirmedPullbackSqueezeConfig.window_size)
    parser.add_argument("--hold-bars", type=int, default=ConfirmedPullbackSqueezeConfig.hold_bars)
    parser.add_argument("--stride-bars", type=int, default=ConfirmedPullbackSqueezeConfig.stride_bars)
    parser.add_argument("--entry-delay-bars", type=int, default=ConfirmedPullbackSqueezeConfig.entry_delay_bars)
    parser.add_argument("--leverage", type=float, default=ConfirmedPullbackSqueezeConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=ConfirmedPullbackSqueezeConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=ConfirmedPullbackSqueezeConfig.slippage_rate)
    parser.add_argument("--range-quantile", type=float, default=ConfirmedPullbackSqueezeConfig.range_quantile)
    parser.add_argument("--overheat-quantile", type=float, default=ConfirmedPullbackSqueezeConfig.overheat_quantile)
    parser.add_argument("--bb-quantile", type=float, default=ConfirmedPullbackSqueezeConfig.bb_quantile)
    parser.add_argument("--quote-volume-quantile", type=float, default=ConfirmedPullbackSqueezeConfig.quote_volume_quantile)
    parser.add_argument("--bootstrap-samples", type=int, default=ConfirmedPullbackSqueezeConfig.bootstrap_samples)
    return parser.parse_args()


def main() -> None:
    result = run(ConfirmedPullbackSqueezeConfig(**vars(parse_args())))
    summary = {
        name: {
            key: result["metrics"][name][key]
            for key in ("absolute_return_pct", "cagr_pct", "strict_mdd_pct", "cagr_to_strict_mdd", "trade_count")
        }
        for name in ("train", "select2023", "test2024", "eval2025_2026", "oos2024_2026", "full")
    }
    print(json.dumps({"output": result["config"]["output"], "rule": result["rule"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
