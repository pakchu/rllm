"""Train/test/eval validator for sparse setup candidate reports.

Candidate discovery is expected to happen before this script, typically on train
folds only.  This validator then sweeps selector configs on test folds and reports
the untouched eval result of the best test-selected config.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from training.sparse_setup_ensemble_audit import EnsembleCfg, _candidate_events, _candidate_key, _load_market, _simulate_events
from training.sparse_setup_walkforward_selector import WalkForwardCfg, _ensemble_cfg, _features, _filter_events, _select_for_history


@dataclass(frozen=True)
class TTEValidatorCfg:
    train_sparse_report: str
    market_csv: str
    output: str
    train_folds_json: str
    test_folds_json: str
    eval_folds_json: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    candidate_limits: tuple[int, ...] = (4, 8, 12, 24)
    ensemble_sizes: tuple[int, ...] = (1, 2, 4)
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_same_bar_signals: int = 1
    min_trades: int = 30
    trade_stop_loss_pct: float = 6.0
    trade_take_profit_pct: float = 6.0
    atr_trailing_stop_mult: float = 0.0
    atr_period: int = 45
    rolling_window_trades: int = 0
    rolling_loss_stop_pct: float = 0.0
    pause_bars: int = 288
    setup_sizing: str = "prior_sharpe"
    min_position_scale: float = 0.35
    max_position_scale: float = 1.0
    execution_horizon_bars: int = 0
    min_history_folds: int = 1
    seed_candidates: str = "top_prior"
    include_external_components: bool = False
    include_price_action_extremes: bool = False
    price_action_lookbacks: str = "36,72,144,288,576,2016"
    min_test_trades: int = 20


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _load_folds(raw: str) -> list[dict[str, str]]:
    folds = json.loads(raw)
    if not isinstance(folds, list):
        raise ValueError("fold json must be a list")
    return sorted([{k: str(v) for k, v in f.items()} for f in folds], key=lambda f: f["eval_start"])


def _fold_names(folds: list[dict[str, str]]) -> list[str]:
    return [str(f["name"]) for f in folds]


def _wf_cfg(cfg: TTEValidatorCfg, *, candidate_limit: int, ensemble_size: int) -> WalkForwardCfg:
    return WalkForwardCfg(
        sparse_report=cfg.train_sparse_report,
        market_csv=cfg.market_csv,
        output=cfg.output,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
        window_size=cfg.window_size,
        candidate_limit=int(candidate_limit),
        max_ensemble_size=int(ensemble_size),
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        cooldown_bars=cfg.cooldown_bars,
        max_same_bar_signals=cfg.max_same_bar_signals,
        min_trades=cfg.min_trades,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
        atr_trailing_stop_mult=cfg.atr_trailing_stop_mult,
        atr_period=cfg.atr_period,
        rolling_window_trades=cfg.rolling_window_trades,
        rolling_loss_stop_pct=cfg.rolling_loss_stop_pct,
        pause_bars=cfg.pause_bars,
        setup_sizing=cfg.setup_sizing,
        min_position_scale=cfg.min_position_scale,
        max_position_scale=cfg.max_position_scale,
        execution_horizon_bars=cfg.execution_horizon_bars,
        min_history_folds=cfg.min_history_folds,
        seed_candidates=cfg.seed_candidates,
        include_external_components=cfg.include_external_components,
        include_price_action_extremes=cfg.include_price_action_extremes,
        price_action_lookbacks=cfg.price_action_lookbacks,
    )


def _period_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    sim = result.get("sim", {})
    stats = result.get("trade_stats", {})
    period = result.get("period", {})
    return {
        "start": period.get("start"),
        "end": period.get("end"),
        "years": period.get("years"),
        "cagr_pct": sim.get("cagr_pct"),
        "strict_mdd_pct": sim.get("strict_mdd_pct"),
        "cagr_to_strict_mdd": sim.get("cagr_to_strict_mdd"),
        "trade_entries": sim.get("trade_entries"),
        "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
        "n_required_for_80pct_power_alpha5pct": stats.get("n_required_for_80pct_power_alpha5pct"),
        "n_gap_to_power_rule": stats.get("n_gap_to_power_rule"),
    }

def _score_period(result: dict[str, Any], *, min_trades: int) -> float:
    sim = result.get("sim", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -100.0) or -100.0)
    mdd = float(sim.get("strict_mdd_pct", 100.0) or 100.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -100.0) or -100.0)
    if trades < int(min_trades) or cagr <= 0.0:
        return -1000.0 + trades / max(1.0, float(min_trades)) + cagr / 100.0 - mdd / 100.0
    return ratio * 10.0 + min(50.0, cagr) / 10.0 - max(0.0, mdd - 15.0) / 5.0 + min(2.0, trades / 100.0)


def _segment_result(events: list[dict[str, Any]], fold_set: set[str], *, dates: pd.Series, market: pd.DataFrame, cfg: EnsembleCfg) -> dict[str, Any]:
    seg = [e for e in events if str(e.get("fold")) in fold_set]
    seg.sort(key=lambda e: (int(e["signal_pos"]), str(e.get("candidate_key", ""))))
    return _simulate_events(seg, dates=dates, market=market, cfg=cfg)


def _run_one_config(
    *,
    candidate_source: list[dict[str, Any]],
    fold_order: list[str],
    train_names: set[str],
    test_names: set[str],
    eval_names: set[str],
    event_cache: dict[int, list[dict[str, Any]]],
    dates: pd.Series,
    market: pd.DataFrame,
    cfg: TTEValidatorCfg,
    candidate_limit: int,
    ensemble_size: int,
) -> dict[str, Any]:
    wf_cfg = _wf_cfg(cfg, candidate_limit=candidate_limit, ensemble_size=ensemble_size)
    base_cfg = _ensemble_cfg(wf_cfg)
    candidate_ids = [int(c["_candidate_index"]) for c in candidate_source[: int(candidate_limit)]]
    final_events: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for pos, fold in enumerate(fold_order):
        history = fold_order[:pos]
        if len(history) < int(cfg.min_history_folds):
            if str(cfg.seed_candidates) == "all_when_cold":
                selected = candidate_ids[: max(1, int(ensemble_size))]
                steps = []
            else:
                selected, steps = _select_for_history(candidate_ids=candidate_ids, event_cache=event_cache, history_folds=[], dates=dates, market=market, cfg=base_cfg)
        else:
            selected, steps = _select_for_history(candidate_ids=candidate_ids, event_cache=event_cache, history_folds=history, dates=dates, market=market, cfg=base_cfg)
        fold_events: list[dict[str, Any]] = []
        for cid in selected:
            fold_events.extend(_filter_events(event_cache[cid], {fold}))
        fold_events.sort(key=lambda e: (int(e["signal_pos"]), str(e.get("candidate_key", ""))))
        final_events.extend(fold_events)
        fold_rows.append({"fold": fold, "history_folds": history, "selected": selected, "steps": steps, "fold_result": {k: v for k, v in _simulate_events(fold_events, dates=dates, market=market, cfg=base_cfg).items() if k != "executed"}})
    periods = {
        "train": {k: v for k, v in _segment_result(final_events, train_names, dates=dates, market=market, cfg=base_cfg).items() if k != "executed"},
        "test": {k: v for k, v in _segment_result(final_events, test_names, dates=dates, market=market, cfg=base_cfg).items() if k != "executed"},
        "eval": {k: v for k, v in _segment_result(final_events, eval_names, dates=dates, market=market, cfg=base_cfg).items() if k != "executed"},
        "all": {k: v for k, v in _simulate_events(sorted(final_events, key=lambda e: (int(e["signal_pos"]), str(e.get("candidate_key", "")))), dates=dates, market=market, cfg=base_cfg).items() if k != "executed"},
    }
    return {
        "candidate_limit": int(candidate_limit),
        "max_ensemble_size": int(ensemble_size),
        "test_score": _score_period(periods["test"], min_trades=int(cfg.min_test_trades)),
        "periods": periods,
        "folds": fold_rows,
    }


def run(cfg: TTEValidatorCfg) -> dict[str, Any]:
    source = json.loads(Path(cfg.train_sparse_report).read_text())
    train_folds = _load_folds(cfg.train_folds_json)
    test_folds = _load_folds(cfg.test_folds_json)
    eval_folds = _load_folds(cfg.eval_folds_json)
    all_folds = sorted(train_folds + test_folds + eval_folds, key=lambda f: f["eval_start"])
    replay_report = dict(source)
    replay_report["folds"] = all_folds

    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
            include_forex_components=bool(cfg.include_external_components),
        )
    feats = _features(market, _wf_cfg(cfg, candidate_limit=max(cfg.candidate_limits), ensemble_size=max(cfg.ensemble_sizes)))
    dates = pd.to_datetime(market["date"])

    candidates = [dict(c, _candidate_index=i) for i, c in enumerate(source.get("top_strict", [])[: max(cfg.candidate_limits)])]
    event_base_cfg = _ensemble_cfg(_wf_cfg(cfg, candidate_limit=max(cfg.candidate_limits), ensemble_size=max(cfg.ensemble_sizes)))
    event_cache = {int(c["_candidate_index"]): _candidate_events(cand=c, report=replay_report, dates=dates, features=feats, market=market, cfg=event_base_cfg) for c in candidates}
    fold_order = _fold_names(all_folds)
    train_names, test_names, eval_names = set(_fold_names(train_folds)), set(_fold_names(test_folds)), set(_fold_names(eval_folds))

    configs: list[dict[str, Any]] = []
    for limit in cfg.candidate_limits:
        for ens in cfg.ensemble_sizes:
            if int(ens) > int(limit):
                continue
            configs.append(
                _run_one_config(
                    candidate_source=candidates,
                    fold_order=fold_order,
                    train_names=train_names,
                    test_names=test_names,
                    eval_names=eval_names,
                    event_cache=event_cache,
                    dates=dates,
                    market=market,
                    cfg=cfg,
                    candidate_limit=int(limit),
                    ensemble_size=int(ens),
                )
            )
    configs.sort(key=lambda r: (float(r["test_score"]), float(r["periods"]["test"].get("sim", {}).get("cagr_pct", -100.0))), reverse=True)
    best = configs[0] if configs else {}
    summary = {name: _period_snapshot(value) for name, value in best.get("periods", {}).items()} if best else {}
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"candidate_limits": list(cfg.candidate_limits), "ensemble_sizes": list(cfg.ensemble_sizes)},
        "candidate_source": {
            "report": cfg.train_sparse_report,
            "source_folds": source.get("folds", []),
            "candidate_count_available": len(source.get("top_strict", [])),
            "candidate_keys": {str(i): _candidate_key(c) for i, c in enumerate(source.get("top_strict", [])[: max(cfg.candidate_limits)])},
        },
        "splits": {"train": train_folds, "test": test_folds, "eval": eval_folds},
        "best_by_test": best,
        "selected_config": {k: best.get(k) for k in ("candidate_limit", "max_ensemble_size", "test_score")} if best else {},
        "summary": summary,
        "all_configs": configs,
        "leakage_guard": {
            "candidate_discovery_report_expected_train_only": True,
            "config_selection_uses_test_score_only": True,
            "eval_not_used_for_config_selection": True,
            "fold_thresholds_and_side_fit_before_each_fold_start": True,
            "eval_selector_history_may_include_train_and_test_past_only": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate train-discovered sparse setup candidates on train/test/eval splits")
    p.add_argument("--train-sparse-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-folds-json", required=True)
    p.add_argument("--test-folds-json", required=True)
    p.add_argument("--eval-folds-json", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=TTEValidatorCfg.external_tolerance)
    p.add_argument("--window-size", type=int, default=TTEValidatorCfg.window_size)
    p.add_argument("--candidate-limits", default="4,8,12,24")
    p.add_argument("--ensemble-sizes", default="1,2,4")
    p.add_argument("--leverage", type=float, default=TTEValidatorCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=TTEValidatorCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=TTEValidatorCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=TTEValidatorCfg.entry_delay_bars)
    p.add_argument("--trade-stop-loss-pct", type=float, default=TTEValidatorCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=TTEValidatorCfg.trade_take_profit_pct)
    p.add_argument("--setup-sizing", choices=["fixed", "prior_sharpe"], default=TTEValidatorCfg.setup_sizing)
    p.add_argument("--execution-horizon-bars", type=int, default=TTEValidatorCfg.execution_horizon_bars)
    p.add_argument("--min-history-folds", type=int, default=TTEValidatorCfg.min_history_folds)
    p.add_argument("--seed-candidates", choices=["top_prior", "all_when_cold"], default=TTEValidatorCfg.seed_candidates)
    p.add_argument("--include-external-components", action="store_true", default=TTEValidatorCfg.include_external_components)
    p.add_argument("--include-price-action-extremes", action="store_true", default=TTEValidatorCfg.include_price_action_extremes)
    p.add_argument("--price-action-lookbacks", default=TTEValidatorCfg.price_action_lookbacks)
    p.add_argument("--min-test-trades", type=int, default=TTEValidatorCfg.min_test_trades)
    ns = p.parse_args()
    ns.candidate_limits = _parse_ints(ns.candidate_limits)
    ns.ensemble_sizes = _parse_ints(ns.ensemble_sizes)
    return ns


def main() -> None:
    rep = run(TTEValidatorCfg(**vars(parse_args())))
    best = rep.get("best_by_test", {})
    print(json.dumps({"best_config": {k: best.get(k) for k in ("candidate_limit", "max_ensemble_size", "test_score")}, "periods": best.get("periods", {})}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
