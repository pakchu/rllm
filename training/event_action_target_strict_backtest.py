"""Strict backtest target actions from event_action_policy rows.

This is an oracle-ceiling diagnostic for the candidate-book structure: targets use
future utility as labels, so results are not live tradable, but they show whether
the past-only candidate book contains actions that could be selected profitably.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventActionTargetBacktestConfig:
    inputs: str
    market_csv: str
    output_predictions: str
    output_report: str
    start: str = ""
    end: str = ""
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_rank_utility: float = -999.0
    min_mfe_to_mae: float = -999.0
    allowed_confidence: str = "LOW,MID,HIGH"


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in str(inputs).split(','):
        path = raw.strip()
        if path:
            rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return sorted(rows, key=lambda r: (str(r.get('date', '')), int(r.get('signal_pos', 0) or 0)))


def _target_action(row: dict[str, Any]) -> dict[str, Any] | None:
    target = row.get('target')
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except Exception:
            target = None
    if not isinstance(target, dict):
        return None
    if str(target.get('gate', 'NO_TRADE')).upper() != 'TRADE':
        return None
    side = str(target.get('side', 'NONE')).upper()
    hold = int(target.get('hold_bars', 0) or 0)
    if side not in {'LONG', 'SHORT'} or hold <= 0:
        return None
    return target


def _predictions(rows: list[dict[str, Any]], cfg: EventActionTargetBacktestConfig) -> list[dict[str, Any]]:
    allowed_conf = {x.strip().upper() for x in str(cfg.allowed_confidence).split(',') if x.strip()}
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        date = str(row.get('date', ''))
        if cfg.start and date < cfg.start:
            continue
        if cfg.end and date > cfg.end:
            continue
        pos = int(row.get('signal_pos', -1) or -1)
        if pos in seen or pos < 0:
            continue
        action = _target_action(row)
        if action is None:
            continue
        audit = row.get('target_action_audit') if isinstance(row.get('target_action_audit'), dict) else {}
        if float(audit.get('rank_utility', 0.0) or 0.0) < float(cfg.min_rank_utility):
            continue
        if float(audit.get('mfe_to_mae', 0.0) or 0.0) < float(cfg.min_mfe_to_mae):
            continue
        if allowed_conf and str(action.get('confidence', '')).upper() not in allowed_conf:
            continue
        pred = {
            'gate': 'TRADE',
            'side': str(action.get('side')).upper(),
            'hold_bars': min(int(action.get('hold_bars', 0) or 0), int(cfg.max_hold_bars)),
            'family': str(action.get('family', 'target_oracle')),
            'confidence': str(action.get('confidence', '')),
        }
        out.append({
            'date': date,
            'signal_pos': pos,
            'prediction': pred,
            'position_scale': 1.0,
            'score': float(audit.get('rank_utility', 0.0) or 0.0),
            'target_action_audit': audit,
        })
        seen.add(pos)
    return out


def run(cfg: EventActionTargetBacktestConfig) -> dict[str, Any]:
    preds = _predictions(_load(cfg.inputs), cfg)
    Path(cfg.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_predictions).write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + ('\n' if preds else ''))
    if not preds:
        report = {'config': asdict(cfg), 'prediction_rows': 0, 'sim': {'trade_entries': 0}, 'trade_stats': {'n_trades': 0}}
    else:
        report = run_overlay(OnlineRiskOverlayConfig(
            predictions_jsonl=cfg.output_predictions,
            market_csv=cfg.market_csv,
            output=cfg.output_report,
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            entry_delay_bars=cfg.entry_delay_bars,
            max_hold_bars=cfg.max_hold_bars,
        ))
        report['config'] = asdict(cfg)
        report['prediction_rows'] = len(preds)
        report['leakage_notice'] = 'oracle target-action diagnostic; not live tradable because targets use future utility labels'
    Path(cfg.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--market-csv', required=True)
    p.add_argument('--output-predictions', required=True)
    p.add_argument('--output-report', required=True)
    p.add_argument('--start', default=EventActionTargetBacktestConfig.start)
    p.add_argument('--end', default=EventActionTargetBacktestConfig.end)
    p.add_argument('--leverage', type=float, default=EventActionTargetBacktestConfig.leverage)
    p.add_argument('--max-hold-bars', type=int, default=EventActionTargetBacktestConfig.max_hold_bars)
    p.add_argument('--entry-delay-bars', type=int, default=EventActionTargetBacktestConfig.entry_delay_bars)
    p.add_argument('--fee-rate', type=float, default=EventActionTargetBacktestConfig.fee_rate)
    p.add_argument('--slippage-rate', type=float, default=EventActionTargetBacktestConfig.slippage_rate)
    p.add_argument('--min-rank-utility', type=float, default=EventActionTargetBacktestConfig.min_rank_utility)
    p.add_argument('--min-mfe-to-mae', type=float, default=EventActionTargetBacktestConfig.min_mfe_to_mae)
    p.add_argument('--allowed-confidence', default=EventActionTargetBacktestConfig.allowed_confidence)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventActionTargetBacktestConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
