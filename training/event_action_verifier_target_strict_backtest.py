"""Strict backtest ALLOW targets from event_action_verifier rows.

Oracle ceiling for post-ranker verifier datasets.  Targets use future audit labels,
so this is not deployable; it measures whether the verifier surface contains
tradable ALLOW decisions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventActionVerifierTargetBacktestConfig:
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
    target_label: str = "ALLOW"


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in str(inputs).split(','):
        path = raw.strip()
        if path:
            rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return sorted(rows, key=lambda r: (str(r.get('date', '')), int(r.get('signal_pos', 0) or 0)))


def _predictions(rows: list[dict[str, Any]], cfg: EventActionVerifierTargetBacktestConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        date = str(row.get('date', ''))
        if cfg.start and date < cfg.start:
            continue
        if cfg.end and date > cfg.end:
            continue
        if str(row.get('target', '')).upper() != str(cfg.target_label).upper():
            continue
        action = row.get('action') if isinstance(row.get('action'), dict) else {}
        side = str(action.get('side', 'NONE')).upper()
        hold = int(action.get('hold_bars', 0) or 0)
        pos = int(row.get('signal_pos', -1) or -1)
        if pos in seen or pos < 0 or side not in {'LONG', 'SHORT'} or hold <= 0:
            continue
        pred = {'gate': 'TRADE', 'side': side, 'hold_bars': min(hold, int(cfg.max_hold_bars)), 'family': str(action.get('family', 'verifier_oracle'))}
        out.append({'date': date, 'signal_pos': pos, 'prediction': pred, 'position_scale': 1.0, 'score': 1.0, 'action_audit': row.get('action_audit')})
        seen.add(pos)
    return out


def run(cfg: EventActionVerifierTargetBacktestConfig) -> dict[str, Any]:
    preds = _predictions(_load(cfg.inputs), cfg)
    Path(cfg.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_predictions).write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + ('\n' if preds else ''))
    if preds:
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
    else:
        report = {'sim': {'trade_entries': 0}, 'trade_stats': {'n_trades': 0}}
    report['config'] = asdict(cfg)
    report['prediction_rows'] = len(preds)
    report['leakage_notice'] = 'oracle verifier target diagnostic; not live tradable because ALLOW/BLOCK targets use future audit labels'
    Path(cfg.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--market-csv', required=True)
    p.add_argument('--output-predictions', required=True)
    p.add_argument('--output-report', required=True)
    p.add_argument('--start', default=EventActionVerifierTargetBacktestConfig.start)
    p.add_argument('--end', default=EventActionVerifierTargetBacktestConfig.end)
    p.add_argument('--leverage', type=float, default=EventActionVerifierTargetBacktestConfig.leverage)
    p.add_argument('--max-hold-bars', type=int, default=EventActionVerifierTargetBacktestConfig.max_hold_bars)
    p.add_argument('--entry-delay-bars', type=int, default=EventActionVerifierTargetBacktestConfig.entry_delay_bars)
    p.add_argument('--fee-rate', type=float, default=EventActionVerifierTargetBacktestConfig.fee_rate)
    p.add_argument('--slippage-rate', type=float, default=EventActionVerifierTargetBacktestConfig.slippage_rate)
    p.add_argument('--target-label', default=EventActionVerifierTargetBacktestConfig.target_label)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventActionVerifierTargetBacktestConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
