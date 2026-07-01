from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class SweepCandidate:
    threshold: float
    trade_stop_loss_pct: float
    trade_take_profit_pct: float
    atr_trailing_stop_mult: float
    monthly_loss_stop_pct: float
    rolling_window_trades: int
    rolling_drawdown_stop_pct: float
    pause_bars: int
    cooldown_bars: int


def _parse_csv(raw: str, cast):
    return [cast(x) for x in str(raw).split(',') if str(x).strip()]


def _strip(out: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in out.items() if k != 'executed'}


def _score(test: dict[str, Any], train: dict[str, Any]) -> float:
    ts = test['sim']
    tr = train['sim']
    tt = test.get('trade_stats', {})
    # Select only by train+test. Eval is never used for ranking.
    if tr['cagr_pct'] <= 0 or ts['cagr_pct'] <= 0:
        return -1e9
    if tr['strict_mdd_pct'] > 45 or ts['strict_mdd_pct'] > 18:
        return -1e9
    if ts['trade_entries'] < 50:
        return -1e9
    ratio = ts['cagr_to_strict_mdd']
    train_ratio = tr['cagr_to_strict_mdd']
    p = float(tt.get('p_value_mean_ret_approx', 1.0) or 1.0)
    # Reward test risk-adjusted return, enough trades, and some train agreement.
    return ratio + 0.20 * train_ratio + min(ts['trade_entries'], 160) / 200.0 - 0.25 * p


def main() -> None:
    ap = argparse.ArgumentParser(description='Sweep online risk overlays for verifier prediction files; rank by train/test only.')
    ap.add_argument('--predictions-dir', required=True)
    ap.add_argument('--market-csv', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--thresholds', default='0.0,0.02,0.06,0.065,0.07')
    ap.add_argument('--stop-loss-grid', default='0,1.5,2.5,4.0')
    ap.add_argument('--take-profit-grid', default='0,2.5,5.0,8.0')
    ap.add_argument('--atr-grid', default='0')
    ap.add_argument('--monthly-loss-grid', default='0,6.0')
    ap.add_argument('--rolling-window-grid', default='0')
    ap.add_argument('--rolling-dd-grid', default='0')
    ap.add_argument('--pause-bars-grid', default='288')
    ap.add_argument('--cooldown-bars-grid', default='0,48')
    ap.add_argument('--max-hold-bars', type=int, default=576)
    ap.add_argument('--top-k', type=int, default=30)
    args = ap.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_path = Path(args.output)
    thresholds = _parse_csv(args.thresholds, float)
    candidates: list[SweepCandidate] = []
    for vals in itertools.product(
        thresholds,
        _parse_csv(args.stop_loss_grid, float),
        _parse_csv(args.take_profit_grid, float),
        _parse_csv(args.atr_grid, float),
        _parse_csv(args.monthly_loss_grid, float),
        _parse_csv(args.rolling_window_grid, int),
        _parse_csv(args.rolling_dd_grid, float),
        _parse_csv(args.pause_bars_grid, int),
        _parse_csv(args.cooldown_bars_grid, int),
    ):
        c = SweepCandidate(*vals)
        # rolling DD threshold is meaningful only with a rolling window.
        if c.rolling_window_trades <= 0 and c.rolling_drawdown_stop_pct > 0:
            continue
        if c.rolling_window_trades > 0 and c.rolling_drawdown_stop_pct <= 0:
            continue
        # Avoid pure same-family duplicates: TP/SL/ATR all zero plus overlay duplicates are still allowed for baseline once.
        candidates.append(c)

    results: list[dict[str, Any]] = []
    tmp_dir = out_path.parent / (out_path.stem + '_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, cand in enumerate(candidates):
        if idx and idx % 50 == 0:
            print(f'swept {idx}/{len(candidates)} candidates', flush=True)
        split_out: dict[str, Any] = {}
        for split in ('train', 'test', 'eval'):
            pred = pred_dir / f'th{cand.threshold:.3f}_{split}.jsonl'
            if not pred.exists():
                # Files for 0.000 are formatted exactly by baseline script; skip missing thresholds robustly.
                raise FileNotFoundError(pred)
            cfg = OnlineRiskOverlayConfig(
                predictions_jsonl=str(pred),
                market_csv=args.market_csv,
                output=str(tmp_dir / f'cand{idx:05d}_{split}.json'),
                max_hold_bars=args.max_hold_bars,
                trade_stop_loss_pct=cand.trade_stop_loss_pct,
                trade_take_profit_pct=cand.trade_take_profit_pct,
                atr_trailing_stop_mult=cand.atr_trailing_stop_mult,
                monthly_loss_stop_pct=cand.monthly_loss_stop_pct,
                rolling_window_trades=cand.rolling_window_trades,
                rolling_drawdown_stop_pct=cand.rolling_drawdown_stop_pct,
                pause_bars=cand.pause_bars,
                cooldown_bars=cand.cooldown_bars,
            )
            split_out[split] = _strip(run_overlay(cfg))
        results.append({
            'candidate': asdict(cand),
            'score_train_test_only': _score(split_out['test'], split_out['train']),
            **split_out,
        })

    results.sort(key=lambda r: r['score_train_test_only'], reverse=True)
    report = {
        'selection_protocol': 'ranked by train+2025 test only; eval reported after selection and never used in score',
        'candidate_count': len(candidates),
        'top': results[: int(args.top_k)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        'candidate_count': len(candidates),
        'best': {
            'candidate': results[0]['candidate'],
            'score': results[0]['score_train_test_only'],
            'train': results[0]['train']['sim'],
            'test': results[0]['test']['sim'],
            'eval': results[0]['eval']['sim'],
        }
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
