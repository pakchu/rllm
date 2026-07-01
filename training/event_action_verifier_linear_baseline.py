"""No-leak sparse linear verifier for event-action prompt tokens.

This is the next step after token reliability: combine many weak symbolic prompt
features with an online logistic model, then use live-compatible post-ranking
(score all exact actions at a signal timestamp, choose the best action, threshold
by test, report eval untouched).
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.event_action_verifier_token_baseline import _load, _tokens
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class LinearVerifierConfig:
    train_inputs: str
    test_inputs: str
    eval_inputs: str
    market_csv: str
    output: str
    predictions_dir: str
    min_token_count: int = 40
    epochs: int = 4
    learning_rate: float = 0.08
    l2: float = 1e-5
    positive_weight: float = 3.0
    top_abs_weight_count: int = 80
    threshold_grid: str = "0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70"
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _is_allow(row: dict[str, Any]) -> bool:
    return str(row.get('target', '')).upper() == 'ALLOW'


def _feature_vocab(rows: list[dict[str, Any]], min_count: int) -> set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(row))
    return {tok for tok, n in counts.items() if n >= min_count}


def _features(row: dict[str, Any], vocab: set[str]) -> list[str]:
    return sorted(tok for tok in _tokens(row) if tok in vocab)


def _fit(rows: list[dict[str, Any]], cfg: LinearVerifierConfig) -> dict[str, Any]:
    vocab = _feature_vocab(rows, cfg.min_token_count)
    weights: dict[str, float] = {}
    grad2: dict[str, float] = {}
    prior = sum(1 for r in rows if _is_allow(r)) / max(1, len(rows))
    bias = math.log(max(1e-6, prior) / max(1e-6, 1 - prior))
    bias_g2 = 0.0
    lr = float(cfg.learning_rate)
    l2 = float(cfg.l2)
    losses: list[float] = []
    for _epoch in range(int(cfg.epochs)):
        total_loss = 0.0
        for row in rows:
            feats = _features(row, vocab)
            y = 1.0 if _is_allow(row) else 0.0
            z = bias + sum(weights.get(f, 0.0) for f in feats)
            p = _sigmoid(z)
            sw = float(cfg.positive_weight) if y > 0.5 else 1.0
            grad = (p - y) * sw
            total_loss += (-y * math.log(max(1e-9, p)) - (1 - y) * math.log(max(1e-9, 1 - p))) * sw
            bias_g2 += grad * grad
            bias -= lr * grad / math.sqrt(bias_g2 + 1e-8)
            for f in feats:
                g = grad + l2 * weights.get(f, 0.0)
                ng = grad2.get(f, 0.0) + g * g
                weights[f] = weights.get(f, 0.0) - lr * g / math.sqrt(ng + 1e-8)
                grad2[f] = ng
        losses.append(total_loss / max(1, len(rows)))
    top_weights = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[: int(cfg.top_abs_weight_count)]
    return {'vocab': vocab, 'weights': weights, 'bias': bias, 'prior': prior, 'losses': losses, 'top_weights': top_weights}


def _score(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    vocab: set[str] = model['vocab']
    weights: dict[str, float] = model['weights']
    feats = _features(row, vocab)
    z = float(model['bias']) + sum(weights.get(f, 0.0) for f in feats)
    scored = sorted(((f, weights.get(f, 0.0)) for f in feats), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    return {'score': _sigmoid(z), 'features': scored}


def _scored_candidates(rows: list[dict[str, Any]], model: dict[str, Any], cfg: LinearVerifierConfig) -> list[dict[str, Any]]:
    best_by_pos: dict[int, tuple[float, dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        action = row.get('action') if isinstance(row.get('action'), dict) else {}
        side = str(action.get('side', 'NONE')).upper()
        hold = int(action.get('hold_bars', 0) or 0)
        pos = int(row.get('signal_pos', -1) or -1)
        if pos < 0 or side not in {'LONG', 'SHORT'} or hold <= 0:
            continue
        sc = _score(row, model)
        score = float(sc['score'])
        prev = best_by_pos.get(pos)
        if prev is None or score > prev[0]:
            best_by_pos[pos] = (score, row, sc)
    out = []
    for pos, (score, row, sc) in sorted(best_by_pos.items()):
        action = row.get('action') if isinstance(row.get('action'), dict) else {}
        side = str(action.get('side', 'NONE')).upper()
        hold = int(action.get('hold_bars', 0) or 0)
        pred = {'gate': 'TRADE', 'side': side, 'hold_bars': min(hold, int(cfg.max_hold_bars)), 'family': str(action.get('family', 'linear_verifier'))}
        out.append({'date': str(row.get('date')), 'signal_pos': pos, 'prediction': pred, 'position_scale': 1.0, 'score': score, 'top_features': sc['features']})
    return out


def _prediction_rows(scored: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [r for r in scored if float(r['score']) >= float(threshold)]


def _backtest(preds: list[dict[str, Any]], cfg: LinearVerifierConfig, tmp: Path, name: str) -> dict[str, Any]:
    if not preds:
        return {'sim': {'trade_entries': 0, 'cagr_pct': 0.0, 'strict_mdd_pct': 0.0, 'cagr_to_strict_mdd': 0.0}, 'trade_stats': {'n_trades': 0}}
    path = Path(cfg.predictions_dir) / f'{name}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + '\n')
    report = run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(path), market_csv=cfg.market_csv, output=str(tmp / f'{name}_bt.json'),
        leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars,
    ))
    report.pop('executed', None)
    return report


def _rank_score(bt: dict[str, Any]) -> float:
    s = bt.get('sim', {})
    t = bt.get('trade_stats', {})
    trades = int(s.get('trade_entries', 0) or 0)
    if trades < 40:
        return -1e9
    return float(s.get('cagr_to_strict_mdd', 0.0) or 0.0) + 0.03 * float(s.get('cagr_pct', 0.0) or 0.0) - 0.25 * float(t.get('p_value_mean_ret_approx', 1.0) or 1.0)


def run(cfg: LinearVerifierConfig) -> dict[str, Any]:
    train = _load(cfg.train_inputs)
    test = _load(cfg.test_inputs)
    eval_rows = _load(cfg.eval_inputs)
    model = _fit(train, cfg)
    train_scored = _scored_candidates(train, model, cfg)
    test_scored = _scored_candidates(test, model, cfg)
    eval_scored = _scored_candidates(eval_rows, model, cfg)
    rows = []
    thresholds = [float(x) for x in str(cfg.threshold_grid).split(',') if x.strip()]
    with tempfile.TemporaryDirectory(prefix='verifier_linear_baseline_') as td:
        tmp = Path(td)
        for th in thresholds:
            train_preds = _prediction_rows(train_scored, th)
            test_preds = _prediction_rows(test_scored, th)
            eval_preds = _prediction_rows(eval_scored, th)
            train_bt = _backtest(train_preds, cfg, tmp, f'th{th:.3f}_train')
            test_bt = _backtest(test_preds, cfg, tmp, f'th{th:.3f}_test')
            eval_bt = _backtest(eval_preds, cfg, tmp, f'th{th:.3f}_eval')
            rows.append({'threshold': th, 'train_rows': len(train_preds), 'test_rows': len(test_preds), 'eval_rows': len(eval_preds), 'train': train_bt, 'test': test_bt, 'eval': eval_bt, 'test_score': _rank_score(test_bt)})
    rows.sort(key=lambda r: float(r['test_score']), reverse=True)
    report = {
        'config': asdict(cfg),
        'rows': {'train': len(train), 'test': len(test), 'eval': len(eval_rows)},
        'train_allow_rate': model['prior'],
        'vocab_size': len(model['vocab']),
        'losses': model['losses'],
        'top_weights': model['top_weights'],
        'scored_candidates': {'train': len(train_scored), 'test': len(test_scored), 'eval': len(eval_scored)},
        'selection_protocol': 'fit sparse linear verifier on train only; choose threshold by test strict score; report eval untouched',
        'top_by_test': rows,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-inputs', required=True)
    p.add_argument('--test-inputs', required=True)
    p.add_argument('--eval-inputs', required=True)
    p.add_argument('--market-csv', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--predictions-dir', required=True)
    p.add_argument('--min-token-count', type=int, default=LinearVerifierConfig.min_token_count)
    p.add_argument('--epochs', type=int, default=LinearVerifierConfig.epochs)
    p.add_argument('--learning-rate', type=float, default=LinearVerifierConfig.learning_rate)
    p.add_argument('--l2', type=float, default=LinearVerifierConfig.l2)
    p.add_argument('--positive-weight', type=float, default=LinearVerifierConfig.positive_weight)
    p.add_argument('--threshold-grid', default=LinearVerifierConfig.threshold_grid)
    p.add_argument('--leverage', type=float, default=LinearVerifierConfig.leverage)
    p.add_argument('--max-hold-bars', type=int, default=LinearVerifierConfig.max_hold_bars)
    p.add_argument('--entry-delay-bars', type=int, default=LinearVerifierConfig.entry_delay_bars)
    p.add_argument('--fee-rate', type=float, default=LinearVerifierConfig.fee_rate)
    p.add_argument('--slippage-rate', type=float, default=LinearVerifierConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(LinearVerifierConfig(**vars(parse_args())))
    print(json.dumps({
        'rows': report['rows'],
        'train_allow_rate': report['train_allow_rate'],
        'vocab_size': report['vocab_size'],
        'losses': report['losses'],
        'scored_candidates': report['scored_candidates'],
        'top_by_test': report['top_by_test'][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
