"""No-leak token reliability baseline for event-action verifier rows.

Train on past-only prompt/action tokens and ALLOW/BLOCK labels from prior periods,
then emit allowed action predictions for later splits.  This is a cheap baseline
before LLM distillation: if token reliability cannot recover any oracle ceiling,
LLM work should focus on richer reasoning/data rather than direct fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class VerifierTokenBaselineConfig:
    train_inputs: str
    test_inputs: str
    eval_inputs: str
    market_csv: str
    output: str
    predictions_dir: str
    min_token_count: int = 20
    smoothing: float = 5.0
    top_token_count: int = 24
    threshold_grid: str = "0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25,0.30"
    score_mode: str = "mean"
    allowed_families: str = ""
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in str(inputs).split(','):
        path = raw.strip()
        if path:
            rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return sorted(rows, key=lambda r: (str(r.get('date', '')), int(r.get('signal_pos', 0) or 0)))


def _clean(x: Any) -> str:
    return re.sub(r'[^A-Za-z0-9_:+.-]+', '_', str(x).strip())[:96]


def _prompt_key_values(row: dict[str, Any]) -> dict[str, str]:
    """Extract past-only categorical prompt facts.

    The verifier prompt intentionally exposes symbolic, LLM-friendly text rather
    than raw numeric candles.  Earlier token baselines only read bullet lines,
    but current prompts store the real alpha surface in semicolon-delimited
    `Regime tokens`, `Candidate book tokens`, and `Selected action tokens`
    lines.  Parse those lines explicitly without touching future-only
    `action_audit` labels.
    """
    out: dict[str, str] = {}
    for raw in str(row.get('prompt', '')).splitlines():
        line = raw.strip()
        if line.startswith('Regime tokens:'):
            payload = line.split(':', 1)[1]
            for part in payload.split(';'):
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                out[f"regime.{_clean(k)}"] = _clean(v)
        elif line.startswith('Selected action tokens:'):
            payload = line.split(':', 1)[1]
            for part in payload.split(';'):
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                out[f"selected.{_clean(k)}"] = _clean(v)
        elif line.startswith('Candidate book tokens:'):
            payload = line.split(':', 1)[1]
            entries = [x.strip() for x in payload.split(';') if x.strip()]
            out['candidate.count_bucket'] = 'many' if len(entries) >= 6 else 'few'
            for entry in entries:
                bits = [b.strip() for b in entry.split(':')]
                if len(bits) >= 3:
                    fam, side, strength = bits[:3]
                    out[f"book.{_clean(fam)}.{_clean(side)}"] = _clean(strength)
        elif line.startswith('- ') and ':' in line:
            k, v = line[2:].split(':', 1)
            out[f"bullet.{_clean(k)}"] = _clean(v)
    state_tokens = row.get('state_tokens') if isinstance(row.get('state_tokens'), dict) else {}
    for k, v in state_tokens.items():
        out[f"state.{_clean(k)}"] = _clean(v)
    return out


def _tokens(row: dict[str, Any]) -> set[str]:
    action = row.get('action') if isinstance(row.get('action'), dict) else {}
    family = _clean(action.get('family', 'unknown'))
    side = _clean(str(action.get('side', 'NONE')).upper())
    hold = int(action.get('hold_bars', 0) or 0)
    kv = _prompt_key_values(row)
    toks = {
        f"family={family}",
        f"side={side}",
        f"hold={hold}",
        f"family_side={family}|{side}",
        f"family_hold={family}|h{hold}",
    }
    for k, v in kv.items():
        toks.add(f"{k}={v}")
        # Low-cardinality interactions let the verifier learn conditional edge:
        # e.g. a SHORT mean-reversion action is different in upper-range vs
        # lower-range contexts.  These are still train-only reliability tokens.
        if k.startswith(('regime.', 'state.')):
            toks.add(f"{family}|{side}|{k}={v}")
            toks.add(f"{side}|{k}={v}")
    return toks


def _is_allow(row: dict[str, Any]) -> bool:
    return str(row.get('target', '')).upper() == 'ALLOW'


def _fit(rows: list[dict[str, Any]], cfg: VerifierTokenBaselineConfig) -> dict[str, Any]:
    allow_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    prior_allow = sum(1 for r in rows if _is_allow(r))
    prior = prior_allow / max(1, len(rows))
    for row in rows:
        toks = _tokens(row)
        total_counts.update(toks)
        if _is_allow(row):
            allow_counts.update(toks)
    reliability: dict[str, float] = {}
    alpha = float(cfg.smoothing)
    for tok, n in total_counts.items():
        if n < int(cfg.min_token_count):
            continue
        p = (allow_counts[tok] + alpha * prior) / (n + alpha)
        reliability[tok] = float(p)
    return {'prior': prior, 'reliability': reliability, 'total_counts': dict(total_counts), 'allow_counts': dict(allow_counts)}


def _score(row: dict[str, Any], model: dict[str, Any], cfg: VerifierTokenBaselineConfig) -> dict[str, Any]:
    prior = float(model['prior'])
    rel = model['reliability']
    vals = []
    for tok in _tokens(row):
        if tok in rel:
            vals.append((tok, float(rel[tok])))
    if not vals:
        return {'score': prior, 'tokens': []}
    vals.sort(key=lambda kv: abs(kv[1] - prior), reverse=True)
    use = vals[: int(cfg.top_token_count)]
    eps = 1e-6
    if cfg.score_mode == 'mean':
        # Conservative reliability average; avoids the extreme sparsity from
        # multiplying many weak token odds.
        score = float(sum(p for _tok, p in use) / max(1, len(use)))
    elif cfg.score_mode == 'max':
        score = float(max(p for _tok, p in use))
    else:
        # Naive Bayes-ish logit aggregation around the train prior.
        logit = math.log(max(eps, min(1 - eps, prior)) / max(eps, 1 - prior))
        prior_logit = logit
        for _tok, p in use:
            p = max(eps, min(1 - eps, p))
            logit += math.log(p / (1 - p)) - prior_logit
        score = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    return {'score': score, 'tokens': use[:8]}


def _allowed_family_set(cfg: VerifierTokenBaselineConfig) -> set[str]:
    return {x.strip() for x in str(cfg.allowed_families).split(',') if x.strip()}


def _scored_candidates(rows: list[dict[str, Any]], model: dict[str, Any], cfg: VerifierTokenBaselineConfig) -> list[dict[str, Any]]:
    # A signal timestamp can have many candidate actions.  Live execution must
    # choose the highest-scored candidate, not the first row in file order.
    # This is a structural verifier/post-ranker step and does not use labels.
    allowed = _allowed_family_set(cfg)
    best_by_pos: dict[int, tuple[float, dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        action = row.get('action') if isinstance(row.get('action'), dict) else {}
        family = str(action.get('family', ''))
        if allowed and family not in allowed:
            continue
        side = str(action.get('side', 'NONE')).upper()
        hold = int(action.get('hold_bars', 0) or 0)
        pos = int(row.get('signal_pos', -1) or -1)
        if pos < 0 or side not in {'LONG', 'SHORT'} or hold <= 0:
            continue
        sc = _score(row, model, cfg)
        score = float(sc['score'])
        prev = best_by_pos.get(pos)
        if prev is None or score > prev[0]:
            best_by_pos[pos] = (score, row, sc)
    out = []
    for pos, (score, row, sc) in sorted(best_by_pos.items()):
        action = row.get('action') if isinstance(row.get('action'), dict) else {}
        side = str(action.get('side', 'NONE')).upper()
        hold = int(action.get('hold_bars', 0) or 0)
        pred = {'gate': 'TRADE', 'side': side, 'hold_bars': min(hold, int(cfg.max_hold_bars)), 'family': str(action.get('family', 'token_verifier'))}
        out.append({'date': str(row.get('date')), 'signal_pos': pos, 'prediction': pred, 'position_scale': 1.0, 'score': score, 'top_tokens': sc['tokens']})
    return out


def _prediction_rows(scored: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [r for r in scored if float(r['score']) >= float(threshold)]


def _backtest(preds: list[dict[str, Any]], cfg: VerifierTokenBaselineConfig, tmp: Path, name: str) -> dict[str, Any]:
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
    # Keep scan reports compact; prediction JSONL is already written for audit.
    report.pop('executed', None)
    return report


def _rank_score(bt: dict[str, Any]) -> float:
    s = bt.get('sim', {})
    t = bt.get('trade_stats', {})
    trades = int(s.get('trade_entries', 0) or 0)
    if trades < 20:
        return -1e9
    return float(s.get('cagr_to_strict_mdd', 0.0) or 0.0) + 0.03 * float(s.get('cagr_pct', 0.0) or 0.0) - 0.25 * float(t.get('p_value_mean_ret_approx', 1.0) or 1.0)


def _robust_rank_score(train_bt: dict[str, Any], test_bt: dict[str, Any]) -> float:
    """Rank thresholds without eval, penalizing 2025-only spikes."""
    tr = train_bt.get('sim', {})
    te = test_bt.get('sim', {})
    tt = test_bt.get('trade_stats', {})
    train_trades = int(tr.get('trade_entries', 0) or 0)
    test_trades = int(te.get('trade_entries', 0) or 0)
    train_cagr = float(tr.get('cagr_pct', 0.0) or 0.0)
    test_cagr = float(te.get('cagr_pct', 0.0) or 0.0)
    train_mdd = float(tr.get('strict_mdd_pct', 0.0) or 0.0)
    test_mdd = float(te.get('strict_mdd_pct', 0.0) or 0.0)
    if train_trades < 120 or test_trades < 40 or train_cagr <= 0.0 or test_cagr <= 0.0:
        return -1e9
    if train_mdd > 40.0 or test_mdd > 18.0:
        return -1e9
    train_ratio = float(tr.get('cagr_to_strict_mdd', 0.0) or 0.0)
    test_ratio = float(te.get('cagr_to_strict_mdd', 0.0) or 0.0)
    p_value = float(tt.get('p_value_mean_ret_approx', 1.0) or 1.0)
    # Prefer balanced evidence.  A high 2025 ratio with weak train support is a
    # validation trap, so the minimum ratio is weighted more than the maximum.
    balance = min(train_ratio, test_ratio) + 0.35 * max(0.0, min(3.0, test_ratio))
    trade_bonus = min(test_trades, 160) / 250.0 + min(train_trades, 700) / 1400.0
    mdd_penalty = 0.015 * max(0.0, train_mdd - 20.0) + 0.02 * max(0.0, test_mdd - 10.0)
    return balance + trade_bonus - 0.25 * p_value - mdd_penalty


def run(cfg: VerifierTokenBaselineConfig) -> dict[str, Any]:
    train = _load(cfg.train_inputs)
    test = _load(cfg.test_inputs)
    eval_rows = _load(cfg.eval_inputs)
    model = _fit(train, cfg)
    thresholds = [float(x) for x in str(cfg.threshold_grid).split(',') if x.strip()]
    train_scored = _scored_candidates(train, model, cfg)
    test_scored = _scored_candidates(test, model, cfg)
    eval_scored = _scored_candidates(eval_rows, model, cfg)
    rows = []
    with tempfile.TemporaryDirectory(prefix='verifier_token_baseline_') as td:
        tmp = Path(td)
        for th in thresholds:
            train_preds = _prediction_rows(train_scored, th)
            test_preds = _prediction_rows(test_scored, th)
            eval_preds = _prediction_rows(eval_scored, th)
            train_bt = _backtest(train_preds, cfg, tmp, f'th{th:.3f}_train')
            test_bt = _backtest(test_preds, cfg, tmp, f'th{th:.3f}_test')
            eval_bt = _backtest(eval_preds, cfg, tmp, f'th{th:.3f}_eval')
            rows.append({'threshold': th, 'train_rows': len(train_preds), 'test_rows': len(test_preds), 'eval_rows': len(eval_preds), 'train': train_bt, 'test': test_bt, 'eval': eval_bt, 'test_score': _rank_score(test_bt), 'robust_score': _robust_rank_score(train_bt, test_bt)})
    rows.sort(key=lambda r: float(r['robust_score']), reverse=True)
    report = {
        'config': asdict(cfg),
        'rows': {'train': len(train), 'test': len(test), 'eval': len(eval_rows)},
        'train_allow_rate': model['prior'],
        'token_count': len(model['reliability']),
        'scored_candidates': {'train': len(train_scored), 'test': len(test_scored), 'eval': len(eval_scored)},
        'score_mode': cfg.score_mode,
        'selection_protocol': 'fit token reliability on train only; choose threshold by robust train+test strict score; report eval untouched',
        'top_by_robust': rows,
        'top_by_test': sorted(rows, key=lambda r: float(r['test_score']), reverse=True),
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
    p.add_argument('--min-token-count', type=int, default=VerifierTokenBaselineConfig.min_token_count)
    p.add_argument('--smoothing', type=float, default=VerifierTokenBaselineConfig.smoothing)
    p.add_argument('--top-token-count', type=int, default=VerifierTokenBaselineConfig.top_token_count)
    p.add_argument('--threshold-grid', default=VerifierTokenBaselineConfig.threshold_grid)
    p.add_argument('--score-mode', choices=('mean','max','logit'), default=VerifierTokenBaselineConfig.score_mode)
    p.add_argument('--allowed-families', default=VerifierTokenBaselineConfig.allowed_families, help='Comma-separated action families allowed before per-signal best-action selection')
    p.add_argument('--leverage', type=float, default=VerifierTokenBaselineConfig.leverage)
    p.add_argument('--max-hold-bars', type=int, default=VerifierTokenBaselineConfig.max_hold_bars)
    p.add_argument('--entry-delay-bars', type=int, default=VerifierTokenBaselineConfig.entry_delay_bars)
    p.add_argument('--fee-rate', type=float, default=VerifierTokenBaselineConfig.fee_rate)
    p.add_argument('--slippage-rate', type=float, default=VerifierTokenBaselineConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(VerifierTokenBaselineConfig(**vars(parse_args())))
    print(json.dumps({'rows': report['rows'], 'train_allow_rate': report['train_allow_rate'], 'token_count': report['token_count'], 'top_by_robust': report['top_by_robust'][:5]}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
