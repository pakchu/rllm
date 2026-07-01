"""Discover simple symbolic candidate-selection rules with chronological splits.

This is a bridge toward deductive LLM usage: rules are explicit symbolic
hypotheses (source/side/state bucket -> follow or invert candidate).  The script
scores many simple rules on train, ranks on test, and reports eval without using
eval for selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.linear_alpha_deductive_candidate_selector import _candidate_from_pair, _numeric_state, _sign_for_side
from training.linear_alpha_meta_stability_diagnostic import _date


@dataclass(frozen=True)
class RuleDiscoveryConfig:
    pairwise_inputs: str
    output: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    min_trades_train: int = 30
    min_trades_test: int = 50
    top_k: int = 50


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in str(inputs).split(','):
        path = raw.strip()
        if path:
            rows.extend(_read_jsonl(path))
    return sorted(rows, key=lambda r: (str(r.get('date', '')), int(r.get('signal_pos', 0) or 0)))


def _split_name(date: str, cfg: RuleDiscoveryConfig) -> str:
    if cfg.train_start <= date <= cfg.train_end:
        return 'train'
    if cfg.test_start <= date <= cfg.test_end:
        return 'test'
    if cfg.eval_start <= date <= cfg.eval_end:
        return 'eval'
    return 'ignore'


def _bucket(v: float, cuts: tuple[float, float]) -> str:
    if not math.isfinite(float(v)):
        return 'nan'
    if v <= cuts[0]:
        return 'low'
    if v >= cuts[1]:
        return 'high'
    return 'mid'


def _candidate_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Deduplicate candidates per timestamp/id while preserving future path metadata for offline scoring.
    seen: set[tuple[int, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        state = _numeric_state(str(row.get('prompt', '')))
        for label in ('A', 'B'):
            cand = _candidate_from_pair(row, label)
            if cand is None:
                continue
            key = (int(row.get('signal_pos', -1) or -1), str(cand.get('id')))
            if key in seen:
                continue
            seen.add(key)
            path = cand.get('path') if isinstance(cand.get('path'), dict) else {}
            ret = float(path.get('realized_return_pct', 0.0) or 0.0)
            adverse = float(path.get('max_adverse_pct', 0.0) or 0.0)
            favorable = float(path.get('max_favorable_pct', 0.0) or 0.0)
            utility = ret + 0.25 * favorable - 0.75 * adverse
            pred = cand['prediction']
            out.append({
                'date': str(row.get('date')),
                'signal_pos': int(row.get('signal_pos', -1) or -1),
                'id': str(cand.get('id')),
                'side': str(pred.get('side', 'NONE')),
                'hold_bars': int(pred.get('hold_bars', 0) or 0),
                'state': state,
                'ret_pct': ret,
                'utility_pct': utility,
            })
    return out


def _features(ex: dict[str, Any]) -> set[str]:
    state = ex['state']
    cid = str(ex['id'])
    side = str(ex['side'])
    s = _sign_for_side(side)
    feats = {
        f'id={cid}',
        f'side={side}',
        f'id_side={cid}|{side}',
        f'hold={ex["hold_bars"]}',
    }
    specs = {
        'trend_96': (-0.02, 0.02),
        'htf_4h_return_4': (-0.02, 0.02),
        'htf_1d_return_4': (-0.03, 0.03),
        'htf_1w_return_4': (-0.06, 0.06),
        'range_pos': (-0.5, 0.5),
        'rex_2016_range_pos': (-0.5, 0.5),
        'rex_8640_range_pos': (-0.5, 0.5),
        'range_vol': (0.02, 0.08),
        'window_drawdown': (0.03, 0.08),
        'dxy_zscore': (-1.0, 1.0),
        'kimchi_premium_zscore': (-1.0, 1.0),
        'usdkrw_zscore': (-1.0, 1.0),
    }
    for key, cuts in specs.items():
        val = float(state.get(key, 0.0) or 0.0)
        feats.add(f'{key}={_bucket(val, cuts)}')
        if s:
            feats.add(f'side_{key}={_bucket(s * val, cuts)}')
    return feats


def _rule_candidates(train: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    counts: Counter[str] = Counter()
    by_ex = []
    for ex in train:
        fs = _features(ex)
        by_ex.append(fs)
        counts.update(fs)
    base = [f for f, n in counts.items() if n >= 25]
    rules: set[tuple[str, ...]] = {(f,) for f in base}
    # Build small two-premise rules anchored by source/side plus one state premise.
    anchors = [f for f in base if f.startswith(('id=', 'side=', 'id_side='))]
    states = [f for f in base if not f.startswith(('id=', 'side=', 'id_side=', 'hold='))]
    for a in anchors:
        for b in states:
            rules.add(tuple(sorted((a, b))))
    return sorted(rules)


def _match(ex: dict[str, Any], rule: tuple[str, ...]) -> bool:
    fs = _features(ex)
    return all(part in fs for part in rule)


def _prepare_cached(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ex in examples:
        row = dict(ex)
        row['_features'] = _features(ex)
        out.append(row)
    return out


def _eval_rule(examples: list[dict[str, Any]], rule: tuple[str, ...], action: str) -> dict[str, Any]:
    vals = []
    n_long = n_short = 0
    rule_set = set(rule)
    for ex in examples:
        if not rule_set.issubset(ex['_features']):
            continue
        # action=follow means take candidate side; action=invert approximates opposite return.
        sign = 1.0 if action == 'follow' else -1.0
        vals.append(sign * float(ex['ret_pct']))
        if (ex['side'] == 'LONG') == (action == 'follow'):
            n_long += 1
        else:
            n_short += 1
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {'n': 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    t = mean / (std / math.sqrt(arr.size)) if std > 1e-12 else 0.0
    win = float(np.mean(arr > 0.0))
    return {'n': int(arr.size), 'mean_ret_pct': mean, 'std_ret_pct': std, 't_stat_like': t, 'win_rate': win, 'long': n_long, 'short': n_short}


def _score(m: dict[str, Any]) -> float:
    if int(m.get('n', 0)) <= 0:
        return -1e9
    return float(m.get('mean_ret_pct', 0.0)) * min(3.0, math.sqrt(float(m['n'])) / 5.0) + 0.02 * float(m.get('t_stat_like', 0.0))


def run(cfg: RuleDiscoveryConfig) -> dict[str, Any]:
    rows = _load(cfg.pairwise_inputs)
    examples = _candidate_examples(rows)
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        name = _split_name(ex['date'], cfg)
        if name != 'ignore':
            splits[name].append(ex)
    for split_name in list(splits):
        splits[split_name] = _prepare_cached(splits[split_name])
    rules = _rule_candidates(splits['train'])
    out = []
    for rule in rules:
        for action in ('follow', 'invert'):
            tr = _eval_rule(splits['train'], rule, action)
            if int(tr.get('n', 0)) < int(cfg.min_trades_train):
                continue
            te = _eval_rule(splits['test'], rule, action)
            if int(te.get('n', 0)) < int(cfg.min_trades_test):
                continue
            ev = _eval_rule(splits['eval'], rule, action)
            out.append({'rule': list(rule), 'action': action, 'train': tr, 'test': te, 'eval': ev, 'test_score': _score(te)})
    out.sort(key=lambda r: float(r['test_score']), reverse=True)
    report = {
        'as_of': datetime.now(timezone.utc).isoformat(),
        'config': asdict(cfg),
        'example_counts': {k: len(v) for k, v in splits.items()},
        'candidate_rule_count': len(rules),
        'evaluated_rule_count': len(out),
        'selection_protocol': 'generate rules from train support; rank by test only; report eval untouched',
        'top_by_test': out[: int(cfg.top_k)],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Discover symbolic candidate rules')
    p.add_argument('--pairwise-inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-start', default=RuleDiscoveryConfig.train_start)
    p.add_argument('--train-end', default=RuleDiscoveryConfig.train_end)
    p.add_argument('--test-start', default=RuleDiscoveryConfig.test_start)
    p.add_argument('--test-end', default=RuleDiscoveryConfig.test_end)
    p.add_argument('--eval-start', default=RuleDiscoveryConfig.eval_start)
    p.add_argument('--eval-end', default=RuleDiscoveryConfig.eval_end)
    p.add_argument('--min-trades-train', type=int, default=RuleDiscoveryConfig.min_trades_train)
    p.add_argument('--min-trades-test', type=int, default=RuleDiscoveryConfig.min_trades_test)
    p.add_argument('--top-k', type=int, default=RuleDiscoveryConfig.top_k)
    return p.parse_args()


def main() -> None:
    report = run(RuleDiscoveryConfig(**vars(parse_args())))
    print(json.dumps({'example_counts': report['example_counts'], 'evaluated_rule_count': report['evaluated_rule_count'], 'top_by_test': report['top_by_test'][:10]}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
