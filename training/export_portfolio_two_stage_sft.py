"""LEGACY / DEPRECATED: kept only to reproduce historical analyzer/trader experiments.

Export portfolio decision rows into separate gate and side SFT tasks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TwoStageCfg:
    input_jsonl: str
    gate_train_output: str
    gate_eval_output: str
    side_train_output: str
    side_eval_output: str
    summary_output: str
    side_eval_all_output: str = ""
    side_train_all_output: str = ""


def _load(path: str) -> list[dict[str, Any]]:
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gate_prompt(prompt: str) -> str:
    return prompt.replace('Output exactly one label: LONG, SHORT, or NO_TRADE.', 'Output exactly one label: TRADE or NO_TRADE.')


def _side_prompt(prompt: str) -> str:
    return prompt.replace('Output exactly one label: LONG, SHORT, or NO_TRADE.', 'Output exactly one label: LONG or SHORT. This row is already gated as TRADE.')


def _best_side(row: dict[str, Any]) -> str:
    audit = row.get('reward_audit', {}) if isinstance(row.get('reward_audit'), dict) else {}
    long_u = float(audit.get('LONG', {}).get('utility', float('-inf')))
    short_u = float(audit.get('SHORT', {}).get('utility', float('-inf')))
    return 'LONG' if long_u >= short_u else 'SHORT'


def _msg(prompt: str, target: str, row: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        'task': task,
        'split': row.get('split'),
        'date': row.get('date'),
        'signal_pos': row.get('signal_pos'),
        'prompt': prompt,
        'target': target,
        'source_target': row.get('target'),
        'candidate': row.get('candidate', {}),
        'leakage_guard': row.get('leakage_guard', {}),
    }


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+('\n' if rows else ''))


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts=Counter(str(r['target']) for r in rows)
    lens=[len(str(r['prompt'])) for r in rows]
    return {'rows':len(rows),'target_counts':dict(sorted(counts.items())),'prompt_chars':{'min':min(lens) if lens else 0,'max':max(lens) if lens else 0,'mean':sum(lens)/max(1,len(lens))}}


def run(cfg: TwoStageCfg) -> dict[str, Any]:
    source=_load(cfg.input_jsonl)
    gate=[]; side=[]; side_all=[]
    for r in source:
        t=str(r.get('target'))
        gate_t='NO_TRADE' if t=='NO_TRADE' else 'TRADE'
        gate.append(_msg(_gate_prompt(str(r['prompt'])), gate_t, r, 'text_state_portfolio_gate'))
        side_t = t if t in {'LONG','SHORT'} else _best_side(r)
        side_all.append(_msg(_side_prompt(str(r['prompt'])), side_t, r, 'text_state_portfolio_side_all'))
        if t in {'LONG','SHORT'}:
            side.append(_msg(_side_prompt(str(r['prompt'])), t, r, 'text_state_portfolio_side'))
    gate_train=[r for r in gate if r.get('split')=='train']; gate_eval=[r for r in gate if r.get('split')=='eval']
    side_train=[r for r in side if r.get('split')=='train']; side_eval=[r for r in side if r.get('split')=='eval']
    side_all_train=[r for r in side_all if r.get('split')=='train']; side_all_eval=[r for r in side_all if r.get('split')=='eval']
    _write(cfg.gate_train_output, gate_train); _write(cfg.gate_eval_output, gate_eval)
    _write(cfg.side_train_output, side_train); _write(cfg.side_eval_output, side_eval)
    if cfg.side_train_all_output:
        _write(cfg.side_train_all_output, side_all_train)
    if cfg.side_eval_all_output:
        _write(cfg.side_eval_all_output, side_all_eval)
    outputs={'gate_train':cfg.gate_train_output,'gate_eval':cfg.gate_eval_output,'side_train':cfg.side_train_output,'side_eval':cfg.side_eval_output}
    if cfg.side_train_all_output:
        outputs['side_train_all']=cfg.side_train_all_output
    if cfg.side_eval_all_output:
        outputs['side_eval_all']=cfg.side_eval_all_output
    report={'config':cfg.__dict__,'gate_train':_summ(gate_train),'gate_eval':_summ(gate_eval),'side_train':_summ(side_train),'side_eval':_summ(side_eval),'side_train_all':_summ(side_all_train),'side_eval_all':_summ(side_all_eval),'outputs':outputs,'contract':'gate predicts TRADE/NO_TRADE; side predicts LONG/SHORT. *_all side files cover every timestamp using past-state prompt and best-side target from reward audit for offline scoring only; deployable validation composes gate model with side model over all eval rows.'}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Export two-stage portfolio SFT rows')
    p.add_argument('--input-jsonl',required=True)
    p.add_argument('--gate-train-output',required=True)
    p.add_argument('--gate-eval-output',required=True)
    p.add_argument('--side-train-output',required=True)
    p.add_argument('--side-eval-output',required=True)
    p.add_argument('--summary-output',required=True)
    p.add_argument('--side-eval-all-output',default='')
    p.add_argument('--side-train-all-output',default='')
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(TwoStageCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
