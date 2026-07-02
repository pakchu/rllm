"""Build binary neutral-choice records labeled by an existing teacher prediction stream."""
from __future__ import annotations

import argparse, json, random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherDistillCfg:
    input_jsonl: str
    teacher_predictions: str
    train_output: str
    eval_output: str
    summary_output: str
    train_start: str = "2023-04-01"
    train_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    randomize_labels: bool = True
    random_seed: int = 42
    skip_side_mismatch: bool = True


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_teacher(raw: str) -> dict[tuple[str, int], dict[str, Any]]:
    out={}
    for path in [x.strip() for x in str(raw).replace('\n', ',').split(',') if x.strip()]:
        for r in _load_jsonl(path):
            out[(str(r.get('date')), int(r.get('signal_pos', -1) or -1))]=r
    return out


def _split(d: str, cfg: TeacherDistillCfg) -> str | None:
    if cfg.train_start <= d < cfg.train_end: return 'train'
    if cfg.eval_start <= d < cfg.eval_end: return 'eval'
    return None


def _teacher_wants_trade(t: dict[str, Any]) -> tuple[bool, str]:
    p=t.get('prediction') if isinstance(t.get('prediction'), dict) else {}
    if str(p.get('gate','NO_TRADE')).upper() != 'TRADE': return False, 'NONE'
    side=str(p.get('side','NONE')).upper()
    return side in {'LONG','SHORT'}, side


def _resolve_trade_choice(row: dict[str, Any]) -> tuple[str | None, str]:
    cmap=row.get('choice_map') if isinstance(row.get('choice_map'), dict) else {}
    for label, cid in cmap.items():
        cid=str(cid).upper()
        if cid != 'NO_TRADE':
            side='SHORT' if cid.endswith('_SHORT') else 'LONG' if cid.endswith('_LONG') else 'NONE'
            return str(label), side
    return None, 'NONE'


def _resolve_no_trade_choice(row: dict[str, Any]) -> str | None:
    cmap=row.get('choice_map') if isinstance(row.get('choice_map'), dict) else {}
    for label, cid in cmap.items():
        if str(cid).upper() == 'NO_TRADE': return str(label)
    return None


def run(cfg: TeacherDistillCfg) -> dict[str, Any]:
    rows=_load_jsonl(cfg.input_jsonl)
    teacher=_load_teacher(cfg.teacher_predictions)
    out={'train':[], 'eval':[]}; skipped=Counter()
    for r in rows:
        d=str(r.get('date','')); sp=_split(d,cfg)
        if sp is None: continue
        t=teacher.get((d, int(r.get('signal_pos',-1) or -1)))
        if t is None:
            skipped['missing_teacher']+=1; continue
        wants_trade, teacher_side=_teacher_wants_trade(t)
        trade_label, trade_side=_resolve_trade_choice(r)
        no_label=_resolve_no_trade_choice(r)
        if trade_label is None or no_label is None:
            skipped['bad_choice_map']+=1; continue
        target = trade_label if wants_trade else no_label
        if wants_trade and cfg.skip_side_mismatch and trade_side != teacher_side:
            skipped['side_mismatch']+=1; continue
        nr=dict(r)
        nr['target']=target
        nr['teacher_prediction']=t.get('prediction')
        nr['teacher_source']='ridge_pairwise_union'
        nr['leakage_guard']={**(nr.get('leakage_guard') or {}), 'target_from_prior_walkforward_teacher': True, 'target_not_raw_future_utility': True}
        out[sp].append(nr)
    for split,path in [('train',cfg.train_output),('eval',cfg.eval_output)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in out[split]))
    summary={
        'config': asdict(cfg), 'rows': {k:len(v) for k,v in out.items()},
        'target_counts': {k: dict(Counter(r['target'] for r in v)) for k,v in out.items()},
        'semantic_target_counts': {k: dict(Counter((r.get('choice_map') or {}).get(r['target'], r['target']) for r in v)) for k,v in out.items()},
        'skipped': dict(skipped),
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', required=True); p.add_argument('--teacher-predictions', required=True)
    p.add_argument('--train-output', required=True); p.add_argument('--eval-output', required=True); p.add_argument('--summary-output', required=True)
    p.add_argument('--train-start', default=TeacherDistillCfg.train_start); p.add_argument('--train-end', default=TeacherDistillCfg.train_end)
    p.add_argument('--eval-start', default=TeacherDistillCfg.eval_start); p.add_argument('--eval-end', default=TeacherDistillCfg.eval_end)
    p.add_argument('--no-skip-side-mismatch', dest='skip_side_mismatch', action='store_false', default=TeacherDistillCfg.skip_side_mismatch)
    return p.parse_args()

if __name__ == '__main__':
    print(json.dumps(run(TeacherDistillCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))
