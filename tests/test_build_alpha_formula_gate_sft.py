import json
from pathlib import Path
from training.build_alpha_formula_gate_sft import build

def write(path: Path, value: dict):
    path.write_text(json.dumps(value))

def test_builder_preserves_stage_order_and_group_split(tmp_path):
    results=tmp_path/"results";results.mkdir();slug="high_volatility_demo"
    write(results/f"{slug}_preregistration_2026-01-01.json",{"policy_id":"DEMO","protocol_version":"v1","mechanism":{"claim":"x","side":"long"},"features":{"x":"prior"},"policy":{},"stopping_rule":"first failure"})
    write(results/f"{slug}_support_2026-01-01.json",{"policy_id":"DEMO","support_passed":True,"support":{"train":{"events":9}},"support_checks":{"train_minimum_events":True}})
    write(results/f"{slug}_gross9_novelty_2026-01-01.json",{"policy_id":"DEMO","advance_to_economic_outcomes":True,"gross9_sleeves":{"a":{"metrics":{},"checks":{"x":True},"passed":True}}})
    write(results/f"{slug}_train_economics_2026-01-01.json",{"policy_id":"DEMO","stage":"train","passed":False,"primary":{"base":{"absolute_return_pct":-1}},"checks":{"absolute_return_positive":False}})
    train=tmp_path/"train.jsonl";ev=tmp_path/"eval.jsonl";summary=tmp_path/"summary.json";report=build(results,train,ev,summary,eval_percent=50)
    rows=[json.loads(x) for x in (train.read_text()+ev.read_text()).splitlines()]
    assert [row["stage"] for row in rows]==["source","gross9","train"]
    assert [row["target"] for row in rows]==["TRADE","TRADE","NO_TRADE"]
    evidence=[row["prompt"].split("current_stage_evidence: ",1)[1] for row in rows]
    assert all('"passed"' not in text and '"checks"' not in text and '"decision"' not in text for text in evidence)
    assert not report["group_overlap"] and report["leakage_guard"]["future_stage_evidence_in_prompt"] is False

def test_builder_never_opens_stage_after_terminal_failure(tmp_path):
    results=tmp_path/"results";results.mkdir();slug="high_volatility_stop"
    write(results/f"{slug}_preregistration_2026-01-01.json",{"policy_id":"STOP","mechanism":{},"features":{},"policy":{}})
    write(results/f"{slug}_support_2026-01-01.json",{"policy_id":"STOP","support_passed":False,"support":{},"support_checks":{}})
    write(results/f"{slug}_gross9_novelty_2026-01-01.json",{"policy_id":"STOP","advance_to_economic_outcomes":True})
    train=tmp_path/"train";ev=tmp_path/"eval";summary=tmp_path/"summary";build(results,train,ev,summary)
    rows=[json.loads(x) for x in (train.read_text()+ev.read_text()).splitlines()]
    assert len(rows)==1 and rows[0]["stage"]=="source" and rows[0]["target"]=="NO_TRADE"
