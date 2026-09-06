import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_doi_research_attention_relay_economics as evaluator


ARTIFACT = Path("results/high_volatility_doi_research_attention_relay_train_economics_2026-08-12.json")


def test_hvdra_train_economics_is_terminal_and_reproducible():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "86181944c03372b42f490e6a98c7d9fb72893dc10d7c4eb495f7a273a2d379a8"
    )
    payload = json.loads(ARTIFACT.read_text())
    assert payload["policy_id"] == "HVDRA-24"
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["advance_to_next_stage"] is False
    assert payload["physical_rows_opened"]["primary_clock"] == 43
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == evaluator.canonical_hash(core)


def test_hvdra_later_stage_artifacts_do_not_exist():
    for stage in ("test", "eval", "final"):
        assert not evaluator.OUTPUTS[stage].exists()
