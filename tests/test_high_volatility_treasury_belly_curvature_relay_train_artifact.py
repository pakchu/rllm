import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_treasury_belly_curvature_relay_train_economics_2026-08-10.json")


def test_train_result_is_terminal_and_later_stages_remain_sealed():
    payload = json.loads(RESULT.read_text())
    assert payload["stage"] == "train"
    assert payload["passed"] is False
    assert payload["decision"] == "terminal_reject_no_repair"
    assert payload["later_stage_outcomes_opened"] is False
    assert payload["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert payload["checks"]["cluster_signflip_p_max_0_1"] is False
    assert payload["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "7b2bc6f9afc14b322a78f341a6d5f741dc3bced5e10fae6434fa70edfd66cfa9"
