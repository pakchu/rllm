import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_trade_arrival_memory_continuation_train_economics_2026-08-10.json")


def test_terminal_train_reject_is_frozen_and_later_stages_stay_sealed():
    artifact = json.loads(RESULT.read_text())
    assert artifact["stage"] == "train"
    assert artifact["passed"] is False
    assert artifact["advance_to_next_stage"] is False
    assert artifact["later_stage_outcomes_opened"] is False
    assert artifact["decision"] == "terminal_reject_no_repair"
    assert artifact["primary"]["base"]["absolute_return_pct"] == -0.8019240576789044
    assert artifact["primary"]["base"]["mean_gross_underlying_bp"] == 9.807811212969467
    assert artifact["primary"]["cluster_signflip"]["pvalue"] == 0.5887741122588774
    assert artifact["primary"]["stress"]["absolute_return_pct"] == -3.3486076765281925
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "c12c998b2b4fa1c2dd03780eb92d8e05c3f273dc468d2fb64e047275a4c36808"
