import hashlib
import json

from training import evaluate_mexican_peso_risk_barometer_relay_economics as economics


def test_mxrbr_train_failure_is_terminal_and_later_stages_are_sealed():
    path = economics.OUTPUTS["train"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "8957f2b6fd703a86ccf2c3fecf0ead9c1e827035ebea1c38347d990371b02e68"
    )
    result = json.loads(path.read_text())
    assert result["policy_id"] == "MXRBR-12"
    assert result["stage"] == "train"
    assert result["passed"] is False
    assert result["advance_to_next_stage"] is False
    assert result["decision"] == "terminal_reject_no_repair"
    assert result["later_stage_outcomes_opened"] is False
    assert result["primary"]["base"]["trades"] == 25
    assert result["primary"]["base"]["absolute_return_pct"] > 0
    assert result["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert result["checks"]["cluster_signflip_p_max_0_1"] is False
    assert result["checks"]["each_calendar_half_positive"] is False
    assert result["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
    assert not economics.OUTPUTS["test"].exists()
    assert not economics.OUTPUTS["eval"].exists()
    assert not economics.OUTPUTS["final"].exists()


def test_mxrbr_train_report_is_hash_bound():
    result = json.loads(economics.OUTPUTS["train"].read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == economics.canonical_hash(core)
    assert result["novelty_authorization"]["sha256"] == economics.NOVELTY_SHA
