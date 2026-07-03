import json

from training.build_rex_regime_thesis_sft import Gate, _target, _context


def test_gate_target_trades_only_when_all_signal_time_gates_match():
    row = {"feature_snapshot": {"range_vol": 0.03, "kimchi_premium_change": -0.01}, "action": {"side": "LONG"}}
    gates = (Gate("range_vol", ">=", 0.02), Gate("kimchi_premium_change", "<=", 0.0))
    target = _target(row, gates)
    assert target["decision"] == "TRADE"
    assert target["action_side"] == "LONG"


def test_gate_target_abstains_and_names_failed_regime():
    row = {"feature_snapshot": {"range_vol": 0.01, "kimchi_premium_change": 0.02}, "action": {"side": "SHORT"}}
    gates = (Gate("range_vol", ">=", 0.02), Gate("kimchi_premium_change", "<=", 0.0))
    target = _target(row, gates)
    assert target["decision"] == "ABSTAIN"
    assert target["action_side"] == "NONE"
    assert "range_vol" in target["rationale_class"]


def test_context_buckets_core_price_action_and_macro_features():
    ctx = _context({"range_vol": 0.03, "kimchi_premium_change": -0.1, "dxy_momentum": 0.0, "window_drawdown": 0.02})
    assert ctx["range_vol"] == "high"
    assert ctx["kimchi_premium_change"] == "non_positive"
    assert ctx["dxy_momentum"] == "nonnegative_or_flat"

from training.build_rex_regime_thesis_sft import _target_text


def test_target_text_can_emit_fast_decision_label():
    target = {"decision": "TRADE", "action_side": "LONG"}
    assert _target_text(target, "decision_label") == "TRADE"
    assert _target_text(target, "label_then_json").startswith("TRADE\n")
