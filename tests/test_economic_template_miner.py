import json

from training.economic_template_miner import collect_rule_values, row_facts, split_rule


def _row(action, utility=0.01):
    return {
        "prompt": json.dumps(
            {
                "regime": "TREND",
                "trend_alignment": "BULL",
                "context_tags": ["ACTIVE"],
                "symbolic_features": {"Macro Dollar State": "MACRO_WEAK"},
            }
        ),
        "action": json.dumps(action),
        "utility": utility,
    }


def test_row_facts_extracts_analyzer_summary_tokens():
    facts = row_facts(_row({"gate": "TRADE", "side": "LONG", "hold_bars": 72}))
    assert "regime=TREND" in facts
    assert "trend_alignment=BULL" in facts
    assert "tag=ACTIVE" in facts
    assert "symbolic.Macro Dollar State=MACRO_WEAK" in facts


def test_collect_rule_values_skips_no_trade_and_pairs_action_with_facts():
    trade = _row({"gate": "TRADE", "side": "LONG", "hold_bars": 72}, 0.02)
    no_trade = _row({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, 0.0)
    vals = collect_rule_values([trade, no_trade], max_terms=1)
    assert vals
    assert all('"NO_TRADE"' not in key for key in vals)
    assert any("trend_alignment=BULL" in key for key in vals)
    assert set(next(iter(vals.values()))) == {0.02}


def test_split_rule_returns_action_and_fact_list():
    rule = '{"gate":"TRADE","hold_bars":72,"side":"LONG"} | regime=TREND & tag=ACTIVE'
    obj = split_rule(rule)
    assert obj["action"] == {"gate": "TRADE", "hold_bars": 72, "side": "LONG"}
    assert obj["facts"] == ["regime=TREND", "tag=ACTIVE"]
