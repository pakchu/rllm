import json

from training.economic_compact_pressure_analyzer_sft_data import compact_summary_from_prompt, compact_pressure_prompt, convert_compact_pressure_rows


def test_compact_summary_keeps_core_fields_and_drops_sequence_text():
    summary = {
        "regime": "RANGE",
        "trend_alignment": "BULL_STACK",
        "evidence": {"momentum_1h_pct": 0.1, "unused": 9},
        "sequence_stats": {"flat": 15},
        "recent_bar_sequence": ["NOISY"] * 16,
        "symbolic_features": {"Macro Dollar State": "DOLLAR_WEAKNESS"},
        "context_tags": ["A", "B"],
    }
    compact = compact_summary_from_prompt("Past-only analyzer summary: " + json.dumps(summary))
    assert compact["state"]["regime"] == "RANGE"
    assert compact["evidence"] == {"momentum_1h_pct": 0.1}
    assert "recent_bar_sequence" not in json.dumps(compact)


def test_compact_pressure_prompt_declares_schema():
    prompt = compact_pressure_prompt({"state": {"regime": "RANGE"}})
    assert "direction_pressure" in prompt
    assert "LONG_FAVORED" in prompt


def test_convert_compact_pressure_rows_outputs_single_label():
    row = {"date": "d", "signal_pos": 1, "prompt": "Past-only analyzer summary: {}", "analyzer_target": {"direction_pressure": "LONG_FAVORED"}}
    out = convert_compact_pressure_rows([row])
    assert out[0]["task"] == "compact_path_pressure_analyzer_sft"
    assert json.loads(out[0]["target"]) == {"direction_pressure": "LONG_FAVORED"}
