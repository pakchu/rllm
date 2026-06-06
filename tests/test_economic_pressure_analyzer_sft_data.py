import json

from training.economic_pressure_analyzer_sft_data import convert_pressure_rows, pressure_prompt


def test_pressure_prompt_declares_allowed_values():
    p = pressure_prompt('{"regime":"RANGE"}')
    assert "direction_pressure" in p
    assert "LONG_FAVORED" in p


def test_convert_pressure_rows_outputs_single_key_target():
    rows = [{"date": "d", "signal_pos": 1, "prompt": "Past-only analyzer summary: {}", "analyzer_target": {"direction_pressure": "SHORT_FAVORED"}}]
    out = convert_pressure_rows(rows)
    assert out[0]["task"] == "path_pressure_analyzer_sft"
    assert json.loads(out[0]["target"]) == {"direction_pressure": "SHORT_FAVORED"}
