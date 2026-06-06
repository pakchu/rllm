import json

from training.economic_path_shape_sft_data import convert_rows, trader_target


def _target(pressure):
    return {
        "direction_pressure": pressure,
        "template": {"target_pct": 1.0, "stop_pct": 0.6, "horizon_bars": 144},
        "long_path": {"grade": "CLEAN_TARGET"},
        "short_path": {"grade": "STOP_FIRST"},
    }


def test_trader_target_maps_pressure_to_template_action():
    assert trader_target(_target("LONG_FAVORED"))["side"] == "LONG"
    assert trader_target(_target("SHORT_FAVORED"))["side"] == "SHORT"
    assert trader_target(_target("NO_TRADE_FAVORED"))["gate"] == "NO_TRADE"


def test_convert_rows_builds_analyzer_and_trader_rows():
    rows = [{"date": "d", "signal_pos": 1, "prompt": "Past-only analyzer summary: {\"regime\":\"RANGE\"}", "analyzer_target": _target("LONG_FAVORED")}]
    analyzer, trader = convert_rows(rows)
    assert analyzer[0]["task"] == "path_shape_analyzer_sft"
    assert trader[0]["task"] == "path_shape_trader_sft"
    assert json.loads(trader[0]["target"])["side"] == "LONG"
    assert "Analyzer path-shape output" in trader[0]["prompt"]
