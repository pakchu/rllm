import json

from training.eval_economic_path_shape_sft import parse_analyzer, parse_trader, analyzer_metrics, trader_metrics


def test_parse_analyzer_extracts_pressure_and_grades():
    pred = parse_analyzer(json.dumps({"direction_pressure": "LONG_FAVORED", "long_path": {"grade": "CLEAN_TARGET"}, "short_path": {"grade": "STOP_FIRST"}}))
    assert pred == {"direction_pressure": "LONG_FAVORED", "long_grade": "CLEAN_TARGET", "short_grade": "STOP_FIRST"}


def test_parse_trader_normalizes_no_trade_side():
    pred = parse_trader(json.dumps({"gate": "NO_TRADE", "side": "LONG", "target_pct": 1, "stop_pct": 0.6, "max_hold_bars": 144}))
    assert pred["gate"] == "NO_TRADE"
    assert pred["side"] == "NONE"


def test_metrics_score_exact_echo():
    analyzer_row = {"target": json.dumps({"direction_pressure": "SHORT_FAVORED", "long_path": {"grade": "STOP_FIRST"}, "short_path": {"grade": "CLEAN_TARGET"}})}
    apred = parse_analyzer(analyzer_row["target"])
    assert analyzer_metrics([analyzer_row], [apred])["per_key"]["direction_pressure"]["accuracy"] == 1.0
    trader_row = {"target": json.dumps({"gate": "TRADE", "side": "SHORT", "target_pct": 1.0, "stop_pct": 0.6, "max_hold_bars": 144})}
    tpred = parse_trader(trader_row["target"])
    assert trader_metrics([trader_row], [tpred])["exact_template_accuracy"] == 1.0


def test_parse_trader_repairs_side_in_gate_schema_slip():
    pred = parse_trader(json.dumps({"gate": "LONG", "side": "LONG", "target_pct": 1, "stop_pct": 0.6, "max_hold_bars": 144}))
    assert pred["gate"] == "TRADE"
    assert pred["side"] == "LONG"
