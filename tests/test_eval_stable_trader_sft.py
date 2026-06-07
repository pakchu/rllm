from training.eval_stable_trader_sft import metrics, parse_stable_trader_json


def test_parse_stable_trader_json_normalizes_bad_action():
    assert parse_stable_trader_json('{"action":"HOLD","risk":"LOW"}') == {"action": "NO_TRADE", "risk": "LOW"}


def test_metrics_counts_action_accuracy():
    rows = [{"target": '{"action":"LONG","risk":"MEDIUM"}'}, {"target": '{"action":"NO_TRADE","risk":"HIGH"}'}]
    m = metrics(rows, [{"action": "LONG", "risk": "HIGH"}, {"action": "NO_TRADE", "risk": "HIGH"}])
    assert m["action_accuracy"] == 1.0
    assert m["exact_accuracy"] == 0.5

from training.eval_stable_trader_sft import _candidate_json


def test_candidate_json_matches_sorted_target_shape():
    assert _candidate_json("LONG", "MEDIUM") == '{"action": "LONG", "risk": "MEDIUM"}'
