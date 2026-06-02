import unittest

from training.sweep_hybrid_stable_policy import _filter_uncovered_records, _merge_action_rows, _parse_indices


class TestSweepHybridStablePolicy(unittest.TestCase):
    def test_parse_indices_handles_empty_and_commas(self):
        self.assertEqual(_parse_indices(""), set())
        self.assertEqual(_parse_indices("1, 3"), {1, 3})

    def test_filter_uncovered_records_uses_base_summary_key(self):
        rows = [
            {"summary": {"regime": "UP", "risk_state": "CALM"}, "actions": {}},
            {"summary": {"regime": "DOWN", "risk_state": "CALM"}, "actions": {}},
        ]
        base = {"regime=UP|risk_state=CALM": {"action": {"side": "LONG", "hold_bars": 48}}}
        out = _filter_uncovered_records(rows, base, ("regime", "risk_state"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["summary"]["regime"], "DOWN")

    def test_merge_action_rows_keeps_both_sources(self):
        merged = _merge_action_rows(
            {"a": {"LONG_48": [{"signal_pos": 1}]}},
            {"a": {"LONG_48": [{"signal_pos": 2}]}, "b": {"SHORT_48": [{"signal_pos": 3}]}},
        )
        self.assertEqual([x["signal_pos"] for x in merged["a"]["LONG_48"]], [1, 2])
        self.assertEqual(merged["b"]["SHORT_48"][0]["signal_pos"], 3)
