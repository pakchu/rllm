import json
import unittest

from training.export_calibrated_policy_labels import _policy_target_for_record, build_policy_trader_input, format_policy_book


class TestExportCalibratedPolicyLabels(unittest.TestCase):
    def test_policy_target_marks_rule_trade_and_overlap_skip(self):
        rules = {"k": {"action": {"side": "LONG", "hold_bars": 48}}}
        target, next_pos = _policy_target_for_record({"signal_pos": 10, "key": "k"}, rules, next_available_pos=-1)
        self.assertEqual(target["gate"], "TRADE")
        self.assertEqual(target["side"], "LONG")
        self.assertEqual(next_pos, 58)
        skipped, _ = _policy_target_for_record({"signal_pos": 12, "key": "k"}, rules, next_available_pos=next_pos)
        self.assertEqual(skipped["reason"], "POSITION_OPEN_SKIP")

    def test_trader_prompt_demands_exact_json_policy_output(self):
        book = format_policy_book({"regime=RANGE": {"action": {"side": "LONG", "hold_bars": 48}}})
        prompt = build_policy_trader_input(
            json.dumps({"regime": "RANGE"}),
            hold_candidates=(48, 96),
            entry_delay_bars=1,
            current_policy_key="regime=RANGE",
            policy_book=book,
        )
        self.assertIn("Imitate the train-calibrated symbolic policy", prompt)
        self.assertIn("Calibrated policy book", prompt)
        self.assertIn("Current policy_key: regime=RANGE", prompt)
        self.assertIn("hold_bars", prompt)


if __name__ == "__main__":
    unittest.main()
