import unittest

from training.export_hybrid_policy_labels import _target_for_hybrid_record, format_hybrid_policy_book


class TestExportHybridPolicyLabels(unittest.TestCase):
    def test_target_prefers_base_before_addon(self):
        row = {"signal_pos": 10, "summary": {"regime": "UP", "risk_state": "CALM", "volume": "HIGH"}}
        base = {"regime=UP|risk_state=CALM": {"action": {"side": "LONG", "hold_bars": 48}}}
        addon = {"regime=UP|volume=HIGH": {"action": {"side": "SHORT", "hold_bars": 96}}}
        target, next_pos = _target_for_hybrid_record(
            row,
            base_rules=base,
            addon_rules=addon,
            base_key_fields=("regime", "risk_state"),
            addon_key_fields=("regime", "volume"),
            next_available_pos=-1,
        )
        self.assertEqual(target["side"], "LONG")
        self.assertEqual(target["reason"], "BASE_CALIBRATED_EDGE")
        self.assertEqual(next_pos, 58)

    def test_target_uses_addon_when_base_missing(self):
        row = {"signal_pos": 10, "summary": {"regime": "UP", "risk_state": "STRESS", "volume": "HIGH"}}
        addon = {"regime=UP|volume=HIGH": {"action": {"side": "SHORT", "hold_bars": 96}}}
        target, _ = _target_for_hybrid_record(
            row,
            base_rules={},
            addon_rules=addon,
            base_key_fields=("regime", "risk_state"),
            addon_key_fields=("regime", "volume"),
            next_available_pos=-1,
        )
        self.assertEqual(target["side"], "SHORT")
        self.assertEqual(target["reason"], "ADDON_CALIBRATED_EDGE")

    def test_policy_book_describes_priority(self):
        text = format_hybrid_policy_book({}, {})
        self.assertIn("Base policy has priority", text)
        self.assertIn("Add-on policy applies only", text)
