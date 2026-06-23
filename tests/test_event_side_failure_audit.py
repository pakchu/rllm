import unittest

from training.event_side_failure_audit import _summary, _trade_action


class TestEventSideFailureAudit(unittest.TestCase):
    def test_summary_reports_win_rate(self):
        s = _summary([1.0, -1.0, 2.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["win_rate"], 2 / 3)

    def test_trade_action_wait_for_no_trade(self):
        self.assertEqual(_trade_action({"prediction": {"gate": "NO_TRADE"}}), "WAIT")
        self.assertEqual(_trade_action({"prediction": {"gate": "TRADE", "side": "LONG"}}), "LONG")


if __name__ == "__main__":
    unittest.main()
