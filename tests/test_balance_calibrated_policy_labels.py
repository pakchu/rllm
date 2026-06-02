import json
import unittest

from training.balance_calibrated_policy_labels import balance_rows


def row(reason):
    return {"target": json.dumps({"reason": reason})}


class TestBalanceCalibratedPolicyLabels(unittest.TestCase):
    def test_balance_rows_repeats_trade_and_samples_negatives(self):
        rows = [row("CALIBRATED_EDGE"), row("NO_CALIBRATED_EDGE"), row("NO_CALIBRATED_EDGE"), row("POSITION_OPEN_SKIP")]
        balanced, summary = balance_rows(rows, trade_repeat=3, no_edge_per_trade=1, skip_per_trade=1, seed=1)
        self.assertEqual(summary["output_reason_counts"]["CALIBRATED_EDGE"], 3)
        self.assertEqual(summary["output_reason_counts"]["NO_CALIBRATED_EDGE"], 1)
        self.assertEqual(summary["output_reason_counts"]["POSITION_OPEN_SKIP"], 1)
        self.assertEqual(len(balanced), 5)


if __name__ == "__main__":
    unittest.main()
