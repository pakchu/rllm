import unittest

from training.eval_vlm_policy import select_action_from_scores, summarize_action_metrics


class TestEvalVlmPolicy(unittest.TestCase):
    def test_summarize_action_metrics(self):
        out = summarize_action_metrics(
            targets=["BUY", "HOLD", "SELL", "BUY"],
            predictions=["BUY", "SELL", "SELL", "HOLD"],
        )
        self.assertEqual(out["num_samples"], 4)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["balanced_recall"], 0.5)
        self.assertAlmostEqual(out["directional_recall_mean"], 0.75)
        self.assertAlmostEqual(out["directional_recall_gap"], 0.5)
        self.assertEqual(out["confusion"]["BUY"]["BUY"], 1)
        self.assertEqual(out["confusion"]["BUY"]["HOLD"], 1)
        self.assertEqual(out["confusion"]["SELL"]["SELL"], 1)

    def test_select_action_from_scores_with_bias(self):
        pred0, adj0 = select_action_from_scores({"BUY": 0.1, "HOLD": 0.2, "SELL": 0.0})
        self.assertEqual(pred0, "HOLD")
        pred1, adj1 = select_action_from_scores(
            {"BUY": 0.1, "HOLD": 0.2, "SELL": 0.0},
            action_biases={"BUY": 0.2},
        )
        self.assertEqual(pred1, "BUY")
        self.assertGreater(adj1["BUY"], adj0["BUY"])

    def test_select_action_from_scores_trade_gate(self):
        pred, adj = select_action_from_scores(
            {"TRADE": 0.1, "NO_TRADE": 0.2},
            action_biases={"TRADE": 0.15},
            labels=("TRADE", "NO_TRADE"),
        )
        self.assertEqual(pred, "TRADE")
        self.assertGreater(adj["TRADE"], adj["NO_TRADE"])


if __name__ == "__main__":
    unittest.main()
