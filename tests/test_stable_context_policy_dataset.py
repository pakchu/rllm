import json
import unittest

from training.stable_context_policy_dataset import StableContextPolicyCfg, context_id, select_contexts, transform_rows


def row(split, token, long_ret, short_ret):
    return {
        "split": split,
        "prompt": "P",
        "target": '{"action":"NO_TRADE"}',
        "state_tokens": {"trend_alignment": token, "risk_state": "normal"},
        "reward_audit": {"LONG": {"net_return_pct": long_ret}, "SHORT": {"net_return_pct": short_ret}},
    }


class TestStableContextPolicyDataset(unittest.TestCase):
    def test_context_id_is_stable(self):
        r = row("train", "aligned_up", 1, -1)
        self.assertEqual(context_id(r, ("trend_alignment", "risk_state")), "trend_alignment=aligned_up|risk_state=normal")

    def test_selects_only_train_test_stable_context(self):
        rows = [row("train", "aligned_up", 0.4, -0.1) for _ in range(4)]
        rows += [row("test", "aligned_up", 0.2, -0.1) for _ in range(2)]
        rows += [row("eval", "aligned_up", -1.0, 1.0) for _ in range(2)]  # must not affect selection
        cfg = StableContextPolicyCfg(input_jsonl="i", output="o", context_keys="trend_alignment,risk_state", min_train_rows=4, min_test_rows=2, min_train_mean_pct=0.1, min_test_mean_pct=0.1, min_train_gap_pct=0.1)
        selected, diag = select_contexts(rows, cfg)
        self.assertEqual(diag["selected_contexts"], 1)
        self.assertEqual(next(iter(selected.values()))["action"], "LONG")

    def test_transform_abstains_unselected(self):
        rows = [row("train", "mixed", -0.1, -0.2)]
        cfg = StableContextPolicyCfg(input_jsonl="i", output="o", context_keys="trend_alignment,risk_state")
        out = transform_rows(rows, {}, cfg)
        target = json.loads(out[0]["target"])
        self.assertEqual(target["action"], "NO_TRADE")
        self.assertFalse(out[0]["context_selection"]["selected"])


if __name__ == "__main__":
    unittest.main()
