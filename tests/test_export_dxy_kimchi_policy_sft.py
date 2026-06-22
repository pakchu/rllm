import json
import tempfile
import unittest
from pathlib import Path

from training.export_dxy_kimchi_policy_sft import ExportDxyKimchiSftCfg, _balanced_train, run


def row(split, i, activate, reason="no_prior_signal", action="NO_TRADE"):
    if activate:
        target = {"activate": True, "action": action, "exit_profile": "FAST", "confidence": "MEDIUM", "reason_code": "prior_signal_path_reward_ok"}
    else:
        target = {"activate": False, "action": "NO_TRADE", "exit_profile": "AVOID", "confidence": "LOW", "reason_code": reason}
    return {"task": "dxy_kimchi_regime_policy_sft", "split": split, "date": f"2025-01-{i+1:02d}", "signal_pos": i, "prompt": "prompt", "target": json.dumps(target)}


class TestExportDxyKimchiPolicySft(unittest.TestCase):
    def test_balanced_train_limits_no_trade_rows(self):
        rows = [row("train", i, True, action="LONG") for i in range(2)] + [row("train", i + 10, False) for i in range(20)]
        selected = _balanced_train(rows, no_trade_per_activate=2.0, seed=1)
        active = sum(json.loads(r["target"])["activate"] for r in selected)
        self.assertEqual(active, 2)
        self.assertLessEqual(len(selected), 6)

    def test_run_writes_train_test_eval_messages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in.jsonl"
            rows = [row("train", 0, True, action="LONG"), row("train", 1, False), row("test", 2, False), row("eval", 3, True, action="SHORT")]
            inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            cfg = ExportDxyKimchiSftCfg(input_jsonl=str(inp), train_output=str(root / "train.jsonl"), test_output=str(root / "test.jsonl"), eval_output=str(root / "eval.jsonl"), summary_output=str(root / "summary.json"))
            report = run(cfg)
            self.assertEqual(report["raw_counts"], {"train": 2, "test": 1, "eval": 1})
            first = json.loads((root / "train.jsonl").read_text().splitlines()[0])
            self.assertEqual(first["messages"][2]["role"], "assistant")
            self.assertIn("prompt", first)
            self.assertIn("target", first)
            self.assertIn("activate", json.loads(first["messages"][2]["content"]))
            self.assertEqual(json.loads(first["target"]), json.loads(first["messages"][2]["content"]))


if __name__ == "__main__":
    unittest.main()
