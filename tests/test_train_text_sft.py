import json
import tempfile
import unittest
from pathlib import Path

from training.train_text_sft import TextSFTConfig, build_training_text, load_jsonl, train_text_sft


class TestTrainTextSFT(unittest.TestCase):
    def test_dry_run_summarizes_trader_jsonl_without_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "trader.jsonl"
            out = Path(td) / "ckpt"
            rows = [
                {"task": "trader", "prompt": "Analyzer summary: {}", "target": '{"gate":"NO_TRADE","side":"NONE"}'},
                {"task": "trader", "prompt": "Analyzer summary: {x}", "target": '{"gate":"TRADE","side":"LONG"}'},
            ]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            summary = train_text_sft(TextSFTConfig(model_name="gemma4", train_jsonl=str(data), output_dir=str(out)), dry_run=True)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["model_name"], "google/gemma-4-E4B-it")
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["tasks"], {"trader": 2})
            self.assertTrue((out / "sft_summary.json").exists())

    def test_build_training_text_uses_user_and_assistant_boundaries(self):
        text = build_training_text({"prompt": "P", "target": "T"})
        self.assertIn("<|user|>", text)
        self.assertIn("<|assistant|>", text)
        self.assertTrue(text.endswith("T"))

    def test_balanced_sampling_spreads_trader_labels(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "trader.jsonl"
            rows = []
            for i in range(10):
                rows.append({"task": "trader", "prompt": f"P{i}", "target": '{"gate":"NO_TRADE","side":"NONE"}'})
            for i in range(2):
                rows.append({"task": "trader", "prompt": f"L{i}", "target": '{"gate":"TRADE","side":"LONG"}'})
                rows.append({"task": "trader", "prompt": f"S{i}", "target": '{"gate":"TRADE","side":"SHORT"}'})
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            sampled = load_jsonl(data, max_samples=6, sample_mode="balanced", seed=7)
            targets = {json.loads(r["target"])["side"] for r in sampled}
            self.assertEqual(targets, {"NONE", "LONG", "SHORT"})


if __name__ == "__main__":
    unittest.main()
