import json
import tempfile
import unittest
from pathlib import Path

from training.train_text_dpo import _dataset_rows, load_preference_jsonl, train_text_dpo, TextDPOConfig


class TestTrainTextDPO(unittest.TestCase):
    def _write_rows(self, path: Path):
        rows = [
            {"prompt": "p1", "chosen": '{"gate":"TRADE","side":"LONG","hold_bars":48}', "rejected": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}'},
            {"prompt": "p2", "chosen": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}', "rejected": '{"gate":"TRADE","side":"SHORT","hold_bars":48}'},
            {"prompt": "p3", "chosen": '{"gate":"TRADE","side":"SHORT","hold_bars":96}', "rejected": '{"gate":"TRADE","side":"LONG","hold_bars":96}'},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_load_preference_jsonl_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pref.jsonl"
            self._write_rows(path)
            rows = load_preference_jsonl(path, max_samples=2, sample_mode="balanced")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("prompt" in r and "chosen" in r and "rejected" in r for r in rows))

    def test_load_preference_jsonl_gate_balanced_overweights_no_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pref.jsonl"
            rows = []
            for i in range(10):
                rows.append({"prompt": f"nt{i}", "chosen": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}', "rejected": '{"gate":"TRADE","side":"LONG","hold_bars":48}'})
            for i in range(30):
                side = "LONG" if i % 2 == 0 else "SHORT"
                hold = [48, 96, 144][i % 3]
                rows.append({"prompt": f"t{i}", "chosen": f'{{"gate":"TRADE","side":"{side}","hold_bars":{hold}}}', "rejected": '{"gate":"NO_TRADE","side":"NONE","hold_bars":0}'})
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            selected = load_preference_jsonl(path, max_samples=12, sample_mode="gate_balanced")
            gates = [json.loads(r["chosen"])["gate"] for r in selected]
            self.assertEqual(gates.count("NO_TRADE"), 6)
            self.assertEqual(gates.count("TRADE"), 6)

    def test_dataset_rows_use_conversational_prompt_chosen_rejected(self):
        rows = [{"prompt": "p", "chosen": "c", "rejected": "r", "extra": 1}]
        self.assertEqual(
            _dataset_rows(rows),
            [
                {
                    "prompt": [{"role": "user", "content": "p"}],
                    "chosen": [{"role": "assistant", "content": "c"}],
                    "rejected": [{"role": "assistant", "content": "r"}],
                }
            ],
        )

    def test_dry_run_writes_summary_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pref.jsonl"
            out = Path(tmp) / "ckpt"
            self._write_rows(path)
            report = train_text_dpo(
                TextDPOConfig(
                    model_name="gemma4",
                    train_jsonl=str(path),
                    output_dir=str(out),
                    max_samples=2,
                    sample_mode="balanced",
                    max_steps=3,
                    init_adapter_dir="checkpoints/schema_adapter",
                ),
                dry_run=True,
            )
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["config"]["init_adapter_dir"], "checkpoints/schema_adapter")
            self.assertTrue((out / "dpo_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
