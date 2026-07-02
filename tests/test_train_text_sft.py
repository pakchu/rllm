import json
import tempfile
import unittest
from pathlib import Path

from training.train_text_sft import (
    TextSFTConfig,
    build_prompt_completion_record,
    build_training_text,
    load_jsonl,
    train_text_sft,
)


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

    def test_prompt_completion_record_separates_loss_target(self):
        record = build_prompt_completion_record({"prompt": "P", "target": "T"})
        self.assertEqual(record["prompt"], [{"role": "user", "content": "P"}])
        self.assertEqual(record["completion"], [{"role": "assistant", "content": "T"}])

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

    def test_balanced_sampling_spreads_side_only_labels(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "side.jsonl"
            rows = [{"task": "side", "prompt": f"L{i}", "target": '{"side":"LONG"}'} for i in range(10)]
            rows += [{"task": "side", "prompt": f"S{i}", "target": '{"side":"SHORT"}'} for i in range(2)]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            sampled = load_jsonl(data, max_samples=4, sample_mode="balanced", seed=7)
            counts = {}
            for row in sampled:
                side = json.loads(row["target"])["side"]
                counts[side] = counts.get(side, 0) + 1
            self.assertEqual(counts, {"LONG": 2, "SHORT": 2})

    def test_balanced_sampling_spreads_compact_path_shape_labels(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "compact.jsonl"
            rows = []
            for i in range(8):
                rows.append({"task": "compact", "prompt": f"N{i}", "target": json.dumps({"action_path": "NONE", "horizon_policy": "SKIP_STEP", "risk_budget": "AVOID_OR_TINY"})})
            for i in range(2):
                rows.append({"task": "compact", "prompt": f"T{i}", "target": json.dumps({"action_path": "TREND", "horizon_policy": "LONG_STEP", "risk_budget": "SMALL"})})
                rows.append({"task": "compact", "prompt": f"F{i}", "target": json.dumps({"action_path": "FADE", "horizon_policy": "SHORT_STEP", "risk_budget": "NORMAL"})})
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            sampled = load_jsonl(data, max_samples=6, sample_mode="balanced", seed=7)
            actions = {json.loads(r["target"])["action_path"] for r in sampled}
            self.assertEqual(actions, {"NONE", "TREND", "FADE"})
            summary = train_text_sft(TextSFTConfig(model_name="gemma4", train_jsonl=str(data), output_dir=str(Path(td) / "out"), max_samples=6, sample_mode="balanced"), dry_run=True)
            self.assertIn("action_path=TREND", summary["target_counts"])

    def test_balanced_sampling_spreads_repaired_router_labels(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "repaired.jsonl"
            rows = []
            for i in range(8):
                rows.append({"task": "repaired", "prompt": f"N{i}", "target": json.dumps({"fade_warning": "NO_FADE_WARNING", "skip_reason": "NO_EDGE", "primary_route": "SKIP"})})
            for i in range(2):
                rows.append({"task": "repaired", "prompt": f"F{i}", "target": json.dumps({"fade_warning": "FADE_STRONG", "skip_reason": "TRADEABLE_FADE", "primary_route": "FADE"})})
                rows.append({"task": "repaired", "prompt": f"W{i}", "target": json.dumps({"fade_warning": "FADE_WATCH", "skip_reason": "LOW_CONFIDENCE", "primary_route": "SKIP"})})
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            sampled = load_jsonl(data, max_samples=6, sample_mode="balanced", seed=7)
            warnings = {json.loads(r["target"])["fade_warning"] for r in sampled}
            self.assertEqual(warnings, {"NO_FADE_WARNING", "FADE_STRONG", "FADE_WATCH"})
            summary = train_text_sft(TextSFTConfig(model_name="gemma4", train_jsonl=str(data), output_dir=str(Path(td) / "out"), max_samples=6, sample_mode="balanced"), dry_run=True)
            self.assertIn("fade_warning=FADE_STRONG", summary["target_counts"])

    def test_balanced_oversample_can_expand_tiny_direction_regime_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "direction.jsonl"
            rows = [
                {"task": "direction", "prompt": "P0", "target": json.dumps({"direction_regime": "HIGH_SCORE_WINS"})},
                {"task": "direction", "prompt": "P1", "target": json.dumps({"direction_regime": "LOW_SCORE_WINS"})},
                {"task": "direction", "prompt": "P2", "target": json.dumps({"direction_regime": "ABSTAIN"})},
            ]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            sampled = load_jsonl(data, max_samples=9, sample_mode="balanced_oversample", seed=7)
            self.assertEqual(len(sampled), 9)
            summary = train_text_sft(TextSFTConfig(model_name="gemma4", train_jsonl=str(data), output_dir=str(Path(td) / "out"), max_samples=9, sample_mode="balanced_oversample"), dry_run=True)
            self.assertEqual(summary["rows"], 9)
            self.assertIn("direction_regime=LOW_SCORE_WINS", summary["target_counts"])


if __name__ == "__main__":
    unittest.main()
