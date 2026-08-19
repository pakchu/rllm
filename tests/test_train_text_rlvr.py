import json
import tempfile
import unittest
from pathlib import Path

from training.train_text_rlvr import (
    TextRLVRConfig,
    apply_reward_variance_guard,
    build_reward_functions,
    exact_target_reward,
    make_economic_utility_reward,
    load_jsonl,
    ordinal_distance_reward,
    parse_args,
    train_text_rlvr,
)


class TestTextRLVR(unittest.TestCase):
    def _jsonl(self, directory: str, rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "train.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_pair_rewards_require_bare_allowed_label_and_exact_target(self):
        format_reward, target_reward = build_reward_functions("pair")
        completions = ["A", " B", "C", [{"role": "assistant", "content": "B"}]]
        targets = ["A", "B", "A", "B"]
        self.assertEqual(format_reward(completions), [0.2, -0.5, -0.5, 0.2])
        self.assertEqual(target_reward(completions, target=targets), [1.0, 0.0, 0.0, 1.0])

    def test_gate_schema_uses_exact_trade_tokens(self):
        format_reward, target_reward = build_reward_functions("gate")
        completions = ["TRADE", "NO_TRADE", "HOLD"]
        self.assertEqual(format_reward(completions), [0.2, 0.2, -0.5])
        self.assertEqual(
            target_reward(completions, target=["TRADE", "TRADE", "NO_TRADE"]),
            [1.0, 0.0, 0.0],
        )

    def test_pposm_state_schema_is_exact_and_nonconstant(self):
        format_reward, target_reward = build_reward_functions("pposm_state")
        completions = ["SKIP", "TP4", "TP12"]
        self.assertEqual(format_reward(completions), [0.2, 0.2, 0.2])
        self.assertEqual(
            target_reward(completions, target=["SKIP", "TP12", "TP12"]),
            [1.0, 0.0, 1.0],
        )

    def test_economic_utility_reward_prefers_trade_only_for_positive_edge(self):
        reward = make_economic_utility_reward(0.01)
        self.assertEqual(
            reward(["TRADE", "NO_TRADE", "TRADE", "BAD"], utility=[0.01, 0.01, -0.005, 0.01]),
            [1.0, 0.0, -0.5, -1.0],
        )

    def test_gate_utility_loader_preserves_train_net_return(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.jsonl"
            path.write_text(json.dumps({"prompt": "p", "target": "TRADE", "metadata": {"net_return": 0.012}}) + "\n")
            rows = load_jsonl(path, label_schema="gate_utility")
            self.assertEqual(rows[0]["utility"], 0.012)

    def test_ordinal_distance_reward_is_monotone_and_invalid_is_zero(self):
        rewards = ordinal_distance_reward(
            ["Q2", "Q1", "Q0", "Q4", "Qx"],
            target=["Q2"] * 5,
        )
        self.assertEqual(rewards, [1.0, 0.5, 0.0, 0.0, -1.0])
        self.assertEqual(len(build_reward_functions("ordinal")), 3)

    def test_jsonl_validation_rejects_wrong_schema_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._jsonl(td, [{"prompt": "choose", "target": "Q0"}])
            with self.assertRaisesRegex(ValueError, "target must be exactly"):
                load_jsonl(path, label_schema="pair")

    def test_jsonl_max_samples_is_deterministic_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._jsonl(
                td,
                [
                    {"prompt": "one", "target": "A"},
                    {"prompt": "two", "target": "B"},
                ],
            )
            self.assertEqual(load_jsonl(path, label_schema="pair", max_samples=1), [{"prompt": "one", "target": "A"}])

    def test_variance_guard_auto_and_error(self):
        generations, notes = apply_reward_variance_guard(num_generations=1, dataset_size=3, mode="auto")
        self.assertEqual(generations, 2)
        self.assertTrue(notes)
        with self.assertRaisesRegex(ValueError, "num_generations >= 2"):
            apply_reward_variance_guard(num_generations=1, dataset_size=3, mode="error")

    def test_dry_run_writes_all_diagnostics_without_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as td:
            train = self._jsonl(
                td,
                [
                    {"prompt": "first", "target": "Q0"},
                    {"prompt": "second", "target": "Q4"},
                ],
            )
            output = Path(td) / "out"
            result = train_text_rlvr(
                TextRLVRConfig(
                    base_model="/models/local-base",
                    sft_adapter_dir="/models/local-adapter",
                    train_jsonl=str(train),
                    output_dir=str(output),
                    label_schema="ordinal",
                    local_files_only=True,
                    seed=17,
                ),
                dry_run=True,
            )
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["target_counts"], {"Q0": 1, "Q4": 1})
            for name in ("config_diagnostics.json", "reward_diagnostics.json", "gradient_diagnostics.json"):
                self.assertTrue((output / name).exists(), name)
            reward = json.loads((output / "reward_diagnostics.json").read_text())
            self.assertEqual(reward["reward_functions"], ["format_reward", "exact_target_reward", "ordinal_distance_reward"])

    def test_cli_has_train_only_and_verification_options(self):
        args = parse_args(
            [
                "--base-model", "/base",
                "--sft-adapter-dir", "/adapter",
                "--train-jsonl", "train.jsonl",
                "--output-dir", "out",
                "--label-schema", "pair",
                "--local-files-only",
                "--require-nonzero-reward-std",
                "--require-nonzero-gradient",
                "--seed", "9",
            ]
        )
        self.assertTrue(args.local_files_only)
        self.assertTrue(args.require_nonzero_reward_std)
        self.assertTrue(args.require_nonzero_gradient)
        self.assertEqual(args.seed, 9)
        self.assertFalse(hasattr(args, "eval_jsonl"))

    def test_exact_target_reward_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            exact_target_reward(["A"], target=["A", "B"])

    def test_conversational_completion_extracts_assistant_content(self):
        format_reward, target_reward = build_reward_functions("ordinal")[:2]
        completion = [[{"role": "assistant", "content": "Q3"}]]
        self.assertEqual(format_reward(completion), [0.2])
        self.assertEqual(target_reward(completion, target=["Q3"]), [1.0])


if __name__ == "__main__":
    unittest.main()
