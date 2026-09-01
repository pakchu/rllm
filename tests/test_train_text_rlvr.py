import json
import tempfile
import unittest
from pathlib import Path

from training.train_text_rlvr import (
    TextRLVRConfig,
    _prompt_token_length,
    apply_reward_variance_guard,
    build_reward_functions,
    exact_target_reward,
    make_economic_utility_reward,
    make_action_utility_reward,
    make_residual_utility_reward,
    load_jsonl,
    ordinal_distance_reward,
    parse_args,
    sample_rows,
    train_text_rlvr,
)


class TestTextRLVR(unittest.TestCase):
    def _jsonl(self, directory: str, rows: list[dict[str, object]]) -> Path:
        path = Path(directory) / "train.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_prompt_token_length_uses_chat_generation_template(self):
        class Tokenizer:
            chat_template = "template"

            @staticmethod
            def apply_chat_template(messages, *, tokenize, add_generation_prompt):
                assert messages == [{"role": "user", "content": "prompt"}]
                assert tokenize is True
                assert add_generation_prompt is True
                return [1, 2, 3]

        self.assertEqual(_prompt_token_length(Tokenizer(), "prompt"), 3)

    def test_prompt_token_length_accepts_batch_encoding_shape(self):
        class Tokenizer:
            chat_template = "template"

            @staticmethod
            def apply_chat_template(*_args, **_kwargs):
                return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

        self.assertEqual(_prompt_token_length(Tokenizer(), "prompt"), 4)

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

    def test_action_utility_reward_scores_selected_route(self):
        reward = make_action_utility_reward(0.01)
        utilities = {"SKIP": 0.0, "TP4": 0.01, "TP12": -0.005}
        self.assertEqual(
            reward(["SKIP", "TP4", "TP12", "BAD"], action_utilities=[utilities] * 4),
            [0.0, 1.0, -0.5, -1.0],
        )

    def test_residual_utility_reward_scores_switch_relative_to_keep(self):
        reward = make_residual_utility_reward(0.01)
        positive = {"KEEP": 0.0, "SWITCH": 0.006}
        negative = {"KEEP": 0.0, "SWITCH": -0.004}
        self.assertEqual(
            reward(
                ["KEEP", "SWITCH", "SWITCH", "BAD"],
                residual_utilities=[positive, positive, negative, positive],
                target=["SWITCH", "SWITCH", "KEEP", "SWITCH"],
            ),
            [0.0, 0.6, -0.4, -1.0],
        )

    def test_residual_utility_reward_breaks_exact_ties_toward_keep(self):
        reward = make_residual_utility_reward(0.01)
        tied = {"KEEP": 0.0, "SWITCH": 0.0}
        self.assertEqual(
            reward(
                ["KEEP", "SWITCH"],
                residual_utilities=[tied, tied],
                target=["KEEP", "KEEP"],
            ),
            [0.0, -0.01],
        )

    def test_residual_target_utility_schema_combines_both_verifiers(self):
        rewards = build_reward_functions(
            "pposm_residual_target_utility", utility_scale=0.01
        )
        self.assertEqual(
            [reward.__name__ for reward in rewards],
            ["format_reward", "exact_target_reward", "residual_utility_reward"],
        )
        utilities = [{"KEEP": 0.0, "SWITCH": 0.006}]
        self.assertEqual(rewards[1](["SWITCH"], target=["SWITCH"]), [1.0])
        self.assertEqual(
            rewards[2](
                ["SWITCH"], residual_utilities=utilities, target=["SWITCH"]
            ),
            [0.6],
        )

    def test_residual_target_utility_loader_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._jsonl(
                td,
                [
                    {
                        "prompt": "candidate SKIP vs TP4",
                        "target": "KEEP",
                        "metadata": {
                            "residual_utilities": {"KEEP": 0.0, "SWITCH": -0.2}
                        },
                    }
                ],
            )
            rows = load_jsonl(
                path, label_schema="pposm_residual_target_utility"
            )
            self.assertEqual(
                rows[0]["residual_utilities"], {"KEEP": 0.0, "SWITCH": -0.2}
            )

    def test_residual_utility_loader_preserves_pairwise_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._jsonl(
                td,
                [
                    {
                        "prompt": "candidate SKIP vs default TP4",
                        "target": "SWITCH",
                        "metadata": {"residual_utilities": {"KEEP": 0.0, "SWITCH": 0.012}},
                    }
                ],
            )
            rows = load_jsonl(path, label_schema="pposm_residual_utility")
            self.assertEqual(
                rows[0]["residual_utilities"],
                {"KEEP": 0.0, "SWITCH": 0.012},
            )

    def test_residual_utility_loader_rejects_malformed_metadata(self):
        cases = [
            ({"SWITCH": 0.01}, "residual_utilities must match labels"),
            ({"KEEP": 0.0, "SWITCH": "nan"}, "non-finite residual utility"),
            ({"KEEP": 0.001, "SWITCH": 0.01}, "residual_utilities.KEEP must be 0"),
        ]
        for utilities, message in cases:
            with self.subTest(utilities=utilities):
                with tempfile.TemporaryDirectory() as td:
                    path = self._jsonl(
                        td,
                        [
                            {
                                "prompt": "candidate TP12 vs default TP4",
                                "target": "KEEP",
                                "metadata": {"residual_utilities": utilities},
                            }
                        ],
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_jsonl(path, label_schema="pposm_residual_utility")

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

    def test_residual_utility_dry_run_writes_schema_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            train = self._jsonl(
                td,
                [
                    {
                        "prompt": "candidate SKIP vs default TP4",
                        "target": "KEEP",
                        "metadata": {"residual_utilities": {"KEEP": 0.0, "SWITCH": -0.003}},
                    },
                    {
                        "prompt": "candidate TP12 vs default TP4",
                        "target": "SWITCH",
                        "metadata": {"residual_utilities": {"KEEP": 0.0, "SWITCH": 0.006}},
                    },
                ],
            )
            output = Path(td) / "out"
            result = train_text_rlvr(
                TextRLVRConfig(
                    base_model="/models/local-base",
                    sft_adapter_dir="/models/local-adapter",
                    train_jsonl=str(train),
                    output_dir=str(output),
                    label_schema="pposm_residual_utility",
                    local_files_only=True,
                    seed=17,
                ),
                dry_run=True,
            )
            self.assertEqual(result["allowed_labels"], ["KEEP", "SWITCH"])
            reward = json.loads((output / "reward_diagnostics.json").read_text())
            self.assertEqual(
                reward["reward_functions"],
                ["format_reward", "residual_utility_reward"],
            )
            self.assertEqual(reward["residual_tie_switch_penalty"], 0.01)
            self.assertEqual(
                reward["deterministic_reward_matrix"]["KEEP"]["KEEP"][
                    "residual_utility_reward"
                ],
                0.0,
            )
            self.assertEqual(
                reward["deterministic_reward_matrix"]["KEEP"]["SWITCH"][
                    "residual_utility_reward"
                ],
                1.0,
            )

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

    def test_balanced_oversample_is_deterministic_and_balanced(self):
        rows = ([{"prompt": "a", "target": "A"}] * 1) + ([{"prompt": "b", "target": "B"}] * 5)
        first = sample_rows(rows, mode="balanced_oversample", max_samples=12, seed=7)
        second = sample_rows(rows, mode="balanced_oversample", max_samples=12, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["target"] == "A" for row in first), 6)
        self.assertEqual(sum(row["target"] == "B" for row in first), 6)

    def test_conversational_completion_extracts_assistant_content(self):
        format_reward, target_reward = build_reward_functions("ordinal")[:2]
        completion = [[{"role": "assistant", "content": "Q3"}]]
        self.assertEqual(format_reward(completion), [0.2])
        self.assertEqual(target_reward(completion, target=["Q3"]), [1.0])


if __name__ == "__main__":
    unittest.main()
