from pathlib import Path
import unittest


ACTIVE_PATHS = [
    Path("training/single_policy_sft_data.py"),
    Path("training/build_multiasset_regime_sft_data.py"),
    Path("training/build_multiasset_regime_preference_data.py"),
    Path("training/train_text_sft.py"),
    Path("docs/wave-execution-bridge.md"),
]

FORBIDDEN_ACTIVE_SNIPPETS = [
    "keys: analyzer, trader",
    "emit no-leak analyzer/trader",
    "monthly regime analyzer/trader",
    "trader stage for a BTCUSDT futures trading bot",
    "gate stage for a BTCUSDT futures trading bot",
]


class TestRllmResetContract(unittest.TestCase):
    def test_active_reset_path_does_not_reintroduce_two_stage_contract(self):
        offenders = []
        for path in ACTIVE_PATHS:
            text = path.read_text()
            lower = text.lower()
            for snippet in FORBIDDEN_ACTIVE_SNIPPETS:
                if snippet.lower() in lower:
                    offenders.append(f"{path}:{snippet}")
        self.assertEqual(offenders, [])

    def test_failure_analysis_doc_is_present(self):
        doc = Path("docs/rllm-reset-failure-analysis-2026-06-23.md")
        text = doc.read_text()
        self.assertIn("The two-stage `analyzer -> trader` architecture is deprecated", text)
        self.assertIn("no live `analyzer -> trader` cascade", text)


if __name__ == "__main__":
    unittest.main()
