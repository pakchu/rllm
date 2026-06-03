import json
import tempfile
import unittest
from pathlib import Path

from training.balance_decision_sft import balance_decision_rows, build_balanced_decision_sft, BalanceDecisionConfig


def _row(decision: str, idx: int) -> dict:
    side = "NONE" if decision == "ABSTAIN" else "LONG"
    return {
        "date": f"2025-01-01 00:{idx:02d}:00",
        "prompt": "past",
        "target": json.dumps({"decision": decision, "action_side": side, "confidence": "LOW", "rationale_class": "LOW_CONFIDENCE_SKIP"}),
    }


class TestBalanceDecisionSFT(unittest.TestCase):
    def test_balances_by_oversampling_minority_and_downsampling_majority(self):
        rows = [_row("ABSTAIN", i) for i in range(5)] + [_row("TRADE_TREND", 10)] + [_row("FADE_TREND", 20)]
        out = balance_decision_rows(rows, BalanceDecisionConfig(target_per_decision=3, seed=1))
        counts = {}
        for row in out:
            label = json.loads(row["target"])["decision"]
            counts[label] = counts.get(label, 0) + 1
            self.assertTrue(row["sampling"]["balanced_decision_train"])
        self.assertEqual(counts, {"ABSTAIN": 3, "FADE_TREND": 3, "TRADE_TREND": 3})

    def test_cli_helper_writes_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "in.jsonl"
            out = root / "out.jsonl"
            summary = root / "summary.json"
            rows = [_row("ABSTAIN", i) for i in range(4)] + [_row("TRADE_TREND", 10)] + [_row("FADE_TREND", 20)]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = build_balanced_decision_sft(input_jsonl=str(src), output=str(out), summary_output=str(summary), target_per_decision=2)
            self.assertEqual(report["balanced_rows"], 6)
            self.assertTrue(summary.exists())
            self.assertEqual(len(out.read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
