import json
import tempfile
import unittest
from pathlib import Path

from training.split_edge_decay_sft import split_edge_decay_sft


def _row(date: str, edge: str = "EDGE_PERSIST") -> dict:
    return {
        "task": "edge_decay_analyzer",
        "date": date,
        "prompt": "past only",
        "target": json.dumps(
            {
                "trend_side": "LONG",
                "edge_decay_label": edge,
                "transition_label": "TREND_CONTINUATION",
                "risk_label": "LOW_ADVERSE_EXCURSION",
                "recommended_router_hint": "ALLOW_TREND_SPECIALIST",
            },
            separators=(",", ":"),
        ),
    }


class TestSplitEdgeDecaySFT(unittest.TestCase):
    def test_chronological_split_counts_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            oos = root / "oos.jsonl"
            summary = root / "summary.json"
            rows = [_row("2025-01-03"), _row("2025-01-01"), _row("2025-02-01", "EDGE_DECAY"), _row("2025-03-01")]
            inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = split_edge_decay_sft(
                input_jsonl=str(inp),
                train_output=str(train),
                val_output=str(val),
                oos_output=str(oos),
                summary_output=str(summary),
                train_start="2025-01-01",
                train_end="2025-01-31",
                val_start="2025-02-01",
                val_end="2025-02-28",
                oos_start="2025-03-01",
                oos_end="2025-03-31",
            )
            self.assertEqual(report["splits"]["train"]["records"], 2)
            self.assertEqual(report["splits"]["val"]["records"], 1)
            self.assertEqual(report["splits"]["oos"]["records"], 1)
            self.assertEqual(len(train.read_text().splitlines()), 2)
            self.assertTrue(json.loads(summary.read_text())["leakage_guard"]["chronological_split"])


if __name__ == "__main__":
    unittest.main()
