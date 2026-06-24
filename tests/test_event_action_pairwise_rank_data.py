import json
import tempfile
import unittest
from pathlib import Path

from training.event_action_pairwise_rank_data import EventActionPairwiseRankConfig, build_pairwise_jsonl, build_pairwise_rows


def row(date, pos, family, side, hold, utility, net=0.0):
    return {
        "date": date,
        "signal_pos": pos,
        "prompt": "\n".join(
            [
                "You are an action value judge for BTCUSDT futures.",
                "Output exactly one label: TAKE or SKIP.",
                "Date: 2025-01-01",
                "Past-only state: {\"trend_24\":1.0}",
                "Action book: [{\"family\":\"breakout\",\"side\":\"LONG\",\"strength\":2.0}]",
                "Candidate action: {\"family\":\"leaky\",\"side\":\"SHORT\",\"hold_bars\":72}",
            ]
        ),
        "action": {"family": family, "side": side, "hold_bars": hold, "strength": 2.0},
        "action_audit": {"rank_utility": utility, "net_return": net, "mae": 0.01, "mfe": 0.02},
    }


class TestEventActionPairwiseRankData(unittest.TestCase):
    def test_builds_balanced_ab_pair_without_future_audit_in_prompt(self):
        rows = [
            row("2025-01-01", 10, "breakout", "LONG", 72, 0.02),
            row("2025-01-01", 10, "fade", "SHORT", 144, -0.01),
            row("2025-01-01", 10, "trend", "LONG", 288, 0.01),
        ]
        pairs = build_pairwise_rows(
            rows,
            cfg=EventActionPairwiseRankConfig(
                input_jsonl="in.jsonl", output_jsonl="out.jsonl", min_utility_gap=0.005, max_pairs_per_signal=2
            ),
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p["target"] for p in pairs}, {"A", "B"})
        self.assertTrue(all("rank_utility" not in p["prompt"] for p in pairs))
        self.assertTrue(all("Candidate action:" not in p["prompt"] for p in pairs))
        self.assertTrue(all("Candidate A:" in p["prompt"] and "Candidate B:" in p["prompt"] for p in pairs))
        self.assertGreater(pairs[0]["chosen_utility"], pairs[0]["rejected_utility"])
        self.assertTrue(pairs[0]["leakage_guard"]["chosen_rejected_use_future_utility_for_training_only"])

    def test_swapped_duplicates_emit_both_orientations_for_same_pair(self):
        rows = [
            row("2025-01-01", 10, "breakout", "LONG", 72, 0.02),
            row("2025-01-01", 10, "fade", "SHORT", 144, -0.01),
        ]
        pairs = build_pairwise_rows(
            rows,
            cfg=EventActionPairwiseRankConfig(
                input_jsonl="in.jsonl",
                output_jsonl="out.jsonl",
                min_utility_gap=0.005,
                max_pairs_per_signal=1,
                emit_swapped_duplicates=True,
            ),
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual([p["target"] for p in pairs], ["A", "B"])
        self.assertEqual({p["orientation_mode"] for p in pairs}, {"swapped_duplicate"})
        self.assertTrue(all(p["leakage_guard"]["swapped_duplicate_pairing"] for p in pairs))
        self.assertNotEqual(
            pairs[0]["prompt"].split("Candidate A: ")[1].split("\n")[0],
            pairs[1]["prompt"].split("Candidate A: ")[1].split("\n")[0],
        )

    def test_cli_writes_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "value.jsonl"
            out = Path(tmp) / "pairs.jsonl"
            summary = Path(tmp) / "summary.json"
            rows = [
                row("2025-01-01", 10, "breakout", "LONG", 72, 0.02),
                row("2025-01-01", 10, "fade", "SHORT", 144, -0.01),
            ]
            inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_pairwise_jsonl(
                input_jsonl=str(inp),
                output_jsonl=str(out),
                summary_output=str(summary),
                min_utility_gap=0.005,
                max_pairs_per_signal=4,
            )
            self.assertEqual(report["pairs"], 1)
            self.assertEqual(report["signals"], 1)
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())
            written = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(written[0]["task"], "event_action_pairwise_rank")


if __name__ == "__main__":
    unittest.main()
