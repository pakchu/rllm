import json
import tempfile
import unittest
from pathlib import Path

from training.build_event_side_rationale_preference import (
    BuildEventSideRationalePreferenceCfg,
    build,
    build_prompt,
    candidate_response,
)


ROW = {
    "date": "2026-01-01 00:00:00",
    "signal_pos": 1,
    "generated_side": "LONG",
    "target": json.dumps({"side_pair": "inverse"}),
    "score_tokens": {"score_side_gap": "high", "score_edge_over_wait": "high_positive"},
    "state_tokens": {"trend_alignment": "aligned_down", "window_drawdown": "high", "pa_event_pressure": "upside_rejection"},
    "label_audit": {"chosen_pct": 99},
}


class TestBuildEventSideRationalePreference(unittest.TestCase):
    def test_prompt_and_response_do_not_include_future_audit(self):
        text = build_prompt(ROW) + candidate_response(ROW, "inverse")
        self.assertIn("aligned_down", text)
        self.assertIn("upside_rejection", text)
        self.assertNotIn("chosen_pct", text)
        self.assertNotIn("99", text)

    def test_builds_chosen_rejected_rationales(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            inp.write_text(json.dumps(ROW) + "\n")
            report = build(BuildEventSideRationalePreferenceCfg(str(inp), str(out)))
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(report["pairs_out"], 1)
            self.assertEqual(rows[0]["chosen_side_pair"], "inverse")
            self.assertEqual(json.loads(rows[0]["chosen"])["side_pair"], "inverse")
            self.assertEqual(json.loads(rows[0]["rejected"])["side_pair"], "normal")
            self.assertTrue(rows[0]["leakage_guard"]["candidate_rationales_use_signal_time_tokens_only"])


if __name__ == "__main__":
    unittest.main()
