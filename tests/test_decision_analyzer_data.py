import json
import tempfile
import unittest
from pathlib import Path

from training.decision_analyzer_data import (
    DecisionAnalyzerConfig,
    build_decision_record,
    build_decision_records,
    derive_decision_target,
    main,
)


def _edge_record(hint="ALLOW_TREND_SPECIALIST", side="LONG", edge="EDGE_PERSIST", transition="TREND_CONTINUATION"):
    return {
        "task": "edge_decay_analyzer",
        "date": "2025-01-01 00:00:00",
        "signal_pos": 10,
        "prompt": "Past-only analyzer summary: trend is up; dxy soft; kimchi neutral",
        "target": json.dumps(
            {
                "trend_side": side,
                "edge_decay_label": edge,
                "transition_label": transition,
                "risk_label": "LOW_ADVERSE_EXCURSION",
                "recommended_router_hint": hint,
            },
            separators=(",", ":"),
        ),
        "path_diagnostics": {
            "long_same": {"net_return": 0.012, "mae": 0.002},
            "long_opposite": {"net_return": -0.004, "mae": 0.005},
        },
        "leakage_guard": {"prompt_uses_future_path": False},
    }


class TestDecisionAnalyzerData(unittest.TestCase):
    def test_derives_trade_fade_and_abstain_targets(self):
        cfg = DecisionAnalyzerConfig()
        trade = derive_decision_target(_edge_record("ALLOW_TREND_SPECIALIST", "LONG"), cfg)
        fade = derive_decision_target(_edge_record("CONSIDER_REVERSAL_SPECIALIST", "LONG", edge="REVERSAL_RISK", transition="TREND_REVERSAL"), cfg)
        abstain = derive_decision_target(_edge_record("REDUCE_OR_SKIP_TREND_SPECIALIST", "SHORT", edge="ADVERSE_STRESS", transition="CHOP_OR_DECAY"), cfg)
        self.assertEqual(trade["decision"], "TRADE_TREND")
        self.assertEqual(trade["action_side"], "LONG")
        self.assertEqual(fade["decision"], "FADE_TREND")
        self.assertEqual(fade["action_side"], "SHORT")
        self.assertEqual(abstain["decision"], "ABSTAIN")
        self.assertEqual(abstain["action_side"], "NONE")
        self.assertEqual(abstain["rationale_class"], "ADVERSE_STRESS_SKIP")

    def test_build_decision_record_keeps_past_prompt_and_future_label_guard(self):
        rec = build_decision_record(_edge_record(), DecisionAnalyzerConfig())
        target = json.loads(rec["target"])
        self.assertEqual(rec["task"], "decision_analyzer")
        self.assertIn(target["decision"], {"TRADE_TREND", "FADE_TREND", "ABSTAIN"})
        self.assertIn("Return exactly one JSON object", rec["prompt"])
        self.assertFalse(rec["leakage_guard"]["prompt_uses_future_path"])
        self.assertTrue(rec["leakage_guard"]["target_uses_future_path"])
        self.assertTrue(rec["leakage_guard"]["decision_target_is_compressed_from_edge_teacher"])

    def test_cli_writes_decision_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "edge.jsonl"
            out = root / "decision.jsonl"
            summary = root / "summary.json"
            rows = [_edge_record(), _edge_record("CONSIDER_REVERSAL_SPECIALIST", "SHORT", edge="REVERSAL_RISK", transition="TREND_REVERSAL")]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            import sys

            old = sys.argv
            try:
                sys.argv = ["prog", "--edge-records", str(src), "--output", str(out), "--summary-output", str(summary)]
                main()
            finally:
                sys.argv = old
            built = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(built), 2)
            payload = json.loads(summary.read_text())
            self.assertEqual(payload["records"]["num_records"], 2)
            self.assertIn("TRADE_TREND", payload["records"]["decision"])


if __name__ == "__main__":
    unittest.main()
