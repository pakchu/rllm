import json
import tempfile
import unittest
from pathlib import Path

from training.eval_code_score_selector import CodeScoreSelectorConfig, evaluate, run


def scored(date, pos, q4, utility):
    return {"date": date, "signal_pos": pos, "score": {"Q1": {"mean": 0.0}, "Q2": {"mean": 0.0}, "Q3": {"mean": 0.0}, "Q4": {"mean": q4}}, "action_audit": {"rank_utility": utility}}


class TestEvalCodeScoreSelector(unittest.TestCase):
    def test_evaluate_selects_highest_expected_rank_candidate(self):
        rows = [scored("d", 1, 0.0, -0.1), scored("d", 1, 2.0, 0.2)]
        out = evaluate(rows, CodeScoreSelectorConfig("in", "out"), {})
        self.assertEqual(out["signals"], 1)
        self.assertAlmostEqual(out["selected_utility"]["mean"], 0.2)
        self.assertGreater(out["selected_minus_first_mean"], 0.0)

    def test_run_reads_report_score_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            scored_path = Path(tmp) / "scores.json"
            out = Path(tmp) / "out.json"
            scored_path.write_text(json.dumps({"score_rows": [scored("d", 1, 1.0, 0.1)]}))
            report = run(CodeScoreSelectorConfig(str(scored_path), str(out)))
            self.assertEqual(report["metrics"]["signals"], 1)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
