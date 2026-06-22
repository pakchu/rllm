import json
import tempfile
import unittest
from pathlib import Path

from training.alpha_candidate_gate import AlphaGateConfig, gate_report, score_candidate


def fold(name, cagr, mdd, ratio, trades=100):
    return {"fold": name, "result": {"sim": {"cagr_pct": cagr, "strict_mdd_pct": mdd, "cagr_to_strict_mdd": ratio, "trade_entries": trades}}}


class TestAlphaCandidateGate(unittest.TestCase):
    def test_candidate_passes_when_all_gates_satisfied(self):
        cfg = AlphaGateConfig(input_report="in", output="out", min_positive_folds=2, min_total_trades=100)
        cand = {"feature": "x", "horizon": 72, "quantile": 0.2, "strict_folds": [fold("a", 50, 10, 5), fold("b", 40, 12, 3.5)]}
        scored = score_candidate(cand, cfg)
        self.assertTrue(scored["passed"])
        self.assertEqual(scored["summary"]["positive_folds"], 2)

    def test_candidate_fails_on_negative_worst_fold_and_mdd(self):
        cfg = AlphaGateConfig(input_report="in", output="out", min_positive_folds=2, min_total_trades=100)
        cand = {"feature": "x", "horizon": 72, "quantile": 0.2, "strict_folds": [fold("a", 50, 10, 5), fold("b", -20, 30, -1)]}
        scored = score_candidate(cand, cfg)
        self.assertFalse(scored["passed"])
        self.assertIn("negative_or_zero_worst_fold_cagr", scored["failures"])
        self.assertIn("mdd_exceeds_limit_in_some_folds", scored["failures"])

    def test_candidate_fails_without_enough_trades(self):
        cfg = AlphaGateConfig(input_report="in", output="out", min_positive_folds=2, min_fold_trades=30, min_total_trades=100)
        cand = {"feature": "x", "strict_folds": [fold("a", 50, 10, 5, trades=10), fold("b", 40, 10, 4, trades=10)]}
        scored = score_candidate(cand, cfg)
        self.assertFalse(scored["passed"])
        self.assertIn("insufficient_trade_count_folds", scored["failures"])
        self.assertIn("insufficient_total_trades", scored["failures"])

    def test_score_candidate_preserves_regime_metadata(self):
        cfg = AlphaGateConfig(input_report="in", output="out", min_positive_folds=2, min_total_trades=100)
        cand = {
            "regime_col": "dxy_zscore",
            "regime_side": "low",
            "signal_col": "kimchi_premium_zscore",
            "horizon": 144,
            "test": {"sim": {"cagr_pct": 40, "strict_mdd_pct": 10, "cagr_to_strict_mdd": 4, "trade_entries": 80}},
            "eval": {"sim": {"cagr_pct": 35, "strict_mdd_pct": 11, "cagr_to_strict_mdd": 3.2, "trade_entries": 90}},
        }
        scored = score_candidate(cand, cfg)
        self.assertEqual(scored["candidate"]["feature"], "kimchi_premium_zscore")
        self.assertEqual(scored["candidate"]["regime_col"], "dxy_zscore")
        self.assertEqual(scored["candidate"]["regime_side"], "low")

    def test_gate_report_reads_top_by_test_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = {
                "top_by_test": [
                    {
                        "test": {"sim": {"cagr_pct": 40, "strict_mdd_pct": 10, "cagr_to_strict_mdd": 4, "trade_entries": 80}},
                        "eval": {"sim": {"cagr_pct": 35, "strict_mdd_pct": 11, "cagr_to_strict_mdd": 3.2, "trade_entries": 90}},
                    }
                ]
            }
            inp = root / "report.json"
            out = root / "gate.json"
            inp.write_text(json.dumps(report))
            gated = gate_report(AlphaGateConfig(input_report=str(inp), output=str(out), min_total_trades=100, min_positive_folds=2))
        self.assertEqual(gated["source_key"], "top_by_test")
        self.assertEqual(gated["decision"], "GO")

    def test_candidate_supports_test_eval_report_shape(self):
        cfg = AlphaGateConfig(input_report="in", output="out", min_positive_folds=2, min_total_trades=100)
        cand = {
            "group": "wave_core",
            "test": {"sim": {"cagr_pct": 40, "strict_mdd_pct": 10, "cagr_to_strict_mdd": 4, "trade_entries": 80}},
            "eval": {"sim": {"cagr_pct": 35, "strict_mdd_pct": 11, "cagr_to_strict_mdd": 3.2, "trade_entries": 90}},
        }
        scored = score_candidate(cand, cfg)
        self.assertTrue(scored["passed"])
        self.assertEqual(scored["summary"]["valid_folds"], 2)
        self.assertEqual(scored["candidate"]["group"], "wave_core")


if __name__ == "__main__":
    unittest.main()
