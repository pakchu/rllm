import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from training.hierarchical_direct_split_search import run_search


class TestHierarchicalDirectSplitSearch(unittest.TestCase):
    def test_run_search_finds_candidate_on_synthetic_reports(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base = datetime(2025, 1, 1, 0, 0, 0)

            def build_gate(path: Path, sign: float, *, start: datetime):
                rows = []
                for i in range(180):
                    dt = start + timedelta(minutes=5 * i)
                    rows.append(
                        {
                            "date": dt.isoformat(sep=" "),
                            "target": "TRADE",
                            "next_return": sign,
                            "adjusted_scores": {"TRADE": 0.8, "NO_TRADE": -0.2},
                        }
                    )
                path.write_text(json.dumps({"action_schema": "trade_gate", "action_scores": rows}))

            def build_side(path: Path, long: bool, *, start: datetime):
                rows = []
                for i in range(180):
                    dt = start + timedelta(minutes=5 * i)
                    rows.append(
                        {
                            "date": dt.isoformat(sep=" "),
                            "target": "LONG" if long else "SHORT",
                            "next_return": 0.002 if long else -0.002,
                            "adjusted_scores": {"LONG": 0.9 if long else -0.9, "SHORT": -0.9 if long else 0.9},
                        }
                    )
                path.write_text(json.dumps({"action_schema": "trade_side", "action_scores": rows}))

            gate_test = tmp / "gate_test.json"
            side_test = tmp / "side_test.json"
            gate_eval = tmp / "gate_eval.json"
            side_eval = tmp / "side_eval.json"
            build_gate(gate_test, 0.002, start=base)
            build_side(side_test, True, start=base)
            eval_base = base + timedelta(days=10)
            build_gate(gate_eval, 0.002, start=eval_base)
            build_side(side_eval, True, start=eval_base)

            out = run_search(
                gate_test_file=str(gate_test),
                side_test_file=str(side_test),
                gate_eval_file=str(gate_eval),
                side_eval_file=str(side_eval),
                output=str(tmp / "out.json"),
                alpha=0.05,
                min_trades=10,
                leverage=2.0,
                fee_rate=0.0004,
                slippage_rate=0.0001,
            )

            self.assertIn(out["selected_from"], {"strict", "relaxed"})
            self.assertIsNotNone(out["selected_params"])
            self.assertTrue(out["leakage_guard"]["eval_strictly_after_test"])


if __name__ == "__main__":
    unittest.main()
