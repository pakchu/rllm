import json
import tempfile
import unittest
from pathlib import Path

from training.threshold_side_rationale_eval import ThresholdSideRationaleEvalCfg, run


class TestThresholdSideRationaleEval(unittest.TestCase):
    def test_low_spread_abstains(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.json"
            out = Path(td) / "out.json"
            inp.write_text(json.dumps({"predictions": [
                {"prediction": "NORMAL", "target": "NORMAL", "scores": {"normal": 0.1, "inverse": 0.09}},
                {"prediction": "INVERSE", "target": "INVERSE", "scores": {"normal": 0.1, "inverse": 0.3}},
            ]}))
            report = run(ThresholdSideRationaleEvalCfg(str(inp), str(out), min_spread=0.05))
            got = json.loads(out.read_text())
            self.assertEqual(got["predictions"][0]["prediction"], "UNRELIABLE")
            self.assertEqual(got["predictions"][1]["prediction"], "INVERSE")
            self.assertEqual(report["kept"], 1)


if __name__ == "__main__":
    unittest.main()
