import json
import tempfile
import unittest
from pathlib import Path

from training.build_event_side_pair_preference import BuildEventSidePairPreferenceCfg, build


class TestBuildEventSidePairPreference(unittest.TestCase):
    def test_builds_opposite_rejected_pair(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            inp.write_text("\n".join([
                json.dumps({"prompt": "P1", "target": json.dumps({"side_pair": "normal"}), "date": "d1"}),
                json.dumps({"prompt": "P2", "target": json.dumps({"side_pair": "inverse"}), "date": "d2"}),
                json.dumps({"prompt": "P3", "target": json.dumps({"side_map": "unreliable"}), "date": "d3"}),
            ]) + "\n")
            report = build(BuildEventSidePairPreferenceCfg(str(inp), str(out)))
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(report["pairs_out"], 2)
            self.assertEqual(json.loads(rows[0]["chosen"]), {"side_pair": "normal"})
            self.assertEqual(json.loads(rows[0]["rejected"]), {"side_pair": "inverse"})
            self.assertEqual(json.loads(rows[1]["chosen"]), {"side_pair": "inverse"})
            self.assertEqual(json.loads(rows[1]["rejected"]), {"side_pair": "normal"})
            self.assertTrue(rows[0]["leakage_guard"]["prompt_reused_from_causal_event_side_pair_record"])


if __name__ == "__main__":
    unittest.main()
