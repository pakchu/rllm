import json
import tempfile
import unittest
from pathlib import Path

from training.build_event_side_pair_sft import BuildEventSidePairCfg, build


class TestBuildEventSidePairSft(unittest.TestCase):
    def test_filters_unreliable_and_projects_target(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            rows = [
                {"target": json.dumps({"side_map": "normal"}), "leakage_guard": {"x": True}},
                {"target": json.dumps({"side_map": "unreliable"})},
                {"target": json.dumps({"side_map": "inverse"})},
            ]
            inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = build(BuildEventSidePairCfg(str(inp), str(out)))
            got = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(report["rows_out"], 2)
            self.assertEqual(report["skipped_counts"], {"unreliable": 1})
            self.assertEqual([json.loads(r["target"])["side_pair"] for r in got], ["normal", "inverse"])
            self.assertTrue(got[0]["leakage_guard"]["target_projected_to_pairwise_side_map"])


if __name__ == "__main__":
    unittest.main()
