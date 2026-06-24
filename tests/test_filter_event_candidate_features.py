import json
import tempfile
import unittest
from pathlib import Path

from training.filter_event_candidate_features import FilterCandidateFeaturesCfg, run


class TestFilterEventCandidateFeatures(unittest.TestCase):
    def test_filters_feature_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.jsonl"
            out = Path(tmp) / "out.jsonl"
            inp.write_text(json.dumps({"feature_snapshot": {"a": 1, "mreg_x": 2, "pa_ext_y": 3}}) + "\n")
            summary = run(FilterCandidateFeaturesCfg(str(inp), str(out), keep_prefixes="pa_ext_", keep_features="a"))
            row = json.loads(out.read_text())
        self.assertEqual(summary["feature_count_min"], 2)
        self.assertEqual(set(row["feature_snapshot"]), {"a", "pa_ext_y"})


if __name__ == "__main__":
    unittest.main()
