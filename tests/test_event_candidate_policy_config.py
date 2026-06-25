import json
import tempfile
import unittest
from pathlib import Path

from training.event_candidate_policy_config import DEFAULT_POLICY_PATH, build_walk_forward_cfg, load_policy_preset


class TestEventCandidatePolicyConfig(unittest.TestCase):
    def test_current_policy_loads_as_non_live_candidate(self):
        preset = load_policy_preset(DEFAULT_POLICY_PATH)
        self.assertFalse(preset["live_ready"])
        self.assertTrue(preset["paper_test_required"])
        self.assertEqual(preset["walk_forward"]["max_feature_names"], "rex_36_range_width_pct")
        self.assertEqual(preset["walk_forward"]["min_feature_names"], "rex_36_cur_to_min_pct")

    def test_build_walk_forward_cfg_applies_policy(self):
        preset = load_policy_preset(DEFAULT_POLICY_PATH)
        cfg = build_walk_forward_cfg(preset, output="out.json", work_dir="work")
        self.assertEqual(cfg.output, "out.json")
        self.assertEqual(cfg.work_dir, "work")
        self.assertEqual(cfg.fit_months, 6)
        self.assertEqual(cfg.ranker_drop_prefixes, "rex_")
        self.assertEqual(cfg.max_feature_quantiles, "0.60,0.70,0.80,0.90")
        self.assertEqual(cfg.min_feature_quantiles, "0.50,0.60,0.70,0.80")

    def test_loader_rejects_live_ready_without_audit(self):
        preset = load_policy_preset(DEFAULT_POLICY_PATH)
        preset["live_ready"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(preset))
            with self.assertRaises(ValueError):
                load_policy_preset(path)


if __name__ == "__main__":
    unittest.main()
