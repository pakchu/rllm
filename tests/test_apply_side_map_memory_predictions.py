import tempfile
import unittest
from pathlib import Path

from training.apply_side_map_memory_predictions import _month, _write_jsonl


class TestApplySideMapMemoryPredictions(unittest.TestCase):
    def test_month(self):
        self.assertEqual(_month({"date": "2026-02-03 01:00:00"}), "2026-02")

    def test_write_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            _write_jsonl(p, [{"a": 1}])
            self.assertIn('"a": 1', p.read_text())


if __name__ == "__main__":
    unittest.main()
