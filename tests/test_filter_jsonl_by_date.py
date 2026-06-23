import json
import tempfile
import unittest
from pathlib import Path

from training.filter_jsonl_by_date import FilterJsonlByDateCfg, run


class TestFilterJsonlByDate(unittest.TestCase):
    def test_filters_exclusive_max_date(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jsonl"
            out = Path(td) / "out.jsonl"
            inp.write_text("\n".join([
                json.dumps({"date": "2025-12-31 21:00:00", "x": 1}),
                json.dumps({"date": "2026-01-01 00:00:00", "x": 2}),
                json.dumps({"date": "2026-01-01 03:00:00", "x": 3}),
            ]) + "\n")
            report = run(FilterJsonlByDateCfg(str(inp), str(out), max_date="2026-01-01"))
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([r["x"] for r in rows], [1])
            self.assertEqual(report["kept_rows"], 1)
            self.assertEqual(report["dropped_rows"], 2)


if __name__ == "__main__":
    unittest.main()
