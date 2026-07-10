import json
import tempfile
import unittest
from pathlib import Path

from training.eval_text_label import _assert_adapter_matches_model


class TestEvalTextLabelAdapter(unittest.TestCase):
    def test_text_adapter_base_mismatch_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "google/gemma-4-E4B-it"})
            )
            with self.assertRaisesRegex(ValueError, "text-only base"):
                _assert_adapter_matches_model("qwen2.5-1.5b-instruct", str(adapter))


if __name__ == "__main__":
    unittest.main()
