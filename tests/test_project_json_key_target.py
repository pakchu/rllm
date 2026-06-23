import unittest

from training.project_json_key_target import _project_target


class TestProjectJsonKeyTarget(unittest.TestCase):
    def test_project_target(self):
        self.assertEqual(_project_target('{"side_map":"normal","confidence":"HIGH"}', "side_map"), {"side_map": "normal"})

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            _project_target('{"x":1}', "side_map")


if __name__ == "__main__":
    unittest.main()
