import unittest

from training.search_strict_bar_policy import _make_filters, _parse_bool_list, _parse_float_list, _parse_int_list


class TestSearchStrictBarPolicy(unittest.TestCase):
    def test_parse_lists(self):
        self.assertEqual(_parse_float_list("1,2.5", []), [1.0, 2.5])
        self.assertEqual(_parse_int_list("1, 3", []), [1, 3])
        self.assertEqual(_parse_bool_list("false,true,yes,0", []), [False, True, True, False])

    def test_make_filters(self):
        filters = _make_filters("tf48_001,mr144_0005,none")
        self.assertEqual([f.name for f in filters], ["tf_trend_48_0p01", "mr_trend_144_0p005", "none"])
        self.assertEqual(filters[0].align_mode, "trend_follow")
        self.assertEqual(filters[1].align_mode, "mean_revert")


if __name__ == "__main__":
    unittest.main()
