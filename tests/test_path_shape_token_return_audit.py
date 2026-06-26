import unittest

from training.path_shape_token_return_audit import _group, _summ


class TestPathShapeTokenReturnAudit(unittest.TestCase):
    def test_group_classifies_augmented_tokens(self):
        self.assertEqual(_group("aug.micro.w12.return=flat"), "micro")
        self.assertEqual(_group("aug.pa.w36.range_pos=TOP"), "price_action")
        self.assertEqual(_group("aug.macro.dxy.z=HIGH"), "macro")

    def test_summ(self):
        out = _summ([1.0, -0.5])
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["mean_ret_pct"], 0.25)
        self.assertEqual(out["win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
