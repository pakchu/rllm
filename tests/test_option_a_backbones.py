import unittest

from models.option_a import (
    ChartScalarExtractor,
    FrozenHFChartScalarExtractor,
    build_policy_kwargs,
)


class TestOptionABackbones(unittest.TestCase):
    def test_build_policy_kwargs_defaults_to_cnn(self):
        kwargs = build_policy_kwargs()
        self.assertIs(kwargs["features_extractor_class"], ChartScalarExtractor)
        self.assertEqual(kwargs["features_extractor_kwargs"]["features_dim"], 320)

    def test_build_policy_kwargs_dinov2_preset(self):
        kwargs = build_policy_kwargs(visual_backbone="dinov2_reg_s", visual_pooling="mean")
        self.assertIs(kwargs["features_extractor_class"], FrozenHFChartScalarExtractor)
        self.assertEqual(
            kwargs["features_extractor_kwargs"]["visual_backbone_model"],
            "facebook/dinov2-with-registers-small",
        )
        self.assertEqual(kwargs["features_extractor_kwargs"]["visual_pooling"], "mean")

    def test_build_policy_kwargs_hf_custom_requires_model(self):
        with self.assertRaises(ValueError):
            build_policy_kwargs(visual_backbone="hf_custom")

    def test_build_policy_kwargs_rejects_unknown_backbone(self):
        with self.assertRaises(ValueError):
            build_policy_kwargs(visual_backbone="not_a_backbone")


if __name__ == "__main__":
    unittest.main()
