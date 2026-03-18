"""Option A model components (CNN baseline + frozen HF vision backbones)."""

from __future__ import annotations

import contextlib
from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

DEFAULT_HF_VISUAL_BACKBONE = "facebook/dinov2-with-registers-small"
VISUAL_BACKBONE_PRESETS = {
    "dinov2_reg_s": "facebook/dinov2-with-registers-small",
    "dinov2_reg_b": "facebook/dinov2-with-registers-base",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "siglip2_base_224": "google/siglip2-base-patch16-224",
}


def _validate_scalar_space(
    observation_space: spaces.Dict,
    scalar_input_dim: int | None,
) -> int:
    scalar_space = observation_space["scalars"]
    if not isinstance(scalar_space, spaces.Box):
        raise ValueError("observation_space['scalars'] must be gymnasium.spaces.Box")
    inferred_scalar_dim = int(scalar_space.shape[0])
    if scalar_input_dim is None:
        return inferred_scalar_dim
    if scalar_input_dim != inferred_scalar_dim:
        raise ValueError(
            f"scalar_input_dim mismatch: got {scalar_input_dim}, expected {inferred_scalar_dim}"
        )
    return scalar_input_dim


def _extract_processor_image_size(processor) -> int:
    for attr_name in ("crop_size", "size"):
        value = getattr(processor, attr_name, None)
        if isinstance(value, dict):
            for key in ("height", "width", "shortest_edge"):
                if key in value:
                    return int(value[key])
        if isinstance(value, (int, float)):
            return int(value)
    return 224


def _resolve_visual_backbone_model(
    visual_backbone: str,
    visual_backbone_model: str | None,
) -> str | None:
    if visual_backbone == "cnn":
        return None
    if visual_backbone == "hf_custom":
        if not visual_backbone_model:
            raise ValueError("visual_backbone_model is required when visual_backbone='hf_custom'")
        return str(visual_backbone_model)
    if visual_backbone_model:
        return str(visual_backbone_model)
    try:
        return VISUAL_BACKBONE_PRESETS[visual_backbone]
    except KeyError as exc:
        raise ValueError(
            f"Unknown visual_backbone='{visual_backbone}'. "
            f"Expected one of {['cnn', *VISUAL_BACKBONE_PRESETS.keys(), 'hf_custom']}"
        ) from exc


class ChartScalarExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations:
      - image: CNN encoder
      - scalars: MLP encoder
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 320,
        scalar_input_dim: int | None = None,
    ):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        if not isinstance(image_space, spaces.Box):
            raise ValueError("observation_space['image'] must be gymnasium.spaces.Box")

        scalar_input_dim = _validate_scalar_space(observation_space, scalar_input_dim)

        image_channels = int(image_space.shape[0])
        self.image_net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self._features_dim = 256 + 64

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        image = obs["image"].float()
        if image.max() > 1.0:
            image = image / 255.0

        scalars = obs["scalars"].float()
        image_feat = self.image_net(image)
        scalar_feat = self.scalar_net(scalars)
        return th.cat([image_feat, scalar_feat], dim=1)


class FrozenHFChartScalarExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that swaps the CNN image tower for a frozen HF vision backbone.

    Latest practical default is DINOv2-with-registers-small. Pooling defaults to
    CLS + mean pooled patch tokens to preserve both global and local chart structure.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 320,
        scalar_input_dim: int | None = None,
        visual_backbone_model: str = DEFAULT_HF_VISUAL_BACKBONE,
        freeze_visual_backbone: bool = True,
        visual_pooling: str = "cls_patch_mean",
    ):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        if not isinstance(image_space, spaces.Box):
            raise ValueError("observation_space['image'] must be gymnasium.spaces.Box")
        if int(image_space.shape[0]) != 3:
            raise ValueError(
                "FrozenHFChartScalarExtractor expects RGB images with shape (3, H, W)"
            )

        scalar_input_dim = _validate_scalar_space(observation_space, scalar_input_dim)
        if visual_pooling not in {"cls", "mean", "cls_patch_mean"}:
            raise ValueError(
                "visual_pooling must be one of {'cls', 'mean', 'cls_patch_mean'}"
            )

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(visual_backbone_model, use_fast=True)
        backbone = AutoModel.from_pretrained(visual_backbone_model)
        vision_model = getattr(backbone, "vision_model", backbone)
        if vision_model is not backbone:
            del backbone

        self.visual_backbone = vision_model
        self.visual_backbone_model = str(visual_backbone_model)
        self.freeze_visual_backbone = bool(freeze_visual_backbone)
        self.visual_pooling = str(visual_pooling)
        self.backbone_image_size = _extract_processor_image_size(processor)
        self.image_mean = th.tensor(
            list(getattr(processor, "image_mean", [0.485, 0.456, 0.406])),
            dtype=th.float32,
        ).view(1, 3, 1, 1)
        self.image_std = th.tensor(
            list(getattr(processor, "image_std", [0.229, 0.224, 0.225])),
            dtype=th.float32,
        ).view(1, 3, 1, 1)
        self.num_register_tokens = int(
            getattr(getattr(self.visual_backbone, "config", None), "num_register_tokens", 0) or 0
        )
        hidden_size = int(getattr(self.visual_backbone.config, "hidden_size"))
        pooled_dim = hidden_size * (2 if self.visual_pooling == "cls_patch_mean" else 1)

        if self.freeze_visual_backbone:
            self.visual_backbone.requires_grad_(False)
            self.visual_backbone.eval()

        self.image_proj = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self._features_dim = 256 + 64

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            self.visual_backbone.eval()
        return self

    def _prepare_images(self, image: th.Tensor) -> th.Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        image = F.interpolate(
            image,
            size=(self.backbone_image_size, self.backbone_image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = self.image_mean.to(device=image.device, dtype=image.dtype)
        std = self.image_std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def _pool_backbone_output(self, outputs) -> th.Tensor:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            tokens = outputs.last_hidden_state
            cls_token = tokens[:, 0]
            patch_start = 1 + self.num_register_tokens
            if tokens.shape[1] > patch_start:
                patch_mean = tokens[:, patch_start:].mean(dim=1)
            else:
                patch_mean = cls_token
            if self.visual_pooling == "cls":
                return cls_token
            if self.visual_pooling == "mean":
                return patch_mean
            return th.cat([cls_token, patch_mean], dim=1)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
            return pooled if self.visual_pooling != "cls_patch_mean" else th.cat([pooled, pooled], dim=1)

        raise ValueError("HF vision backbone output did not provide last_hidden_state or pooler_output")

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        image = self._prepare_images(obs["image"])
        scalars = obs["scalars"].float()

        grad_ctx = th.no_grad() if self.freeze_visual_backbone else contextlib.nullcontext()
        with grad_ctx:
            outputs = self.visual_backbone(pixel_values=image)
            pooled = self._pool_backbone_output(outputs).float()

        image_feat = self.image_proj(pooled)
        scalar_feat = self.scalar_net(scalars)
        return th.cat([image_feat, scalar_feat], dim=1)


def build_policy_kwargs(
    visual_backbone: str = "cnn",
    visual_backbone_model: str | None = None,
    features_dim: int = 320,
    scalar_input_dim: int | None = None,
    freeze_visual_backbone: bool = True,
    visual_pooling: str = "cls_patch_mean",
) -> dict:
    """Return stable-baselines3 PPO policy kwargs for Option A."""
    resolved_model = _resolve_visual_backbone_model(visual_backbone, visual_backbone_model)
    if visual_backbone == "cnn":
        extractor_class = ChartScalarExtractor
        extractor_kwargs = {
            "features_dim": features_dim,
            "scalar_input_dim": scalar_input_dim,
        }
    else:
        extractor_class = FrozenHFChartScalarExtractor
        extractor_kwargs = {
            "features_dim": features_dim,
            "scalar_input_dim": scalar_input_dim,
            "visual_backbone_model": resolved_model,
            "freeze_visual_backbone": freeze_visual_backbone,
            "visual_pooling": visual_pooling,
        }
    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        "activation_fn": nn.ReLU,
    }
