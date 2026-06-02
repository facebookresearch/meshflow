# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ImageType = Union[Image.Image, torch.Tensor]

_HUB_EMBED_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}


def empty_visual_embeds_shape(image_size: int, embed_dim: int) -> tuple[int, int]:
    seq_len = 1 + 4 + (image_size // 16) ** 2
    return seq_len, embed_dim


def make_empty_visual_embeds(image_size: int, embed_dim: int) -> torch.Tensor:
    seq_len, dim = empty_visual_embeds_shape(image_size, embed_dim)
    return torch.zeros(1, seq_len, dim)


class DINOv3Encoder(nn.Module):
    """ViT-based visual encoder."""

    def __init__(
        self,
        image_size: int,
        hub_model: str,
        hub_dir: str = "/root/.cache/torch/hub/facebookresearch_dinov3_main",
        hub_weights: str | None = None,
        pretrained: bool = True,
        freeze: bool = True,
        empty_embeds_ratio: float = 0.1,
    ):
        super().__init__()
        self.empty_embeds_ratio = empty_embeds_ratio

        hub_kwargs: dict = {"pretrained": pretrained}
        if hub_weights is not None:
            hub_kwargs["weights"] = hub_weights
        self.dino_model = torch.hub.load(
            hub_dir, hub_model, source="local", **hub_kwargs
        )
        embed_dim = self.dino_model.embed_dim

        self.transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        seq_len, _ = empty_visual_embeds_shape(image_size, embed_dim)
        self.register_buffer(
            "empty_image_embeds",
            torch.zeros(1, seq_len, embed_dim),
            persistent=False,
        )

        self.dino_model.eval()
        if freeze:
            self.dino_model.requires_grad_(False)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode_image(self, image: ImageType) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            x = self.transform(image.permute(0, 3, 1, 2))
        else:
            x = self.transform(transforms.ToTensor()(image).unsqueeze(0))

        ref = next(self.dino_model.parameters())
        x = x.to(device=ref.device, dtype=ref.dtype)
        with torch.autocast(device_type=ref.device.type, enabled=False):
            out = self.dino_model.forward_features(x)["x_prenorm"]
        return F.layer_norm(out, out.shape[-1:])

    def forward(self, batch):
        bs = batch["image"].shape[0]
        if self.training and random.random() < self.empty_embeds_ratio:
            return self.empty_image_embeds.repeat(bs, 1, 1).to(batch["image"].device)
        return self.encode_image(batch["image"])


def build_dinov3_encoder(
    encoder_type: str,
    cfg: dict,
) -> DINOv3Encoder:
    if encoder_type != "dinov3-encoder":
        raise ValueError(f"Unknown visual_condition_type: {encoder_type!r}")

    return DINOv3Encoder(
        image_size=cfg["image_size"],
        hub_model=cfg["hub_model"],
        hub_dir=cfg.get("hub_dir", "/root/.cache/torch/hub/facebookresearch_dinov3_main"),
        hub_weights=cfg.get("hub_weights"),
        pretrained=cfg.get("pretrained", True),
        freeze=cfg.get("freeze", True),
        empty_embeds_ratio=cfg.get("empty_embeds_ratio", 0.1),
    )


def resolve_visual_embed_dim(cfg: dict, fallback: int | None = None) -> int:
    if "embed_dim" in cfg:
        return int(cfg["embed_dim"])
    hub_model = cfg.get("hub_model")
    if hub_model in _HUB_EMBED_DIMS:
        return _HUB_EMBED_DIMS[hub_model]
    if fallback is not None:
        return int(fallback)
    raise ValueError(
        f"Cannot infer DINOv3 embed_dim for hub_model={hub_model!r}; "
        "set visual_condition.embed_dim in config.yaml"
    )
