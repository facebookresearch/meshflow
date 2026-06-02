# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union

import torch

AUTOCAST_DTYPE_CHOICES = ("bf16", "fp16", "fp32")

_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Map ``bf16`` / ``fp16`` / ``fp32`` (or matching ``torch.dtype``) to autocast dtype."""
    if isinstance(dtype, str):
        key = dtype.lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(
                f"Unsupported dtype: {dtype!r}. Choose from: {', '.join(AUTOCAST_DTYPE_CHOICES)}"
            )
        return _DTYPE_ALIASES[key]
    if dtype in _DTYPE_ALIASES.values():
        return dtype
    raise ValueError(
        f"Unsupported dtype: {dtype!r}. Choose from: {', '.join(AUTOCAST_DTYPE_CHOICES)}"
    )


def torch_dtype_to_name(dtype: torch.dtype) -> str:
    for name, torch_dtype in _DTYPE_ALIASES.items():
        if dtype == torch_dtype:
            return name
    raise ValueError(
        f"Unsupported dtype: {dtype!r}. Choose from: {', '.join(AUTOCAST_DTYPE_CHOICES)}"
    )
