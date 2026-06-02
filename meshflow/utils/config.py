# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, List, Optional

from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    name: str = "default"
    tag: str = ""
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    resume: Optional[str] = None
    data: dict = field(default_factory=dict)
    system_type: str = ""
    system: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                if self.n_gpus > 1:
                    print(
                        "Timestamp is disabled when using multiple GPUs; set a unique tag."
                    )
                else:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)


def _to_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def structured_config(cls, cfg: Any):
    """Build a structured config object from a dataclass type and plain dict."""
    return OmegaConf.structured(cls(**_to_dict(cfg)))


def load_config(
    path: str,
    cli_args: Optional[List[str]] = None,
    **overrides,
) -> ExperimentConfig:
    cfg = OmegaConf.load(path)
    if cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(cli_args))
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)
    raw = _to_dict(cfg)
    valid = {f.name for f in fields(ExperimentConfig)}
    return ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})
