# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

DEFAULT_RMBG_PATH = "/mnt/jfs-ssd/pretrained_models/RMBG-2.0"

_rmbg_model = None
_rmbg_lock = threading.Lock()


def preprocess_image(
    images: Union[Image.Image, List[Image.Image]],
    background_color: Optional[List[int]] = None,
    foreground_ratio: float = 0.9,
    rmbg_path: str = DEFAULT_RMBG_PATH,
) -> Union[Image.Image, List[Image.Image]]:
    """Remove background (if needed), crop foreground, center and square-pad."""
    global _rmbg_model

    bg = tuple(background_color or [255, 255, 255])
    single = isinstance(images, Image.Image)
    batch = [images] if single else list(images)
    out: List[Image.Image] = []

    for im in batch:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "A" in im.mode else "RGB")

        # Run matting when alpha channel is not fully opaque.
        if not (im.mode == "RGBA" and im.getextrema()[3][0] >= 255):
            with _rmbg_lock:
                if _rmbg_model is None:
                    if not os.path.isdir(rmbg_path):
                        raise FileNotFoundError(f"RMBG model not found: {rmbg_path}")
                    from transformers import AutoModelForImageSegmentation

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    _rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                        rmbg_path, trust_remote_code=True
                    ).eval().to(device)

            model = _rmbg_model
            device = next(model.parameters()).device
            rgb = im.convert("RGB")
            norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            inp = norm(transforms.ToTensor()(transforms.Resize((1024, 1024))(rgb))).unsqueeze(0).to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"), torch.no_grad():
                mask = model(inp)[-1].sigmoid()[0, 0].cpu()
            im = rgb.copy()
            im.putalpha(transforms.ToPILImage()(mask).resize(rgb.size))

        alpha = im.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            raise ValueError("No foreground found after background removal.")

        w0, h0 = im.size
        x1, y1, x2, y2 = bbox
        dw, dh = x2 - x1, y2 - y1
        scale = min(h0 * foreground_ratio / dh, w0 * foreground_ratio / dw)
        tw, th = max(1, int(dw * scale)), max(1, int(dh * scale))

        fg = Image.alpha_composite(Image.new("RGBA", im.size, (*bg, 255)), im).crop(bbox)
        fg = fg.resize((tw, th))
        alpha = alpha.crop(bbox).resize((tw, th))

        canvas = Image.new("RGBA", (w0, h0), (*bg, 255))
        canvas.paste(fg, ((w0 - tw) // 2, (h0 - th) // 2), alpha)
        side = max(w0, h0)
        if w0 != h0:
            square = Image.new("RGBA", (side, side), (*bg, 255))
            square.paste(canvas, ((side - w0) // 2, (side - h0) // 2))
            canvas = square
        out.append(canvas)

    return out[0] if single else out


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"{scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"{scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def flow_sample(
    scheduler: FlowMatchEulerDiscreteScheduler,
    diffusion_model: torch.nn.Module,
    shape: Union[List[int], Tuple[int]],
    steps: int,
    visual_cond: Optional[torch.Tensor] = None,
    shape_cond: Optional[torch.Tensor] = None,
    guidance_scale: float = 3.0,
    do_classifier_free_guidance: bool = True,
    generator: Optional[torch.Generator] = None,
    device: torch.device = "cuda:0",
    disable_prog: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    denoiser_model_kwargs: Optional[Dict[str, Any]] = None,
):
    assert steps > 0, f"{steps} must > 0."

    bsz_from_doubled_cond = False
    if visual_cond is not None:
        bsz = visual_cond.shape[0]
        device = visual_cond.device
        dtype = visual_cond.dtype
        bsz_from_doubled_cond = do_classifier_free_guidance
    elif shape_cond is not None:
        bsz = shape_cond.shape[0]
        device = shape_cond.device
        dtype = shape_cond.dtype
        bsz_from_doubled_cond = do_classifier_free_guidance
    elif joint_attention_kwargs is not None:
        bsz, device, dtype = None, None, None
        if "rope_input" in joint_attention_kwargs:
            v = joint_attention_kwargs["rope_input"]
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bsz, device, dtype = v.shape[0], v.device, v.dtype
        if bsz is None:
            for v in joint_attention_kwargs.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    bsz, device, dtype = v.shape[0], v.device, v.dtype
                    break
        if bsz is None:
            raise ValueError(
                "flow_sample: joint_attention_kwargs must contain a batch-sized tensor."
            )
        bsz_from_doubled_cond = False
    else:
        raise ValueError("flow_sample: at least one condition tensor is required.")

    if do_classifier_free_guidance and bsz_from_doubled_cond:
        bsz = bsz // 2

    latents = torch.randn((bsz, *shape), generator=generator, device=device, dtype=dtype)
    try:
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    joint_attention_kwargs_for_forward = joint_attention_kwargs
    if joint_attention_kwargs_for_forward is not None and do_classifier_free_guidance:
        joint_attention_kwargs_for_forward = {
            k: (
                torch.cat([v, v], dim=0)
                if isinstance(v, torch.Tensor) and v.shape[0] == bsz
                else v
            )
            for k, v in joint_attention_kwargs.items()
        }

    timesteps, _ = retrieve_timesteps(scheduler, steps + 1, device)
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps

    for i, t in enumerate(
        tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)
    ):
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        timestep_tensor = timestep_tensor / scheduler.config.num_train_timesteps

        noise_pred = diffusion_model(
            latent_model_input,
            timestep_tensor,
            visual_condition=visual_cond,
            joint_attention_kwargs=joint_attention_kwargs_for_forward,
            **(denoiser_model_kwargs or {}),
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        latents = latents - distance[i] * noise_pred
        yield latents, t
