# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

from ..models.condition_encoder import (
    build_dinov3_encoder,
    make_empty_visual_embeds,
    resolve_visual_embed_dim,
)
from ..models.meshflow_dit import build_meshflow_dit, voxelize_pc
from ..models.meshflow_vae import build_mesh_model
from ..utils.mesh import Mesh
from .utils import flow_sample, preprocess_image as rmbg_preprocess_image


class MeshFlowPipeline:
    """
    MeshFlow DiT image-to-mesh inference pipeline.

    Conditions:
      - RoPE: surface points sampled from an input mesh, voxelized to 3D indices
      - Visual: optional reference image encoded by the visual encoder (or empty tokens)
    """

    model_names_to_load = ["mesh_model", "visual_encoder", "denoiser"]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
        num_verts: int = 4096,
        voxelize_resolution: int = 32,
        num_inference_steps: int = 24,
        guidance_scale: float = 2.5,
        dtype: torch.dtype = torch.float16,
        use_rmbg: bool = False,
        rmbg_preprocess: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if models is None:
            return

        self.models = models
        self.scheduler = scheduler
        self.num_verts = int(num_verts)
        self.voxelize_resolution = int(voxelize_resolution)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.dtype = dtype
        self.use_rmbg = use_rmbg
        self.rmbg_preprocess = rmbg_preprocess or rmbg_preprocess_image
        self._device = torch.device("cpu")
        self._visual_encoder_type: str | None = None
        self._visual_encoder_cfg: dict | None = None
        self._empty_visual_embeds: torch.Tensor | None = None
        self._compile_models = False
        self.num_verts_max: int | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "float16",
        compile_models: bool = False,
        num_verts: Optional[int] = None,
        num_verts_max: Optional[int] = None,
    ) -> "MeshFlowPipeline":
        """
        Load MeshFlow DiT + VAE + visual encoder from a model bundle directory.

        Args:
            model_path: Directory that must contain `config.yaml` and `model.pth`.
            device: Target torch device.
            dtype: `float16` or `float32`.
            compile_models: Whether to `torch.compile` hot paths (CUDA only).
            num_verts: If set and ``denoiser_model.use_proj_cond_on_temb`` is enabled,
                override `data.n_verts`, `mesh_model.n_samples`, `mesh_model.num_latents`,
                and `denoiser_model.input_size`.
            num_verts_max: Normalizer for `proj_cond_on_temb`; only used when
                ``denoiser_model.use_proj_cond_on_temb`` is enabled (default: config
                `system.num_verts_max` or training `denoiser_model.input_size`).
        """
        root = Path(model_path)
        if not root.is_dir():
            raise NotADirectoryError(f"model_path must be a directory: {root}")

        config_path = root / "config.yaml"
        ckpt_path = root / "model.pth"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[MeshFlowPipeline] Loading from {root.resolve()}")
        print(f"  config: {config_path.name}")
        print(f"  weights: {ckpt_path.name} ({ckpt_path.stat().st_size / (1024 ** 3):.2f} GB)")

        cfg = OmegaConf.load(config_path)
        data_cfg, scfg = cfg.data, cfg.system

        config_num_verts = int(data_cfg.n_verts)
        n_samples = int(scfg.mesh_model.n_samples)
        input_size = int(scfg.denoiser_model.input_size)
        use_proj_cond_on_temb = bool(scfg.denoiser_model.get("use_proj_cond_on_temb", False))

        if use_proj_cond_on_temb:
            if num_verts_max is None:
                num_verts_max = int(scfg.get("num_verts_max", input_size))
            else:
                num_verts_max = int(num_verts_max)
            if num_verts is not None:
                num_verts = int(num_verts)
                config_num_verts = n_samples = input_size = num_verts
        else:
            if num_verts is not None:
                print(
                    "[MeshFlowPipeline] Ignoring num_verts: "
                    "denoiser_model.use_proj_cond_on_temb is disabled in config"
                )
            if num_verts_max is not None:
                print(
                    "[MeshFlowPipeline] Ignoring num_verts_max: "
                    "denoiser_model.use_proj_cond_on_temb is disabled in config"
                )
            num_verts = None
            num_verts_max = None

        if isinstance(dtype, str):
            if dtype == "float16":
                dtype = torch.float16
            elif dtype == "float32":
                dtype = torch.float32
            else:
                raise ValueError(f"Unsupported dtype: {dtype!r}")
        elif dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if num_verts is not None:
            print(
                f"  test-time scaling: num_verts={config_num_verts} "
                f"n_samples={n_samples} input_size={input_size}"
            )

        print(f"[MeshFlowPipeline] Reading checkpoint on CPU...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        print(f"  state_dict: {len(state)} tensors")

        mesh_cfg = OmegaConf.to_container(scfg.mesh_model, resolve=True)
        mesh_cfg["pretrained_model_name_or_path"] = str(ckpt_path)
        mesh_cfg["n_samples"] = n_samples
        if num_verts is not None:
            mesh_cfg["num_latents"] = num_verts
        print(f"[MeshFlowPipeline] Building mesh_model (VAE)...")
        mesh_model = build_mesh_model(mesh_cfg).to(device).eval()
        print(f"  mesh_model: {type(mesh_model).__name__} on {device}")

        visual_cfg = OmegaConf.to_container(scfg.visual_condition, resolve=True)
        embed_dim = resolve_visual_embed_dim(
            visual_cfg,
            fallback=int(scfg.denoiser_model.visual_condition_dim),
        )
        empty_visual_embeds = make_empty_visual_embeds(int(visual_cfg["image_size"]), embed_dim)
        print(
            "[MeshFlowPipeline] visual_encoder deferred "
            f"(loads on first reference image; zero cond shape={tuple(empty_visual_embeds.shape)})"
        )

        print(f"[MeshFlowPipeline] Building denoiser (DiT)...")
        denoiser_cfg = OmegaConf.to_container(scfg.denoiser_model, resolve=True)
        denoiser_cfg["pretrained_model_name_or_path"] = str(ckpt_path)
        denoiser_cfg["input_size"] = input_size
        denoiser = build_meshflow_dit(denoiser_cfg).to(device).eval()
        if hasattr(denoiser, "set_rope_input_resolution"):
            denoiser.set_rope_input_resolution(int(scfg.voxelize_resolution))
        print(
            f"  denoiser: {type(denoiser).__name__} "
            f"(layers={denoiser.depth}, input_size={denoiser.input_size}) on {device}"
        )

        if compile_models:
            if device.type != "cuda":
                print("[MeshFlowPipeline] torch.compile skipped (CUDA required)")
            else:
                print("[MeshFlowPipeline] torch.compile enabled:")
                print("  compiling mesh_model.decoder ...")
                mesh_model.decoder = torch.compile(mesh_model.decoder)
                print("  compiling denoiser ...")
                denoiser = torch.compile(denoiser)
                print("[MeshFlowPipeline] torch.compile done (first run will warm up)")

        scheduler = FlowMatchEulerDiscreteScheduler(
            **OmegaConf.to_container(scfg.denoise_scheduler, resolve=True)
        )

        pipeline = cls(
            models={
                "mesh_model": mesh_model,
                "visual_encoder": None,
                "denoiser": denoiser,
            },
            scheduler=scheduler,
            num_verts=config_num_verts,
            voxelize_resolution=int(scfg.voxelize_resolution),
            num_inference_steps=int(scfg.num_inference_steps),
            guidance_scale=float(scfg.guidance_scale),
            dtype=dtype,
        )
        pipeline._device = device
        pipeline._visual_encoder_type = str(scfg.visual_condition_type)
        pipeline._visual_encoder_cfg = visual_cfg
        pipeline._empty_visual_embeds = empty_visual_embeds
        pipeline._compile_models = compile_models
        pipeline.num_verts_max = num_verts_max
        num_verts_max_repr = num_verts_max if num_verts_max is not None else "n/a"
        print(
            f"[MeshFlowPipeline] Ready  device={device} dtype={dtype} "
            f"num_verts={config_num_verts} num_verts_max={num_verts_max_repr} "
            f"use_proj_cond_on_temb={use_proj_cond_on_temb} "
            f"steps={pipeline.num_inference_steps} "
            f"guidance_scale={pipeline.guidance_scale} compile={compile_models}"
        )
        return pipeline

    def to(self, device: Union[str, torch.device]) -> "MeshFlowPipeline":
        device = torch.device(device)
        self._device = device
        for model in self.models.values():
            if model is not None:
                model.to(device)
        return self

    def _ensure_visual_encoder(self) -> nn.Module:
        visual_encoder = self.models.get("visual_encoder")
        if visual_encoder is not None:
            return visual_encoder
        if self._visual_encoder_type is None or self._visual_encoder_cfg is None:
            raise RuntimeError("visual encoder config is not initialized")

        print(f"[MeshFlowPipeline] Loading visual_encoder ({self._visual_encoder_type})...")
        visual_encoder = build_dinov3_encoder(
            self._visual_encoder_type,
            self._visual_encoder_cfg,
        ).to(self._device).eval()
        if self._compile_models and self._device.type == "cuda":
            print("  compiling visual_encoder.dino_model ...")
            visual_encoder.dino_model = torch.compile(visual_encoder.dino_model)
        self.models["visual_encoder"] = visual_encoder
        print(f"  visual_encoder: {type(visual_encoder).__name__} on {self._device}")
        return visual_encoder

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Optional RMBG + foreground crop for reference images."""
        if not self.use_rmbg:
            return image.convert("RGB")
        output = self.rmbg_preprocess(image)
        if isinstance(output, list):
            output = output[0]
        return output.convert("RGB")

    def sample_surface_points(
        self,
        mesh: Union[str, Mesh],
        num_verts: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build points for RoPE conditioning.

        Meshes are surface-sampled; point clouds are normalized and randomly resampled directly,
        then voxelized in ``get_rope_cond``.

        Returns:
            Tensor of shape (1, N, 3) on pipeline device.
        """
        num_verts = num_verts or self.num_verts
        if isinstance(mesh, str):
            surface_pts = Mesh.load_rope_points(
                mesh,
                n_verts=num_verts,
                normalize_by="bsphere",
                obj_scale=2.0,
                preprocess=True,
                device=self.device,
            )
        elif int(mesh.faces.shape[0]) == 0:
            from ..utils.mesh import _normalize_points, _resample_point_cloud

            points = _normalize_points(mesh.verts, normalize_by="bsphere", size=2.0)
            surface_pts = _resample_point_cloud(points, num_verts)
        else:
            surface_pts, _ = mesh.sample_surface(num_verts)
        return surface_pts.unsqueeze(0).to(self.device)

    def get_rope_cond(self, surface_pc: torch.Tensor) -> dict:
        """Voxelize surface points for RoPE conditioning."""
        _, vox_idx = voxelize_pc(surface_pc, self.voxelize_resolution)
        return {"rope_input": vox_idx.to(self.device)}

    def get_visual_cond(
        self,
        image: Optional[Image.Image] = None,
        guidance_scale: Optional[float] = None,
        preprocess_image: bool = True,
    ) -> torch.Tensor:
        """
        Build visual tokens for DiT cross-attention.

        When `guidance_scale != 1`, returns concatenated [uncond, cond] for CFG.
        """
        guidance_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        use_cfg = guidance_scale != 1.0
        empty_visual_embeds = self._empty_visual_embeds
        if empty_visual_embeds is None:
            raise RuntimeError("empty visual embeds are not initialized")

        with torch.autocast(device_type=self.device.type, enabled=False):
            if image is not None:
                if preprocess_image:
                    image = self.preprocess_image(image)
                visual_encoder = self._ensure_visual_encoder()
                visual_cond = visual_encoder.encode_image(image.convert("RGB"))
            else:
                visual_cond = empty_visual_embeds
            visual_cond = visual_cond.to(device=self.device, dtype=self.dtype)

        if use_cfg:
            uncond = empty_visual_embeds.to(visual_cond).expand(
                visual_cond.shape[0], -1, -1
            )
            visual_cond = torch.cat([uncond, visual_cond], dim=0)
        return visual_cond

    def get_proj_cond_on_temb(
        self,
        num_verts: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Build normalized vertex-count condition for timestep embedding projection."""
        denoiser = self.models["denoiser"]
        if not denoiser.use_proj_cond_on_temb:
            return None
        if self.num_verts_max is None:
            raise RuntimeError("num_verts_max is required when use_proj_cond_on_temb is enabled")

        num_verts = int(num_verts or self.num_verts)
        guidance_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        proj_cond = torch.tensor(
            [[num_verts / self.num_verts_max]],
            dtype=torch.float32,
            device=self.device,
        )
        if guidance_scale != 1.0:
            proj_cond = proj_cond.repeat(2, 1)
        return proj_cond

    def sample_latent(
        self,
        visual_cond: torch.Tensor,
        joint_attention_kwargs: dict,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        disable_prog: bool = False,
    ) -> torch.Tensor:
        """Run flow-matching sampling and return final latents."""
        denoiser = self.models["denoiser"]
        steps = self.num_inference_steps if steps is None else int(steps)
        guidance_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        use_cfg = guidance_scale != 1.0
        sample_shape = (denoiser.input_size, denoiser.in_channels)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        denoiser_model_kwargs = {}
        proj_cond_on_temb = self.get_proj_cond_on_temb(guidance_scale=guidance_scale)
        if proj_cond_on_temb is not None:
            denoiser_model_kwargs["proj_cond_on_temb"] = proj_cond_on_temb

        latents = None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for latents, _ in flow_sample(
                self.scheduler,
                denoiser,
                sample_shape,
                steps=steps,
                visual_cond=visual_cond,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=use_cfg,
                generator=generator,
                device=self.device,
                disable_prog=disable_prog,
                joint_attention_kwargs=joint_attention_kwargs,
                denoiser_model_kwargs=denoiser_model_kwargs or None,
            ):
                pass
        return latents

    def decode_latent(
        self,
        latents: torch.Tensor,
        fill_holes: bool = False,
    ) -> Mesh:
        """Decode sampled latents with the MeshFlow VAE and extract a mesh."""
        mesh_model = self.models["mesh_model"]
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            decoded = mesh_model.decode(mesh_model.denormalize(latents))
            out_mesh = mesh_model.extract_mesh(
                decoded,
                fill_holes=fill_holes,
            )[0]
        return out_mesh

    @torch.no_grad()
    def run(
        self,
        mesh: Union[str, Mesh],
        image: Optional[Image.Image] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        preprocess_image: bool = True,
        return_latent: bool = False,
        disable_prog: bool = False,
    ):
        """
        End-to-end MeshFlow inference.

        Args:
            mesh: Input mesh/point-cloud path or `Mesh` instance for RoPE surface sampling.
            image: Optional reference image for visual conditioning.
            steps: Flow sampling steps (default: from config).
            guidance_scale: CFG scale (default: from config).
            seed: Random seed.
            preprocess_image: Whether to run image preprocessing before encoding.
            return_latent: If True, return `(mesh, latents)` instead of mesh only.
            disable_prog: Disable tqdm progress bar during sampling.
        """
        surface_pc = self.sample_surface_points(mesh)
        joint_kwargs = self.get_rope_cond(surface_pc)
        visual_cond = self.get_visual_cond(
            image,
            guidance_scale=guidance_scale,
            preprocess_image=preprocess_image,
        )
        latents = self.sample_latent(
            visual_cond,
            joint_kwargs,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            disable_prog=disable_prog,
        )
        out_mesh = self.decode_latent(latents)
        if return_latent:
            return out_mesh, latents
        return out_mesh
