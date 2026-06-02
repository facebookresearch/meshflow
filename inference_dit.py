# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from pathlib import Path

import torch
import trimesh
from PIL import Image

from meshflow.pipelines import MeshFlowPipeline
from meshflow.utils.dtype import AUTOCAST_DTYPE_CHOICES
from meshflow.utils.mesh import (
    DEFAULT_NUM_VERTS,
    GEOMETRY_EXTS,
    Mesh,
    resolve_num_verts_for_mesh,
)

_MESH_EXTS = set(GEOMETRY_EXTS)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MeshFlow DiT image-to-mesh inference")
    parser.add_argument(
        "--model_path",
        required=True,
        help="model bundle directory (must contain config.yaml and model.pth)",
    )
    parser.add_argument("--input", required=True, help="mesh file or directory (RoPE / shape cond)")
    parser.add_argument(
        "--ref_image",
        default=None,
        help="reference image file or directory (matched by stem); if omitted, use zero visual cond",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=None, help="sampling steps (default: from config)")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale (default: from config)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=AUTOCAST_DTYPE_CHOICES,
        help="autocast dtype: bf16, fp16, or fp32 (default: bf16)",
    )
    parser.add_argument(
        "--num_verts",
        type=int,
        default=DEFAULT_NUM_VERTS,
        help=(
            "target mesh vertex count sent to the DiT as proj_cond_on_temb control; "
            "roughly controls generated mesh resolution (proj_cond = num_verts / "
            "mesh_model.num_latents from config). For .glb inputs with fewer verts "
            "than num_latents, the file's vertex count is used instead. "
            "Only effective when denoiser_model.use_proj_cond_on_temb is enabled in config "
            f"(default: {DEFAULT_NUM_VERTS})"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile models for faster inference (CUDA only, default off)",
    )
    args = parser.parse_args()

    pipeline = MeshFlowPipeline.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        compile_models=args.compile,
        num_verts=args.num_verts,
    )
    steps = args.steps or pipeline.num_inference_steps
    guidance_scale = args.guidance_scale or pipeline.guidance_scale
    denoiser = pipeline.models["denoiser"]
    mesh_model = pipeline.models["mesh_model"]

    print(f"Loaded models from {args.model_path}")
    print(
        f"  num_verts={pipeline.num_verts} num_latents={pipeline.num_latents} "
        f"n_samples={mesh_model.encoder.num_enc_latents} "
        f"input_size={denoiser.input_size} use_proj_cond_on_temb={denoiser.use_proj_cond_on_temb} "
        f"steps={steps} guidance_scale={guidance_scale} "
        f"compile={args.compile} device={pipeline.device} dtype={args.dtype}"
    )

    input_root = Path(args.input)
    if input_root.is_file():
        assert input_root.suffix.lower() in _MESH_EXTS, f"Unsupported mesh: {input_root}"
        meshes = [input_root]
    else:
        meshes = sorted(p for p in input_root.iterdir() if p.suffix.lower() in _MESH_EXTS)
    print(f"Found {len(meshes)} meshes in {args.input}")

    has_ref_image = args.ref_image is not None
    ref_root = Path(args.ref_image) if has_ref_image else None
    out_root = Path(args.output)
    (out_root / "input_meshes").mkdir(parents=True, exist_ok=True)
    if has_ref_image:
        (out_root / "input_images").mkdir(parents=True, exist_ok=True)
    (out_root / "generated_meshes").mkdir(parents=True, exist_ok=True)

    for mesh_path in meshes:
        ref_path = None
        if has_ref_image:
            if ref_root.is_file():
                ref_path = ref_root
            else:
                ref_path = next(
                    (
                        ref_root / f"{mesh_path.stem}{ext}"
                        for ext in _IMAGE_EXTS
                        if (ref_root / f"{mesh_path.stem}{ext}").is_file()
                    ),
                    None,
                )
            if ref_path is None:
                print(f"Skip {mesh_path.name}: no ref_image")
                continue

        pil_img = Image.open(ref_path).convert("RGB") if ref_path is not None else None

        proj_num_verts = resolve_num_verts_for_mesh(
            mesh_path,
            pipeline.num_verts,
            pipeline.num_latents,
        )
        if denoiser.use_proj_cond_on_temb:
            proj = pipeline.get_proj_cond_on_temb(
                num_verts=proj_num_verts,
                guidance_scale=guidance_scale,
            )
            print(
                f"  {mesh_path.name}: proj_num_verts={proj_num_verts} "
                f"proj_cond_on_temb={proj} (num_verts / num_latents={pipeline.num_latents})"
            )

        # RoPE must match denoiser input_size; proj_cond may use per-mesh vertex count.
        surface_pc = pipeline.sample_surface_points(str(mesh_path), num_verts=pipeline.num_verts)
        joint_kwargs = pipeline.get_rope_cond(surface_pc)
        visual_cond = pipeline.get_visual_cond(
            pil_img,
            guidance_scale=guidance_scale,
            preprocess_image=False,
        )

        start_time = time.time()
        latents = pipeline.sample_latent(
            visual_cond,
            joint_kwargs,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=args.seed,
            num_verts=proj_num_verts,
        )
        denoise_time = time.time() - start_time
        print(f"Denoise time taken: {denoise_time:.2f} seconds")

        out_mesh = pipeline.decode_latent(latents)

        stem = mesh_path.stem
        input_ply = out_root / "input_meshes" / f"{stem}.ply"
        if Mesh.is_point_cloud_file(mesh_path):
            pts = Mesh.load_rope_points(
                str(mesh_path),
                n_verts=pipeline.num_verts,
                normalize_by="bsphere",
                obj_scale=2.0,
                preprocess=True,
            )
            trimesh.PointCloud(pts.cpu().numpy()).export(str(input_ply))
        else:
            Mesh.load_mesh(
                str(mesh_path),
                normalize=True,
                normalize_by="bsphere",
                obj_scale=2.0,
                preprocess=True,
            ).to_trimesh().export(str(input_ply))
        if pil_img is not None:
            pil_img.save(str(out_root / "input_images" / f"{stem}.png"))
        out_mesh.to_trimesh().export(str(out_root / "generated_meshes" / f"{stem}.glb"))
        print(f"Saved {stem}")


if __name__ == "__main__":
    main()