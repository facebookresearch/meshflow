# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from meshflow.utils.mesh import Mesh
from meshflow.models.meshflow_vae import build_mesh_model

_MESH_EXTS = {".ply", ".glb", ".stl", ".obj"}


def select_latents_by_mask(
    z: torch.Tensor,
    verts_mask: torch.Tensor,
    downsample_ratio: int = 1,
) -> torch.Tensor:
    """Keep latent tokens whose vertex group has verts_mask > 0."""
    mask = verts_mask.squeeze(-1).bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    ratio = max(int(downsample_ratio), 1)
    if ratio > 1:
        pad = (-mask.shape[1]) % ratio
        if pad:
            mask = torch.cat(
                [mask, torch.zeros(mask.shape[0], pad, dtype=torch.bool, device=mask.device)],
                dim=1,
            )
        mask = mask.view(mask.shape[0], -1, ratio).any(dim=-1)
    mask = mask[:, : z.shape[1]]
    out = []
    for b in range(z.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        out.append(z[b, idx])
    max_len = max(t.shape[0] for t in out)
    if any(t.shape[0] != max_len for t in out):
        raise ValueError("select_latents_by_mask requires the same number of valid latents per batch item.")
    return torch.stack(out, dim=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MeshFlow VAE reconstruction inference")
    parser.add_argument(
        "--model_path",
        required=True,
        help="model bundle directory (must contain config.yaml and model.pth)",
    )
    parser.add_argument("--input", required=True, help="mesh file (.ply/.glb/.stl/.obj) or directory")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--sample_posterior",
        type=lambda s: s.lower() not in ("0", "false", "no", "off"),
        default=True,
        help="sample latent z from posterior (default: true; use false/0 to disable)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16"],
        help="autocast dtype (default: float16)",
    )
    parser.add_argument(
        "--no-dynamic_latent",
        action="store_true",
        help="decode all latents including padded slots (default: only verts_mask==1)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    sample_posterior = args.sample_posterior
    dynamic_latent = not args.no_dynamic_latent

    # Load config and checkpoint from model bundle.
    model_root = Path(args.model_path)
    if not model_root.is_dir():
        raise NotADirectoryError(f"model_path must be a directory: {model_root}")
    config_path = model_root / "config.yaml"
    ckpt_path = model_root / "model.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(config_path)
    data_cfg = cfg.data
    mesh_cfg = OmegaConf.to_container(cfg.system.mesh_model, resolve=True)
    mesh_cfg["pretrained_model_name_or_path"] = str(ckpt_path)
    mesh_model = build_mesh_model(mesh_cfg).to(device).eval()
    print(f"Loaded {type(mesh_model).__name__} from {args.model_path}")
    print(f"  n_verts={data_cfg.n_verts} max_degree={data_cfg.max_degree} point_feats_type={mesh_cfg.get('point_feats_type')}")
    print(f"  sample_posterior={sample_posterior} dynamic_latent={dynamic_latent} device={device} dtype={dtype}")

    # Collect input mesh paths.
    if Path(args.input).is_file():
        assert Path(args.input).suffix.lower() in _MESH_EXTS, f"Unsupported mesh file: {args.input}"
        meshes = [Path(args.input)]
    else:
        meshes = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in _MESH_EXTS)
    print(f"Found {len(meshes)} meshes in {args.input}")

    # Process each mesh.
    n_verts = int(data_cfg.n_verts)
    for mesh_path in meshes:
        # Build topology batch for the encoder.
        mesh = Mesh.load_mesh(str(mesh_path), normalize=True, normalize_by="bsphere", obj_scale=2.0, preprocess=True)
        num_verts = int(mesh.verts.shape[0])
        if n_verts >= 0 and num_verts > n_verts:
            print(f"Skip {mesh_path.name}: {num_verts} verts > n_verts={n_verts}")
            continue
        topo = mesh.build_topology(
            topology_feats_type=list(data_cfg.topology_feats_type),
            n_verts=n_verts,
            max_degree=int(data_cfg.max_degree),
        )
        batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in topo.items()
        }

        # Encode latents and decode a reconstructed mesh.
        with torch.autocast(device_type=device.type, dtype=dtype):
            input_feats = mesh_model.collect_input(batch)
            enc = mesh_model.encode(
                input_feats,
                batch["padded_verts"],
                sample_posterior=sample_posterior,
            )
            z = enc["z"]
            if dynamic_latent:
                z = select_latents_by_mask(
                    z,
                    batch["verts_mask"],
                    downsample_ratio=mesh_model.encoder.downsample_ratio,
                )
            print(f"{mesh_path.name}: decoder z.shape={tuple(z.shape)}")
            dec = mesh_model.decode(z)
            out_mesh = mesh_model.extract_mesh(dec)[0]

        # Save input and reconstruction.
        out_root = Path(args.output)
        input_ply = out_root / "inputs_meshes" / f"{mesh_path.stem}.ply"
        recon_ply = out_root / "vae_recon" / f"{mesh_path.stem}.ply"
        input_ply.parent.mkdir(parents=True, exist_ok=True)
        recon_ply.parent.mkdir(parents=True, exist_ok=True)
        mesh.to_trimesh().export(str(input_ply))
        out_mesh.to_trimesh().export(str(recon_ply))
        print(f"Saved {input_ply} and {recon_ply}")

if __name__ == "__main__":
    main()