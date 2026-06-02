# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from meshflow.pipelines import MeshFlowPipeline
from meshflow.utils.dtype import AUTOCAST_DTYPE_CHOICES
from meshflow.utils.mesh import (
    DEFAULT_NUM_VERTS,
    GEOMETRY_EXTS,
    MESH_EXTS,
    POINT_CLOUD_EXTS,
    resolve_num_verts_for_mesh,
)

NUM_VERTS_MIN = 1024
NUM_VERTS_MAX = 8192
NUM_VERTS_STEP = 256
CHECKPOINT_REPO_ID = "facebook/meshflow"
LOCAL_CHECKPOINT_DIR = "ckpt/meshflow"
CHECKPOINT_CONFIG_FILENAME = "config.yaml"
CHECKPOINT_WEIGHTS_FILENAME = "model.pth"

NUM_VERTS_CONTROL_NOTE = (
    "`num_verts` is injected via `proj_cond_on_temb` as `num_verts / num_latents` "
    "(normalization uses `mesh_model.num_latents` from config, e.g. 4096). "
    "For `.glb` with fewer verts than `num_latents`, the file vertex count is used. "
    f"({NUM_VERTS_MIN}–{NUM_VERTS_MAX}, default {DEFAULT_NUM_VERTS})."
)
NUM_VERTS_UNSUPPORTED_NOTE = (
    "This checkpoint has `denoiser_model.use_proj_cond_on_temb: false`, so generated "
    "mesh resolution cannot be controlled from the UI."
)

# --- Plot settings ---

PLOT_SCENE_BG = "#0b1220"
PLOT_MESH_COLOR = "#64748b"
PLOT_MESH_OPACITY = 0.48
PLOT_WIRE_COLOR = "#f8fafc"
PLOT_WIRE_HALO_COLOR = "rgba(15, 23, 42, 0.72)"
PLOT_WIRE_WIDTH = 1.8
PLOT_WIRE_HALO_WIDTH = 3.2
PLOT_AXIS_RANGE = 1.15
PLOT_CAMERA = dict(
    eye=dict(x=0.0, y=-2.4, z=0.0),
    center=dict(x=0.0, y=0.0, z=0.0),
    up=dict(x=0.0, y=0.0, z=1.0),
)

# --- App copy ---

APP_TITLE = "MeshFlow — Artistic Mesh Generation in Under 1 Second"
APP_TAB_TITLE = "MeshFlow — Artistic Mesh Generation"
PAPER_SUBTITLE = (
    "Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer"
)
PAPER_AUTHORS = (
    "Weiyu Li, Antoine Toisoul, Tom Monnier, Roman Shapovalov, "
    "Rakesh Ranjan, Ping Tan, Andrea Vedaldi"
)
HERO_DESC = (
    "Upload a mesh or point cloud, optionally add a reference image, "
    "and generate a new artistic mesh with flow-based diffusion."
)

# --- Gradio theme & assets ---

MESHFLOW_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_700",
    block_border_color_dark="*neutral_700",
    block_border_width="1px",
    block_label_text_size="*text_sm",
    block_label_text_weight="600",
    block_title_text_weight="700",
    block_shadow="0 10px 30px rgba(0, 0, 0, 0.25)",
    block_shadow_dark="0 10px 30px rgba(0, 0, 0, 0.25)",
    body_text_color="*neutral_200",
    body_text_color_dark="*neutral_200",
    button_large_padding="14px 24px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    slider_color="*primary_500",
)

FORCE_DARK_MODE_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.href);
    }
}
"""

CUSTOM_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
}

.dark .gradio-container,
.dark .main,
.dark .contain {
    background: #0f172a !important;
    color: #e2e8f0 !important;
}

.mf-hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 48%, #0e7490 100%);
    border-radius: 18px;
    padding: 28px 32px;
    color: #f8fafc;
    margin-bottom: 18px;
    box-shadow: 0 18px 45px rgba(30, 27, 75, 0.22);
}

.mf-hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: linear-gradient(135deg, #fde047 0%, #facc15 55%, #eab308 100%);
    border: 1px solid rgba(253, 224, 71, 0.85);
    color: #422006;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(234, 179, 8, 0.35);
}

.mf-hero h1 {
    margin: 0 0 8px 0;
    font-size: 2.35rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.mf-subtitle {
    margin: 0 0 10px 0;
    max-width: 860px;
    color: rgba(248, 250, 252, 0.88);
    font-size: 1.02rem;
    line-height: 1.55;
}

.mf-authors {
    margin: 0 0 14px 0;
    max-width: 860px;
    color: rgba(203, 213, 225, 0.88);
    font-size: 0.92rem;
    line-height: 1.5;
}

.mf-desc {
    margin: 0;
    color: rgba(226, 232, 240, 0.92);
    font-size: 0.95rem;
    line-height: 1.6;
}

.mf-panel {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 10px 12px 12px !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22) !important;
}

.mf-panel .block {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

.mf-panel > .gap,
.mf-input-column > .gap {
    gap: 10px !important;
}

.mf-section-head {
    margin: 0 0 6px 0;
    padding: 8px 0 0 12px;
}

.mf-section-title {
    margin: 0 0 3px 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.25;
}

.mf-section-note {
    margin: 0;
    color: #94a3b8;
    font-size: 0.84rem;
    line-height: 1.4;
}

.mf-section-note code {
    font-size: 0.82rem;
    color: #cbd5e1;
    background: #334155;
    padding: 1px 5px;
    border-radius: 4px;
}

.mf-input-column .label-wrap span,
.mf-input-column label,
.mf-input-column .prose,
.mf-input-column .markdown-text,
.mf-viewer-panel .label-wrap span {
    color: #e2e8f0 !important;
}

.mf-input-column .block-label span,
.mf-input-column .accordion .label-wrap span {
    color: #f1f5f9 !important;
}

.mf-panel .label-wrap,
.mf-panel .gr-file,
.mf-panel .gr-image,
.mf-panel .gr-plot {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.mf-generate-wrap {
    margin-top: 2px;
}

.mf-generate-wrap button {
    width: 100%;
    border: none !important;
    background: linear-gradient(90deg, #4f46e5 0%, #0891b2 100%) !important;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.28) !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}

.mf-generate-wrap button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

.mf-download-wrap {
    margin-top: 8px;
}

.mf-download-wrap button {
    width: 100%;
    font-weight: 600 !important;
}

.mf-viewer-panel {
    min-height: 430px;
}

.mf-viewer-panel > .gap {
    gap: 8px !important;
}

.mf-viewer-panel .plot-container,
.mf-viewer-panel .gr-panel {
    min-height: 390px;
}
"""


# --- HTML builders ---


def format_exts(exts: tuple[str, ...]) -> str:
    return ", ".join(sorted(exts))


def panel_header(title: str, note: str = "") -> str:
    note_html = f'  <div class="mf-section-note">{note}</div>\n' if note else ""
    return f"""
<div class="mf-section-head">
  <div class="mf-section-title">{title}</div>
{note_html}</div>
"""


def build_hero_html() -> str:
    return f"""
<div class="mf-hero">
  <div class="mf-hero-badge">CVPR 2026 Highlight</div>
  <h1>{APP_TITLE}</h1>
  <p class="mf-subtitle">{PAPER_SUBTITLE}</p>
  <p class="mf-authors">{PAPER_AUTHORS}</p>
  <p class="mf-desc">{HERO_DESC}</p>
</div>
"""


# --- Upload helpers ---


def resolve_upload_path(upload: Any) -> Optional[str]:
    if upload is None:
        return None
    if isinstance(upload, str):
        return upload
    if isinstance(upload, dict):
        return upload.get("path") or upload.get("name")
    if hasattr(upload, "name"):
        return upload.name
    return str(upload)


def validate_geometry_upload(upload: Any) -> str:
    path = resolve_upload_path(upload)
    if path is None:
        raise gr.Error(
            "Please upload a mesh (.glb/.obj/.stl/.ply) or point cloud (.ply/.pcd/.xyz/.npz)."
        )

    ext = Path(path).suffix.lower()
    if ext not in GEOMETRY_EXTS:
        raise gr.Error(f"Unsupported format: {ext}. Supported: {format_exts(GEOMETRY_EXTS)}")
    return path


# --- Mesh visualization ---


def plotly_scene_layout(fig: go.Figure) -> go.Figure:
    axis = dict(
        visible=False,
        showbackground=False,
        range=[-PLOT_AXIS_RANGE, PLOT_AXIS_RANGE],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            bgcolor=PLOT_SCENE_BG,
            aspectmode="cube",
            camera=PLOT_CAMERA,
        ),
    )
    return fig


def _extract_wireframe_segments(
    verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    wire_x: list[float] = []
    wire_y: list[float] = []
    wire_z: list[float] = []
    for i0, i1 in edges:
        p0, p1 = verts[i0], verts[i1]
        wire_x.extend((float(p0[0]), float(p1[0]), None))
        wire_y.extend((float(p0[1]), float(p1[1]), None))
        wire_z.extend((float(p0[2]), float(p1[2]), None))
    return wire_x, wire_y, wire_z


def _make_wireframe_traces(
    wire_x: list[float],
    wire_y: list[float],
    wire_z: list[float],
) -> list[go.Scatter3d]:
    common = dict(x=wire_x, y=wire_y, z=wire_z, mode="lines", hoverinfo="skip")
    return [
        go.Scatter3d(
            **common,
            line=dict(color=PLOT_WIRE_HALO_COLOR, width=PLOT_WIRE_HALO_WIDTH),
            name="wireframe-halo",
        ),
        go.Scatter3d(
            **common,
            line=dict(color=PLOT_WIRE_COLOR, width=PLOT_WIRE_WIDTH),
            name="wireframe",
        ),
    ]


def canonicalize_mesh_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    rot_x = trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1.0, 0.0, 0.0])
    mesh.apply_transform(rot_x)
    return mesh


def mesh_to_plotly(mesh: trimesh.Trimesh) -> go.Figure:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    wire_x, wire_y, wire_z = _extract_wireframe_segments(verts, faces)

    mesh_trace = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=PLOT_MESH_COLOR,
        opacity=PLOT_MESH_OPACITY,
        flatshading=True,
        lighting=dict(
            ambient=0.72,
            diffuse=0.45,
            specular=0.08,
            roughness=0.85,
            fresnel=0.05,
        ),
        lightposition=dict(x=0.35, y=-0.6, z=1.8),
        name="mesh",
        showscale=False,
    )
    fig = go.Figure(data=[mesh_trace, *_make_wireframe_traces(wire_x, wire_y, wire_z)])
    return plotly_scene_layout(fig)


# --- Inference ---


def _checkpoint_bundle_valid(root: Path) -> bool:
    return (root / CHECKPOINT_CONFIG_FILENAME).is_file() and (
        root / CHECKPOINT_WEIGHTS_FILENAME
    ).is_file()


def resolve_model_path(model_path: Optional[str]) -> str:
    """Return a local checkpoint directory, downloading from Hugging Face Hub if needed."""
    if model_path:
        root = Path(model_path)
        if _checkpoint_bundle_valid(root):
            return str(root.resolve())
        raise FileNotFoundError(
            f"model_path must contain {CHECKPOINT_CONFIG_FILENAME} and "
            f"{CHECKPOINT_WEIGHTS_FILENAME}: {root}"
        )

    local_default = Path(__file__).resolve().parent / LOCAL_CHECKPOINT_DIR
    if _checkpoint_bundle_valid(local_default):
        print(f"[MeshFlow] Using local checkpoint bundle at {local_default.resolve()}")
        return str(local_default.resolve())

    bundle_dir = Path(
        os.environ.get("MESHFLOW_MODEL_PATH", Path.home() / ".cache" / "meshflow")
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for filename in (CHECKPOINT_CONFIG_FILENAME, CHECKPOINT_WEIGHTS_FILENAME):
        if not (bundle_dir / filename).is_file():
            downloaded = hf_hub_download(
                repo_id=CHECKPOINT_REPO_ID,
                filename=filename,
                local_dir=str(bundle_dir),
            )
            print(f"[MeshFlow] Downloaded {filename} to {downloaded}")
    if not _checkpoint_bundle_valid(bundle_dir):
        raise FileNotFoundError(
            f"Failed to prepare checkpoint bundle at {bundle_dir} "
            f"from {CHECKPOINT_REPO_ID}"
        )
    print(f"[MeshFlow] Using checkpoint bundle at {bundle_dir.resolve()}")
    return str(bundle_dir.resolve())


def read_config_num_verts(model_path: str) -> int:
    cfg = OmegaConf.load(Path(model_path) / "config.yaml")
    return int(cfg.data.n_verts)


def read_config_num_latents(model_path: str) -> int:
    cfg = OmegaConf.load(Path(model_path) / "config.yaml")
    return int(cfg.system.mesh_model.num_latents)


def read_use_proj_cond_on_temb(model_path: str) -> bool:
    cfg = OmegaConf.load(Path(model_path) / "config.yaml")
    return bool(cfg.system.denoiser_model.get("use_proj_cond_on_temb", False))


def supports_num_verts_control(pipeline: MeshFlowPipeline) -> bool:
    return bool(pipeline.models["denoiser"].use_proj_cond_on_temb)


def clamp_num_verts(value: int) -> int:
    value = int(value)
    if value < NUM_VERTS_MIN or value > NUM_VERTS_MAX:
        raise ValueError(
            f"num_verts must be between {NUM_VERTS_MIN} and {NUM_VERTS_MAX}, got {value}"
        )
    remainder = (value - NUM_VERTS_MIN) % NUM_VERTS_STEP
    if remainder:
        value -= remainder
    return value


def load_pipeline(
    args: argparse.Namespace,
    num_verts: Optional[int] = None,
) -> MeshFlowPipeline:
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if num_verts is not None:
        num_verts = clamp_num_verts(num_verts)
    return MeshFlowPipeline.from_pretrained(
        args.model_path,
        device=device,
        dtype=args.dtype,
        compile_models=args.compile,
        num_verts=num_verts,
    )


def resolve_default_num_verts(args: argparse.Namespace) -> int:
    if args.num_verts is not None:
        return clamp_num_verts(args.num_verts)
    return clamp_num_verts(DEFAULT_NUM_VERTS)


@torch.no_grad()
def run_meshflow(
    pipeline_state: dict,
    runtime_args: argparse.Namespace,
    input_file: Any,
    input_image: Optional[Any],
    steps: int,
    guidance_scale: float,
    seed: int,
    num_verts: Optional[int] = None,
) -> tuple[go.Figure, str, dict]:
    supports_num_verts_scaling = getattr(runtime_args, "supports_num_verts_scaling", False)
    pipeline = pipeline_state["pipeline"]
    geometry_path = validate_geometry_upload(input_file)

    proj_num_verts = None
    if supports_num_verts_scaling:
        if num_verts is None:
            raise ValueError("num_verts is required when use_proj_cond_on_temb is enabled")
        num_verts = clamp_num_verts(num_verts)
        if pipeline_state.get("num_verts") != num_verts:
            pipeline = load_pipeline(runtime_args, num_verts=num_verts)
            pipeline_state = {
                "pipeline": pipeline,
                "num_verts": num_verts,
            }
        proj_num_verts = resolve_num_verts_for_mesh(
            Path(geometry_path),
            num_verts,
            pipeline.num_latents,
        )

    out_mesh = pipeline.run(
        geometry_path,
        image=input_image,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        preprocess_image=False,
        disable_prog=False,
        num_verts=proj_num_verts,
    )
    mesh = canonicalize_mesh_orientation(out_mesh.to_trimesh())

    fd, download_path = tempfile.mkstemp(suffix=".glb", prefix="meshflow_")
    os.close(fd)
    mesh.export(download_path)

    return mesh_to_plotly(mesh), download_path, pipeline_state


# --- UI ---


def build_ui(
    pipeline: MeshFlowPipeline,
    args: argparse.Namespace,
    default_num_verts: int,
    config_num_latents: int,
    supports_num_verts_scaling: bool,
) -> gr.Blocks:
    mesh_exts = format_exts(MESH_EXTS)
    pc_exts = format_exts(POINT_CLOUD_EXTS)
    pipeline_state = gr.State(
        {
            "pipeline": pipeline,
            "num_verts": default_num_verts,
        }
    )

    with gr.Blocks(
        title=APP_TAB_TITLE,
        theme=MESHFLOW_THEME,
        css=CUSTOM_CSS,
        js=FORCE_DARK_MODE_JS,
    ) as demo:
        gr.HTML(build_hero_html())

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="mf-input-column"):
                with gr.Group(elem_classes="mf-panel"):
                    gr.HTML(
                        panel_header(
                            "Input Geometry",
                            f"Supported mesh: <code>{mesh_exts}</code> · "
                            f"point cloud: <code>{pc_exts}</code>",
                        )
                    )
                    input_file = gr.File(
                        label="Upload file",
                        file_types=list(GEOMETRY_EXTS),
                    )

                with gr.Group(elem_classes="mf-panel"):
                    gr.HTML(
                        panel_header(
                            "Reference Image (Optional)",
                            "Optional visual conditioning from <code>.png</code>, "
                            "<code>.jpg</code>, or <code>.webp</code>.",
                        )
                    )
                    input_image = gr.Image(type="pil", label="Reference image", height=240)

                with gr.Group(elem_classes="mf-generate-wrap"):
                    run_btn = gr.Button("Generate Mesh", variant="primary", size="lg")

                with gr.Accordion("Advanced options", open=False):
                    gr.Markdown(
                        NUM_VERTS_UNSUPPORTED_NOTE,
                        visible=not supports_num_verts_scaling,
                    )
                    gr.Markdown(
                        NUM_VERTS_CONTROL_NOTE,
                        visible=supports_num_verts_scaling,
                    )
                    gr.Markdown(
                        f"Normalization divisor: **num_latents = {config_num_latents}** "
                        f"(from `mesh_model.num_latents` in config).",
                        visible=supports_num_verts_scaling,
                    )
                    num_verts = gr.Slider(
                        NUM_VERTS_MIN,
                        NUM_VERTS_MAX,
                        value=default_num_verts,
                        step=NUM_VERTS_STEP,
                        label="num_verts (proj_cond numerator, num_verts / num_latents)",
                        visible=supports_num_verts_scaling,
                    )
                    seed = gr.Number(value=args.seed, precision=0, label="Random seed")
                    guidance = gr.Slider(
                        1.0,
                        15.0,
                        value=args.guidance_scale or pipeline.guidance_scale,
                        step=0.1,
                        label="Classifier-free guidance",
                    )
                    steps = gr.Slider(
                        1,
                        100,
                        value=args.steps or pipeline.num_inference_steps,
                        step=1,
                        label="Sampling steps",
                    )

            with gr.Column(scale=6):
                with gr.Group(elem_classes="mf-panel mf-viewer-panel"):
                    gr.HTML(
                        panel_header(
                            "Generated Mesh",
                            "Generated artist-like mesh.",
                        )
                    )
                    mesh_out_plot = gr.Plot(label="Generated Mesh")
                    with gr.Group(elem_classes="mf-download-wrap"):
                        mesh_download = gr.DownloadButton("Download GLB", variant="secondary")

        run_inputs = [
            pipeline_state,
            gr.State(args),
            input_file,
            input_image,
            steps,
            guidance,
            seed,
        ]
        if supports_num_verts_scaling:
            run_inputs.append(num_verts)

        run_btn.click(
            fn=run_meshflow,
            inputs=run_inputs,
            outputs=[mesh_out_plot, mesh_download, pipeline_state],
            show_progress="full",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeshFlow Gradio demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            f"Model bundle directory (config.yaml + model.pth). "
            f"If omitted, use local {LOCAL_CHECKPOINT_DIR}/ or download from "
            f"Hugging Face ({CHECKPOINT_REPO_ID})."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=AUTOCAST_DTYPE_CHOICES,
        help="autocast dtype: bf16, fp16, or fp32 (default: fp16)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--num_verts",
        type=int,
        default=None,
        help=(
            "initial proj_cond numerator (num_verts / mesh_model.num_latents). "
            "Only effective when denoiser_model.use_proj_cond_on_temb is enabled in config "
            f"({NUM_VERTS_MIN}-{NUM_VERTS_MAX}; default: {DEFAULT_NUM_VERTS})"
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile models for faster inference (CUDA only, default on)",
    )
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    supports_num_verts_scaling = read_use_proj_cond_on_temb(args.model_path)
    if not supports_num_verts_scaling and args.num_verts is not None:
        print(
            "[MeshFlow] Ignoring --num_verts: "
            "denoiser_model.use_proj_cond_on_temb is disabled in config"
        )

    if supports_num_verts_scaling:
        default_num_verts = resolve_default_num_verts(args)
        pipeline = load_pipeline(args, num_verts=default_num_verts)
    else:
        default_num_verts = read_config_num_verts(args.model_path)
        pipeline = load_pipeline(args)

    supports_num_verts_scaling = supports_num_verts_control(pipeline)
    args.supports_num_verts_scaling = supports_num_verts_scaling

    config_num_latents = read_config_num_latents(args.model_path)
    demo = build_ui(
        pipeline,
        args,
        default_num_verts,
        config_num_latents,
        supports_num_verts_scaling,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
