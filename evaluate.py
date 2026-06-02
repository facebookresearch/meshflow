# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import point_cloud_utils as pcu
import trimesh
from numpy.typing import NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MeshMetrics:
    """Container for mesh evaluation metrics."""

    def __init__(
        self,
        filename: str,
        chamfer_distance: float,
        hausdorff_distance: float,
        gt_path: str = "",
    ) -> None:
        self.filename = filename
        self.chamfer_distance = chamfer_distance
        self.hausdorff_distance = hausdorff_distance
        self.gt_path = gt_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "filename": self.filename,
            "chamfer_distance": self.chamfer_distance,
            "hausdorff_distance": self.hausdorff_distance,
            "gt_path": self.gt_path,
        }


def load_mesh_from_path(path: str) -> Optional[trimesh.Trimesh]:
    """Load a mesh from a local file path."""
    try:
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        return mesh
    except Exception as e:
        logger.error(f"Failed to load mesh from {path}: {e}")
        return None


def normalize_mesh_to_unit_cube(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, float]:
    """
    Normalize mesh to fit within [-0.5, 0.5] unit cube.

    Args:
        mesh: Input mesh

    Returns:
        Tuple of (normalized_mesh, scale_factor)
    """
    # Get bounding box
    bounds = mesh.bounds  # shape: (2, 3) - min and max corners
    bbox_min = bounds[0]
    bbox_max = bounds[1]

    # Compute center and extent
    center = (bbox_max + bbox_min) / 2.0
    extent = bbox_max - bbox_min
    max_extent = np.max(extent)

    # Compute scale factor to fit in unit cube [-0.5, 0.5]
    scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0

    # Create normalized mesh (center at origin, scale to [-0.5, 0.5])
    normalized_vertices = (mesh.vertices - center) * scale_factor
    normalized_mesh = trimesh.Trimesh(
        vertices=normalized_vertices, faces=mesh.faces, process=False
    )

    return normalized_mesh, scale_factor


def sample_points_from_mesh(
    mesh: trimesh.Trimesh, n_points: int
) -> NDArray[np.float64]:
    """
    Sample points uniformly from mesh surface.

    Args:
        mesh: Input mesh
        n_points: Number of points to sample

    Returns:
        Array of sampled points with shape (n_points, 3)
    """
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def compute_chamfer_distance(
    points_a: NDArray[np.float64], points_b: NDArray[np.float64]
) -> float:
    """
    Compute bidirectional Chamfer distance between two point clouds using point-cloud-utils.

    Args:
        points_a: First point cloud (N, 3)
        points_b: Second point cloud (M, 3)

    Returns:
        Chamfer distance
    """
    return float(pcu.chamfer_distance(points_a, points_b))


def compute_hausdorff_distance(
    points_a: NDArray[np.float64], points_b: NDArray[np.float64]
) -> float:
    """
    Compute Hausdorff distance between two point clouds using point-cloud-utils.

    Args:
        points_a: First point cloud (N, 3)
        points_b: Second point cloud (M, 3)

    Returns:
        Hausdorff distance
    """
    return float(pcu.hausdorff_distance(points_a, points_b))


def evaluate_mesh_pair(
    gt_path: str, pred_path: str, n_sample_points: int, normalize: bool = False
) -> Optional[MeshMetrics]:
    """
    Evaluate a single pair of ground truth and predicted meshes.

    Args:
        gt_path: Path to ground truth mesh
        pred_path: Path to predicted mesh
        n_sample_points: Number of points to sample from each mesh
        normalize: If True, normalize gt mesh to [-0.5, 0.5] unit cube and apply
                   the same scale to pred mesh for unified metric computation

    Returns:
        MeshMetrics object with computed metrics, or None if evaluation fails
    """
    # Load meshes
    gt_mesh = load_mesh_from_path(gt_path)
    pred_mesh = load_mesh_from_path(pred_path)

    if gt_mesh is None or pred_mesh is None:
        return None

    try:
        # Apply normalization if requested
        if normalize:
            # Normalize gt mesh to unit cube and get scale factor
            gt_mesh_normalized, scale_factor = normalize_mesh_to_unit_cube(gt_mesh)

            # Apply same transformation to pred mesh
            # Use gt mesh's center for centering
            gt_bounds = gt_mesh.bounds
            gt_center = (gt_bounds[0] + gt_bounds[1]) / 2.0

            pred_vertices_normalized = (pred_mesh.vertices - gt_center) * scale_factor
            pred_mesh_normalized = trimesh.Trimesh(
                vertices=pred_vertices_normalized,
                faces=pred_mesh.faces,
                process=False,
            )

            # Use normalized meshes for evaluation
            gt_mesh = gt_mesh_normalized
            pred_mesh = pred_mesh_normalized

        # Sample points
        gt_points = sample_points_from_mesh(gt_mesh, n_sample_points)
        pred_points = sample_points_from_mesh(pred_mesh, n_sample_points)

        # Compute distances
        chamfer_dist = compute_chamfer_distance(gt_points, pred_points)
        hausdorff_dist = compute_hausdorff_distance(gt_points, pred_points)

        metrics = MeshMetrics(
            filename=os.path.basename(pred_path),
            chamfer_distance=chamfer_dist,
            hausdorff_distance=hausdorff_dist,
            gt_path=gt_path,
        )

        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate mesh pair {pred_path}: {e}")
        return None


_MESH_EXTENSIONS = {".glb", ".obj", ".ply", ".stl", ".off"}


def list_files_in_directory(directory: str) -> Sequence[str]:
    """List mesh files in a local directory."""
    try:
        root = Path(directory)
        if not root.is_dir():
            logger.error(f"Not a directory: {directory}")
            return []
        return sorted(
            f.name
            for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in _MESH_EXTENSIONS
        )
    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        return []


def find_matching_gt_file(pred_filename: str, gt_files: Sequence[str]) -> Optional[str]:
    """
    Find matching ground truth file for a prediction file.

    If filename contains '_pred_', takes the part before '_pred_' for matching.
    For example: 'model_pred_0.glb' -> matches 'model.glb'

    Args:
        pred_filename: Prediction file name
        gt_files: List of ground truth file names

    Returns:
        Matching GT file name or None
    """
    # Try exact match first
    if pred_filename in gt_files:
        return pred_filename

    # Extract stem and extension
    pred_path = Path(pred_filename)
    pred_stem = pred_path.stem
    pred_ext = pred_path.suffix

    # If filename contains '_pred_', take the part before it
    if "_pred_" in pred_stem:
        pred_stem = pred_stem.split("_pred_")[0]

    # Try matching with modified stem
    for gt_file in gt_files:
        gt_path = Path(gt_file)
        gt_stem = gt_path.stem

        # Match stem (with part after _pred_ removed if applicable)
        if gt_stem == pred_stem:
            return gt_file

        # Also try matching with same extension
        if gt_stem == pred_stem and gt_path.suffix == pred_ext:
            return gt_file

    return None


def _scalar_stats(arr: NDArray[Any]) -> Dict[str, float]:
    """Mean/std/median/min/max for a 1D metric array."""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_statistics(metrics_list: Sequence[MeshMetrics]) -> Dict[str, Any]:
    """Compute aggregate statistics from per-mesh metrics."""
    if not metrics_list:
        return {}

    return {
        "total_evaluated": len(metrics_list),
        "chamfer_distance": _scalar_stats(
            np.array([m.chamfer_distance for m in metrics_list])
        ),
        "hausdorff_distance": _scalar_stats(
            np.array([m.hausdorff_distance for m in metrics_list])
        ),
    }


def save_results(
    output_path: str,
    metrics_list: Sequence[MeshMetrics],
    statistics: Dict[str, Any],
) -> None:
    """
    Save evaluation results to text file.

    Args:
        output_path: Output file path
        metrics_list: List of individual metrics
        statistics: Computed statistics
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare output content
    lines = []
    lines.append("=" * 80)
    lines.append("MESH EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Overall statistics
    lines.append(f"Total meshes evaluated: {statistics['total_evaluated']}")
    lines.append("")

    # Chamfer Distance stats
    cd_stats = statistics["chamfer_distance"]
    lines.append("-" * 80)
    lines.append("CHAMFER DISTANCE")
    lines.append("-" * 80)
    lines.append(f"Mean:   {cd_stats['mean']:.6f}")
    lines.append(f"Std:    {cd_stats['std']:.6f}")
    lines.append(f"Median: {cd_stats['median']:.6f}")
    lines.append(f"Min:    {cd_stats['min']:.6f}")
    lines.append(f"Max:    {cd_stats['max']:.6f}")
    lines.append("")

    # Hausdorff Distance stats
    hd_stats = statistics["hausdorff_distance"]
    lines.append("-" * 80)
    lines.append("HAUSDORFF DISTANCE")
    lines.append("-" * 80)
    lines.append(f"Mean:   {hd_stats['mean']:.6f}")
    lines.append(f"Std:    {hd_stats['std']:.6f}")
    lines.append(f"Median: {hd_stats['median']:.6f}")
    lines.append(f"Min:    {hd_stats['min']:.6f}")
    lines.append(f"Max:    {hd_stats['max']:.6f}")
    lines.append("")

    # Individual results
    lines.append("=" * 80)
    lines.append("INDIVIDUAL RESULTS")
    lines.append("=" * 80)
    lines.append("")
    for metrics in metrics_list:
        lines.append(f"File: {metrics.filename}")
        lines.append(f"  Chamfer Distance:   {metrics.chamfer_distance:.6f}")
        lines.append(f"  Hausdorff Distance: {metrics.hausdorff_distance:.6f}")
        lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    json_path = output_path.replace(".txt", ".json")
    json_data = {
        "statistics": statistics,
        "individual_results": [m.to_dict() for m in metrics_list],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Results saved to {output_path} and {json_path}")

    # Print summary to console (excluding INDIVIDUAL RESULTS)
    summary_lines = []
    for line in lines:
        if "INDIVIDUAL RESULTS" in line:
            break
        summary_lines.append(line)

    logger.info("\n" + "\n".join(summary_lines))


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate mesh generation results by computing distance metrics"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Ground truth meshes directory",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Predicted meshes directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output results file path (.txt; .json written alongside)",
    )
    parser.add_argument(
        "--n_sample_points",
        type=int,
        default=5000,
        help="Number of points to sample from each mesh (default: 5000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize GT mesh to [-0.5, 0.5] unit cube and apply same scale to pred mesh",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting mesh evaluation...")
    logger.info(f"GT path: {args.gt_path}")
    logger.info(f"Pred path: {args.pred_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Sample points: {args.n_sample_points}")
    logger.info(f"Normalize: {args.normalize}")

    # List files
    logger.info("Listing prediction files...")
    pred_files = list_files_in_directory(args.pred_path)
    logger.info(f"Found {len(pred_files)} prediction files")

    logger.info("Listing ground truth files...")
    gt_files = list_files_in_directory(args.gt_path)
    logger.info(f"Found {len(gt_files)} ground truth files")

    if not pred_files:
        logger.error("No prediction files found")
        return

    if not gt_files:
        logger.error("No ground truth files found")
        return

    # Evaluate each prediction file
    metrics_list = []
    skipped = 0

    logger.info("Starting evaluation of mesh pairs...")
    logger.info("-" * 80)

    for idx, pred_file in enumerate(tqdm(pred_files, desc="Evaluating meshes"), 1):
        # Find matching GT file
        gt_file = find_matching_gt_file(pred_file, gt_files)

        if gt_file is None:
            logger.warning(f"No matching GT file found for {pred_file}")
            skipped += 1
            continue

        gt_path = str(Path(args.gt_path) / gt_file)
        pred_path = str(Path(args.pred_path) / pred_file)

        # Evaluate
        metrics = evaluate_mesh_pair(
            gt_path, pred_path, args.n_sample_points, args.normalize
        )

        if metrics is not None:
            metrics_list.append(metrics)

            # Print individual result immediately for visualization
            logger.info(f"\n[{idx}/{len(pred_files)}] {metrics.filename}")
            logger.info(f"  GT file:            {gt_file}")
            logger.info(f"  Chamfer Distance:   {metrics.chamfer_distance:.6f}")
            logger.info(f"  Hausdorff Distance: {metrics.hausdorff_distance:.6f}")
        else:
            skipped += 1

    logger.info("-" * 80)

    logger.info(f"Successfully evaluated {len(metrics_list)} mesh pairs")
    logger.info(f"Skipped {skipped} files due to errors or missing matches")

    if not metrics_list:
        logger.error("No meshes were successfully evaluated")
        return

    # Compute statistics
    logger.info("Computing statistics...")
    statistics = compute_statistics(metrics_list)

    # Save results
    logger.info("Saving results...")
    save_results(args.output_path, metrics_list, statistics)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()