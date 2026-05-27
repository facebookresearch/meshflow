# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

MESH_EXTS = (".stl", ".obj", ".ply", ".glb", ".gltf")
POINT_CLOUD_EXTS = (".pcd", ".xyz", ".pts", ".npy", ".npz")
GEOMETRY_EXTS = MESH_EXTS + POINT_CLOUD_EXTS


def _normalize_points(
    points: torch.Tensor,
    normalize_by: str = "bsphere",
    size: float = 2.0,
) -> torch.Tensor:
    if normalize_by == "bbox":
        bbox_min, bbox_max = points.min(0).values, points.max(0).values
        return (points - (bbox_min + bbox_max) / 2) * (size / (bbox_max - bbox_min).max())
    if normalize_by == "bsphere":
        center = (points.min(0).values + points.max(0).values) / 2
        radius = torch.linalg.norm(points - center, dim=-1).max() * 2
        return (points - center) * (size / radius)
    raise NotImplementedError(f"Invalid normalize_by: {normalize_by}")


def _resample_points(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    if points.shape[0] >= num_points:
        idx = torch.randperm(points.shape[0], device=points.device)[:num_points]
        return points[idx]
    idx = torch.randint(0, points.shape[0], (num_points,), device=points.device)
    return points[idx]


def _resample_point_cloud(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    if points.shape[0] < num_points:
        raise ValueError(
            f"Point cloud must have at least {num_points} points, got {int(points.shape[0])}"
        )
    idx = torch.randperm(points.shape[0], device=points.device)[:num_points]
    return points[idx]


def _read_point_cloud_file(filename: str) -> torch.Tensor:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npy":
        arr = np.load(filename)
    elif ext == ".npz":
        data = np.load(filename)
        for key in ("points", "pts", "xyz", "vertices", "data"):
            if key in data:
                arr = data[key]
                break
        else:
            raise ValueError(f"No point array found in {filename}")
    elif ext in (".xyz", ".pts"):
        arr = np.loadtxt(filename, dtype=np.float32)
    elif ext == ".pcd":
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        start = 0
        for i, line in enumerate(lines):
            header = line.upper()
            if header.startswith("POINTS") or header.startswith("WIDTH"):
                start = i + 1
                break
        arr = np.loadtxt(lines[start:], dtype=np.float32)
    else:
        raise NotImplementedError(f"Unsupported point cloud format: {ext}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] % 3 != 0:
            raise ValueError(f"Invalid point array shape in {filename}")
        arr = arr.reshape(-1, 3)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 coordinates per point in {filename}")
    return torch.from_numpy(arr[..., :3].copy())


def _try_read_point_cloud_ply(filename: str) -> Optional[torch.Tensor]:
    loaded = trimesh.load(filename, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        return torch.from_numpy(np.asarray(loaded.vertices, dtype=np.float32))
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) == 0:
        return torch.from_numpy(np.asarray(loaded.vertices, dtype=np.float32))
    if isinstance(loaded, trimesh.Scene):
        points = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.PointCloud):
                points.append(geometry.vertices)
            elif isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) == 0:
                points.append(geometry.vertices)
        if points:
            return torch.from_numpy(np.concatenate(points, axis=0).astype(np.float32))
    return None


def detect_and_triangulate_holes(
    triangles: List[Tuple[int, int, int]],
    max_cycle_size: int = 20,
) -> List[Tuple[int, int, int]]:
    """Find boundary loops and fan-triangulate holes in a triangle soup."""
    edge_to_triangles: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for tri in triangles:
        edges_in_tri = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[0], tri[2]])),
        ]
        for edge in edges_in_tri:
            edge_to_triangles.setdefault(edge, []).append(tri)

    boundary_edges = {edge for edge, tris in edge_to_triangles.items() if len(tris) == 1}
    if not boundary_edges:
        return []

    boundary_vertices: set[int] = set()
    boundary_adjacency: Dict[int, List[int]] = {}
    for v1, v2 in boundary_edges:
        boundary_vertices.update((v1, v2))
        boundary_adjacency.setdefault(v1, []).append(v2)
        boundary_adjacency.setdefault(v2, []).append(v1)

    def find_cycles_dfs(start_vertex: int) -> List[List[int]]:
        cycles: List[List[int]] = []
        visited_in_path: set[int] = set()
        path: List[int] = []

        def dfs(current: int) -> None:
            visited_in_path.add(current)
            path.append(current)
            has_unvisited_neighbor = False
            for neighbor in boundary_adjacency.get(current, []):
                if neighbor == start_vertex and len(path) >= 3:
                    if 4 <= len(path) <= max_cycle_size:
                        cycles.append(path[:])
                elif neighbor not in visited_in_path and len(path) < max_cycle_size:
                    has_unvisited_neighbor = True
                    dfs(neighbor)
            if not has_unvisited_neighbor and len(path) >= 3:
                if 4 <= len(path) + 1 <= max_cycle_size:
                    if start_vertex not in boundary_adjacency.get(current, []):
                        missing_edge = tuple(sorted([current, start_vertex]))
                        if missing_edge in edge_to_triangles:
                            cycles.append(path[:])
            path.pop()
            visited_in_path.remove(current)

        dfs(start_vertex)
        return cycles

    all_cycles: List[List[int]] = []
    processed_cycles: set[Tuple[int, ...]] = set()
    for start_v in boundary_vertices:
        for cycle in find_cycles_dfs(start_v):
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            reversed_cycle = tuple(reversed(normalized))
            if normalized not in processed_cycles and reversed_cycle not in processed_cycles:
                processed_cycles.add(normalized)
                all_cycles.append(cycle)

    new_triangles: List[Tuple[int, int, int]] = []
    for loop in all_cycles:
        v0 = loop[0]
        for i in range(1, len(loop) - 1):
            new_triangles.append(tuple(sorted([v0, loop[i], loop[i + 1]])))
    return new_triangles


def extract_mesh_from_verts_normals_edges(
    vertices: torch.Tensor,
    vertices_normal: torch.Tensor,
    edges: torch.Tensor,
    fill_holes: bool = False,
    max_cycle_size: int = 10,
    device: torch.device | None = None,
) -> Mesh:
    """Build a Mesh from predicted vertices, normals, and edge pairs."""
    if device is None:
        device = vertices.device

    vertices = vertices.to(device, dtype=torch.float32)
    vertices_normal = vertices_normal.to(device, dtype=torch.float32)
    edges = edges.to(device, dtype=torch.long)

    num_vertices = vertices.shape[0]
    if edges.numel() == 0 or num_vertices < 3:
        raise ValueError(
            f"Cannot extract mesh: invalid input (edges={edges.numel()}, vertices={num_vertices})"
        )

    edges = torch.clamp(edges, 0, num_vertices - 1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.numel() == 0:
        raise ValueError("Cannot extract mesh: no edges after removing self-loops")

    adjacency_list: List[List[int]] = [[] for _ in range(num_vertices)]
    edge_set: set[Tuple[int, int]] = set()
    for e in edges.cpu().numpy():
        v1, v2 = int(e[0]), int(e[1])
        if (v1, v2) not in edge_set and (v2, v1) not in edge_set:
            adjacency_list[v1].append(v2)
            adjacency_list[v2].append(v1)
            edge_set.add((v1, v2))

    max_neighbors = max(len(adj) for adj in adjacency_list) if adjacency_list else 0
    if max_neighbors == 0:
        raise ValueError("Cannot extract mesh: no vertex neighbors in edge graph")

    adj_tensor = torch.full((num_vertices, max_neighbors), -1, dtype=torch.long, device=device)
    for i, neighbors in enumerate(adjacency_list):
        if neighbors:
            adj_tensor[i, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.long, device=device)

    triangles: List[Tuple[int, int, int]] = []
    for v1, v2 in edge_set:
        n1_mask = adj_tensor[v1] >= 0
        n2_mask = adj_tensor[v2] >= 0
        if not (n1_mask.any() and n2_mask.any()):
            continue
        neighbors_v1 = adj_tensor[v1][n1_mask]
        neighbors_v2 = adj_tensor[v2][n2_mask]
        matches = neighbors_v1.unsqueeze(1) == neighbors_v2.unsqueeze(0)
        if not matches.any():
            continue
        v1_indices, _ = torch.where(matches)
        for v3 in neighbors_v1[v1_indices].unique():
            if v3 != v1 and v3 != v2:
                triangles.append(tuple(sorted([v1, v2, int(v3.item())])))

    triangles = list(set(triangles))
    if not triangles:
        raise ValueError("Cannot extract mesh: no triangles formed from edges")

    if fill_holes:
        extra = detect_and_triangulate_holes(triangles, max_cycle_size=max_cycle_size)
        if extra:
            triangles = list(set(triangles + extra))

    triangle_tensor = torch.tensor(triangles, dtype=torch.long, device=device)
    triangle_vertices = vertices[triangle_tensor]
    edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    face_normals = torch.linalg.cross(edge1, edge2, dim=1)
    face_normals_norm = torch.linalg.norm(face_normals, dim=1, keepdim=True)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    face_normals = torch.where(
        face_normals_norm > 1e-8,
        face_normals / face_normals_norm,
        fallback.expand_as(face_normals),
    )

    triangle_normals = vertices_normal[triangle_tensor]
    avg_vertex_normals = triangle_normals.mean(dim=1)
    avg_normals_norm = torch.linalg.norm(avg_vertex_normals, dim=1, keepdim=True)
    avg_vertex_normals = torch.where(
        avg_normals_norm > 1e-8,
        avg_vertex_normals / avg_normals_norm,
        fallback.expand_as(avg_vertex_normals),
    )

    faces = triangle_tensor.clone()
    flip_mask = (face_normals * avg_vertex_normals).sum(dim=1) < 0
    faces[flip_mask, 1], faces[flip_mask, 2] = faces[flip_mask, 2], faces[flip_mask, 1]
    faces = torch.clamp(faces, 0, num_vertices - 1)

    mesh = Mesh(verts=vertices, faces=faces, device=device)
    mesh._v_nrm = vertices_normal
    return mesh


class Mesh:
    """Mesh class for loading, augmentation, and topology feature extraction."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, device: torch.device | None = None) -> None:
        """Args:
            verts: (N, 3) vertex positions.
            faces: (F, 3) triangle indices.
            device: target device; defaults to verts.device.
        """
        self.device = device if device is not None else verts.device
        self.verts = verts.to(self.device, dtype=torch.float32)
        self.faces = faces.to(self.device, dtype=torch.int64)
        self._v_nrm: Optional[torch.Tensor] = None
        self._edges: Optional[torch.Tensor] = None

    @classmethod
    def is_point_cloud_file(cls, filename: str) -> bool:
        """True for point-cloud formats, including faceless `.ply` point clouds."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in POINT_CLOUD_EXTS:
            return True
        if ext == ".ply":
            return _try_read_point_cloud_ply(filename) is not None
        return False

    @classmethod
    def load_mesh(
        cls,
        filename: str,
        normalize: bool = False,
        normalize_by: str = "bbox",
        obj_scale: float = 2.0,
        preprocess: bool = True,
        device: torch.device | None = None,
    ) -> Mesh:
        """Load mesh from disk (.stl/.obj/.ply/.glb/.gltf).

        STL files are rotated -90 degrees about X during load.
        Scene graphs are flattened into a single mesh.

        Args:
            normalize: whether to call normalize() after load.
            normalize_by: "bbox" or "bsphere".
            obj_scale: target extent after normalization (passed as size).
            preprocess: whether to merge vertices and drop degenerate faces.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext not in (".stl", ".obj", ".ply", ".glb", ".gltf"):
            raise NotImplementedError(f"Unsupported mesh format: {ext}")
        if cls.is_point_cloud_file(filename):
            raise ValueError(
                f"{filename} is a point cloud; use load_rope_points() or pass the path to "
                "MeshFlowPipeline.run() instead of load_mesh()."
            )

        mesh = trimesh.load(filename, force="mesh", process=False, skip_materials=True)
        if ext == ".stl":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
        if isinstance(mesh, trimesh.Scene):
            parts = []
            for _, node in mesh.graph.to_flattened().items():
                name = node["geometry"]
                if name in mesh.geometry and isinstance(mesh.geometry[name], trimesh.Trimesh):
                    parts.append(mesh.geometry[name].apply_transform(node["transform"]))
            mesh = trimesh.util.concatenate(parts)

        m = cls(
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64),
            device=device,
        )
        if preprocess:
            m.preprocess()
        if normalize:
            m.normalize(normalize_by=normalize_by, size=obj_scale)
        return m

    @classmethod
    def load_rope_points(
        cls,
        filename: str,
        n_verts: int,
        normalize_by: str = "bsphere",
        obj_scale: float = 2.0,
        preprocess: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Load a mesh or point cloud and return fixed-size points for RoPE conditioning."""
        device = device or torch.device("cpu")
        ext = os.path.splitext(filename)[1].lower()

        if ext in POINT_CLOUD_EXTS:
            points = _read_point_cloud_file(filename).to(device)
        elif ext == ".ply":
            points = _try_read_point_cloud_ply(filename)
            if points is None:
                mesh = cls.load_mesh(
                    filename,
                    normalize=True,
                    normalize_by=normalize_by,
                    obj_scale=obj_scale,
                    preprocess=preprocess,
                    device=device,
                )
                surface_pts, _ = mesh.sample_surface(n_verts)
                return surface_pts
            points = points.to(device)
        else:
            mesh = cls.load_mesh(
                filename,
                normalize=True,
                normalize_by=normalize_by,
                obj_scale=obj_scale,
                preprocess=preprocess,
                device=device,
            )
            surface_pts, _ = mesh.sample_surface(n_verts)
            return surface_pts

        points = _normalize_points(points, normalize_by=normalize_by, size=obj_scale)
        return _resample_point_cloud(points, n_verts)

    def to_trimesh(self) -> trimesh.Trimesh:
        """Export to a host-side mesh object (detached from autograd)."""
        return trimesh.Trimesh(
            vertices=self.verts.detach().cpu().numpy(),
            faces=self.faces.detach().cpu().numpy(),
            process=False,
        )

    @property
    def v_nrm(self) -> torch.Tensor:
        """Per-vertex normals (area-weighted face normal sum), shape (N, 3)."""
        if self._v_nrm is None:
            v0, v1, v2 = self.verts[self.faces[:, 0]], self.verts[self.faces[:, 1]], self.verts[self.faces[:, 2]]
            fn = torch.linalg.cross(v1 - v0, v2 - v0)
            v_nrm = torch.zeros_like(self.verts)
            v_nrm.scatter_add_(0, self.faces[:, 0:1].expand(-1, 3), fn)
            v_nrm.scatter_add_(0, self.faces[:, 1:2].expand(-1, 3), fn)
            v_nrm.scatter_add_(0, self.faces[:, 2:3].expand(-1, 3), fn)
            v_nrm = torch.where(
                (v_nrm * v_nrm).sum(-1, keepdim=True) > 1e-20,
                v_nrm,
                torch.tensor([0.0, 0.0, 1.0], device=self.device),
            )
            self._v_nrm = F.normalize(v_nrm, dim=1)
        return self._v_nrm

    @property
    def edges(self) -> torch.Tensor:
        """Unique undirected edges, shape (E, 2)."""
        if self._edges is None:
            e = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]], dim=0)
            self._edges = torch.unique(torch.sort(e, dim=1)[0], dim=0)
        return self._edges

    def preprocess(self) -> Mesh:
        """Clean mesh in-place: merge vertices, remove degenerate/unreferenced geometry."""
        tm = self.to_trimesh()
        tm.process()
        tm.merge_vertices(merge_tex=True, merge_norm=True)
        tm.update_faces(tm.nondegenerate_faces())
        tm.update_faces(tm.unique_faces())
        tm.remove_unreferenced_vertices()
        self.verts = torch.from_numpy(tm.vertices).to(self.verts).float()
        self.faces = torch.from_numpy(tm.faces).to(self.faces).long()
        self._v_nrm = self._edges = None
        return self

    def normalize(self, normalize_by: str = "bsphere", size: float = 2.0) -> Mesh:
        """Center and scale vertices in-place to fit a canonical volume.

        Args:
            normalize_by: "bbox" (axis-aligned box) or "bsphere" (bounding sphere).
            size: target extent along the longest axis / diameter.
        """
        if normalize_by == "bbox":
            bbox_min, bbox_max = self.verts.min(0).values, self.verts.max(0).values
            self.verts = (self.verts - (bbox_min + bbox_max) / 2) * (size / (bbox_max - bbox_min).max())
        elif normalize_by == "bsphere":
            center = (self.verts.min(0).values + self.verts.max(0).values) / 2
            self.verts = (self.verts - center) * (
                size / (torch.linalg.norm(self.verts - center, dim=-1).max() * 2)
            )
        else:
            raise NotImplementedError(f"Invalid normalize_by: {normalize_by}")
        self._v_nrm = self._edges = None
        return self

    def rotate(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> Mesh:
        """Apply ZYX Euler rotation (degrees) and return a new Mesh."""
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        R = torch.tensor(
            [
                [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
                [cp * sr, cp * cr, -sp],
                [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return Mesh(self.verts @ R.T, self.faces, device=self.device)

    @torch.no_grad()
    def sample_surface(self, num_points: int, return_face_idx: bool = False):
        """Uniformly sample points on the mesh surface.

        Returns:
            points, face_normals; with return_face_idx=True also per-sample face indices.
        """
        tm = self.to_trimesh()
        points, face_idx = tm.sample(num_points, return_index=True)
        points = torch.from_numpy(points.astype(np.float32)).to(self.device)
        normals = torch.from_numpy(tm.face_normals[face_idx].astype(np.float32)).to(self.device)
        if return_face_idx:
            return points, normals, torch.from_numpy(face_idx).to(self.device, dtype=torch.long)
        return points, normals

    @torch.no_grad()
    def build_topology(
        self,
        topology_feats_type: Optional[List[str]] = None,
        n_verts: int = 8192,
        max_degree: int = 50,
    ) -> Dict[str, Any]:
        """Build padded topology tensors for MeshFlow VAE / inference.

        Runs preprocess + bsphere normalize, pads to ``n_verts`` with surface-sampled
        negative vertices (adaptive padding), and masks real mesh vertices via ``verts_mask``.

        Adjacency, degree, and neighbor features on padded slots are copied from the
        nearest real mesh vertex.

        Supported features: verts_normal, neighbor_points, degree, adjacency_matrix.

        Returns:
            Dict with padded_verts, verts_mask, verts, faces, num_verts, and requested features.
        """
        if topology_feats_type is None:
            topology_feats_type = ["verts_normal", "neighbor_points", "degree", "adjacency_matrix"]

        m = self.preprocess().normalize(size=2.0)
        num_verts = int(m.verts.shape[0])
        if num_verts > n_verts:
            raise ValueError(f"Mesh has {num_verts} verts > n_verts={n_verts}")
        if num_verts == 0:
            raise ValueError("Mesh has no vertices")

        device = m.device
        edges, v_nrm = m.edges, m.v_nrm
        n_negative_verts = n_verts - num_verts

        negative_verts = torch.zeros(0, 3, dtype=m.verts.dtype, device=device)
        negative_normals = torch.zeros(0, 3, dtype=v_nrm.dtype, device=device)
        nearest_verts_idx = None
        if n_negative_verts > 0:
            negative_verts, negative_normals, negative_face_idx = m.sample_surface(
                n_negative_verts, return_face_idx=True
            )
            face_vertices = m.faces[negative_face_idx]
            dist2 = (face_vertices - negative_verts[:, None, :]).pow(2).sum(dim=-1)
            nearest_choice = dist2.argmin(dim=1)
            nearest_verts_idx = face_vertices[
                torch.arange(n_negative_verts, device=device),
                nearest_choice,
            ]

        padded = torch.zeros(n_verts, 3, dtype=m.verts.dtype, device=device)
        padded[:num_verts] = m.verts
        if n_negative_verts > 0:
            padded[num_verts:] = negative_verts.to(dtype=m.verts.dtype, device=device)

        verts_mask = (torch.arange(n_verts, device=device) < num_verts).float().unsqueeze(-1)
        ret: Dict[str, Any] = {
            "padded_verts": padded,
            "verts_mask": verts_mask,
            "num_verts": num_verts,
            "num_faces": int(m.faces.shape[0]),
            "num_edges": int(edges.shape[0]),
            "verts": m.verts,
            "faces": m.faces,
        }

        for feat in topology_feats_type:
            if feat == "verts_normal":
                out = torch.zeros(n_verts, 3, dtype=v_nrm.dtype, device=device)
                out[:num_verts] = v_nrm
                if n_negative_verts > 0:
                    out[num_verts:] = negative_normals.to(dtype=v_nrm.dtype, device=device)
            elif feat == "adjacency_matrix":
                out = torch.zeros(n_verts, n_verts, dtype=torch.bool, device=device)
                out[edges[:, 0], edges[:, 1]] = True
                out[edges[:, 1], edges[:, 0]] = True
                if n_negative_verts > 0:
                    neg_slice = slice(num_verts, n_verts)
                    out[neg_slice, :] = out[nearest_verts_idx, :]
                    out[:, neg_slice] = out[:, nearest_verts_idx]
            elif feat == "degree":
                degrees = torch.bincount(edges.flatten(), minlength=num_verts)
                if (degrees > max_degree).any():
                    raise ValueError(f"Vertex degree exceeds max_degree ({max_degree}).")
                out = torch.zeros(n_verts, dtype=torch.long, device=device)
                out[:num_verts] = degrees
                if n_negative_verts > 0:
                    out[num_verts:] = out[nearest_verts_idx]
                out = out.unsqueeze(-1)
            elif feat == "neighbor_points":
                all_e = torch.cat([edges, edges.flip(1)], dim=0)
                src, dst = all_e[:, 0], all_e[:, 1]
                order = torch.argsort(src)
                src, dst = src[order], dst[order]
                if torch.bincount(src, minlength=num_verts).max() > max_degree:
                    raise ValueError(f"Vertex degree exceeds max_degree ({max_degree}).")
                first = torch.searchsorted(src, src)
                out = torch.zeros(n_verts, max_degree, 3, dtype=m.verts.dtype, device=device)
                out[src, torch.arange(len(src), device=device) - first] = m.verts[dst]
                if n_negative_verts > 0:
                    out[num_verts:] = out[nearest_verts_idx]
                out = out.view(n_verts, -1)
            else:
                raise ValueError(f"Unknown topology_feats_type: {feat}")
            ret[feat] = out
        return ret
