"""Building3D loader and weak-label projection (ADR-0003).

Building3D stores per-building point clouds (``.xyz``, space-separated XYZ +
extra columns) and wireframes (``.obj`` with only ``v`` vertices and ``l``
line segments — no ``f`` face directives). We derive per-point weak instance
labels by:

1. Parsing the wireframe into a graph.
2. Extracting planar faces from the XY-projected edge graph via the
   rotation-system algorithm (each undirected edge contributes two directed
   half-edges; walking the next-CW half-edge at each vertex traces one face).
3. Fitting a plane to each face's 3D vertices and assigning each point to the
   face whose plane is closest *and* whose polygon contains the projected point.

The labels are **weak** — points within ``ambiguity_margin`` of two faces'
planes are flagged as ambiguous and excluded from downstream metrics. ADR-0003
requires this audit accompany every external-validation eval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WireFrame:
    vertices: np.ndarray  # (V, 3)
    edges: list[tuple[int, int]]


@dataclass
class Building3DScene:
    scene_id: str
    points: np.ndarray  # (N, 3) XYZ
    points_extra: np.ndarray  # (N, F) RGB + intensity etc.
    labels: np.ndarray  # (N,) face index in [0..K-1] or -1 (ambiguous / outside)
    n_faces: int
    n_ambiguous: int

    @property
    def ambiguity_rate(self) -> float:
        return float(self.n_ambiguous) / max(self.points.shape[0], 1)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def load_wireframe(path: str | Path) -> WireFrame:
    """Parse a Building3D ``.obj`` wireframe."""
    vertices: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tag, *rest = line.split()
            if tag == "v" and len(rest) >= 3:
                vertices.append([float(x) for x in rest[:3]])
            elif tag == "l" and len(rest) >= 2:
                # OBJ vertex indices are 1-based.
                u = int(rest[0]) - 1
                v = int(rest[1]) - 1
                if u != v:
                    edges.append((u, v))
    return WireFrame(
        vertices=np.array(vertices, dtype=np.float64),
        edges=edges,
    )


def load_xyz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Building3D ``.xyz`` file.

    First 3 columns are XYZ; remaining columns (RGB + intensity etc.) are
    returned separately so the loader is agnostic to the column count.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3]
    extra = data[:, 3:] if data.shape[1] > 3 else np.zeros((data.shape[0], 0))
    return xyz, extra


# --------------------------------------------------------------------------
# Planar face extraction
# --------------------------------------------------------------------------


def _signed_polygon_area(poly_xy: np.ndarray) -> float:
    n = poly_xy.shape[0]
    a = 0.0
    for i in range(n):
        x0, y0 = poly_xy[i]
        x1, y1 = poly_xy[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def find_planar_faces(
    vertices_xy: np.ndarray, edges: list[tuple[int, int]]
) -> list[list[int]]:
    """Find bounded planar faces of an undirected graph embedded in 2D.

    Algorithm: for each vertex, sort incident edges by angle (CCW). For each
    directed half-edge ``u -> v``, the next half-edge in the same face is
    ``v -> w`` where ``w`` is the **CW** neighbour of ``u`` in ``v``'s
    rotation system. Walking until we return to the start traces one face.
    Faces with negative signed area are the outer (unbounded) face and are
    discarded.

    Args:
        vertices_xy: ``(V, 2)`` 2D coordinates.
        edges: undirected ``(u, v)`` pairs.

    Returns:
        List of faces; each face is a list of vertex indices in CCW order.
    """
    if not edges:
        return []

    incident: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for u, v in edges:
        du = vertices_xy[v] - vertices_xy[u]
        incident[u].append((float(np.arctan2(du[1], du[0])), v))
        dv = vertices_xy[u] - vertices_xy[v]
        incident[v].append((float(np.arctan2(dv[1], dv[0])), u))
    neighbour_order: dict[int, list[int]] = {}
    for u, lst in incident.items():
        lst.sort()
        neighbour_order[u] = [n for _, n in lst]

    def next_half_edge(u: int, v: int) -> tuple[int, int]:
        nbrs = neighbour_order[v]
        i = nbrs.index(u)
        return (v, nbrs[(i - 1) % len(nbrs)])

    visited: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for u, v in edges:
        for start in ((u, v), (v, u)):
            if start in visited:
                continue
            face: list[int] = []
            e = start
            # Walk forward until we close the loop (or hit a visited edge).
            for _ in range(len(edges) * 2 + 4):  # safety bound
                if e in visited:
                    break
                visited.add(e)
                face.append(e[0])
                e = next_half_edge(*e)
                if e == start:
                    break
            if len(face) < 3:
                continue
            area = _signed_polygon_area(vertices_xy[face])
            if area > 0:
                faces.append(face)
    return faces


# --------------------------------------------------------------------------
# Per-point label assignment (weak)
# --------------------------------------------------------------------------


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm > 0:
        normal = normal / norm
    offset = float(normal @ centroid)
    return normal, offset


def _point_in_polygon_2d(px: float, py: float, polygon: np.ndarray) -> bool:
    n = polygon.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        x_i, y_i = polygon[i]
        x_j, y_j = polygon[j]
        if ((y_i > py) != (y_j > py)) and (
            px < (x_j - x_i) * (py - y_i) / (y_j - y_i + 1e-30) + x_i
        ):
            inside = not inside
        j = i
    return inside


def assign_labels_to_points(
    points_xyz: np.ndarray,
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    ambiguity_margin: float = 0.3,
) -> tuple[np.ndarray, int]:
    """Assign per-point face-index labels (weak labels).

    For each point, compute distance to each face's plane and check whether
    its projection onto the plane lies inside the face polygon. Among the
    faces whose polygon contains the projected point, the point is assigned
    to the closest-plane face; if the next-closest containing face is within
    ``ambiguity_margin``, the point is labelled ``-1`` (ambiguous).

    Args:
        points_xyz: ``(N, 3)`` input points.
        vertices: ``(V, 3)`` wireframe vertices.
        faces: list of face vertex-index lists (CCW).
        ambiguity_margin: distance threshold (same units as ``points_xyz``)
            below which a second face's plane is considered competing.

    Returns:
        ``(labels (N,), n_ambiguous)``. ``labels`` are face indices or ``-1``.
    """
    n = points_xyz.shape[0]
    if not faces:
        return np.full(n, -1, dtype=np.int64), 0

    cached = []
    for face in faces:
        face_verts = vertices[face]
        normal, offset = _fit_plane(face_verts)
        # In-plane 2D basis, robust to vertical normals.
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(normal @ helper)) > 0.95:
            helper = np.array([1.0, 0.0, 0.0])
        u_axis = np.cross(normal, helper)
        u_axis = u_axis / max(float(np.linalg.norm(u_axis)), 1e-12)
        v_axis = np.cross(normal, u_axis)
        centroid = face_verts.mean(axis=0)
        poly_2d = np.column_stack(
            [
                (face_verts - centroid) @ u_axis,
                (face_verts - centroid) @ v_axis,
            ]
        )
        cached.append(
            (normal, offset, u_axis, v_axis, centroid, poly_2d)
        )

    labels = np.full(n, -1, dtype=np.int64)
    n_ambiguous = 0
    for i in range(n):
        p = points_xyz[i]
        candidates: list[tuple[float, int]] = []
        for j, (normal, offset, u_axis, v_axis, centroid, poly_2d) in enumerate(cached):
            dist = abs(float(normal @ p) - offset)
            p_rel = p - centroid
            u = float(p_rel @ u_axis)
            v = float(p_rel @ v_axis)
            if _point_in_polygon_2d(u, v, poly_2d):
                candidates.append((dist, j))
        if not candidates:
            continue
        candidates.sort()
        if len(candidates) == 1:
            labels[i] = candidates[0][1]
        elif candidates[1][0] - candidates[0][0] < ambiguity_margin:
            n_ambiguous += 1  # labels[i] stays -1
        else:
            labels[i] = candidates[0][1]
    return labels, n_ambiguous


# --------------------------------------------------------------------------
# High-level scene loader
# --------------------------------------------------------------------------


def load_scene(
    scene_id: str,
    xyz_path: str | Path,
    wireframe_path: str | Path,
    *,
    ambiguity_margin: float = 0.3,
) -> Building3DScene:
    """Load one Building3D scene and produce weak per-point instance labels."""
    points_xyz, points_extra = load_xyz(xyz_path)
    wire = load_wireframe(wireframe_path)
    faces = find_planar_faces(wire.vertices[:, :2], wire.edges)
    labels, n_ambiguous = assign_labels_to_points(
        points_xyz,
        wire.vertices,
        faces,
        ambiguity_margin=ambiguity_margin,
    )
    return Building3DScene(
        scene_id=scene_id,
        points=points_xyz,
        points_extra=points_extra,
        labels=labels,
        n_faces=len(faces),
        n_ambiguous=n_ambiguous,
    )
