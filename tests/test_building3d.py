"""Tests for the Building3D weak-label loader (ADR-0003).

The tests run on synthetic wireframes + an integration test that loads one
real sample from ``data/Building3D/Entry-level/train/`` if available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roofseg.data.building3d import (
    assign_labels_to_points,
    find_planar_faces,
    load_scene,
    load_wireframe,
)


def _square_with_diagonal():
    # Unit square (0,0)-(1,0)-(1,1)-(0,1) split by diagonal (0,0)-(1,1).
    # Two triangular faces sharing one edge.
    verts_xy = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    return verts_xy, edges


def test_find_planar_faces_square_with_diagonal_yields_two_triangles():
    verts_xy, edges = _square_with_diagonal()
    faces = find_planar_faces(verts_xy, edges)
    assert len(faces) == 2
    sizes = sorted(len(f) for f in faces)
    assert sizes == [3, 3]


def test_find_planar_faces_two_disjoint_triangles():
    # Triangle A around origin, triangle B far away — disjoint components.
    verts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [10.5, 11.0],
    ])
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    faces = find_planar_faces(verts, edges)
    assert len(faces) == 2


def test_find_planar_faces_no_cycles_returns_empty():
    # Just a chain — no closed cycles, no faces.
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edges = [(0, 1), (1, 2)]
    assert find_planar_faces(verts, edges) == []


def test_load_wireframe_parses_v_and_l(tmp_path: Path):
    obj = tmp_path / "test.obj"
    obj.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "l 1 2\n"
        "l 2 3\n"
        "l 3 4\n"
        "l 4 1\n"
    )
    wire = load_wireframe(obj)
    assert wire.vertices.shape == (4, 3)
    assert len(wire.edges) == 4
    assert wire.edges[0] == (0, 1)  # 1-indexed in OBJ -> 0-indexed


def test_assign_labels_single_square_face():
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = [[0, 1, 2, 3]]
    # Three points: inside, on edge, far outside.
    points = np.array([
        [0.5, 0.5, 0.0],   # inside
        [2.0, 2.0, 0.0],   # outside
        [0.5, 0.5, 0.05],  # close to plane, inside polygon -> still labelled
    ])
    labels, n_amb = assign_labels_to_points(points, verts, faces, ambiguity_margin=0.3)
    assert labels[0] == 0
    assert labels[1] == -1
    assert labels[2] == 0
    assert n_amb == 0


def test_assign_labels_flags_ambiguous_between_close_planes():
    # Two coplanar squares (same z=0) overlapping in their middle region.
    # A point in the overlap should be ambiguous since both planes are equally
    # close.
    verts = np.array([
        # Square A: x in [0,1]
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        # Square B: x in [0.5, 1.5]
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    points = np.array([
        [0.25, 0.5, 0.0],  # only in A
        [1.25, 0.5, 0.0],  # only in B
        [0.75, 0.5, 0.0],  # in both -> ambiguous
    ])
    labels, n_amb = assign_labels_to_points(points, verts, faces, ambiguity_margin=0.3)
    assert labels[0] == 0
    assert labels[1] == 1
    assert labels[2] == -1
    assert n_amb == 1


SAMPLE_XYZ = Path(
    "/Users/hdong/Projects/LidarRoofSegNet/data/Building3D/Entry-level/train/xyz/10.xyz"
)
SAMPLE_OBJ = Path(
    "/Users/hdong/Projects/LidarRoofSegNet/data/Building3D/Entry-level/train/wireframe/10.obj"
)


@pytest.mark.skipif(
    not (SAMPLE_XYZ.exists() and SAMPLE_OBJ.exists()),
    reason="real Building3D sample not available",
)
def test_load_scene_real_sample():
    scene = load_scene("10", SAMPLE_XYZ, SAMPLE_OBJ, ambiguity_margin=0.3)
    assert scene.points.shape[1] == 3
    assert scene.labels.shape == (scene.points.shape[0],)
    # Sample 10 has two clearly separated cycles -> two faces.
    assert scene.n_faces >= 2
    # Most points should be assigned (not all ambiguous / unlabelled).
    assigned = (scene.labels >= 0).sum()
    assert assigned > 0
