"""
Marching tetrahedra — guaranteed-manifold iso-surface extraction.

Fallback when the (smoother, faster) surface-nets pass detects a saddle
pinch. Each cubic cell is split into 6 tetrahedra sharing the *same* main
diagonal (Kuhn fan), provably consistent across neighbouring cells: both
cells split their shared face along the identical face-diagonal, so the
triangles emitted on shared faces match exactly → the result is watertight
and edge-manifold for **any** sign field. No lookup tables: cases are built
generically, orientation fixed against the local inside→outside direction.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# corner id = dx + 2·dy + 4·dz
CORNER_OFFSETS = np.array([[dx, dy, dz]
                           for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                          dtype=np.int64)
# Kuhn 6-tet fan around cube diagonal 0 → 7
TETS = np.array([
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
], dtype=np.int64)


def _lerp(pa: np.ndarray, fa: np.ndarray, pb: np.ndarray,
          fb: np.ndarray) -> np.ndarray:
    """Zero-crossing interpolation on edges (fa·fb < 0 → denom ≠ 0)."""
    t = fa / np.where(fa - fb != 0, fa - fb, 1e-30)
    return pa + t[..., None] * (pb - pa)


def marching_tets(sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (vertices, faces) with f < 0 inside. Vertices are in sample
    index units — the same convention as ``surface_nets``."""
    sdf = np.asarray(sdf, dtype=np.float64)
    nx, ny, nz = sdf.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    ns = nx * ny * nz

    ii, jj, kk = np.meshgrid(np.arange(cx), np.arange(cy), np.arange(cz),
                             indexing="ij")
    vals, poss, gids, negs = [], [], [], []
    for g in range(8):
        ox, oy, oz = CORNER_OFFSETS[g]
        vals.append(sdf[ox:ox + cx, oy:oy + cy, oz:oz + cz])
        poss.append(np.stack([ii + ox, jj + oy, kk + oz], axis=-1))
        gids.append((((ii + ox) * ny) + (jj + oy)) * nz + (kk + oz))
        negs.append(vals[-1] < 0)

    all_keys: List[np.ndarray] = []
    all_pos: List[np.ndarray] = []
    all_tris: List[np.ndarray] = []
    offset = 0

    def emit(keys: np.ndarray, pts: np.ndarray, d: np.ndarray) -> None:
        """keys/pts: (T,3) triangle corners; d: (T,3) inside→outside dir."""
        nonlocal offset
        if keys.shape[0] == 0:
            return
        nrm = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        flip = np.einsum("ij,ij->i", nrm, d) < 0
        if flip.any():
            keys = np.where(flip[:, None], keys[:, [0, 2, 1]], keys)
            pts = np.where(flip[:, None, None], pts[:, [0, 2, 1], :], pts)
        t = keys.shape[0]
        all_keys.append(keys.reshape(-1))
        all_pos.append(pts.reshape(-1, 3).astype(np.float64))
        all_tris.append(np.arange(offset, offset + t * 3,
                                  dtype=np.int32).reshape(t, 3))
        offset += t * 3

    for tet in TETS:
        V = [vals[g] for g in tet]
        P = [poss[g] for g in tet]
        G = [gids[g] for g in tet]
        NEG = [negs[g] for g in tet]
        bitmask = (NEG[0].astype(np.int16) | (NEG[1].astype(np.int16) << 1) |
                   (NEG[2].astype(np.int16) << 2) | (NEG[3].astype(np.int16) << 3))
        for case in range(1, 15):
            mask = bitmask == case
            if not mask.any():
                continue
            inside = sorted(c for c in range(4) if (case >> c) & 1)
            outside = sorted(c for c in range(4) if not ((case >> c) & 1))
            Vm = [vc[mask] for vc in V]
            Pm = [pp[mask] for pp in P]
            Gm = [gg[mask] for gg in G]

            def edge_key(a: int, b: int) -> np.ndarray:
                ga, gb = Gm[a].astype(np.int64), Gm[b].astype(np.int64)
                return np.minimum(ga, gb) * ns + np.maximum(ga, gb)

            def edge_pt(a: int, b: int) -> np.ndarray:
                return _lerp(Pm[a], Vm[a], Pm[b], Vm[b])

            if len(inside) == 1:
                i = inside[0]
                keys = np.stack([edge_key(i, o) for o in outside], axis=1)
                pts = np.stack([edge_pt(i, o) for o in outside], axis=1)
                d = pts.mean(axis=1) - Pm[i]
                emit(keys, pts, d)
            elif len(outside) == 1:
                o = outside[0]
                keys = np.stack([edge_key(i, o) for i in inside], axis=1)
                pts = np.stack([edge_pt(i, o) for i in inside], axis=1)
                d = Pm[o] - pts.mean(axis=1)
                emit(keys, pts, d)
            else:                              # 2-in / 2-out → quad (2 tris)
                i1, i2 = inside
                o1, o2 = outside
                k4 = np.stack([edge_key(i1, o1), edge_key(i1, o2),
                               edge_key(i2, o2), edge_key(i2, o1)], axis=1)
                p4 = np.stack([edge_pt(i1, o1), edge_pt(i1, o2),
                               edge_pt(i2, o2), edge_pt(i2, o1)], axis=1)
                d = 0.5 * (Pm[o1] + Pm[o2]) - 0.5 * (Pm[i1] + Pm[i2])
                keys2 = np.concatenate([k4[:, [0, 1, 2]], k4[:, [0, 2, 3]]], axis=0)
                pts2 = np.concatenate([p4[:, [0, 1, 2]], p4[:, [0, 2, 3]]], axis=0)
                d2 = np.concatenate([d, d], axis=0)
                emit(keys2, pts2, d2)

    if not all_keys:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    keys_flat = np.concatenate(all_keys)
    pos_flat = np.concatenate(all_pos, axis=0)
    tris_flat = np.concatenate(all_tris, axis=0)
    _, first, inverse = np.unique(keys_flat, return_index=True,
                                  return_inverse=True)
    verts = pos_flat[first]
    faces = inverse[tris_flat].astype(np.int32)
    return verts, faces
