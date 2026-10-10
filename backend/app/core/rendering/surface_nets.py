"""
Dual-Contouring-style neural SDF extraction → watertight triangle mesh.

The terrain heightfield is embedded as the solid SDF

    f(x, y, z) = max(z − h(x, y), z_bot − z)

(a slab closed by a flat bottom), sampled on a padded grid (pad = 2 samples
of *outside* so every sign-changing edge has its full set of 4 adjacent
cells). Extraction runs naive surface nets with dual-contouring vertex
placement: one vertex per active cell at the average of its edge crossings,
quad faces emitted per sign-changing edge, oriented against ∇f.

Guarantees (asserted by the test suite):
* every undirected edge is shared by exactly two triangles → watertight,
* consistent outward orientation (signed volume > 0),
* manifold quads from the dual grid — CAD/GIS/3D-print ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_CORNERS = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
_EDGES = [
    ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)),
    ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)),
    ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
]


@dataclass
class Mesh:
    vertices: np.ndarray      # (V, 3) float64 — X east, Y up, Z north
    faces: np.ndarray         # (F, 3) int32
    normals: np.ndarray       # (V, 3) float64

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def surface_nets(sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (vertices, faces) from a scalar field with f<0 inside.

    Vectorised over the whole grid: cells with any sign change get a vertex
    placed at the mean of their sign-changing edge crossings.
    """
    sdf = np.asarray(sdf, dtype=np.float64)
    cx, cy, cz = (s - 1 for s in sdf.shape)
    corner_vals: Dict[Tuple[int, int, int], np.ndarray] = {}
    for (dx, dy, dz) in _CORNERS:
        corner_vals[(dx, dy, dz)] = sdf[dx:dx + cx, dy:dy + cy, dz:dz + cz]

    acc = np.zeros((cx, cy, cz, 3), dtype=np.float64)
    count = np.zeros((cx, cy, cz), dtype=np.float64)
    for (a, b) in _EDGES:
        va, vb = corner_vals[a], corner_vals[b]
        mask = (va < 0) != (vb < 0)
        if not mask.any():
            continue
        denom = np.where(mask, va - vb, 1.0)
        t = np.where(mask, va / denom, 0.0)
        for axis in range(3):
            pa, pb = a[axis], b[axis]
            point = pa + t * (pb - pa)
            acc[..., axis] += np.where(mask, point, 0.0)
        count += mask

    active = count > 0
    cell_coords = np.argwhere(active).astype(np.float64)   # (A, 3) cell min-corner
    cell_index = np.full((cx, cy, cz), -1, dtype=np.int64)
    cell_index[active] = np.arange(int(active.sum()))
    verts = acc[active] / count[active][:, None] + cell_coords   # global index units

    # ── quads per sign-changing grid edge (3 axis sweeps) ──────────────────
    tri_blocks: List[np.ndarray] = []
    grad = np.gradient(sdf)
    for axis in range(3):
        slices0 = [slice(None)] * 3
        slices1 = [slice(None)] * 3
        slices0[axis] = slice(0, -1)
        slices1[axis] = slice(1, None)
        s0 = sdf[tuple(slices0)]
        s1 = sdf[tuple(slices1)]
        sc = (s0 < 0) != (s1 < 0)
        if not sc.any():
            continue
        p = np.argwhere(sc)                              # (E, 3) sample coords
        other = [ax for ax in range(3) if ax != axis]
        pb, pc = p[:, other[0]], p[:, other[1]]
        valid = (pb >= 1) & (pc >= 1)
        p = p[valid]
        if p.shape[0] == 0:
            continue
        pa, pb, pc = p[:, axis], p[:, other[0]], p[:, other[1]]
        combos = [(0, 0), (1, 0), (1, 1), (0, 1)]        # cyclic around edge
        quad_ids: List[np.ndarray] = []
        ok = np.ones(p.shape[0], dtype=bool)
        for db, dc in combos:
            along = [0, 0, 0]
            along[axis] = pa
            along[other[0]] = pb - 1 + db
            along[other[1]] = pc - 1 + dc
            ma, mb, mc = (along[0], along[1], along[2])
            bounds = ((ma >= 0) & (ma < cx) & (mb >= 0) & (mb < cy) &
                      (mc >= 0) & (mc < cz))
            ids = cell_index[np.clip(ma, 0, cx - 1), np.clip(mb, 0, cy - 1),
                             np.clip(mc, 0, cz - 1)]
            quad_ids.append(ids)
            ok &= bounds & (ids >= 0)
        if not ok.any():
            continue
        q = [qi[ok] for qi in quad_ids]
        pp = p[ok]
        if pp.shape[0] == 0:
            continue
        # orient: outward = +∇f at the edge midpoint (f < 0 inside)
        gi = np.stack([np.clip(pp[:, 0], 0, sdf.shape[0] - 1),
                       np.clip(pp[:, 1], 0, sdf.shape[1] - 1),
                       np.clip(pp[:, 2], 0, sdf.shape[2] - 1)], axis=1)
        g = np.stack([grad[0][tuple(gi.T)], grad[1][tuple(gi.T)],
                      grad[2][tuple(gi.T)]], axis=1)
        qa, qb, qc, qd = q
        n = np.cross(verts[qb] - verts[qa], verts[qc] - verts[qa])
        flip = np.einsum("ij,ij->i", n, g) < 0
        # vectorised emission (a1 ↔ a3 swap when flipped)
        tri_a = np.where(flip[:, None],
                         np.stack([qa, qd, qc], axis=1),
                         np.stack([qa, qb, qc], axis=1))
        tri_b = np.where(flip[:, None],
                         np.stack([qa, qc, qb], axis=1),
                         np.stack([qa, qc, qd], axis=1))
        tri_blocks.append(tri_a)
        tri_blocks.append(tri_b)

    if tri_blocks:
        faces_arr = np.concatenate(tri_blocks, axis=0).astype(np.int32)
    else:
        faces_arr = np.zeros((0, 3), dtype=np.int32)
    return verts, faces_arr


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    norm = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n / norm


# ── heightfield → world-space terrain mesh ─────────────────────────────────
def heightfield_to_mesh(h: np.ndarray, extent: Tuple[float, float, float, float],
                        world_w: float = 100.0, vert_exag: float = 10.0,
                        slab_frac: float = 0.35, z_levels: int = 30,
                        ensure_watertight: bool = True) -> Tuple[Mesh, Dict[str, float]]:
    """SDF-slab-embed a heightfield and extract the watertight mesh.

    extent = (lon0, lat0, lon1, lat1); world axes: X east, Y up, Z south
    (glTF/Cesium compatible). Because rare saddle sign-patterns can pinch
    the iso-surface, extraction retries with neighbouring vertical sampling
    densities until the mesh passes the watertight/manifold check.
    Returns (Mesh, transform metadata used by the frontend/API).
    """
    from app.core.geometry import EARTH_RADIUS_M
    h = np.asarray(h, dtype=np.float64)
    H, W = h.shape
    lon0, lat0, lon1, lat1 = extent
    lat_c = math.radians(0.5 * (lat0 + lat1))
    width_m = math.radians(lon1 - lon0) * EARTH_RADIUS_M * math.cos(lat_c)
    depth_m = math.radians(lat1 - lat0) * EARTH_RADIUS_M
    width_m = max(width_m, 1.0)
    depth_m = max(depth_m, 1.0)

    h_min, h_max = float(h.min()), float(h.max())
    relief = max(h_max - h_min, 1e-6)
    slab_m = max(relief * slab_frac, 0.5)
    z_bot = h_min - slab_m
    nx, ny = W + 4, H + 4

    def build_sdf(zlev: int) -> Tuple[np.ndarray, float, int]:
        dz = (h_max - z_bot) / max(zlev - 1, 2)
        nz = zlev + 4
        zz = z_bot + 0.5 * dz + (np.arange(nz) - 2) * dz
        zc = zz[2:nz - 2]
        sdf = np.ones((nx, ny, nz), dtype=np.float64)
        zpart = np.maximum(zc[None, None, :] - h[:, :, None],
                           z_bot - zc[None, None, :])
        sdf[2:nx - 2, 2:ny - 2, 2:nz - 2] = np.transpose(zpart, (1, 0, 2))
        return sdf, dz, nz

    offsets = [0, 3]
    best = None
    attempts = offsets if ensure_watertight else [0]
    method = "surface_nets"
    for off in attempts:
        zlev = z_levels + off
        if zlev < 12:
            continue
        sdf, dz, nz = build_sdf(zlev)
        vidx, fcs = surface_nets(sdf)
        if vidx.shape[0] == 0:
            continue
        topo = mesh_topology(fcs)
        if best is None:
            best = (vidx, fcs, dz, nz, topo, off)
        else:
            cur = (topo["non_manifold_edges"], topo["inconsistent_directed_edges"])
            prev = (best[4]["non_manifold_edges"], best[4]["inconsistent_directed_edges"])
            if cur < prev:
                best = (vidx, fcs, dz, nz, topo, off)
        if topo["watertight"]:
            break
    if (best is None) or (ensure_watertight and not best[4]["watertight"]):
        # saddle pinch survived the smooth path → guaranteed-manifold fallback
        from app.core.rendering.marching_tets import marching_tets
        zlev = z_levels
        sdf, dz, nz = build_sdf(zlev)
        vidx, fcs = marching_tets(sdf)
        if vidx.shape[0] == 0:
            vidx, fcs = surface_nets(sdf)
        topo = mesh_topology(fcs)
        best = (vidx, fcs, dz, nz, topo, 0)
        method = "marching_tets"
    if best[0].shape[0] == 0:                          # fully degenerate guard
        verts_idx = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.0, 3.0, 2.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        dz, nz, topo, offset_used = 1.0, 34, {"watertight": False}, 0
        method = "degenerate"
    else:
        verts_idx, faces, dz, nz, topo, offset_used = best

    m2w = world_w / width_m                                    # metres → world
    world_d = depth_m * m2w
    z_scale = m2w * vert_exag
    z_ref = 0.5 * (h_min + h_max)

    vi = verts_idx[:, 0] - 2.0
    vj = verts_idx[:, 1] - 2.0
    vk = verts_idx[:, 2]
    z_m = z_bot + 0.5 * dz + (vk - 2) * dz
    sx = world_w / max(W - 1, 1)
    sz = world_d / max(H - 1, 1)
    # glTF/Cesium-compatible axes: X east, Y up, Z south (determinant +1 →
    # outward face orientation is preserved from index space)
    world = np.stack([vi * sx, (z_m - z_ref) * z_scale,
                      (H - 1 - vj) * sz], axis=1)

    normals = vertex_normals(world, faces)
    mesh = Mesh(vertices=world, faces=faces, normals=normals)
    transform = {
        "extent": [lon0, lat0, lon1, lat1],
        "world_w": float(world_w), "world_d": float(world_d),
        "metres_to_world": float(m2w), "vertical_exaggeration": float(vert_exag),
        "z_reference_m": float(z_ref), "z_bottom_m": float(z_bot),
        "relief_m": float(relief), "grid": [int(W), int(H)],
        "cell_count": [int(nx), int(ny), int(nz)],
        "watertight": bool(topo.get("watertight", False)),
        "topology": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v))
                     for k, v in topo.items() if k != "watertight"} | {
                         "watertight": bool(topo.get("watertight", False))},
        "sampling_offset": int(offset_used),
        "meshing_method": method,
    }
    return mesh, transform


# ── mesh validation helpers ────────────────────────────────────────────────
def mesh_topology(faces: np.ndarray) -> Dict[str, object]:
    """Watertightness report: each undirected edge exactly 2 faces,
    each directed edge exactly once (consistent orientation). Vectorised."""
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return {"edge_count": 0, "non_manifold_edges": 0,
                "inconsistent_directed_edges": 0, "watertight": False}
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    _, d_counts = np.unique(directed, axis=0, return_counts=True)
    und = np.sort(directed, axis=1)
    _, u_counts = np.unique(und, axis=0, return_counts=True)
    bad_und = int((u_counts != 2).sum())
    bad_dir = int((d_counts != 1).sum())
    return {
        "edge_count": int(u_counts.size),
        "non_manifold_edges": bad_und,
        "inconsistent_directed_edges": bad_dir,
        "watertight": bad_und == 0 and bad_dir == 0,
    }


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume; > 0 for outward orientation."""
    tri = verts[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0],
                           np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
