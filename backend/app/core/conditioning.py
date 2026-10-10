"""
Conditioning rasteriser — grounded spatial context → physical bound grids.

Input : the RRF-fused retrieval (nodes with geometry + physical props)
Output: a multi-channel coarse grid the latent engine is conditioned on

    channels: elevation | relief | hardness | rainfall | uplift |
              roughness | influence-weight

The elevation channel is further decomposed through spherical harmonics at
degree ≤ SH_DEGREE: the low-degree Y_l^m component is the *planetary global
shape* (projection-distortion free), while local detail is supplied later by
the DCT latent diffusion — a clean split between global curvature and local
geomorphology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.core.latent.spherical_harmonics import (encode_directions,
                                                 real_basis_count)

CHANNELS = ("elevation", "relief", "hardness", "rainfall", "uplift",
            "roughness", "weight")

DEFAULT_ATTRS: Dict[str, float] = {
    "elevation_m": 1800.0, "relief_m": 1400.0, "hardness": 5.0,
    "rainfall_mm": 520.0, "uplift_mm_yr": 4.0, "roughness": 0.55,
    "vp_ms": 5400.0, "permeability_mD": 180.0,
}


@dataclass
class Conditioning:
    attrs: Dict[str, float]
    extent: Tuple[float, float, float, float]        # lon0, lat0, lon1, lat1
    grid: np.ndarray                                 # (C, H, W)
    sh_coeffs: np.ndarray                            # (K,)
    sh_degree: int
    influences: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def channels(self) -> Dict[str, np.ndarray]:
        return {name: self.grid[i] for i, name in enumerate(CHANNELS)}

    def meta(self) -> Dict[str, Any]:
        return {
            "attrs": self.attrs, "extent": list(self.extent),
            "shape": list(self.grid.shape), "channels": list(CHANNELS),
            "sh_degree": self.sh_degree,
            "sh_coeffs": [round(float(c), 6) for c in self.sh_coeffs],
            "influences": len(self.influences),
        }


# ── geometry → distance fields (planar degrees, vectorised) ───────────────
def _distance_field(geom: Dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    X, Y = xs, ys
    t = geom["type"]
    if t == "Point":
        cx, cy = geom["coordinates"][:2]
        return np.hypot(X - cx, Y - cy)
    if t == "LineString":
        pts = geom["coordinates"]
        best = np.full(X.shape, np.inf, dtype=np.float64)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy or 1e-12
            tt = np.clip(((X - x1) * dx + (Y - y1) * dy) / L2, 0.0, 1.0)
            best = np.minimum(best, np.hypot(X - (x1 + tt * dx), Y - (y1 + tt * dy)))
        return best
    # Polygon: 0 inside, else distance to ring
    c = geom["coordinates"]
    ring = c if c and isinstance(c[0][0], (int, float)) else c[0]
    inside = np.zeros(X.shape, dtype=bool)
    best = np.full(X.shape, np.inf, dtype=np.float64)
    n = len(ring) - 1
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        cond = ((y1 > Y) != (y2 > Y))                # ray casting (vectorised)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (Y - y1) / np.where(y2 - y1 != 0, y2 - y1, 1e-12) + x1
        inside ^= cond & (X < xin)
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy or 1e-12
        tt = np.clip(((X - x1) * dx + (Y - y1) * dy) / L2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(X - (x1 + tt * dx), Y - (y1 + tt * dy)))
    return np.where(inside, 0.0, best)


def _influence_radius(geom: Dict[str, Any]) -> float:
    t = geom["type"]
    if t == "Point":
        return 0.45
    c = geom["coordinates"]
    pts = c if (t == "LineString" or (c and isinstance(c[0][0], (int, float)))) else c[0]
    arr = np.asarray(pts, dtype=np.float64)
    diag = float(np.hypot(arr[:, 0].max() - arr[:, 0].min(),
                          arr[:, 1].max() - arr[:, 1].min()))
    return max(0.18, diag * 0.55)


# ── main entry ─────────────────────────────────────────────────────────────
def build_conditioning(fused: List[Dict[str, Any]],
                       resolution: int = 32,
                       fallback_extent: Optional[Tuple[float, float, float, float]] = None
                       ) -> Conditioning:
    # 1) weighted scalar attributes from retrieved physical nodes
    attrs = {k: 0.0 for k in DEFAULT_ATTRS}
    wsum = 0.0
    influences: List[Dict[str, Any]] = []
    for item in fused:
        props = item.get("props") or {}
        has_phys = any(k in props for k in ("elevation_m", "relief_m", "thickness_m", "depth_m"))
        w = float(item.get("score", 0.0))
        if not has_phys:
            w *= 0.15                       # reports contribute weak priors only
        if item.get("geom"):
            influences.append({"geom": item["geom"], "weight": max(w, 1e-4),
                               "kind": item.get("kind", "unknown"), "id": item.get("id")})
        if has_phys:
            wsum += w
            for key in attrs:
                v = props.get(key)
                if v is None and key == "permeability_mD":
                    v = props.get("perm_mD", props.get("transmissivity_mDm"))
                if v is not None:
                    attrs[key] += w * float(v)
    if wsum > 0:
        attrs = {k: (v / wsum if v else DEFAULT_ATTRS[k]) for k, v in attrs.items()}
    else:
        attrs = dict(DEFAULT_ATTRS)
    for k, v in DEFAULT_ATTRS.items():
        if not attrs.get(k):
            attrs[k] = v

    # 2) spatial extent from retrieved footprints
    boxes = [item["bbox"] for item in fused if item.get("bbox")]
    if boxes:
        extent = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                  max(b[2] for b in boxes), max(b[3] for b in boxes))
    else:
        extent = fallback_extent or (71.0, 32.2, 77.0, 36.2)
    lon0, lat0, lon1, lat1 = extent
    pad_x = max((lon1 - lon0) * 0.08, 0.05)
    pad_y = max((lat1 - lat0) * 0.08, 0.05)
    extent = (lon0 - pad_x, lat0 - pad_y, lon1 + pad_x, lat1 + pad_y)

    # 3) rasterise channels
    H = W = int(resolution)
    xs = np.linspace(extent[0], extent[2], W)
    ys = np.linspace(extent[1], extent[3], H)
    X, Y = np.meshgrid(xs, ys)

    phys_keys = ("elevation_m", "relief_m", "hardness", "rainfall_mm",
                 "uplift_mm_yr", "roughness")
    acc = {k: np.zeros((H, W)) for k in phys_keys}
    wacc = np.zeros((H, W))
    props_by_id = {item.get("id"): (item.get("props") or {}) for item in fused}

    for inf in influences:
        d = _distance_field(inf["geom"], X, Y)
        r = _influence_radius(inf["geom"])
        w = inf["weight"] * np.exp(-((d / r) ** 2))
        props = props_by_id.get(inf["id"], {})
        wacc += w
        for k in phys_keys:
            v = props.get(k)
            if v is None and k == "permeability_mD":
                v = props.get("perm_mD")
            if v is not None:
                acc[k] += w * float(v)

    smooth = wacc > 1e-9
    weight_ch = np.clip(wacc / max(float(wacc.max()), 1e-9), 0.0, 1.0)
    elev_local = np.where(smooth, acc["elevation_m"] / np.maximum(wacc, 1e-9),
                          attrs["elevation_m"])
    elev = elev_local * weight_ch + attrs["elevation_m"] * (1.0 - weight_ch)

    channels = [elev]
    for k in phys_keys[1:]:
        local = np.where(smooth, acc[k] / np.maximum(wacc, 1e-9), attrs[k])
        channels.append(local * weight_ch + attrs[k] * (1.0 - weight_ch))
    channels.append(weight_ch)
    grid = np.stack(channels).astype(np.float64)

    # 4) spherical-harmonic global decomposition of the elevation channel
    degree = settings.sh_degree
    lon_c = 0.5 * (extent[0] + extent[2])
    lat_c = 0.5 * (extent[1] + extent[3])
    dlon = np.radians(X - lon_c)
    dlat = np.radians(Y - lat_c)
    dirs = np.stack([np.cos(dlat) * np.sin(dlon),
                     -np.cos(dlat) * np.cos(dlon),
                     np.sin(dlat) + 0.0 * dlon], axis=-1).reshape(-1, 3)
    design = encode_directions(degree, dirs)          # (H·W, K)
    coeffs, *_ = np.linalg.lstsq(design, grid[0].reshape(-1), rcond=None)
    grid[0] = (design @ coeffs).reshape(H, W)         # projection-free elevation

    attrs_out = {
        "elevation_m": float(attrs["elevation_m"]),
        "relief_m": float(attrs["relief_m"]),
        "hardness": float(attrs["hardness"]),
        "rainfall_mm": float(attrs["rainfall_mm"]),
        "uplift_mm_yr": float(attrs["uplift_mm_yr"]),
        "roughness": float(attrs["roughness"]),
        "influence_weight": float(weight_ch.mean()),
    }
    return Conditioning(attrs=attrs_out, extent=extent, grid=grid,
                        sh_coeffs=coeffs, sh_degree=degree,
                        influences=influences)
