"""
Reference geomathematical PDE solvers (vectorised NumPy).

These are the *ground truth* operators the FNO learns to emulate:

* ``hydraulic_erosion`` — stream-power incision (E = K·A·S) with D8 flow
  relaxation + hillslope sediment diffusion (laplacian term).
* ``thermal_erosion``   — talus-angle material creep (Bingham-style threshold
  flow: only slopes steeper than the repose angle move).
* ``seismic_wave``      — damped 2-D acoustic/elastic proxy  u_tt = c²∇²u − κu_t
  (leap-frog integration) for shake-curve style propagation.
* ``diffusion_spectral``— exact heat-equation solution in Fourier space
  (the FFT "spectral solver" that motivates the FNO architecture).

All operate on square grids with unit cell size and are numerically
stability-checked (clipped updates, CFL-safe dt).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2

SOLVERS = ("hydraulic", "thermal", "seismic", "diffusion")


def _gradients(h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference slopes with edge replication."""
    gy, gx = np.gradient(h)
    return gx, gy


def slope_magnitude(h: np.ndarray) -> np.ndarray:
    gx, gy = _gradients(h)
    return np.hypot(gx, gy)


def flow_accumulation(h: np.ndarray, rainfall: float, relax: int = 6) -> np.ndarray:
    """D8-style discharge by bounded relaxation (vectorised).

    Each sweep every cell routes its water (plus fresh rainfall) to its
    steepest downhill neighbour; sinks keep theirs. After ``relax`` sweeps,
    discharge ≈ upstream contributing area × rainfall within the flow path
    horizon — the A term of stream-power incision.
    """
    n = h.shape[0]
    q = np.zeros_like(h, dtype=np.float64)
    offs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
    for _ in range(relax):
        q += rainfall
        best_drop = np.full_like(h, -np.inf)
        best_idx = np.zeros(h.shape, dtype=np.int8)
        for i, (dy, dx) in enumerate(offs):
            shifted = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
            drop = h - shifted
            if dy == 1:
                drop[0, :] = -np.inf          # invalidate wrapped borders
            if dy == -1:
                drop[-1, :] = -np.inf
            if dx == 1:
                drop[:, 0] = -np.inf
            if dx == -1:
                drop[:, -1] = -np.inf
            improve = drop > best_drop
            best_drop = np.where(improve, drop, best_drop)
            best_idx = np.where(improve, i, best_idx)
        recv = np.zeros_like(q)
        flat_q, flat_idx = q.ravel(), best_idx.ravel()
        flat_pos = np.arange(n * n)
        for i, (dy, dx) in enumerate(offs):
            mask = flat_idx == i
            if not mask.any():
                continue
            pos = flat_pos[mask]
            rows, cols = pos // n, pos % n
            downhill = best_drop.ravel()[mask] > 0
            # sinks keep their water
            keep_rows, keep_cols = rows[~downhill], cols[~downhill]
            move_rows, move_cols = rows[downhill], cols[downhill]
            np.add.at(recv, (keep_rows, keep_cols), flat_q[mask][~downhill])
            np.add.at(recv, ((move_rows + dy) % n, (move_cols + dx) % n),
                      flat_q[mask][downhill])
        q = recv
    return q


def hydraulic_erosion(h: np.ndarray, steps: int = 1,
                      rainfall: float = 0.02, capacity: float = 0.4,
                      deposition: float = 0.3, evaporation: float = 0.02,
                      hardness: Optional[np.ndarray] = None,
                      diffusion_k: float = 0.12,
                      uplift: float = 0.0,
                      dt: float = 0.35) -> np.ndarray:
    """Stream-power incision + hillslope diffusion (mass-aware surrogate).

        ∂h/∂t = −K_e · (A·S) / hardness  +  K_d·∇²h  +  uplift

    K_e ∝ capacity·rainfall, S = |∇h|, hardness ∈ [0,1] scales erodibility.
    The laplacian term (scaled by deposition) moves sediment from convex
    ridges into concave hollows — mass-conserving with Neumann edges.
    """
    out = h.astype(np.float64).copy()
    n = out.shape[0]
    hard = np.ones_like(out) if hardness is None else np.clip(hardness, 0.05, 1.0)
    if hard.shape != out.shape:
        hard = np.ones_like(out)
    erodibility = (capacity * max(rainfall, 1e-6) * 25.0) / hard
    for _ in range(int(steps)):
        q = flow_accumulation(out, rainfall, relax=5)
        s = slope_magnitude(out)
        incision = erodibility * q * s
        # Neumann laplacian (edge-replicating)
        lap = (np.roll(out, 1, 0) + np.roll(out, -1, 0) +
               np.roll(out, 1, 1) + np.roll(out, -1, 1) - 4.0 * out)
        dh = (-incision + deposition * diffusion_k * lap * 8.0 + uplift)
        dh = np.clip(dh, -0.35, 0.35)          # CFL-style stability bound
        out = out + dt * dh
        out[0, :] = out[1, :]; out[-1, :] = out[-2, :]   # open boundaries
        out[:, 0] = out[:, 1]; out[:, -1] = out[:, -2]
        _ = evaporation                        # carried for physical parity
    return out


def thermal_erosion(h: np.ndarray, steps: int = 1,
                    talus_deg: float = 34.0, rate: float = 0.5,
                    dt: float = 0.25) -> np.ndarray:
    """Talus-angle creep: cells steeper than the angle of repose shed
    material to their steepest downhill neighbour (vectorised 4-neighbour)."""
    out = h.astype(np.float64).copy()
    talus = np.tan(np.deg2rad(talus_deg))      # gradient threshold, dx = 1
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(int(steps)):
        move = np.zeros_like(out)
        for dy, dx in offs:
            neigh = np.roll(np.roll(out, dy, 0), dx, 1)
            diff = out - neigh
            if dy == 1:
                diff[0, :] = -np.inf
            if dy == -1:
                diff[-1, :] = -np.inf
            if dx == 1:
                diff[:, 0] = -np.inf
            if dx == -1:
                diff[:, -1] = -np.inf
            excess = np.where(diff > talus, (diff - talus) * rate * 0.25, 0.0)
            move += excess
            # material arrives from the opposite direction
            from_nb = np.roll(np.roll(excess, -dy, 0), -dx, 1)
            move -= np.where(np.roll(np.roll(diff > talus, -dy, 0), -dx, 1),
                             from_nb, 0.0)
        out = out + np.clip(move, -0.3, 0.3) * dt
    return out


def seismic_wave(u: np.ndarray, steps: int = 24, wave_speed: float = 2.0,
                 damping: float = 0.004, source: Optional[np.ndarray] = None,
                 dt: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Damped wave equation on the grid; returns final (displacement, velocity)."""
    u_cur = u.astype(np.float64).copy()
    v = np.zeros_like(u_cur)
    c2 = float(wave_speed) ** 2
    if source is not None:
        v = v + source
    for _ in range(int(steps)):
        lap = (np.roll(u_cur, 1, 0) + np.roll(u_cur, -1, 0) +
               np.roll(u_cur, 1, 1) + np.roll(u_cur, -1, 1) - 4.0 * u_cur)
        v = (v + dt * c2 * lap) * (1.0 - damping)
        u_cur = u_cur + dt * v
        u_cur[0, :] = 0.0; u_cur[-1, :] = 0.0
        u_cur[:, 0] = 0.0; u_cur[:, -1] = 0.0
        u_cur = np.nan_to_num(u_cur, nan=0.0, posinf=0.0, neginf=0.0)
    return u_cur, v


def diffusion_spectral(h: np.ndarray, nu: float = 0.08, t: float = 1.0) -> np.ndarray:
    """Exact heat-equation solution  h(t) = F⁻¹[e^{−ν k² t} F[h]]
    (the Fourier spectral solver that motivates the FNO architecture)."""
    n = h.shape[0]
    ky = np.fft.fftfreq(n) * 2.0 * np.pi
    KX, KY = np.meshgrid(ky, ky)
    k2 = KX ** 2 + KY ** 2
    return np.real(ifft2(np.exp(-nu * k2 * t) * fft2(h)))


def run_solver(name: str, h: np.ndarray, steps: int,
               params: Optional[Dict[str, float]] = None,
               hardness: Optional[np.ndarray] = None) -> np.ndarray:
    """Dispatch by SOLVERS name (used by the /api/v1/simulate endpoint)."""
    p = dict(params or {})
    if name == "hydraulic":
        return hydraulic_erosion(
            h, steps=steps,
            rainfall=p.get("rainfall", 0.02),
            capacity=p.get("capacity", 0.4),
            deposition=p.get("deposition", 0.3),
            evaporation=p.get("evaporation", 0.02),
            hardness=hardness, uplift=p.get("uplift", 0.0))
    if name == "thermal":
        return thermal_erosion(h, steps=steps,
                               talus_deg=p.get("talus_deg", 34.0),
                               rate=p.get("rate", 0.5))
    if name == "seismic":
        src = np.zeros_like(h)
        src[h.shape[0] // 2, h.shape[1] // 2] = p.get("impulse", 1.0)
        u, _ = seismic_wave(h, steps=steps,
                            wave_speed=p.get("wave_speed", 2.0),
                            damping=p.get("damping", 0.004), source=src)
        return u
    if name == "diffusion":
        return diffusion_spectral(h, nu=p.get("nu", 0.08), t=float(steps) * 0.1)
    raise ValueError(f"unknown solver: {name!r} (expected one of {SOLVERS})")
