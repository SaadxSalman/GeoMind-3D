"""
Spherical Harmonic positional encoders  Y_l^m(θ, φ).

Standard complex spherical harmonics computed from normalised associated
Legendre functions (recurrence, no scipy special dependency):

    Y_l^m(θ,φ) = N_lm · P_l^m(cosθ) · e^{i m φ}
    N_lm       = sqrt( (2l+1)/(4π) · (l−m)!/(l+m)! )

On top of the complex basis we expose the *real* form used for terrain
encoding (graphics convention):

    m = 0 → Re(Y),   m > 0 → √2·Re(Y_l^m),   m < 0 → √2·Im(Y_l^|m|)

Why: Fourier coordinate encodings (NeRF-style) assume a flat plane and warp
badly when a scene spans a planetary curved surface. Expanding positions in
Y_l^m keeps the encoding isotropic on the sphere — no map-projection
distortion over large extents. The pipeline uses these basis functions for:

* the low-degree global curvature term of the conditioning raster,
* degree-3 colour SH of exported 3D Gaussians (f_dc / f_rest coefficients).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

# ── associated Legendre recurrence ─────────────────────────────────────────
@lru_cache(maxsize=32)
def _normalisation(l: int, m: int) -> float:
    log_nm = (math.log(2 * l + 1) - math.log(4 * math.pi)
              + _log_fact(l - m) - _log_fact(l + m))
    return math.sqrt(math.exp(log_nm))


@lru_cache(maxsize=512)
def _log_fact(n: int) -> float:
    return math.lgamma(n + 1)


def associated_legendre(l_max: int, x: np.ndarray) -> dict:
    """P_l^m(x) for 0≤m≤l≤l_max, Condon–Shortley phase included.

    Returns {(l, m): array_like(x)}.
    """
    x = np.asarray(x, dtype=np.float64)
    p: dict = {}
    p[(0, 0)] = np.ones_like(x)
    sin_theta = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    for m in range(1, l_max + 1):
        p[(m, m)] = -(2 * m - 1) * sin_theta * p[(m - 1, m - 1)]
    for m in range(0, l_max + 1):
        for l in range(m + 1, l_max + 1):
            p[(l, m)] = ((2 * l - 1) * x * p[(l - 1, m)]
                         - (l + m - 1) * (p[(l - 2, m)] if l - 2 >= m else 0.0)) / (l - m)
    return p


def complex_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Complex Y_l^m. θ ∈ [0, π] colatitude, φ ∈ [0, 2π) azimuth."""
    if abs(m) > l:
        return np.zeros_like(np.asarray(theta, dtype=np.float64))
    if m < 0:  # Y_l^{-m} = (−1)^m conj(Y_l^m)
        return ((-1) ** -m) * np.conj(complex_sph_harm(l, -m, theta, phi))
    p = associated_legendre(l, np.cos(np.asarray(theta, dtype=np.float64)))[(l, m)]
    return _normalisation(l, m) * p * np.exp(1j * m * np.asarray(phi))


def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal Y_l^m (graphics convention, index m ∈ [−l, l])."""
    if m == 0:
        return np.real(complex_sph_harm(l, 0, theta, phi))
    if m > 0:
        return np.sqrt(2.0) * np.real(complex_sph_harm(l, m, theta, phi))
    return np.sqrt(2.0) * np.imag(complex_sph_harm(l, -m, theta, phi))


def real_basis_count(degree: int) -> int:
    return (degree + 1) ** 2


def real_sph_design(degree: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Design matrix (N, (deg+1)^2) of real SH evaluations at N directions."""
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    cols = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            cols.append(real_sph_harm(l, m, theta, phi))
    return np.stack(cols, axis=-1)


def encode_directions(degree: int, dirs: np.ndarray) -> np.ndarray:
    """Unit vectors (N,3) → real SH coefficients (N, (deg+1)^2) by quadrature
    of the field sampled on the sphere (least squares for sparse samples)."""
    dirs = np.asarray(dirs, dtype=np.float64)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0]) % (2 * np.pi)
    return real_sph_design(degree, theta, phi)
