"""
Latent basis: DCT autoencoder pair (exact linear autoencoder).

The "VAE" of the engine is an orthonormal DCT-II/III basis: the encoder
projects a conditioning grid onto the lowest ``LATENT_DIM`` frequency
coefficients, the decoder reconstructs. Because the basis is orthonormal the
reconstruction is exact for retained coefficients (Parseval-optimal — the
same optimality property PCA has, without needing a training corpus).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom


def _square(dim: int) -> int:
    k = int(math.isqrt(dim))
    return max(k * k, 4)


def encode_field(field: np.ndarray, latent_dim: int = 64) -> np.ndarray:
    """(H, W) field → (latent_dim,) DCT coefficients (ortho-normalised)."""
    coeffs = dctn(np.asarray(field, dtype=np.float64), type=2, norm="ortho")
    k = int(math.sqrt(_square(latent_dim)))
    flat = coeffs[:k, :k].reshape(-1)
    out = np.zeros(latent_dim, dtype=np.float64)
    out[:flat.size] = flat
    return out


def decode_field(z: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """(latent_dim,) → (H, W) field via inverse DCT."""
    h, w = shape
    k = int(math.sqrt(_square(len(z))))
    grid = np.zeros((k, k), dtype=np.float64)
    flat = np.asarray(z, dtype=np.float64)[:k * k]
    grid[: flat.size // k, :][...] = flat.reshape(-1)[: (flat.size // k) * k].reshape(
        flat.size // k, k)
    full = np.zeros((h, w), dtype=np.float64)
    kh, kw = min(k, h), min(k, w)
    full[:kh, :kw] = grid[:kh, :kw]
    return idctn(full, type=2, norm="ortho")


def upsample(field: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Cubic-zoom resampling to target shape."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape == tuple(shape):
        return field
    factors = (shape[0] / field.shape[0], shape[1] / field.shape[1])
    out = zoom(field, factors, order=3, grid_mode=False)
    # zoom can overshoot the requested size by a pixel — crop/pad safely
    out = out[: shape[0], : shape[1]]
    if out.shape != tuple(shape):
        pad = np.zeros(shape)
        pad[: out.shape[0], : out.shape[1]] = out
        out = pad
    return out


def spectral_tilt(shape: Tuple[int, int], pinkness: float = 1.0) -> np.ndarray:
    """1/(1+r^p) radial spectral envelope in DCT space — used to colour the
    latent prior so detail is pink-noise-like (low-frequency dominated)."""
    h, w = shape
    ky = np.arange(h)[:, None]
    kx = np.arange(w)[None, :]
    r = np.sqrt((ky / max(h, 1)) ** 2 + (kx / max(w, 1)) ** 2)
    return 1.0 / (1.0 + (r * 6.0) ** pinkness)
