"""
Conditional latent diffusion — exact reverse chain with a conjugate prior.

Forward (DDPM):
    z_t = √ᾱ_t · z_0 + √(1−ᾱ_t) · ε ,   ε ~ N(0, I)
    ᾱ_t = ∏_{s≤t} (1 − β_s),   β_s linear from 1e-4 → 0.02

Conditional prior from the grounding stage (diagonal Gaussian in latent
space):
    z_0 ~ N(μ, Σ),   μ = DCT-mean shaped by elevation,   Σ = diag(σ²)
    σ_k = relief · spectral_tilt(k) · (0.55 + roughness)

Because prior and likelihood are Gaussian, the *exact* reverse marginal is
closed-form (no learned score network needed):

    p(z_{t−1} | z_t) = N( a_t · z_t + b_t ,  v_t )
        a_t = √α_t (1−ᾱ_{t−1}) / (1−ᾱ_t)
        b_t = √ᾱ_{t−1} β_t / (1−ᾱ_t) · μ
        v_t = [ √ᾱ_{t−1} β_t / (1−ᾱ_t) ]² · σ² + β̃_t

Iterating T steps from pure noise samples z_T ~ N(√ᾱ_T μ, (1−ᾱ_T)I+ᾱ_T Σ)
converges to the conditional prior — i.e. this IS posterior sampling under
the conditioning, implemented as a diffusion chain. Reproducible per seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.core.latent.basis import spectral_tilt


@dataclass
class LatentPrior:
    """Diagonal Gaussian conditioning prior in DCT-latent space."""
    mean: np.ndarray            # (latent_dim,)
    std: np.ndarray             # (latent_dim,)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])


def build_prior(conditioning: Dict[str, float], latent_dim: int,
                grid: Tuple[int, int] = (8, 8)) -> LatentPrior:
    """Map physical conditioning scalars to a diagonal latent prior.

    elevation → DC term; relief → overall detail amplitude; hardness/rainfall
    → spectral slope (hard rock & dry climate keep high-frequency ridges,
    soft wet terrain smooths out); roughness → variance floor.
    """
    elev = float(conditioning.get("elevation_m", 1500.0)) / 6000.0
    relief = float(conditioning.get("relief_m", 1000.0)) / 3500.0
    hardness = float(conditioning.get("hardness", 5.0)) / 10.0
    rainfall = float(conditioning.get("rainfall_mm", 500.0)) / 1200.0
    rough = float(conditioning.get("roughness", 0.5))

    pinkness = 1.9 - 1.1 * hardness - 0.4 * rainfall      # sharper → more HF
    tilt = spectral_tilt(grid, pinkness=float(np.clip(pinkness, 0.4, 2.2)))
    tilt_flat = tilt.reshape(-1)[:latent_dim]
    if tilt_flat.size < latent_dim:                        # pad safety
        tilt_flat = np.pad(tilt_flat, (0, latent_dim - tilt_flat.size))

    sigma = relief * tilt_flat * (0.55 + rough) + 1e-3
    mean = np.zeros(latent_dim, dtype=np.float64)
    mean[0] = elev * float(np.prod(grid)) * 0.15           # DC elevation anchor
    return LatentPrior(mean=mean, std=sigma.astype(np.float64))


def _beta_schedule(steps: int) -> np.ndarray:
    return np.linspace(settings.diffusion_beta_start,
                       settings.diffusion_beta_end, steps)


def sample(prior: LatentPrior, steps: Optional[int] = None, seed: int = 0,
           return_trajectory: bool = False) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
    """Run the reverse diffusion chain; returns z_0 (and optional trajectory)."""
    steps = int(steps or settings.diffusion_steps)
    rng = np.random.default_rng(seed if seed is not None else 0)
    beta = _beta_schedule(steps)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)

    mu, sig = prior.mean, prior.std
    # start from the exact marginal at t = T
    z = np.sqrt(alpha_bar[-1]) * mu + np.sqrt(1.0 - alpha_bar[-1]) * rng.normal(size=prior.dim)
    if np.any(sig > 0):
        z += np.sqrt(alpha_bar[-1]) * sig * rng.normal(size=prior.dim)

    traj: List[np.ndarray] = [z.copy()] if return_trajectory else []
    for t in range(steps - 1, -1, -1):
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else 1.0
        b_t = beta[t]
        a_t = np.sqrt(alpha[t]) * (1.0 - ab_prev) / (1.0 - ab_t)
        coef = np.sqrt(ab_prev) * b_t / (1.0 - ab_t)
        mean = a_t * z + coef * mu
        var = (coef ** 2) * (sig ** 2)
        if t > 0:
            beta_tilde = b_t * (1.0 - ab_prev) / (1.0 - ab_t)
            var = var + beta_tilde
            z = mean + np.sqrt(var) * rng.normal(size=prior.dim)
        else:
            z = mean                                   # deterministic at t=0
        if return_trajectory:
            traj.append(z.copy())
    return (z, traj) if return_trajectory else z
