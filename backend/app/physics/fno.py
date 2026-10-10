"""
Fourier Neural Operator (FNO-2D) — NumPy implementation with exact
hand-derived gradients + Adam.

Architecture (Li et al. 2021, adapted):

    lift   :  y = W₁·x + b₁                                   (1×1 conv)
    layerₗ :  y ← σ( IFFT( W_c ⊙ FFT(y) ) + W_p·y + b_p )     (spectral + pointwise)
    project:  ŷ = W₃·y + b₃                                   (1×1 conv)

with truncated Fourier modes: W_c acts only on |k_y| < modes, |k_x| < modes.

Gradients (validated by finite differences in tests, for real signals):
    y = Re IFFT(M)            →  ∂L/∂M = FFT(g) / (H·W)
    M = W_c ⊙ X,  X = FFT(x)  →  ∂L/∂W_c = Σ_b g_M · conj(X)
                                  ∂L/∂X   = Σ_o g_M · conj(W_c)
    X = FFT(y)                →  ∂L/∂y   = Re( H·W · IFFT(∂L/∂X) )

The model learns the erosion *operator*  (h, hardness, rain) → Δh  from
rollouts of the reference solver, then serves as a fast surrogate in the
simulation endpoint. Weights cache to FNO_CACHE_PATH as .npz.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import fft as sfft

from app.config import settings

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def _gelu(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x2 = x * x
    x3 = x2 * x
    t = np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x3))
    return (0.5 * x * (1.0 + t)), t        # (value, tanh state)


def _gelu_grad_from_t(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """GELU derivative reusing the tanh state computed in forward."""
    x2 = x * x
    dt = (1.0 - t * t) * _SQRT_2_OVER_PI * (1.0 + 0.134145 * x2)
    return 0.5 * (1.0 + t) + 0.5 * x * dt


def _mode_indices(size: int, modes: int) -> np.ndarray:
    m = min(modes, size // 2)
    return np.concatenate([np.arange(0, m), np.arange(size - m, size)])


class FNO2d:
    def __init__(self, cin: int = 3, cout: int = 1, modes: Optional[int] = None,
                 width: Optional[int] = None, layers: Optional[int] = None,
                 seed: Optional[int] = None, dtype: Any = np.float32):
        self.cin = cin
        self.cout = cout
        self.modes = int(modes if modes is not None else settings.fno_modes)
        self.width = int(width if width is not None else settings.fno_width)
        self.layers = int(layers if layers is not None else settings.fno_layers)
        self.seed = int(seed if seed is not None else settings.fno_seed)
        self.dtype = np.dtype(dtype)
        self.rtype = np.complex64 if self.dtype == np.float32 else np.complex128
        self.params: Dict[str, np.ndarray] = self._init_params()

    # ── initialisation ─────────────────────────────────────────────────────
    def _init_params(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        w, c, modes = self.width, self.cin, self.modes
        f, cx = self.dtype, self.rtype
        p: Dict[str, np.ndarray] = {}
        p["lift_w"] = (rng.standard_normal((w, c)) / np.sqrt(c)).astype(f)
        p["lift_b"] = np.zeros(w, dtype=f)
        for l in range(self.layers):
            scale = 1.0 / np.sqrt(w)
            p[f"spc{l}_w"] = ((rng.standard_normal((w, w, 2 * modes, 2 * modes))
                               + 1j * rng.standard_normal((w, w, 2 * modes, 2 * modes)))
                              * scale).astype(cx)
            p[f"ptw{l}_w"] = (rng.standard_normal((w, w)) * scale).astype(f)
            p[f"ptw{l}_b"] = np.zeros(w, dtype=f)
        p["proj_w"] = (rng.standard_normal((self.cout, w)) / np.sqrt(w)).astype(f)
        p["proj_b"] = np.zeros(self.cout, dtype=f)
        return p

    def param_count(self) -> int:
        return int(sum(np.prod(v.shape) for v in self.params.values()))

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray,
                params: Optional[Dict[str, np.ndarray]] = None
                ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        p = params or self.params
        x = np.asarray(x, dtype=self.dtype)
        B, _, H, W = x.shape
        iy, ix = _mode_indices(H, self.modes), _mode_indices(W, self.modes)
        cache: List[Dict[str, Any]] = []

        y = (p["lift_w"] @ x.reshape(B, self.cin, -1)).reshape(B, self.width, H, W)
        y = y + p["lift_b"][:, None, None]
        cache.append({"kind": "lift", "x": x})

        for l in range(self.layers):
            X = sfft.fft2(y, workers=-1)
            Wfull = np.zeros((self.width, self.width, H, W), dtype=self.rtype)
            Wfull[:, :, iy[:, None], ix[None, :]] = p[f"spc{l}_w"]
            M = np.einsum("oiyx,bixy->boxy", Wfull, X, optimize=True)
            spec = np.real(sfft.ifft2(M, workers=-1))
            point = (p[f"ptw{l}_w"] @ y.reshape(B, self.width, -1)).reshape(B, self.width, H, W)
            point = point + p[f"ptw{l}_b"][:, None, None]
            pre = spec + point
            out, t_state = _gelu(pre)
            cache.append({"kind": "layer", "y_in": y, "X": X, "Wfull": Wfull,
                          "pre": pre, "t": t_state})
            y = out

        out_final = (p["proj_w"] @ y.reshape(B, self.width, -1)).reshape(B, self.cout, H, W)
        out_final = out_final + p["proj_b"][:, None, None]
        cache.append({"kind": "proj", "y": y})
        return out_final, cache

    # ── backward ───────────────────────────────────────────────────────────
    def backward(self, cache: List[Dict[str, Any]], g_out: np.ndarray,
                 params: Optional[Dict[str, np.ndarray]] = None
                 ) -> Dict[str, np.ndarray]:
        p = params or self.params
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}
        g = np.asarray(g_out, dtype=self.dtype)
        B, _, H, W = g.shape
        N = H * W

        # projection layer:  out = W₃·y + b₃   (1×1 conv as batched matmul)
        y = cache[-1]["y"]
        g2 = g.reshape(B, self.cout, N)
        y2 = y.reshape(B, self.width, N)
        grads["proj_w"] = np.matmul(g2, np.swapaxes(y2, -1, -2)).sum(axis=0)
        grads["proj_b"] = g.sum(axis=(0, 2, 3))
        g_y = (np.swapaxes(p["proj_w"], -1, -2) @ g2).reshape(B, self.width, H, W)

        # hidden layers (reverse)
        iy, ix = _mode_indices(H, self.modes), _mode_indices(W, self.modes)
        for l in range(self.layers - 1, -1, -1):
            lc = cache[l + 1]
            y_in, X, Wfull, pre, t_state = (lc["y_in"], lc["X"], lc["Wfull"],
                                            lc["pre"], lc["t"])
            g_pre = g_y * _gelu_grad_from_t(pre, t_state)

            # spectral branch:  spec = Re IFFT(M),  M = W_c ⊙ FFT(y_in)
            g_M = sfft.fft2(g_pre, workers=-1) / N              # adjoint of Re∘IFFT
            gW_full = np.einsum("boxy,bixy->oiyx", g_M, np.conj(X), optimize=True)
            grads[f"spc{l}_w"] = gW_full[:, :, iy[:, None], ix[None, :]]
            g_X = np.einsum("boxy,oiyx->bixy", g_M, np.conj(Wfull), optimize=True)
            g_from_spec = np.real(sfft.ifft2(g_X, workers=-1)) * N   # adjoint of FFT

            # pointwise branch:  point = W_p·y_in + b_p
            gp2 = g_pre.reshape(B, self.width, N)
            yi2 = y_in.reshape(B, self.width, N)
            grads[f"ptw{l}_w"] = np.matmul(gp2, np.swapaxes(yi2, -1, -2)).sum(axis=0)
            grads[f"ptw{l}_b"] = g_pre.sum(axis=(0, 2, 3))
            g_from_point = (np.swapaxes(p[f"ptw{l}_w"], -1, -2) @ gp2) \
                .reshape(B, self.width, H, W)

            g_y = g_from_spec + g_from_point

        # lift:  y = W₁·x + b₁  (input gradient not needed for training)
        x0 = cache[0]["x"]
        gx2 = g_y.reshape(B, self.width, N)
        x2 = x0.reshape(B, self.cin, N)
        grads["lift_w"] = np.matmul(gx2, np.swapaxes(x2, -1, -2)).sum(axis=0)
        grads["lift_b"] = g_y.sum(axis=(0, 2, 3))
        return grads

    # ── objective ──────────────────────────────────────────────────────────
    def loss_grads(self, x: np.ndarray, y_true: np.ndarray,
                   params: Optional[Dict[str, np.ndarray]] = None
                   ) -> Tuple[float, Dict[str, np.ndarray]]:
        pred, cache = self.forward(x, params)
        diff = pred - np.asarray(y_true, dtype=self.dtype)
        loss = float(np.mean(diff ** 2))
        g = (2.0 * diff / diff.size).astype(self.dtype)
        return loss, self.backward(cache, g, params)

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path, info: Optional[Dict[str, Any]] = None) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.params)
        payload["__meta__"] = np.asarray(json.dumps({
            "cin": self.cin, "cout": self.cout, "modes": self.modes,
            "width": self.width, "layers": self.layers, "seed": self.seed,
            "dtype": str(self.dtype),
            "info": info or {},
        }))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> Tuple["FNO2d", Dict[str, Any]]:
        import json
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        dtype = np.float32 if meta.get("dtype", "float32") == "float32" else np.float64
        model = cls(cin=meta["cin"], cout=meta["cout"], modes=meta["modes"],
                    width=meta["width"], layers=meta["layers"],
                    seed=meta["seed"], dtype=dtype)
        for k in model.params:
            model.params[k] = np.asarray(data[k])
        model.dtype = np.dtype(model.params["lift_w"].dtype)
        model.rtype = np.complex64 if model.dtype == np.float32 else np.complex128
        return model, meta.get("info", {})


# ── dataset: operator learning from the reference solver ───────────────────
TARGET_SCALE = 100.0          # amplifies tiny erosion deltas for stable MSE


def _fbm_field(size: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    field = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        sigma = max(size / (3.0 * 2 ** o), 0.8)
        layer = gaussian_filter(rng.standard_normal((size, size)), sigma=sigma)
        layer /= max(float(layer.std()), 1e-9)
        field += amp * layer
        total += amp
        amp *= 0.55
    field /= total
    field -= field.min()
    field /= max(field.max(), 1e-9)
    return field


def make_dataset(n: int = 64, size: int = 32,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y): x=(3,H,W) [h_zscore, hardness, rain], y=(1,H,W) = 100·Δh."""
    from app.physics.solvers import hydraulic_erosion
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        h = _fbm_field(size, rng)
        hard = rng.uniform(0.3, 1.0, (size, size))
        rain = float(rng.uniform(0.01, 0.05))
        h1 = hydraulic_erosion(h, steps=1, rainfall=rain, capacity=0.4,
                               deposition=0.3, hardness=hard)
        mu, sd = float(h.mean()), float(h.std()) + 1e-9
        xs.append(np.stack([(h - mu) / sd, hard, np.full_like(h, rain * 20.0)]))
        ys.append(((h1 - h) * TARGET_SCALE)[None, ...])
    return np.asarray(xs), np.asarray(ys)


# ── Adam + training loop ───────────────────────────────────────────────────
def _adam_step(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
               state: Dict[str, Dict[str, np.ndarray]], lr: float,
               t: int, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8) -> None:
    for k, g in grads.items():
        st = state.setdefault(k, {"m": np.zeros_like(params[k]),
                                  "v": np.zeros_like(params[k])})
        st["m"] = beta1 * st["m"] + (1 - beta1) * g
        st["v"] = beta2 * st["v"] + (1 - beta2) * (g * g)
        mhat = st["m"] / (1 - beta1 ** t)
        vhat = st["v"] / (1 - beta2 ** t)
        params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)


def train_fno(steps: Optional[int] = None, n_samples: int = 48,
              batch_size: Optional[int] = None, lr: Optional[float] = None,
              size: int = 32, seed: Optional[int] = None,
              verbose: bool = False) -> Tuple[FNO2d, Dict[str, Any]]:
    """Train the erosion-operator FNO against the reference solver."""
    steps = int(steps if steps is not None else settings.fno_train_steps)
    batch = int(batch_size or settings.fno_batch_size)
    lr = float(lr or settings.fno_lr)
    seed = int(seed if seed is not None else settings.fno_seed)

    model = FNO2d(seed=seed)
    xs, ys = make_dataset(n=n_samples, size=size, seed=seed + 11)
    rng = np.random.default_rng(seed + 23)
    state: Dict[str, Dict[str, np.ndarray]] = {}
    losses: List[float] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(xs), size=min(batch, len(xs)))
        loss, grads = model.loss_grads(xs[idx], ys[idx])
        _adam_step(model.params, grads, state, lr, step)
        losses.append(loss)
        if verbose and step % max(steps // 10, 1) == 0:
            print(f"  [fno] step {step:4d}  loss={loss:.3e}")
    info = {
        "steps": steps, "samples": n_samples, "params": model.param_count(),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses),
        "improved": bool(losses[-1] < losses[0]),
        "seconds": round(time.perf_counter() - t0, 3),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return model, info


# ── global accessor ────────────────────────────────────────────────────────
_FNO_CACHE: Dict[str, Any] = {}


def get_fno(train_steps: Optional[int] = None) -> Tuple[FNO2d, Dict[str, Any]]:
    """Load cached weights, or (auto)train once and persist to FNO_CACHE_PATH."""
    if "model" in _FNO_CACHE:
        return _FNO_CACHE["model"], _FNO_CACHE["info"]
    cache_path = settings.fno_cache_path
    if cache_path.exists():
        model, info = FNO2d.load(cache_path)
        info.setdefault("source", "cache")
        _FNO_CACHE.update(model=model, info=info)
        return model, info
    if not settings.fno_auto_train:
        model = FNO2d()
        info = {"source": "untrained"}
        _FNO_CACHE.update(model=model, info=info)
        return model, info
    model, info = train_fno(steps=train_steps)
    info["source"] = "trained"
    model.save(cache_path, info)
    _FNO_CACHE.update(model=model, info=info)
    return model, info


def fno_field_step(model: FNO2d, h: np.ndarray, hardness: np.ndarray,
                   rainfall: float) -> np.ndarray:
    """One surrogate step: h → h + FNO(h, hardness, rain)/TARGET_SCALE."""
    mu, sd = float(h.mean()), float(h.std()) + 1e-9
    x = np.stack([[(h - mu) / sd, np.clip(hardness, 0.05, 1.0),
                   np.full_like(h, rainfall * 20.0)]])
    pred, _ = model.forward(x)
    return h + pred[0, 0] / TARGET_SCALE


def run_fno(model: FNO2d, h: np.ndarray, steps: int,
            hardness: Optional[np.ndarray] = None,
            rainfall: float = 0.02) -> np.ndarray:
    out = h.astype(np.float64).copy()
    hard = np.ones_like(out) if hardness is None else np.clip(hardness, 0.05, 1.0)
    for _ in range(int(steps)):
        out = fno_field_step(model, out, hard, rainfall)
    return out


def benchmark(model: FNO2d, h: np.ndarray, steps: int = 4,
              hardness: Optional[np.ndarray] = None,
              rainfall: float = 0.02, repeats: int = 3) -> Dict[str, Any]:
    """Wall-clock reference vs FNO comparison on identical inputs."""
    from app.physics.solvers import hydraulic_erosion
    hard = np.ones_like(h) if hardness is None else hardness
    t0 = time.perf_counter()
    ref = hydraulic_erosion(h, steps=steps, rainfall=rainfall, hardness=hard)
    ref_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    t0 = time.perf_counter()
    for _ in range(repeats):
        sur = run_fno(model, h, steps, hardness=hard, rainfall=rainfall)
    fno_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    denom = max(float(np.linalg.norm(ref)), 1e-12)
    return {
        "steps": steps, "grid": list(h.shape),
        "reference_ms": round(ref_ms, 3), "fno_ms": round(fno_ms, 3),
        "speedup": round(ref_ms / max(fno_ms, 1e-6), 2),
        "rel_l2_error": round(float(np.linalg.norm(sur - ref) / denom), 4),
    }
