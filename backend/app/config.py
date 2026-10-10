"""
GeoMind-3D central configuration.

Every key lives in ONE file: the repository-root `.env`. This module locates
that file (repo root, or CWD walking upwards), loads it with python-dotenv and
exposes a typed `settings` singleton. Nothing else in the codebase reads
environment variables directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# backend/app/config.py -> parents[0]=app, [1]=backend, [2]=repo root
_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), *_PACKAGE_DIR.parents]
    for cand in candidates:
        if (cand / ".env").exists():
            return cand
        if (cand / "backend").is_dir() and (cand / "run.py").exists():
            return cand
    return _PACKAGE_DIR.parents[1]


REPO_ROOT = _find_repo_root()
load_dotenv(REPO_ROOT / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    return default if val is None else val


def _int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(key: str, default: str) -> List[str]:
    raw = _env(key, default)
    return [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]


def _path(key: str, default: str) -> Path:
    raw = _env(key, default)
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: _env("APP_NAME", "GeoMind-3D"))
    app_env: str = field(default_factory=lambda: _env("APP_ENV", "development"))
    app_debug: bool = field(default_factory=lambda: _bool("APP_DEBUG", True))
    app_host: str = field(default_factory=lambda: _env("APP_HOST", "127.0.0.1"))
    app_port: int = field(default_factory=lambda: _int("APP_PORT", 8000))
    app_root_path: str = field(default_factory=lambda: _env("APP_ROOT_PATH", ""))
    app_log_level: str = field(default_factory=lambda: _env("APP_LOG_LEVEL", "info"))
    secret_key: str = field(default_factory=lambda: _env("APP_SECRET_KEY", "dev"))
    api_require_key: bool = field(default_factory=lambda: _bool("API_REQUIRE_KEY", False))
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))
    cors_origins: List[str] = field(default_factory=lambda: _list(
        "CORS_ORIGINS", "http://localhost:8000"))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 64))

    @property
    def artifacts_dir(self) -> Path:
        return _path("ARTIFACTS_DIR", "artifacts")

    @property
    def frontend_dir(self) -> Path:
        return _path("FRONTEND_DIR", "frontend")

    # ── embeddings ──
    @property
    def embedding_provider(self) -> str:
        return _env("EMBEDDING_PROVIDER", "local").strip().lower()

    @property
    def embedding_dim(self) -> int:
        return _int("EMBEDDING_DIM", 1536)

    @property
    def embedding_fallback(self) -> bool:
        return _bool("EMBEDDING_FALLBACK", True)

    @property
    def openai_api_key(self) -> str:
        return _env("OPENAI_API_KEY", "")

    @property
    def openai_base_url(self) -> str:
        return _env("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def openai_embedding_model(self) -> str:
        return _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def openai_chat_model(self) -> str:
        return _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    @property
    def anthropic_api_key(self) -> str:
        return _env("ANTHROPIC_API_KEY", "")

    @property
    def anthropic_model(self) -> str:
        return _env("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    @property
    def huggingface_api_key(self) -> str:
        return _env("HUGGINGFACE_API_KEY", "")

    @property
    def huggingface_model(self) -> str:
        return _env("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # ── vector store ──
    @property
    def vector_store(self) -> str:
        return _env("VECTOR_STORE", "local-hnsw").strip().lower()

    @property
    def hnsw_m(self) -> int:
        return _int("HNSW_M", 16)

    @property
    def hnsw_ef_construction(self) -> int:
        return _int("HNSW_EF_CONSTRUCTION", 100)

    @property
    def hnsw_ef_search(self) -> int:
        return _int("HNSW_EF_SEARCH", 64)

    @property
    def hnsw_index_path(self) -> Path:
        return _path("HNSW_INDEX_PATH", "artifacts/geomind_hnsw.npz")

    @property
    def qdrant_url(self) -> str:
        return _env("QDRANT_URL", "http://localhost:6333")

    @property
    def qdrant_api_key(self) -> str:
        return _env("QDRANT_API_KEY", "")

    @property
    def qdrant_collection(self) -> str:
        return _env("QDRANT_COLLECTION", "geomind_spatial_chunks")

    @property
    def supabase_url(self) -> str:
        return _env("SUPABASE_URL", "")

    @property
    def supabase_key(self) -> str:
        return _env("SUPABASE_SERVICE_KEY", "") or _env("SUPABASE_ANON_KEY", "")

    @property
    def supabase_table(self) -> str:
        return _env("SUPABASE_TABLE", "spatial_chunks")

    # ── spatial db ──
    @property
    def spatial_db_path(self) -> Path:
        return _path("SPATIAL_DB_PATH", "artifacts/geomind_spatial.db")

    # ── grounding ──
    @property
    def rrf_k(self) -> int:
        return _int("RRF_K", 60)

    @property
    def rrf_weights(self) -> dict:
        return {
            "vector": _float("RRF_WEIGHT_VECTOR", 1.0),
            "spatial": _float("RRF_WEIGHT_SPATIAL", 1.0),
            "gnn": _float("RRF_WEIGHT_GNN", 1.0),
        }

    @property
    def retrieval_top_k(self) -> int:
        return _int("RETRIEVAL_TOP_K", 8)

    @property
    def gnn_layers(self) -> int:
        return _int("GNN_LAYERS", 2)

    @property
    def gnn_seed(self) -> int:
        return _int("GNN_SEED", 1337)

    # ── latent diffusion ──
    @property
    def latent_dim(self) -> int:
        return _int("LATENT_DIM", 64)

    @property
    def diffusion_steps(self) -> int:
        return _int("DIFFUSION_STEPS", 48)

    @property
    def diffusion_beta_start(self) -> float:
        return _float("DIFFUSION_BETA_START", 1e-4)

    @property
    def diffusion_beta_end(self) -> float:
        return _float("DIFFUSION_BETA_END", 0.02)

    @property
    def diffusion_seed(self) -> int:
        return _int("DIFFUSION_SEED", 42)

    @property
    def sh_degree(self) -> int:
        return _int("SH_DEGREE", 4)

    # ── FNO ──
    @property
    def fno_modes(self) -> int:
        return _int("FNO_MODES", 8)

    @property
    def fno_width(self) -> int:
        return _int("FNO_WIDTH", 16)

    @property
    def fno_layers(self) -> int:
        return _int("FNO_LAYERS", 4)

    @property
    def fno_lr(self) -> float:
        return _float("FNO_LR", 1e-3)

    @property
    def fno_train_steps(self) -> int:
        return _int("FNO_TRAIN_STEPS", 400)

    @property
    def fno_batch_size(self) -> int:
        return _int("FNO_BATCH_SIZE", 8)

    @property
    def fno_cache_path(self) -> Path:
        return _path("FNO_CACHE_PATH", "artifacts/fno/fno_eroder.npz")

    @property
    def fno_auto_train(self) -> bool:
        return _bool("FNO_AUTO_TRAIN", True)

    @property
    def fno_seed(self) -> int:
        return _int("FNO_SEED", 7)

    # ── physics ──
    @property
    def erosion_rainfall(self) -> float:
        return _float("EROSION_RAINFALL", 0.02)

    @property
    def erosion_capacity(self) -> float:
        return _float("EROSION_CAPACITY", 0.4)

    @property
    def erosion_deposition(self) -> float:
        return _float("EROSION_DEPOSITION", 0.3)

    @property
    def erosion_evaporation(self) -> float:
        return _float("EROSION_EVAPORATION", 0.02)

    @property
    def thermal_talus_deg(self) -> float:
        return _float("THERMAL_TALUS_DEG", 34.0)

    @property
    def thermal_rate(self) -> float:
        return _float("THERMAL_RATE", 0.5)

    @property
    def seismic_wave_speed(self) -> float:
        return _float("SEISMIC_WAVE_SPEED", 2.0)

    @property
    def seismic_damping(self) -> float:
        return _float("SEISMIC_DAMPING", 0.004)

    @property
    def simulation_steps(self) -> int:
        return _int("SIMULATION_STEPS", 24)

    @property
    def solver_default(self) -> str:
        return _env("SOLVER_DEFAULT", "hydraulic").strip().lower()

    # ── rendering ──
    @property
    def mesh_resolution(self) -> int:
        return _int("MESH_RESOLUTION", 128)

    @property
    def splat_count(self) -> int:
        return _int("SPLAT_COUNT", 20000)

    @property
    def splat_opacity(self) -> float:
        return _float("SPLAT_OPACITY", 0.85)

    @property
    def mesh_exports(self) -> List[str]:
        return _list("MESH_EXPORTS", "obj,ply,glb,usdz")

    @property
    def mesh_slab_thickness(self) -> float:
        return _float("MESH_SLAB_THICKNESS", 8.0)

    # ── seed / corpus ──
    @property
    def seed_random_seed(self) -> int:
        return _int("SEED_RANDOM_SEED", 2026)

    @property
    def corpus_chunk_size(self) -> int:
        return _int("CORPUS_CHUNK_SIZE", 220)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

