"""
GeoMind-3D — single-command launcher.

    python run.py                 # start the server (auto-trains the FNO if missing)
    python run.py --no-fno-train  # skip FNO warm-up training
    python run.py --reload        # uvicorn auto-reload (dev)
    python run.py --host 0.0.0.0 --port 8080

Puts `backend/` on sys.path, warms the engines, then serves the API +
frontend from one origin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GeoMind-3D server")
    p.add_argument("--host", default=None, help="bind host (default: APP_HOST)")
    p.add_argument("--port", type=int, default=None, help="bind port (default: APP_PORT)")
    p.add_argument("--reload", action="store_true", help="uvicorn autoreload")
    p.add_argument("--no-fno-train", action="store_true", help="skip FNO warm-up training")
    p.add_argument("--fno-train-steps", type=int, default=None, help="override FNO warm-up steps")
    return p.parse_args()


def warm_engines(train_fno: bool, steps: int | None) -> None:
    """Pre-build the spatial DB, vector index and (optionally) the FNO."""
    from app.config import settings  # noqa: F401  (ensures .env is loaded)
    from app.core.knowledge_graph import get_repository

    repo = get_repository()
    print(f"[GeoMind-3D] spatial db   : {repo.spatial.db_path}")
    print(f"[GeoMind-3D] corpus       : {repo.vector_store.size} chunks, "
          f"{repo.graph.node_count} graph nodes, {repo.graph.edge_count} edges")

    if train_fno:
        from app.config import settings as s
        from app.physics.fno import get_fno
        n = steps if steps is not None else s.fno_train_steps
        try:
            model, info = get_fno(train_steps=n)
            print(f"[GeoMind-3D] FNO          : {info} (cache={s.fno_cache_path})")
        except Exception as exc:  # pragma: no cover - warm-up is best effort
            print(f"[GeoMind-3D] FNO warm-up skipped: {exc}")


def main() -> None:
    args = parse_args()
    import uvicorn

    from app.config import settings

    host = args.host or settings.app_host
    port = args.port or settings.app_port
    warm_engines(train_fno=not args.no_fno_train, steps=args.fno_train_steps)
    print(f"[GeoMind-3D] serving on http://{host}:{port}  (env={settings.app_env})")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=settings.app_log_level,
    )


if __name__ == "__main__":
    main()
