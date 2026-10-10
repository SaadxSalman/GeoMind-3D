"""
Hybrid Spatial Vector Store.

``VECTOR_STORE`` in ``.env`` selects one of three interchangeable backends:

* ``local-hnsw`` — the in-process HNSW graph (default; zero infrastructure)
* ``qdrant``     — Qdrant REST API (``/collections/{c}/points/search``)
* ``supabase``   — Supabase PostgREST RPC over pgvector

All backends share one interface: ``add(records)`` / ``search(vector, k)``.
Records are report chunks carrying a bbox anchor so vector hits can be
re-projected into Spatial SQL.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.core.hnsw import HNSWIndex


@dataclass
class ChunkRecord:
    id: str
    text: str
    source_id: str
    source_name: str
    kind: str = "report"
    bbox: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore:
    backend = "base"

    def add(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


class LocalVectorStore(BaseVectorStore):
    backend = "local-hnsw"

    def __init__(self, dim: int = 1536):
        self.dim = dim
        index_path = settings.hnsw_index_path
        if index_path.exists():
            try:
                self.index = HNSWIndex.load(index_path)
                if self.index.dim != dim:
                    raise ValueError("dim drift")
            except Exception:
                self.index = HNSWIndex(dim, m=settings.hnsw_m,
                                       ef_construction=settings.hnsw_ef_construction,
                                       ef_search=settings.hnsw_ef_search)
        else:
            self.index = HNSWIndex(dim, m=settings.hnsw_m,
                                   ef_construction=settings.hnsw_ef_construction,
                                   ef_search=settings.hnsw_ef_search)
        self.records: Dict[str, ChunkRecord] = {}
        self._records_path = index_path.with_suffix(".records.json")
        if self._records_path.exists():
            try:
                raw = json.loads(self._records_path.read_text(encoding="utf-8"))
                self.records = {r["id"]: ChunkRecord(**r) for r in raw}
            except Exception:
                self.records = {}

    def add(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> None:
        for rec, vec in zip(records, vectors):
            self.index.add(np.asarray(vec, np.float32), rec.id)
            self.records[rec.id] = rec
        self.save()

    def search(self, vector: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        return self.index.search(vector, k=k)

    @property
    def size(self) -> int:
        return len(self.index)

    def save(self) -> None:
        self.index.save(settings.hnsw_index_path)
        payload = [{"id": c.id, "text": c.text, "source_id": c.source_id,
                    "source_name": c.source_name, "kind": c.kind,
                    "bbox": c.bbox, "meta": c.meta}
                   for c in self.records.values()]
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        self._records_path.write_text(json.dumps(payload, ensure_ascii=False),
                                      encoding="utf-8")


class QdrantVectorStore(BaseVectorStore):
    """Qdrant REST adapter (no client library required — plain httpx)."""
    backend = "qdrant"

    def __init__(self, url: str, api_key: str, collection: str, dim: int = 1536):
        import httpx
        self._url = url.rstrip("/")
        self._collection = collection
        self._dim = dim
        headers = {"Content-Type": "application/json"}
        if api_key and "REPLACE" not in api_key:
            headers["api-key"] = api_key
        self._client = httpx.Client(base_url=self._url, headers=headers, timeout=30.0)
        self._count = 0

    def add(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> None:
        points = [
            {"id": i, "vector": [float(x) for x in vec], "payload": {
                "chunk_id": rec.id, "text": rec.text, "source_id": rec.source_id,
                "source_name": rec.source_name, "kind": rec.kind, "bbox": rec.bbox,
                "meta": rec.meta}}
            for i, (rec, vec) in enumerate(zip(records, vectors))
        ]
        r = self._client.put(f"/collections/{self._collection}", json={
            "vectors": {"size": self._dim, "distance": "Cosine"}})
        if r.status_code not in (200, 409):
            r.raise_for_status()
        r = self._client.put(f"/collections/{self._collection}/points",
                             json={"points": points})
        r.raise_for_status()
        self._count += len(points)

    def search(self, vector: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        r = self._client.post(
            f"/collections/{self._collection}/points/search",
            json={"vector": [float(x) for x in vector], "limit": k,
                  "with_payload": ["chunk_id"]})
        r.raise_for_status()
        return [(p["payload"]["chunk_id"], float(p["score"]))
                for p in r.json().get("result", [])]

    @property
    def size(self) -> int:
        try:
            r = self._client.get(f"/collections/{self._collection}")
            if r.status_code == 200:
                return int(r.json()["result"]["points_count"])
        except Exception:
            pass
        return self._count


class SupabaseVectorStore(BaseVectorStore):
    """Supabase (pgvector) adapter via PostgREST RPC."""
    backend = "supabase"

    def __init__(self, url: str, key: str, table: str = "spatial_chunks",
                 rpc: str = "match_spatial_chunks", dim: int = 1536):
        import httpx
        self._url = url.rstrip("/")
        self._table = table
        self._rpc = rpc
        self._dim = dim
        headers = {
            "Content-Type": "application/json",
            "apikey": key or "anon",
            "Authorization": f"Bearer {key or 'anon'}",
        }
        self._client = httpx.Client(base_url=self._url, headers=headers, timeout=30.0)

    def add(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> None:
        rows = [
            {"id": rec.id, "text": rec.text, "source_id": rec.source_id,
             "source_name": rec.source_name, "kind": rec.kind, "bbox": rec.bbox,
             "meta": rec.meta, "embedding": [float(x) for x in vec]}
            for rec, vec in zip(records, vectors)
        ]
        r = self._client.post(f"/rest/v1/{self._table}", json=rows,
                              headers={"Prefer": "resolution=merge-duplicates"})
        r.raise_for_status()

    def search(self, vector: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        r = self._client.post(f"/rest/v1/rpc/{self._rpc}", json={
            "query_embedding": [float(x) for x in vector],
            "match_count": k})
        r.raise_for_status()
        return [(row["id"], float(row.get("similarity", 1.0 - row.get("distance", 0.0))))
                for row in r.json()]

    @property
    def size(self) -> int:
        try:
            r = self._client.get(f"/rest/v1/{self._table}",
                                 headers={"Prefer": "count=exact"})
            content = r.headers.get("content-range", "")
            if "/" in content:
                return int(content.split("/")[1])
        except Exception:
            pass
        return 0


def get_vector_store() -> BaseVectorStore:
    """Factory honouring ``VECTOR_STORE`` in ``.env``."""
    backend = settings.vector_store
    if backend == "qdrant":
        return QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key,
                                 settings.qdrant_collection, settings.embedding_dim)
    if backend == "supabase":
        return SupabaseVectorStore(settings.supabase_url, settings.supabase_key,
                                   settings.supabase_table, dim=settings.embedding_dim)
    return LocalVectorStore(settings.embedding_dim)

