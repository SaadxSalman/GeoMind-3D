"""
Spatial Knowledge Graph Engine + grounding repository.

Responsibilities
----------------
* seed the SQLite spatial DB (features, reports, typed edges) idempotently,
* chunk + embed report text into the hybrid vector index,
* build the graph node-feature matrix (projected text embedding + physical
  props) and run relational GNN message passing,
* expose ``Repository.retrieve()`` — the full grounding pass that returns the
  three source rankings plus the RRF fusion consumed by the latent engine.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.core import geometry as g
from app.core.embeddings import BaseEmbedder, get_embedder
from app.core.gnn import RELATIONS, RelationalGNN
from app.core.rrf import rrf_fuse
from app.core.spatial_sql import SpatialDB
from app.core.vector_store import (BaseVectorStore, ChunkRecord,
                                   LocalVectorStore, get_vector_store)
from app.data.seed_features import AQUIFERS, BOREHOLES, FAULTS, REGIONS, UNITS
from app.data.seed_reports import REPORTS

_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Physical property channels used by the conditioning rasteriser.
PROP_KEYS = [
    ("elevation_m", 1.0 / 6000.0), ("relief_m", 1.0 / 3500.0),
    ("hardness", 1.0 / 10.0), ("rainfall_mm", 1.0 / 1200.0),
    ("uplift_mm_yr", 1.0 / 8.0), ("roughness", 1.0),
    ("vp_ms", 1.0 / 7000.0), ("perm_mD", 1.0 / 1500.0),
]


def chunk_text(text: str, size: int) -> List[str]:
    """Sentence-aware chunking (~size chars) used for the vector index."""
    sentences = [s.strip() for s in _SENT_RE.split(text.replace("\n", " ")) if s.strip()]
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + 1 + len(sent) > size:
            chunks.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        chunks.append(buf)
    return chunks or [text[:size]]


def _props_vector(props: Dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(PROP_KEYS), dtype=np.float32)
    for i, (key, scale) in enumerate(PROP_KEYS):
        v = props.get(key)
        if v is None and key == "perm_mD":
            v = props.get("transmissivity_mDm")
        if v is None:
            v = props.get("permeability_mD")
        if v is None:
            v = 0.0
        out[i] = float(v) * scale
    return np.clip(out, 0.0, 2.0)


def _feature_text(name: str, kind: str, props: Dict[str, Any]) -> str:
    parts = [f"{name} ({kind})", f"properties: {', '.join(f'{k}={v}' for k, v in props.items())}"]
    return ". ".join(parts)


def seed_database(spatial: SpatialDB, embedder: BaseEmbedder,
                  vector_store: BaseVectorStore) -> Dict[str, int]:
    """Idempotent corpus seeding. Returns counts of inserted objects."""
    stats = {"features": 0, "edges": 0, "chunks": 0}
    already = len(spatial.all_features()) > 0

    if not already:
        for entries, kind in ((REGIONS, "region"), (FAULTS, "fault"),
                              (AQUIFERS, "aquifer"), (UNITS, "unit"),
                              (BOREHOLES, "borehole")):
            for e in entries:
                spatial.upsert_feature(e["id"], e["name"], kind, e["geom"],
                                       e["props"], _feature_text(e["name"], kind, e["props"]))
                stats["features"] += 1
        for rep in REPORTS:
            x0, y0, x1, y1 = rep["bbox"]
            geom = {"type": "Polygon",
                    "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
            spatial.upsert_feature(rep["id"], rep["name"], "report", geom,
                                   {"bbox": rep["bbox"]}, rep["text"])
            stats["features"] += 1

    if len(spatial.all_edges()) == 0:
        stats["edges"] = spatial.build_spatial_edges(
            ("ST_Intersects", "ST_Contains", "ST_Touches"))
        stats["edges"] += _semantic_edges(spatial)

    if vector_store.size == 0:
        records: List[ChunkRecord] = []
        for rep in REPORTS:
            for i, chunk in enumerate(chunk_text(rep["text"], settings.corpus_chunk_size)):
                records.append(ChunkRecord(
                    id=f"{rep['id']}#c{i:02d}", text=chunk,
                    source_id=rep["id"], source_name=rep["name"],
                    kind="report", bbox=rep["bbox"]))
        if records:
            vecs = embedder.embed([r.text for r in records])
            vector_store.add(records, vecs)
            stats["chunks"] = len(records)
    return stats


def _semantic_edges(spatial: SpatialDB) -> int:
    """`mentions` edges: report text referencing a feature name, plus
    `similar` edges between features sharing lithology keywords."""
    count = 0
    features = spatial.all_features()
    reports = [f for f in features if f["kind"] == "report"]
    others = [f for f in features if f["kind"] != "report"]
    for rep in reports:
        low = rep["text"].lower()
        for feat in others:
            if feat["name"].lower() in low or feat["id"].split("-")[-1].lower() in low:
                spatial.upsert_edge(rep["id"], feat["id"], "mentions", 1.0)
                spatial.upsert_edge(feat["id"], rep["id"], "mentions", 0.5)
                count += 1
    for i, a in enumerate(others):
        la = set(re.findall(r"[a-z]{5,}", (a["text"] or "").lower()))
        for b in others[i + 1:]:
            lb = set(re.findall(r"[a-z]{5,}", (b["text"] or "").lower()))
            shared = la & lb
            if len(shared) >= 3:
                spatial.upsert_edge(a["id"], b["id"], "similar",
                                    min(1.0, len(shared) / 8.0))
                count += 1
    return count


class KnowledgeGraph:
    """In-memory graph with GNN-propagated node embeddings."""

    CONTENT_DIM = 64

    def __init__(self, spatial: SpatialDB, embedder: BaseEmbedder):
        self.spatial = spatial
        self.embedder = embedder
        self.nodes: List[Dict[str, Any]] = spatial.all_features()
        self.index: Dict[str, int] = {n["id"]: i for i, n in enumerate(self.nodes)}
        self.edges: Dict[str, List[Tuple[int, int]]] = self._load_edges()
        # query/content projection aligned with the content block (seeded, fixed)
        rng = np.random.default_rng(settings.gnn_seed + 1)
        proj = rng.standard_normal((self.embedder.dim, self.CONTENT_DIM))
        q, _ = np.linalg.qr(proj)                      # orthonormal basis
        self.query_proj = q.astype(np.float32)
        self.features0 = self._build_features()
        self.gnn = RelationalGNN(
            in_dim=self.features0.shape[1], relations=RELATIONS,
            layers=settings.gnn_layers, seed=settings.gnn_seed)
        self.embeddings = self.gnn.normalised(self.features0, self.edges)

    # ── construction helpers ───────────────────────────────────────────────
    def _load_edges(self) -> Dict[str, List[Tuple[int, int]]]:
        out: Dict[str, List[Tuple[int, int]]] = {r: [] for r in RELATIONS}
        for e in self.spatial.all_edges():
            rel = e["rel"] if e["rel"] in out else "similar"
            if e["src"] in self.index and e["dst"] in self.index:
                out[rel].append((self.index[e["src"]], self.index[e["dst"]]))
        return out

    def _build_features(self) -> np.ndarray:
        texts = [n["text"] or n["name"] for n in self.nodes]
        embeds = self.embedder.embed(texts)                       # (N, 1536)
        content = embeds @ self.query_proj                        # (N, 64)
        props = np.stack([_props_vector(n["props"]) for n in self.nodes]) \
            if self.nodes else np.zeros((0, len(PROP_KEYS)), np.float32)
        return np.hstack([content, props]).astype(np.float32)

    # ── queries ────────────────────────────────────────────────────────────
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.edges.values())

    def rank_query(self, query_vec: np.ndarray) -> List[Tuple[str, float]]:
        q = np.asarray(query_vec, np.float32).reshape(-1) @ self.query_proj
        qn = float(np.linalg.norm(q)) or 1.0
        sims = (self.embeddings[:, :self.CONTENT_DIM] @ q) / \
            (np.linalg.norm(self.embeddings[:, :self.CONTENT_DIM], axis=1) * qn + 1e-9)
        order = np.argsort(-sims)
        return [(self.nodes[i]["id"], float(sims[i])) for i in order]

    def neighbours_of(self, node_id: str, hops: int = 1) -> set:
        frontier = {node_id}
        seen = set(frontier)
        for _ in range(hops):
            nxt = set()
            for a, b in self.spatial.all_edges():
                if a in frontier and b not in seen:
                    nxt.add(b)
                if b in frontier and a not in seen:
                    nxt.add(a)
            seen |= nxt
            frontier = nxt
        return seen

    def to_json(self) -> Dict[str, Any]:
        """Payload for the frontend graph view."""
        edges = self.spatial.all_edges()
        return {
            "nodes": [{
                "id": n["id"], "label": n["name"], "kind": n["kind"],
                "props": n["props"], "bbox": n["bbox"], "geom": n["geom"],
            } for n in self.nodes],
            "edges": [{"source": e["src"], "target": e["dst"],
                       "rel": e["rel"], "weight": e["weight"]} for e in edges],
        }


@dataclass
class Repository:
    """Everything the grounding stage needs, wired together."""
    spatial: SpatialDB
    embedder: BaseEmbedder
    vector_store: BaseVectorStore
    graph: KnowledgeGraph
    seed_stats: Dict[str, int] = field(default_factory=dict)

    # ── the grounding pass ─────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        top_k = top_k or settings.retrieval_top_k
        qvec = self.embedder.embed_one(query)

        # 1) dense semantic retrieval over report chunks
        chunk_hits = self.vector_store.search(qvec, k=max(top_k * 3, 12))
        evidence: List[Dict[str, Any]] = []
        vec_scores: Dict[str, float] = {}
        for cid, score in chunk_hits:
            rec = self.vector_store.records.get(cid) if isinstance(
                self.vector_store, LocalVectorStore) else None
            if rec is None:
                src = cid.split("#")[0]
                text, name, bbox = "", src, []
            else:
                text, name, bbox = rec.text, rec.source_name, rec.bbox
                src = rec.source_id
            evidence.append({"chunk_id": cid, "source_id": src, "source_name": name,
                             "score": round(float(score), 4), "text": text[:240],
                             "bbox": bbox})
            vec_scores[src] = max(vec_scores.get(src, -1.0), float(score))
        vector_ranking = sorted(vec_scores.items(), key=lambda kv: -kv[1])

        # 2) exact Spatial SQL anchoring from the retrieved report footprints
        spatial_scores: Dict[str, Tuple[int, float]] = {}
        anchors = [e for e in evidence if e["bbox"]][:5]
        for anchor in anchors:
            hits = self._spatial_anchor(anchor["source_id"])
            for fid, dist in hits:
                m, d = spatial_scores.get(fid, (0, float("inf")))
                spatial_scores[fid] = (m + 1, min(d, dist))
        spatial_ranking = sorted(
            spatial_scores.items(),
            key=lambda kv: (-kv[1][0], kv[1][1]))
        spatial_ranking = [(fid, 1.0 / (1.0 + d)) for fid, (m, d) in spatial_ranking]

        # 3) GNN topology traversal ranking
        gnn_ranking = self.graph.rank_query(qvec)

        # 4) Reciprocal Rank Fusion
        fused = rrf_fuse(
            {"vector": vector_ranking, "spatial": spatial_ranking,
             "gnn": gnn_ranking},
            weights=settings.rrf_weights, k=settings.rrf_k, top=top_k)

        for item in fused:
            node = self.spatial.get(item["id"])
            item["name"] = node["name"] if node else item["id"]
            item["kind"] = node["kind"] if node else "unknown"
            item["props"] = node["props"] if node else {}
            item["geom"] = node["geom"] if node else None
            item["bbox"] = node["bbox"] if node else []

        return {
            "query": query,
            "embedding_backend": self.embedder.name,
            "evidence": evidence[: top_k * 2],
            "rankings": {
                "vector": [{"id": i, "score": round(s, 4)} for i, s in vector_ranking[:top_k]],
                "spatial": [{"id": i, "score": round(s, 4)} for i, s in spatial_ranking[:top_k]],
                "gnn": [{"id": i, "score": round(s, 4)} for i, s in gnn_ranking[:top_k]],
            },
            "fused": fused,
        }

    def _spatial_anchor(self, report_id: str) -> List[Tuple[str, float]]:
        """Exact predicate join from a report footprint to physical features,
        widened by one hop through the Spatial SQL neighbour table."""
        out: List[Tuple[str, float]] = []
        report = self.spatial.get(report_id)
        if not report:
            return out
        for pred in ("ST_Intersects",):
            for fid, dist in self.spatial.neighbors(report_id, rel=pred):
                if fid != report_id:
                    out.append((fid, dist))
                # one topology hop (fault-to-aquifer style transitivity)
                for fid2, d2 in self.spatial.neighbors(fid, rel=pred)[:6]:
                    if fid2 != report_id:
                        out.append((fid2, d2 + 0.5))
        # de-duplicate keeping best distance
        best: Dict[str, float] = {}
        for fid, dist in out:
            best[fid] = min(best.get(fid, float("inf")), dist)
        return sorted(best.items(), key=lambda kv: kv[1])

    def graph_payload(self) -> Dict[str, Any]:
        return self.graph.to_json()


# ── singleton wiring ───────────────────────────────────────────────────────
_repo: Optional[Repository] = None
_repo_lock = threading.Lock()


def get_repository() -> Repository:
    global _repo
    with _repo_lock:
        if _repo is None:
            spatial = SpatialDB(settings.spatial_db_path)
            embedder = get_embedder()
            store = get_vector_store()
            stats = seed_database(spatial, embedder, store)
            graph = KnowledgeGraph(spatial, embedder)
            _repo = Repository(spatial=spatial, embedder=embedder,
                               vector_store=store, graph=graph, seed_stats=stats)
        return _repo


def reset_repository() -> None:
    """Test hook — drop the cached singleton."""
    global _repo
    with _repo_lock:
        if _repo is not None:
            _repo.spatial.close()
        _repo = None



