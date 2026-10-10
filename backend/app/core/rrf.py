"""
Reciprocal Rank Fusion (RRF) — Cormack et al. 2009.

Fuses the three grounding rankings:

    vector  — dense semantic retrieval from the HNSW / Qdrant index
    spatial — exact Spatial SQL predicate matches (ST_Intersects & friends)
    gnn     — topology-aware graph embeddings after message passing

score(d) = Σ_source  w_source / (k + rank_source(d))

RRF is scale-free: it needs no score calibration between cosine similarities,
metre distances and graph affinities — only ordinal ranks matter.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Ranked = Sequence[Tuple[str, float]]  # (id, score) sorted best-first


def rrf_fuse(rankings: Dict[str, Ranked], weights: Optional[Dict[str, float]] = None,
             k: int = 60, top: Optional[int] = None) -> List[Dict[str, object]]:
    """Fuse named rankings into a single RRF-scored list.

    Returns items sorted by fused score, each annotated with per-source rank
    and score so the UI can show *why* something was retrieved.
    """
    weights = weights or {name: 1.0 for name in rankings}
    fused: Dict[str, Dict[str, object]] = {}

    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        for rank, (rid, score) in enumerate(ranking, start=1):
            item = fused.setdefault(rid, {"id": rid, "score": 0.0, "sources": {}})
            item["score"] = float(item["score"]) + w / (k + rank)
            item["sources"][name] = {"rank": rank, "score": float(score), "weight": w}

    out = sorted(fused.values(), key=lambda it: -float(it["score"]))
    if top is not None:
        out = out[:top]
    return out


def rank_of(ranking: Ranked, rid: str) -> Optional[int]:
    for i, (x, _) in enumerate(ranking, start=1):
        if x == rid:
            return i
    return None
