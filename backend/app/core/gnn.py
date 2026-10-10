"""
Relational GNN — topology traversal engine over the spatial knowledge graph.

Message passing over typed edges (``intersects``, ``contains``, ``overlies``,
``abuts``, ``mentions``):

    m_r(v)  = mean_{u ∈ N_r(v)}  h(u)
    h'(v)   = h(v) + γ · tanh( ( Σ_r m_r(v) W_r ) / R )

The residual form anchors every node to its own content embedding while
neighbours' attributes bleed in along *physical* relationships — a query about
"permeable aquifer near a fault" therefore ranks aquifers that are topologically
adjacent to faults even if their report text never co-occurs.

Weights are seeded (GNN_SEED) and act as a fixed graph filter: no training
loop is required for the retrieval objective, keeping boot time at zero while
still being a genuine multi-layer message-passing network.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

RELATIONS = ("intersects", "contains", "overlies", "abuts", "mentions", "similar")


class RelationalGNN:
    def __init__(self, in_dim: int, relations: Sequence[str] = RELATIONS,
                 layers: int = 2, seed: int = 1337, gamma: float = 0.45):
        self.in_dim = in_dim
        self.relations = tuple(relations)
        self.layers = layers
        self.gamma = gamma
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(in_dim)
        # one projection per (layer, relation) + a self projection
        self.w_rel = [
            {r: (rng.standard_normal((in_dim, in_dim)) * scale).astype(np.float32)
             for r in self.relations}
            for _ in range(layers)
        ]
        self.w_self = [
            (rng.standard_normal((in_dim, in_dim)) * scale).astype(np.float32)
            for _ in range(layers)
        ]

    def propagate(self, features: np.ndarray,
                  edges: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """features: (N, F); edges: relation -> [(src, dst)] (dst receives)."""
        h = features.astype(np.float32).copy()
        n = h.shape[0]
        for layer in range(self.layers):
            agg = np.zeros_like(h)
            counts = np.zeros((n, 1), dtype=np.float32)
            for rel in self.relations:
                pairs = edges.get(rel, [])
                if not pairs:
                    continue
                src = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
                dst = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
                np.add.at(agg, dst, h[src] @ self.w_rel[layer][rel])
                np.add.at(counts, dst, 1.0)
            counts = np.maximum(counts, 1.0)
            agg /= counts
            update = agg + h @ self.w_self[layer]
            h = h + self.gamma * np.tanh(update / np.sqrt(2.0))
        return h

    def normalised(self, features: np.ndarray,
                   edges: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        h = self.propagate(features, edges)
        norms = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        return h / norms
