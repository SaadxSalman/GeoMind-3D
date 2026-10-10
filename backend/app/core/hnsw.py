"""
HNSW — Hierarchical Navigable Small World graph index (NumPy implementation).

Implements Malkov & Yashunin (2018) with:
* multi-layer graph with exponential level assignment,
* greedy descending search on upper layers,
* ef-construction candidate selection with the select-nearest heuristic,
* cosine similarity on L2-normalised vectors (stored as inner product),
* npz persistence.

Expected search complexity O(log N) keeps grounding latency flat as the
spatial corpus grows.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class HNSWIndex:
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 100,
                 ef_search: int = 64, seed: int = 0):
        self.dim = dim
        self.m = m
        self.m_max0 = m * 2          # denser base layer, as per the paper
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._rng = np.random.default_rng(seed)
        self._level_mult = 1.0 / math.log(max(m, 2))

        self.vectors: List[np.ndarray] = []          # unit vectors
        self.ids: List[str] = []                     # external ids
        self.levels: List[int] = []
        self.neighbors: List[List[List[int]]] = []   # [node][level] -> ids
        self.entry: int = -1
        self.max_level: int = -1
        self._id_to_pos: Dict[str, int] = {}

    # ── metrics ────────────────────────────────────────────────────────────
    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))          # cosine on normalised vectors

    @staticmethod
    def normalise(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    # ── insertion ──────────────────────────────────────────────────────────
    def _random_level(self) -> int:
        u = self._rng.random()
        return int(-math.log(max(u, 1e-12)) * self._level_mult)

    def add(self, vector: np.ndarray, external_id: str) -> int:
        if external_id in self._id_to_pos:   # upsert
            pos = self._id_to_pos[external_id]
            self.vectors[pos] = self.normalise(np.asarray(vector, np.float32))
            return pos
        v = self.normalise(np.asarray(vector, np.float32).reshape(-1))
        if v.shape[0] != self.dim:
            raise ValueError(f"dim mismatch: {v.shape[0]} != {self.dim}")
        pos = len(self.vectors)
        level = self._random_level()

        self.vectors.append(v)
        self.ids.append(external_id)
        self.levels.append(level)
        self.neighbors.append([[] for _ in range(level + 1)])
        self._id_to_pos[external_id] = pos

        if self.entry < 0:                   # first node becomes entry point
            self.entry = pos
            self.max_level = level
            return pos

        ep = self.entry
        for lc in range(self.max_level, level, -1):     # greedy descend
            ep = self._greedy(v, ep, lc)
        for lc in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(v, ep, self.ef_construction, lc)
            chosen = self._select_heuristic(candidates, self.m)
            self.neighbors[pos][lc] = [n for _, n in chosen]
            m_cap = self.m_max0 if lc == 0 else self.m
            for _, n in chosen:                          # reciprocal links
                nb = self.neighbors[n]
                if lc < len(nb):
                    nb[lc].append(pos)
                    if len(nb[lc]) > m_cap:
                        pruned = self._select_heuristic(
                            [(self._sim(self.vectors[n], self.vectors[x]), x)
                             for x in nb[lc]], m_cap)
                        nb[lc] = [x for _, x in pruned]
            if candidates:
                ep = max(candidates, key=lambda t: t[0])[1]
        if level > self.max_level:
            self.max_level = level
            self.entry = pos
        return pos

    def add_batch(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        for vec, ext_id in zip(vectors, ids):
            self.add(vec, ext_id)

    def _greedy(self, v: np.ndarray, ep: int, level: int) -> int:
        cur, cur_sim = ep, self._sim(v, self.vectors[ep])
        improved = True
        while improved:
            improved = False
            adj = self.neighbors[cur][level] if level < len(self.neighbors[cur]) else []
            for n in adj:
                s = self._sim(v, self.vectors[n])
                if s > cur_sim:
                    cur, cur_sim, improved = n, s, True
        return cur

    # ── layer search ───────────────────────────────────────────────────────
    def _search_layer(self, v: np.ndarray, ep: int, ef: int, level: int) -> List[Tuple[float, int]]:
        visited = {ep}
        ep_sim = self._sim(v, self.vectors[ep])
        candidates: List[Tuple[float, int]] = [(-ep_sim, ep)]  # min-heap on sim
        best: List[Tuple[float, int]] = [(ep_sim, ep)]          # worst on top
        heapq.heapify(best)
        while candidates:
            neg_s, n = heapq.heappop(candidates)
            if -neg_s < best[0][0]:
                break
            adj = self.neighbors[n][level] if level < len(self.neighbors[n]) else []
            for nb in adj:
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._sim(v, self.vectors[nb])
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(best, (s, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(best, key=lambda t: -t[0])

    def _select_heuristic(self, candidates: Sequence[Tuple[float, int]],
                          m: int) -> List[Tuple[float, int]]:
        """Malkov's select-nearest heuristic: prefer well-spread neighbours."""
        ordered = sorted(candidates, key=lambda t: -t[0])
        good: List[Tuple[float, int]] = []
        for s_new, n_new in ordered:
            if len(good) >= m:
                break
            v_new = self.vectors[n_new]
            if any(self._sim(v_new, self.vectors[n_g]) > s_new for _, n_g in good):
                continue
            good.append((s_new, n_new))
        if len(good) < min(m, len(ordered)):     # degree-stability fallback
            have = {n for _, n in good}
            for item in ordered:
                if len(good) >= m:
                    break
                if item[1] not in have:
                    good.append(item)
                    have.add(item[1])
        return good

    # ── public search ──────────────────────────────────────────────────────
    def search(self, vector: np.ndarray, k: int = 8,
               ef: Optional[int] = None) -> List[Tuple[str, float]]:
        if self.entry < 0:
            return []
        v = self.normalise(np.asarray(vector, np.float32).reshape(-1))
        ef = max(ef or self.ef_search, k)
        ep = self.entry
        for lc in range(self.max_level, 0, -1):
            ep = self._greedy(v, ep, lc)
        results = self._search_layer(v, ep, ef, 0)
        return [(self.ids[n], s) for s, n in results[:k]]

    def __len__(self) -> int:
        return len(self.vectors)

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        max_lvl = max(self.max_level, 0)
        link = np.full((len(self.vectors), max_lvl + 1, self.m_max0), -1, dtype=np.int32)
        for n, levels in enumerate(self.neighbors):
            for lc, adj in enumerate(levels):
                link[n, lc, :len(adj)] = adj[: self.m_max0]
        np.savez_compressed(
            path,
            vectors=np.asarray(self.vectors, dtype=np.float32),
            ids=np.asarray(self.ids),
            levels=np.asarray(self.levels, dtype=np.int32),
            links=link,
            entry=np.asarray(self.entry, dtype=np.int64),
            max_level=np.asarray(self.max_level, dtype=np.int64),
            meta=np.asarray([self.dim, self.m, self.ef_construction, self.ef_search],
                            dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        data = np.load(path, allow_pickle=False)
        dim, m, efc, efs = (int(x) for x in data["meta"])
        idx = cls(dim, m=m, ef_construction=efc, ef_search=efs)
        idx.vectors = [np.asarray(v, np.float32) for v in data["vectors"]]
        idx.ids = [str(i) for i in data["ids"]]
        idx.levels = [int(l) for l in data["levels"]]
        link = data["links"]
        idx.neighbors = [
            [list(int(x) for x in link[n, lc] if int(x) >= 0)
             for lc in range(int(idx.levels[n]) + 1)]
            for n in range(len(idx.ids))
        ]
        idx.entry = int(data["entry"])
        idx.max_level = int(data["max_level"])
        idx._id_to_pos = {i: p for p, i in enumerate(idx.ids)}
        return idx

