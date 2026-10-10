"""
Spatial SQL — PostGIS-flavoured predicates registered into SQLite.

The grounding layer issues *real SQL* such as:

    SELECT b.* FROM features a, features b
    WHERE a.id = :anchor AND ST_Intersects(a.geom, b.geom) = 1

…while the geometry engine underneath is ``app.core.geometry``. This gives
the GNN topology traversal engine deterministic, set-based edge extraction
(``ST_Intersects``, ``ST_Contains``, ``ST_Within``, ``ST_Touches``,
``ST_Overlaps``, ``ST_Distance``) without an external database server.
FTS5 provides the lexical full-text half of hybrid search.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core import geometry as g

_PREDICATES = {
    "ST_Intersects": g.intersects,
    "ST_Contains": g.contains,
    "ST_Within": g.within,
    "ST_Touches": g.touches,
    "ST_Overlaps": g.overlaps,
}


class SpatialDB:
    """SQLite connection with geometry functions + spatial feature tables."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._register_functions()
        self._create_schema()

    # ── setup ──────────────────────────────────────────────────────────────
    def _register_functions(self) -> None:
        for name, fn in _PREDICATES.items():
            self.conn.create_function(name, 2, self._wrap_geom2(fn), deterministic=True)
        self.conn.create_function(
            "ST_Distance", 2, lambda a, b: g.distance(self._load(a), self._load(b)),
            deterministic=True)
        self.conn.create_function(
            "ST_Area", 1, lambda a: g.area_m2(self._load(a)), deterministic=True)
        self.conn.create_function(
            "ST_Length", 1, lambda a: g.length_m(self._load(a)), deterministic=True)
        self.conn.create_function(
            "ST_Centroid", 1,
            lambda a: json.dumps(list(g.centroid(self._load(a)))), deterministic=True)

    @staticmethod
    def _wrap_geom2(fn):
        def inner(a, b):
            return 1 if fn(SpatialDB._load(a), SpatialDB._load(b)) else 0
        return inner

    @staticmethod
    def _load(blob: str) -> Dict:
        return json.loads(blob) if isinstance(blob, str) else blob

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
                geom TEXT NOT NULL, bbox TEXT NOT NULL,
                props TEXT NOT NULL DEFAULT '{}', text TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_features_kind ON features(kind);

            CREATE TABLE IF NOT EXISTS feature_edges (
                src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (src, dst, rel)
            );

            CREATE TABLE IF NOT EXISTS scenes (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS features_fts USING fts5(
                id UNINDEXED, name, text
            );
            """
        )
        self._ensure_fts()

    def _ensure_fts(self) -> None:
        """Rebuild the FTS table if an older contentless schema is present
        (contentless FTS5 cannot DELETE/UPSERT, which seeding requires)."""
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='features_fts'").fetchone()
        if row and row[0] and "content=''" in row[0].replace(" ", ""):
            self.conn.execute("DROP TABLE features_fts")
            row = None
        if row is None:
            self.conn.execute(
                "CREATE VIRTUAL TABLE features_fts USING fts5(id UNINDEXED, name, text)")
        n = self.conn.execute("SELECT count(*) FROM features_fts").fetchone()[0]
        if n == 0:
            rows = self.conn.execute("SELECT id, name, text FROM features").fetchall()
            self.conn.executemany(
                "INSERT INTO features_fts(id, name, text) VALUES(?,?,?)", rows)
        self.conn.commit()

    # ── writes ─────────────────────────────────────────────────────────────
    def upsert_feature(self, fid: str, name: str, kind: str, geom: Dict,
                       props: Optional[Dict] = None, text: str = "") -> None:
        bb = ",".join(str(x) for x in g.bbox(geom))
        self.conn.execute(
            "INSERT INTO features(id,name,kind,geom,bbox,props,text) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind, "
            "geom=excluded.geom, bbox=excluded.bbox, props=excluded.props, text=excluded.text",
            (fid, name, kind, json.dumps(geom), bb, json.dumps(props or {}), text))
        old = self.conn.execute("SELECT rowid FROM features_fts WHERE id=?", (fid,)).fetchone()
        if old:
            self.conn.execute("DELETE FROM features_fts WHERE rowid=?", (old[0],))
        self.conn.execute("INSERT INTO features_fts(id, name, text) VALUES(?,?,?)",
                          (fid, name, text))
        self.conn.commit()

    def upsert_edge(self, src: str, dst: str, rel: str, weight: float = 1.0) -> None:
        self.conn.execute(
            "INSERT INTO feature_edges(src,dst,rel,weight) VALUES(?,?,?,?) "
            "ON CONFLICT(src,dst,rel) DO UPDATE SET weight=excluded.weight",
            (src, dst, rel, weight))
        self.conn.commit()

    # ── reads ──────────────────────────────────────────────────────────────
    def get(self, fid: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM features WHERE id=?", (fid,)).fetchone()
        return self._row(row) if row else None

    def all_features(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM features", ()
        if kind:
            q += " WHERE kind=?"
            args = (kind,)
        return [self._row(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": r["id"], "name": r["name"], "kind": r["kind"],
            "geom": json.loads(r["geom"]), "props": json.loads(r["props"]),
            "text": r["text"], "bbox": [float(x) for x in r["bbox"].split(",")],
        }

    # ── spatial SQL queries ────────────────────────────────────────────────
    def neighbors(self, fid: str, rel: str = "ST_Intersects",
                  exclude_self: bool = True) -> List[Tuple[str, float]]:
        """Run `rel`-predicate join of fid against every other feature."""
        sql = (
            f"SELECT b.id AS id, ST_Distance(a.geom, b.geom) AS d "
            f"FROM features a, features b "
            f"WHERE a.id = :anchor AND b.id != a.id AND {rel}(a.geom, b.geom) = 1 "
            f"ORDER BY d ASC"
        ) if exclude_self else (
            f"SELECT b.id AS id, ST_Distance(a.geom, b.geom) AS d "
            f"FROM features a, features b "
            f"WHERE a.id = :anchor AND {rel}(a.geom, b.geom) = 1 ORDER BY d ASC")
        return [(r["id"], float(r["d"])) for r in
                self.conn.execute(sql, {"anchor": fid})]

    def bbox_candidates(self, minx: float, miny: float, maxx: float,
                        maxy: float) -> List[str]:
        """Cheap bbox pre-filter used before exact predicates
        (portable R-tree emulation over the materialised bbox column)."""
        out = []
        for r in self.conn.execute("SELECT id, bbox FROM features"):
            x0, y0, x1, y1 = (float(v) for v in r["bbox"].split(","))
            if not (x1 < minx or maxx < x0 or y1 < miny or maxy < y0):
                out.append(r["id"])
        return out

    def fulltext(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """FTS5 lexical search (bm25 ranks; lower = better → negated score).
        Tokens are OR-ed so partial matches still surface."""
        import re as _re
        tokens = _re.findall(r"\w+", query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:12])
        try:
            rows = self.conn.execute(
                "SELECT id, bm25(features_fts) AS s FROM features_fts "
                "WHERE features_fts MATCH ? ORDER BY s LIMIT ?",
                (match, limit)).fetchall()
            return [(r["id"], -float(r["s"])) for r in rows]
        except sqlite3.OperationalError:
            return []

    # ── edge materialisation for the GNN ───────────────────────────────────
    def build_spatial_edges(self, predicates: Sequence[str] = ("ST_Intersects", "ST_Contains")) -> int:
        """Recompute typed topological edges with Spatial SQL joins."""
        rel_map = {"ST_Intersects": "intersects", "ST_Contains": "contains",
                   "ST_Within": "overlies", "ST_Touches": "abuts"}
        count = 0
        for pred in predicates:
            rel = rel_map.get(pred, pred.lower())
            rows = self.conn.execute(
                f"SELECT a.id AS src, b.id AS dst FROM features a, features b "
                f"WHERE a.id < b.id AND {pred}(a.geom, b.geom) = 1").fetchall()
            for r in rows:
                self.upsert_edge(r["src"], r["dst"], rel)
                self.upsert_edge(r["dst"], r["src"], rel)
                count += 1
        # faults/units: "overlies" derived from containment of centroids
        rows = self.conn.execute(
            "SELECT a.id AS src, b.id AS dst FROM features a, features b "
            "WHERE a.kind IN ('fault','unit') AND b.kind = 'region' "
            "AND ST_Intersects(a.geom, b.geom) = 1").fetchall()
        for r in rows:
            self.upsert_edge(r["src"], r["dst"], "abuts")
            count += 1
        return count

    def all_edges(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT src, dst, rel, weight FROM feature_edges")]

    # ── scene persistence (used by the API) ────────────────────────────────
    def save_scene(self, scene_id: str, payload: Dict[str, Any], created_at: str) -> None:
        self.conn.execute(
            "INSERT INTO scenes(id,payload,created_at) VALUES(?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (scene_id, json.dumps(payload), created_at))
        self.conn.commit()

    def load_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT payload FROM scenes WHERE id=?",
                                (scene_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_scenes(self) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "created_at": r["created_at"]}
                for r in self.conn.execute(
                    "SELECT id, created_at FROM scenes ORDER BY created_at DESC")]

    def close(self) -> None:
        self.conn.close()

