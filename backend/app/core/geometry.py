"""
Planar spatial predicates — the computational core of the Spatial SQL layer.

Geometries are GeoJSON-like dicts on lon/lat coordinates:

    {"type": "Point",      "coordinates": [lon, lat]}
    {"type": "LineString", "coordinates": [[lon, lat], ...]}
    {"type": "Polygon",    "coordinates": [[lon, lat], ...]}   # single ring

Distances/areas use an equirectangular approximation scaled by cos(lat0)
accurate to <0.5% at regional (≤500 km) scales — plenty for grounding, and
it keeps the engine dependency-free.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Geom = Dict
Coord = Tuple[float, float]

EARTH_RADIUS_M = 6_371_008.8


def bbox(geom: Geom) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy)."""
    xs, ys = _coords(geom)
    return (min(xs), min(ys), max(xs), max(ys))


def _coords(geom: Geom) -> Tuple[List[float], List[float]]:
    t, c = geom["type"], geom["coordinates"]
    if t == "Point":
        return [c[0]], [c[1]]
    if t == "LineString":
        return [p[0] for p in c], [p[1] for p in c]
    if t == "Polygon":
        ring = _ring(geom)
        return [p[0] for p in ring], [p[1] for p in ring]
    raise ValueError(f"unsupported geometry type: {t}")


def _ring(geom: Geom) -> List[Coord]:
    """Polygon ring, tolerating both GeoJSON ([ring]) and bare-ring forms."""
    if geom["type"] != "Polygon":
        return []
    c = geom["coordinates"]
    if c and isinstance(c[0][0], (int, float)):
        pts = c                      # coordinates: [[x,y], ...]
    else:
        pts = c[0]                   # coordinates: [[[x,y], ...]]
    ring = [tuple(p) for p in pts]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _line(geom: Geom) -> List[Coord]:
    return [tuple(p) for p in geom["coordinates"]] if geom["type"] == "LineString" else []


def bbox_intersects(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_in_ring(pt: Coord, ring: Sequence[Coord]) -> bool:
    """Ray casting; points on the boundary count as inside."""
    x, y = pt
    inside = False
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 and \
           min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if abs(cross) < 1e-12:
                return True
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def _seg_pt(p: Coord, q: Coord, pt: Coord) -> bool:
    cross = (q[0] - p[0]) * (pt[1] - p[1]) - (q[1] - p[1]) * (pt[0] - p[0])
    if abs(cross) > 1e-12:
        return False
    return (min(p[0], q[0]) - 1e-12 <= pt[0] <= max(p[0], q[0]) + 1e-12 and
            min(p[1], q[1]) - 1e-12 <= pt[1] <= max(p[1], q[1]) + 1e-12)


def _segments_intersect(p1: Coord, p2: Coord, p3: Coord, p4: Coord) -> bool:
    def orient(a: Coord, b: Coord, c: Coord) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    def on(a: Coord, b: Coord, c: Coord) -> bool:
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12 and
                abs(orient(a, b, c)) < 1e-12)
    return on(p1, p2, p3) or on(p1, p2, p4) or on(p3, p4, p1) or on(p3, p4, p2)


def _points(geom: Geom) -> List[Coord]:
    if geom["type"] == "Point":
        return [tuple(geom["coordinates"])]
    if geom["type"] == "LineString":
        return _line(geom)
    if geom["type"] == "Polygon":
        return _ring(geom)
    return []


def _dim(geom: Geom) -> int:
    return {"Point": 0, "LineString": 1, "Polygon": 2}[geom["type"]]


def _pairs(pts: Sequence[Coord]) -> List[Tuple[Coord, Coord]]:
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _line_in_poly(line: Geom, poly: Geom) -> bool:
    ring = _ring(poly)
    pts = _line(line)
    if any(_point_in_ring(p, ring) for p in pts):
        return True
    return any(_segments_intersect(p, q, r, s) for p, q in _pairs(pts) for r, s in _pairs(ring))


def intersects(a: Geom, b: Geom) -> bool:
    if not bbox_intersects(bbox(a), bbox(b)):
        return False
    da, db = _dim(a), _dim(b)
    if da > db:  # order by dimension: point ⊂ line ⊂ polygon
        a, b, da, db = b, a, db, da
    if da == 0 and db == 0:
        return abs(a["coordinates"][0] - b["coordinates"][0]) < 1e-12 and \
               abs(a["coordinates"][1] - b["coordinates"][1]) < 1e-12
    if da == 0 and db == 1:
        return any(_seg_pt(p, q, tuple(a["coordinates"])) for p, q in _pairs(_line(b)))
    if da == 0 and db == 2:
        return _point_in_ring(tuple(a["coordinates"]), _ring(b))
    if da == 1 and db == 1:
        return any(_segments_intersect(p, q, r, s)
                   for p, q in _pairs(_line(a)) for r, s in _pairs(_line(b)))
    if da == 1 and db == 2:
        return _line_in_poly(a, b)
    ra, rb = _ring(a), _ring(b)  # polygon/polygon
    if any(_point_in_ring(p, rb) for p in ra) or any(_point_in_ring(p, ra) for p in rb):
        return True
    return any(_segments_intersect(p, q, r, s) for p, q in _pairs(ra) for r, s in _pairs(rb))


def contains(a: Geom, b: Geom) -> bool:
    """True when b lies entirely inside a (boundary counts as inside)."""
    if a["type"] != "Polygon" or not bbox_intersects(bbox(a), bbox(b)):
        return False
    pts = _points(b)
    if not pts or not all(_point_in_ring(p, _ring(a)) for p in pts):
        return False
    if b["type"] == "Polygon":  # edges must not leave a
        ra = _ring(a)
        for p, q in _pairs(_ring(b)):
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if not _point_in_ring(mid, ra):
                return False
    return True


def within(a: Geom, b: Geom) -> bool:
    return contains(b, a)


def _interiors_overlap(a: Geom, b: Geom) -> bool:
    if a["type"] == "Polygon" and b["type"] == "Polygon":
        ra, rb = _ring(a), _ring(b)
        if any(_point_in_ring(p, rb) for p in ra[:-1]):
            return True
        if any(_point_in_ring(p, ra) for p in rb[:-1]):
            return True
        return False
    if a["type"] == "Point":
        return contains(b, a)
    if b["type"] == "Point":
        return contains(a, b)
    pts = _points(a)
    return any(contains(b, {"type": "Point", "coordinates": list(p)}) for p in pts[:1])


def touches(a: Geom, b: Geom) -> bool:
    return intersects(a, b) and not _interiors_overlap(a, b)


def overlaps(a: Geom, b: Geom) -> bool:
    if a["type"] != "Polygon" or b["type"] != "Polygon":
        return False
    return intersects(a, b) and not contains(a, b) and not contains(b, a)


def distance(a: Geom, b: Geom) -> float:
    """Approximate geodesic distance in metres (equirectangular)."""
    pts_a, pts_b = _points(a), _points(b)
    if not pts_a or not pts_b:
        return float("inf")
    lat0 = math.radians((bbox(a)[1] + bbox(a)[3] + bbox(b)[1] + bbox(b)[3]) / 4)

    def m(p: Coord) -> Coord:
        return (math.radians(p[0]) * math.cos(lat0) * EARTH_RADIUS_M,
                math.radians(p[1]) * EARTH_RADIUS_M)

    best = float("inf")
    samples_a = _resample(pts_a, _dim(a) >= 1)
    samples_b = _resample(pts_b, _dim(b) >= 1)
    for p in samples_a:
        pm = m(p)
        for q in samples_b:
            qm = m(q)
            d = math.hypot(pm[0] - qm[0], pm[1] - qm[1])
            if d < best:
                best = d
    return best


def _resample(pts: Sequence[Coord], dense: bool) -> List[Coord]:
    if not dense or len(pts) < 2:
        return list(pts)
    out: List[Coord] = []
    for p, q in _pairs(pts):
        out.append(p)
        out.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
    out.append(pts[-1])
    return out


def area_m2(geom: Geom) -> float:
    """Polygon area in m² (shoelace × cos(lat0) metric scale)."""
    if geom["type"] != "Polygon":
        return 0.0
    ring = _ring(geom)
    lat0 = math.radians((bbox(geom)[1] + bbox(geom)[3]) / 2)
    mx = math.cos(lat0) * EARTH_RADIUS_M * math.pi / 180.0
    my = EARTH_RADIUS_M * math.pi / 180.0
    s = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0] * mx, ring[i][1] * my
        x2, y2 = ring[i + 1][0] * mx, ring[i + 1][1] * my
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def length_m(geom: Geom) -> float:
    """LineString length in metres."""
    if geom["type"] != "LineString":
        return 0.0
    pts = _line(geom)
    return sum(_seg_len(p, q) for p, q in _pairs(pts))


def _seg_len(p: Coord, q: Coord) -> float:
    lat0 = math.radians((p[1] + q[1]) / 2)
    dx = math.radians(q[0] - p[0]) * math.cos(lat0) * EARTH_RADIUS_M
    dy = math.radians(q[1] - p[1]) * EARTH_RADIUS_M
    return math.hypot(dx, dy)


def centroid(geom: Geom) -> Coord:
    xs, ys = _coords(geom)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def buffer(geom: Geom, metres: float, quad_segs: int = 16) -> Geom:
    """Approximate buffer: convex-hull-of-offset-points for lines/points,
    outward offset for polygons (bbox-safe implementation)."""
    lat0 = math.radians(bbox(geom)[1])
    dlat = math.degrees(metres / EARTH_RADIUS_M)
    dlon = dlat / max(math.cos(lat0), 1e-6)
    cx, cy = centroid(geom)
    if geom["type"] == "Polygon":
        ring = _ring(geom)
        pts = []
        for x, y in ring:
            vx, vy = x - cx, y - cy
            n = math.hypot(vx, vy) or 1.0
            pts.append((x + vx / n * dlon, y + vy / n * dlat))
        pts.append(pts[0])
        return {"type": "Polygon", "coordinates": [pts]}
    ring = []
    for i in range(quad_segs * 2):
        ang = 2 * math.pi * i / (quad_segs * 2)
        ring.append((cx + math.cos(ang) * dlon, cy + math.sin(ang) * dlat))
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}

