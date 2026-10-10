"""
Seed corpus — a synthetic geological survey of the Karakoram–Indus frontal
zone (lon 71–77°E, lat 31–36°N).

Everything the grounding layer knows at boot lives here:

* ``REGIONS``    — tectonic provinces (polygons)
* ``FAULTS``     — named fault traces (linestrings) with slip/rake props
* ``AQUIFERS``   — hydrogeological units (polygons)
* ``UNITS``      — lithological formations (polygons)
* ``BOREHOLES``  — point logs with depth/lithology/permeability
* ``REPORTS``    — unstructured survey-report text (chunked into the
                   vector index) carrying bbox anchors for spatial SQL

Physical props (elevation, relief, hardness, rainfall, uplift, permeability,
vp) feed the conditioning rasteriser that steers the latent engine.
"""
from __future__ import annotations

from typing import Any, Dict, List

REGION = "region"
FAULT = "fault"
AQUIFER = "aquifer"
UNIT = "unit"
BOREHOLE = "borehole"
REPORT = "report"

REGIONS: List[Dict[str, Any]] = [
    {"id": "reg-karakoram", "name": "Karakoram Fold-Thrust Belt", "geom":
     {"type": "Polygon", "coordinates": [[74.0, 35.9], [76.6, 35.7], [76.9, 34.6],
                                         [75.2, 34.2], [73.8, 34.8], [74.0, 35.9]]},
     "props": {"elevation_m": 4800, "relief_m": 3200, "hardness": 7.2,
               "rainfall_mm": 320, "uplift_mm_yr": 7.5, "roughness": 0.82,
               "vp_ms": 5900, "permeability_mD": 40}},
    {"id": "reg-indus", "name": "Indus Suture Zone", "geom":
     {"type": "Polygon", "coordinates": [[72.4, 35.2], [74.4, 35.4], [74.6, 34.3],
                                         [73.9, 33.4], [72.2, 33.7], [72.4, 35.2]]},
     "props": {"elevation_m": 2600, "relief_m": 1900, "hardness": 5.8,
               "rainfall_mm": 480, "uplift_mm_yr": 4.1, "roughness": 0.66,
               "vp_ms": 5400, "permeability_mD": 120}},
    {"id": "reg-chitral", "name": "Chitral Accretionary Prism", "geom":
     {"type": "Polygon", "coordinates": [[71.2, 36.0], [72.6, 35.9], [72.5, 34.6],
                                         [71.0, 34.4], [71.2, 36.0]]},
     "props": {"elevation_m": 3100, "relief_m": 2400, "hardness": 4.9,
               "rainfall_mm": 720, "uplift_mm_yr": 5.6, "roughness": 0.74,
               "vp_ms": 5100, "permeability_mD": 260}},
    {"id": "reg-potwar", "name": "Potwar–Kohistan Forearc", "geom":
     {"type": "Polygon", "coordinates": [[72.0, 33.6], [73.8, 33.5], [73.7, 32.4],
                                         [72.1, 32.5], [72.0, 33.6]]},
     "props": {"elevation_m": 900, "relief_m": 800, "hardness": 3.6,
               "rainfall_mm": 950, "uplift_mm_yr": 2.2, "roughness": 0.41,
               "vp_ms": 4700, "permeability_mD": 540}},
    {"id": "reg-gilgit", "name": "Gilgit Gneiss Dome", "geom":
     {"type": "Polygon", "coordinates": [[74.2, 36.1], [75.4, 36.2], [75.5, 35.4],
                                         [74.3, 35.3], [74.2, 36.1]]},
     "props": {"elevation_m": 3900, "relief_m": 2700, "hardness": 8.1,
               "rainfall_mm": 260, "uplift_mm_yr": 6.3, "roughness": 0.77,
               "vp_ms": 6300, "permeability_mD": 15}},
]

FAULTS: List[Dict[str, Any]] = [
    {"id": "flt-mkt", "name": "Main Karakoram Thrust", "geom":
     {"type": "LineString", "coordinates": [[73.6, 35.9], [74.4, 35.5], [75.1, 35.2],
                                            [75.9, 34.9], [76.7, 34.6]]},
     "props": {"dip_deg": 32, "slip_mm_yr": 9.0, "rake_deg": 110, "seismicity": "high"}},
    {"id": "flt-mbt", "name": "Main Boundary Thrust", "geom":
     {"type": "LineString", "coordinates": [[72.1, 34.1], [72.9, 33.7], [73.6, 33.3],
                                            [74.3, 33.0]]},
     "props": {"dip_deg": 25, "slip_mm_yr": 4.5, "rake_deg": 95, "seismicity": "moderate"}},
    {"id": "flt-chaman", "name": "Chaman Transform Fault", "geom":
     {"type": "LineString", "coordinates": [[71.4, 35.6], [71.6, 34.8], [71.8, 34.0],
                                            [72.0, 33.2]]},
     "props": {"dip_deg": 88, "slip_mm_yr": 12.0, "rake_deg": 178, "seismicity": "high"}},
    {"id": "flt-tsangpo", "name": "Indus–Tsangpo Suture Fault", "geom":
     {"type": "LineString", "coordinates": [[72.6, 35.1], [73.5, 34.8], [74.2, 34.4],
                                            [75.0, 34.1]]},
     "props": {"dip_deg": 41, "slip_mm_yr": 3.2, "rake_deg": 60, "seismicity": "low"}},
    {"id": "flt-besham", "name": "Besham Duplex Fault", "geom":
     {"type": "LineString", "coordinates": [[72.8, 34.9], [73.2, 34.5], [73.5, 34.1]]},
     "props": {"dip_deg": 47, "slip_mm_yr": 2.1, "rake_deg": 84, "seismicity": "moderate"}},
    {"id": "flt-gilgit", "name": "Gilgit Detachment", "geom":
     {"type": "LineString", "coordinates": [[74.4, 35.9], [74.9, 35.7], [75.3, 35.5]]},
     "props": {"dip_deg": 18, "slip_mm_yr": 1.4, "rake_deg": 20, "seismicity": "low"}},
    {"id": "flt-kunhar", "name": "Kunhar River Fault", "geom":
     {"type": "LineString", "coordinates": [[73.1, 33.9], [73.6, 33.6], [74.1, 33.4]]},
     "props": {"dip_deg": 60, "slip_mm_yr": 1.8, "rake_deg": 130, "seismicity": "moderate"}},
    {"id": "flt-nang", "name": "Nanga Parbat Thrust Front", "geom":
     {"type": "LineString", "coordinates": [[74.0, 35.2], [74.1, 34.7], [74.3, 34.2],
                                            [74.5, 33.7]]},
     "props": {"dip_deg": 36, "slip_mm_yr": 7.1, "rake_deg": 102, "seismicity": "high"}},
]

AQUIFERS: List[Dict[str, Any]] = [
    {"id": "aqn-indus-valley", "name": "Indus Valley Alluvial Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[72.6, 34.6], [73.6, 34.5], [73.7, 33.8],
                                         [72.7, 33.9], [72.6, 34.6]]},
     "props": {"thickness_m": 85, "transmissivity_mDm": 3200, "storage": 0.22,
               "water_table_m": 14, "quality": "fresh"}},
    {"id": "aqn-kohistan", "name": "Kohistan Fractured Rock Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[73.4, 35.2], [74.3, 35.1], [74.4, 34.5],
                                         [73.5, 34.6], [73.4, 35.2]]},
     "props": {"thickness_m": 240, "transmissivity_mDm": 480, "storage": 0.011,
               "water_table_m": 61, "quality": "mineralised"}},
    {"id": "aqn-potwar", "name": "Potwar Sedimentary Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[72.3, 33.4], [73.4, 33.3], [73.3, 32.6],
                                         [72.4, 32.7], [72.3, 33.4]]},
     "props": {"thickness_m": 150, "transmissivity_mDm": 2100, "storage": 0.18,
               "water_table_m": 22, "quality": "brackish"}},
    {"id": "aqn-gilgit", "name": "Gilgit Glacial Melt Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[74.5, 35.9], [75.3, 35.8], [75.3, 35.3],
                                         [74.6, 35.4], [74.5, 35.9]]},
     "props": {"thickness_m": 60, "transmissivity_mDm": 900, "storage": 0.09,
               "water_table_m": 38, "quality": "fresh"}},
    {"id": "aqn-chitral", "name": "Chitral Terrace Gravel Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[71.4, 35.6], [72.2, 35.5], [72.2, 34.9],
                                         [71.5, 35.0], [71.4, 35.6]]},
     "props": {"thickness_m": 110, "transmissivity_mDm": 2700, "storage": 0.19,
               "water_table_m": 18, "quality": "fresh"}},
    {"id": "aqn-skar", "name": "Skardu Karst Aquifer", "geom":
     {"type": "Polygon", "coordinates": [[75.4, 35.4], [76.1, 35.3], [76.1, 34.8],
                                         [75.5, 34.9], [75.4, 35.4]]},
     "props": {"thickness_m": 320, "transmissivity_mDm": 6400, "storage": 0.31,
               "water_table_m": 44, "quality": "fresh"}},
]

UNITS: List[Dict[str, Any]] = [
    {"id": "unit-gilgit-gneiss", "name": "Gilgit Paragneiss Unit", "geom":
     {"type": "Polygon", "coordinates": [[74.4, 36.0], [75.2, 36.0], [75.2, 35.5],
                                         [74.5, 35.5], [74.4, 36.0]]},
     "props": {"lithology": "gneiss", "hardness": 8.0, "age_ma": 40, "vp_ms": 6400}},
    {"id": "unit-chilas", "name": "Chilas Mafic Complex", "geom":
     {"type": "Polygon", "coordinates": [[74.0, 35.6], [74.8, 35.5], [74.8, 35.0],
                                         [74.1, 35.1], [74.0, 35.6]]},
     "props": {"lithology": "gabbro", "hardness": 7.4, "age_ma": 100, "vp_ms": 6600}},
    {"id": "unit-kohistan", "name": "Kohistan Island Arc Granitoids", "geom":
     {"type": "Polygon", "coordinates": [[73.2, 35.1], [74.0, 35.0], [74.0, 34.5],
                                         [73.3, 34.6], [73.2, 35.1]]},
     "props": {"lithology": "granodiorite", "hardness": 6.9, "age_ma": 50, "vp_ms": 6100}},
    {"id": "unit-slate", "name": "Chitral Slate Belt", "geom":
     {"type": "Polygon", "coordinates": [[71.3, 35.8], [72.1, 35.7], [72.1, 35.1],
                                         [71.4, 35.2], [71.3, 35.8]]},
     "props": {"lithology": "slate", "hardness": 4.2, "age_ma": 200, "vp_ms": 5200}},
    {"id": "unit-murree", "name": "Murree Red Bed Formation", "geom":
     {"type": "Polygon", "coordinates": [[72.4, 33.5], [73.3, 33.4], [73.3, 32.8],
                                         [72.5, 32.9], [72.4, 33.5]]},
     "props": {"lithology": "sandstone", "hardness": 3.8, "age_ma": 30, "vp_ms": 4600}},
    {"id": "unit-siwalik", "name": "Siwalik Conglomerate", "geom":
     {"type": "Polygon", "coordinates": [[72.6, 32.7], [73.6, 32.6], [73.6, 32.2],
                                         [72.7, 32.3], [72.6, 32.7]]},
     "props": {"lithology": "conglomerate", "hardness": 3.1, "age_ma": 12, "vp_ms": 4300}},
]

BOREHOLES: List[Dict[str, Any]] = [
    {"id": "bh-skardu-01", "name": "Skardu Deep Survey BH-01", "geom":
     {"type": "Point", "coordinates": [75.6, 35.3]},
     "props": {"depth_m": 420, "lithology": "limestone", "perm_mD": 180,
               "elevation_m": 2230, "vp_ms": 5300}},
    {"id": "bh-gilgit-02", "name": "Gilgit Geothermal BH-02", "geom":
     {"type": "Point", "coordinates": [74.9, 35.9]},
     "props": {"depth_m": 780, "lithology": "gneiss", "perm_mD": 12,
               "elevation_m": 2450, "vp_ms": 6200}},
    {"id": "bh-potwar-03", "name": "Potwar Groundwater BH-03", "geom":
     {"type": "Point", "coordinates": [72.9, 33.1]},
     "props": {"depth_m": 160, "lithology": "sandstone", "perm_mD": 640,
               "elevation_m": 510, "vp_ms": 4500}},
    {"id": "bh-chitral-04", "name": "Chitral Mineral Spring BH-04", "geom":
     {"type": "Point", "coordinates": [71.7, 35.3]},
     "props": {"depth_m": 240, "lithology": "slate", "perm_mD": 95,
               "elevation_m": 1480, "vp_ms": 5100}},
    {"id": "bh-indus-05", "name": "Indus Dam Axis BH-05", "geom":
     {"type": "Point", "coordinates": [73.1, 34.4]},
     "props": {"depth_m": 310, "lithology": "gabbro", "perm_mD": 25,
               "elevation_m": 1060, "vp_ms": 6000}},
    {"id": "bh-nanga-06", "name": "Nanga Parbat Massif BH-06", "geom":
     {"type": "Point", "coordinates": [74.2, 34.8]},
     "props": {"depth_m": 550, "lithology": "granodiorite", "perm_mD": 40,
               "elevation_m": 3100, "vp_ms": 6300}},
    {"id": "bh-kunhar-07", "name": "Kunhar River Alluvial BH-07", "geom":
     {"type": "Point", "coordinates": [73.5, 33.7]},
     "props": {"depth_m": 95, "lithology": "alluvium", "perm_mD": 1200,
               "elevation_m": 640, "vp_ms": 3900}},
    {"id": "bh-skar-08", "name": "Skardu Karst Probe BH-08", "geom":
     {"type": "Point", "coordinates": [75.8, 35.1]},
     "props": {"depth_m": 360, "lithology": "limestone", "perm_mD": 890,
               "elevation_m": 2510, "vp_ms": 5400}},
    {"id": "bh-besham-09", "name": "Besham Structural BH-09", "geom":
     {"type": "Point", "coordinates": [73.3, 34.7]},
     "props": {"depth_m": 500, "lithology": "schist", "perm_mD": 55,
               "elevation_m": 1320, "vp_ms": 5700}},
    {"id": "bh-dasu-10", "name": "Dasu Dam Foundation BH-10", "geom":
     {"type": "Point", "coordinates": [73.6, 34.1]},
     "props": {"depth_m": 610, "lithology": "granite", "perm_mD": 18,
               "elevation_m": 890, "vp_ms": 6500}},
]

