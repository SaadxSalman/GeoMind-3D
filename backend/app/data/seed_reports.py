"""
Seed survey reports — unstructured multi-modal text documents.

Each report carries a bbox anchor so the Spatial SQL layer can bind report
chunks to physical features (report → region/fault mentions become graph
edges via ``ST_Intersects`` + name matching). Reports are chunked into the
vector index by ``knowledge_graph.seed_database``.
"""
from __future__ import annotations

from typing import Any, Dict, List

REPORTS: List[Dict[str, Any]] = [
    {
        "id": "rep-karakoram-2024",
        "name": "Karakoram Fold-Thrust Belt Geotechnical Survey 2024",
        "bbox": [73.8, 34.2, 76.9, 35.9],
        "text": (
            "The Karakoram Fold-Thrust Belt records ongoing collisional deformation north of the "
            "Indus Suture Zone. Field mapping along the Gilgit–Skardu transect identifies the Main "
            "Karakoram Thrust as the dominant crustal-scale structure, dipping 32 degrees south with "
            "cumulative slip near 9 mm per year. Ridges composed of the Chilas Mafic Complex and "
            "Gilgit Paragneiss Unit reach elevations above 4800 metres, with local relief exceeding "
            "3200 metres across deeply incised gorges.\n"
            "Aerial photograph interpretation and LiDAR differencing show that fluvial incision by the "
            "Indus River keeps pace with tectonic uplift over the last 1.2 million years. Landslide "
            "scars cluster within one kilometre of the Main Karakoram Thrust trace, indicating that "
            "fault damage zones control slope instability. The Gilgit Detachment accommodates late "
            "normal-mode extension above the gneiss dome.\n"
            "Borehole BH-02 near Gilgit penetrated 780 metres of fractured gneiss with low primary "
            "permeability of 12 millidarcy, while karst windows in the Skardu limestone host a highly "
            "transmissive aquifer. Seismic refraction gives P-wave velocities of 6100 to 6600 metres "
            "per second through the mafic core complex."
        ),
    },
    {
        "id": "rep-indus-suture",
        "name": "Indus Suture Zone Lithospheric Structure Report",
        "bbox": [72.2, 33.4, 74.6, 35.4],
        "text": (
            "The Indus Suture Zone separates the Kohistan Island Arc from the Karakoram block and is "
            "exposed as a 180 kilometre wide belt of ophiolitic mélange, serpentinite and arc "
            "granitoids. The Kohistan Island Arc Granitoids yield crystallisation ages near 50 Ma and "
            "seismic velocities around 6100 metres per second. The Indus–Tsangpo Suture Fault bounds "
            "the belt on the south with a 41 degree dip and oblique slip of 3.2 millimetres per year.\n"
            "Geomorphometric analysis of the 30 metre digital elevation model gives a mean slope of "
            "27 degrees and roughness indices of 0.66. Channel steepness indices spike across the "
            "suture, consistent with spatially variable uplift of 4.1 millimetres per year. Thermal "
            "erosion modelling at a talus angle of 34 degrees reproduces the observed ridge profiles "
            "after 24 thousand annual steps.\n"
            "Groundwater in the fractured rock aquifer is mineralised, with transmissivity near 480 "
            "millidarcy metres. The Besham Duplex Fault and Kunhar River Fault partition shortening "
            "into imbricate slices that focus aquifer recharge along fault-permeable corridors."
        ),
    },
    {
        "id": "rep-potwar-forearc",
        "name": "Potwar–Kohistan Forearc Hydrogeology Memoir",
        "bbox": [72.0, 32.2, 73.8, 33.6],
        "text": (
            "The Potwar–Kohistan Forearc is a low-relief plateau built on the Murree Red Bed "
            "Formation and Siwalik Conglomerate. Mean elevation is 900 metres with relief near 800 "
            "metres, and rainfall reaches 950 millimetres per year monsoon-dominated. Soft sandstones "
            "and conglomerates have bulk hardness between 3.1 and 3.8 Mohs, producing smooth, low "
            "roughness terrain with an index of 0.41.\n"
            "The Potwar Sedimentary Aquifer is the region's principal groundwater resource: 150 metre "
            "thick, transmissivity 2100 millidarcy metres, storage 0.18, brackish quality. The Main "
            "Boundary Thrust carries 4.5 millimetres per year of slip and marks the southern limit of "
            "deformation. Shallow borehole BH-03 at 160 metres depth logs alluvium and sandstone with "
            "640 millidarcy permeability.\n"
            "Numerical groundwater flow models couple recharge from the Murree hills to discharge at "
            "the Soan River, reproducing observed water table depths of 14 to 22 metres. Uplift rates "
            "of 2.2 millimetres per year maintain the forearc gradient without triggering large "
            "landslides in the weak sediments."
        ),
    },
    {
        "id": "rep-chitral-accretion",
        "name": "Chitral Accretionary Prism Engineering Geological Report",
        "bbox": [71.0, 34.4, 72.6, 36.0],
        "text": (
            "The Chitral Accretionary Prism is composed of the Chitral Slate Belt, sheared phyllites "
            "and detached trench sediments. Slate hardness near 4.2 Mohs and pervasive cleavage make "
            "slopes prone to shallow failures during snowmelt. Rainfall of 720 millimetres per year "
            "is the highest in the study area, and uplift of 5.6 millimetres per year sustains relief "
            "of 2400 metres.\n"
            "The Chaman Transform Fault forms the western boundary as a near-vertical strike-slip "
            "wall with 12 millimetres per year of right-lateral motion and high seismicity. Long-term "
            "seismic hazard assessment maps peak ground acceleration above 0.4 g within 20 kilometres "
            "of the fault trace.\n"
            "The Chitral Terrace Gravel Aquifer supplies irrigation water with transmissivity of 2700 "
            "millidarcy metres. Borehole BH-04 at 240 metres encountered mineral springs along the "
            "slate–phyllite contact. Slope stability back-analysis gives effective cohesion of 18 kPa "
            "and friction angle of 28 degrees for the weathered slate."
        ),
    },
    {
        "id": "rep-dasu-dam",
        "name": "Dasu Hydropower Dam Foundation Investigation",
        "bbox": [73.3, 33.8, 74.0, 34.4],
        "text": (
            "The Dasu hydropower project sits on the Indus River where it cuts the Kaghan–Indus "
            "structural valley. Foundation exploration drilled ten boreholes including BH-10 at 610 "
            "metres into fresh granite with 6500 metres per second P-wave velocity and 18 millidarcy "
            "permeability. The Besham Duplex Fault and Kunhar River Fault pass within the reservoir "
            "area and require grout curtain treatment.\n"
            "Open excavation reaches competent rock below 45 metres of weathered granite. Maximum "
            "flood discharge design assumes a 1 in 10,000 year event routed through the narrow "
            "canyon. Static settlement of the gravity section is computed at 32 millimetres.\n"
            "The Indus Valley Alluvial Aquifer downstream of the dam is managed for baseline "
            "conditions: water table 14 metres deep, storage 0.22, fresh quality. Induced seismicity "
            "monitoring uses a 12 station array with 0.1 magnitude completeness thresholds."
        ),
    },
    {
        "id": "rep-regional-dem",
        "name": "Regional Digital Elevation Model and Terrain Analytics Note",
        "bbox": [71.0, 32.2, 76.9, 36.2],
        "text": (
            "This terrain analytics note summarises the regional 30 metre digital elevation model "
            "covering the entire study area from the Chaman Transform Fault to the Karakoram ranges. "
            "Shaded relief reveals five tectonic provinces: the Karakoram Fold-Thrust Belt, Indus "
            "Suture Zone, Chitral Accretionary Prism, Potwar–Kohistan Forearc and Gilgit Gneiss Dome.\n"
            "Terrain roughness correlates strongly with rock hardness: gneiss and gabbro ridges hold "
            "roughness indices of 0.77 to 0.82, while Murree sandstone slopes sit near 0.41. Uplift "
            "gradients from thermochronology range from 2.2 to 7.5 millimetres per year and explain "
            "most of the variance in channel steepness after controlling for rainfall.\n"
            "The derived product set includes slope, aspect, curvature, topographic wetness index and "
            "drainage network vectors. These layers feed hydraulic erosion simulations, sediment "
            "connectivity analysis and 3D visualisation of the terrain in WebGL and Unreal Engine 5 "
            "using Cesium tiles. All elevation values are referenced to EGM96 geoid."
        ),
    },
]

