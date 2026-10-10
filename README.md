# GeoMind-3D

### Autonomous Neural-Spatial Agent Engine for Geomathematical Rendering

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.5-013243?logo=numpy&logoColor=white">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.18-8CAAE6?logo=scipy&logoColor=white">
  <img alt="Three.js" src="https://img.shields.io/badge/Three.js-r170-000000?logo=threedotjs&logoColor=white">
  <img alt="WebGL" src="https://img.shields.io/badge/WebGL-WebGPU-990000">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="secrets" src="https://img.shields.io/badge/secrets-git--ignored-critical">
</p>

> **GeoMind-3D** is an autonomous, multi-modal GenAI platform that integrates a
> **Spatial Knowledge Graph Engine** as a core memory and context layer. It
> converts multi-modal inputs — natural language queries, satellite imagery,
> geological survey reports and elevation models — into physically grounded 3D
> environments, real-time PDE simulations and watertight CAD meshes.
>
> *You do not ask a text model to hallucinate a mountain. You ask GeoMind-3D to
> ground the request in a real spatial corpus, then generate the geometry that
> the physics of that corpus implies.*

---

## Table of Contents

1. [What GeoMind-3D Actually Does](#1-what-geomind-3d-actually-does)
2. [Feature Highlights](#2-feature-highlights)
3. [System Architecture](#3-system-architecture)
4. [The Pipeline, Stage by Stage](#4-the-pipeline-stage-by-stage)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [Installation & Quick Start](#7-installation--quick-start)
8. [Configuration — the Single `.env`](#8-configuration--the-single-env)
9. [API Reference](#9-api-reference)
10. [The WebGL Studio (Frontend)](#10-the-webgl-studio-frontend)
11. [Data Model & Seed Corpus](#11-data-model--seed-corpus)
12. [The Mathematics](#12-the-mathematics)
13. [Determinism & Reproducibility](#13-determinism--reproducibility)
14. [Testing](#14-testing)
15. [Deployment](#15-deployment)
16. [Performance Notes](#16-performance-notes)
17. [Extending the Engine](#17-extending-the-engine)
18. [Troubleshooting](#18-troubleshooting)
19. [Roadmap](#19-roadmap)
20. [License & Citation](#20-license--citation)

---

## 1. What GeoMind-3D Actually Does

Type this into the studio:

> *"Simulate hydraulic erosion across the high-relief fractured-rock terrain
> north of the Main Karakoram Thrust, then export a watertight mesh."*

and GeoMind-3D does all of the following, end to end, in a single HTTP request:

| # | Stage | What really happens |
|---|-------|---------------------|
| 1 | **Ground** | The query is embedded into a 1536-d vector and searched against a **hybrid spatial vector store** (HNSW). Retrieved report chunks are re-projected into **Spatial SQL** (`ST_Intersects`, `ST_Distance`) to pull the physical features they mention. A **relational Graph Neural Network** propagates embeddings along typed topology edges (`intersects`, `contains`, `overlays`, `abuts`, `mentions`). The three rankings are merged with **Reciprocal Rank Fusion**. |
| 2 | **Condition** | The fused nodes are rasterised into a 7-channel physical bound grid (elevation, relief, hardness, rainfall, uplift, roughness, influence weight). The elevation channel is decomposed into **spherical harmonics** $Y_l^m(\theta,\varphi)$ up to degree `SH_DEGREE` so planetary curvature survives the projection. |
| 3 | **Generate** | A **conditional latent diffusion** chain — DDPM forward schedule with a *conjugate Gaussian prior* in DCT-latent space — samples a coarse global geometry prior. The prior's mean comes from grounded elevation; its variance is coloured by a spectral tilt driven by rock hardness and rainfall. |
| 4 | **Simulate** | A **Fourier Neural Operator** (or the exact reference PDE solver) runs erosion / landslide / seismic propagation, orders of magnitude faster than finite elements. |
| 5 | **Render** | The implicit field is converted into **oriented 3D Gaussians** for sub-millisecond WebGL rendering, *and* into a **watertight dual-contouring mesh** exported as `OBJ` / `PLY` / `GLB` / `USDZ`. |

Every number in the output is traceable back to a piece of evidence in the
corpus, and the whole chain is deterministic for a given seed.

---

## 2. Feature Highlights

### Grounding & Memory
- **Hybrid Spatial Vector Store** — dense 1536-d embeddings with **HNSW** indexing (`M`, `ef_construction`, `ef_search` all configurable), plus interchangeable `qdrant` and `supabase`/pgvector backends selected by one env var.
- **Real Spatial SQL** — PostGIS-flavoured predicates (`ST_Intersects`, `ST_Contains`, `ST_Within`, `ST_Touches`, `ST_Overlaps`, `ST_Distance`, `ST_Area`, `ST_Length`, `ST_Centroid`) registered as SQLite functions, so the topology engine issues genuine `SELECT … WHERE ST_Intersects(a.geom, b.geom) = 1` joins.
- **FTS5 lexical index** as an additional retrieval channel over feature names and report text.
- **Relational GNN topology traversal** with per-relation projection weights, residual anchoring and tanh-gated aggregation.
- **Reciprocal Rank Fusion** with per-source weights and full provenance (which source, which rank, which score) for every fused item.

### Geomathematical Latent Engine
- **Spherical Harmonic positional encoders** $Y_l^m(\theta,\varphi)$ computed from an associated-Legendre recurrence — no `scipy.special` dependency, Condon–Shortley phase included, real orthonormal basis available.
- **DCT latent autoencoder** — an orthonormal DCT-II/III basis gives Parseval-optimal compression with no training corpus required.
- **Conditional latent diffusion** with a *closed-form exact reverse chain* under a conjugate Gaussian prior — no learned score network, fully reproducible.
- **Spectral tilt colouring** so generated detail is pink-noise-like, with hardness and rainfall controlling the spectral slope.

### Physics
- **Stream-power hydraulic erosion** $E = K\,A\,S$ with D8 flow accumulation and hillslope sediment diffusion.
- **Thermal talus-angle creep** with a Bingham-style repose threshold.
- **Damped acoustic wave propagation** (leap-frog, CFL-safe).
- **Exact Fourier spectral heat-equation solver** — the operator the FNO is trained to emulate.
- **FNO-2D** with hand-derived analytic gradients, Adam, dataset generation *from the reference solver itself*, weight caching and a built-in speed/accuracy benchmark.

### Rendering & Export
- **Dual-contouring neural SDF extraction** with automatic watertightness verification and a **guaranteed-manifold marching-tetrahedra fallback** (Kuhn 6-tet fan).
- **Oriented 3D Gaussians** with degree-3 spherical-harmonic colour coefficients, exportable as binary `.ply` / `.splat` and streamable to a WebGL rasterizer.
- **Watertight CAD/GIS exports**: `OBJ`, `PLY` (binary + ASCII), `GLB` (binary glTF 2.0), `USDZ` (zip-wrapped USDA) — all written by hand, zero external geometry libraries.
- **Topology proof** attached to every mesh: `non_manifold_edges`, `inconsistent_directed_edges`, signed volume.

### Engineering
- **One `.env` file.** Every key, token and switch lives in the repository-root `.env`, which is git-ignored. There is deliberately **no** `.env.example`.
- **Zero required infrastructure.** SQLite + in-process HNSW + NumPy/SciPy. No Postgres, no Docker, no GPU, no model downloads to boot.
- **Runs with or without PyTorch.** The FNO and GNN are pure NumPy implementations with exact analytic gradients; `torch`, `torch-geometric`, `gsplat`, `open3d` and `trimesh` are optional accelerators.
- **Deterministic.** Every stochastic component is seeded from `.env`.
- **Fully tested.** Topology watertightness is asserted, not assumed.

---

## 3. System Architecture

```
                               ┌────────────────────────────────────────┐
                               │     Multi-Modal Query & Spatial Data   │
                               │  (Text, Satellite DEMs, Geological PDFs)│
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Grounding & Memory Core                                                                        │
│  ├── Multi-Modal Spatial Vector Index (1536-dim embeddings for unstructured reports & imagery)   │
│  ├── GNN Topology Traversal Engine (Extracts physical edges: intersects, overlies, abuts)       │
│  └── Reciprocal Rank Fusion (RRF) Reranker (Fuses dense semantic vectors with spatial SQL)        │
└──────────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │ Grounded Spatial Context & Physical Bounds
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Geomathematical Latent Engine                                                                  │
│  ├── 3D Latent Diffusion Model (Generates coarse global geometry prior)                          │
│  ├── Spherical Harmonic Encoders: Y_l^m(θ, φ) (Preserves spherical/planetary curvature)          │
│  └── Fourier Neural Operator (FNO) (Solves erosion & fluid flow PDEs in real-time)              │
└──────────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Multi-Representation Rendering Pipeline                                                         │
│  ├── 3D Gaussian Splatting (3DGS) ──► Ultra-high FPS WebGL / Unreal Engine 5 view synthesis       │
│  └── Dual Contouring Neural SDF ──► Watertight OBJ/USDZ for CAD & GIS export                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Module map

```
run.py                     ── single-command launcher (warms engines, serves API + frontend)
backend/
├── app/
│   ├── config.py          ── typed `settings` singleton, loads the ONE root `.env`
│   ├── main.py            ── FastAPI app: CORS, routers, static frontend, lifespan warm-up
│   ├── pipeline.py        ── orchestration engine (ground → condition → generate → render)
│   ├── api/               ── versioned REST surface under /api/v1
│   ├── core/
│   │   ├── geometry.py        ── planar spatial predicates (the ST_* kernel)
│   │   ├── spatial_sql.py     ── SQLite + PostGIS-flavoured functions + FTS5 + scenes
│   │   ├── embeddings.py      ── LocalEmbedder (BLAKE2b) / OpenAIEmbedder (fallback)
│   │   ├── hnsw.py            ── HNSW graph index (Malkov & Yashunin 2018)
│   │   ├── vector_store.py    ── local-hnsw | qdrant | supabase adapters
│   │   ├── rrf.py             ── Reciprocal Rank Fusion with provenance
│   │   ├── gnn.py             ── Relational GNN message passing
│   │   ├── knowledge_graph.py ── seeding, graph build, Repository.retrieve()
│   │   ├── conditioning.py    ── fused nodes → 7-channel physical bound grid
│   │   ├── latent/            ── spherical_harmonics · basis (DCT) · diffusion
│   │   └── rendering/         ── surface_nets · marching_tets · gaussians · exporters
│   ├── physics/           ── solvers.py (reference PDEs) · fno.py (Fourier Neural Operator)
│   └── data/              ── seed_features.py · seed_reports.py (the grounded corpus)
└── tests/                 ── pytest suite (geometry, SQL, RRF, GNN, SH, FNO, meshing, API)
frontend/
├── index.html             ── the WebGL studio shell
├── css/style.css          ── layout, panels, HUD
└── js/                    ── app.js · viewer.js · splats.js · graph.js · api.js
```

---

## 4. The Pipeline, Stage by Stage

Every request flows through five stages. Each stage is independently callable
through the REST API, so you can inspect intermediate artefacts.

```
 query ─► ground() ─► build_conditioning() ─► diffusion.sample() ─► simulate() ─► render()
          └─ RRF        └─ 7-ch grid           └─ latent z₀         └─ PDE       └─ mesh + splats
              + GNN         + SH degree L                          + FNO        + OBJ/GLB/USDZ
```

### 4.1 Grounding & Memory Core

**Embedding.** `app/core/embeddings.py` ships two interchangeable backends.

* `LocalEmbedder` — offline, deterministic feature-hashing embedder. Tokens,
  word bigrams and character trigrams are hashed with **BLAKE2b** (stable across
  processes, unlike Python's randomised `hash()`) into a 1536-dim signed vector
  with sqrt-inverse-frequency weighting and stopword down-weighting, then
  L2-normalised. Cosine similarity behaves like a soft lexical/semantic overlap
  metric. No network, no weights, fully reproducible.
* `OpenAIEmbedder` — calls `text-embedding-3-small` (1536 dims) when
  `OPENAI_API_KEY` is configured, falling back to the local embedder when
  `EMBEDDING_FALLBACK=true`.

**Vector index.** `app/core/hnsw.py` is a complete NumPy HNSW implementation
(Malkov & Yashunin, 2018): multi-layer exponential level assignment, greedy
descent on upper layers, ef-construction candidate selection with the
select-nearest heuristic, reciprocal link pruning, cosine similarity on
L2-normalised vectors, and `.npz` persistence. Expected search complexity
$O(\log N)$ keeps grounding latency flat as the corpus grows.

`app/core/vector_store.py` wraps it behind one interface
(`add(records, vectors)` / `search(vector, k)`) with three backends selected by
`VECTOR_STORE`: `local-hnsw`, `qdrant`, `supabase` (pgvector via PostgREST RPC).
Records are report chunks carrying a **bbox anchor** so vector hits can be
re-projected back into Spatial SQL.

**Spatial SQL.** `app/core/spatial_sql.py` registers real PostGIS-flavoured
predicates as SQLite functions and the topology engine therefore issues genuine
SQL:

```sql
SELECT b.id AS id, ST_Distance(a.geom, b.geom) AS d
FROM   features a, features b
WHERE  a.id = :anchor AND b.id != a.id
  AND  ST_Intersects(a.geom, b.geom) = 1
ORDER  BY d ASC;
```

The geometry kernel underneath (`app/core/geometry.py`) implements
`intersects`, `contains`, `within`, `touches`, `overlaps`, `distance`,
`area_m2`, `length_m`, `centroid` and `buffer` on GeoJSON-like Point /
LineString / Polygon dicts in lon/lat, using an equirectangular metric scaled by
$\cos(\varphi_0)$ — accurate to <0.5 % at regional (≤500 km) scales, which is
plenty for grounding and keeps the engine dependency-free.

Tables: `features`, `feature_edges`, `scenes`, plus an FTS5 virtual table
`features_fts(id UNINDEXED, name, text)` for the lexical channel.

**GNN topology traversal.** `app/core/gnn.py` performs relational message
passing over typed edges:

$$
m_r(v) = \operatorname{mean}_{u \in N_r(v)} h(u)
\qquad
h'(v) = h(v) + \gamma \tanh\!\left(\frac{\sum_r m_r(v) W_r + h(v)W_0}{\sqrt{2}}\right)
$$

The residual form anchors every node to its own content embedding while
neighbours' attributes bleed in along *physical* relationships. A query about
"permeable aquifer near a fault" therefore ranks aquifers that are topologically
adjacent to faults even if their report text never co-occurs. Weights are seeded
from `GNN_SEED` and act as a fixed graph filter — no training loop is required
for the retrieval objective, so boot time is zero while remaining a genuine
multi-layer message-passing network.

**Reciprocal Rank Fusion.** `app/core/rrf.py` merges the vector, spatial and GNN
rankings with:

$$
\operatorname{score}(d) = \sum_{s \in \{\text{vector, spatial, gnn}\}} \frac{w_s}{k + \operatorname{rank}_s(d)}
$$

RRF is scale-free — it needs no score calibration between cosine similarities,
metre distances and graph affinities, because only ordinal ranks matter. Every
fused item is annotated with the per-source rank/score/weight so the UI can show
*why* each node was retrieved.

### 4.2 Geomathematical Latent Engine

**Conditioning rasteriser** (`app/core/conditioning.py`). The RRF-fused nodes are
turned into a coarse multi-channel grid:

| Channel | Meaning |
|---|---|
| `elevation` | mean sea-level elevation, m |
| `relief` | local relief, m |
| `hardness` | bulk rock hardness (Mohs) |
| `rainfall` | mean annual rainfall, mm |
| `uplift` | tectonic uplift, mm/yr |
| `roughness` | surface roughness index ∈ [0,1] |
| `weight` | normalised influence weight ∈ [0,1] |

Each retrieved feature contributes a Gaussian influence kernel
$w(x) = \sigma \exp\!\left(-\left(d/r\right)^2\right)$ in lon/lat space, weighted
by its RRF score. Features without physical properties (reports) contribute only
a weak prior (`×0.15`). The spatial extent is derived from the union of the
retrieved footprints, padded by 8 %.

The elevation channel is then decomposed through **spherical harmonics** at
degree ≤ `SH_DEGREE`: the low-degree $Y_l^m$ component is the *planetary global
shape* (projection-distortion free), while local detail is supplied later by the
DCT latent diffusion — a clean split between global curvature and local
geomorphology.

**Spherical Harmonic encoders** (`app/core/latent/spherical_harmonics.py`).
Standard complex spherical harmonics computed from a normalised
associated-Legendre recurrence:

$$
Y_l^m(\theta,\varphi) = N_{lm}\, P_l^m(\cos\theta)\, e^{i m \varphi},
\qquad
N_{lm} = \sqrt{\tfrac{2l+1}{4\pi}\tfrac{(l-m)!}{(l+m)!}}
$$

plus the *real* form used for terrain encoding (graphics convention):

$$
m = 0 \rightarrow \operatorname{Re}(Y_l^m), \qquad
m > 0 \rightarrow \sqrt{2}\,\operatorname{Re}(Y_l^m), \qquad
m < 0 \rightarrow \sqrt{2}\,\operatorname{Im}(Y_l^{|m|})
$$

**Why it matters.** Fourier coordinate encodings (NeRF-style) assume a flat
plane and warp badly when a scene spans a planetary curved surface. Expanding
positions in $Y_l^m$ keeps the encoding isotropic on the sphere — no map
projection distortion over large extents. The pipeline uses these basis
functions for the low-degree global curvature term of the conditioning raster
and for the degree-3 colour SH of exported 3D Gaussians (`f_dc` / `f_rest`).

**Latent basis** (`app/core/latent/basis.py`). The engine's "VAE" is an
orthonormal DCT-II/III basis: the encoder projects a conditioning grid onto the
lowest `LATENT_DIM` frequency coefficients, the decoder reconstructs. Because
the basis is orthonormal, the reconstruction is exact for retained coefficients
— Parseval-optimal, the same optimality property PCA has, without needing a
training corpus. `upsample()` uses cubic (`order=3`) zoom resampling, and
`spectral_tilt()` produces the $1/(1+r^p)$ radial envelope that colours the
latent prior.

**Conditional latent diffusion** (`app/core/latent/diffusion.py`).

Forward (DDPM):

$$
z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad
\bar\alpha_t = \prod_{s\le t}(1-\beta_s),
\qquad
\beta_s \in [10^{-4},\,0.02] \text{ linear}
$$

Conditional prior from the grounding stage (diagonal Gaussian in latent space):

$$
z_0 \sim \mathcal N(\mu, \Sigma), \qquad
\mu = \text{DCT-mean shaped by elevation}, \qquad
\Sigma = \operatorname{diag}(\sigma_k^2)
$$
$$
\sigma_k = \text{relief}\cdot \text{tilt}_k \cdot (0.55 + \text{roughness})
$$

Because prior and likelihood are Gaussian, the **exact** reverse marginal is
closed-form — no learned score network is needed:

$$
p(z_{t-1}\mid z_t) = \mathcal N\!\left(a_t z_t + b_t,\; v_t\right)
$$
$$
a_t = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},
\qquad
b_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\mu,
\qquad
v_t = \left(\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\right)^{\!2}\sigma^2 + \tilde\beta_t
$$

Iterating `DIFFUSION_STEPS` steps from pure noise
$z_T \sim \mathcal N(\sqrt{\bar\alpha_T}\mu,\; (1-\bar\alpha_T)I + \bar\alpha_T\Sigma)$
converges to the conditional prior — i.e. this **is** posterior sampling under
the conditioning, implemented as a diffusion chain, and it is reproducible per
seed. The pinkness of the prior is computed as
$p = 1.9 - 1.1\,\widehat{\text{hardness}} - 0.4\,\widehat{\text{rainfall}}$, so
hard dry rock keeps sharp high-frequency ridges while soft wet terrain smooths
out.

### 4.3 Physics Engine

`app/physics/solvers.py` implements the reference PDEs; `app/physics/fno.py`
implements a Fourier Neural Operator trained to reproduce them.

#### Hydraulic erosion

Stream power $E = K\,A\,m\,S^n$, with drainage area $A$ accumulated over the D8
flow graph and hillslope soil creep:

```python
E = K * (A ** m) * (S ** n)                      # stream-power incision
dep = A ** m * S ** n * RELIEF_DEPOSITION        # deposition component
dt = min(0.3, 0.25 * dx ** 2 / (D + DEPOSITION))
height += uplift_growth * dt
# sediment-diffusion on the height field
```

D8 flow routing comes first (gradient-descent neighbour selection with local
pit filling), then $A$ is accumulated topologically by scanning cells in
descending order. `HARDNESS_SCALING` converts rock hardness into a per-cell
erodibility mask, so the quartzite ridges survive while the phyllites gorge —
which is exactly what the retrieved corpus predicts.

#### Landslide creep

$$
c \frac{\partial h}{\partial t} = K_s \nabla\!\cdot\!\Big(\big(1 - \tfrac{\tan\theta_c}{\lvert\nabla h\rvert}\big)\,\nabla h\Big)
$$

implicitly stabilised with a semi-implicit update and CFL-clamped
$\Delta t$; areas already below the repose angle are frozen out.

#### Seismic propagation

2-D acoustic wave equation $\partial^2 u/\partial t^2 = c^2 \nabla^2 u$ with
damping, leap-frog in time, Courant-safe
$\Delta t \le \Delta x /(\sqrt{2}\,c_\max)$.

#### Fourier spectral heat solver

The exact reference solution to $\partial h/\partial t = -\kappa\nabla^2 h$ via
FFT multiplication by $e^{-\kappa |k|^2 t}$. This is the operator the FNO is
trained against.

#### Fourier Neural Operator

`app/physics/fno.py` implements an **FNO-2D** — Li et al., 2020 — with spectral
convolutions:

$$
(\mathcal F h)(x) = \mathcal F^{-1}\big(R_\phi\,(\mathcal F h)(k)\big)_{k \le k_{\max}}
$$

* **Backend:** PyTorch if available (`USE_TORCH=true`), otherwise a pure-NumPy
  implementation with the same forward pass *and hand-derived analytic
  gradients* — `W1`, `W2`, `W3`, `W4` spectral kernels, `C1`, `C2` channel
  lifts, biases. Adam optimizer. Dropout. Weight decay. Early stopping. LR
  scheduling. It's a real, trainable FNO.
* **Dataset:** generated from the reference solver itself (`make_dataset`), so
  the FNO learns to emulate the physics with $O(N\log N)$ cost.
* **Batching:** mini-batch gradient descent with shuffling.
* **Cache:** trained weights are persisted to `FNO_WEIGHTS_PATH` and reloaded on
  boot, so training happens once.
* **Benchmark:** `benchmark()` measures the speedup of the FNO over the reference
  solver at the resolution in `.env` and returns the MSE.

Typical measured behaviour (see §16): the FNO reproduces the spectral solution
to within a few percent at a 16–64× speedup.

### 4.4 Multi-Representation Rendering Pipeline

The simulated implicit height/SDF field is rendered in two complementary
representations simultaneously.

#### A. 3D Gaussian Splatting

`app/core/rendering/gaussians.py` builds oriented 3D Gaussians from the terrain
field:

* Position $(x, y, h)$ from the elevation grid; normals from central differences.
* Anisotropic scale following terrain slope (thin normal to the surface, spread
  in the tangential plane), so slopes use elongated Gaussians and flats use
  compact ones.
* Opacity from a combination of the local slope and the physical roughness
  channel.
* **Degree-3 spherical harmonic colour** — `f_dc` from the mean colour and
  `f_rest` from the SH expansion — exactly matching the standard 3DGS layout used
  by Unreal Engine 5's Cesium plugin and `gsplat`.

Exports:

* `PLY` binary (little-endian, 3DGS standard layout: position, normals, DC + 45 SH
  coefficients, opacity, log-scale, rotation quaternion) — directly loadable by
  Postshot, Nerfstudio, SuperSplat, or the UE5 `LumaAI` / Splat importers.
* `SPLAT` — the compact runtime format consumed by the bundled WebGL
  rasterizer (`frontend/js/splats.js`).

#### B. Dual Contouring Neural SDF → Watertight Mesh

`app/core/rendering/surface_nets.py`:

1. **Signed distance field.** Bilinear sample of the height field into an
   $N_x \times N_y \times N_z$ scalar grid; negative inside the solid, positive
   outside.
2. **Watertightness proof.** Before extraction, the code **verifies** that the
   SDF sign change on every boundary face is consistent — if the field can leak
   at the borders, the domain is padded until it cannot.
3. **Dual-contouring / surface-nets extraction.** Cells with a sign change
   produce a vertex placed at the minimiser of squared QEF error (the
   dual-vertex formulation that gives manifold, watertight meshes — unlike
   naive surface nets with degenerate non-manifold output).
4. **Automatic verification.** The extracted mesh is checked for
   `non_manifold_edges` and `inconsistent_directed_edges`. If both are zero, it's
   proven watertight.
5. **Guaranteed fallback.** If dual contouring *ever* produced a non-manifold
   edge (rare — degenerate zero-function value cases), the pipeline
   automatically falls back to **marching tetrahedra** (Kuhn 6-tet fan), which
   is *provably* manifold: every tet surface is a closed 2-manifold, edge-sharing
   tets produce matching edges, and no non-manifold configuration can arise.
   The fallback is exercised and asserted in the test suite.

#### C. Exporters

All hand-written, no geometry dependencies:

* `OBJ` (ASCII, with vertex normals and faces)
* `PLY` binary + ASCII
* `GLB` — full binary glTF 2.0 with JSON chunk + BIN chunk + accessors + mesh
  primitives
* `USDZ` — zip-wrapped USDA (Universal Scene Description ASCII) with `UsdGeomMesh`
  prims, ready for Apple AR Quick Look / USDZ converter

The API returns either base64-encoded mesh bytes or a downloadable file.

---

## 5. Technology Stack

| Layer | Technologies |
| --- | --- |
| **Grounding & Retrieval** | PyTorch Geometric *(optional)*, Supabase (pgvector + PostGIS) / Qdrant, LlamaIndex *(optional)*, HNSW, SQLite + FTS5 |
| **Deep Learning & Solvers** | PyTorch, `torch-harmonics` (Fourier Neural Operators), Taichi (CUDA) — all **optional** |
| **3D Rendering & Geometry** | `gsplat` (3DGS), Open3D, Trimesh, PyVista — all **optional** |
| **Frontend / Engine Integration** | Three.js / WebGPU, Unreal Engine 5 (via Cesium Plugin) |
| **Core (always required)** | Python 3.11+, FastAPI, Uvicorn, NumPy, SciPy, Pydantic |

**Why optional?** Everything works out of the box with just the core stack. The
optional accelerators are wrapped in try/except imports so a missing `torch` or
`gsplat` degrades gracefully instead of crashing.

---

## 6. Project Structure

```
GeoMind-3D/
├── .env                          # THE single config + all secrets (GIT-IGNORED)
├── .gitignore                    # explicitly ignores .env, data/, __pycache__, *.npz …
├── README.md                     # ← you are here
├── run.py                        # one-command launcher: `python run.py`
├── requirements.txt              # backend requirements (root convenience shim)
├── pytest.ini
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py             # pydantic-settings; loads ../../.env
│   │   ├── main.py              # FastAPI application factory + lifespan warm-up
│   │   ├── pipeline.py           # Engine: ground() → condition() → sample() → render()
│   │   │
│   │   ├── api/                  # routers mounted under /api/v1
│   │   │   ├── __init__.py
│   │   │   ├── routes_generate.py    # POST /generate
│   │   │   ├── routes_graph.py       # GET  /graph
│   │   │   ├── routes_scenes.py      # GET/POST /scenes
│   │   │   ├── routes_physics.py     # POST /simulate, GET /benchmark
│   │   │   └── routes_status.py      # GET  /status, /buildings, /imagery
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── geometry.py            # planar spatial predicates (ST_* kernel)
│   │   │   ├── spatial_sql.py         # SQLite + PostGIS-style UDFs + FTS5
│   │   │   ├── embeddings.py          # LocalEmbedder / OpenAIEmbedder
│   │   │   ├── hnsw.py                # HNSW graph index
│   │   │   ├── vector_store.py        # local-hnsw | qdrant | supabase
│   │   │   ├── rrf.py                 # Reciprocal Rank Fusion
│   │   │   ├── gnn.py                 # relational GNN message passing
│   │   │   ├── knowledge_graph.py     # Repository: seed, build, retrieve
│   │   │   ├── conditioning.py       # fused nodes → 7-ch physical grid
│   │   │   ├── latent/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── spherical_harmonics.py   # Y_l^m encoders
│   │   │   │   ├── basis.py                 # DCT latent autoencoder
│   │   │   │   └── diffusion.py             # conditional DDPM sampler
│   │   │   └── rendering/
│   │   │       ├── __init__.py
│   │   │       ├── surface_nets.py        # dual contouring + SDF
│   │   │       ├── marching_tets.py      # Kuhn 6-tet manifold fallback
│   │   │       ├── gaussians.py          # 3DGS generation
│   │   │       └── exporters.py          # OBJ / PLY / GLB / USDZ
│   │   │
│   │   ├── physics/
│   │   │   ├── __init__.py
│   │   │   ├── solvers.py               # reference PDEs
│   │   │   └── fno.py                   # Fourier Neural Operator
│   │   │
│   │   └── data/
│   │       ├── __init__.py
│   │       ├── seed_features.py         # 12 physical features with real geometry
│   │       └── seed_reports.py          # 10 geological report chunks
│   │
│   ├── requirements.txt                  # pinned dependency set
│   └── tests/                            # 40+ tests
│       ├── conftest.py
│       ├── test_geometry.py
│       ├── test_spatial_sql.py
│       ├── test_rrf.py
│       ├── test_gnn.py
│       ├── test_spherical_harmonics.py
│       ├── test_diffusion.py
│       ├── test_physics.py
│       ├── test_meshing.py
│       └── test_api.py
│
└── frontend/
    ├── index.html                        # WebGL studio shell
    ├── css/style.css                      # dark HUD layout
    └── js/
        ├── api.js                         # typed REST client
        ├── app.js                         # controller / state machine
        ├── viewer.js                      # Three.js WebGL scene
        ├── splats.js                      # WebGL 3DGS rasterizer
        └── graph.js                       # 2D canvas knowledge-graph renderer
```

---

## 7. Installation & Quick Start

### Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11+ | FastAPI, Pydantic v2, typing |
| Node.js | 18+ *(optional)* | only if you want to rebuild the frontend |
| RAM | 2 GB+ | NumPy grids |
| GPU | **not required** | the FNO runs on NumPy |

### 1. Clone

```bash
git clone https://github.com/SaadxSalman/GeoMind-3D.git
cd GeoMind-3D
```

### 2. Virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

Optionally install the accelerators (each degrades gracefully if absent):

```bash
pip install torch torch-geometric          # GPU FNO / PyG GNN backend
pip install gsplat                          # production 3DGS rasterisation
pip install open3d trimesh pyvista        # alternative meshing pipelines
pip install llama-index torch-harmonics   # retrieval / spherical-harmonic FNO
```

### 4. Create the `.env`

The repository ships **one** `.env` file at the root containing *every* key and
switch. It is listed in `.gitignore`, so it never reaches GitHub. Fill in the
values you have; anything left blank falls back to a safe local default.

```bash
# Windows
notepad .env
# macOS / Linux
nano .env
```

### 5. Run

```bash
python run.py
```

`run.py` is a convenience launcher that:
1. Loads the root `.env` into the environment.
2. Warms the embedder, HNSW index, FNO weights and graph so the first request is
   not slow.
3. Binds Uvicorn to `BACKEND_HOST:BACKEND_PORT` with the `BACKEND_WORKERS` value.

Then open **http://localhost:8000** — the FastAPI app serves the WebGL studio
directly.

Interact at http://localhost:8000/docs (Swagger UI) or
http://localhost:8000/redoc (ReDoc) if `DEBUG=true`.

---

## 8. Configuration — the Single `.env`

GeoMind-3D has **exactly one** configuration file:

```
GeoMind-3D/
└── .env          ← every key, token and switch lives here
```

There is intentionally **no** `.env.example`, no `.env.development`, no
`.env.production` and no `appsettings.json`. Splitting secrets across files is how
they leak into git history; a single git-ignored file is auditable in one
`cat` and impossible to accidentally publish.

### 8.1 How it is loaded

`backend/app/config.py` is the only module that touches the environment:

```python
REPO_ROOT = <repo root discovered by walking up from cwd / the package>
load_dotenv(REPO_ROOT / ".env", override=False)
```

* `_find_repo_root()` looks for a directory containing `.env` (or
  `backend/` + `run.py`) among `Path.cwd()` and the package's parents — so
  `python run.py`, `uvicorn app.main:app` from `backend/`, and `pytest` from
  anywhere all resolve the same file.
* `override=False` means a real OS environment variable always wins over `.env` —
  ideal for Docker/CI where you inject secrets without writing them to disk.
* Everything is exposed through a typed, frozen `Settings` dataclass with
  `@lru_cache`d `settings` singleton, and defensive `_int` / `_float` / `_bool` /
  `_list` parsers that fall back to sane defaults on malformed input instead of
  raising on boot.

```python
from app.config import settings

settings.app_port          # 8000
settings.retrieval_top_k   # 8
settings.fno_modes         # 8
settings.artifacts_dir     # PosixPath('/…/GeoMind-3D/artifacts')
```

**Never** `import os; os.environ[...]` anywhere else — add a property to
`Settings` instead.

### 8.2 Git-ignored — guaranteed

`.gitignore` leads with a warning banner and then:

```gitignore
# ── Secrets ──
.env
.env.*
!.env.example

# ── Python ──
__pycache__/
*.py[cod]
.venv/
venv/

# ── Runtime artifacts (generated scenes, meshes, indexes, DBs) ──
artifacts/
*.db
*.sqlite
*.sqlite3
*.npz
!backend/**/*.npz

# ── Node / frontend ──
node_modules/
frontend/dist/

# ── IDE / OS ──
.vscode/*
!.vscode/extensions.json
.idea/
.DS_Store
Thumbs.db

# ── Logs / temp ──
*.log
npm-debug.log*
tmp/
temp/
```

Notes on the rules:

* `.env` and `.env.*` cover the master file plus any stray variants. **No
  `.env.example` file is present in this repository** (there is nothing to
  un-ignore; the `.env` you create is the whole contract, and this README §8.3
  documents every key).
* `artifacts/` swallows the generated spatial DB, the HNSW `.npz` index and the
  FNO weight cache — all rebuildable, all secret-free, all large.

Verify at any time:

```bash
git check-ignore -v .env
# → .gitignore:8:.env	.env
git status --short       # must not list .env
git ls-files | grep env   # must print nothing
```

If you ever *did* commit it by accident, purge it with
`git rm --cached .env && git filter-repo --invert-paths --path .env`, then rotate
every key that was in it.

### 8.3 Every Key in `.env`

The shipped `.env` is organised into 13 numbered blocks. Values below are the
**defaults already in your file**; replace the `REPLACE_*` placeholders with real
credentials as needed.

#### Block 1 — Application / Server

| Key | Default | Purpose |
|---|---|---|
| `APP_NAME` | `GeoMind-3D` | Title used in OpenAPI docs and the status endpoint |
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `APP_DEBUG` | `true` | Swagger/ReDoc enabled, verbose error bodies |
| `APP_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose) |
| `APP_PORT` | `8000` | Bind port |
| `APP_ROOT_PATH` | *(empty)* | Set when behind a proxy prefix, e.g. `/geomind` |
| `APP_WORKERS` | `1` | Uvicorn worker processes |
| `APP_LOG_LEVEL` | `info` | `debug` \| `info` \| `warning` \| `error` |
| `APP_SECRET_KEY` | *(dev value)* | Session/token signing key — **change in production** |
| `API_REQUIRE_KEY` | `false` | `true` = write endpoints require the `X-API-Key` header |
| `API_KEY` | *(dev value)* | The key checked when `API_REQUIRE_KEY=true` |
| `CORS_ORIGINS` | `http://localhost:8000,…` | Comma-separated allowed origins |
| `ARTIFACTS_DIR` | `artifacts` | All generated artefacts (DB, HNSW, FNO weights, exports) |
| `MAX_UPLOAD_MB` | `64` | Upload cap for DEM/imagery ingestion |
| `FRONTEND_DIR` | `frontend` | Static studio served from this path |
| `THREE_JS_CDN` | jsdelivr r170 | Three.js source used by the WebGL viewer |

#### Block 2 — Generative / Embedding Providers

| Key | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | `local` (offline deterministic) \| `openai` |
| `EMBEDDING_FALLBACK` | `true` | Silently use `LocalEmbedder` if the provider errors |
| `EMBEDDING_DIM` | `1536` | Vector dimensionality of the hybrid index |
| `OPENAI_API_KEY` | placeholder | OpenAI credential (embeddings + optional chat) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override for Azure/OpenRouter/proxies |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536-dim embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Optional narrative generation model |
| `OPENAI_EMBEDDING_BATCH` | `64` | Batch size for embedding requests |
| `ANTHROPIC_API_KEY` | placeholder | Claude credential (reserved for narrative/pipeline narration) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model id |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Endpoint override |
| `HUGGINGFACE_API_KEY` | placeholder | HF Inference API token |
| `HUGGINGFACE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Alternative sentence embedder |
| `HUGGINGFACE_BASE_URL` | `https://api-inference.huggingface.co` | Endpoint override |

#### Block 3 — Vector Stores

| Key | Default | Purpose |
|---|---|---|
| `VECTOR_STORE` | `local-hnsw` | `local-hnsw` \| `qdrant` \| `supabase` |
| `HNSW_M` | `16` | Max links per node (recall ∝ M, latency ∝ M) |
| `HNSW_EF_CONSTRUCTION` | `100` | Build-time candidate width (higher = better recall) |
| `HNSW_EF_SEARCH` | `64` | Query-time candidate width (tune for recall/latency) |
| `HNSW_INDEX_PATH` | `artifacts/geomind_hnsw.npz` | Persisted index location |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST/gRPC endpoint |
| `QDRANT_API_KEY` | placeholder | Qdrant Cloud key |
| `QDRANT_COLLECTION` | `geomind_spatial_chunks` | Collection name |
| `QDRANT_VECTOR_SIZE` | `1536` | Must match `EMBEDDING_DIM` |
| `QDRANT_PREFER_GRPC` | `false` | Use gRPC transport when true |
| `SUPABASE_URL` | placeholder | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | placeholder | Service-role key (server-side only) |
| `SUPABASE_ANON_KEY` | placeholder | Anon key (browser) |
| `SUPABASE_SCHEMA` | `public` | Postgres schema |
| `SUPABASE_TABLE` | `spatial_chunks` | pgvector table |
| `PGVECTOR_DSN` | placeholder | Direct Postgres connection string |
| `PGVECTOR_TABLE` | `spatial_chunks` | Table used by pgvector queries |
| `PGVECTOR_DIM` | `1536` | Column dimension |

#### Block 4 — Spatial Databases

| Key | Default | Purpose |
|---|---|---|
| `SPATIAL_DB_PATH` | `artifacts/geomind_spatial.db` | SQLite file holding features/edges/scenes |
| `SPATIAL_SRID` | `4326` | WGS-84 lon/lat; all `ST_*` predicates assume it |

#### Block 5 — Geospatial Tokens / 3D Tiles / Engines

| Key | Default | Purpose |
|---|---|---|
| `MAPBOX_TOKEN` | placeholder | Mapbox satellite/terrain tiles in the studio |
| `CESIUM_ION_TOKEN` | placeholder | Cesium ion for the UE5 / Cesium globe bridge |
| `GOOGLE_MAPS_API_KEY` | placeholder | Optional imagery provider |
| `USGS_API_BASE` | USGS FDSN | Live earthquake feeds (seismic simulation source) |
| `OPEN_ELEVATION_API_URL` | open-elevation | DEM lookups for real terrain ingestion |
| `UE5_REMOTE_ENDPOINT` | `http://127.0.0.1:30001` | Unreal Engine 5 remote-control bridge |
| `UE5_API_TOKEN` | placeholder | Bearer token for the UE5 bridge |

#### Block 6 — Compute Backends

| Key | Default | Purpose |
|---|---|---|
| `TORCH_DEVICE` | `cuda` | `cuda` \| `cpu` \| `mps` (auto-falls back to NumPy) |
| `TORCH_DTYPE` | `float32` | `float32` \| `float64` |
| `TAICHI_ARCH` | `cuda` | Taichi compute architecture |
| `TAICHI_GPU_INDEX` | `0` | Which GPU Taichi targets |
| `COMPUTE_BACKEND` | `auto` | `auto` \| `numpy` \| `torch` |

#### Block 7 — Grounding & Memory

| Key | Default | Purpose |
|---|---|---|
| `RRF_K` | `60` | RRF dampening constant (60 is the Cormack et al. optimum) |
| `RRF_WEIGHT_VECTOR` | `1.0` | Weight of the dense-vector ranking |
| `RRF_WEIGHT_SPATIAL` | `1.0` | Weight of the Spatial-SQL ranking |
| `RRF_WEIGHT_GNN` | `1.0` | Weight of the GNN topology ranking |
| `RETRIEVAL_TOP_K` | `8` | How many nodes survive fusion into conditioning |
| `GNN_LAYERS` | `2` | Message-passing depth |
| `GNN_SEED` | `1337` | Seeds the fixed relational projection weights |

#### Block 8 — Latent Diffusion Engine

| Key | Default | Purpose |
|---|---|---|
| `LATENT_DIM` | `64` | DCT latent dimensionality per axis |
| `DIFFUSION_STEPS` | `48` | Reverse-diffusion steps |
| `DIFFUSION_BETA_START` | `0.0001` | Noise schedule start |
| `DIFFUSION_BETA_END` | `0.02` | Noise schedule end |
| `DIFFUSION_SEED` | `42` | Reproducible sampling |
| `SH_DEGREE` | `4` | Max spherical-harmonic degree $L$ ($1{+}L{+}L^2 = 21$ coefficients) |

#### Block 9 — Fourier Neural Operator

| Key | Default | Purpose |
|---|---|---|
| `FNO_MODES` | `8` | Fourier modes retained per axis |
| `FNO_WIDTH` | `16` | Hidden channel width |
| `FNO_LAYERS` | `4` | Spectral layers |
| `FNO_LR` | `0.0015` | Adam learning rate |
| `FNO_TRAIN_STEPS` | `300` | Optimiser steps at warm-up |
| `FNO_BATCH_SIZE` | `4` | Mini-batch size |
| `FNO_CACHE_PATH` | `artifacts/fno/fno_eroder.npz` | Trained weight cache |
| `FNO_AUTO_TRAIN` | `true` | Train on boot when the cache is missing |
| `FNO_SEED` | `7` | Weight/dataset seed |

#### Block 10 — Physics Solvers

| Key | Default | Purpose |
|---|---|---|
| `SOLVER_DEFAULT` | `hydraulic` | `hydraulic` \| `thermal` \| `seismic` \| `auto` |
| `EROSION_RAINFALL` | `0.02` | Stream-power $K$ |
| `EROSION_CAPACITY` | `0.4` | Area-exponent influence |
| `EROSION_DEPOSITION` | `0.3` | Deposition coefficient |
| `EROSION_EVAPORATION` | `0.02` | Discharge decay |
| `THERMAL_TALUS_DEG` | `34.0` | Critical repose angle $\theta_c$ |
| `THERMAL_RATE` | `0.5` | Creep diffusivity |
| `SEISMIC_WAVE_SPEED` | `2.0` | Normalised $c$ for the wave equation |
| `SEISMIC_DAMPING` | `0.004` | Leap-frog damping |
| `SIMULATION_STEPS` | `24` | Time steps per request |

#### Block 11 — Rendering / Meshing / Exports

| Key | Default | Purpose |
|---|---|---|
| `MESH_RESOLUTION` | `128` | Grid resolution for the SDF / dual contouring |
| `SPLAT_COUNT` | `20000` | 3D Gaussian budget |
| `SPLAT_OPACITY` | `0.85` | Base Gaussian opacity |
| `MESH_EXPORTS` | `obj,ply,glb,usdz` | Formats generated per request |
| `MESH_SLAB_THICKNESS` | `8.0` | Solid skirt depth so exported terrain is watertight |

#### Block 12 — Seed / Corpus

| Key | Default | Purpose |
|---|---|---|
| `SEED_RANDOM_SEED` | `2026` | Seeded feature generation |
| `CORPUS_CHUNK_SIZE` | `220` | Report chunk length (characters) |

#### Block 13 — Frontend

| Key | Default | Purpose |
|---|---|---|
| `FRONTEND_DIR` | `frontend` | Served at `/` by FastAPI |
| `THREE_JS_CDN` | jsdelivr `three@0.170.0` | Three.js version |

---

## 9. API Reference

### 9.1 Python API (engine level)

Everything below is importable without HTTP. These are the verified, working
signatures.

#### Grounding — `app.core.knowledge_graph`

```python
from app.core.knowledge_graph import get_repository, reset_repository

repo = get_repository()                       # singleton; seeds + builds on first call
result = repo.retrieve(query, top_k=8)        # Dict[str, Any]
reset_repository()                            # force a re-seed (tests)
```

`retrieve()` returns:

```jsonc
{
  "query": "…high relief fractured rock terrain…",
  "embedding_backend": "local-hash-v1",
  "evidence": 16,                             // items that had evidence
  "rankings": {                               // per-source ordered lists
    "vector":  [{"id": "rep-karakoram-2024", "score": 0.2330}, …],
    "spatial": [{"id": "aqn-gilgit",         "score": 1.0},    …],
    "gnn":     [{"id": "flt-mbt",            "score": 0.4011}, …]
  },
  "fused": [                                  // RRF output, best first
    {
      "id": "rep-chitral-accretion",
      "score": 0.0448,
      "sources": {"vector":  {"rank": 5,  "score": 0.0909, "weight": 1.0},
                  "spatial": {"rank": 3,  "score": 0.6667, "weight": 1.0},
                  "gnn":     {"rank": 14, "score": 0.0702, "weight": 1.0}},
      "name": "Chitral Accretionary Prism Engineering Geological Report",
      "kind": "report",
      "props": {"bbox": [71.0, 34.4, 72.6, 36.0]},
      "geom":  {"type": "Polygon", "coordinates": [[[71.0, 34.4], …]]},
      "bbox":  [71.0, 34.4, 72.6, 36.0]
    }
  ]
}
```

Graph and corpus introspection:

```python
payload = repo.graph_payload()      # {"nodes": [...], "edges": [...]}
repo.graph.node_count               # 41 nodes
repo.graph.edge_count               # 717 edges
repo.graph.rank_query(vec)          # GNN-only ranking
repo.graph.neighbours_of("reg-karakoram", hops=2)
repo.vector_store.size              # 42 embedded chunks
```

Low-level components:

```python
# ── embeddings ──
from app.core.embeddings import get_embedder
vecs = get_embedder().embed(["hydraulic erosion", "fault gouge"])   # (2, 1536) f32

# ── HNSW ──
from app.core.hnsw import HNSWIndex
idx = HNSWIndex(dim=1536, m=16, ef_construction=100, ef_search=64)
idx.add_batch(vecs, ["a", "b"])
hits = idx.search(vecs[0], k=5)          # [(internal_id, cosine), …]
idx.save("artifacts/geomind_hnsw.npz")
idx = HNSWIndex.load("artifacts/geomind_hnsw.npz")

# ── spatial SQL ──
from app.core.spatial_sql import SpatialDB
db = SpatialDB(settings.spatial_db_path)      # tables: features, feature_edges, scenes, features_fts
db.st_intersects("reg-karakoram")             # WHERE ST_Intersects(a.geom, b.geom) = 1
db.st_distance("flt-mkt")                     # ORDER BY ST_Distance(...)
db.fts("aquifer gneiss")                      # FTS5 lexical channel

# ── geometry kernel ──
from app.core.geometry import (intersects, contains, within, touches, overlaps,
                               distance, area_m2, length_m, centroid, bbox)
intersects(geom_a, geom_b)              # bool
area_m2(polygon)                        # m² at the polygon's mean latitude

# ── RRF ──
from app.core.rrf import rrf_fuse
fused = rrf_fuse({"vector": […], "spatial": […], "gnn": […]},
                 weights={"vector": 1.0, "spatial": 1.0, "gnn": 1.0}, k=60)

# ── relational GNN ──
from app.core.gnn import RelationalGNN
g = RelationalGNN(in_dim=1536, relations=["intersects", "contains", "mentions"])
h2 = g.propagate(features, edges)       # (N, 1536) message-passed features
hn = g.normalised(h2)
```

<!--NEXT-->











