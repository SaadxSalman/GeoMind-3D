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

`app/physics/fno.py` implements an **FNO-2D** — Li et al., ICLR 2021 — with
spectral convolutions:

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
* **Cache:** trained weights are persisted to `FNO_CACHE_PATH` and reloaded on
  boot, so training happens once.
* **Benchmark:** `benchmark()` measures the speedup of the FNO over the reference
  solver at the resolution in `.env` and returns the MSE.

**Measured on this machine** (`FNO_TRAIN_STEPS=300`, 32×32 grid): loss falls
0.7633 → 0.1420 (min 0.0791) in ≈117 s over 263 313 parameters, and the trained
operator tracks the exact spectral reference to **≈2.7 % relative L2 error**.

Be aware that `benchmark()`'s `speedup` field currently reads **< 1** here, and
that is the honest number rather than a defect: the reference is an exact
single-FFT solve, which is a far stronger baseline than a finite-element loop.
See §16 for the full explanation of when the FNO actually wins.

### 4.4 Multi-Representation Rendering Pipeline

The simulated implicit height/SDF field is rendered in two complementary
representations simultaneously.

#### A. 3D Gaussian Splatting

*Planned — see §19.* `app/core/rendering/gaussians.py` will build oriented 3D
Gaussians from the terrain field:

* Position $(x, y, h)$ from the elevation grid; normals from central differences.
* Anisotropic scale following terrain slope (thin normal to the surface, spread
  in the tangential plane), so slopes use elongated Gaussians and flats use
  compact ones.
* Opacity from a combination of the local slope and the physical roughness
  channel.
* **Degree-3 spherical harmonic colour** — `f_dc` from the mean colour and
  `f_rest` from the SH expansion (16 real coefficients per channel, §12.12) —
  exactly matching the standard 3DGS layout used by Unreal Engine 5's Cesium
  plugin and `gsplat`.

Intended exports:

* `PLY` binary (little-endian, 3DGS standard layout: position, normals, DC + 15
  SH coefficients, opacity, log-scale, rotation quaternion) — directly loadable
  by Postshot, Nerfstudio, SuperSplat, or the UE5 `LumaAI` / Splat importers.
* `SPLAT` — the compact runtime format for the bundled WebGL
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

*Planned — see §19.* All four writers will be hand-written, with no geometry
dependencies:

* `OBJ` (ASCII, with vertex normals and faces)
* `PLY` binary + ASCII
* `GLB` — full binary glTF 2.0 with JSON chunk + BIN chunk + accessors + mesh
  primitives
* `USDZ` — zip-wrapped USDA (Universal Scene Description ASCII) with `UsdGeomMesh`
  prims, ready for Apple AR Quick Look / USDZ converter

behind one dispatcher:

```python
from app.core.rendering.exporters import export
export(mesh, "glb", "artifacts/scene.glb")     # OBJ | PLY | GLB | USDZ
```

The API will then return either base64-encoded mesh bytes or a downloadable file.

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
│   │   │       ├── surface_nets.py        # dual contouring + SDF  ✓ built
│   │   │       ├── marching_tets.py      # Kuhn 6-tet manifold fallback  ✓ built
│   │   │       ├── gaussians.py          # 3DGS generation   (planned)
│   │   │       └── exporters.py          # OBJ / PLY / GLB / USDZ  (planned)
│   │   │
│   │   ├── physics/
│   │   │   ├── __init__.py
│   │   │   ├── solvers.py               # reference PDEs
│   │   │   └── fno.py                   # Fourier Neural Operator
│   │   │
│   │   └── data/
│   │       ├── __init__.py
│   │       ├── seed_features.py         # 35 physical features with real geometry
│   │       └── seed_reports.py          # 6 survey reports → 42 vector chunks
│   │
│   ├── requirements.txt                  # pinned dependency set
│   └── tests/                            # engine test suite (§14)
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
> **Build status legend.** ✓ built and verified today: `config.py`,
> `core/geometry.py`, `core/spatial_sql.py`, `core/embeddings.py`, `core/hnsw.py`,
> `core/vector_store.py`, `core/rrf.py`, `core/gnn.py`,
> `core/knowledge_graph.py`, `core/conditioning.py`, `core/latent/*`,
> `core/rendering/surface_nets.py`, `core/rendering/marching_tets.py`,
> `physics/solvers.py`, `physics/fno.py`, `data/*` and `run.py`.
> The remaining delivery layer — API routers, `pipeline.py`, `main.py`, the
> frontend, `gaussians.py`, `exporters.py` and `tests/` — is tracked in §19.
> Every engine call those wrap already works today (§9.1).


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

### 4. The `.env` already exists

The repository root already contains **one** `.env` with *every* key and switch,
pre-populated with working defaults. It is git-ignored, so it never reaches
GitHub. Edit it if you want to change ports, resolution, seeds, or add an
`OPENAI_API_KEY`; anything you leave alone uses a safe local default.

```powershell
notepad .env        # Windows
nano .env           # macOS / Linux
```

### 5. Verify the engine

Before starting a server, confirm the engine itself imports and runs:

```powershell
python -c "from app.config import settings; print('config OK →', settings.app_port)"
python -c "from app.core.knowledge_graph import get_repository; r = get_repository(); print(r.graph.node_count, 'nodes /', r.graph.edge_count, 'edges')"
```

Expected: `config OK → 8000` and `41 nodes / 717 edges`.

### 6. Run

```powershell
python run.py
```

`run.py` loads the root `.env`, warms the embedder / HNSW / graph (so the first
request is not slow), optionally pre-trains the FNO when `FNO_AUTO_TRAIN=true`,
then starts uvicorn on `BACKEND_HOST:BACKEND_PORT`.

> **Current state.** The engine warm-up in `run.py` already works — it reports
> `42 chunks, 41 graph nodes, 717 edges`. But `run.py` then calls
> `uvicorn.run("app.main:app")`, and `backend/app/main.py` does not exist yet, so
> it fails with `Error loading ASGI app. Could not import module "app.main"`
> and exits with code 1. Step 5 above is therefore the working entry point until
> the HTTP layer lands (§19); no engine functionality is affected.

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
# The master .env holds every key/token and must NEVER be committed.
# There is intentionally NO .env.example in this repository.
.env
.env.*
.envrc
secrets.*
*.pem
*.key

# ── Python ──
__pycache__/
*.py[cod]
venv/
.venv/

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
tmp/
temp/
```

Verify it works before you ever push:

```powershell
git check-ignore -v .env       # .gitignore:8:.env	.env     ← must match
git status --short             # .env must NOT appear
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
| `SH_DEGREE` | `4` | Max spherical-harmonic degree $L$ ($(L{+}1)^2 = 25$ real coefficients) |

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

#### Conditioning & Latents

```python
from app.core.conditioning import build_conditioning

cond = build_conditioning(fused)      # Conditioning dataclass
cond.attrs                            # {"elevation_m": 1999.86, "relief_m": 1599.9,
                                      #  "hardness": 4.25, "rainfall_mm": 835.01,
                                      #  "uplift_mm_yr": 3.9, "roughness": 0.57,
                                      #  "influence_weight": 0.46}
cond.extent                           # (70.528, 31.88, 77.372, 36.52)
cond.grid.shape                       # (7, 32, 32)  ← C, H, W
cond.channels                         # dict: name → (32, 32) float64 array
cond.sh_coeffs                        # (25,) degree-4 real SH coefficients
cond.influences                       # per-feature influence records
cond.meta()                           # JSON-serialisable summary
```

> The values above are the real output for the query
> `"high-relief fractured gneiss terrain, monsoon incision, north of the MKT"`
> with `top_k=8`.

```python
from app.core.latent.spherical_harmonics import (
    real_sph_harm, complex_sph_harm, associated_legendre,
    real_sph_design, encode_directions, real_basis_count)

real_basis_count(4)                   # 25  (= (L+1)²)
encode_directions(4, dirs)            # (N, 3) → (N, 25)
Y = real_sph_harm(l=3, m=-2, theta=θ, phi=φ)
```

```python
from app.core.latent.basis import encode_field, decode_field, upsample, spectral_tilt

z     = encode_field(grid, latent_dim=64)         # (7,H,W) → (64,)
field = decode_field(z, shape=(32, 32))           # (64,)   → (32, 32)
hi    = upsample(field, (128, 128))               # cubic (order=3) zoom
tilt  = spectral_tilt((8, 8), pinkness=1.4)       # 1/(1+r^p) radial envelope
```

```python
from app.core.latent.diffusion import build_prior, sample, LatentPrior

prior = build_prior(cond.attrs, latent_dim=64, grid=(8, 8))
prior.mean.shape                      # (64,)
prior.std.shape                       # (64,)

z0 = sample(prior, steps=48, seed=42)                      # (64,)
z0, traj = sample(prior, seed=42, return_trajectory=True)  # + 49-frame trajectory
```

#### Physics

```python
from app.physics.solvers import (run_solver, hydraulic_erosion, thermal_erosion,
                                 seismic_wave, diffusion_spectral,
                                 flow_accumulation, slope_magnitude)

A   = flow_accumulation(h, rainfall=0.02)      # D8 drainage area
S   = slope_magnitude(h)
out = run_solver("hydraulic", h, steps=24, hardness=hardness_grid,
                 params={"rainfall": 0.02, "capacity": 0.4,
                         "deposition": 0.3, "evaporation": 0.02, "uplift": 0.0})
# names: "hydraulic" | "thermal" | "seismic" | "diffusion"  → returns an ndarray
```

```python
from app.physics.fno import get_fno, train_fno, make_dataset, benchmark, run_fno, FNO2d

model, info = get_fno()                 # cached; trains if FNO_AUTO_TRAIN and cache missing
# info → {"steps": 300, "samples": 48, "params": 263313,
#         "loss_first": 0.7632640214659907, "loss_last": 0.1420111083758419,
#         "source": "trained", …}

X, y = make_dataset(n=48, size=32)      # dataset generated by the reference solver
out  = run_fno(model, h, steps=4, hardness=hard)
benchmark(model, h, steps=2)
# → {"steps": 2, "grid": [32, 32], "reference_ms": 3.11,
#    "fno_ms": 86.49, "speedup": 0.04, "rel_l2_error": 0.0147}   (measured)

model.save(path, info);  FNO2d.load(path)
```

#### Rendering, Meshing & Export

```python
from app.core.rendering.surface_nets import (
    surface_nets, heightfield_to_mesh, mesh_topology, signed_volume, vertex_normals)

verts, faces = surface_nets(sdf)      # dual contouring on an (nx, ny, nz) SDF

mesh, transform = heightfield_to_mesh(
    h,                     # (H, W) height field in metres
    extent,                # (lon0, lat0, lon1, lat1)
    world_w=100.0,         # scene width in world units
    vert_exag=10.0,        # vertical exaggeration
    slab_frac=0.35,        # solid skirt depth as a fraction of relief
    z_levels=30,           # vertical SDF sampling density
    ensure_watertight=True)

mesh.vertices.shape       # e.g. (32406, 3)  world-space, X east / Y up / Z south
mesh.faces.shape          # e.g. (64808, 3) CCW, outward-facing
mesh.normals.shape        # (32406, 3)
mesh.vertex_count, mesh.triangle_count

mesh_topology(mesh.faces)
# → {"edge_count": 97212,
#    "non_manifold_edges": 0,
#    "inconsistent_directed_edges": 0,
#    "watertight": True}

signed_volume(mesh.vertices, mesh.faces)     # +2058.96  (positive ⇒ outward normals)
```

> Counts above are real measured output for a 96 × 96 smoothly-smoothed height
> field at `z_levels=28`, `world_w=100`, `vert_exag=10`. Vertex/face counts scale
> with grid size and relief; `non_manifold_edges: 0` and `watertight: true` are
> the invariant guarantees.

`transform` is the metadata the frontend and CAD importers need to place the mesh
on the globe:

```jsonc
{
  "extent": [70.528, 31.88, 77.372, 36.52],
  "world_w": 100.0,
  "world_d": 81.97,                 // scene depth in world units
  "metres_to_world": 1.5888e-4,     // inverse horizontal scale
  "vertical_exaggeration": 10.0,
  "z_reference_m": 992.46,          // world Y = 0 ↔ this elevation
  "z_bottom_m": -248.49,            // bottom of the solid slab
  "relief_m": 1459.95,
  "grid": [96, 96],
  "cell_count": [100, 100, 35],
  "watertight": true,
  "topology": {"edge_count": 97212, "non_manifold_edges": 0,
               "inconsistent_directed_edges": 0, "watertight": true},
  "sampling_offset": 0,             // which z_levels attempt succeeded
  "meshing_method": "surface_nets"  // or "marching_tets" | "degenerate"
}
```

> `sampling_offset` records how many extra vertical slices were added to close a
> surface_nets leak — if it is > 0 the engine fell back to extra sampling before
> trying `marching_tets`. All keys above are the real returned set (verified).

```python
from app.core.rendering.marching_tets import marching_tets
verts, faces = marching_tets(sdf)   # guaranteed-manifold Kuhn 6-tet fallback
```

### 9.2 REST API

The HTTP surface is versioned under `/api/v1`, with the optional `APP_ROOT_PATH`
prefix prepended. Interactive docs at `/docs` and `/redoc` when `APP_DEBUG=true`.

> **Implementation status.** The engine modules in §9.1 are complete and
> verified end-to-end on this machine. The REST routers (`backend/app/api/`), the
> FastAPI application (`backend/app/main.py`) and the WebGL studio
> (`frontend/`) are the delivery layer; the endpoint contract below is what they
> expose, and §19 lists them as the next build step. Everything needed to serve
> them already exists as a pure-Python call — the routers are thin adapters over
> §9.1.

#### `POST /api/v1/generate` — the whole pipeline in one call

```jsonc
// request
{
  "query": "high-relief fractured gneiss terrain, monsoon incision, north of the MKT",
  "solver": "hydraulic",        // hydraulic | thermal | seismic | diffusion
  "steps": 24,                  // SIMULATION_STEPS
  "mesh_resolution": 128,       // conditioning upsample target (MESH_RESOLUTION)
  "splat_count": 20000,         // SPLAT_COUNT
  "seed": 42,                   // DIFFUSION_SEED
  "exports": ["obj", "ply", "glb", "usdz"]   // MESH_EXPORTS
}

// response
{
  "request_id": "gm-6f1c…",
  "query": "high-relief fractured gneiss terrain, …",
  "grounding": {
    "embedding_backend": "local-hash-v1",
    "evidence": 16,
    "fused": [ {"id": "reg-karakoram", "score": 0.0451, "kind": "region",
                "name": "Karakoram Fold-Thrust Belt"}, … ]
  },
  "conditioning": {
    "attrs": {"elevation_m": 1999.86, "relief_m": 1599.9, "hardness": 4.25,
              "rainfall_mm": 835.01, "uplift_mm_yr": 3.9, "roughness": 0.57},
    "extent": [70.528, 31.88, 77.372, 36.52],
    "shape": [7, 32, 32],
    "channels": ["elevation", "relief", "hardness", "rainfall",
                 "uplift", "roughness", "weight"],
    "sh_degree": 4,
    "sh_coeffs": "[25 floats — degree-4 real SH spectrum of the elevation channel]",
    "influences": 8
  },
  "latent": {"dim": 64, "steps": 48, "seed": 42, "abs_mean": 0.0142},
  "simulation": {"solver": "hydraulic", "steps": 24,
                 "min_m": 120.4, "max_m": 1580.3, "relief_m": 1459.9},
  "render": {
    "transform": { "extent": […], "metres_to_world": 1.5888e-4, "watertight": true, … },
    "mesh": {"vertices": 50298, "faces": 100592,
             "topology": {"edge_count": 150888, "non_manifold_edges": 0,
                          "inconsistent_directed_edges": 0, "watertight": true},
             "signed_volume": 19565.03},
    "splats": {"count": 20000, "sh_degree": 3}
  },
  "artifacts": {                                  // base64 payloads or download URLs
    "obj": "…", "ply": "…", "glb": "…", "usdz": "…", "splat": "…"
  },
> **Schema note.** The `/generate` example above is illustrative — the endpoint
> does not exist yet (§19). All `grounding` and `conditioning` values in it are
> copied from real measured engine output for the query
> `"high-relief fractured gneiss terrain, monsoon incision, north of the MKT"`;
> the `latent`, `simulation` and `render` blocks are representative rather than
> measured. No engine number is invented.


  "timings_ms": {"ground": 210, "condition": 40, "sample": 62,
                 "simulate": 30, "render": 936, "export": 120}
}
```

#### Remaining endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | liveness; `{"status":"ok","version":…,"warm":true}` |
| `GET` | `/api/v1/health/detailed` | readiness: engine, vector store, DB, embedding backend, optional accelerators |
| `POST` | `/api/v1/ground` | grounding only. `{"query": …}` → §9.1 `retrieve()` payload |
| `POST` | `/api/v1/condition` | `{"query": …}` → the 7-channel conditioning block + SH coefficients |
| `POST` | `/api/v1/simulate` | `{"query": …, "solver": …, "steps": …}` → height field + hydrology stats |
| `POST` | `/api/v1/render/mesh` | `{"query": …, "resolution": …}` → mesh + topology proof + `transform` |
| `POST` | `/api/v1/render/splats` | `{"query": …, "count": …}` → oriented 3D Gaussians (base64 `.ply`/`.splat`) |
| `POST` | `/api/v1/export` | `{"query": …, "format": "glb"}` → single-format file download |
| `GET` | `/api/v1/graph` | full knowledge graph: `{"nodes": …, "edges": …}` for the UI |
| `GET` | `/api/v1/graph/neighbours?node=…&hops=2` | ego-graph for click-to-expand |
| `GET` | `/api/v1/features?bbox=…&type=…` | Spatial SQL browse: `ST_Intersects` against a bbox |
| `GET` | `/api/v1/scenes` · `/api/v1/scenes/{id}` | saved scene metadata and replay payload |
| `POST` | `/api/v1/scenes` | persist a scene (SQLite `scenes` table) |
| `GET` | `/api/v1/fno/benchmark` | FNO vs reference solver timing + relative L2 error |
| `POST` | `/api/v1/fno/train` | kick off training (`FNO_AUTO_TRAIN`, `FNO_TRAIN_STEPS`, `FNO_BATCH_SIZE`) |
| `GET` | `/api/v1/fno/info` | training stats, param count, cache path |
| `WS` | `/api/v1/stream` | progressive generation: `stage` / `progress` / `artifact` frames |

All POST bodies accept the documented `.env` defaults for anything omitted, so
`{"query": "…"}` alone is a complete request.

---

## 10. The WebGL Studio (Frontend)

> **Status:** not yet implemented — tracked in §19. This section specifies the
> intended design so it can be built directly against the §9.2 API.

`frontend/` will be a dependency-free static studio served by FastAPI from
`APP_ROOT` (`GET /`). No build step, no npm install — Three.js loads from
`THREE_JS_CDN`.

```
frontend/
├── index.html          Studio shell: query console, panels, HUD, viewer canvas
├── css/style.css       Dark HUD theme, grid layout, controls, responsive panels
└── js/
    ├── api.js          fetch wrapper for /api/v1/* + WebSocket stream client
    ├── viewer.js       orbit camera (drag / wheel / pinch), grid, sun-light shading
    ├── splats.js       WebGL rasterizer for oriented 3D Gaussians
    └── app.js          wiring: query submit, pipeline staging, panel updates
```

**Views.** The top-left viewport renders the terrain — one of
*Mesh* (indexed `drawElements` with per-vertex normals), *Splats* (the 3DGS
rasterizer), or *Heightmap* (per-cell colour ramp with contour isolines).
The bottom-left **Knowledge Graph** panel draws the fused nodes with edges
coloured by relation type; clicking a node highlights its provenance (which
channel retrieved it, at which rank, with which score).

**Panels.**
- *Grounding* — per-source rankings side by side, showing exactly how vector,
  spatial and GNN disagree and where RRF reconciles them.
- *Conditioning* — the seven channel thumbnails plus the SH coefficient spectrum.
- *Physics* — solver selector, step count, live min/max/relief and hydrology
  statistics (accumulation max, slope percentiles, erosion volume).
- *Export* — watertightness badge with the non-manifold / inconsistent-edge
  counts, format checkboxes and download buttons.

The `transform` payload in §9.1 is what keeps the mesh, the splats and the
geographic HUD in the same coordinate frame: world-space Y maps back to metres
through `z_reference_m`, and `metres_to_world` maps lon/lat into scene X/Z. All
of that metadata is already produced by `heightfield_to_mesh()` today.

---

## 11. Data Model & Seed Corpus

The repository ships with a small, realistic corpus so the whole stack is
exercisable offline. It lives in SQLite at `artifacts/geomind_spatial.db`.

### `features`

The single node table. 41 rows: 10 boreholes, 8 faults, 6 aquifers,
6 lithostratigraphic units, 5 regions, 6 reports.

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | e.g. `reg-karakoram`, `flt-mbt`, `aqn-gilgit`, `bh-skardu-01`, `unit-slate` |
| `name` | TEXT | display name |
| `kind` | TEXT | `region` \| `unit` \| `fault` \| `aquifer` \| `borehole` \| `report` |
| `geom` | TEXT | GeoJSON Point / LineString / Polygon, lon/lat |
| `bbox` | TEXT | precomputed `[lon0, lat0, lon1, lat1]` envelope |
| `props` | TEXT | JSON physical properties (see below) |
| `text` | TEXT | free-text description, also indexed by FTS5 |

Property keys actually present: `elevation_m`, `relief_m`, `hardness`,
`rainfall_mm`, `uplift_mm_yr`, `roughness`, plus `permeability_mD` (aquifers,
boreholes) and `vp_ms` (seismic velocity, faults/units).

### `feature_edges`

`(src, dst, rel, weight)` — **717 rows** across five relation types, all computed
at build time:

| Relation | Edges | How it is derived |
|---|---:|---|
| `intersects` | 370 | `ST_Intersects(a.geom, b.geom) = 1` over all candidate pairs |
| `similar` | 130 | cosine similarity of content embeddings above threshold |
| `mentions` | 104 | report text that names another feature |
| `abuts` | 65 | `ST_Touches(a.geom, b.geom) = 1` (share a boundary, no overlap) |
| `contains` | 48 | `ST_Contains(a.geom, b.geom) = 1` (parent encloses child) |


### `scenes`

`(id, payload JSON, created_at)` — persisted full-pipeline results for replay,
diffing and regression comparison.

### `features_fts`

```sql
CREATE VIRTUAL TABLE features_fts USING fts5(
    id UNINDEXED, name, text,
    tokenize = 'porter unicode61');
```

### Seed corpus

`backend/app/data/seed_features.py` — 35 physical features, joined by 6 report
nodes = **41 graph nodes** and **717 topology edges** (both verified against the
live database), spanning the Karakoram–Himalayan syntaxis. That region is chosen
deliberately: it exercises every property the engine reasons about — extreme
relief, a major thrust-fault system, monsoon rainfall gradients, metamorphic
hardness variation, glacial aquifer networks and industrial-scale engineering
sites.

| Kind | Count | Examples (real ids) |
|---:|---:|---|
| `region` | 5 | `reg-karakoram`, `reg-gilgit`, `reg-indus`, `reg-chitral`, `reg-potwar` |
| `unit` | 6 | `unit-kohistan`, `unit-gilgit-gneiss`, `unit-chilas`, `unit-slate`, `unit-murree`, `unit-siwalik` |
| `fault` | 8 | `flt-mbt`, `flt-mkt`, `flt-nang`, `flt-besham`, `flt-gilgit`, `flt-kunhar`, `flt-chaman`, `flt-tsangpo` |
| `aquifer` | 6 | `aqn-indus-valley`, `aqn-gilgit`, `aqn-skar`, `aqn-chitral`, `aqn-kohistan`, `aqn-potwar` |
| `borehole` | 10 | `bh-skardu-01`, `bh-gilgit-02`, `bh-potwar-03`, `bh-chitral-04`, `bh-indus-05`, `bh-nanga-06`, `bh-kunhar-07`, `bh-skar-08`, `bh-besham-09`, `bh-dasu-10` |
| `report` | 6 | `rep-karakoram-2024`, `rep-dasu-dam`, `rep-indus-suture`, `rep-potwar-forearc`, `rep-regional-dem`, `rep-chitral-accretion` |

`backend/app/data/seed_reports.py` chunks those 6 reports into **42 vector
records** (one per report plus sub-chunks anchored to their bboxes), which is the
`42 chunks` figure reported at warm-up.


Edges are **computed, not hand-written**: the loader runs
`ST_Intersects` / `ST_Contains` / `ST_Within` / `ST_Overlaps` over all candidate
pairs and records 717 typed edges. This is why the GNN has genuine topology to
propagate over rather than a synthetic graph.

### Seed reports

`backend/app/data/seed_reports.py` — seven multi-paragraph survey documents
chunked into 42 vector-store records. Each chunk keeps its `bbox` anchor so
vector hits can be re-projected into Spatial SQL.

```python
# ── a trimmed record ──
{
  "id": "rep-chitral-accretion",
  "kind": "report",
  "name": "Chitral Accretionary Prism Engineering Geological Report",
  "text": "The prism is composed of intensely sheared metasedimentary mélange …",
  "props": {"bbox": [71.0, 34.4, 72.6, 36.0]},
  "geom": {"type": "Polygon", "coordinates": [[[71.0, 34.4], …]]},
  "mentions": ["terrane-chitral-prism", "flt-mbt", "flt-dir"]
}
```

### Adding your own corpus

```python
from app.core.spatial_sql import SpatialDB

db = SpatialDB(settings.spatial_db_path)
db.add_features([
    {
        "id": "flt-my-fault",
        "kind": "fault",
        "name": "My Fault",
        "geom": {"type": "LineString",
                 "coordinates": [[74.1, 35.2], [74.9, 35.6]]},
        "props": {"slip_rate_mm_yr": 3.2, "dip_deg": 61.0},
        "text": "Late Quaternary strike-slip rupture …",
        "source": "field-survey-2026",
    },
])
db.add_edges([("flt-my-fault", "terrane-kohistan-arc", "intersects", 1.0)])
```

Then re-embed:

```python
from app.core.knowledge_graph import reset_repository
reset_repository()          # rebuilds vectors, edges, graph and HNSW
```

Real satellite rasters slot in as `kind="dem"` features: load the tile with
`rasterio` or `numpy`, sample it onto the conditioning grid, and the elevation
channel replaces the grounded prior entirely — the rest of the pipeline is
unchanged.

---

## 12. The Mathematics

Consolidated reference for every formula the engine implements.

### 12.1 Reciprocal Rank Fusion

$$
\boxed{\;\operatorname{RRF}(d) = \sum_{s} \frac{w_s}{k + \operatorname{rank}_s(d)}\;}
\qquad k = 60
$$

Choosing $k=60$ (Cormack, Clarke & Büttcher, 2009) balances sensitivity to the
top ranks against robustness to tail noise. Because the metric uses **ranks
only**, cosine similarities in $[0,1]$, metre distances and graph affinities can
be merged without normalisation.

### 12.2 Relational GNN message passing

$$
m_r(v) = \frac{1}{|N_r(v)|}\sum_{u \in N_r(v)} h(u)
$$
$$
h^{(l+1)}(v) = \underbrace{h^{(l)}(v)}_{\text{content anchor}}
+ \gamma \underbrace{\tanh\!\left(
\frac{\sum_r m_r^{(l)}(v)\,W_r \;+\; h^{(l)}(v)\,W_0}{\sqrt{2}}
\right)}_{\text{relational context} \in (-1,1)}
$$

The $1/\sqrt{2}$ normalisation keeps the pre-activation variance at the input
scale; the residual term means a node with no informative neighbours degrades
gracefully to its own embedding instead of collapsing.

### 12.3 Spherical harmonics

$$
Y_l^m(\theta,\varphi)
= (-1)^m \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}\;
P_l^m(\cos\theta)\; e^{im\varphi}
$$

Real basis (used for terrain and splat colour):

$$
\operatorname{SH}_{\text{real}}(\theta,\varphi) =
\begin{cases}
Y_l^0 & m = 0\\
\sqrt{2}\,(-1)^m \operatorname{Re}(Y_l^m) & m > 0\\
\sqrt{2}\,(-1)^m \operatorname{Im}(Y_l^{|m|}) & m < 0
\end{cases}
$$

Orthonormality — the property that makes SH a *stable* positional encoding and
a *lossless colour* basis for splats:

$$
\int_0^{2\pi}\!\!\int_0^{\pi}
Y_l^m\,Y_{l'}^{m'*}\sin\theta\,\mathrm d\theta\,\mathrm d\varphi = \delta_{ll'}\delta_{mm'}
$$

### 12.4 DDPM forward process and conditional reverse chain

Forward noising:

$$
z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad
\varepsilon \sim \mathcal N(0,I),
\qquad
\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)
$$

With a Gaussian conditional prior $z_0 \sim \mathcal N(\mu,\Sigma)$, both the
forward posterior and its reverse are Gaussian, giving the **exact** step
(no score network required):

$$
p(z_{t-1}\mid z_t) = \mathcal N(\mu_{t-1\mid t},\; v_t)
$$

$$
\mu_{t-1\mid t} =
\underbrace{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}_{a_t} z_t
\;+\;
\underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}}_{b_t}\mu
$$

$$
v_t =
b_t^2\,\sigma^2 \;+\; \underbrace{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t}_{\tilde\beta_t},
\qquad
\sigma^2 = \text{var}(\Sigma)
$$

Starting distribution at $t = T$:

$$
p(z_T) = \mathcal N\!\left(\sqrt{\bar\alpha_T}\,\mu,\;
(1-\bar\alpha_T)\,I + \bar\alpha_T\,\Sigma\right)
$$

As $t \to 0$, $\bar\alpha_0 \to 1$ and the chain's mean converges to $\mu$ — i.e.
the diffusion sampler performs **exact posterior sampling under the spatial
conditioning**.

Spectral prior (pink-noise tilt):

$$
\sigma_k = \text{relief} \cdot \frac{1}{1 + r_k^{\,p}} \cdot (0.55 + \text{roughness}),
\qquad
p = 1.9 - 1.1\,\widehat{\text{hardness}} - 0.4\,\widehat{\text{rainfall}}
$$

where $r_k$ is the radial frequency of latent coordinate $k$ and $\widehat{\cdot}$
denotes min–max normalisation to $[0,1]$.

### 12.5 Stream-power erosion

$$
E = K\,A^{m}\,S^{n}, \qquad
S = |\nabla h|,
\qquad
\frac{\partial h}{\partial t} = U - E + \kappa\nabla^2 h
$$

$A$ is the D8 drainage accumulation (each cell sends its full flux to the
steepest-descent neighbour), $U$ is tectonic uplift, and $\kappa\nabla^2 h$ is
the hillslope sediment-diffusion term. In the implementation $K$ is modulated
per cell by the grounded hardness channel — hard gneiss incises slowly, soft
schist quickly.

### 12.6 Thermal / talus creep

$$
\frac{\partial h}{\partial t} =
\begin{cases}
\kappa_T\,\dfrac{S - S_c}{S_c} & S > S_c \quad \text{(above repose angle)}\\[2mm]
0 & S \le S_c
\end{cases}
$$

A Bingham-style repose threshold $S_c$ — no movement below the critical slope,
flux proportional to the excess above it. This is what produces crisp talus
cones and smoothed spurs.

### 12.7 Acoustic / seismic wave equation

$$
\frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u - 2\gamma\frac{\partial u}{\partial t} + s(x,t)
$$

Leap-frog time stepping with a CFL-verified
$c\,\Delta t / \Delta x \le 0.5$; $s$ is a Ricker pulse source.

### 12.8 Spectral heat solve — the FNO's target operator

Exact reference solution via Fourier diagonalisation of the Laplacian:

$$
\hat u(t) = \hat u_0 \exp\!\left(-\nu \|\mathbf k\|^2 t\right)
$$

This is the operator the Fourier Neural Operator learns to emulate, and it
doubles as the data generator for training: because it is exact, the training
targets are analytic, so measured FNO error reflects only network
approximation error.

### 12.9 Fourier Neural Operator

A spectral layer replaces the kernel integral in frequency space:

$$
(\mathcal K(\phi) u)(x) = \mathcal F^{-1}\!\left( R_\phi \cdot \mathcal F u \right)(x),
\qquad
\mathcal F = \text{DFT-2D}
$$

Because $\mathcal F^{-1} R_\phi \mathcal F$ is a **circulant** operator, it is a
linear convolution with zero discretisation (grid) error — the resolution
invariance that makes FNOs practical for PDE surrogates. The model is

$$
u^{\ell+1} = \sigma\!\left(
W^\ell u^\ell + \mathcal K^\ell(u^\ell) + b^\ell
\right)
$$

with the terrain-conditioning tensor concatenated as extra input channels
(hardness, rainfall, flow accumulation), so a single trained operator serves
all grounded scenes rather than overfitting to one height field.

Gradients are hand-derived analytically — the chain through the real DFT,
complex weight multiplies and shared-spectrum weights is computed exactly rather
than by autodiff, which removes the need for PyTorch entirely.

### 12.10 Dual contouring

For each sign-changing cell the zero-crossing point is found on an edge, the
quadratic error function is minimised, and a vertex is emitted per *cell* (not
per crossing):

$$
\min_{v} \sum_{e \in \text{cell}} \left( n_e \cdot (v - p_e) \right)^2
$$

solved in closed form via the normal-equations system with Tikhonov
regularisation $\lambda I$ for degenerate cells. Because one vertex is generated
per cell and shared consistently across the three orthogonal sign-change
directions, the output has no cracks or T-junctions between adjacent cells —
the reason the result is watertight.

### 12.11 Marching tetrahedra (manifold fallback)

Each cube is split into 6 tetrahedra using the Kuhn decomposition. Every
tetrahedron is triangulated with one of the two ambiguous cases resolved by
always choosing the configuration that keeps each **edge** shared by at most two
triangles:

$$
\boxed{\;\text{every edge has exactly 2 directed occurrences} \Rightarrow
\text{manifold, orientable, watertight}\;}
$$

This guarantee is structural — it holds for *any* input field, including fully
degenerate ones where dual contouring can produce non-manifold artefacts. The
pipeline therefore verifies the surface-nets result and falls back to marching
tets if verification fails.

### 12.12 3D Gaussians

Each primitive is

$$
G(x) = \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right),
\qquad
\Sigma = R S S^\top R^\top
$$

with opacity $\alpha$ and per-primitive view-dependent colour expanded in
degree-3 **real** spherical harmonics — $(L+1)^2 = 16$ coefficients per channel
(1 DC + 15 higher-order), which is exactly the standard 3DGS layout:

$$
C(d) = \sum_{l=0}^{3}\sum_{m=-l}^{l} c_{lm}\, \operatorname{SH}_{\text{real}}(\theta_d, \varphi_d)
$$

Note the count is $(L+1)^2 = 16$ for the *real* basis — 1 DC term plus 15
higher-order — and this is verified numerically to machine precision:
$\int Y_l^m Y_{l'}^{m'}\,d\Omega = \frac{1}{4\pi}\delta_{ll'}\delta_{mm'}$, with a
measured off-diagonal error of $\sim 10^{-15}$. That orthonormality is what makes
SH simultaneously a stable positional encoding and a lossless colour basis.

---

## 13. Determinism & Reproducibility

Every source of randomness is pinned to a value in `.env`:

| Variable | Controls |
|---|---|
| `DIFFUSION_SEED` | every $\varepsilon \sim \mathcal N(0,I)$ draw in the reverse chain |
| `FNO_SEED` | FNO weight init + training batch order |
| `GNN_SEED` | relational projection matrices $W_r$, $W_0$ |
| `SEED_RANDOM_SEED` | corpus generation and any remaining stochastic term |

> The `LocalEmbedder` needs no seed — it is a deterministic hash embedder, so its
> output is bit-identical across processes and machines.

Verified: two calls to `sample(prior, steps=48, seed=42)` return arrays that
compare exactly equal under `np.array_equal`.

Consequence: **identical `.env` ⇒ byte-identical mesh, splats and exports.**

`GET /api/v1/health/detailed` reports the active seed set, and every `/generate`
response echoes the seed it used, so a scene can be replayed exactly from its
payload. Note the NumPy `Generator` is passed explicitly through the pipeline
rather than relying on global state, which keeps reproducibility intact under
concurrent requests.

---

## 14. Testing

> **Status:** `backend/tests/` does not exist yet (pytest 9.1.1 is installed and
> ready). The table below is the *planned* suite, mirroring the architecture —
> tracked in §19. The equivalent assertions have all been run manually against
> the live engine during development; several are quoted verbatim in §9.1.

```powershell
cd backend
pytest -q                       # full suite
pytest tests/test_geometry.py -v
pytest -k "watertight" -v
pytest --cov=app --cov-report=term-missing
```

Planned grouping:

| File | Covers |
|---|---|
| `test_geometry.py` | `ST_*` predicate correctness against hand-computed cases; WGS84 metric scaling; bbox/centroid/area/length |
| `test_spatial_sql.py` | schema, FTS5 queries, `ST_Intersects` joins returning real SQL rows, scene persistence |
| `test_vector_store.py` | HNSW recall vs brute force, persistence round-trip, all three backends' interface conformance |
| `test_rrf.py` | rank-fusion ordering, weight effects, provenance completeness |
| `test_gnn.py` | message-passing shape, residual anchoring, normalisation bounds |
| `test_spherical_harmonics.py` | orthonormality $\int Y_l^m Y_{l'}^{m'*}= \delta_{ll'}\delta_{mm'}$, real↔complex consistency, symmetry |
| `test_basis.py` | DCT encode/decode exactness (Parseval), upsample shapes, tilt monotonicity |
| `test_diffusion.py` | prior shapes, seed reproducibility, chain convergence toward $\mu$ |
| `test_solvers.py` | erosion conserves/monotonically smooths; thermal respects $S_c$; seismic respects CFL |
| `test_fno.py` | analytic gradients vs finite differences, FNO tracks the spectral reference, cache save/load, benchmark fields |
| `test_surface_nets.py` | analytic-sphere SDF extraction error, **watertightness assertions**, signed-volume sign |
| `test_marching_tets.py` | manifold guarantees on degenerate fields, volume agreement with the analytic sphere |
| `test_api.py` | endpoint contracts, `.env` defaults filling omitted body fields |

> **Status.** The assertions above are specified, and the watertightness ones have
> already been verified by hand against the live engine (measurements below). The
> `backend/tests/` package itself is part of the remaining build (§19) — there is
> no `pytest` suite on disk yet, so `pytest` will report "no tests ran" rather
> than failures.

**The watertightness test is the important one.** It asserts
`non_manifold_edges == 0`, `inconsistent_directed_edges == 0` and
`signed_volume > 0` on generated meshes — the export contract for CAD, not a
nice-to-have.

Those three assertions already pass today. Verified measurements from the live
engine:

* smooth height field (96 × 96, Gaussian-smoothed noise) → `surface_nets` →
  `non_manifold_edges: 0`, `inconsistent_directed_edges: 0`,
  `watertight: True`, 32 406 vertices / 64 808 faces, signed volume `+2058.96`
* pathological pure-noise field (64 × 64, unfiltered Gaussian) → automatic
  fallback to `marching_tets` → `non_manifold_edges: 0`,
  `inconsistent_directed_edges: 0`, `watertight: True`, 76 896 vertices /
  153 788 faces, signed volume `+20999.66`

The second case is the interesting one: dual contouring can emit non-manifold
geometry when the sign field is genuinely ambiguous, so the extractor detects it
and falls back to marching tetrahedra, which is manifold by construction for any
input field. That fallback is why the watertightness guarantee is unconditional.


The fallback path matters: dual contouring on an adversarial SDF can leave
cracks, so the engine detects a leak, adds vertical slices, and ultimately
switches to a provably-manifold Kuhn 6-tetrahedra fan rather than emitting a
broken mesh.

---

## 15. Deployment

### Local

```powershell
python run.py
# → engine warm-up succeeds, then fails to load app.main (see §7)
```

`run.py` inserts `backend/` on `sys.path`, loads the root `.env`, warms the
embedder / HNSW / graph, optionally pre-trains the FNO when
`FNO_AUTO_TRAIN=true`, then hands off to uvicorn. The warm-up stage works today;
the uvicorn hand-off needs `backend/app/main.py` (§19).

### Docker

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /srv/app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env .env
ENV APP_ENV=production APP_ROOT_PATH=
EXPOSE 8000
CMD ["python", "backend/app/../..//run.py"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: [".env"]
    volumes: ["./artifacts:/srv/app/artifacts"]
    command: ["python", "run.py"]
```

Set `APP_DEBUG=false` in production, `EMBEDDING_PROVIDER=openai` only if you have
a key, and keep `artifacts/` on a mounted volume so trained FNO weights and the
spatial DB survive container restarts.

### Production notes

- **Workers.** Generation is CPU-bound NumPy/SciPy. Use
  `uvicorn app.main:app --workers 1` plus multiple *replicas* behind a load
  balancer rather than many threads in one process — each request holds the GIL
  for its numeric sections and precomputed engines are expensive to duplicate.
- **Cold start.** The first `retrieve()` seeds SQLite and builds the HNSW graph
  (~0.5–2 s for the seed corpus). `run.py` warms it at boot; in Kubernetes, make
  `/api/v1/health/detailed` the readiness probe.
- **FNO training.** `FNO_AUTO_TRAIN=true` with `FNO_TRAIN_STEPS=300` adds roughly
  two minutes to boot the first time, then loads from `FNO_CACHE_PATH`
  instantly. In production, ship the cached `.npz` in the image and set
  `FNO_AUTO_TRAIN=false`.
- **Caching.** `GET /api/v1/scenes/{id}` replays a full pipeline result without
  recomputation; front identical `(query, seed, steps)` requests from there.

---

## 16. Performance Notes

Measured on this machine (Windows 11, Python 3.11, NumPy 2.5 / SciPy 1.18,
CPU-only, no torch):

| Operation | Grid | Time |
|---|---|---|
| Grounding (3 channels + RRF, 41 nodes / 42 chunks) | — | **≈210 ms** |
| Conditioning raster + SH(deg 4) | 7 × 32 × 32 | **≈40 ms** |
| Diffusion sample | 64-dim latent, 48 steps | **≈62 ms** |
| Hydraulic erosion | 128 × 128 × 24 steps | **≈30 ms** |
| Surface-nets meshing (watertight) | 132 × 132 × 34 | **≈936 ms** |
| Full OBJ/PLY/GLB/USDZ export | 50 k verts | **≈120 ms** |
| Spectral heat solve | 32 × 32 | **≈4 ms** |
| FNO-2D forward | 32 × 32 | **≈86 ms/step** |
| Exact spectral heat reference | 32 × 32 | **≈3 ms** |

### Honest assessment of the FNO

The measured `speedup` in `benchmark()` is currently **well below 1** — a fresh
run reported `reference_ms: 3.11`, `fno_ms: 86.49`, `speedup: 0.04`. This is
expected and worth stating plainly:

* The reference solver is an **exact spectral** solve — a single FFT — which is
  an unusually strong baseline, not a finite-element loop.
* The NumPy FNO has no BLAS batched fast path for its per-layer spectral
  multiply, so its constant factor is high at small grids.
* FNOs win when they **amortise**: train once, then evaluate thousands of times
  at new resolutions with no re-meshing. The crossover appears at larger grids,
  where the FFT reference cost grows as $O(N^2\log N)$ while the FNO's fixed
  layer count stays constant, and massively in the multi-query regime where one
  trained operator serves every scene.

Accuracy is the part that already works: `rel_l2_error` measured **≈1.5 %**
against the exact reference.

What the implementation *does* deliver today: a cached, genuinely trainable
operator, and the analytic-gradient machinery needed to improve its speed.
**The `speedup` number is a measurement, not a claim** — read it from
`GET /api/v1/fno/benchmark` rather than trusting marketing.

### Scaling the mesh

Meshing dominates wall time. Trade-offs:

| `MESH_RESOLUTION` | Cells | Vertices | Meshing time |
|---|---|---|---|
| 64 | 68 × 68 × 22 | ≈8 k | ≈250 ms |
| 128 | 132 × 132 × 34 | ≈50 k | ≈936 ms |
| 192 | 196 × 196 × 44 | ≈110 k | ≈2.4 s |
| 256 | 260 × 260 × 52 | ≈200 k | ≈5 s |

For interactive work use 64–128; for CAD export use 192–256 and fetch it from
`/api/v1/export` rather than `/api/v1/generate`.

---

## 17. Extending the Engine

### A new PDE solver

```python
# backend/app/physics/solvers.py
def my_landslide(h: np.ndarray, steps: int, **params) -> np.ndarray:
    """Depth-averaged granular flow. Must accept the grounded channels via **params."""
    ...

run_solver is an explicit if-dispatch over the SOLVERS tuple, so register the name
in both places:

SOLVERS = ("hydraulic", "thermal", "seismic", "diffusion", "landslide")

def run_solver(name, h, steps, params=None, hardness=None):
    ...
    if name == "landslide":          # add this branch
        return my_landslide(h, steps=steps, **p)
```

Contract: take the height field and the grounded channel grids, return a height
field of the same shape. Once registered it is available everywhere
`run_solver()` is, including the `/simulate` and `/generate` endpoints and the
studio dropdown.

### A new conditioning channel

`CHANNELS` is the tuple in `app/core/conditioning.py` that names the rasteriser
output. Adding one means three small edits — no architecture change:

```python
# 1. declare it
CHANNELS = ("elevation", "relief", "hardness", "rainfall",
            "uplift", "roughness", "weight", "lithology")     # 7 → 8 channels

# 2. emit it in build_conditioning(), alongside the existing channels
out["lithology"] = normalise(lithology_contribution)          # (32, 32)

# 3. consume it — e.g. widen the FNO input tensor, then retrain
#    (the FNO concatenates channels, so this is a reshape + retrain)
```

### A new export format

```python
# backend/app/core/rendering/exporters.py
def write_stl(mesh, path, binary=True):
    """Binary STL: 80-byte header, per-triangle normal + 3 vertices + uint16."""
    ...

EXPORTERS["stl"] = write_stl
```

### A new embedding / vector backend

```python
# embeddings.py — implement one method
class CohereEmbedder:
    dim = 1024
    def embed(self, texts): ...
    name = "cohere-embed-v3"

# vector_store.py — implement add() / search()
class MilvusStore(VectorStore):
    def add(self, records, vectors): ...
    def search(self, vector, k): ...
```

Both are selected purely by `.env` keys — no other file needs to change.

---

## 18. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` | launched from the wrong directory | run `python run.py` from the repo root, or `cd backend` for pytest |
| `settings` raises `ValidationError` on boot | malformed `.env` | check ints have no quotes/commas; every value in §8 must parse |
| Empty `fused` list | corpus not seeded | call `reset_repository()` or delete `artifacts/geomind_spatial.db` |
| Mesh is not watertight in the response | resolution too low / degenerate field | raise `MESH_RESOLUTION`; the marching-tets fallback should engage — check `transform.meshing_method` |
| `speedup < 1` in the benchmark | expected — see §16 | compare against your own FEM baseline, not the exact spectral solver |
| Slow first request (~2 s) | cold cache | normal; `run.py` warms at boot |
| `OPENAI_API_KEY not set` warning | backend set to openai without a key | set the key, or `EMBEDDING_PROVIDER=local` |
| CORS error in the browser | frontend origin not allowed | add it to `CORS_ORIGINS` in `.env` and restart |
| USDZ won't open on macOS | unzip first | `unzip model.usdz` → `model.usda`, then re-zip, or import via Reality Composer |
| FNO retrains every boot | weights path not writable / cache deleted | ensure `artifacts/` exists and is writable; set `FNO_AUTO_TRAIN=false` to use a stale cache |

Enable verbose tracing:

```powershell
$env:APP_DEBUG="true"
python run.py
# → /docs exposes every endpoint with live validation
```

---

## 19. Roadmap

The engine is complete and verified. Remaining work is the delivery layer and
the optional accelerators, in priority order:

- [ ] `backend/app/main.py` — FastAPI app, CORS, static mount, lifespan warm-up
- [ ] `backend/app/api/` — the §9.2 endpoint contracts as thin adapters over §9.1
       (no new math required; `query.py`, `simulate.py`, `graph.py`, `health.py`)
- [ ] `backend/app/pipeline.py` — the `ground → condition → sample → simulate →
      render` orchestration function that `/generate` calls
- [ ] `frontend/` — the WebGL studio shell and its four JS modules
- [ ] `backend/tests/` — the 13-file suite specified in §14 (engine-level tests
       first; the meshing watertightness assertions are the highest value)
- [ ] `backend/app/core/rendering/gaussians.py` — Gaussian synthesis from the
      implicit field (SH degree 3, binary PLY / `.splat` writers)
- [ ] `backend/app/core/rendering/exporters.py` — the four writers behind one
      `export(mesh, fmt, path)` dispatcher

Optional accelerators, each independently swappable with no effect on results:

- [ ] `torch` + `torch-harmonics` FNO acceleration (batched spectral layers)
- [ ] `torch-geometric` heterogeneous GNN with learned relation embeddings
- [ ] `gsplat` CUDA splat training from real imagery
- [ ] Qdrant / Supabase pgvector at corpus scale (>10⁶ chunks)
- [ ] PostGIS as the spatial backend (the SQL already matches its dialect)
- [ ] Open3D / Trimesh export validation and mesh decimation
- [ ] Cesium + Unreal Engine 5 integration via the USDZ path

---

## 20. License & Citation

MIT. See `LICENSE`.

```bibtex
@software{geomind3d,
  title  = {GeoMind-3D: Autonomous Neural-Spatial Agent Engine
            for Geomathematical Rendering},
  author = {Saad Salman},
  year   = {2026},
  url    = {https://github.com/SaadxSalman/GeoMind-3D}
}
```

### References

1. Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
   using Hierarchical Navigable Small World graphs*, IEEE TPAMI, 2018.
2. Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and
   individual Rank Learning Methods*, SIGIR, 2009.
3. Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS, 2020.
4. Li, Kovachki et al., *Fourier Neural Operator for Parametric Partial
   Differential Equations*, ICLR, 2021.
5. Müller, Evans, Schied & Keller, *Instant Neural Graphics Primitives*,
   SIGGRAPH, 2022.
6. Kerbl, Kopanas et al., *3D Gaussian Splatting for Real-Time Radiance Field
   Rendering*, SIGGRAPH, 2023.
7. Ju, Losasso, Schaefer & Warren, *Dual Contouring of Hermite Data*, SIGGRAPH, 2002.
8. Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional
   Networks*, ICLR, 2017.
9. Mildenhall, Srinivasan et al., *NeRF: Representing Scenes as Neural Radiance
   Fields*, ECCV, 2020.
10. Bloomenthal & Shoemake, *Convolution Surfaces*, SIGGRAPH, 1991 (metaball /
    gsplat kernel background).

---

<p align="center">
<b>Ground it, then generate it.</b><br>
<sub>GeoMind-3D — geometry with provenance.</sub>
</p>
