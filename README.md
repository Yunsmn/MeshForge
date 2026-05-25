# MeshForge — Automated 2D-to-3D Floor Plan Reconstruction

<img width="1600" height="539" alt="WhatsApp Image 2026-05-21 at 5 53 37 PM" src="https://github.com/user-attachments/assets/b2bae4c9-ca37-43c0-b716-811e7fc02816" />

MeshForge is an end-to-end pipeline that converts 2D CAD floor plans (DWG/DXF) into unified 3D mesh models. The pipeline is split into four sequential phases, each implemented as a standalone Jupyter notebook designed to run on Google Colab with GPU support.

## Pipeline Overview

```
DWG/DXF Floor Plan
       |
       v
 Phase 1: Segmentation (SAM)
       |
       v
 Phase 2: ControlNet Rendering
       |
       v
 Phase 3: 3D Mesh Generation (Hunyuan3D)
       |
       v
 Phase 4: Mesh Stitching
       |
       v
  Unified 3D Mesh (.glb)
```

## Phase 1 — DXF Density-Based SAM Segmentation

**Notebook:** `phase 1.ipynb`

Reads a DXF floor plan and segments it into meaningful regions using Meta's Segment Anything Model (SAM). The process works as follows:

1. Parses the DXF file and renders it to a high-resolution overview image via custom SVG-to-PNG conversion.
2. Builds a multi-scale density map using Gaussian blurring to locate areas with concentrated geometry.
3. Applies watershed segmentation to identify distinct region candidates.
4. Classifies regions by size: **normal** regions are fed to SAM with box prompts, **large** regions are kept as-is for downstream handling, and **noise** (tiny fragments) is discarded.
5. For each valid region, renders a high-resolution tile from the original DXF data.
6. Computes spatial adjacency between segments for use in Phase 4.

**Key outputs:** `segments_meta.json` (segment IDs, DXF bounding boxes, adjacency graph), tile PNGs in `density_tiles/`.

**Main dependencies:** `ezdxf`, `cairosvg`, `segment-anything`, `torch`, `opencv-python`, `scipy`

---

## Phase 2 — ControlNet CAD-to-Render

**Notebook:** `phase 2.ipynb`

Transforms 2D CAD wireframe tile images into realistic 3D-looking rendered images using Stable Diffusion 1.5 with ControlNet (lineart conditioning). Each tile from Phase 1 is processed to produce a clean SolidWorks-style 3D render that Hunyuan3D can interpret in Phase 3.

1. Extracts a line-art control map from the input CAD tile using the LineartDetector.
2. Runs the ControlNet-conditioned Stable Diffusion pipeline with a prompt engineered for clean, studio-lit CAD renders on a white background.
3. Generates multiple render variants per tile and saves results to Google Drive for persistence.

**Key outputs:** Rendered PNG images in `/content/outputs/` and backed up to Google Drive.

**Main dependencies:** `diffusers`, `transformers`, `controlnet-aux`, `accelerate`, `torch`

**Requires:** GPU runtime (T4 or better). A runtime restart is needed after installing packages.

---

## Phase 3 — Hunyuan3D Mesh Generation

**Notebook:** `phase 3.ipynb`

Converts the rendered 2D images from Phase 2 into 3D mesh geometry using Tencent's Hunyuan3D-2 Mini model. Each rendered tile image is turned into a `.glb` mesh file.

1. Removes the image background using `rembg` (or a simple white-threshold fallback).
2. Loads the Hunyuan3D-2 Mini pipeline with weights cached to Google Drive (~25 GB on first run).
3. Runs single-image-to-3D inference with 30 diffusion steps and octree resolution of 380.
4. Exports the resulting trimesh geometry as a GLB file.

**Key outputs:** `.glb` mesh files in `/content/mesh_results/`.

**Main dependencies:** `hunyuan3d (hy3dgen)`, `trimesh`, `rembg`, `torch`, `safetensors`, `pymeshlab`

**Requires:** GPU runtime. Model weights are downloaded once and cached on Google Drive.

---

## Phase 4 — Mesh Stitching

**Notebook:** `phase 4.ipynb`

Assembles all individual segment meshes from Phase 3 into a single unified 3D model using the spatial metadata from Phase 1.

1. Loads all `.obj`/`.glb` mesh files and matches them to segments via `segments_meta.json`.
2. Transforms each mesh from its local normalized coordinate space into global DXF coordinates using the bounding boxes from Phase 1.
3. Resolves overlaps between neighboring segments by clipping meshes at the midpoint of overlap zones.
4. Detects boundary vertices near segment edges and aligns them between neighboring meshes using a KD-tree nearest-neighbor search.
5. Concatenates all meshes, welds duplicate vertices, and performs Laplacian boundary smoothing.
6. Cleans up degenerate/duplicate faces, fixes normals, and exports the final unified mesh.

**Key outputs:** `unified_mesh.glb`, `unified_mesh_preview.png` (multi-view matplotlib render).

**Main dependencies:** `trimesh`, `scipy`, `numpy`, `networkx`

---

## Testing Sample

The `testing sample/` folder contains a sample CAD file for testing the pipeline:

| File | Description |
|------|-------------|
| `HY2M - Leap.dwg` | Original AutoCAD drawing file (native DWG format) |
| `HY2M - Leap.dxf` | DXF version of the same file (used as input to Phase 1) |

To test the pipeline, use `HY2M - Leap.dxf` as the input for Phase 1.

---

## Step-by-Step Testing Guide (using `HY2M - Leap.dxf`)

This walkthrough uses the sample file in `testing sample/` to run the full pipeline from DXF to unified 3D mesh.

### Prerequisites

- A Google account with ~30 GB of free Drive space (for model weight caching).
- Upload all four notebooks (`phase 1.ipynb` through `phase 4.ipynb`) to Google Colab.
- Upload `testing sample/HY2M - Leap.dxf` to Colab (you will be prompted, or can upload via the Files panel).

---

### Phase 1 — Segment the floor plan

**Runtime:** CPU is enough, but GPU speeds up SAM inference.

1. Open `phase 1.ipynb` in Colab.
2. Run **Cell 1** — installs `ezdxf`, `cairosvg`, `segment-anything`, etc.
3. Run **Cell 2** — downloads the SAM `vit_h` checkpoint (~2.5 GB). Skips if already present.
4. Run **Cell 3** — imports libraries.
5. Run **Cell 4** — you will be prompted to **upload your `.dxf` file**. Upload `HY2M - Leap.dxf` from `testing sample/`. The cell also sets all pipeline parameters:
   - `OVERVIEW_PX = 2048` — resolution of the overview render.
   - `TILE_LONG_EDGE_PX = 8192` — resolution of each segment tile.
   - `DENSITY_THRESHOLD = 0.06` — sensitivity for region detection.
   - No changes needed for the test file; defaults work out of the box.
6. Run **Cells 5–12** sequentially (Run All is fine). The pipeline will:
   - Render the full DXF as a 2048 px overview image.
   - Build a density map and detect region candidates.
   - Run SAM on each normal-sized region to refine masks.
   - Save high-res tiles and compute segment adjacency.
7. Run **Cell 13** — downloads `density_sam_results.zip` containing:

| Output file | What it is |
|---|---|
| `segments_meta.json` | Segment IDs, DXF bounding boxes, neighbor lists — **needed by Phase 4** |
| `density_tiles/tile_0000.png`, `tile_0001.png`, ... | High-res tile PNGs of each segment — **feed these into Phase 2** |
| `density_overlay.png` | Color overlay showing all detected segments |
| `density_comparison.png` | Side-by-side of overview vs. segmented overlay |

**Save `segments_meta.json` and the `density_tiles/` PNGs** — you will need both later.

---

### Phase 2 — Render tiles into 3D-looking images

**Runtime:** GPU required (T4 or better). Change via `Runtime → Change runtime type → T4 GPU`.

You must run Phase 2 **once per tile** from Phase 1. Pick a tile to start (e.g., `tile_0000.png`).

1. Open `phase 2.ipynb` in Colab.
2. Run **Cell 1** — installs `diffusers`, `transformers`, `controlnet-aux`, etc. **After this cell finishes, restart the runtime** (`Runtime → Restart session`).
3. Run **Cell 2** — mounts Google Drive and sets up the model cache directory at `MyDrive/colab_model_cache/`. First run downloads ~5 GB of model weights to Drive; subsequent runs load from cache.
4. Run **Cell 3** — **before running**, upload your tile PNG to Colab and update the path:
   ```python
   INPUT_IMAGE_PATH = "/content/tile_0000.png"  # <-- change to your tile filename
   ```
   The other parameters can stay at their defaults:
   - `NUM_IMAGES = 4` — generates 4 render variants to choose from.
   - `TARGET_SIZE = 768` — resize resolution for Stable Diffusion.
   - `GUIDANCE_SCALE = 12.0` and `CONTROLNET_CONDITIONING_SCALE = 1.3` — tuned for CAD input.
5. Run **Cells 4–7** — loads models and generates rendered images (~2–3 minutes on T4).
6. Run **Cell 8** — displays all renders and saves them:
   - `/content/outputs/render_01.png` through `render_04.png`
   - Also backed up to `MyDrive/colab_model_cache/renders/`.

**Pick the best render** (cleanest 3D appearance, no artifacts) for each tile. **Save these render PNGs** — they are the input for Phase 3.

Repeat for each tile from Phase 1.

---

### Phase 3 — Generate 3D meshes

**Runtime:** GPU required (T4 or better).

You must run Phase 3 **once per rendered image** from Phase 2.

1. Open `phase 3.ipynb` in Colab.
2. Run **Cell 1** — mounts Google Drive. Cache directory: `MyDrive/3D_Models_Cache/`.
3. Run **Cell 2** — installs `trimesh`, `rembg`, `pymeshlab`, etc.
4. Run **Cell 3** — **before running**, upload your best render PNG to `/content/input/` or just upload it when prompted. The cell auto-detects images in `/content/input/`.
5. Run **Cell 4** — downloads Hunyuan3D-2 Mini weights (~25 GB) to Google Drive on first run. Cached for future runs.
6. Run **Cell 5** — runs the image-to-3D pipeline:
   - Removes the background with `rembg`.
   - Generates a 3D mesh using 30 inference steps at octree resolution 380.
   - Exports the mesh as `hunyuan3d_output.glb` in `/content/mesh_results/`.
7. Run **Cell 6** — zips and downloads the result.

**Save the `.glb` file.** Name it to match the segment (e.g., rename `hunyuan3d_output.glb` to `seg_0000.glb`) so Phase 4 can match it to `segments_meta.json`.

Repeat for each rendered tile.

---

### Phase 4 — Stitch all meshes into one model

**Runtime:** CPU is enough.

1. Open `phase 4.ipynb` in Colab.
2. Run **Cell 1** — installs `trimesh`, `scipy`, `networkx`.
3. Run **Cell 2** — sets configuration. The defaults are fine:
   ```python
   MESHES_DIR = '/content/meshes'
   SEGMENTS_META_PATH = '/content/segments_meta.json'
   UPLOAD_FILES = True
   OUTPUT_FILENAME = 'unified_mesh.glb'
   ```
4. Run **Cell 3** — you will be prompted to upload files in two steps:
   - **First prompt:** upload **all your `.glb` mesh files** from Phase 3 (e.g., `seg_0000.glb`, `seg_0001.glb`, ...).
   - **Second prompt:** upload **`segments_meta.json`** from Phase 1.
5. Run **Cells 4–14** sequentially. The pipeline will:
   - Match each mesh to its segment ID using the metadata.
   - Transform meshes from local space to global DXF coordinates.
   - Clip overlapping meshes at their shared boundaries.
   - Align and weld boundary vertices between neighbors.
   - Smooth boundary seams and clean up the final mesh.
6. Run **Cell 15** — renders a multi-view preview (top, front, perspective).
7. Run **Cell 16** — downloads the final outputs:

| Output file | What it is |
|---|---|
| `unified_mesh.glb` | The final stitched 3D model — open in any GLB viewer (e.g., [glTF Viewer](https://gltf-viewer.donmccurdy.com/)) |
| `unified_mesh_preview.png` | Multi-angle matplotlib preview |

---

### Quick Reference: What to Feed Each Notebook

| Phase | Input file(s) | Where to set / upload | Output file(s) |
|-------|---------------|----------------------|-----------------|
| 1 | `HY2M - Leap.dxf` | Upload when prompted (Cell 4) | `segments_meta.json` + `density_tiles/*.png` |
| 2 | One tile PNG (e.g., `tile_0000.png`) | Set `INPUT_IMAGE_PATH` in Cell 3 | `render_01.png` ... `render_04.png` |
| 3 | One render PNG (best from Phase 2) | Upload to `/content/input/` or when prompted (Cell 3) | `hunyuan3d_output.glb` |
| 4 | All `.glb` meshes + `segments_meta.json` | Upload when prompted (Cell 3) | `unified_mesh.glb` |

---

## Requirements

- Python 3.8+
- Google Colab with GPU runtime (T4 or better recommended)
- Google Drive (for model weight caching in Phases 2 and 3)
- ~30 GB Drive space for model weights on first run
