# Melina — Cropping & Filtering Pipeline

## Purpose

Generate experiment-ready stimuli for a **color oddball task** and a **texture matching task** from the STUFF enhanced dataset (200 material categories, 3514 images). The pipeline produces cropped image patches and mosaic-scrambled versions at controlled difficulty levels.

---

> **TODO:** When the color task is complete, ask user if we should delete `color_outputs_archive.zip` (671 MB archive of all v1/v2 color outputs).

---

## GitHub Repository

**URL:** <https://github.com/melinaomueller/color_texture_tasks>

The public GitHub repo hosts the **v1 output assets** on the `main` branch (the only branch). It contains:

| Folder / File | Contents |
|---|---|
| `cropped_colors/` | v1 color target + oddball crop PNGs |
| `color_mosaics/` | v1 mosaic-scrambled PNGs |
| `cropped_textures/` | v1 texture crop PNGs |
| `color_trial_examples.pdf` | Example trial layout PDF |

The local working directory (this Dropbox folder) is ahead of the GitHub repo and contains all v2/v3 scripts, bitmap trials, texture trials, and related outputs that have not yet been pushed.

---

## Pipeline Overview

The project has evolved through multiple approaches:

### v1 — Random crop + mosaic pipeline
```
materials_list.xlsx
        |
        v
 color_sampler.py
   |           |
   v           v
 TEXTURE     COLOR TARGET
 CROPS       CROPS + MOSAICS
               |
               v
      color_oddball_and_pdf.py
               |
               v
         ODDBALL CROPS
         + MOSAICS + PDF
```

### v2 — Anchor + Structured Rings sampling
```
materials_list.xlsx
        |
        v
 color_sampler_v2.py
        |
        v
  v2_crops/ + v2_mosaics/
        |
        v
  sifted color trials/    (hand-curated subsets)
```

### v3 — Bitmap palette trials (current)
```
 scripts/trial_manifest.csv
        |
        v
 scripts/bitmap_trials.py
        |
        v
  bitmap_trials/           (86 trials, 3 versions each)
        |
        v
 scripts/generate_match_bitmaps.py
        |
        v
  bitmap_trials/*/match_v1–v3   (matching-task stimuli)
```

### Texture oddball pipeline
```
 scripts/texture_manifest.csv
        |
        v
 scripts/generate_texture_crops.py
        |
        v
  texture_trials/          (17 trials, grayscale 4AFC)
```

---

## Scripts

### Root-level scripts (v1/v2)

#### 1. `color_sampler.py` — Main pipeline (v1)

Reads `materials_list.xlsx` (sheets: `texture` and `color`) and the STUFF image dataset.

**Texture processing:**
- 10 random 200x200 crops per texture trial
- Minimum 85% non-black content per crop
- Minimum 80 px distance between crop centers
- Output: `cropped_textures/<category>/<id>/crop_1.png` ...

**Color target processing:**
- 20 target crops per color trial (5 crops x 4 overlap tiers)
- Stricter filtering: 95% content threshold, 50 black threshold
- Crops sampled from center 70% of image only
- Output: `color_trials/<cat>_<id>_target_<tier>_<cropnum>.png`
- Mosaic output: `color_trial_mosaics/mosaic_<cat>_<id>_target_<tier>_<cropnum>.png`

**Shared utilities:**
- `compute_color_profile()` — 3D CIELAB histogram (8 bins/channel = 512 bins)
- `color_distance()` — chi-squared distance between histogram profiles
- `mean_color_mosaic()` — tile-scramble that preserves color palette but destroys spatial structure
- `_rgb_to_lab()` — RGB to CIELAB conversion (uses skimage if available, otherwise manual)

**Usage:**
```bash
python color_sampler.py
```

---

#### 2. `color_sampler_v2.py` — Anchor + Structured Rings sampling (v2)

Replaces v1's random-crop approach with deterministic, spatially structured sampling. Crops are placed in concentric rings around fixed anchor points, so crops near the same anchor naturally share overlapping color distributions.

**Sampling method:**
1. **5 anchors** placed in a quincunx pattern (center + 4 quadrant midpoints)
2. **4 concentric rings** per anchor at increasing radii (fractions: 0.10, 0.25, 0.45, 0.70 of max usable radius)
3. **5 crops per ring** at evenly spaced angles (72° apart), with a random angular offset per anchor
4. **Total: 100 crops per image** (5 anchors x 4 rings x 5 crops)
5. Outer radius clamped to 150 px max to guarantee spatial overlap at every ring level
6. Each crop also gets a mosaic-scrambled version (same mosaic logic as v1)

**File naming:**
```
<category>_<imgid>_a<anchor>_r<ring>_c<crop>.png
mosaic_<category>_<imgid>_a<anchor>_r<ring>_c<crop>.png
```

**Output:**
- `v2_crops/` — crop PNGs
- `v2_mosaics/` — mosaic-scrambled PNGs

**Applies to:** both texture and color tasks.

**Usage:**
```bash
python color_sampler_v2.py
```

---

#### 3. `color_oddball_and_pdf.py` — Oddball generation + PDF

For each target trial, selects **same-category distractor images** at 4 difficulty levels based on LAB color similarity.

**Oddball selection logic:**
1. Profile all unused images from the same material category
2. Compute chi-squared color distance to the target
3. Sort by distance, split into quartiles
4. Pick one image per quartile:
   - **hard** — most similar color (smallest distance)
   - **medhard** — moderately similar
   - **medeasy** — moderately different
   - **easy** — most different color (largest distance)

**Output per oddball:** 20 crops (5 crops x 4 overlap tiers) + corresponding mosaics.

**PDF generation:** Creates `color_trial_examples.pdf` showing example trial layouts (study image + 2AFC choices) for each difficulty tier, in both crop and mosaic formats.

**Usage:**
```bash
python color_oddball_and_pdf.py
```

---

#### 4. `mosaic_scramble.py` — Standalone mosaic tool

Prototype/standalone version of the mosaic scrambling logic. Takes a single input image, samples 5 crops, saves originals + scrambled versions.

**Usage:**
```bash
# Edit INPUT_IMAGE and OUTPUT_DIR in the CONFIG section, then:
python mosaic_scramble.py
```

---

#### 5. `test_2trials.py` — Quick test

Runs the full pipeline (targets + oddballs) on only 2 categories to verify correctness without processing the entire dataset. Clears previous output before running.

**Usage:**
```bash
python test_2trials.py
```

---

### `scripts/` — Bitmap & texture trial generators (v3)

#### `bitmap_trials.py` — Bitmap color-oddball generator

Generates synthetic bitmap stimuli for the color oddball task. Extracts k=5 color palettes from STUFF images, merges to 3–4 colors, and creates 4AFC trials (1 oddball + 3 distractors) at a specified difficulty. Produces three versions of each trial image: original, L*-equalized, and chroma-boost + L*-equalized.

**Input:** `trial_manifest.csv` (86 rows: trial_id, source_image, category, image_id, difficulty, n_colors, chroma_boost, min_lightness)

**Output:** `bitmap_trials/` — one subfolder per trial containing palette PNGs, oddball + distractor PNGs in 3 versions, and `trial_info.txt`.

---

#### `generate_match_bitmaps.py` — Match-task bitmap generator

Creates matching-task stimuli: 3 new bitmaps per trial that share the same oddball color proportions but use different random spatial arrangements (correct match = same palette ratios, different layout).

---

#### `generate_regen_bitmaps.py` / `generate_regen_v3.py` — Difficulty regeneration

Regenerates oddball + match bitmaps at alternative difficulty levels for flagged trials. `generate_regen_bitmaps.py` uses shifts of 0.15/0.17/0.20; `generate_regen_v3.py` uses 0.25/0.30/0.35 for 7 problematic trials.

---

#### `generate_texture_crops.py` — Texture oddball crop generator

Generates 200x200 grayscale crops for the texture oddball task. Each 4AFC trial uses 3 distractor crops from one image ("same3") and 1 oddball crop from a different image ("diff1") of the same category.

**Input:** `texture_manifest.csv` (17 rows: trial_id, category, same3_image, diff1_image, crop parameters)

**Output:** `texture_trials/` — one subfolder per trial with distractor PNGs, oddball PNGs, source images, and `trial_info.txt`.

---

#### `generate_texture_review.py` — Texture review page

Generates `review_index.html` with thumbnail grids for reviewing texture oddball trials (4 layout variants per trial with oddball in different positions).

---

#### `generate_click_crop.py` — Interactive crop placement

Creates `click_crop.html`, an interactive browser tool for manually placing crop rectangles on source images. Grayscale conversion and downscaling happen client-side via Canvas API.

---

#### `generate_crop_gallery.py` — Candidate crop gallery

Generates a large gallery of candidate texture crops (12 non-overlapping 200x200 grayscale crops) and an interactive `index.html` for selecting the best 3 distractors + 1 oddball per trial.

---

## Key Parameters

| Parameter | Texture | Color | Description |
|---|---|---|---|
| `CROP_SIZE` | 200 px | 200 px | Width/height of each crop |
| `TILE_SIZE` | — | 20 px | Mosaic tile size |
| `MIN_CONTENT` | 0.85 | 0.95 | Min fraction of non-black pixels |
| `BLACK_THRESH` | 15 | 50 | Max RGB value considered "black" |
| `CENTER_PCT` | 1.0 | 0.7 | Fraction of image to sample from (centered) |
| `RANDOM_SEED` | 42 | 42 (targets), 99 (oddballs) | Reproducibility seeds |

---

## Overlap Tiers

Controls spatial overlap between crops sampled from the same image:

| Tier | Min distance between crops | Effect |
|---|---|---|
| `extreme_overlap` | 0 px | Crops can fully overlap |
| `medium_overlap` | ~67 px (CROP_SIZE / 3) | Moderate spatial separation |
| `small_overlap` | ~133 px (2 * CROP_SIZE / 3) | Mostly non-overlapping |
| `nonoverlap` | 200 px (CROP_SIZE) | Fully non-overlapping |

---

## Folder Contents

```
Melina/
  # --- Root-level scripts ---
  color_sampler.py              # Main pipeline script (v1)
  color_sampler_v2.py           # Anchor + Structured Rings sampling (v2)
  color_oddball_and_pdf.py      # Oddball generation + PDF
  mosaic_scramble.py            # Standalone mosaic tool (prototype)
  test_2trials.py               # Quick 2-category test
  materials_list.xlsx           # Trial definitions (texture + color sheets)

  # --- v3 scripts ---
  scripts/
    bitmap_trials.py            # Bitmap color-oddball generator
    generate_match_bitmaps.py   # Match-task bitmap generator
    generate_regen_bitmaps.py   # Difficulty regeneration (0.15–0.20)
    generate_regen_v3.py        # Difficulty regeneration (0.25–0.35)
    generate_texture_crops.py   # Texture oddball crop generator
    generate_texture_review.py  # Texture review page generator
    generate_click_crop.py      # Interactive crop placement tool
    generate_crop_gallery.py    # Candidate crop gallery
    trial_manifest.csv          # Color trial definitions (86 trials)
    texture_manifest.csv        # Texture trial definitions (17 trials)

  # --- Source data ---
  STUFF_enhanced_dataset_3514_images/   # Source images (200 categories)

  # --- Archive (v1 + v2 color outputs) ---
  color_outputs_archive.zip     # Archived color outputs from v1/v2/v3 iterations
                                # Contains: color_trials/, color_trial_mosaics/,
                                # v2_crops/, v2_mosaics/, new_crops/, new_mosaics/,
                                # sifted color trials/, trial_versions/,
                                # new_trial_versions/
                                # TODO: When color task is complete, ask user
                                #       whether to delete this archive.

  # --- v3 output (bitmap trials) ---
  bitmap_trials/                # 86 color oddball trials (bitmap stimuli)
    trial_manifest.csv          # Trial manifest
    palettes_review/            # Palette PNGs + TXT for all trials
    jspsych_stimuli/            # jsPsych-ready stimulus versions
    trial_01_brick_0291/        # Per-trial folders with 3 versions each
    ...

  # --- Texture output ---
  texture_trials/               # 17 texture oddball trials (grayscale 4AFC)
    trial_01_rhinestone/        # Per-trial: distractors, oddballs, sources
    ...

  # --- Documentation / review ---
  colortrials_v2.pptx           # v2 color trial visuals
  v2 color_trials.rtf           # v2 trial composition notes (oddball + distractor assignments)
  inkscape_/                    # Stimulus layout files
    colors.svg / colors.pptx
    textures.svg / textures.pptx
    filtered stimuli/           # Grayscale texture stimuli
    unfiltered stimuli/         # Color texture stimuli
    make_texture_pdf.py         # Generates texture_sets_overview.pdf
    texture_pairs_overview.pdf
    texture_sets_overview.pdf
```

---

## Dependencies

- Python 3
- `numpy`, `Pillow`, `openpyxl`
- `scikit-image` (optional — used for RGB-to-LAB if available; manual fallback included)
