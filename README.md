# Color & Texture Oddball Tasks

Stimulus generation pipeline and experiment files for a color oddball and texture oddball study using the STUFF Enhanced Dataset (3,514 material images). Participants detect an oddball bitmap among distractors in a color task, then match it to a comparison in a follow-up matching task. A parallel texture task uses grayscale scene crops. The project aims to measure individual differences in color and texture perception for objects/materials.

## Repository Structure

```
color_texture_tasks/
├── scripts/                  # All generation scripts, manifests, and source data
│   ├── bitmap_trials.py              # Core bitmap pipeline (palette extraction → bitmap generation)
│   ├── generate_match_bitmaps.py     # Matching-task bitmap variants
│   ├── generate_regen_bitmaps.py     # Difficulty recalibration (pilot v2)
│   ├── generate_regen_v3.py          # Larger difficulty shifts (pilot v3)
│   ├── generate_texture_crops.py     # Texture oddball crop generation
│   ├── generate_texture_review.py    # Texture review HTML pages
│   ├── generate_crop_gallery.py      # Interactive crop picker with quality filters
│   ├── color_sampler.py              # Legacy v1 color crop pipeline
│   ├── color_sampler_v2.py           # Legacy v1 color mosaic pipeline
│   ├── trial_manifest.csv            # 86 color trial definitions
│   ├── texture_manifest.csv          # 17 texture trial definitions
│   └── materials_list.xlsx           # Category/image selection reference
├── bitmap_trials/            # Generated color bitmap stimuli (per-trial dirs + jspsych_stimuli/)
├── texture_trials/           # Generated texture oddball stimuli (per-trial dirs + jspsych_stimuli/)
├── experiment/               # jsPsych HTML experiment files
├── trial_versions/           # Early trial version outputs
├── new_trial_versions/       # Revised trial version outputs
├── cropped_colors/           # Legacy v1 color crops (by category)
├── cropped_textures/         # Legacy v1 texture crops (by category)
├── color_mosaics/            # Legacy v1 color mosaics (by category)
├── new_crops/                # Additional cropped stimuli
├── new_mosaics/              # Additional mosaic stimuli
├── METHODS.md                # Detailed methods documentation for the bitmap pipeline
├── v2_color_trials.rtf       # Pilot v2 review notes
└── color_trial_examples.pdf  # Example trial visuals
```

## Pipeline 1: Bitmap Color Task

The primary stimulus pipeline. See [METHODS.md](METHODS.md) for full technical documentation.

### Scripts

| Script | Purpose |
|---|---|
| `bitmap_trials.py` | Core pipeline: extracts k-means palettes from STUFF images, generates distractor/oddball bitmaps with shifted color proportions. Supports luminance equalization and chroma boosting. |
| `generate_match_bitmaps.py` | For each of the 54 matching trials, generates 3 match bitmaps that share the oddball's color proportions but with different spatial arrangements. |
| `generate_regen_bitmaps.py` | Regenerates oddball + match bitmaps for 21 flagged trials at 3 difficulty levels (0.15, 0.17, 0.20) for pilot recalibration. |
| `generate_regen_v3.py` | Regenerates bitmaps for 7 remaining problematic trials at larger difficulty shifts (0.25, 0.30, 0.35). |

### Data

- **`trial_manifest.csv`** — Defines all 86 trials: category, image ID, difficulty (proportion shift), palette size (3–5 colors), optional chroma boost and min-lightness overrides.
- **Output** — `bitmap_trials/` contains per-trial directories with palette visualizations and a `jspsych_stimuli/` subdirectory with the final PNGs used in the experiment.

## Pipeline 2: Texture Oddball Task

Generates grayscale texture crops for an oddball detection task.

### Scripts

| Script | Purpose |
|---|---|
| `generate_texture_crops.py` | Core script: for each of 17 trials, extracts 3 distractor crops and 1 oddball crop from different STUFF category images. All crops are converted to grayscale to isolate texture from color. Each trial produces 4 oddball variants at different spatial positions. |
| `generate_texture_review.py` | Generates an HTML review page per trial showing all 4 oddball variants side-by-side for researcher evaluation. |
| `generate_crop_gallery.py` | Interactive HTML gallery for selecting crops. Applies sharpening filters and rejects crops with excessive blank regions (>15% near-white pixels). |

### Data

- **`texture_manifest.csv`** — Defines 17 texture trials: distractor category/images and oddball category/image.
- **Output** — `texture_trials/` contains per-trial directories and a `jspsych_stimuli/` subdirectory with the final stimuli.

## Pipeline 3: Legacy V1 Color Crops & Mosaics

Earlier exploration pipeline before the bitmap approach was adopted.

| Script | Purpose |
|---|---|
| `color_sampler.py` | Extracts color palette crops from STUFF images and saves per-category crop grids. |
| `color_sampler_v2.py` | Generates color mosaic images from extracted palettes. |

**Output** — `cropped_colors/`, `cropped_textures/`, `color_mosaics/`, `new_crops/`, `new_mosaics/`

## Experiment

The `experiment/` directory contains jsPsych 7.3.4 HTML files for running the tasks in a browser:

| File | Task |
|---|---|
| `index_oddball.html` | Color oddball detection (main task) |
| `index_matching.html` | Color oddball-to-match comparison |
| `matching_pilot.html` | Matching pilot v1 |
| `matching_pilot_v2.html` | Matching pilot v2 (recalibrated difficulties) |
| `matching_pilot_v3.html` | Matching pilot v3 (larger shifts) |
| `comparison.html` / `comparison_v2.html` | Side-by-side comparison viewers |
| `index.html` / `index_v2.html` / `index_v3.html` | Experiment entry points across versions |

## Dependencies

**Python 3** with:
- numpy
- Pillow (PIL)
- scipy
- scikit-image
- openpyxl

**JavaScript:**
- jsPsych 7.3.4 (loaded via CDN in experiment HTML files)

## Reproducibility

All random operations use deterministic seeds (base seed = 42) with per-trial offsets, so re-running any script produces identical outputs. See [METHODS.md](METHODS.md) Section 13 for full details on the seeding strategy.

**Note:** The generation scripts reference the STUFF Enhanced Dataset directory, which is not included in this repository due to size. To regenerate stimuli, place the dataset at the path expected by `bitmap_trials.py` (see `STUFF_DIR` variable) or update the path accordingly.
