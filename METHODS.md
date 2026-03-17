# Stimulus Generation Methods

This document describes the full pipeline from raw material images to the final set of bitmap stimuli used in the color oddball and matching experiment (v3).

## 1. Source Materials

### 1.1 Image Database

Stimuli were derived from the **STUFF Enhanced Dataset**, a collection of 3,514 photographs across approximately 200 material categories (e.g., brick, silk, marble, leather). Each image depicts a close-up surface texture of a named material.

### 1.2 Category Selection

From the full dataset, 57 texture categories and 44 color categories were curated based on the materials list (`materials_list.xlsx`). Specific images within each category were selected for adequate color variation, avoiding grayscale or monochromatic photographs (assessed by chroma spread in CIELAB a\*b\* space, with a practical threshold of ~15 units).

### 1.3 Trial Manifest

All trial parameters are defined in `scripts/trial_manifest.csv`. Each row specifies:

| Field | Description |
|---|---|
| `trial_id` | Unique integer (1–86) |
| `category` | STUFF dataset category name |
| `image_id` | Image filename stem within the category folder |
| `difficulty` | Proportion shift for oddball generation (0.10–0.30) |
| `n_colors` | Target palette size (3, 4, or 5) |
| `chroma_boost` | Optional override for chroma scaling factor (default 1.5) |
| `min_lightness` | Optional minimum L\* threshold for dark-color filtering |

The final manifest contains **86 trials**: 60 base material trials + 8 paint + 8 play_dough + 1 high-chroma play_dough (chroma_boost=2.5) + 9 supplementary trials (soap, algae, crepe_paper, foliage, flannel, microfiber). The palette size distribution is 30 five-color, 29 four-color, and 27 three-color trials.

## 2. Palette Extraction

### 2.1 Pixel Sampling

For each source image, all pixels are flattened into an (N, 3) array in RGB space. If the image exceeds 50,000 pixels, a random subsample of exactly 50,000 pixels is drawn (using a trial-consistent RandomState seeded at 42) to keep clustering tractable.

### 2.2 K-Means Clustering

Dominant colors are extracted via k-means clustering with the following parameters:

- **k = 5** clusters (always; merging to fewer colors happens in a later step)
- **Initialization:** kmeans++ (`minit="++"` in `scipy.cluster.vq.kmeans2`)
- **Iterations:** 30
- **Seed:** 42 (deterministic initialization and convergence)

Cluster centroids are rounded to the nearest integer and clipped to the [0, 255] range to yield a valid RGB palette. Proportions are computed from cluster membership counts, then sorted in descending order.

### 2.3 Dark-Color Filtering

For categories with very dark dominant colors that obscure meaningful hue information (e.g., algae, moss), an optional `min_lightness` threshold is applied. Colors with L\* below this threshold in CIELAB space are removed from the palette, and their proportions are redistributed to the remaining colors. A minimum of 2 colors is always retained. In practice, `min_lightness=15` was used for moss (trial 14) and algae (trial 81).

## 3. Color Merging

For trials specifying 3 or 4 palette colors, the initial 5-color palette is hierarchically merged:

1. **Identify** the color with the smallest proportion.
2. **Find** its nearest neighbor by Euclidean distance in CIELAB space.
3. **Merge** via proportion-weighted averaging in RGB space. The merged color inherits the combined proportion.
4. **Repeat** until the target number of colors (3 or 4) is reached.
5. **Re-sort** by proportion (descending).

This preserves perceptual similarity structure while reducing palette complexity.

## 4. Color Processing Versions

Each trial produces bitmaps under three color processing conditions:

### 4.1 Original

The extracted RGB palette is used as-is. Colors retain their native luminance and chroma.

### 4.2 Luminance-Equalized (`lum_eq`)

All palette colors are converted to CIELAB. The L\* channel is set to the mean L\* across all colors in the palette, while a\* and b\* are preserved. The result is converted back to RGB. This removes luminance differences between colors, isolating chromatic discrimination.

### 4.3 Chroma-Boosted + Luminance-Equalized (`chroma_boost_lum_eq`)

All palette colors are converted to CIELAB. The a\* and b\* channels are each multiplied by 1.5 to increase color saturation. L\* is then set to the mean L\* across all colors. The result is converted back to RGB and clipped to valid gamut. This ensures sufficient chromatic signal for materials with low native saturation. One trial (trial 77, play_dough) uses a 2.5x chroma boost as specified in the manifest.

## 5. Bitmap Generation

### 5.1 Grid Layout

Each bitmap is a **200 x 200 pixel** image divided into a **10 x 10 grid** of 20 x 20 pixel cells (100 cells total). Each cell is filled with a single solid color from the palette.

### 5.2 Cell Allocation

Cells are allocated to colors in proportion to the palette proportions. Specifically, each proportion is multiplied by 100 (total cells) and rounded to the nearest integer. Any rounding remainder is absorbed by the largest-proportion color.

### 5.3 Spatial Randomization

The ordered cell list is spatially shuffled using a per-trial pseudorandom number generator (`numpy.random.RandomState`). The seed for each trial is `42 + trial_id`, ensuring reproducible but unique arrangements for each trial while allowing the same palette to produce visually distinct bitmaps.

## 6. Difficulty Manipulation: Proportion Shifting

### 6.1 Method

The oddball bitmap differs from distractor bitmaps only in the relative proportions of palette colors, not in the colors themselves. The shifting procedure is:

1. The **donor** color is always the largest-proportion color in the distractor palette (guaranteed to have sufficient mass to donate).
2. A **recipient** color is randomly chosen from the remaining colors.
3. Exactly `shift_amount / 2` proportion mass is moved from the donor to the recipient.
4. A small floor of 0.005 is applied to prevent any color from reaching zero, followed by renormalization to sum to 1.0.

This produces a total absolute difference of `shift_amount` between oddball and distractor distributions.

### 6.2 Difficulty Levels

Five shift amounts define the difficulty levels:

| Shift | Difficulty |
|---|---|
| 0.10 | Hardest |
| 0.15 | Hard |
| 0.20 | Medium |
| 0.25 | Easy |
| 0.30 | Easiest |

Larger shifts produce more visually distinct oddballs.

### 6.3 Two Dimensions of Difficulty

Trial difficulty is determined by two factors in combination: the **proportion shift** and the **color version**.

**Proportion shift** controls how many cells differ between oddball and distractors (see Section 6.1). **Color version** controls what perceptual information is available to support discrimination:

- **Original:** Retains native luminance differences between palette colors. Participants can rely on lightness contrast as a cue — e.g., noticing that the oddball has "more bright cells" or "less dark area" — without needing to discriminate hue. These trials are intentionally the easiest.
- **Luminance-equalized (`lum_eq`):** All colors share the same L\*, removing brightness as a cue. Discrimination must rely on chromatic (hue/saturation) differences between colors. These trials are harder.
- **Chroma-boosted + L\*-equalized (`chroma_boost_lum_eq`):** Luminance is equalized as above, but a\* and b\* are scaled by 1.5x to amplify chromatic differences. This compensates for low-saturation palettes that would become nearly indistinguishable under luminance equalization alone. These trials are intermediate in difficulty — harder than original (no lightness cue) but easier than plain lum_eq for desaturated materials.

This design is motivated by individual differences: some participants may be highly sensitive to lightness-based proportion changes but poor at hue-based discrimination, or vice versa. By deliberately including easy original-version trials alongside harder luminance-equalized trials, the experiment captures a range of color discrimination strategies rather than measuring only one.

### 6.4 Rationale for Proportion Shifting

Proportion shifting isolates color *ratio* perception as the discrimination cue. Unlike hue or saturation manipulations, this method keeps the exact same set of colors in both oddball and distractor bitmaps, varying only their relative areas. This provides parametric difficulty control without introducing qualitative color differences.

## 7. Trial Structure

### 7.1 Base Trial Composition

Each trial generates **12 bitmaps** arranged as follows:

- **Part 1** (oddball): 1 bitmap with shifted proportions
- **Parts 2–4** (distractors): 3 bitmaps with base proportions

Each of these 4 parts is rendered in all 3 color versions (original, lum_eq, chroma_boost_lum_eq), yielding 4 x 3 = 12 bitmaps per trial.

### 7.2 Output Formats

Bitmaps are saved in two locations:

1. **Per-trial folders** (`bitmap_trials/trial_{id}_{category}_{image_id}/`): organized hierarchically with palette PNGs and a `trial_info.txt` metadata file.
2. **Flat stimulus folder** (`bitmap_trials/jspsych_stimuli/`): all bitmaps with naming convention `t{id}_part{n}_{type}_{version}.png` for direct use by the jsPsych experiment.

## 8. Task Assignment

### 8.1 Two-Task Design

The 86 base trials (each appearing in up to 3 color versions) are split across two tasks:

- **Oddball task** (54 trial-versions): 4AFC — four bitmaps are shown simultaneously; the participant clicks the one that "looks different."
- **Matching task** (54 trial-versions): Study a single bitmap for 2 seconds, view a grayscale noise mask for 500 ms, then 3AFC — select which of three bitmaps matches the studied image's color proportions.

Trial-versions were hand-assigned to tasks based on pilot evaluation of discriminability and match quality. Some base trial IDs appear in both tasks but always in different color versions (e.g., trial 5 appears in the oddball task as chroma_boost_lum_eq and in the matching task as original). A few trial IDs appear more than once within a task in different color versions.

### 8.2 Attention Checks

Each task includes one designated attention check trial:

- **Oddball:** Trial 86 (microfiber, lum_eq, difficulty=0.15)
- **Matching:** Trial 47 (moonstone, lum_eq, difficulty=0.35 — regenerated at a very large shift for easy detection)

## 9. Match Bitmap Generation

### 9.1 Standard Match Variants

For the matching task, three additional "match" bitmaps are generated per trial-version (`match_v1`, `match_v2`, `match_v3`). Each match bitmap uses the **same oddball color proportions** as the original part 1 oddball but a **different spatial arrangement**, achieved through independent RandomState seeds.

The match seed space uses an offset of **5000** from the base seed to avoid collisions with the original trial generation:

```
seed = 5000 + trial_id * 10 + version_offset + match_version
```

where `version_offset` is 0 (original), 100 (lum_eq), or 200 (chroma_boost_lum_eq), and `match_version` is 1, 2, or 3. This produced 162 match bitmaps across 54 matching trial-versions.

During pilot v1, a researcher evaluated all three match candidates alongside the oddball and distractors to select the best match for each trial.

### 9.2 Regenerated Variants

For trials where no standard match variant was satisfactory, regenerated oddball-match pairs were produced at alternative difficulty levels (see Section 10). These use separate seed offsets (8000 for pilot v2 difficulties, 9000 for pilot v3 difficulties) with the formula:

```
shift_seed = offset + trial_id * 100 + int(difficulty * 100)
bitmap_seed = offset + trial_id * 1000 + int(difficulty * 100) * 10 + version_offset [+ 500 for match]
```

## 10. Iterative Pilot Calibration

### 10.1 Pilot v1: Match Selection

All 54 matching trial-versions were evaluated via free-response review. The researcher viewed the oddball, two distractors, and all three match candidates (match_v1, match_v2, match_v3) and judged which match best reproduced the oddball's appearance.

- **30 trials:** Clear best match identified, used directly.
- **22 trials:** Flagged as problematic (matches too similar to distractors, difficulty too high, or poor color rendering).
- **2 trials:** Dropped from the matching task (absorbed into oddball only).

### 10.2 Pilot v2: Difficulty Recalibration

The 22 flagged trials were regenerated at three difficulty levels: **0.15, 0.17, and 0.20**. For each trial-version, new oddball and match bitmaps were created via `generate_regen_bitmaps.py`, producing 6 images per trial (oddball + match at each of 3 difficulties).

Researcher notes for the flagged trials included:
- "makes no sense" (emerald/original)
- "very difficult" (multiple trials at 0.10–0.15 shift)
- "need more similar" / "matches not great" (poor match quality)
- "adjust hue" (flannel/chroma_boost_lum_eq)

Outcomes:
- **15 trials:** Resolved at one of the new difficulty levels.
- **7 trials:** Still problematic even at 0.20 shift.

Final trial assignments used mix-and-match strategies for some resolved trials (e.g., study a d15 oddball, correct answer is a d17 match; or use a distractor as the match and regen images as foils).

### 10.3 Pilot v3: Larger Shifts

The 7 remaining trials were regenerated at **0.25, 0.30, and 0.35** via `generate_regen_v3.py`. The flagged trials and notes were:

| Trial | Category | Version | Note |
|---|---|---|---|
| 38 | wool | original | "too similar, need more red" |
| 47 | moonstone | lum_eq | "change hue, still not right" |
| 48 | coral | original | "need more red or white space" |
| 54 | fluorine | original | "still very difficult" |
| 57 | ruby | original | "still look very similar" |
| 58 | paint | chroma_boost_lum_eq | "need more/less blue" |
| 85 | flannel | chroma_boost_lum_eq | "purple and gray too similar" |

All 7 trials were resolved at the larger shift levels. Trial 47 at 0.35 shift was designated as the matching task attention check.

## 11. Experiment Flow

### 11.1 Oddball Task

1. **Pattern mask** (500 ms): grayscale noise pattern (10 px blocks)
2. **4AFC display** (untimed): 1 oddball + 3 distractors in a 2x2 grid; participant clicks the odd one out
3. **Feedback** (1.2 s): correct/incorrect with colored borders

### 11.2 Matching Task

1. **Study phase** (2 s): single bitmap displayed
2. **Pattern mask** (500 ms): grayscale noise pattern
3. **3AFC display** (untimed): 1 correct match + 2 foils (distractor bitmaps); participant selects the match
4. **Feedback** (1.2 s): correct/incorrect with colored borders

### 11.3 Trial Order

Within each task, trials are presented in a randomized order (jsPsych shuffle). The attention check trial is embedded within the randomized sequence.

## 12. Design Rationale

### 12.1 Why Bitmap Mosaics

Bitmap mosaics provide full experimental control over color distributions. Unlike cropped photographs, mosaics eliminate shape, texture, and spatial frequency cues, isolating color proportion as the sole discrimination dimension. The 10x10 grid offers sufficient spatial resolution to represent proportions ranging from 1% to ~50% while keeping individual cells large enough (20 px) to be individually resolvable.

### 12.2 Why K-Means Followed by Merging

K-means at k=5 captures the dominant color structure of material photographs while remaining robust to minor color variations. Hierarchical merging (smallest into nearest CIELAB neighbor) preserves perceptual relationships: merged colors retain a weighted average that stays close to both originals in color space, rather than introducing artificial colors. This two-stage approach (extract 5, merge to target) is more stable than directly clustering to k=3 or k=4, which can produce inconsistent results with different random initializations.

### 12.3 Why Proportion Shifting

Proportion shifting provides a continuous, parametric difficulty dimension that is orthogonal to the color palette itself. All four bitmaps in a trial contain exactly the same set of colors; only the relative areas differ. This design isolates sensitivity to color *distribution* rather than sensitivity to color *identity*, and allows difficulty to be precisely titrated across trials.

### 12.4 Why Three Color Versions

The three color versions (original, lum_eq, chroma_boost_lum_eq) serve as a second difficulty dimension that targets different perceptual strategies. Original versions deliberately preserve lightness cues, making them easier and providing a performance baseline. Luminance-equalized versions force reliance on chromatic discrimination by setting all palette colors to the same L\*. Chroma-boosted versions rescue low-saturation materials (e.g., sandstone, bone, flint) whose colors cluster near the achromatic axis and would become nearly indistinguishable under luminance equalization alone. Because the experiment is designed to measure individual differences in color perception, including trials that span the full range from lightness-accessible to hue-only discrimination is essential.

## 13. Reproducibility

All stochastic operations use deterministic seeds:

| Operation | Seed |
|---|---|
| Pixel subsampling for k-means | 42 |
| K-means clustering | 42 |
| Per-trial bitmap generation | 42 + trial_id |
| Match bitmap generation | 5000 + trial_id * 10 + version_offset + match_version |
| Regen v2 (proportion shift) | 8000 + trial_id * 100 + int(difficulty * 100) |
| Regen v2 (bitmap layout) | 8000 + trial_id * 1000 + int(difficulty * 100) * 10 + version_offset |
| Regen v2 (match layout) | ... + 500 |
| Regen v3 (proportion shift) | 9000 + trial_id * 100 + int(difficulty * 100) |
| Regen v3 (bitmap layout) | 9000 + trial_id * 1000 + int(difficulty * 100) * 10 + version_offset |
| Regen v3 (match layout) | ... + 500 |

The generation scripts (`bitmap_trials.py`, `generate_match_bitmaps.py`, `generate_regen_bitmaps.py`, `generate_regen_v3.py`) and the trial manifest (`trial_manifest.csv`) together fully specify all stimuli.

## 14. Software Dependencies

- **Python 3** with NumPy, Pillow, SciPy (`scipy.cluster.vq.kmeans2`), scikit-image (`skimage.color.rgb2lab`, `skimage.color.lab2rgb`)
- **jsPsych** for experiment delivery (HTML/JavaScript)
