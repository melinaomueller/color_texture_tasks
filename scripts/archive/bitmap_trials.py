"""
bitmap_trials.py
----------------
Generate synthetic bitmap color-oddball trials from STUFF dataset images
using hue-family palettes with 2-level difficulty (easy / hard).

For each source image:
  1. Extract all k=5 palette colors via k-means and convert to LCH
  2. For each palette color (selected by palette_index), generate N shades
     by stepping hue symmetrically (L* and C* held constant)
  3. Generate a 4AFC trial at one of two difficulty levels:
       Easy — oddball has a unique cue shade (distant hue) + shifted proportions
       Hard — oddball uses shifted proportions only (same palette)
  4. Output three versions: original, L*-equalized, chroma-boost + L*-equalized

Manifest columns:
  trial_id, category, image_id, palette_index, n_shades, hue_step,
  difficulty_type, distribution_shift, cue_offset

  palette_index: 0–4, which of the k=5 k-means centroids (sorted by
                 proportion descending) to use as the base color
  n_shades:      number of hue-stepped shades (varies per trial, e.g. 5–11)

Modes:
  --batch-csv FILE              Read trial specs from CSV and generate all trials
  --batch-csv FILE --palettes-only   Extract and save hue palettes only (for review)

USAGE:
    # Palette review step
    python3 bitmap_trials.py --batch-csv trial_manifest.csv --palettes-only --output-dir bitmap_trials/

    # Full trial generation
    python3 bitmap_trials.py --batch-csv trial_manifest.csv --output-dir bitmap_trials/
"""

import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2lab, lab2rgb, lab2lch, lch2lab

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Melina/
STUFF_DIR = os.path.join(PROJECT_DIR, "STUFF_enhanced_dataset_3514_images")

BITMAP_SIZE = 400          # width and height of each bitmap tile (pixels)
TILE_GRID = 20             # 20x20 grid of color cells per bitmap
CELL_SIZE = BITMAP_SIZE // TILE_GRID  # 20 px per cell

RANDOM_SEED = 42
CHROMA_BOOST_FACTOR = 1.5
KMEANS_ATTEMPTS = 10       # number of k-means restarts for stability
SAMPLE_PIXELS = 50000      # max pixels to sample for k-means (speed)
MIN_CHROMA = 15            # drop palette colors with C* below this (too gray)
CHROMA_FLOOR = 40          # boost base-color C* to at least this before hue stepping
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    """Set the global random seed for reproducibility."""
    np.random.seed(seed)


def load_stuff_image(category, image_id):
    """Load an image from the STUFF dataset. Returns RGB numpy array or None."""
    path = os.path.join(STUFF_DIR, category, f"{int(image_id):04d}.jpg")
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found")
        return None
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ── PALETTE EXTRACTION ───────────────────────────────────────────────────────

def extract_palette(image_arr, k=5, seed=42, min_lightness=0):
    """Extract k dominant colors and their proportions via k-means.

    Args:
        image_arr: (H, W, 3) uint8 RGB image
        k: number of clusters
        seed: random seed for reproducibility
        min_lightness: minimum L* value (0-100). Colors below this are
            dropped and their proportions redistributed. Use e.g. 10
            to exclude near-black colors.

    Returns:
        colors: (n, 3) uint8 RGB array (n <= k after filtering)
        proportions: (n,) float array summing to 1.0, sorted descending
    """
    pixels = image_arr.reshape(-1, 3).astype(np.float64)

    # Subsample for speed if image is large
    if len(pixels) > SAMPLE_PIXELS:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pixels), SAMPLE_PIXELS, replace=False)
        pixels = pixels[idx]

    # Run k-means with fixed seed
    centroids, labels = kmeans2(pixels, k, minit="++", iter=30, seed=seed)

    # Compute proportions
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    proportions = counts / counts.sum()

    # Sort by proportion (descending)
    order = np.argsort(-proportions)
    centroids = centroids[order]
    proportions = proportions[order]

    colors = np.clip(np.round(centroids), 0, 255).astype(np.uint8)

    # Filter out dark colors if min_lightness is set
    if min_lightness > 0:
        lab = rgb2lab(colors.reshape(1, -1, 3).astype(np.float64) / 255.0)[0]
        keep = lab[:, 0] >= min_lightness
        if keep.sum() >= 2:  # keep at least 2 colors
            colors = colors[keep]
            proportions = proportions[keep]
            proportions /= proportions.sum()

    return colors, proportions


# ── HUE-FAMILY PALETTE GENERATION ────────────────────────────────────────────

def extract_all_base_colors(image_arr, k=5, seed=42, min_chroma=MIN_CHROMA): #k=5
    """Extract dominant colors from an image, filter by chroma, return in LCH.

    Runs k-means (k centroids), converts each RGB centroid → LAB → LCH,
    drops colors with C* < min_chroma (too gray for meaningful hue stepping),
    and returns the rest sorted by proportion (descending).

    Args:
        image_arr: (H, W, 3) uint8 RGB image
        k: number of k-means clusters
        seed: random seed for reproducibility
        min_chroma: minimum C* value; colors below this are dropped

    Returns:
        base_colors_lch: list of (3,) arrays [L*, C*, H*] (may be < k if
                         some centroids were too gray)
    """
    colors, proportions = extract_palette(image_arr, k=k, seed=seed)
    base_colors_lch = []
    for rgb in colors:
        lab = rgb2lab(rgb.reshape(1, 1, 3).astype(np.float64) / 255.0)[0, 0]
        lch = lab2lch(lab.reshape(1, 1, 3))[0, 0]
        if lch[1] >= min_chroma:
            base_colors_lch.append(lch)
    return base_colors_lch


def _max_in_gamut_chroma(L, H_rad, tol=0.5):
    """Binary-search for the maximum C* at given L* and hue that stays in sRGB."""
    lo, hi = 0.0, 150.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        lch = np.array([[[L, mid, H_rad]]])
        lab = lch2lab(lch)
        rgb = lab2rgb(lab)[0, 0]
        if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
            lo = mid
        else:
            hi = mid
    return lo


def generate_hue_palette(base_lch, n_shades, hue_step, chroma_floor=CHROMA_FLOOR):
    """Generate n_shades colors by stepping hue symmetrically around a base.

    L* and C* are held constant; only hue varies.  If the base color's
    chroma is below chroma_floor it is boosted so that small hue steps
    produce visually distinct shades that stay within the same color category.

    Chroma is clamped to the minimum in-gamut maximum across all shades
    so that saturation appears uniform (no clipping at certain hue angles).

    Args:
        base_lch: (3,) array [L*, C*, H*]
        n_shades: int, number of shades to generate (e.g. 5–11)
        hue_step: float, degrees between consecutive shades
        chroma_floor: minimum C* applied to the base before stepping

    Returns:
        colors_rgb: (n_shades, 3) uint8 RGB array
    """
    L, C, H = base_lch
    C = max(C, chroma_floor)
    step_rad = np.deg2rad(hue_step)
    offsets = np.arange(n_shades) - (n_shades - 1) / 2.0  # centered around 0
    hues = (H + offsets * step_rad) % (2 * np.pi)

    # Find the max in-gamut chroma at each hue, then use the minimum
    max_chromas = [_max_in_gamut_chroma(L, h) for h in hues]
    C = min(C, min(max_chromas))

    lch_arr = np.zeros((1, n_shades, 3))
    lch_arr[0, :, 0] = L
    lch_arr[0, :, 1] = C
    lch_arr[0, :, 2] = hues

    lab_arr = lch2lab(lch_arr)
    rgb_float = lab2rgb(lab_arr)[0]  # (n_shades, 3) in [0, 1]
    return np.clip(np.round(rgb_float * 255), 0, 255).astype(np.uint8)


# ── COLOR MERGING ────────────────────────────────────────────────────────────

def _lab_distance(rgb1, rgb2):
    """Euclidean distance in CIELAB between two RGB colors (uint8)."""
    lab1 = rgb2lab(rgb1.reshape(1, 1, 3).astype(np.float64) / 255.0)[0, 0]
    lab2 = rgb2lab(rgb2.reshape(1, 1, 3).astype(np.float64) / 255.0)[0, 0]
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def merge_colors(colors, proportions, target_k):
    """Merge smallest-proportion colors into nearest LAB neighbor until target_k remain.

    Args:
        colors: (n, 3) uint8 RGB array
        proportions: (n,) float array
        target_k: desired number of colors (3 or 4)

    Returns:
        merged_colors: (target_k, 3) uint8 RGB
        merged_proportions: (target_k,) float
    """
    colors = colors.copy().astype(np.float64)
    proportions = proportions.copy()

    while len(colors) > target_k:
        # Find the smallest-proportion color
        smallest_idx = np.argmin(proportions)
        smallest_rgb = colors[smallest_idx]

        # Find nearest neighbor in LAB space
        best_dist = np.inf
        best_idx = None
        for i in range(len(colors)):
            if i == smallest_idx:
                continue
            d = _lab_distance(
                np.round(smallest_rgb).astype(np.uint8),
                np.round(colors[i]).astype(np.uint8),
            )
            if d < best_dist:
                best_dist = d
                best_idx = i

        # Weighted average merge (in RGB space, weighted by proportion)
        total_prop = proportions[smallest_idx] + proportions[best_idx]
        w_small = proportions[smallest_idx] / total_prop
        w_big = proportions[best_idx] / total_prop
        colors[best_idx] = w_big * colors[best_idx] + w_small * smallest_rgb
        proportions[best_idx] = total_prop

        # Remove the merged color
        colors = np.delete(colors, smallest_idx, axis=0)
        proportions = np.delete(proportions, smallest_idx)

    # Re-sort by proportion descending
    order = np.argsort(-proportions)
    colors = colors[order]
    proportions = proportions[order]

    return np.clip(np.round(colors), 0, 255).astype(np.uint8), proportions


# ── COLOR TRANSFORMATIONS ────────────────────────────────────────────────────

def equalize_luminance(colors_rgb):
    """Set all colors to the same mean L* value, preserving a* and b*.

    Args:
        colors_rgb: (n, 3) uint8 RGB

    Returns:
        equalized_rgb: (n, 3) uint8 RGB
    """
    lab = rgb2lab(colors_rgb.reshape(1, -1, 3).astype(np.float64) / 255.0)[0]
    mean_L = np.mean(lab[:, 0])
    lab[:, 0] = mean_L
    rgb_float = lab2rgb(lab.reshape(1, -1, 3))[0]
    return np.clip(np.round(rgb_float * 255), 0, 255).astype(np.uint8)


def chroma_boost_and_equalize(colors_rgb, boost=CHROMA_BOOST_FACTOR):
    """Scale a* and b* by boost factor, then equalize L*.

    Args:
        colors_rgb: (n, 3) uint8 RGB
        boost: multiplier for a* and b* channels

    Returns:
        boosted_rgb: (n, 3) uint8 RGB
    """
    lab = rgb2lab(colors_rgb.reshape(1, -1, 3).astype(np.float64) / 255.0)[0]
    lab[:, 1] *= boost  # a*
    lab[:, 2] *= boost  # b*
    mean_L = np.mean(lab[:, 0])
    lab[:, 0] = mean_L
    rgb_float = lab2rgb(lab.reshape(1, -1, 3))[0]
    return np.clip(np.round(rgb_float * 255), 0, 255).astype(np.uint8)


# ── BITMAP GENERATION ────────────────────────────────────────────────────────

def generate_bitmap(colors_rgb, proportions, rng):
    """Generate a BITMAP_SIZE x BITMAP_SIZE image with colored cells.

    The TILE_GRID x TILE_GRID grid is filled with colors according to
    the given proportions, then spatially shuffled.

    Args:
        colors_rgb: (n, 3) uint8 RGB colors
        proportions: (n,) float proportions summing to 1.0
        rng: numpy RandomState for reproducibility

    Returns:
        bitmap: (BITMAP_SIZE, BITMAP_SIZE, 3) uint8 array
    """
    total_cells = TILE_GRID * TILE_GRID

    # Allocate cells to colors according to proportions
    cell_counts = np.round(proportions * total_cells).astype(int)

    # Fix rounding: adjust the largest-proportion color
    diff = total_cells - cell_counts.sum()
    cell_counts[0] += diff

    # Build list of color indices
    cell_colors = []
    for i, count in enumerate(cell_counts):
        cell_colors.extend([i] * count)

    # Shuffle spatially
    cell_colors = np.array(cell_colors)
    rng.shuffle(cell_colors)

    # Paint the bitmap
    bitmap = np.zeros((BITMAP_SIZE, BITMAP_SIZE, 3), dtype=np.uint8)
    for idx, color_idx in enumerate(cell_colors):
        row = idx // TILE_GRID
        col = idx % TILE_GRID
        y0 = row * CELL_SIZE
        x0 = col * CELL_SIZE
        bitmap[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE] = colors_rgb[color_idx]

    return bitmap


def shift_proportions(base_proportions, shift_amount, rng):
    """Create oddball proportions by shifting mass between colors.

    Moves exactly `shift_amount / 2` probability mass from a donor color
    to a recipient color, producing a total absolute difference of
    `shift_amount` between oddball and distractor distributions.

    The donor is always the largest-proportion color (guaranteed to have
    enough mass). The recipient is chosen randomly from the remaining colors.

    Args:
        base_proportions: (n,) float, the distractor proportions
        shift_amount: float, total proportion mass to shift (0.10 to 0.30)
        rng: numpy RandomState

    Returns:
        oddball_proportions: (n,) float
    """
    n = len(base_proportions)
    props = base_proportions.copy()
    half_shift = shift_amount / 2.0

    # Donor = largest-proportion color (always has enough to give)
    donor = int(np.argmax(props))

    # Recipient = random color that is NOT the donor
    others = [i for i in range(n) if i != donor]
    recipient = others[rng.randint(len(others))]

    # Move exactly half_shift (total absolute change = shift_amount)
    props[donor] -= half_shift
    props[recipient] += half_shift

    # Safety: ensure no proportion goes negative, then renormalize
    props = np.maximum(props, 0.005)
    props /= props.sum()

    return props


# ── PALETTE VISUALIZATION ────────────────────────────────────────────────────

def draw_palette_png(colors_rgb, proportions, path, width=400, height=80):
    """Save a palette visualization: horizontal bars proportional to weight.

    Args:
        colors_rgb: (n, 3) uint8 RGB
        proportions: (n,) float
        path: output PNG path
        width: image width
        height: image height
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for i, (color, prop) in enumerate(zip(colors_rgb, proportions)):
        w = int(round(prop * width))
        if i == len(colors_rgb) - 1:
            w = width - x  # fill remainder
        if w > 0:
            img[:, x:x + w] = color
            x += w
    Image.fromarray(img).save(path)


# ── PROPORTIONS GENERATION ───────────────────────────────────────────────────

def generate_base_proportions(n_shades, rng):
    """Generate base proportions for n shades using a Dirichlet draw.

    Produces a mildly uneven distribution (alpha=3) so shades aren't
    perfectly equal but no single shade dominates.

    Returns:
        proportions: (n_shades,) float array summing to 1.0, sorted descending
    """
    alpha = np.full(n_shades, 3.0)
    props = rng.dirichlet(alpha)
    props.sort()
    return props[::-1]  # descending


# ── TRIAL GENERATION ─────────────────────────────────────────────────────────

def generate_hue_trial(base_lch, n_shades, hue_step, difficulty_type,
                       distribution_shift, cue_offset, trial_id,
                       category, image_id, output_dir, rng,
                       chroma_boost=CHROMA_BOOST_FACTOR):
    """Generate a 4AFC hue-family trial (easy or hard) with three versions.

    Easy: oddball replaces its least-frequent shade with a cue shade
          (hue offset by cue_offset degrees) AND shifts proportions.
    Hard: oddball uses shifted proportions only (same palette as distractors).

    Args:
        base_lch: (3,) array [L*, C*, H*] from extract_base_color
        n_shades: int, number of hue-stepped shades (5–10)
        hue_step: float, degrees between consecutive shades
        difficulty_type: "easy" or "hard"
        distribution_shift: float, proportion mass to shift
        cue_offset: float or None, extra hue degrees for easy-trial cue shade
        trial_id: int
        category: str
        image_id: str
        output_dir: str
        rng: numpy RandomState
        chroma_boost: float

    Returns:
        info_dict: dict with trial metadata
    """
    trial_folder = os.path.join(
        output_dir, f"trial_{trial_id:03d}_{category}_{image_id}_{difficulty_type}")
    os.makedirs(trial_folder, exist_ok=True)

    # Generate the shared shade palette
    base_colors = generate_hue_palette(base_lch, n_shades, hue_step)

    # Generate base proportions
    distractor_props = generate_base_proportions(n_shades, rng)
    oddball_props = shift_proportions(distractor_props, distribution_shift, rng)

    # Build oddball colors
    if difficulty_type == "easy" and cue_offset is not None:
        # Replace the least-frequent shade with a cue shade
        oddball_colors = base_colors.copy()
        least_freq_idx = int(np.argmin(distractor_props))
        # Generate cue shade: base hue + cue_offset
        cue_lch = base_lch.copy()
        cue_lch[2] = (base_lch[2] + np.deg2rad(cue_offset)) % (2 * np.pi)
        cue_rgb = generate_hue_palette(cue_lch, 1, 0.0)  # single color
        oddball_colors[least_freq_idx] = cue_rgb[0]
    else:
        # Hard: same palette, only proportions differ
        oddball_colors = base_colors.copy()

    # Three color versions
    versions = {
        "original": (base_colors.copy(), oddball_colors.copy()),
        "lum_eq": (equalize_luminance(base_colors),
                   equalize_luminance(oddball_colors)),
        "chroma_boost_lum_eq": (
            chroma_boost_and_equalize(base_colors, boost=chroma_boost),
            chroma_boost_and_equalize(oddball_colors, boost=chroma_boost)),
    }

    info_lines = [
        f"trial_id: {trial_id}",
        f"source: {category}/{image_id}",
        f"difficulty_type: {difficulty_type}",
        f"n_shades: {n_shades}",
        f"hue_step: {hue_step:.1f}",
        f"distribution_shift: {distribution_shift:.2f}",
        f"cue_offset: {cue_offset if cue_offset else 'N/A'}",
        f"distractor_proportions: {', '.join(f'{p:.4f}' for p in distractor_props)}",
        f"oddball_proportions: {', '.join(f'{p:.4f}' for p in oddball_props)}",
        "",
    ]

    for version_name, (dist_colors, odd_colors) in versions.items():
        # Palette PNGs
        draw_palette_png(
            dist_colors, distractor_props,
            os.path.join(trial_folder, f"palette_distractor_{version_name}.png"),
        )
        draw_palette_png(
            odd_colors, oddball_props,
            os.path.join(trial_folder, f"palette_oddball_{version_name}.png"),
        )

        # Part 1 = oddball
        bitmap = generate_bitmap(odd_colors, oddball_props, rng)
        Image.fromarray(bitmap).save(
            os.path.join(trial_folder, f"part1_oddball_{version_name}.png"))

        # Parts 2-4 = distractors
        for part in range(2, 5):
            bitmap = generate_bitmap(dist_colors, distractor_props, rng)
            Image.fromarray(bitmap).save(
                os.path.join(trial_folder,
                             f"part{part}_distractor_{version_name}.png"))

        # Record version colors
        info_lines.append(f"[{version_name}]")
        info_lines.append("  Distractor shades:")
        for i, (c, p) in enumerate(zip(dist_colors, distractor_props)):
            info_lines.append(
                f"    shade_{i+1}: RGB({c[0]}, {c[1]}, {c[2]}) prop={p:.4f}")
        if difficulty_type == "easy":
            info_lines.append("  Oddball shades:")
            for i, (c, p) in enumerate(zip(odd_colors, oddball_props)):
                info_lines.append(
                    f"    shade_{i+1}: RGB({c[0]}, {c[1]}, {c[2]}) prop={p:.4f}")
        info_lines.append("")

    # Write trial info
    with open(os.path.join(trial_folder, "trial_info.txt"), "w") as f:
        f.write("\n".join(info_lines))

    return {
        "trial_id": trial_id,
        "category": category,
        "image_id": image_id,
        "difficulty_type": difficulty_type,
        "n_shades": n_shades,
        "hue_step": hue_step,
        "distribution_shift": distribution_shift,
        "cue_offset": cue_offset,
        "distractor_proportions": distractor_props.tolist(),
        "oddball_proportions": oddball_props.tolist(),
    }


# ── BATCH PROCESSING ─────────────────────────────────────────────────────────

def process_batch(csv_path, output_dir, palettes_only=False, seed=RANDOM_SEED):
    """Process all trials defined in a CSV manifest.

    Manifest columns: trial_id, category, image_id, palette_index, n_shades,
                      hue_step, difficulty_type, distribution_shift, cue_offset

    Args:
        csv_path: path to trial_manifest.csv
        output_dir: base output directory
        palettes_only: if True, only extract and save hue palettes
        seed: random seed for reproducibility
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Copy manifest to output dir for reference
    manifest_dest = os.path.join(output_dir, "trial_manifest.csv")
    if os.path.abspath(csv_path) != os.path.abspath(manifest_dest):
        shutil.copy2(csv_path, manifest_dest)

    # Read the manifest
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        trials = list(reader)

    print(f"Loaded {len(trials)} trials from {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    if palettes_only:
        print("Mode: PALETTES ONLY")
    print()

    if palettes_only:
        palettes_dir = os.path.join(output_dir, "palettes_review")
        os.makedirs(palettes_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    # Cache: (category, image_id) → list of LCH base colors
    palette_cache = {}

    for row in trials:
        trial_id = int(row["trial_id"])
        category = row["category"].strip()
        image_id = row["image_id"].strip()
        palette_index = int(row["palette_index"])
        n_shades = int(row["n_shades"])
        hue_step = float(row["hue_step"])
        difficulty_type = row["difficulty_type"].strip()
        distribution_shift = float(row["distribution_shift"])
        cue_offset_str = row.get("cue_offset", "").strip()
        cue_offset = float(cue_offset_str) if cue_offset_str else None

        print(f"  Trial {trial_id:03d}: {category}/{image_id} "
              f"(pal={palette_index}, {difficulty_type}, {n_shades} shades, "
              f"step={hue_step}, shift={distribution_shift:.2f}"
              f"{f', cue={cue_offset}' if cue_offset else ''}) ... ", end="")

        # Load image and extract palette (with caching)
        cache_key = (category, image_id)
        if cache_key not in palette_cache:
            arr = load_stuff_image(category, image_id)
            if arr is None:
                print("SKIP (image not found)")
                fail_count += 1
                continue
            palette_cache[cache_key] = extract_all_base_colors(arr, k=5, seed=seed)

        all_base_colors = palette_cache[cache_key]
        if palette_index >= len(all_base_colors):
            print(f"SKIP (palette_index {palette_index} >= {len(all_base_colors)} colors)")
            fail_count += 1
            continue
        base_lch = all_base_colors[palette_index]

        if palettes_only:
            # Generate and save the hue palette for review
            colors = generate_hue_palette(base_lch, n_shades, hue_step)
            props = np.ones(n_shades) / n_shades  # equal for visualization

            palette_path = os.path.join(
                palettes_dir,
                f"palette_{trial_id:03d}_{category}_{image_id}_p{palette_index}_{difficulty_type}.png",
            )
            draw_palette_png(colors, props, palette_path)

            info_path = os.path.join(
                palettes_dir,
                f"palette_{trial_id:03d}_{category}_{image_id}_p{palette_index}_{difficulty_type}.txt",
            )
            with open(info_path, "w") as f:
                f.write(f"Source: {category}/{image_id}\n")
                f.write(f"Trial {trial_id}: palette_index={palette_index}, "
                        f"{difficulty_type}, "
                        f"{n_shades} shades, hue_step={hue_step}\n")
                f.write(f"Base LCH: L={base_lch[0]:.1f}, "
                        f"C={base_lch[1]:.1f}, H={base_lch[2]:.1f}\n\n")
                for i, c in enumerate(colors):
                    f.write(f"  Shade {i+1}: RGB({c[0]:3d}, {c[1]:3d}, "
                            f"{c[2]:3d})\n")

            print("palette saved")
            success_count += 1
            continue

        # Generate the trial
        trial_rng = np.random.RandomState(seed + trial_id)
        generate_hue_trial(
            base_lch, n_shades, hue_step, difficulty_type,
            distribution_shift, cue_offset, trial_id,
            category, image_id, output_dir, trial_rng,
        )
        print("OK")
        success_count += 1

    print(f"\nDone: {success_count} succeeded, {fail_count} failed")
    if palettes_only:
        print(f"Palettes saved to {os.path.join(output_dir, 'palettes_review')}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bitmap color-oddball trials")
    parser.add_argument(
        "--batch-csv", type=str, required=True,
        help="Path to trial_manifest.csv")
    parser.add_argument(
        "--output-dir", type=str, default="bitmap_trials",
        help="Output directory (default: bitmap_trials/)")
    parser.add_argument(
        "--palettes-only", action="store_true",
        help="Only extract and save palettes (for review)")
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    process_batch(
        csv_path=args.batch_csv,
        output_dir=args.output_dir,
        palettes_only=args.palettes_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
