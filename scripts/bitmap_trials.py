"""
bitmap_trials.py
----------------
Generate synthetic bitmap color-oddball trials from STUFF dataset images.

For each source image:
  1. Extract a k=5 color palette via k-means clustering
  2. Optionally merge down to 3 or 4 colors (smallest into nearest LAB neighbor)
  3. Generate a 4AFC trial: 1 oddball + 3 distractors at a specified difficulty
  4. Output three versions: original, L*-equalized, chroma-boost + L*-equalized

Modes:
  --batch-csv FILE              Read trial specs from CSV and generate all trials
  --batch-csv FILE --palettes-only   Extract and save palettes only (for review)

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
from skimage.color import rgb2lab, lab2rgb

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MELINA_DIR = os.path.join(PROJECT_DIR, "Melina")
STUFF_DIR = os.path.join(MELINA_DIR, "STUFF_enhanced_dataset_3514_images")

BITMAP_SIZE = 200          # width and height of each bitmap tile (pixels)
TILE_GRID = 10             # 10x10 grid of color cells per bitmap
CELL_SIZE = BITMAP_SIZE // TILE_GRID  # 20 px per cell

RANDOM_SEED = 42
CHROMA_BOOST_FACTOR = 1.5
KMEANS_ATTEMPTS = 10       # number of k-means restarts for stability
SAMPLE_PIXELS = 50000      # max pixels to sample for k-means (speed)
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


# ── TRIAL GENERATION ─────────────────────────────────────────────────────────

def generate_trial(colors_rgb, proportions, difficulty, n_colors, trial_id,
                   category, image_id, output_dir, rng,
                   chroma_boost=CHROMA_BOOST_FACTOR):
    """Generate a complete 4AFC trial with all three versions.

    Args:
        colors_rgb: (k, 3) uint8 palette (already merged if needed)
        proportions: (k,) float proportions
        difficulty: float, proportion shift amount
        n_colors: int, number of colors (3, 4, or 5)
        trial_id: int, trial number
        category: str, source category
        image_id: str, source image ID
        output_dir: str, base output directory
        rng: numpy RandomState
        chroma_boost: float, boost factor for a*b* channels (default 1.5)

    Returns:
        info_dict: dict with trial metadata
    """
    trial_folder = os.path.join(
        output_dir, f"trial_{trial_id:02d}_{category}_{image_id}")
    os.makedirs(trial_folder, exist_ok=True)

    # Base proportions = distractor proportions
    distractor_props = proportions.copy()
    oddball_props = shift_proportions(distractor_props, difficulty, rng)

    # Generate the three color versions
    versions = {
        "original": colors_rgb.copy(),
        "lum_eq": equalize_luminance(colors_rgb),
        "chroma_boost_lum_eq": chroma_boost_and_equalize(colors_rgb, boost=chroma_boost),
    }

    info_lines = [
        f"trial_id: {trial_id}",
        f"source: {category}/{image_id}",
        f"n_colors: {n_colors}",
        f"difficulty_shift: {difficulty:.2f}",
        f"distractor_proportions: {', '.join(f'{p:.4f}' for p in distractor_props)}",
        f"oddball_proportions: {', '.join(f'{p:.4f}' for p in oddball_props)}",
        "",
    ]

    for version_name, version_colors in versions.items():
        # Palette PNG
        draw_palette_png(
            version_colors, distractor_props,
            os.path.join(trial_folder, f"palette_{version_name}.png"),
        )

        # Part 1 = oddball
        bitmap = generate_bitmap(version_colors, oddball_props, rng)
        Image.fromarray(bitmap).save(
            os.path.join(trial_folder, f"part1_oddball_{version_name}.png"))

        # Parts 2-4 = distractors
        for part in range(2, 5):
            bitmap = generate_bitmap(version_colors, distractor_props, rng)
            Image.fromarray(bitmap).save(
                os.path.join(trial_folder,
                             f"part{part}_distractor_{version_name}.png"))

        # Record version colors
        info_lines.append(f"[{version_name}]")
        for i, (c, p) in enumerate(zip(version_colors, distractor_props)):
            info_lines.append(
                f"  color_{i+1}: RGB({c[0]}, {c[1]}, {c[2]}) prop={p:.4f}")
        info_lines.append("")

    # Write trial info
    with open(os.path.join(trial_folder, "trial_info.txt"), "w") as f:
        f.write("\n".join(info_lines))

    return {
        "trial_id": trial_id,
        "category": category,
        "image_id": image_id,
        "n_colors": n_colors,
        "difficulty": difficulty,
        "distractor_proportions": distractor_props.tolist(),
        "oddball_proportions": oddball_props.tolist(),
    }


# ── BATCH PROCESSING ─────────────────────────────────────────────────────────

def process_batch(csv_path, output_dir, palettes_only=False, seed=RANDOM_SEED):
    """Process all trials defined in a CSV manifest.

    Args:
        csv_path: path to trial_manifest.csv
        output_dir: base output directory
        palettes_only: if True, only extract and save palettes
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

    for row in trials:
        trial_id = int(row["trial_id"])
        category = row["category"].strip()
        image_id = row["image_id"].strip()
        difficulty = float(row["difficulty"])
        n_colors = int(row["n_colors"])
        chroma_boost = float(row.get("chroma_boost") or CHROMA_BOOST_FACTOR)
        min_lightness = float(row.get("min_lightness") or 0)

        extras = []
        if chroma_boost != CHROMA_BOOST_FACTOR:
            extras.append(f"chroma={chroma_boost}x")
        if min_lightness > 0:
            extras.append(f"min_L={min_lightness}")
        extra_str = f", {', '.join(extras)}" if extras else ""

        print(f"  Trial {trial_id:02d}: {category}/{image_id} "
              f"(k={n_colors}, shift={difficulty:.2f}{extra_str}) ... ", end="")

        # Load source image
        arr = load_stuff_image(category, image_id)
        if arr is None:
            print("SKIP (image not found)")
            fail_count += 1
            continue

        # Extract k=5 palette (always), optionally filtering dark colors
        colors, proportions = extract_palette(
            arr, k=5, seed=seed, min_lightness=min_lightness)

        if palettes_only:
            # Save palette PNG only
            palette_path = os.path.join(
                palettes_dir,
                f"palette_{category}_{image_id}.png",
            )
            draw_palette_png(colors, proportions, palette_path)

            # Also save a text file with color info
            info_path = os.path.join(
                palettes_dir,
                f"palette_{category}_{image_id}.txt",
            )
            with open(info_path, "w") as f:
                f.write(f"Source: {category}/{image_id}\n")
                f.write(f"Trial {trial_id}: n_colors={n_colors}, "
                        f"difficulty={difficulty:.2f}\n\n")
                for i, (c, p) in enumerate(zip(colors, proportions)):
                    f.write(f"  Color {i+1}: RGB({c[0]:3d}, {c[1]:3d}, "
                            f"{c[2]:3d})  prop={p:.4f}\n")

            print("palette saved")
            success_count += 1
            continue

        # Merge colors if needed
        if n_colors < 5:
            colors, proportions = merge_colors(colors, proportions, n_colors)

        # Generate the trial
        # Use a trial-specific sub-seed for deterministic per-trial randomness
        trial_rng = np.random.RandomState(seed + trial_id)
        generate_trial(
            colors, proportions, difficulty, n_colors,
            trial_id, category, image_id, output_dir, trial_rng,
            chroma_boost=chroma_boost,
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
