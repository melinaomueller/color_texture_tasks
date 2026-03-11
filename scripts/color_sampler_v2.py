"""
color_sampler_v2.py
-------------------
Anchor + Structured Rings sampling.

Places 5 anchors in a quincunx pattern (center + 4 quadrant midpoints),
then samples 5 crops at each of 4 concentric rings around each anchor.
Total: 100 crops per image (5 anchors x 4 rings x 5 crops).

Output folders:
    v2_crops/     — crop PNGs
    v2_mosaics/   — mosaic-scrambled PNGs

File naming:
    <category>_<imgid>_a<anchor>_r<ring>_c<crop>.png
    mosaic_<category>_<imgid>_a<anchor>_r<ring>_c<crop>.png

USAGE:
    python color_sampler_v2.py                    # all categories
    python color_sampler_v2.py --max-categories 2  # first 2 categories only
"""

import argparse
import csv
import os
import math
import re
from itertools import combinations

import numpy as np
from PIL import Image

from color_sampler import (
    read_trials,
    load_image,
    resolve_folder_name,
    is_valid_crop,
    mean_color_mosaic,
    compute_color_profile,
    color_distance,
    _rgb_to_lab,
    MATERIALS_LIST,
    STUFF_DIR,
    CROP_SIZE,
    TILE_SIZE,
    EXTRA_TEXTURE_TRIALS,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_CROPS_OUT = os.path.join(SCRIPT_DIR, "v2_crops")
V2_MOSAICS_OUT = os.path.join(SCRIPT_DIR, "v2_mosaics")

RING_FRACTIONS = [0.10, 0.25, 0.45, 0.70]  # fraction of max usable radius
MAX_RING_RADIUS = 150  # px — outer radius clamped here
CROPS_PER_RING = 5
RANDOM_SEED = 42

BLACK_THRESH = 15
MIN_CONTENT = 0.85
DEFAULT_MIN_TILE_STD = 8.0
# ──────────────────────────────────────────────────────────────────────────────


def mosaic_tile_colors(mosaic_arr, tile_size):
    """Extract non-black tile colors from a generated mosaic.

    Each tile in the mosaic is a uniform color block. We sample the center
    pixel of each tile and keep only non-black tiles (where the tile has
    content). Returns an Nx3 uint8 RGB array.
    """
    h, w = mosaic_arr.shape[:2]
    bh = h // tile_size
    bw = w // tile_size
    colors = []
    half = tile_size // 2
    for r in range(bh):
        for c in range(bw):
            cy = r * tile_size + half
            cx = c * tile_size + half
            pixel = mosaic_arr[cy, cx]
            # Skip black (empty) tiles
            if pixel.max() > 0:
                colors.append(pixel)
    if len(colors) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.array(colors, dtype=np.uint8)


def mosaic_color_std(tile_colors_rgb):
    """Compute color spread across mosaic tiles in LAB space.

    Converts tile colors to CIELAB and returns the combined standard
    deviation: sqrt(var_L + var_a + var_b). Higher values indicate more
    color variation across tiles.
    """
    if len(tile_colors_rgb) < 2:
        return 0.0
    # Shape tile colors as 1-row image for _rgb_to_lab: (1, N, 3)
    rgb_row = tile_colors_rgb[np.newaxis, :, :]
    lab_row = _rgb_to_lab(rgb_row)  # (1, N, 3)
    lab = lab_row[0]  # (N, 3)
    var_per_channel = np.var(lab, axis=0)  # (3,)
    return float(np.sqrt(var_per_channel.sum()))


def compute_anchors(img_w, img_h, crop_size, max_ring_radius):
    """Place 5 anchor points in a quincunx pattern within the usable area.

    The usable area is the region where an anchor can sit such that even
    the outermost ring crop stays inside the image.  If the image is too
    small for the quincunx, falls back to a single center anchor.

    Returns a list of (cx, cy) tuples.
    """
    half = crop_size // 2
    margin = half + max_ring_radius  # anchor must be this far from edges

    usable_x0 = margin
    usable_y0 = margin
    usable_x1 = img_w - margin
    usable_y1 = img_h - margin

    if usable_x1 <= usable_x0 or usable_y1 <= usable_y0:
        # Image too small for even one anchor with full rings —
        # fall back to center only
        cx = img_w // 2
        cy = img_h // 2
        return [(cx, cy)]

    mid_x = (usable_x0 + usable_x1) // 2
    mid_y = (usable_y0 + usable_y1) // 2
    q1_x = (usable_x0 + mid_x) // 2
    q3_x = (mid_x + usable_x1) // 2
    q1_y = (usable_y0 + mid_y) // 2
    q3_y = (mid_y + usable_y1) // 2

    return [
        (mid_x, mid_y),   # center
        (q1_x, q1_y),     # top-left quadrant
        (q3_x, q1_y),     # top-right quadrant
        (q1_x, q3_y),     # bottom-left quadrant
        (q3_x, q3_y),     # bottom-right quadrant
    ]


def compute_ring_radii(img_w, img_h, crop_size):
    """Compute 4 ring radii in pixels, clamped to MAX_RING_RADIUS.

    The max usable radius is the largest ring radius such that a crop
    centered on the ring perimeter can still fit inside the image
    (relative to a centered anchor). This is then clamped to MAX_RING_RADIUS.
    """
    half = crop_size // 2
    max_r_x = img_w // 2 - half
    max_r_y = img_h // 2 - half
    max_usable = min(max_r_x, max_r_y, MAX_RING_RADIUS)
    max_usable = max(max_usable, 1)  # avoid zero

    return [int(round(f * max_usable)) for f in RING_FRACTIONS]


def place_ring_crops(arr, anchor_cx, anchor_cy, radii, crop_size,
                     crops_per_ring, angle_offset, black_thresh, min_content):
    """Place crops_per_ring crops at evenly spaced angles for each ring.

    Returns a list of (ring_idx, crop_idx, crop_array) tuples.
    Only valid, in-bounds crops are included.
    """
    h, w = arr.shape[:2]
    half = crop_size // 2
    results = []

    for ring_idx, radius in enumerate(radii):
        angle_step = 2 * math.pi / crops_per_ring
        crop_count = 0
        for ci in range(crops_per_ring):
            angle = angle_offset + ci * angle_step
            cx = int(round(anchor_cx + radius * math.cos(angle)))
            cy = int(round(anchor_cy + radius * math.sin(angle)))

            # Top-left corner of the crop
            x0 = cx - half
            y0 = cy - half

            # Bounds check
            if x0 < 0 or y0 < 0 or x0 + crop_size > w or y0 + crop_size > h:
                continue

            if not is_valid_crop(arr, x0, y0, crop_size, black_thresh, min_content):
                continue

            crop = arr[y0:y0 + crop_size, x0:x0 + crop_size].copy()
            crop_count += 1
            results.append((ring_idx, crop_count, crop))

    return results


def process_image_v2(arr, folder, img_id, crop_size, crops_out, mosaics_out,
                     black_thresh, min_content, tile_size, min_tile_std=0.0,
                     make_mosaics=True):
    """Orchestrate anchors -> rings -> save crops (+ mosaics) for one image.

    If make_mosaics is True, generates mosaic-scrambled versions and applies
    the variance filter (min_tile_std). If False, saves crops only.
    """
    h, w = arr.shape[:2]

    if h < crop_size or w < crop_size:
        print(f"    Image too small ({w}x{h}), skipping.")
        return 0, 0, 0

    radii = compute_ring_radii(w, h, crop_size)
    anchors = compute_anchors(w, h, crop_size, max(radii))

    total_crops = 0
    total_mosaics = 0
    total_rejected = 0

    for ai, (acx, acy) in enumerate(anchors):
        # Random angular offset per anchor for variety
        angle_offset = np.random.uniform(0, 2 * math.pi)

        ring_crops = place_ring_crops(
            arr, acx, acy, radii, crop_size,
            CROPS_PER_RING, angle_offset, black_thresh, min_content,
        )

        for ring_idx, crop_idx, crop_arr in ring_crops:
            if make_mosaics:
                mosaic = mean_color_mosaic(crop_arr, tile_size, black_thresh)

                # Variance filter: reject homogeneous mosaics
                if min_tile_std > 0:
                    tile_colors = mosaic_tile_colors(mosaic, tile_size)
                    std_val = mosaic_color_std(tile_colors)
                    if std_val < min_tile_std:
                        total_rejected += 1
                        continue

            fname = f"{folder}_{img_id:04d}_a{ai}_r{ring_idx}_c{crop_idx}.png"
            Image.fromarray(crop_arr).save(os.path.join(crops_out, fname))
            total_crops += 1

            if make_mosaics:
                Image.fromarray(mosaic).save(
                    os.path.join(mosaics_out, f"mosaic_{fname}"))
                total_mosaics += 1

    return total_crops, total_mosaics, total_rejected


def process_trials_v2(trials, task_label, min_tile_std=0.0, make_mosaics=True):
    """Loop over all trials, call process_image_v2 for each."""
    print("=" * 60)
    print(f"V2 STRUCTURED RINGS — {task_label.upper()}")
    print("=" * 60)
    if make_mosaics and min_tile_std > 0:
        print(f"  Variance filter: min_tile_std = {min_tile_std:.1f} LAB units")
    if not make_mosaics:
        print("  Crops only (no mosaics)")

    os.makedirs(V2_CROPS_OUT, exist_ok=True)
    if make_mosaics:
        os.makedirs(V2_MOSAICS_OUT, exist_ok=True)

    grand_crops = 0
    grand_mosaics = 0
    grand_rejected = 0

    for name, img_id in trials:
        arr, folder = load_image(name, img_id)
        if arr is None:
            continue

        print(f"  {name} / {img_id:04d} ({arr.shape[1]}x{arr.shape[0]})")

        nc, nm, nr = process_image_v2(
            arr, folder, img_id, CROP_SIZE,
            V2_CROPS_OUT, V2_MOSAICS_OUT,
            BLACK_THRESH, MIN_CONTENT, TILE_SIZE,
            min_tile_std=min_tile_std,
            make_mosaics=make_mosaics,
        )
        grand_crops += nc
        grand_mosaics += nm
        grand_rejected += nr
        if nr > 0:
            print(f"    accepted={nc}, rejected={nr}")

    print(f"  -> {grand_crops} crops saved to {V2_CROPS_OUT}/")
    if make_mosaics:
        print(f"  -> {grand_mosaics} mosaics saved to {V2_MOSAICS_OUT}/")
        if grand_rejected > 0:
            print(f"  -> {grand_rejected} mosaics rejected (below variance threshold)")
    print()


def filter_by_categories(trials, max_categories):
    """Keep only trials whose category is among the first *max_categories* unique categories."""
    seen = []
    for name, _ in trials:
        if name not in seen:
            seen.append(name)
    keep = set(seen[:max_categories])
    return [(n, i) for n, i in trials if n in keep]


def _parse_mosaic_filename(fname):
    """Parse a mosaic filename into (category, image_id).

    Expected format: mosaic_<category>_<imgid>_a<anchor>_r<ring>_c<crop>.png
    Returns (category, image_id_str) or None if parsing fails.
    """
    m = re.match(r'^mosaic_(.+?)_(\d{4})_a\d+_r\d+_c\d+\.png$', fname)
    if m:
        return m.group(1), m.group(2)
    return None


def _find_best_distractor_triplet(distance_matrix, indices):
    """Find the 3 indices (from `indices`) with lowest avg pairwise distance."""
    best_triplet = None
    best_avg = float('inf')
    idx_list = list(indices)
    for triplet in combinations(idx_list, 3):
        i, j, k = triplet
        avg = (distance_matrix[i, j] + distance_matrix[i, k] +
               distance_matrix[j, k]) / 3.0
        if avg < best_avg:
            best_avg = avg
            best_triplet = triplet
    return best_triplet, best_avg


def assemble_trials(mosaics_dir, color_trials, max_trials_per_image=3,
                    min_oddball_distance=0.02):
    """Assemble 4AFC oddball trials from filtered mosaics.

    Only mosaics whose category matches a color trial are included.
    Groups mosaics by (category, image_id), computes pairwise color
    distances, finds distractor triplets, and selects oddballs at
    easy/medium/hard difficulty levels. Writes v2_trials.csv.
    """
    print("=" * 60)
    print("TRIAL ASSEMBLY")
    print("=" * 60)

    # Build set of (resolved_folder, image_id_str) from color trials
    color_keys = set()
    for name, img_id in color_trials:
        folder = resolve_folder_name(name)
        color_keys.add((folder, f"{img_id:04d}"))

    # Collect mosaic files grouped by (category, image_id), color only
    groups = {}
    skipped = 0
    for fname in sorted(os.listdir(mosaics_dir)):
        if not fname.endswith('.png'):
            continue
        parsed = _parse_mosaic_filename(fname)
        if parsed is None:
            continue
        category, img_id = parsed
        if (category, img_id) not in color_keys:
            skipped += 1
            continue
        key = (category, img_id)
        groups.setdefault(key, []).append(fname)

    print(f"  Found {sum(len(v) for v in groups.values())} color mosaics "
          f"across {len(groups)} image groups (skipped {skipped} texture mosaics)")

    trials_csv = os.path.join(os.path.dirname(mosaics_dir), "v2_trials.csv")
    trial_rows = []
    trial_id = 0

    for (category, img_id), fnames in sorted(groups.items()):
        if len(fnames) < 4:
            continue  # need at least 4 mosaics for a trial

        # Load color profiles
        profiles = []
        for fname in fnames:
            img = Image.open(os.path.join(mosaics_dir, fname)).convert("RGB")
            profiles.append(compute_color_profile(np.array(img)))

        n = len(profiles)
        # Compute pairwise distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = color_distance(profiles[i], profiles[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        used = set()
        trials_for_image = 0

        while trials_for_image < max_trials_per_image:
            available = [i for i in range(n) if i not in used]
            if len(available) < 4:
                break

            # Find best distractor triplet from available mosaics
            triplet, _ = _find_best_distractor_triplet(dist_matrix, available)
            if triplet is None:
                break

            # Score remaining mosaics as oddball candidates
            remaining = [i for i in available if i not in triplet]
            if not remaining:
                break

            # Compute avg distance of each candidate to the distractor triplet
            oddball_scores = []
            for idx in remaining:
                avg_dist = np.mean([dist_matrix[idx, t] for t in triplet])
                if avg_dist >= min_oddball_distance:
                    oddball_scores.append((idx, avg_dist))

            if not oddball_scores:
                break

            # Sort by distance (ascending)
            oddball_scores.sort(key=lambda x: x[1])

            # Select oddballs at different difficulty levels
            difficulties = {}
            difficulties['hard'] = oddball_scores[0]  # smallest distance
            difficulties['easy'] = oddball_scores[-1]  # largest distance
            if len(oddball_scores) >= 3:
                mid = len(oddball_scores) // 2
                difficulties['medium'] = oddball_scores[mid]
            elif len(oddball_scores) >= 2:
                difficulties['medium'] = oddball_scores[-1]
                difficulties['easy'] = oddball_scores[-1]

            for difficulty, (oddball_idx, dist_val) in difficulties.items():
                trial_id += 1
                trial_rows.append({
                    'trial_id': trial_id,
                    'category': category,
                    'image_id': img_id,
                    'difficulty': difficulty,
                    'color_distance': f"{dist_val:.6f}",
                    'oddball': fnames[oddball_idx],
                    'distractor_1': fnames[triplet[0]],
                    'distractor_2': fnames[triplet[1]],
                    'distractor_3': fnames[triplet[2]],
                })

            # Mark used mosaics
            used.update(triplet)
            # Also mark the oddball with the largest distance to avoid reuse
            used.add(oddball_scores[-1][0])
            if len(oddball_scores) >= 3:
                used.add(oddball_scores[0][0])
                used.add(oddball_scores[len(oddball_scores) // 2][0])

            trials_for_image += 1

    # Write CSV
    fieldnames = ['trial_id', 'category', 'image_id', 'difficulty',
                  'color_distance', 'oddball', 'distractor_1',
                  'distractor_2', 'distractor_3']
    with open(trials_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)

    print(f"  -> {len(trial_rows)} trials written to {trials_csv}")
    # Summary by difficulty
    for diff in ['easy', 'medium', 'hard']:
        count = sum(1 for r in trial_rows if r['difficulty'] == diff)
        if count > 0:
            print(f"     {diff}: {count} trials")
    print()


def main():
    parser = argparse.ArgumentParser(description="V2 structured-rings color/texture sampler")
    parser.add_argument("--max-categories", type=int, default=None,
                        help="Process only the first N unique categories (default: all)")
    parser.add_argument("--min-tile-std", type=float, default=DEFAULT_MIN_TILE_STD,
                        help=f"Min tile color std in LAB space; mosaics below this are "
                             f"rejected (default: {DEFAULT_MIN_TILE_STD})")
    parser.add_argument("--assemble-trials", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Assemble 4AFC oddball trials after generation (default: on)")
    parser.add_argument("--max-trials-per-image", type=int, default=3,
                        help="Max trials assembled per source image (default: 3)")
    parser.add_argument("--min-oddball-distance", type=float, default=0.02,
                        help="Min color distance for an oddball to be usable (default: 0.02)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    parser.add_argument("--regenerate-original", action="store_true",
                        help="Reproduce the original v2 run (mosaics for all, no filter)")
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    texture_trials = read_trials(MATERIALS_LIST, "texture") + EXTRA_TEXTURE_TRIALS
    color_trials = read_trials(MATERIALS_LIST, "color")

    if args.max_categories is not None:
        texture_trials = filter_by_categories(texture_trials, args.max_categories)
        color_trials = filter_by_categories(color_trials, args.max_categories)

    print(f"Texture trials: {len(texture_trials)}")
    print(f"Color trials:   {len(color_trials)}\n")

    if args.regenerate_original:
        # Reproduce the original v2 run: mosaics for all, no variance filter
        print("** REGENERATE-ORIGINAL MODE: mosaics for all, no filter **\n")
        process_trials_v2(texture_trials, "texture", make_mosaics=True)
        process_trials_v2(color_trials, "color", make_mosaics=True)
    else:
        process_trials_v2(texture_trials, "texture", make_mosaics=False)
        process_trials_v2(color_trials, "color", min_tile_std=args.min_tile_std, make_mosaics=True)

    if args.assemble_trials:
        assemble_trials(
            V2_MOSAICS_OUT,
            color_trials,
            max_trials_per_image=args.max_trials_per_image,
            min_oddball_distance=args.min_oddball_distance,
        )

    print("All done.")


if __name__ == "__main__":
    main()
