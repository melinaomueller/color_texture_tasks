"""
generate_match_bitmaps.py
-------------------------
Generate additional oddball-match bitmaps for the matching task.

For each matching trial, generates 3 new bitmaps that use the SAME oddball
color proportions as the original part1_oddball, but with DIFFERENT random
spatial arrangements. These serve as the "correct match" in the matching task
(looks similar to the oddball but is a different image).

Output naming: t{id}_match_v{1,2,3}_{version}.png  in jspsych_stimuli/

Usage:
    python3 generate_match_bitmaps.py
"""

import os
import sys
import csv
import numpy as np
from PIL import Image

# Import from bitmap_trials.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitmap_trials import (
    extract_palette, merge_colors, equalize_luminance,
    chroma_boost_and_equalize, generate_bitmap, shift_proportions,
    load_stuff_image, RANDOM_SEED, CHROMA_BOOST_FACTOR,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bitmap_trials", "jspsych_stimuli")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "trial_manifest.csv")

# ── Matching trial list (54 trials from the matching split) ──────────────────
MATCHING_TRIALS = [
    (1, "chroma_boost_lum_eq"),
    (3, "chroma_boost_lum_eq"),
    (5, "original"),
    (6, "chroma_boost_lum_eq"),
    (13, "chroma_boost_lum_eq"),
    (14, "original"),
    (15, "original"),
    (16, "original"),
    (18, "original"),
    (19, "original"),
    (20, "original"),
    (23, "chroma_boost_lum_eq"),
    (24, "chroma_boost_lum_eq"),
    (25, "chroma_boost_lum_eq"),
    (28, "original"),
    (32, "chroma_boost_lum_eq"),
    (32, "lum_eq"),
    (34, "chroma_boost_lum_eq"),
    (34, "original"),
    (35, "original"),
    (38, "chroma_boost_lum_eq"),
    (38, "original"),
    (39, "original"),
    (44, "original"),
    (45, "original"),
    (46, "original"),
    (47, "lum_eq"),
    (48, "original"),
    (52, "chroma_boost_lum_eq"),
    (52, "lum_eq"),
    (54, "original"),
    (55, "original"),
    (57, "original"),
    (58, "chroma_boost_lum_eq"),
    (58, "lum_eq"),
    (58, "original"),
    (61, "lum_eq"),
    (63, "original"),
    (65, "chroma_boost_lum_eq"),
    (65, "original"),
    (67, "original"),
    (68, "original"),
    (70, "chroma_boost_lum_eq"),
    (70, "lum_eq"),
    (71, "chroma_boost_lum_eq"),
    (74, "original"),
    (79, "chroma_boost_lum_eq"),
    (79, "original"),
    (81, "original"),
    (82, "lum_eq"),
    (83, "chroma_boost_lum_eq"),
    (84, "original"),
    (85, "chroma_boost_lum_eq"),
    (85, "original"),
]

N_MATCH_VERSIONS = 3
MATCH_SEED_OFFSET = 5000  # separate seed space from originals


def load_manifest():
    """Load trial manifest into a dict keyed by trial_id."""
    manifest = {}
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["trial_id"])
            manifest[tid] = row
    return manifest


def recompute_oddball_props(manifest_row, seed=RANDOM_SEED):
    """Recompute palette and oddball proportions for a trial, matching the
    exact process used in bitmap_trials.py."""
    category = manifest_row["category"].strip()
    image_id = manifest_row["image_id"].strip()
    n_colors = int(manifest_row["n_colors"])
    difficulty = float(manifest_row["difficulty"])
    min_lightness = float(manifest_row.get("min_lightness") or 0)
    chroma_boost = float(manifest_row.get("chroma_boost") or CHROMA_BOOST_FACTOR)
    trial_id = int(manifest_row["trial_id"])

    # Load source image
    arr = load_stuff_image(category, image_id)
    if arr is None:
        return None

    # Extract k=5 palette (same seed as original)
    colors, proportions = extract_palette(arr, k=5, seed=seed,
                                          min_lightness=min_lightness)

    # Merge if needed
    if n_colors < 5:
        colors, proportions = merge_colors(colors, proportions, n_colors)

    # Recompute oddball proportions using same trial RNG
    trial_rng = np.random.RandomState(seed + trial_id)
    oddball_props = shift_proportions(proportions, difficulty, trial_rng)

    # Build version colors
    versions = {
        "original": colors.copy(),
        "lum_eq": equalize_luminance(colors),
        "chroma_boost_lum_eq": chroma_boost_and_equalize(colors,
                                                          boost=chroma_boost),
    }

    return {
        "colors": versions,
        "oddball_props": oddball_props,
        "distractor_props": proportions,
        "trial_id": trial_id,
        "category": category,
        "image_id": image_id,
    }


def main():
    manifest = load_manifest()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Deduplicate: we only need to recompute palette once per trial_id
    unique_ids = sorted(set(tid for tid, _ in MATCHING_TRIALS))
    trial_data = {}
    for tid in unique_ids:
        if tid not in manifest:
            print(f"  WARNING: trial {tid} not in manifest, skipping")
            continue
        data = recompute_oddball_props(manifest[tid])
        if data is None:
            print(f"  WARNING: could not load image for trial {tid}, skipping")
            continue
        trial_data[tid] = data

    generated = 0
    for trial_id, version in MATCHING_TRIALS:
        if trial_id not in trial_data:
            continue
        data = trial_data[trial_id]
        version_colors = data["colors"][version]
        oddball_props = data["oddball_props"]
        tid_str = f"t{trial_id:02d}"

        for v in range(1, N_MATCH_VERSIONS + 1):
            # Use a unique seed per (trial_id, version, match_version)
            version_offset = {"original": 0, "lum_eq": 100,
                              "chroma_boost_lum_eq": 200}[version]
            match_rng = np.random.RandomState(
                MATCH_SEED_OFFSET + trial_id * 10 + version_offset + v)

            bitmap = generate_bitmap(version_colors, oddball_props, match_rng)
            fname = f"{tid_str}_match_v{v}_{version}.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            Image.fromarray(bitmap).save(out_path)
            generated += 1

        print(f"  Trial {trial_id:02d}/{version}: 3 match variants generated")

    print(f"\nDone: {generated} match bitmaps saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
