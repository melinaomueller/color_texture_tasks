"""
generate_regen_bitmaps.py
-------------------------
For flagged matching trials, regenerate oddball + match bitmaps at 3 difficulty
levels (0.15, 0.17, 0.20) so the researcher can compare and pick the best.

Output naming:
  t{id}_regen_oddball_d{15,17,20}_{version}.png
  t{id}_regen_match_d{15,17,20}_{version}.png

Usage:
    python3 generate_regen_bitmaps.py
"""

import os
import sys
import csv
import numpy as np
from PIL import Image

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

DIFFICULTIES = [0.15, 0.17, 0.20]
REGEN_SEED_OFFSET = 8000  # separate seed space

# Trials with negative pilot feedback (trial_id, version)
FLAGGED_TRIALS = [
    (5, "original"),               # "makes no sense"
    (25, "chroma_boost_lum_eq"),   # "need one more similar to study"
    (32, "chroma_boost_lum_eq"),   # "distractor 1 better as match"
    (32, "lum_eq"),                # "need more pink"
    (38, "chroma_boost_lum_eq"),   # "tricky, not sure"
    (38, "original"),              # "maybe B?" — borderline
    (39, "original"),              # "pretty hard"
    (47, "lum_eq"),                # "dull, adjust hue, swap"
    (48, "original"),              # "hard"
    (52, "chroma_boost_lum_eq"),   # "difficult, adjust difficulty"
    (52, "lum_eq"),                # "very difficult"
    (54, "original"),              # "very difficult"
    (55, "original"),              # "very difficult"
    (57, "original"),              # "very difficult"
    (58, "chroma_boost_lum_eq"),   # "very difficult"
    (58, "lum_eq"),                # "very difficult"
    (58, "original"),              # "very difficult"
    (65, "chroma_boost_lum_eq"),   # "very difficult"
    (68, "original"),              # "very difficult"
    (70, "chroma_boost_lum_eq"),   # "hard"
    (79, "original"),              # "matches not great"
    (85, "chroma_boost_lum_eq"),   # "adjust hue"
]


def load_manifest():
    manifest = {}
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[int(row["trial_id"])] = row
    return manifest


def get_palette_and_versions(manifest_row, seed=RANDOM_SEED):
    """Extract palette and build all 3 color versions."""
    category = manifest_row["category"].strip()
    image_id = manifest_row["image_id"].strip()
    n_colors = int(manifest_row["n_colors"])
    min_lightness = float(manifest_row.get("min_lightness") or 0)
    chroma_boost = float(manifest_row.get("chroma_boost") or CHROMA_BOOST_FACTOR)

    arr = load_stuff_image(category, image_id)
    if arr is None:
        return None

    colors, proportions = extract_palette(arr, k=5, seed=seed,
                                          min_lightness=min_lightness)
    if n_colors < 5:
        colors, proportions = merge_colors(colors, proportions, n_colors)

    versions = {
        "original": colors.copy(),
        "lum_eq": equalize_luminance(colors),
        "chroma_boost_lum_eq": chroma_boost_and_equalize(colors,
                                                          boost=chroma_boost),
    }

    return {
        "colors": versions,
        "base_proportions": proportions,
        "n_colors": n_colors,
    }


def main():
    manifest = load_manifest()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cache palette data per trial_id
    palette_cache = {}
    unique_ids = sorted(set(tid for tid, _ in FLAGGED_TRIALS))
    for tid in unique_ids:
        if tid not in manifest:
            print(f"  WARNING: trial {tid} not in manifest, skipping")
            continue
        data = get_palette_and_versions(manifest[tid])
        if data is None:
            print(f"  WARNING: could not load image for trial {tid}, skipping")
            continue
        palette_cache[tid] = data

    generated = 0
    for trial_id, version in FLAGGED_TRIALS:
        if trial_id not in palette_cache:
            continue

        data = palette_cache[trial_id]
        version_colors = data["colors"][version]
        base_props = data["base_proportions"]
        tid_str = f"t{trial_id:02d}"

        for diff in DIFFICULTIES:
            d_label = f"d{int(diff * 100)}"

            # Compute oddball proportions at this difficulty
            # Use a consistent RNG per (trial_id, difficulty) so results are stable
            shift_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 100 + int(diff * 100))
            oddball_props = shift_proportions(base_props, diff, shift_rng)

            # Version offset for seed uniqueness
            version_offset = {"original": 0, "lum_eq": 1,
                              "chroma_boost_lum_eq": 2}[version]

            # Generate oddball bitmap
            oddball_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 1000 + int(diff * 100) * 10
                + version_offset)
            oddball_bmp = generate_bitmap(version_colors, oddball_props,
                                          oddball_rng)
            oddball_fname = f"{tid_str}_regen_oddball_{d_label}_{version}.png"
            Image.fromarray(oddball_bmp).save(
                os.path.join(OUTPUT_DIR, oddball_fname))

            # Generate match bitmap (same proportions, different spatial layout)
            match_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 1000 + int(diff * 100) * 10
                + version_offset + 500)
            match_bmp = generate_bitmap(version_colors, oddball_props,
                                        match_rng)
            match_fname = f"{tid_str}_regen_match_{d_label}_{version}.png"
            Image.fromarray(match_bmp).save(
                os.path.join(OUTPUT_DIR, match_fname))

            generated += 2

        print(f"  Trial {trial_id:02d}/{version}: "
              f"3 difficulty levels x (oddball + match) generated")

    print(f"\nDone: {generated} bitmaps saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
