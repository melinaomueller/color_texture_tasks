"""
generate_regen_v3.py
--------------------
Regenerate match bitmaps for the 7 remaining problematic trials at larger
difficulty shifts: 0.25, 0.30, 0.35.

Output naming:
  t{id}_regen_oddball_d{25,30,35}_{version}.png
  t{id}_regen_match_d{25,30,35}_{version}.png
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

DIFFICULTIES = [0.25, 0.30, 0.35]
REGEN_SEED_OFFSET = 9000

FLAGGED_TRIALS = [
    (38, "original"),              # "too similar, need more red"
    (47, "lum_eq"),                # "change hue, still not right"
    (48, "original"),              # "need more red or white space"
    (54, "original"),              # "still very difficult, rescramble"
    (57, "original"),              # "still look very similar"
    (58, "chroma_boost_lum_eq"),   # "need more/less blue"
    (85, "chroma_boost_lum_eq"),   # "purple and gray too similar"
]


def load_manifest():
    manifest = {}
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[int(row["trial_id"])] = row
    return manifest


def get_palette_and_versions(manifest_row, seed=RANDOM_SEED):
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
    return {"colors": versions, "base_proportions": proportions}


def main():
    manifest = load_manifest()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    palette_cache = {}
    for tid in sorted(set(t for t, _ in FLAGGED_TRIALS)):
        if tid not in manifest:
            print(f"  WARNING: trial {tid} not in manifest")
            continue
        data = get_palette_and_versions(manifest[tid])
        if data is None:
            print(f"  WARNING: could not load trial {tid}")
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
            version_offset = {"original": 0, "lum_eq": 1,
                              "chroma_boost_lum_eq": 2}[version]

            shift_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 100 + int(diff * 100))
            oddball_props = shift_proportions(base_props, diff, shift_rng)

            oddball_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 1000 + int(diff * 100) * 10
                + version_offset)
            oddball_bmp = generate_bitmap(version_colors, oddball_props,
                                          oddball_rng)
            Image.fromarray(oddball_bmp).save(
                os.path.join(OUTPUT_DIR,
                             f"{tid_str}_regen_oddball_{d_label}_{version}.png"))

            match_rng = np.random.RandomState(
                REGEN_SEED_OFFSET + trial_id * 1000 + int(diff * 100) * 10
                + version_offset + 500)
            match_bmp = generate_bitmap(version_colors, oddball_props,
                                        match_rng)
            Image.fromarray(match_bmp).save(
                os.path.join(OUTPUT_DIR,
                             f"{tid_str}_regen_match_{d_label}_{version}.png"))
            generated += 2

        print(f"  Trial {trial_id:02d}/{version}: 3 difficulty levels generated")

    print(f"\nDone: {generated} bitmaps saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
