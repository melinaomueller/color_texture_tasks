"""
generate_texture_crops.py
-------------------------
Generate 200x200 grayscale crop stimuli for the texture oddball task.

Each 4AFC trial has:
  - 3 distractor crops from the "same3" image
  - 1 oddball crop from the "diff1" image (possibly a different category)

Reads trial specs from texture_manifest.csv. Outputs:
  1. texture_trials/<trial_id>_<category>/  — per-trial folders with crops + metadata
  2. texture_trials/jspsych_stimuli/        — flat folder for jsPsych experiment

USAGE:
    python3 generate_texture_crops.py
    python3 generate_texture_crops.py --manifest texture_manifest.csv --output-dir texture_trials
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
STUFF_DIR = os.path.join(PROJECT_DIR, "STUFF_enhanced_dataset_3514_images")

CROP_SIZE = 200
MIN_CONTENT = 0.85
BLACK_THRESH = 15
MAX_ATTEMPTS = 10000
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_stuff_image(category, image_id):
    """Load an image from the STUFF dataset. Returns RGB numpy array or None."""
    path = os.path.join(STUFF_DIR, category, f"{int(image_id):04d}.jpg")
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found")
        return None
    img = Image.open(path).convert("RGB")
    return np.array(img)


def is_valid_crop(arr, x, y, size, black_thresh, min_content):
    """Check that a crop has enough non-black content."""
    crop = arr[y:y + size, x:x + size]
    return (crop.max(axis=2) >= black_thresh).mean() > min_content


def crops_too_close(new_x, new_y, existing, min_dist):
    """Check if a new crop position is too close to existing ones."""
    for ex, ey in existing:
        if abs(new_x - ex) < min_dist and abs(new_y - ey) < min_dist:
            return True
    return False


def sample_crops(arr, n, rng, min_dist=200, source_size=None):
    """Sample n valid, non-overlapping crops from an RGB image.

    Crops are extracted at source_size x source_size, then downscaled to
    CROP_SIZE x CROP_SIZE using LANCZOS resampling.

    Args:
        arr: (H, W, 3) uint8 RGB image
        n: number of crops to sample
        rng: numpy RandomState
        min_dist: minimum distance between crop origins (200 = non-overlapping)
        source_size: size of the region to crop from the source image
                     (default: CROP_SIZE for backwards compatibility)

    Returns:
        list of (crop_array, x, y) tuples
    """
    if source_size is None:
        source_size = CROP_SIZE
    h, w = arr.shape[:2]
    if h < source_size or w < source_size:
        print(f"    Image too small ({w}x{h}) for {source_size}x{source_size} crops, skipping.")
        return []

    x_hi = w - source_size
    y_hi = h - source_size
    centers = []
    results = []
    attempts = 0

    while len(results) < n and attempts < MAX_ATTEMPTS:
        x = rng.randint(0, x_hi + 1)
        y = rng.randint(0, y_hi + 1)
        if (is_valid_crop(arr, x, y, source_size, BLACK_THRESH, MIN_CONTENT)
                and not crops_too_close(x, y, centers, min_dist)):
            centers.append((x, y))
            crop = arr[y:y + source_size, x:x + source_size].copy()
            if source_size != CROP_SIZE:
                crop = np.array(Image.fromarray(crop).resize(
                    (CROP_SIZE, CROP_SIZE), Image.LANCZOS))
            results.append((crop, x, y))
        attempts += 1

    if len(results) < n:
        print(f"    Warning: only found {len(results)}/{n} valid crops "
              f"after {attempts} attempts.")
    return results


def to_grayscale(rgb_arr):
    """Convert an RGB array to single-channel grayscale (uint8)."""
    return np.dot(rgb_arr[..., :3].astype(np.float64),
                  [0.2989, 0.5870, 0.1140]).round().clip(0, 255).astype(np.uint8)


def normalize_luminance(oddball_gray, distractor_grays):
    """Match oddball's mean and std to the average of distractors.

    Args:
        oddball_gray: (H, W) uint8 grayscale array
        distractor_grays: list of (H, W) uint8 grayscale arrays

    Returns:
        normalized: (H, W) uint8 grayscale array
    """
    odd = oddball_gray.astype(np.float64)
    odd_mean = odd.mean()
    odd_std = odd.std()

    dist_means = [d.astype(np.float64).mean() for d in distractor_grays]
    dist_stds = [d.astype(np.float64).std() for d in distractor_grays]
    target_mean = np.mean(dist_means)
    target_std = np.mean(dist_stds)

    if odd_std < 1e-6:
        # Flat image — just shift to target mean
        normalized = np.full_like(odd, target_mean)
    else:
        normalized = (odd - odd_mean) * (target_std / odd_std) + target_mean

    return normalized.clip(0, 255).round().astype(np.uint8)


def process_trial(row, output_dir, jspsych_dir):
    """Process a single trial row from the manifest.

    Returns True on success, False on failure.
    """
    trial_id = int(row["trial_id"])
    category = row["category"].strip()
    same3_image = row["same3_image"].strip()
    diff1_category = row["diff1_category"].strip()
    diff1_image = row["diff1_image"].strip()
    min_dist = int(row.get("min_dist") or CROP_SIZE)
    normalize = row.get("normalize", "").strip().lower() in ("true", "1", "yes")
    crop_source_size = int(row.get("crop_source_size") or CROP_SIZE)
    notes = row.get("notes", "").strip()

    # Trial-specific RNG (matches color pipeline convention)
    rng = np.random.RandomState(RANDOM_SEED + trial_id)

    print(f"  Trial {trial_id:02d}: {category}/{same3_image} vs "
          f"{diff1_category}/{diff1_image}"
          f" (source crop {crop_source_size}x{crop_source_size})"
          f"{' [normalize]' if normalize else ''}"
          f"{f' ({notes})' if notes else ''} ... ", end="")

    # Load source images
    same3_arr = load_stuff_image(category, same3_image)
    if same3_arr is None:
        print("SKIP (same3 image not found)")
        return False

    diff1_arr = load_stuff_image(diff1_category, diff1_image)
    if diff1_arr is None:
        print("SKIP (diff1 image not found)")
        return False

    # Sample 3 distractor crops from same3 image
    distractor_results = sample_crops(same3_arr, 3, rng, min_dist=min_dist,
                                      source_size=crop_source_size)
    if len(distractor_results) < 3:
        print(f"SKIP (only got {len(distractor_results)}/3 distractor crops)")
        return False

    # Sample 1 oddball crop from diff1 image
    oddball_results = sample_crops(diff1_arr, 1, rng, min_dist=0,
                                   source_size=crop_source_size)
    if len(oddball_results) < 1:
        print("SKIP (could not get oddball crop)")
        return False

    # Convert to grayscale
    distractor_grays = [to_grayscale(r[0]) for r in distractor_results]
    oddball_gray = to_grayscale(oddball_results[0][0])

    # Optionally normalize oddball luminance
    if normalize:
        oddball_gray = normalize_luminance(oddball_gray, distractor_grays)

    # Create trial output folder
    trial_folder = os.path.join(output_dir, f"trial_{trial_id:02d}_{category}")
    os.makedirs(trial_folder, exist_ok=True)

    # Save crops
    for i, gray in enumerate(distractor_grays, start=1):
        img = Image.fromarray(gray, mode="L")
        img.save(os.path.join(trial_folder, f"distractor_{i}.png"))
        img.save(os.path.join(jspsych_dir, f"t{trial_id:02d}_distractor_{i}.png"))

    oddball_img = Image.fromarray(oddball_gray, mode="L")
    oddball_img.save(os.path.join(trial_folder, "oddball.png"))
    oddball_img.save(os.path.join(jspsych_dir, f"t{trial_id:02d}_oddball.png"))

    # Write metadata
    info_lines = [
        f"trial_id: {trial_id}",
        f"category: {category}",
        f"same3_source: {category}/{same3_image}",
        f"diff1_source: {diff1_category}/{diff1_image}",
        f"min_dist: {min_dist}",
        f"crop_source_size: {crop_source_size}",
        f"normalize: {normalize}",
        f"notes: {notes}",
        "",
        "distractor crops (x, y):",
    ]
    for i, (_, x, y) in enumerate(distractor_results, start=1):
        info_lines.append(f"  distractor_{i}: ({x}, {y})")
    ox, oy = oddball_results[0][1], oddball_results[0][2]
    info_lines.append(f"  oddball: ({ox}, {oy})")

    if normalize:
        info_lines.append("")
        info_lines.append("luminance normalization applied:")
        odd_raw = to_grayscale(oddball_results[0][0])
        info_lines.append(f"  oddball original: mean={odd_raw.mean():.1f}, std={odd_raw.std():.1f}")
        info_lines.append(f"  oddball normalized: mean={oddball_gray.mean():.1f}, std={oddball_gray.std():.1f}")
        info_lines.append(f"  distractor target: mean={np.mean([d.mean() for d in distractor_grays]):.1f}, "
                          f"std={np.mean([d.std() for d in distractor_grays]):.1f}")

    with open(os.path.join(trial_folder, "trial_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    print("OK")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate grayscale texture crops for oddball task")
    parser.add_argument(
        "--manifest", type=str,
        default=os.path.join(SCRIPT_DIR, "texture_manifest.csv"),
        help="Path to texture_manifest.csv")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJECT_DIR, "texture_trials"),
        help="Output directory (default: Melina/texture_trials/)")
    args = parser.parse_args()

    manifest_path = args.manifest
    output_dir = args.output_dir

    # Read manifest
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        trials = list(reader)

    print(f"Loaded {len(trials)} trials from {manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    jspsych_dir = os.path.join(output_dir, "jspsych_stimuli")
    os.makedirs(jspsych_dir, exist_ok=True)

    success = 0
    fail = 0

    for row in trials:
        if process_trial(row, output_dir, jspsych_dir):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} succeeded, {fail} failed")
    print(f"Trial folders: {output_dir}/")
    print(f"jsPsych stimuli: {jspsych_dir}/")


if __name__ == "__main__":
    main()
