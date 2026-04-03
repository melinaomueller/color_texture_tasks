"""
mosaic_scramble.py
------------------
For a given image, samples 5 content-aware 200x200 crops and saves:
  - crop_1.png ... crop_5.png          (original crops)
  - crop_1_scrambled.png ... etc.      (mean-color tile scrambled versions)

USAGE:
    python mosaic_scramble.py

Edit the CONFIG section below to change inputs/outputs.
"""

import os
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_IMAGE   = "your_image.png"   # path to input image
OUTPUT_DIR    = "output_crops"     # folder where crops will be saved
N_CROPS       = 5                  # number of crops to sample
CROP_SIZE     = 200                # width and height of each crop (pixels)
TILE_SIZE     = 20                 # mosaic tile size (pixels)
MIN_CONTENT   = 0.85               # minimum fraction of non-black pixels in crop
BLACK_THRESH  = 15                 # pixel max RGB value considered "black"
MIN_DIST      = 80                 # minimum distance between crop centers (pixels)
RANDOM_SEED   = 42                 # set to None for different results each run
# ─────────────────────────────────────────────────────────────────────────────


def is_valid_crop(arr, x, y, size, black_thresh, min_content):
    crop = arr[y:y+size, x:x+size]
    return (crop.max(axis=2) >= black_thresh).mean() > min_content


def crops_too_close(new_x, new_y, existing, min_dist):
    for (ex, ey) in existing:
        if abs(new_x - ex) < min_dist and abs(new_y - ey) < min_dist:
            return True
    return False


def sample_crops(arr, n, size, black_thresh, min_content, min_dist, max_attempts=10000):
    h, w = arr.shape[:2]
    centers, crops = [], []
    attempts = 0
    while len(crops) < n and attempts < max_attempts:
        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)
        if (is_valid_crop(arr, x, y, size, black_thresh, min_content) and
                not crops_too_close(x, y, centers, min_dist)):
            centers.append((x, y))
            crops.append(arr[y:y+size, x:x+size].copy())
        attempts += 1
    if len(crops) < n:
        print(f"Warning: only found {len(crops)}/{n} valid crops after {attempts} attempts.")
        print("Try reducing MIN_DIST or MIN_CONTENT.")
    return crops


def mean_color_mosaic(crop_arr, tile_size, black_thresh):
    sh, sw = crop_arr.shape[:2]
    bh = sh // tile_size
    bw = sw // tile_size

    # Build list of mean colors per tile
    positions, colors = [], []
    for r in range(bh):
        for c in range(bw):
            block = crop_arr[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]
            non_black = block.max(axis=2) >= black_thresh
            if non_black.mean() > 0.5:
                mean_color = block[non_black].mean(axis=0).astype(np.uint8)
                positions.append((r, c))
                colors.append(mean_color)

    # Shuffle colors across positions
    np.random.shuffle(colors)

    result = np.zeros_like(crop_arr)
    for (r, c), color in zip(positions, colors):
        result[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = color

    return result


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    img = Image.open(INPUT_IMAGE).convert("RGB")
    arr = np.array(img)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Sampling {N_CROPS} crops from {INPUT_IMAGE} ({arr.shape[1]}x{arr.shape[0]})...")
    crops = sample_crops(arr, N_CROPS, CROP_SIZE, BLACK_THRESH, MIN_CONTENT, MIN_DIST)

    for i, crop in enumerate(crops, start=1):
        # Save original crop
        orig_path = os.path.join(OUTPUT_DIR, f"crop_{i}.png")
        Image.fromarray(crop).save(orig_path)

        # Save scrambled version
        scrambled = mean_color_mosaic(crop, TILE_SIZE, BLACK_THRESH)
        scram_path = os.path.join(OUTPUT_DIR, f"crop_{i}_scrambled.png")
        Image.fromarray(scrambled).save(scram_path)

        print(f"  Saved crop {i}: {orig_path} + {scram_path}")

    print(f"\nDone. {len(crops)*2} files saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
