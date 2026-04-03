"""
color_sampler.py
----------------
Reads materials_list.xlsx (both 'texture' and 'color' tabs) and produces
three output folders:

  1. cropped_textures/     — 5 random 200x200 crops per texture-tab trial
  2. color_trials/         — 20 target crops per color-tab trial (flat folder)
                              with filenames: <cat>_<id>_target_<tier>_<cropnum>.png
  3. color_trial_mosaics/  — mosaic-scrambled versions of all target crops
                              with filenames: mosaic_<cat>_<id>_target_<tier>_<cropnum>.png

USAGE:
    python color_sampler.py
"""

import os
import numpy as np
import openpyxl
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MATERIALS_LIST = os.path.join(SCRIPT_DIR, "materials_list.xlsx")
STUFF_DIR      = os.path.join(SCRIPT_DIR, "STUFF_enhanced_dataset_3514_images")

TEXTURE_OUT        = os.path.join(SCRIPT_DIR, "cropped_textures")
COLOR_OUT          = os.path.join(SCRIPT_DIR, "cropped_colors")       # legacy nested
MOSAIC_OUT         = os.path.join(SCRIPT_DIR, "color_mosaics")        # legacy nested
COLOR_TRIALS_OUT   = os.path.join(SCRIPT_DIR, "color_trials")
MOSAIC_TRIALS_OUT  = os.path.join(SCRIPT_DIR, "color_trial_mosaics")

CROP_SIZE     = 200        # width and height of each crop (pixels)
TILE_SIZE     = 20         # mosaic tile size for scrambled versions
MIN_CONTENT   = 0.85       # minimum fraction of non-black pixels in crop
BLACK_THRESH  = 15         # pixel max RGB value considered "black"
MAX_ATTEMPTS  = 10000      # max random attempts per image
RANDOM_SEED   = 42         # set to None for different results each run

# Manual name corrections: materials_list name -> STUFF folder name
NAME_CORRECTIONS = {
    "oil paper":  "oilpaper",
    "fluroine":   "fluorine",
    "playdough":  "play_dough",
    "tinofil":    "tinfoil",
    "cordoroy":   "corduroy",
    "beewax":     "beewax",
}

# Extra trials not in the spreadsheet: (name, image_id)
EXTRA_TEXTURE_TRIALS = [
    ("algae", 2),
]
# ─────────────────────────────────────────────────────────────────────────────


def resolve_folder_name(name):
    """Map a materials-list name to the corresponding STUFF folder name."""
    name = name.strip().lower()
    if name in NAME_CORRECTIONS:
        return NAME_CORRECTIONS[name]
    underscore = name.replace(" ", "_")
    if os.path.isdir(os.path.join(STUFF_DIR, underscore)):
        return underscore
    return name


def is_valid_crop(arr, x, y, size, black_thresh, min_content):
    crop = arr[y:y+size, x:x+size]
    return (crop.max(axis=2) >= black_thresh).mean() > min_content


def crops_too_close(new_x, new_y, existing, min_dist):
    for ex, ey in existing:
        if abs(new_x - ex) < min_dist and abs(new_y - ey) < min_dist:
            return True
    return False


def sample_crops(arr, n, size, black_thresh, min_content, min_dist,
                 max_attempts=MAX_ATTEMPTS, center_pct=1.0):
    """Sample n valid crops. Use min_dist=0 for overlapping, >=size for non-overlapping.
    center_pct: fraction of image to sample from (centered). E.g. 0.5 = middle 50%."""
    h, w = arr.shape[:2]
    if h < size or w < size:
        print(f"    Image too small ({w}x{h}), skipping.")
        return []
    # Compute sampling bounds (constrained to center region)
    margin_x = int(w * (1 - center_pct) / 2)
    margin_y = int(h * (1 - center_pct) / 2)
    x_lo = margin_x
    x_hi = max(w - size - margin_x, x_lo + 1)
    y_lo = margin_y
    y_hi = max(h - size - margin_y, y_lo + 1)

    centers, crops = [], []
    attempts = 0
    while len(crops) < n and attempts < max_attempts:
        x = np.random.randint(x_lo, x_hi)
        y = np.random.randint(y_lo, y_hi)
        if (is_valid_crop(arr, x, y, size, black_thresh, min_content)
                and not crops_too_close(x, y, centers, min_dist)):
            centers.append((x, y))
            crops.append(arr[y:y+size, x:x+size].copy())
        attempts += 1
    if len(crops) < n:
        print(f"    Warning: only found {len(crops)}/{n} valid crops after {attempts} attempts.")
    return crops


def mean_color_mosaic(crop_arr, tile_size, black_thresh):
    sh, sw = crop_arr.shape[:2]
    bh = sh // tile_size
    bw = sw // tile_size
    positions, colors = [], []
    for r in range(bh):
        for c in range(bw):
            block = crop_arr[r*tile_size:(r+1)*tile_size,
                             c*tile_size:(c+1)*tile_size]
            non_black = block.max(axis=2) >= black_thresh
            if non_black.mean() > 0.5:
                mean_color = block[non_black].mean(axis=0).astype(np.uint8)
                positions.append((r, c))
                colors.append(mean_color)
    np.random.shuffle(colors)
    result = np.zeros_like(crop_arr)
    for (r, c), color in zip(positions, colors):
        result[r*tile_size:(r+1)*tile_size,
               c*tile_size:(c+1)*tile_size] = color
    return result


def _rgb_to_lab(rgb_arr):
    """Convert RGB array (uint8) to CIELAB via XYZ. Returns float64 array.
    Uses skimage if available, otherwise a manual conversion."""
    try:
        from skimage.color import rgb2lab
        return rgb2lab(rgb_arr)
    except ImportError:
        pass
    # Manual conversion: RGB -> linear RGB -> XYZ (D65) -> CIELAB
    arr = rgb_arr.astype(np.float64) / 255.0
    # sRGB gamma decode
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] = arr[~mask] / 12.92
    # RGB to XYZ (sRGB D65 matrix)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
    shape = arr.shape
    xyz = arr.reshape(-1, 3) @ M.T
    xyz = xyz.reshape(shape)
    # D65 reference white
    xyz[..., 0] /= 0.95047
    xyz[..., 1] /= 1.00000
    xyz[..., 2] /= 1.08883
    # XYZ to Lab
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    mask = xyz > eps
    f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0    # L
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])  # a
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])  # b
    return lab


def compute_color_profile(image_arr):
    """Compute a normalized 3D LAB histogram (8 bins per channel = 512 bins).
    Returns a flattened float64 histogram vector."""
    lab = _rgb_to_lab(image_arr)
    # Bin ranges: L [0,100], a [-128,127], b [-128,127]
    L = np.clip(lab[..., 0], 0, 100)
    a = np.clip(lab[..., 1], -128, 127)
    b = np.clip(lab[..., 2], -128, 127)
    # Digitize into 8 bins each
    L_bins = np.floor(L / (100.0 / 8)).astype(int).clip(0, 7)
    a_bins = np.floor((a + 128) / (255.0 / 8)).astype(int).clip(0, 7)
    b_bins = np.floor((b + 128) / (255.0 / 8)).astype(int).clip(0, 7)
    flat_idx = L_bins.ravel() * 64 + a_bins.ravel() * 8 + b_bins.ravel()
    hist = np.bincount(flat_idx, minlength=512).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def color_distance(profile1, profile2):
    """Chi-squared distance between two histogram profiles. Lower = more similar."""
    s = profile1 + profile2
    nonzero = s > 0
    diff = (profile1[nonzero] - profile2[nonzero]) ** 2
    return 0.5 * np.sum(diff / s[nonzero])


def read_trials(path, sheet_name):
    """Return list of (name, image_id) tuples from a specific sheet."""
    wb = openpyxl.load_workbook(path)
    ws = wb[sheet_name]
    trials = []
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        name = row[3].value  # column D: Name
        same3 = row[4].value  # column E: same 3
        diff1 = row[5].value  # column F: different 1
        if name is None:
            continue
        name = str(name).strip()
        for img_id in [same3, diff1]:
            if img_id is not None:
                try:
                    trials.append((name, int(img_id)))
                except (ValueError, TypeError):
                    pass  # skip non-numeric entries
    return trials


def load_image(name, img_id):
    """Resolve name and load image, returning (arr, folder) or (None, None)."""
    folder = resolve_folder_name(name)
    img_path = os.path.join(STUFF_DIR, folder, f"{img_id:04d}.jpg")
    if not os.path.isfile(img_path):
        print(f"  SKIP: {name}/{img_id:04d}.jpg not found")
        return None, folder
    img = Image.open(img_path).convert("RGB")
    return np.array(img), folder


def process_textures(trials):
    """Folder 1: cropped_textures/<category>/ — 10 crops per texture trial."""
    print("=" * 60)
    print("CROPPED TEXTURES")
    print("=" * 60)
    os.makedirs(TEXTURE_OUT, exist_ok=True)
    total = 0
    for name, img_id in trials:
        arr, folder = load_image(name, img_id)
        if arr is None:
            continue
        trial_dir = os.path.join(TEXTURE_OUT, folder, f"{img_id:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        print(f"  {name} / {img_id:04d} ({arr.shape[1]}x{arr.shape[0]})")

        crops = sample_crops(arr, 10, CROP_SIZE, BLACK_THRESH, MIN_CONTENT,
                             min_dist=80)
        for i, crop in enumerate(crops, start=1):
            Image.fromarray(crop).save(
                os.path.join(trial_dir, f"crop_{i}.png"))
            total += 1

    print(f"  -> {total} files saved to {TEXTURE_OUT}/\n")


def process_colors(trials):
    """Flat folders: color_trials/ + color_trial_mosaics/.
    20 target crops per trial across 4 overlap tiers (5 crops each).
    Filenames: <cat>_<id>_target_<tier>_<cropnum>.png"""
    print("=" * 60)
    print("COLOR TRIALS (TARGET CROPS + MOSAICS)")
    print("=" * 60)
    os.makedirs(COLOR_TRIALS_OUT, exist_ok=True)
    os.makedirs(MOSAIC_TRIALS_OUT, exist_ok=True)
    total_crops = 0
    total_mosaics = 0

    # 4 overlap tiers: (label, min_dist)
    tiers = [
        ("extreme_overlap", 0),
        ("medium_overlap",  CROP_SIZE // 3),       # ~67 px
        ("small_overlap",   2 * CROP_SIZE // 3),   # ~133 px
        ("nonoverlap",      CROP_SIZE),             # 200 px
    ]

    for name, img_id in trials:
        arr, folder = load_image(name, img_id)
        if arr is None:
            continue

        print(f"  {name} / {img_id:04d} ({arr.shape[1]}x{arr.shape[0]})")

        color_min_content = 0.95  # stricter to avoid dark patches
        color_black_thresh = 50   # higher threshold to reject dark regions

        for tier_label, min_dist in tiers:
            crops = sample_crops(arr, 5, CROP_SIZE, color_black_thresh,
                                 color_min_content, min_dist=min_dist,
                                 center_pct=0.7)
            for i, crop in enumerate(crops, start=1):
                fname = f"{folder}_{img_id:04d}_target_{tier_label}_{i}.png"
                Image.fromarray(crop).save(
                    os.path.join(COLOR_TRIALS_OUT, fname))
                total_crops += 1
                mosaic = mean_color_mosaic(crop, TILE_SIZE, BLACK_THRESH)
                Image.fromarray(mosaic).save(
                    os.path.join(MOSAIC_TRIALS_OUT, f"mosaic_{folder}_{img_id:04d}_target_{tier_label}_{i}.png"))
                total_mosaics += 1

    print(f"  -> {total_crops} files saved to {COLOR_TRIALS_OUT}/")
    print(f"  -> {total_mosaics} files saved to {MOSAIC_TRIALS_OUT}/\n")


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    texture_trials = read_trials(MATERIALS_LIST, "texture") + EXTRA_TEXTURE_TRIALS
    color_trials = read_trials(MATERIALS_LIST, "color")
    print(f"Texture trials: {len(texture_trials)}")
    print(f"Color trials:   {len(color_trials)}\n")

    process_textures(texture_trials)
    process_colors(color_trials)

    print("All done.")


if __name__ == "__main__":
    main()
