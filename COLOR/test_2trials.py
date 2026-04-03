"""Quick test: run the color pipeline on 2 categories (all their trials)."""
import os, sys, shutil
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from color_sampler import (
    read_trials, resolve_folder_name, process_colors, load_image,
    sample_crops, mean_color_mosaic, compute_color_profile, color_distance,
    MATERIALS_LIST, STUFF_DIR, COLOR_TRIALS_OUT, MOSAIC_TRIALS_OUT,
    CROP_SIZE, TILE_SIZE,
)
from color_oddball_and_pdf import (
    select_oddballs,
    COLOR_BLACK_THRESH, COLOR_MIN_CONTENT, CENTER_PCT,
    CROP_TIERS, MAX_CANDIDATES_PER_CAT,
)

# Wipe previous test output
for d in [COLOR_TRIALS_OUT, MOSAIC_TRIALS_OUT]:
    if os.path.isdir(d):
        shutil.rmtree(d)
        print(f"  Cleared {d}")

np.random.seed(42)

# ── Pick 2 categories and grab ALL their trials ─────────────────────────
all_trials = read_trials(MATERIALS_LIST, "color")

seen_cats = set()
test_cats = []
for name, img_id in all_trials:
    cat = resolve_folder_name(name)
    if cat not in seen_cats:
        test_cats.append(cat)
        seen_cats.add(cat)
    if len(test_cats) == 2:
        break

# Collect every trial belonging to those 2 categories
subset = [(name, img_id) for name, img_id in all_trials
          if resolve_folder_name(name) in seen_cats]

print(f"\n=== TEST: {len(subset)} trials across categories {test_cats} ===")
for name, img_id in subset:
    print(f"  {resolve_folder_name(name)} / {img_id}")

# ── Step 1: target crops ─────────────────────────────────────────────────
process_colors(subset)

# ── Step 2: profile candidates for those 2 categories ───────────────────
np.random.seed(99)

used_ids = {}
for name, img_id in subset:
    cat = resolve_folder_name(name)
    used_ids.setdefault(cat, set()).add(img_id)

print("\n  Building candidate profiles...")
candidate_profiles = {}
for cat in test_cats:
    folder_path = os.path.join(STUFF_DIR, cat)
    if not os.path.isdir(folder_path):
        continue
    used = used_ids.get(cat, set())
    candidates = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".jpg"):
            continue
        iid = int(fname.replace(".jpg", ""))
        if iid in used:
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        arr = np.array(img)
        profile = compute_color_profile(arr)
        candidates.append((iid, profile))
        if len(candidates) >= MAX_CANDIDATES_PER_CAT:
            break
    candidate_profiles[cat] = candidates
    print(f"    {cat}: {len(candidates)} candidates")

# ── Step 3: generate oddballs for every trial ────────────────────────────
total_crops = 0
total_mosaics = 0

for name, img_id in subset:
    cat = resolve_folder_name(name)
    target_arr, _ = load_image(name, img_id)
    if target_arr is None:
        continue
    target_profile = compute_color_profile(target_arr)
    oddballs = select_oddballs(cat, target_profile, candidate_profiles)
    if not oddballs:
        print(f"  SKIP {cat}/{img_id}: no candidates")
        continue

    print(f"\n  Target: {cat}/{img_id:04d}")
    for diff_label, odd_id, dist in oddballs:
        odd_arr, _ = load_image(cat, odd_id)
        if odd_arr is None:
            continue
        print(f"    {diff_label}: {cat}/{odd_id:04d} (dist={dist:.4f})")
        for tier_label, min_dist in CROP_TIERS:
            crops = sample_crops(odd_arr, 5, CROP_SIZE, COLOR_BLACK_THRESH,
                                 COLOR_MIN_CONTENT, min_dist=min_dist,
                                 center_pct=CENTER_PCT)
            for i, crop in enumerate(crops, start=1):
                fname = f"{cat}_{odd_id:04d}_oddball_{diff_label}_{tier_label}_{i}.png"
                Image.fromarray(crop).save(os.path.join(COLOR_TRIALS_OUT, fname))
                total_crops += 1
                mosaic = mean_color_mosaic(crop, TILE_SIZE, COLOR_BLACK_THRESH)
                Image.fromarray(mosaic).save(
                    os.path.join(MOSAIC_TRIALS_OUT,
                                 f"mosaic_{cat}_{odd_id:04d}_oddball_{diff_label}_{tier_label}_{i}.png"))
                total_mosaics += 1

print(f"\n  -> {total_crops} oddball crops, {total_mosaics} oddball mosaics")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n=== FILES IN color_trials/ ===")
if os.path.isdir(COLOR_TRIALS_OUT):
    for f in sorted(os.listdir(COLOR_TRIALS_OUT)):
        print(f"  {f}")
    print(f"  TOTAL: {len(os.listdir(COLOR_TRIALS_OUT))}")

print("\nDone.")
