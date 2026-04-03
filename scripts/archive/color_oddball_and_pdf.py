"""
color_oddball_and_pdf.py
------------------------
1. Generates same-category oddball (distractor) crops + mosaics for each of
   the 34 color categories, using data-driven LAB color histogram similarity
   to select oddballs at 4 difficulty levels (hard, medhard, medeasy, easy).
   Oddballs are different exemplars of the SAME material category.
2. Creates a PDF (color_trial_examples.pdf) showing example trial layouts
   from each difficulty tier.

All output goes to flat folders:
  color_trials/          — <cat>_<id>_oddball_<difficulty>_<tier>_<cropnum>.png
  color_trial_mosaics/   — mosaic_<cat>_<id>_oddball_<difficulty>_<tier>_<cropnum>.png

USAGE:
    python color_oddball_and_pdf.py
"""

import os
import sys
import glob as globmod
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import crop/mosaic/color functions from color_sampler
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from color_sampler import (
    sample_crops, mean_color_mosaic, load_image, read_trials,
    resolve_folder_name, compute_color_profile, color_distance,
    STUFF_DIR, MATERIALS_LIST,
    CROP_SIZE, TILE_SIZE, COLOR_TRIALS_OUT, MOSAIC_TRIALS_OUT,
)

PDF_OUT = os.path.join(SCRIPT_DIR, "color_trial_examples.pdf")

# Color pipeline parameters (same as process_colors in color_sampler.py)
COLOR_BLACK_THRESH = 50
COLOR_MIN_CONTENT  = 0.95
CENTER_PCT         = 0.7

RANDOM_SEED = 99  # different seed from main pipeline to avoid overlap

# 4 overlap tiers for crop sampling (must match color_sampler.py)
CROP_TIERS = [
    ("extreme_overlap", 0),
    ("medium_overlap",  CROP_SIZE // 3),
    ("small_overlap",   2 * CROP_SIZE // 3),
    ("nonoverlap",      CROP_SIZE),
]

# 4 difficulty labels (sorted from hardest to easiest)
DIFFICULTY_LABELS = ["hard", "medhard", "medeasy", "easy"]

MAX_CANDIDATES_PER_CAT = 10  # max unused images to profile per category


def get_used_ids():
    """Return dict of {folder_name: set of used image IDs} from the color tab."""
    trials = read_trials(MATERIALS_LIST, "color")
    used = {}
    for name, img_id in trials:
        folder = resolve_folder_name(name)
        used.setdefault(folder, set()).add(img_id)
    return used


def get_all_color_categories():
    """Return sorted list of unique color category folder names."""
    trials = read_trials(MATERIALS_LIST, "color")
    cats = set()
    for name, _img_id in trials:
        cats.add(resolve_folder_name(name))
    return sorted(cats)


def build_candidate_profiles(used_ids):
    """For each color category, load up to MAX_CANDIDATES_PER_CAT unused images
    from the SAME category and compute their LAB color profiles.
    Returns: {category: [(img_id, profile), ...]}"""
    print("  Building candidate color profiles (same-category)...")
    categories = get_all_color_categories()
    profiles = {}

    for cat in categories:
        folder_path = os.path.join(STUFF_DIR, cat)
        if not os.path.isdir(folder_path):
            print(f"    WARNING: STUFF folder not found: {cat}")
            continue

        used = used_ids.get(cat, set())
        candidates = []

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".jpg"):
                continue
            img_id = int(fname.replace(".jpg", ""))
            if img_id in used:
                continue
            # Load and profile
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            profile = compute_color_profile(arr)
            candidates.append((img_id, profile))
            if len(candidates) >= MAX_CANDIDATES_PER_CAT:
                break

        profiles[cat] = candidates
        print(f"    {cat}: {len(candidates)} candidates profiled")

    return profiles


def select_oddballs(target_cat, target_profile, candidate_profiles, n_tiers=4):
    """Select one oddball image per difficulty tier from the SAME category.

    Gets candidates from candidate_profiles[target_cat], computes color distance
    to the target, sorts by distance, divides into quartiles, and picks one
    from each quartile.

    Returns: list of (difficulty_label, img_id, distance) tuples,
    ordered from hard (most similar) to easy (most different).
    """
    candidates = candidate_profiles.get(target_cat, [])
    if not candidates:
        return []

    # Compute distances and sort
    scored = []
    for img_id, profile in candidates:
        dist = color_distance(target_profile, profile)
        scored.append((dist, img_id))

    scored.sort(key=lambda x: x[0])

    n = len(scored)
    results = []
    for tier_idx in range(min(n_tiers, n)):
        q_start = int(tier_idx * n / n_tiers)
        q_end = int((tier_idx + 1) * n / n_tiers)
        if q_start >= q_end:
            continue
        mid = (q_start + q_end) // 2
        dist, img_id = scored[mid]
        label = DIFFICULTY_LABELS[tier_idx]
        results.append((label, img_id, dist))

    return results


def generate_oddballs():
    """Generate same-category oddball crops + mosaics for all categories
    at 4 difficulty levels. Saves to flat color_trials/ and color_trial_mosaics/."""
    print("=" * 60)
    print("GENERATING SAME-CATEGORY ODDBALL CROPS + MOSAICS")
    print("=" * 60)

    os.makedirs(COLOR_TRIALS_OUT, exist_ok=True)
    os.makedirs(MOSAIC_TRIALS_OUT, exist_ok=True)

    used_ids = get_used_ids()
    candidate_profiles = build_candidate_profiles(used_ids)

    # Track IDs picked as oddballs per target image so we don't reuse
    picked_ids = {}

    total_crops = 0
    total_mosaics = 0

    # Iterate over ALL trials (every target image gets its own oddballs)
    trials = read_trials(MATERIALS_LIST, "color")

    for name, tid in trials:
        target_cat = resolve_folder_name(name)

        # Load target image and compute its profile
        target_arr, _ = load_image(name, tid)
        if target_arr is None:
            continue
        target_profile = compute_color_profile(target_arr)

        # Select 4 same-category oddballs at different difficulty levels
        oddballs = select_oddballs(target_cat, target_profile, candidate_profiles)
        if not oddballs:
            print(f"  SKIP {target_cat}/{tid:04d}: no same-category oddball candidates found")
            continue

        print(f"\n  Target: {target_cat}/{tid:04d}")
        for diff_label, odd_id, dist in oddballs:
            # Check if this image was already picked for this target
            pick_key = (target_cat, tid)
            if odd_id in picked_ids.get(pick_key, set()):
                print(f"    {diff_label}: {target_cat}/{odd_id:04d} already used, skipping")
                continue

            # Load oddball image (same category)
            odd_arr, _ = load_image(target_cat, odd_id)
            if odd_arr is None:
                continue

            # Quick check that we can get crops
            test_crops = sample_crops(odd_arr, 5, CROP_SIZE, COLOR_BLACK_THRESH,
                                      COLOR_MIN_CONTENT, min_dist=0,
                                      center_pct=CENTER_PCT)
            if len(test_crops) == 0:
                print(f"    {diff_label}: {target_cat}/{odd_id:04d} — no valid crops, skipping")
                continue

            picked_ids.setdefault(pick_key, set()).add(odd_id)
            print(f"    {diff_label}: {target_cat}/{odd_id:04d} (dist={dist:.4f})")

            # Sample 20 crops across 4 overlap tiers
            for tier_label, min_dist in CROP_TIERS:
                crops = sample_crops(odd_arr, 5, CROP_SIZE, COLOR_BLACK_THRESH,
                                     COLOR_MIN_CONTENT, min_dist=min_dist,
                                     center_pct=CENTER_PCT)
                for i, crop in enumerate(crops, start=1):
                    fname = f"{target_cat}_{odd_id:04d}_oddball_{diff_label}_{tier_label}_{i}.png"
                    Image.fromarray(crop).save(
                        os.path.join(COLOR_TRIALS_OUT, fname))
                    total_crops += 1
                    mosaic = mean_color_mosaic(crop, TILE_SIZE, COLOR_BLACK_THRESH)
                    Image.fromarray(mosaic).save(
                        os.path.join(MOSAIC_TRIALS_OUT, f"mosaic_{target_cat}_{odd_id:04d}_oddball_{diff_label}_{tier_label}_{i}.png"))
                    total_mosaics += 1

    print(f"\n  -> {total_crops} oddball crops saved to {COLOR_TRIALS_OUT}/")
    print(f"  -> {total_mosaics} oddball mosaics saved to {MOSAIC_TRIALS_OUT}/\n")


# ── PDF GENERATION ───────────────────────────────────────────────────────────

# Page dimensions (portrait letter at 150 DPI)
PAGE_W, PAGE_H = 1275, 1650
MARGIN = 60

# Try to get fonts
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception:
    font_title = ImageFont.load_default()
    font_label = font_title
    font_small = font_title


def fit_image(img, max_w, max_h):
    """Resize image to fit within max_w x max_h, preserving aspect ratio."""
    ratio = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def draw_centered_text(draw, text, y, font, fill="black", page_w=PAGE_W):
    """Draw text centered horizontally at vertical position y."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((page_w - tw) // 2, y), text, fill=fill, font=font)


def make_trial_page(title, study_path, choices, labels):
    """
    Create one trial example page.
    - title: page title string
    - study_path: path to the study image
    - choices: list of image paths
    - labels: list of label strings
    """
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)

    # Title
    y_cursor = MARGIN
    draw_centered_text(draw, title, y_cursor, font_title)
    y_cursor += 50

    # "Study image" label
    draw_centered_text(draw, "Study Image", y_cursor, font_label, fill="gray")
    y_cursor += 30

    # Study image (centered, large)
    study_display_size = 400
    if study_path and os.path.isfile(study_path):
        study_img = Image.open(study_path)
        study_img = fit_image(study_img, study_display_size, study_display_size)
        x = (PAGE_W - study_img.width) // 2
        page.paste(study_img, (x, y_cursor))
        draw.rectangle(
            [x - 1, y_cursor - 1, x + study_img.width, y_cursor + study_img.height],
            outline="#cccccc"
        )
        y_cursor += study_img.height + 20
    else:
        draw_centered_text(draw, "[study image not found]", y_cursor + 50, font_small, fill="red")
        y_cursor += study_display_size + 20

    # Separator
    draw.line([(MARGIN, y_cursor), (PAGE_W - MARGIN, y_cursor)], fill="#dddddd", width=2)
    y_cursor += 20

    # "Choose the match:" label
    draw_centered_text(draw, "Choose the match:", y_cursor, font_label, fill="gray")
    y_cursor += 35

    # Answer choices, evenly spaced
    choice_display_size = 300
    n_choices = len(choices)
    total_width = n_choices * choice_display_size + (n_choices - 1) * 40
    x_start = (PAGE_W - total_width) // 2

    for i, (img_path, label) in enumerate(zip(choices, labels)):
        x = x_start + i * (choice_display_size + 40)

        if img_path and os.path.isfile(img_path):
            choice_img = Image.open(img_path)
            choice_img = fit_image(choice_img, choice_display_size, choice_display_size)
            cx = x + (choice_display_size - choice_img.width) // 2
            cy = y_cursor + (choice_display_size - choice_img.height) // 2
            page.paste(choice_img, (cx, cy))
            draw.rectangle(
                [cx - 1, cy - 1, cx + choice_img.width, cy + choice_img.height],
                outline="#cccccc"
            )
        else:
            draw_centered_text(draw, "[missing]",
                               y_cursor + choice_display_size // 2,
                               font_small, fill="red",
                               page_w=x + choice_display_size)

        # Label below choice
        label_y = y_cursor + choice_display_size + 8
        bbox = draw.textbbox((0, 0), label, font=font_small)
        lw = bbox[2] - bbox[0]
        lx = x + (choice_display_size - lw) // 2
        color = "#228B22" if "correct" in label.lower() else "#333333"
        draw.text((lx, label_y), label, fill=color, font=font_small)

    return page


def find_flat_file(directory, pattern):
    """Find first file matching a glob pattern in a flat directory."""
    matches = sorted(globmod.glob(os.path.join(directory, pattern)))
    return matches[0] if matches else None


def generate_pdf():
    """Create a PDF with one example page per difficulty tier (4 mosaic pages),
    showing target mosaic vs same-category oddball mosaic."""
    print("=" * 60)
    print("GENERATING TRIAL EXAMPLES PDF")
    print("=" * 60)

    pages = []
    categories = get_all_color_categories()

    # Build {category: first target image id} so we can find target files
    trials = read_trials(MATERIALS_LIST, "color")
    cat_first_id = {}
    for name, img_id in trials:
        folder = resolve_folder_name(name)
        if folder not in cat_first_id:
            cat_first_id[folder] = img_id

    tier_display = {
        "hard":    "HARD (most similar — same category)",
        "medhard": "MEDIUM-HARD (same category)",
        "medeasy": "MEDIUM-EASY (same category)",
        "easy":    "EASY (most different — same category)",
    }

    page_num = 0

    # One mosaic page per difficulty tier
    for diff_label, display_label in tier_display.items():
        found = False
        for cat in categories:
            tid = cat_first_id.get(cat)
            if tid is None:
                continue

            # Find target mosaic
            target_study = find_flat_file(
                MOSAIC_TRIALS_OUT,
                f"mosaic_{cat}_{tid:04d}_target_extreme_overlap_1.png")
            target_match = find_flat_file(
                MOSAIC_TRIALS_OUT,
                f"mosaic_{cat}_{tid:04d}_target_extreme_overlap_2.png")

            # Find oddball mosaic for this difficulty
            oddball_mosaic = find_flat_file(
                MOSAIC_TRIALS_OUT,
                f"mosaic_{cat}_*_oddball_{diff_label}_extreme_overlap_1.png")

            if target_study and target_match and oddball_mosaic:
                # Extract oddball image ID from filename
                odd_fname = os.path.basename(oddball_mosaic)
                page_num += 1
                pages.append(make_trial_page(
                    f"{display_label} — Mosaics — {cat}",
                    target_study,
                    [target_match, oddball_mosaic],
                    ["Target (correct)", f"Oddball ({diff_label})"]
                ))
                print(f"  Page {page_num}: {display_label} — {cat}")
                found = True
                break
        if not found:
            print(f"  WARNING: No example found for {display_label}")

    # One crop page per difficulty tier
    for diff_label, display_label in tier_display.items():
        found = False
        for cat in categories:
            tid = cat_first_id.get(cat)
            if tid is None:
                continue

            target_study = find_flat_file(
                COLOR_TRIALS_OUT,
                f"{cat}_{tid:04d}_target_extreme_overlap_1.png")
            target_match = find_flat_file(
                COLOR_TRIALS_OUT,
                f"{cat}_{tid:04d}_target_extreme_overlap_2.png")
            oddball_crop = find_flat_file(
                COLOR_TRIALS_OUT,
                f"{cat}_*_oddball_{diff_label}_extreme_overlap_1.png")

            if target_study and target_match and oddball_crop:
                page_num += 1
                pages.append(make_trial_page(
                    f"{display_label} — Crops — {cat}",
                    target_study,
                    [target_match, oddball_crop],
                    ["Target (correct)", f"Oddball ({diff_label})"]
                ))
                print(f"  Page {page_num}: {display_label} crops — {cat}")
                found = True
                break
        if not found:
            print(f"  WARNING: No example found for {display_label} crops")

    # Save PDF
    if pages:
        pages[0].save(PDF_OUT, save_all=True, append_images=pages[1:], resolution=150)
        print(f"\n  -> Saved {len(pages)}-page PDF to: {PDF_OUT}\n")
    else:
        print("\n  WARNING: No pages generated — check that crops/mosaics exist.\n")


def main():
    np.random.seed(RANDOM_SEED)
    generate_oddballs()
    generate_pdf()
    print("All done.")


if __name__ == "__main__":
    main()
