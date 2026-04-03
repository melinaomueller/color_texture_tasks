"""
generate_texture_review.py
--------------------------
Generate a visual review page for texture oddball trials.

For each trial:
  - Samples 3 distractor crops from the same3 image (identical to generate_texture_crops.py)
  - Samples 4 oddball crops from the diff1 image
  - Saves source image thumbnails for reference
  - Creates 4 versions per trial, each with a different oddball in a different position
  - Generates review_index.html with all trials + free-response feedback boxes

USAGE:
    python3 generate_texture_review.py
"""

import csv
import html
import os

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
THUMB_HEIGHT = 300   # thumbnail height for source images
# ──────────────────────────────────────────────────────────────────────────────


def load_stuff_image(category, image_id):
    path = os.path.join(STUFF_DIR, category, f"{int(image_id):04d}.jpg")
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found")
        return None
    return Image.open(path).convert("RGB")


def is_valid_crop(arr, x, y, size):
    crop = arr[y:y + size, x:x + size]
    return (crop.max(axis=2) >= BLACK_THRESH).mean() > MIN_CONTENT


def crops_too_close(new_x, new_y, existing, min_dist):
    for ex, ey in existing:
        if abs(new_x - ex) < min_dist and abs(new_y - ey) < min_dist:
            return True
    return False


def sample_crops(arr, n, rng, min_dist=200):
    h, w = arr.shape[:2]
    if h < CROP_SIZE or w < CROP_SIZE:
        print(f"    Image too small ({w}x{h}), skipping.")
        return []
    x_hi = w - CROP_SIZE
    y_hi = h - CROP_SIZE
    centers, results = [], []
    attempts = 0
    while len(results) < n and attempts < MAX_ATTEMPTS:
        x = rng.randint(0, x_hi + 1)
        y = rng.randint(0, y_hi + 1)
        if (is_valid_crop(arr, x, y, CROP_SIZE)
                and not crops_too_close(x, y, centers, min_dist)):
            centers.append((x, y))
            results.append(arr[y:y + CROP_SIZE, x:x + CROP_SIZE].copy())
        attempts += 1
    if len(results) < n:
        print(f"    Warning: only got {len(results)}/{n} crops after {attempts} attempts.")
    return results


def to_grayscale(rgb_arr):
    return np.dot(rgb_arr[..., :3].astype(np.float64),
                  [0.2989, 0.5870, 0.1140]).round().clip(0, 255).astype(np.uint8)


def save_thumbnail(pil_img, path, max_h=THUMB_HEIGHT):
    w, h = pil_img.size
    ratio = max_h / h
    new_w = int(w * ratio)
    pil_img.resize((new_w, max_h), Image.LANCZOS).save(path)


def process_trial(row, output_dir):
    """Generate review crops for one trial. Returns trial info dict or None."""
    trial_id = int(row["trial_id"])
    category = row["category"].strip()
    same3_image = row["same3_image"].strip()
    diff1_category = row["diff1_category"].strip()
    diff1_image = row["diff1_image"].strip()
    min_dist = int(row.get("min_dist") or CROP_SIZE)
    notes = row.get("notes", "").strip()

    # Same seed as generate_texture_crops.py — distractors will be identical
    rng = np.random.RandomState(RANDOM_SEED + trial_id)

    print(f"  Trial {trial_id:02d}: {category}/{same3_image} vs "
          f"{diff1_category}/{diff1_image} ... ", end="")

    same3_pil = load_stuff_image(category, same3_image)
    diff1_pil = load_stuff_image(diff1_category, diff1_image)
    if same3_pil is None or diff1_pil is None:
        print("SKIP")
        return None

    same3_arr = np.array(same3_pil)
    diff1_arr = np.array(diff1_pil)

    # Sample 3 distractors (same RNG state → same crops as original script)
    dist_crops = sample_crops(same3_arr, 3, rng, min_dist=min_dist)
    if len(dist_crops) < 3:
        print(f"SKIP (only {len(dist_crops)}/3 distractors)")
        return None

    # Sample 4 oddball crops from diff1 image
    # The first oddball uses the same RNG state as the original script,
    # so oddball_1 matches the original oddball.png
    odd_crops = sample_crops(diff1_arr, 4, rng, min_dist=100)
    if len(odd_crops) < 4:
        # Fall back: allow more overlap
        rng2 = np.random.RandomState(RANDOM_SEED + trial_id + 1000)
        odd_crops = sample_crops(diff1_arr, 4, rng2, min_dist=0)
    if len(odd_crops) < 4:
        print(f"SKIP (only {len(odd_crops)}/4 oddballs)")
        return None

    # Convert to grayscale
    dist_grays = [to_grayscale(c) for c in dist_crops]
    odd_grays = [to_grayscale(c) for c in odd_crops]

    # Save to trial folder
    trial_folder = os.path.join(output_dir, f"trial_{trial_id:02d}_{category}")
    os.makedirs(trial_folder, exist_ok=True)

    for i, g in enumerate(dist_grays, 1):
        Image.fromarray(g, mode="L").save(
            os.path.join(trial_folder, f"distractor_{i}.png"))

    for i, g in enumerate(odd_grays, 1):
        Image.fromarray(g, mode="L").save(
            os.path.join(trial_folder, f"oddball_{i}.png"))

    # Save source thumbnails
    save_thumbnail(same3_pil, os.path.join(trial_folder, "source_same3.jpg"))
    save_thumbnail(diff1_pil, os.path.join(trial_folder, "source_diff1.jpg"))

    print("OK")
    return {
        "trial_id": trial_id,
        "category": category,
        "same3_image": same3_image,
        "diff1_category": diff1_category,
        "diff1_image": diff1_image,
        "notes": notes,
        "folder": f"trial_{trial_id:02d}_{category}",
    }


def generate_html(trials, output_path):
    """Generate the review HTML page."""

    trial_blocks = []
    for t in trials:
        tid = t["trial_id"]
        folder = t["folder"]
        cat = html.escape(t["category"])
        s3 = html.escape(t["same3_image"])
        d1cat = html.escape(t["diff1_category"])
        d1 = html.escape(t["diff1_image"])
        notes = html.escape(t["notes"])
        notes_html = f'<span class="notes">({notes})</span>' if notes else ""

        # Build 4 versions: oddball rotates through positions 1-4
        versions_html = ""
        for v in range(4):
            oddball_pos = v  # 0-indexed position of oddball in this version
            cells = []
            dist_idx = 0
            for pos in range(4):
                if pos == oddball_pos:
                    img_path = f"{folder}/oddball_{v + 1}.png"
                    cells.append(
                        f'<div class="crop-cell oddball">'
                        f'<img src="{img_path}" alt="oddball {v+1}">'
                        f'<div class="label">Oddball {v+1}</div></div>')
                else:
                    dist_idx += 1
                    img_path = f"{folder}/distractor_{dist_idx}.png"
                    cells.append(
                        f'<div class="crop-cell distractor">'
                        f'<img src="{img_path}" alt="distractor {dist_idx}">'
                        f'<div class="label">D{dist_idx}</div></div>')

            versions_html += f"""
        <div class="version-row">
          <div class="version-label">Version {v + 1}</div>
          <div class="crop-grid">
            {"".join(cells)}
          </div>
        </div>"""

        block = f"""
    <div class="trial-block" id="trial-{tid}">
      <h2>Trial {tid}: {cat} {notes_html}</h2>
      <div class="source-info">
        Distractors: <strong>{cat}/{s3}</strong> &nbsp;|&nbsp;
        Oddball: <strong>{d1cat}/{d1}</strong>
      </div>
      <div class="source-images">
        <div class="source-img">
          <img src="{folder}/source_same3.jpg" alt="same3 source">
          <div class="label">Same3: {cat}/{s3}</div>
        </div>
        <div class="source-img">
          <img src="{folder}/source_diff1.jpg" alt="diff1 source">
          <div class="label">Diff1: {d1cat}/{d1}</div>
        </div>
      </div>
      {versions_html}
      <div class="feedback-section">
        <label for="feedback-{tid}">Feedback for Trial {tid}:</label>
        <textarea id="feedback-{tid}" rows="3"
          placeholder="e.g. oddball 2 is best, too similar, crop quality..."></textarea>
      </div>
    </div>"""
        trial_blocks.append(block)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Texture Trials Review</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f5;
      color: #333;
      padding: 20px 40px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      text-align: center;
      margin-bottom: 10px;
      font-size: 24px;
    }}
    .page-info {{
      text-align: center;
      color: #666;
      margin-bottom: 8px;
      font-size: 14px;
    }}
    .nav-bar {{
      text-align: center;
      margin-bottom: 30px;
      padding: 10px;
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .nav-bar a {{
      margin: 0 6px;
      text-decoration: none;
      color: #1565c0;
      font-size: 14px;
    }}
    .nav-bar a:hover {{ text-decoration: underline; }}
    .trial-block {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      padding: 24px;
      margin-bottom: 30px;
    }}
    .trial-block h2 {{
      font-size: 18px;
      margin-bottom: 6px;
      border-bottom: 2px solid #e0e0e0;
      padding-bottom: 6px;
    }}
    .notes {{
      color: #e65100;
      font-weight: normal;
      font-size: 14px;
    }}
    .source-info {{
      font-size: 13px;
      color: #666;
      margin-bottom: 12px;
    }}
    .source-images {{
      display: flex;
      gap: 20px;
      margin-bottom: 18px;
      justify-content: center;
    }}
    .source-img {{
      text-align: center;
    }}
    .source-img img {{
      max-height: 200px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }}
    .source-img .label {{
      font-size: 12px;
      color: #666;
      margin-top: 4px;
    }}
    .version-row {{
      display: flex;
      align-items: center;
      margin-bottom: 10px;
      gap: 12px;
    }}
    .version-label {{
      font-size: 13px;
      font-weight: 600;
      color: #555;
      min-width: 80px;
      text-align: right;
    }}
    .crop-grid {{
      display: flex;
      gap: 10px;
    }}
    .crop-cell {{
      text-align: center;
    }}
    .crop-cell img {{
      width: 200px;
      height: 200px;
      display: block;
      border: 3px solid transparent;
      border-radius: 4px;
    }}
    .crop-cell.oddball img {{
      border-color: #e53935;
    }}
    .crop-cell .label {{
      font-size: 11px;
      color: #888;
      margin-top: 2px;
    }}
    .crop-cell.oddball .label {{
      color: #e53935;
      font-weight: 600;
    }}
    .feedback-section {{
      margin-top: 16px;
    }}
    .feedback-section label {{
      display: block;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #444;
    }}
    .feedback-section textarea {{
      width: 100%;
      font-family: inherit;
      font-size: 14px;
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 6px;
      resize: vertical;
    }}
    .feedback-section textarea:focus {{
      outline: none;
      border-color: #1565c0;
      box-shadow: 0 0 0 2px rgba(21,101,192,0.15);
    }}
    .export-bar {{
      text-align: center;
      margin: 30px 0;
    }}
    .export-bar button {{
      padding: 10px 28px;
      font-size: 15px;
      font-weight: 600;
      background: #1565c0;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }}
    .export-bar button:hover {{
      background: #0d47a1;
    }}
  </style>
</head>
<body>
  <h1>Texture Trials Review</h1>
  <p class="page-info">{len(trials)} trials &mdash; 4 versions each (oddball highlighted in red)</p>
  <div class="nav-bar">
    {"".join(f'<a href="#trial-{t["trial_id"]}">T{t["trial_id"]}</a>' for t in trials)}
  </div>

  {"".join(trial_blocks)}

  <div class="export-bar">
    <button onclick="exportFeedback()">Export Feedback</button>
  </div>

  <script>
    function exportFeedback() {{
      const data = {{}};
      document.querySelectorAll('textarea').forEach(ta => {{
        if (ta.value.trim()) {{
          data[ta.id] = ta.value.trim();
        }}
      }});
      if (Object.keys(data).length === 0) {{
        alert('No feedback entered yet.');
        return;
      }}
      const blob = new Blob([JSON.stringify(data, null, 2)],
                            {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'texture_trial_feedback.json';
      a.click();
      URL.revokeObjectURL(url);
    }}

    // Auto-save feedback to localStorage
    document.querySelectorAll('textarea').forEach(ta => {{
      const saved = localStorage.getItem('tex_' + ta.id);
      if (saved) ta.value = saved;
      ta.addEventListener('input', () => {{
        localStorage.setItem('tex_' + ta.id, ta.value);
      }});
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"\nReview page: {output_path}")


def main():
    manifest_path = os.path.join(SCRIPT_DIR, "texture_manifest.csv")
    output_dir = os.path.join(PROJECT_DIR, "texture_trials")

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} trials from {manifest_path}")
    print(f"Output: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    trials = []
    for row in rows:
        info = process_trial(row, output_dir)
        if info:
            trials.append(info)

    html_path = os.path.join(output_dir, "review_index.html")
    generate_html(trials, html_path)
    print(f"Done: {len(trials)} trials processed.")


if __name__ == "__main__":
    main()
