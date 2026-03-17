"""
generate_crop_gallery.py
------------------------
Generate a large gallery of candidate crops for each texture trial,
then build an interactive index.html where you can select the best
3 distractors + 1 oddball per trial.

For each trial in texture_manifest.csv:
  - Generates ~12 non-overlapping 200x200 grayscale crops from same3 image
  - Generates ~12 non-overlapping 200x200 grayscale crops from diff1 image
  - Saves source image thumbnails
  - Outputs texture_trials/index.html with interactive crop picker

USAGE:
    python3 generate_crop_gallery.py
"""

import csv
import html
import json
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
MAX_ATTEMPTS = 20000
RANDOM_SEED = 42
THUMB_HEIGHT = 300
N_CANDIDATES = 12      # candidate crops per image
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


def sample_crops(arr, n, rng, min_dist=100):
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
            results.append((arr[y:y + CROP_SIZE, x:x + CROP_SIZE].copy(), x, y))
        attempts += 1
    if len(results) < n:
        print(f"    Got {len(results)}/{n} crops after {attempts} attempts.")
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
    trial_id = int(row["trial_id"])
    category = row["category"].strip()
    same3_image = row["same3_image"].strip()
    diff1_category = row["diff1_category"].strip()
    diff1_image = row["diff1_image"].strip()
    notes = row.get("notes", "").strip()

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

    # Gallery folder
    gallery_dir = os.path.join(output_dir, f"trial_{trial_id:02d}_{category}",
                               "gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    # Sample candidate crops from both images
    same3_crops = sample_crops(same3_arr, N_CANDIDATES, rng, min_dist=100)
    diff1_crops = sample_crops(diff1_arr, N_CANDIDATES, rng, min_dist=100)

    # Save grayscale crops
    same3_info = []
    for i, (crop, x, y) in enumerate(same3_crops):
        gray = to_grayscale(crop)
        fname = f"same3_{i+1:02d}.png"
        Image.fromarray(gray, mode="L").save(os.path.join(gallery_dir, fname))
        same3_info.append({"idx": i + 1, "file": fname, "x": x, "y": y})

    diff1_info = []
    for i, (crop, x, y) in enumerate(diff1_crops):
        gray = to_grayscale(crop)
        fname = f"diff1_{i+1:02d}.png"
        Image.fromarray(gray, mode="L").save(os.path.join(gallery_dir, fname))
        diff1_info.append({"idx": i + 1, "file": fname, "x": x, "y": y})

    # Save source thumbnails
    trial_dir = os.path.join(output_dir, f"trial_{trial_id:02d}_{category}")
    save_thumbnail(same3_pil, os.path.join(trial_dir, "source_same3.jpg"))
    save_thumbnail(diff1_pil, os.path.join(trial_dir, "source_diff1.jpg"))

    print(f"OK ({len(same3_crops)} same3, {len(diff1_crops)} diff1)")
    return {
        "trial_id": trial_id,
        "category": category,
        "same3_image": same3_image,
        "diff1_category": diff1_category,
        "diff1_image": diff1_image,
        "notes": notes,
        "folder": f"trial_{trial_id:02d}_{category}",
        "same3_count": len(same3_crops),
        "diff1_count": len(diff1_crops),
    }


def generate_html(trials, output_path):
    """Generate the interactive crop picker HTML."""

    # Build trial data as JSON for the JS side
    trials_json = json.dumps(trials, indent=2)

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

        # Same3 crop grid
        same3_cells = ""
        for i in range(1, t["same3_count"] + 1):
            img = f"{folder}/gallery/same3_{i:02d}.png"
            same3_cells += (
                f'<div class="crop-cell" data-trial="{tid}" data-role="same3" '
                f'data-idx="{i}" onclick="toggleCrop(this)">'
                f'<img src="{img}"><div class="label">S{i}</div></div>\n')

        # Diff1 crop grid
        diff1_cells = ""
        for i in range(1, t["diff1_count"] + 1):
            img = f"{folder}/gallery/diff1_{i:02d}.png"
            diff1_cells += (
                f'<div class="crop-cell" data-trial="{tid}" data-role="diff1" '
                f'data-idx="{i}" onclick="toggleCrop(this)">'
                f'<img src="{img}"><div class="label">O{i}</div></div>\n')

        block = f"""
    <div class="trial-block" id="trial-{tid}">
      <h2>Trial {tid}: {cat} {notes_html}</h2>
      <div class="source-info">
        Distractors from: <strong>{cat}/{s3}</strong> &nbsp;|&nbsp;
        Oddball from: <strong>{d1cat}/{d1}</strong>
      </div>
      <div class="source-images">
        <div class="source-img">
          <img src="{folder}/source_same3.jpg">
          <div class="label">Same3: {cat}/{s3}</div>
        </div>
        <div class="source-img">
          <img src="{folder}/source_diff1.jpg">
          <div class="label">Diff1: {d1cat}/{d1}</div>
        </div>
      </div>

      <h3>Select 3 distractors <span class="count-badge" id="count-same3-{tid}">0/3</span></h3>
      <div class="crop-grid">{same3_cells}</div>

      <h3>Select 1 oddball <span class="count-badge" id="count-diff1-{tid}">0/1</span></h3>
      <div class="crop-grid">{diff1_cells}</div>

      <div class="preview-section" id="preview-{tid}">
        <h3>Preview (selected crops)</h3>
        <div class="preview-row" id="preview-row-{tid}"></div>
      </div>

      <div class="feedback-section">
        <label for="feedback-{tid}">Notes:</label>
        <textarea id="feedback-{tid}" rows="2"
          placeholder="e.g. needs luminance normalization, crop quality notes..."></textarea>
      </div>
    </div>"""
        trial_blocks.append(block)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Texture Crop Picker</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f5;
      color: #333;
      padding: 20px 40px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    h1 {{ text-align: center; margin-bottom: 6px; font-size: 24px; }}
    .page-info {{
      text-align: center; color: #666; margin-bottom: 8px; font-size: 14px;
    }}
    .nav-bar {{
      text-align: center; margin-bottom: 20px; padding: 10px;
      background: #fff; border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .nav-bar a {{ margin: 0 6px; text-decoration: none; color: #1565c0; font-size: 14px; }}
    .nav-bar a:hover {{ text-decoration: underline; }}
    .trial-block {{
      background: #fff; border-radius: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      padding: 24px; margin-bottom: 30px;
    }}
    .trial-block h2 {{
      font-size: 18px; margin-bottom: 6px;
      border-bottom: 2px solid #e0e0e0; padding-bottom: 6px;
    }}
    .trial-block h3 {{
      font-size: 14px; margin: 14px 0 8px; color: #444;
    }}
    .notes {{ color: #e65100; font-weight: normal; font-size: 14px; }}
    .count-badge {{
      display: inline-block; padding: 2px 8px; border-radius: 10px;
      font-size: 12px; font-weight: 600; background: #e0e0e0; color: #555;
    }}
    .count-badge.complete {{ background: #c8e6c9; color: #2e7d32; }}
    .source-info {{ font-size: 13px; color: #666; margin-bottom: 12px; }}
    .source-images {{
      display: flex; gap: 20px; margin-bottom: 14px; justify-content: center;
    }}
    .source-img {{ text-align: center; }}
    .source-img img {{ max-height: 180px; border: 1px solid #ccc; border-radius: 4px; }}
    .source-img .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
    .crop-grid {{
      display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;
    }}
    .crop-cell {{
      text-align: center; cursor: pointer; position: relative;
      border: 3px solid transparent; border-radius: 6px;
      transition: border-color 0.15s, opacity 0.15s;
    }}
    .crop-cell:hover {{ border-color: #bbb; }}
    .crop-cell.selected {{ border-color: #1565c0; }}
    .crop-cell.selected[data-role="diff1"] {{ border-color: #e53935; }}
    .crop-cell img {{
      width: 160px; height: 160px; display: block; border-radius: 3px;
    }}
    .crop-cell .label {{
      font-size: 11px; color: #888; margin-top: 2px;
    }}
    .crop-cell.selected .label {{ color: #1565c0; font-weight: 600; }}
    .crop-cell.selected[data-role="diff1"] .label {{ color: #e53935; }}
    .crop-cell .check {{
      display: none; position: absolute; top: 4px; right: 4px;
      width: 22px; height: 22px; border-radius: 50%;
      background: #1565c0; color: #fff; font-size: 14px;
      line-height: 22px; text-align: center;
    }}
    .crop-cell.selected[data-role="diff1"] .check {{ background: #e53935; }}
    .crop-cell.selected .check {{ display: block; }}
    .preview-section {{
      margin-top: 14px; padding: 12px; background: #fafafa;
      border-radius: 6px; border: 1px solid #e0e0e0;
    }}
    .preview-section h3 {{ margin: 0 0 8px; }}
    .preview-row {{
      display: flex; gap: 12px; min-height: 170px; align-items: center;
    }}
    .preview-item {{ text-align: center; }}
    .preview-item img {{
      width: 160px; height: 160px; border: 3px solid transparent; border-radius: 4px;
    }}
    .preview-item.oddball img {{ border-color: #e53935; }}
    .preview-item .label {{ font-size: 11px; color: #666; margin-top: 2px; }}
    .preview-item.oddball .label {{ color: #e53935; font-weight: 600; }}
    .feedback-section {{ margin-top: 12px; }}
    .feedback-section label {{
      display: block; font-size: 13px; font-weight: 600; margin-bottom: 3px; color: #444;
    }}
    .feedback-section textarea {{
      width: 100%; font-family: inherit; font-size: 13px; padding: 6px;
      border: 1px solid #ccc; border-radius: 6px; resize: vertical;
    }}
    .feedback-section textarea:focus {{
      outline: none; border-color: #1565c0;
      box-shadow: 0 0 0 2px rgba(21,101,192,0.15);
    }}
    .export-bar {{
      text-align: center; margin: 30px 0; display: flex;
      gap: 12px; justify-content: center;
    }}
    .export-bar button {{
      padding: 10px 28px; font-size: 14px; font-weight: 600;
      background: #1565c0; color: #fff; border: none;
      border-radius: 6px; cursor: pointer;
    }}
    .export-bar button:hover {{ background: #0d47a1; }}
    .export-bar button.secondary {{
      background: #fff; color: #1565c0; border: 2px solid #1565c0;
    }}
    .export-bar button.secondary:hover {{ background: #e3f2fd; }}
    .empty-msg {{
      color: #999; font-style: italic; font-size: 13px; padding: 20px;
    }}
  </style>
</head>
<body>
  <h1>Texture Crop Picker</h1>
  <p class="page-info">{len(trials)} trials &mdash; click crops to select 3 distractors (blue) + 1 oddball (red) per trial</p>
  <div class="nav-bar">
    {"".join(f'<a href="#trial-{t["trial_id"]}">T{t["trial_id"]}</a>' for t in trials)}
  </div>

  {"".join(trial_blocks)}

  <div class="export-bar">
    <button class="secondary" onclick="exportSelections()">Export Selections (JSON)</button>
    <button onclick="exportFeedback()">Export All (Selections + Notes)</button>
  </div>

  <script>
    // Selection state: {{ trialId: {{ same3: [idx, ...], diff1: [idx] }} }}
    const selections = {{}};

    function toggleCrop(el) {{
      const tid = el.dataset.trial;
      const role = el.dataset.role;
      const idx = parseInt(el.dataset.idx);
      const maxSel = role === 'same3' ? 3 : 1;

      if (!selections[tid]) selections[tid] = {{ same3: [], diff1: [] }};
      const arr = selections[tid][role];
      const pos = arr.indexOf(idx);

      if (pos >= 0) {{
        // Deselect
        arr.splice(pos, 1);
        el.classList.remove('selected');
      }} else if (arr.length < maxSel) {{
        // Select
        arr.push(idx);
        el.classList.add('selected');
      }} else {{
        // At max — deselect oldest, select new
        const oldest = arr.shift();
        const oldEl = document.querySelector(
          `.crop-cell[data-trial="${{tid}}"][data-role="${{role}}"][data-idx="${{oldest}}"]`);
        if (oldEl) oldEl.classList.remove('selected');
        arr.push(idx);
        el.classList.add('selected');
      }}

      updateCount(tid, role);
      updatePreview(tid);
      saveState();
    }}

    function updateCount(tid, role) {{
      const maxSel = role === 'same3' ? 3 : 1;
      const badge = document.getElementById(`count-${{role}}-${{tid}}`);
      const count = (selections[tid] && selections[tid][role]) ? selections[tid][role].length : 0;
      badge.textContent = `${{count}}/${{maxSel}}`;
      badge.classList.toggle('complete', count === maxSel);
    }}

    function updatePreview(tid) {{
      const row = document.getElementById(`preview-row-${{tid}}`);
      if (!selections[tid]) {{ row.innerHTML = '<p class="empty-msg">Click crops above to select</p>'; return; }}

      const sel = selections[tid];
      let html = '';

      // Show distractors first, then oddball
      const folder = document.querySelector(`#trial-${{tid}} .crop-cell`).closest('.trial-block')
        .querySelector('.crop-cell').parentElement.querySelector('.crop-cell').querySelector('img')
        .src.split('/gallery/')[0].split('/').pop();

      // Get folder from first crop cell
      const firstCell = document.querySelector(`.crop-cell[data-trial="${{tid}}"][data-role="same3"]`);
      const basePath = firstCell ? firstCell.querySelector('img').src.replace(/gallery\\/.*/, '') : '';

      for (const idx of sel.same3) {{
        const src = basePath + `gallery/same3_${{String(idx).padStart(2,'0')}}.png`;
        html += `<div class="preview-item"><img src="${{src}}"><div class="label">D${{sel.same3.indexOf(idx)+1}}</div></div>`;
      }}
      for (const idx of sel.diff1) {{
        const src = basePath + `gallery/diff1_${{String(idx).padStart(2,'0')}}.png`;
        html += `<div class="preview-item oddball"><img src="${{src}}"><div class="label">Oddball</div></div>`;
      }}

      if (!html) html = '<p class="empty-msg">Click crops above to select</p>';
      row.innerHTML = html;
    }}

    function saveState() {{
      const data = {{ selections, feedback: {{}} }};
      document.querySelectorAll('textarea').forEach(ta => {{
        if (ta.value.trim()) data.feedback[ta.id] = ta.value.trim();
      }});
      localStorage.setItem('texture_crop_picker', JSON.stringify(data));
    }}

    function loadState() {{
      const raw = localStorage.getItem('texture_crop_picker');
      if (!raw) return;
      try {{
        const data = JSON.parse(raw);
        if (data.selections) {{
          Object.assign(selections, data.selections);
          // Restore visual state
          for (const [tid, sel] of Object.entries(selections)) {{
            for (const role of ['same3', 'diff1']) {{
              for (const idx of (sel[role] || [])) {{
                const el = document.querySelector(
                  `.crop-cell[data-trial="${{tid}}"][data-role="${{role}}"][data-idx="${{idx}}"]`);
                if (el) el.classList.add('selected');
              }}
              updateCount(tid, role);
            }}
            updatePreview(tid);
          }}
        }}
        if (data.feedback) {{
          for (const [id, val] of Object.entries(data.feedback)) {{
            const ta = document.getElementById(id);
            if (ta) ta.value = val;
          }}
        }}
      }} catch(e) {{ console.error('Failed to load state:', e); }}
    }}

    function exportSelections() {{
      const out = {{}};
      for (const [tid, sel] of Object.entries(selections)) {{
        if (sel.same3.length || sel.diff1.length) {{
          out[`trial_${{tid}}`] = {{
            distractors: sel.same3.map(i => `same3_${{String(i).padStart(2,'0')}}.png`),
            oddball: sel.diff1.map(i => `diff1_${{String(i).padStart(2,'0')}}.png`),
          }};
        }}
      }}
      download('texture_crop_selections.json', out);
    }}

    function exportFeedback() {{
      const out = {{ selections: {{}}, feedback: {{}} }};
      for (const [tid, sel] of Object.entries(selections)) {{
        if (sel.same3.length || sel.diff1.length) {{
          out.selections[`trial_${{tid}}`] = {{
            distractors: sel.same3.map(i => `same3_${{String(i).padStart(2,'0')}}.png`),
            oddball: sel.diff1.map(i => `diff1_${{String(i).padStart(2,'0')}}.png`),
          }};
        }}
      }}
      document.querySelectorAll('textarea').forEach(ta => {{
        if (ta.value.trim()) out.feedback[ta.id] = ta.value.trim();
      }});
      download('texture_crop_feedback.json', out);
    }}

    function download(filename, data) {{
      const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
    }}

    // Auto-save feedback on input
    document.querySelectorAll('textarea').forEach(ta => {{
      ta.addEventListener('input', saveState);
    }});

    // Load saved state
    loadState();
  </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"\nCrop picker: {output_path}")


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

    html_path = os.path.join(output_dir, "index.html")
    generate_html(trials, html_path)
    print(f"Done: {len(trials)} trials processed.")


if __name__ == "__main__":
    main()
