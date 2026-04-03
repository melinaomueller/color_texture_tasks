"""
generate_click_crop.py
----------------------
Generate an interactive click-to-crop HTML tool for texture trials.

Instead of random candidate crops, the user clicks directly on source images
to place crop rectangles. Grayscale conversion and downscaling happen
client-side via Canvas API — no server needed.

For each trial in texture_manifest.csv:
  - Copies full-resolution source images into trial folders
  - Embeds trial metadata as JSON in the generated HTML

Outputs:
  texture_trials/click_crop.html — interactive crop picker

USAGE:
    python3 generate_click_crop.py
"""

import csv
import json
import os
import shutil

from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
STUFF_DIR = os.path.join(PROJECT_DIR, "STUFF_enhanced_dataset_3514_images")
CROP_SIZE = 200
# ──────────────────────────────────────────────────────────────────────────────


def load_stuff_image_path(category, image_id):
    """Return the path to a STUFF image, or None if not found."""
    path = os.path.join(STUFF_DIR, category, f"{int(image_id):04d}.jpg")
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found")
        return None
    return path


def process_trial(row, output_dir):
    """Copy source images and return trial metadata dict."""
    trial_id = int(row["trial_id"])
    category = row["category"].strip()
    same3_image = row["same3_image"].strip()
    diff1_category = row["diff1_category"].strip()
    diff1_image = row["diff1_image"].strip()
    crop_source_size = int(row.get("crop_source_size") or CROP_SIZE)
    notes = row.get("notes", "").strip()

    folder = f"trial_{trial_id:02d}_{category}"
    trial_dir = os.path.join(output_dir, folder)
    os.makedirs(trial_dir, exist_ok=True)

    print(f"  Trial {trial_id:02d}: {category}/{same3_image} vs "
          f"{diff1_category}/{diff1_image} ... ", end="")

    # Load and copy source images
    same3_path = load_stuff_image_path(category, same3_image)
    diff1_path = load_stuff_image_path(diff1_category, diff1_image)
    if same3_path is None or diff1_path is None:
        print("SKIP")
        return None

    same3_dest = os.path.join(trial_dir, "source_same3_full.jpg")
    diff1_dest = os.path.join(trial_dir, "source_diff1_full.jpg")
    shutil.copy2(same3_path, same3_dest)
    shutil.copy2(diff1_path, diff1_dest)

    # Get image dimensions
    with Image.open(same3_path) as img:
        same3_w, same3_h = img.size
    with Image.open(diff1_path) as img:
        diff1_w, diff1_h = img.size

    print("OK")
    return {
        "trial_id": trial_id,
        "category": category,
        "same3_image": same3_image,
        "diff1_category": diff1_category,
        "diff1_image": diff1_image,
        "crop_source_size": crop_source_size,
        "notes": notes,
        "folder": folder,
        "same3_src": f"{folder}/source_same3_full.jpg",
        "diff1_src": f"{folder}/source_diff1_full.jpg",
        "same3_w": same3_w,
        "same3_h": same3_h,
        "diff1_w": diff1_w,
        "diff1_h": diff1_h,
    }


def generate_html(trials, output_path):
    """Generate the interactive click-to-crop HTML."""
    trials_json = json.dumps(trials, indent=2)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Click-to-Crop Tool</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f5; color: #333;
      padding: 20px 40px; max-width: 1900px; margin: 0 auto;
    }}
    h1 {{ text-align: center; margin-bottom: 6px; font-size: 24px; }}
    .page-info {{
      text-align: center; color: #666; margin-bottom: 8px; font-size: 14px;
    }}
    .nav-bar {{
      text-align: center; margin-bottom: 20px; padding: 10px;
      background: #fff; border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      position: sticky; top: 0; z-index: 100;
    }}
    .nav-bar a {{
      margin: 0 6px; text-decoration: none; color: #1565c0; font-size: 14px;
    }}
    .nav-bar a:hover {{ text-decoration: underline; }}
    .nav-bar a.complete {{ color: #2e7d32; font-weight: 600; }}

    .trial-block {{
      background: #fff; border-radius: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      padding: 24px; margin-bottom: 30px;
    }}
    .trial-block h2 {{
      font-size: 18px; margin-bottom: 6px;
      border-bottom: 2px solid #e0e0e0; padding-bottom: 6px;
    }}
    .notes {{ color: #e65100; font-weight: normal; font-size: 14px; }}
    .source-info {{ font-size: 13px; color: #666; margin-bottom: 12px; }}

    .image-panels {{
      display: flex; gap: 30px; margin-bottom: 16px; flex-wrap: wrap;
    }}
    .image-panel {{
      flex: 1; min-width: 400px;
    }}
    .image-panel h3 {{
      font-size: 14px; margin-bottom: 8px; color: #444;
    }}
    .count-badge {{
      display: inline-block; padding: 2px 8px; border-radius: 10px;
      font-size: 12px; font-weight: 600; background: #e0e0e0; color: #555;
    }}
    .count-badge.complete {{ background: #c8e6c9; color: #2e7d32; }}

    .source-container {{
      position: relative; display: inline-block; cursor: crosshair;
      border: 2px solid #ccc; border-radius: 4px; overflow: hidden;
    }}
    .source-container img {{
      display: block; max-width: 700px; height: auto;
    }}
    .crop-overlay {{
      position: absolute; border: 2px solid; pointer-events: none;
      opacity: 0.7;
    }}
    .crop-overlay.distractor {{ border-color: #1565c0; background: rgba(21,101,192,0.15); }}
    .crop-overlay.oddball {{ border-color: #e53935; background: rgba(229,57,53,0.15); }}
    .crop-overlay .crop-label {{
      position: absolute; top: 2px; left: 2px;
      font-size: 11px; font-weight: 700; padding: 1px 4px;
      border-radius: 3px; color: #fff;
    }}
    .crop-overlay.distractor .crop-label {{ background: #1565c0; }}
    .crop-overlay.oddball .crop-label {{ background: #e53935; }}

    .preview-row {{
      display: flex; gap: 14px; margin-top: 12px; min-height: 320px;
      align-items: flex-start; flex-wrap: wrap;
    }}
    .preview-item {{
      text-align: center; cursor: pointer; position: relative;
      border: 3px solid transparent; border-radius: 6px;
      transition: border-color 0.15s;
    }}
    .preview-item:hover {{ border-color: #e53935; }}
    .preview-item canvas {{
      display: block; border-radius: 3px;
      width: 300px; height: 300px;
    }}
    .preview-item .label {{
      font-size: 13px; color: #666; margin-top: 4px;
    }}
    .preview-item .remove-hint {{
      font-size: 11px; color: #e53935; margin-top: 2px;
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
    .export-bar button.danger {{
      background: #fff; color: #e53935; border: 2px solid #e53935;
    }}
    .export-bar button.danger:hover {{ background: #ffebee; }}

    .empty-msg {{
      color: #999; font-style: italic; font-size: 13px; padding: 20px;
    }}
  </style>
</head>
<body>
  <h1>Click-to-Crop Tool</h1>
  <p class="page-info">Click on source images to place crop regions. Click previews to remove them.</p>
  <div class="nav-bar" id="nav-bar"></div>
  <div id="trials-container"></div>

  <div class="export-bar">
    <button class="secondary" onclick="exportJSON()">Export JSON</button>
    <button class="danger" onclick="clearAll()">Clear All</button>
  </div>

  <script>
    const TRIALS = {trials_json};
    const CROP_SIZE = {CROP_SIZE};

    // State: {{ trialId: {{ distractors: [{{x, y}}], oddballs: [{{x, y}}] }} }}
    let state = {{}};

    // ── Build UI ──────────────────────────────────────────────────────────────

    function buildUI() {{
      const nav = document.getElementById('nav-bar');
      const container = document.getElementById('trials-container');
      let navHtml = '';

      for (const t of TRIALS) {{
        const tid = t.trial_id;
        navHtml += `<a href="#trial-${{tid}}" id="nav-${{tid}}">T${{tid}}</a>`;

        const notesHtml = t.notes ? `<span class="notes">(${{t.notes}})</span>` : '';
        const block = document.createElement('div');
        block.className = 'trial-block';
        block.id = `trial-${{tid}}`;
        block.innerHTML = `
          <h2>Trial ${{tid}}: ${{t.category}} ${{notesHtml}}</h2>
          <div class="source-info">
            Distractors from: <strong>${{t.category}}/${{t.same3_image}}</strong> &nbsp;|&nbsp;
            Oddball from: <strong>${{t.diff1_category}}/${{t.diff1_image}}</strong> &nbsp;|&nbsp;
            Source crop: <strong>${{t.crop_source_size}}&times;${{t.crop_source_size}}</strong>
            &rarr; ${{CROP_SIZE}}&times;${{CROP_SIZE}}
          </div>
          <div class="image-panels">
            <div class="image-panel">
              <h3>Same3 (distractors)
                <span class="count-badge" id="count-dist-${{tid}}">0/3</span>
              </h3>
              <div class="source-container" id="src-same3-${{tid}}"
                   data-trial="${{tid}}" data-role="same3"
                   data-natw="${{t.same3_w}}" data-nath="${{t.same3_h}}"
                   data-cropsize="${{t.crop_source_size}}">
                <img src="${{t.same3_src}}" draggable="false">
              </div>
              <div class="preview-row" id="prev-dist-${{tid}}">
                <p class="empty-msg">Click the image above to place crops</p>
              </div>
            </div>
            <div class="image-panel">
              <h3>Diff1 (oddballs)
                <span class="count-badge" id="count-odd-${{tid}}">0/3</span>
              </h3>
              <div class="source-container" id="src-diff1-${{tid}}"
                   data-trial="${{tid}}" data-role="diff1"
                   data-natw="${{t.diff1_w}}" data-nath="${{t.diff1_h}}"
                   data-cropsize="${{t.crop_source_size}}">
                <img src="${{t.diff1_src}}" draggable="false">
              </div>
              <div class="preview-row" id="prev-odd-${{tid}}">
                <p class="empty-msg">Click the image above to place crops</p>
              </div>
            </div>
          </div>
        `;
        container.appendChild(block);
      }}

      nav.innerHTML = navHtml;

      // Attach click handlers after images load
      document.querySelectorAll('.source-container').forEach(sc => {{
        const img = sc.querySelector('img');
        img.addEventListener('load', () => setupClick(sc));
        if (img.complete) setupClick(sc);
      }});
    }}

    // ── Click handler on source images ────────────────────────────────────────

    function setupClick(container) {{
      // Only attach once
      if (container.dataset.ready) return;
      container.dataset.ready = '1';

      container.addEventListener('click', (e) => {{
        const img = container.querySelector('img');
        const rect = img.getBoundingClientRect();
        const displayW = rect.width;
        const displayH = rect.height;
        const natW = parseInt(container.dataset.natw);
        const natH = parseInt(container.dataset.nath);
        const cropSize = parseInt(container.dataset.cropsize);
        const tid = parseInt(container.dataset.trial);
        const role = container.dataset.role;

        // Scale factor: display -> original
        const scaleX = natW / displayW;
        const scaleY = natH / displayH;

        // Click position in original-image pixels
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        // Center crop on click, clamp to image bounds
        let x = Math.round(clickX - cropSize / 2);
        let y = Math.round(clickY - cropSize / 2);
        x = Math.max(0, Math.min(x, natW - cropSize));
        y = Math.max(0, Math.min(y, natH - cropSize));

        // Check limits
        const key = role === 'same3' ? 'distractors' : 'oddballs';
        if (!state[tid]) state[tid] = {{ distractors: [], oddballs: [] }};
        if (state[tid][key].length >= 3) {{
          alert(`Already have 3 ${{key}} for trial ${{tid}}. Remove one first.`);
          return;
        }}

        state[tid][key].push({{ x, y }});
        renderTrial(tid);
        saveState();
      }});
    }}

    // ── Render overlays and previews for a trial ──────────────────────────────

    function renderTrial(tid) {{
      const trial = TRIALS.find(t => t.trial_id === tid);
      if (!trial) return;
      const ts = state[tid] || {{ distractors: [], oddballs: [] }};

      // Render both panels
      renderPanel(tid, 'same3', 'distractors', trial.same3_src,
                  trial.same3_w, trial.same3_h, trial.crop_source_size, ts.distractors);
      renderPanel(tid, 'diff1', 'oddballs', trial.diff1_src,
                  trial.diff1_w, trial.diff1_h, trial.crop_source_size, ts.oddballs);

      // Update counts
      const distBadge = document.getElementById(`count-dist-${{tid}}`);
      distBadge.textContent = `${{ts.distractors.length}}/3`;
      distBadge.classList.toggle('complete', ts.distractors.length === 3);

      const oddBadge = document.getElementById(`count-odd-${{tid}}`);
      oddBadge.textContent = `${{ts.oddballs.length}}/3`;
      oddBadge.classList.toggle('complete', ts.oddballs.length >= 1);

      // Update nav
      updateNav(tid);
    }}

    function renderPanel(tid, role, key, imgSrc, natW, natH, cropSize, crops) {{
      const container = document.getElementById(`src-${{role}}-${{tid}}`);
      const img = container.querySelector('img');

      // Remove old overlays
      container.querySelectorAll('.crop-overlay').forEach(o => o.remove());

      // Display scale
      const displayW = img.clientWidth;
      const displayH = img.clientHeight;
      const scaleX = displayW / natW;
      const scaleY = displayH / natH;

      const overlayClass = key === 'distractors' ? 'distractor' : 'oddball';
      const labelPrefix = key === 'distractors' ? 'D' : 'O';

      crops.forEach((crop, i) => {{
        const ov = document.createElement('div');
        ov.className = `crop-overlay ${{overlayClass}}`;
        ov.style.left = (crop.x * scaleX) + 'px';
        ov.style.top = (crop.y * scaleY) + 'px';
        ov.style.width = (cropSize * scaleX) + 'px';
        ov.style.height = (cropSize * scaleY) + 'px';
        ov.innerHTML = `<span class="crop-label">${{labelPrefix}}${{i + 1}}</span>`;
        container.appendChild(ov);
      }});

      // Render previews
      const prevId = key === 'distractors' ? `prev-dist-${{tid}}` : `prev-odd-${{tid}}`;
      const prevRow = document.getElementById(prevId);

      if (crops.length === 0) {{
        prevRow.innerHTML = '<p class="empty-msg">Click the image above to place crops</p>';
        return;
      }}

      prevRow.innerHTML = '';
      crops.forEach((crop, i) => {{
        const item = document.createElement('div');
        item.className = 'preview-item';
        item.title = 'Click to remove';

        const canvas = document.createElement('canvas');
        canvas.width = CROP_SIZE;
        canvas.height = CROP_SIZE;
        item.appendChild(canvas);

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = `${{labelPrefix}}${{i + 1}} (${{crop.x}}, ${{crop.y}})`;
        item.appendChild(label);

        const hint = document.createElement('div');
        hint.className = 'remove-hint';
        hint.textContent = 'click to remove';
        item.appendChild(hint);

        // Remove handler
        item.addEventListener('click', () => {{
          state[tid][key].splice(i, 1);
          renderTrial(tid);
          saveState();
        }});

        prevRow.appendChild(item);

        // Draw grayscale crop on canvas
        drawGrayscaleCrop(canvas, img, crop.x, crop.y, cropSize, natW, natH);
      }});
    }}

    function drawGrayscaleCrop(canvas, sourceImg, x, y, cropSize, natW, natH) {{
      // Use an offscreen canvas at native resolution to extract the crop
      const offscreen = document.createElement('canvas');
      offscreen.width = natW;
      offscreen.height = natH;
      const offCtx = offscreen.getContext('2d');
      offCtx.drawImage(sourceImg, 0, 0, natW, natH);

      // Extract crop region and downscale to CROP_SIZE
      const ctx = canvas.getContext('2d');
      ctx.drawImage(offscreen, x, y, cropSize, cropSize, 0, 0, CROP_SIZE, CROP_SIZE);

      // Apply grayscale
      const imageData = ctx.getImageData(0, 0, CROP_SIZE, CROP_SIZE);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {{
        const gray = Math.round(0.2989 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        data[i] = data[i + 1] = data[i + 2] = gray;
      }}
      ctx.putImageData(imageData, 0, 0);
    }}

    // ── Nav badges ────────────────────────────────────────────────────────────

    function updateNav(tid) {{
      const ts = state[tid] || {{ distractors: [], oddballs: [] }};
      const link = document.getElementById(`nav-${{tid}}`);
      if (link) {{
        link.classList.toggle('complete', ts.distractors.length === 3 && ts.oddballs.length >= 1);
      }}
    }}

    // ── Persistence ───────────────────────────────────────────────────────────

    function saveState() {{
      localStorage.setItem('click_crop_state', JSON.stringify(state));
    }}

    function loadState() {{
      const raw = localStorage.getItem('click_crop_state');
      if (!raw) return;
      try {{
        state = JSON.parse(raw);
        for (const tid of Object.keys(state)) {{
          renderTrial(parseInt(tid));
        }}
      }} catch (e) {{
        console.error('Failed to load state:', e);
      }}
    }}

    // ── Export ─────────────────────────────────────────────────────────────────

    function exportJSON() {{
      const out = {{}};
      for (const t of TRIALS) {{
        const tid = t.trial_id;
        const ts = state[tid];
        if (!ts) continue;
        if (ts.distractors.length === 0 && ts.oddballs.length === 0) continue;
        out[`trial_${{tid}}`] = {{
          crop_source_size: t.crop_source_size,
          distractors: ts.distractors.map(c => ({{ x: c.x, y: c.y }})),
          oddballs: ts.oddballs.map(c => ({{ x: c.x, y: c.y }})),
        }};
      }}
      download('click_crop_selections.json', out);
    }}

    function download(filename, data) {{
      const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
    }}

    function clearAll() {{
      if (!confirm('Clear all crop selections? This cannot be undone.')) return;
      state = {{}};
      localStorage.removeItem('click_crop_state');
      for (const t of TRIALS) renderTrial(t.trial_id);
    }}

    // ── Re-render overlays on window resize ───────────────────────────────────

    let resizeTimer;
    window.addEventListener('resize', () => {{
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {{
        for (const tid of Object.keys(state)) renderTrial(parseInt(tid));
      }}, 200);
    }});

    // ── Init ──────────────────────────────────────────────────────────────────

    buildUI();
    // Wait for all images to load, then restore state
    window.addEventListener('load', loadState);
  </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)
    print(f"\nClick-to-crop tool: {output_path}")


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

    html_path = os.path.join(output_dir, "click_crop.html")
    generate_html(trials, html_path)
    print(f"Done: {len(trials)} trials processed.")


if __name__ == "__main__":
    main()
