"""
Create a PDF showing texture sets. Each set = 4 consecutive textures.
Each PDF page shows one set: top row = 4 color (unfiltered), bottom row = 4 grayscale (filtered).
36 textures / 4 per set = 9 pages.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re

base = Path(__file__).parent
filtered_dir = base / "filtered stimuli"
unfiltered_dir = base / "unfiltered stimuli"
output_path = base / "texture_sets_overview.pdf"

# Collect and sort pages by number
def page_num(fname):
    m = re.search(r"Page (\d+)", fname)
    return int(m.group(1)) if m else 0

filtered_files = sorted(filtered_dir.glob("*.png"), key=lambda p: page_num(p.name))
unfiltered_files = sorted(unfiltered_dir.glob("*.png"), key=lambda p: page_num(p.name))

# Match by page number
filtered_map = {page_num(f.name): f for f in filtered_files}
unfiltered_map = {page_num(f.name): f for f in unfiltered_files}
page_numbers = sorted(set(filtered_map.keys()) & set(unfiltered_map.keys()))

# Group into sets of 4
SET_SIZE = 4
sets = [page_numbers[i:i + SET_SIZE] for i in range(0, len(page_numbers), SET_SIZE)]

# PDF page dimensions (landscape letter at 150 DPI)
PAGE_W, PAGE_H = 1650, 1275
MARGIN = 50
TITLE_H = 45
ROW_LABEL_H = 30
GAP_X = 20  # gap between images horizontally
GAP_Y = 30  # gap between rows

# Try to get fonts
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_num = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception:
    font_title = ImageFont.load_default()
    font_label = font_title
    font_num = font_title

def fit(img, max_w, max_h):
    ratio = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)

pages = []
for set_idx, texture_pages in enumerate(sets):
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)

    # Title
    title = f"Set {set_idx + 1}  (Textures {texture_pages[0]}\u2013{texture_pages[-1]})"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((PAGE_W - tw) // 2, MARGIN // 2), title, fill="black", font=font_title)

    n_cols = len(texture_pages)
    # Available area for each image
    content_top = MARGIN + TITLE_H
    img_area_w = (PAGE_W - 2 * MARGIN - (n_cols - 1) * GAP_X) // n_cols
    img_area_h = (PAGE_H - content_top - MARGIN - GAP_Y - 2 * ROW_LABEL_H) // 2

    for row_idx, (label_text, img_map) in enumerate([("Color", unfiltered_map), ("Grayscale", filtered_map)]):
        row_top = content_top + row_idx * (img_area_h + GAP_Y + ROW_LABEL_H)

        # Row label
        bbox = draw.textbbox((0, 0), label_text, font=font_label)
        lw = bbox[2] - bbox[0]
        draw.text(((PAGE_W - lw) // 2, row_top), label_text, fill="gray", font=font_label)

        img_top = row_top + ROW_LABEL_H

        for col_idx, pn in enumerate(texture_pages):
            if pn not in img_map:
                continue
            img = Image.open(img_map[pn])
            img_resized = fit(img, img_area_w, img_area_h)

            x = MARGIN + col_idx * (img_area_w + GAP_X) + (img_area_w - img_resized.width) // 2
            y = img_top + (img_area_h - img_resized.height) // 2
            page.paste(img_resized, (x, y))

            # Light border
            draw.rectangle(
                [x - 1, y - 1, x + img_resized.width, y + img_resized.height],
                outline="#cccccc"
            )

            # Small page number label below image
            num_label = str(pn)
            bbox = draw.textbbox((0, 0), num_label, font=font_num)
            nw = bbox[2] - bbox[0]
            nx = MARGIN + col_idx * (img_area_w + GAP_X) + (img_area_w - nw) // 2
            ny = img_top + img_area_h + 2
            draw.text((nx, ny), num_label, fill="#999999", font=font_num)

    pages.append(page)
    print(f"  Set {set_idx + 1} done (textures {texture_pages})")

# Save as multi-page PDF
if pages:
    pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=150)
    print(f"\nSaved {len(pages)}-page PDF to: {output_path}")
else:
    print("No matching texture sets found.")
