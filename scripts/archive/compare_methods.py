"""
compare_methods.py
------------------
Side-by-side comparison of the two bitmap generation methods
for the same base color anchors.

Method 1 (Hue Rotation): distractor and oddball come from different
  hue regions — genuinely different colors.

Method 2 (Platykurtic): one shared palette; distractor and oddball
  use the same colors but with the peak of the distribution shifted.

Run:
    python3 scripts/compare_methods.py
Output:
    scripts/method_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import lab2rgb, rgb2lab
from PIL import Image as PILImage

# ── Config ────────────────────────────────────────────────────────────────────

BITMAP_SIZE = 200   # px per bitmap (smaller for the comparison grid)
TILE_GRID   = 20
SEED        = 42

# Base color anchors to demonstrate: (label, L, a, b)
EXAMPLES = [
    ("Red-Pink\n(L=60, a=70, b=−60)", 60,  70, -60),
    ("Teal\n(L=60, a=−80, b=−20)",    60, -80, -20),
    ("Yellow-Green\n(L=60, a=60, b=80)", 60,  60,  80),
]

# Method 1 parameters
M1_PALETTE_DE = 4    # ΔE between adjacent shades within each palette
M1_ODDBALL_DE = 15   # ΔE separation between distractor and oddball anchor
M1_N_SHADES   = 10

# Method 2 parameters
M2_PALETTE_DE = 4    # ΔE between adjacent shades (same palette for both)
M2_N_COLORS   = 20   # total shades in the shared palette
M2_SIGMA      = 5    # distribution width
M2_MEAN_SHIFT = 7    # how many shades to shift the oddball peak

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "method_comparison.png")

# ── Core functions (ported from notebook) ─────────────────────────────────────

def lab_palette_colors(L, a, b, palette_de, n_shades):
    """n_shades colors stepping tangentially (hue direction) by palette_de ΔE."""
    C = np.sqrt(a**2 + b**2)
    offsets = np.arange(n_shades) - (n_shades - 1) / 2.0
    lab = np.stack([
        np.full(n_shades, L),
        a + offsets * palette_de * (-b / C),
        b + offsets * palette_de * (a / C),
    ], axis=1)
    rgb = np.clip(lab2rgb(lab.reshape(1, -1, 3))[0], 0, 1)
    return np.round(rgb * 255).astype(np.uint8)


def rotate_ab(a, b, de):
    """Rotate (a*, b*) anchor by de ΔE around the origin."""
    C = np.sqrt(a**2 + b**2)
    deg = np.degrees(2 * np.arcsin(np.clip(de / (2 * C), 0, 1)))
    h = np.arctan2(b, a) + np.deg2rad(deg)
    return C * np.cos(h), C * np.sin(h)


def platykurtic_proportions(n_colors, mean_idx, sigma):
    """Uniform floor + Gaussian peak (alpha=0.5)."""
    indices  = np.arange(n_colors, dtype=float)
    uniform  = np.ones(n_colors) / n_colors
    gaussian = np.exp(-0.5 * ((indices - mean_idx) / sigma) ** 2)
    gaussian /= gaussian.sum()
    weights  = 0.5 * uniform + 0.5 * gaussian
    return weights / weights.sum()


def generate_bitmap(colors_rgb, proportions, rng, size=BITMAP_SIZE, tile_grid=TILE_GRID):
    cell_size   = size // tile_grid
    out_size    = cell_size * tile_grid
    total_cells = tile_grid * tile_grid

    counts = np.round(proportions * total_cells).astype(int)
    counts[0] += total_cells - counts.sum()

    cell_colors = np.repeat(np.arange(len(colors_rgb)), counts)
    rng.shuffle(cell_colors)

    bmp = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    for idx, ci in enumerate(cell_colors):
        row, col = divmod(idx, tile_grid)
        y0, x0 = row * cell_size, col * cell_size
        bmp[y0:y0 + cell_size, x0:x0 + cell_size] = colors_rgb[ci]
    return bmp


def equal_props(n):
    return np.ones(n) / n


# ── Generate trials ───────────────────────────────────────────────────────────

def make_m1_trial(L, a, b):
    """Method 1: two separate hue anchors."""
    rng = np.random.RandomState(SEED)
    dist_colors = lab_palette_colors(L, a, b, M1_PALETTE_DE, M1_N_SHADES)
    a_odd, b_odd = rotate_ab(a, b, M1_ODDBALL_DE)
    odd_colors  = lab_palette_colors(L, a_odd, b_odd, M1_PALETTE_DE, M1_N_SHADES)

    dist_props = equal_props(M1_N_SHADES)
    odd_props  = equal_props(M1_N_SHADES)

    oddball = generate_bitmap(odd_colors,  odd_props,  np.random.RandomState(SEED))
    dists   = [generate_bitmap(dist_colors, dist_props, np.random.RandomState(SEED + i + 1))
               for i in range(3)]
    return dist_colors, odd_colors, dist_props, odd_props, oddball, dists


def make_m2_trial(L, a, b):
    """Method 2: shared palette, shifted distribution."""
    colors    = lab_palette_colors(L, a, b, M2_PALETTE_DE, M2_N_COLORS)
    midpoint  = (M2_N_COLORS - 1) / 2.0
    dist_props = platykurtic_proportions(M2_N_COLORS, midpoint,              M2_SIGMA)
    odd_props  = platykurtic_proportions(M2_N_COLORS, midpoint + M2_MEAN_SHIFT, M2_SIGMA)

    oddball = generate_bitmap(colors, odd_props,  np.random.RandomState(SEED))
    dists   = [generate_bitmap(colors, dist_props, np.random.RandomState(SEED + i + 1))
               for i in range(3)]
    return colors, dist_props, odd_props, oddball, dists


# ── Layout ────────────────────────────────────────────────────────────────────

def palette_strip(ax, colors_rgb, title, highlight_split=None):
    """Draw a horizontal palette strip. highlight_split draws a dividing line."""
    n = len(colors_rgb)
    for i, c in enumerate(colors_rgb.astype(float) / 255.0):
        ax.add_patch(patches.Rectangle((i, 0), 1, 1, color=c))
    if highlight_split is not None:
        ax.axvline(highlight_split, color='white', lw=2, ls='--')
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=7, pad=2)


def dist_bar(ax, props, colors_rgb, title):
    """Bar chart of proportions, colored by the palette colors."""
    x = np.arange(len(props))
    colors = colors_rgb.astype(float) / 255.0
    ax.bar(x, props, color=colors, edgecolor='none', width=1.0)
    ax.set_xlim(-0.5, len(props) - 0.5)
    ax.set_ylim(0, props.max() * 1.4)
    ax.axis('off')
    ax.set_title(title, fontsize=7, pad=2)


def show_bitmap(ax, bmp, label):
    ax.imshow(bmp)
    ax.set_title(label, fontsize=7, pad=2)
    ax.axis('off')


# ── Main figure ───────────────────────────────────────────────────────────────

n_examples = len(EXAMPLES)

# Row layout per example (2 methods × each row has: palette | dist | oddball | d1 | d2 | d3)
# We'll make a tall figure: for each example, 2 method-rows + a separator

# Columns: [palette_dist] [palette_odd] [dist_bar] [oddball] [dist1] [dist2] [dist3]
N_COLS = 7
N_ROWS_PER_EX = 3  # method1, method2, + label row
TOTAL_ROWS = n_examples * N_ROWS_PER_EX

fig = plt.figure(figsize=(N_COLS * 2.2, n_examples * 5.2))
fig.patch.set_facecolor('#f5f5f5')

col_widths = [1.5, 1.5, 1.5, 2, 2, 2, 2]  # relative widths

for ex_idx, (label, L, a, b) in enumerate(EXAMPLES):

    # ── Method 1 ──────────────────────────────────────────────────────────────
    dist_colors, odd_colors, dist_props, odd_props, oddball, dists = make_m1_trial(L, a, b)

    base_row = ex_idx * N_ROWS_PER_EX

    # Title row for this example
    ax_title = fig.add_subplot(TOTAL_ROWS, 1, base_row + 1)
    ax_title.axis('off')
    ax_title.text(0.01, 0.5, label, fontsize=11, fontweight='bold',
                  va='center', transform=ax_title.transAxes)

    # -- This is getting complex with add_subplot; use gridspec instead
    # Rebuild using GridSpec
    pass

plt.close(fig)

# ── Cleaner layout with GridSpec ──────────────────────────────────────────────

import matplotlib.gridspec as gridspec

# For each example: 4 rows
#   row 0: title spanning all cols
#   row 1: METHOD 1 — palette_dist | palette_odd | [spacer] | oddball | d1 | d2 | d3
#   row 2: METHOD 2 — shared_palette spanning 2 | dist_bar | oddball | d1 | d2 | d3
#   row 3: thin separator

ROW_HEIGHTS_PER_EX = [0.25, 1, 1, 0.15]   # title, m1, m2, sep
all_heights = []
for _ in range(n_examples):
    all_heights += ROW_HEIGHTS_PER_EX

fig = plt.figure(figsize=(15, n_examples * 4.5))
fig.patch.set_facecolor('#f5f5f5')

gs_outer = gridspec.GridSpec(
    len(all_heights), 1,
    height_ratios=all_heights,
    hspace=0.05,
    left=0.01, right=0.99, top=0.97, bottom=0.02
)

BITMAP_COLS = 4    # oddball + 3 distractors
PAL_COLS    = 3    # palette_dist, palette_odd, dist_bar (or shared pal + bar)

for ex_idx, (label, L, a, b) in enumerate(EXAMPLES):

    base = ex_idx * len(ROW_HEIGHTS_PER_EX)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs_outer[base, :])
    ax_t.axis('off')
    ax_t.set_facecolor('#e0e0e0')
    ax_t.text(0.0, 0.5, f"  {label.replace(chr(10), '  |  ')}",
              fontsize=12, fontweight='bold', va='center',
              transform=ax_t.transAxes)

    # ── Method 1 row ──────────────────────────────────────────────────────────
    dist_colors, odd_colors, dist_props, odd_props, oddball_m1, dists_m1 = make_m1_trial(L, a, b)

    gs_m1 = gridspec.GridSpecFromSubplotSpec(
        1, PAL_COLS + BITMAP_COLS,
        subplot_spec=gs_outer[base + 1, :],
        wspace=0.04,
        width_ratios=[1.2, 1.2, 0.6, 1.5, 1.5, 1.5, 1.5],
    )

    # Palette strips: distractor palette | oddball palette | (empty)
    ax_pd = fig.add_subplot(gs_m1[0, 0])
    palette_strip(ax_pd, dist_colors,
                  f"Distractor palette\n({M1_N_SHADES} shades, pΔE={M1_PALETTE_DE})")
    ax_pd.set_ylabel("METHOD 1\n(different hue\nregions)",
                     fontsize=8, rotation=90, labelpad=4, va='center')
    ax_pd.yaxis.set_label_coords(-0.18, 0.5)

    ax_po = fig.add_subplot(gs_m1[0, 1])
    palette_strip(ax_po, odd_colors,
                  f"Oddball palette\n(rotated {M1_ODDBALL_DE} ΔE)")

    ax_sp = fig.add_subplot(gs_m1[0, 2])
    ax_sp.axis('off')

    bm_labels = ["ODDBALL", "distractor 1", "distractor 2", "distractor 3"]
    bitmaps_m1 = [oddball_m1] + dists_m1
    for col_i, (bm, lbl) in enumerate(zip(bitmaps_m1, bm_labels)):
        ax_bm = fig.add_subplot(gs_m1[0, PAL_COLS + col_i])
        show_bitmap(ax_bm, bm, lbl)
        if lbl == "ODDBALL":
            for spine in ax_bm.spines.values():
                spine.set_edgecolor('crimson')
                spine.set_linewidth(2)

    # ── Method 2 row ──────────────────────────────────────────────────────────
    colors_m2, dist_props_m2, odd_props_m2, oddball_m2, dists_m2 = make_m2_trial(L, a, b)

    gs_m2 = gridspec.GridSpecFromSubplotSpec(
        1, PAL_COLS + BITMAP_COLS,
        subplot_spec=gs_outer[base + 2, :],
        wspace=0.04,
        width_ratios=[1.2, 1.2, 0.6, 1.5, 1.5, 1.5, 1.5],
    )

    # Shared palette spanning 2 columns
    ax_sh = fig.add_subplot(gs_m2[0, 0:2])
    palette_strip(ax_sh, colors_m2,
                  f"Shared palette ({M2_N_COLORS} shades, pΔE={M2_PALETTE_DE})\n"
                  f"— same colors in ALL bitmaps")
    ax_sh.set_ylabel("METHOD 2\n(same colors,\nshifted amounts)",
                     fontsize=8, rotation=90, labelpad=4, va='center')
    ax_sh.yaxis.set_label_coords(-0.09, 0.5)

    # Distribution bar chart
    ax_bar = fig.add_subplot(gs_m2[0, 2])
    x = np.arange(M2_N_COLORS)
    midpoint = (M2_N_COLORS - 1) / 2.0
    ax_bar.bar(x, dist_props_m2, color='steelblue', alpha=0.7, width=1.0, label='distractors')
    ax_bar.bar(x, odd_props_m2,  color='tomato',    alpha=0.7, width=1.0, label='oddball')
    ax_bar.set_xlim(-0.5, M2_N_COLORS - 0.5)
    ax_bar.set_ylim(0, max(dist_props_m2.max(), odd_props_m2.max()) * 1.5)
    ax_bar.legend(fontsize=5, loc='upper right', framealpha=0.5)
    ax_bar.tick_params(labelsize=5)
    ax_bar.set_title("Proportion\ndistributions", fontsize=7, pad=2)
    ax_bar.spines[['top', 'right']].set_visible(False)

    bitmaps_m2 = [oddball_m2] + dists_m2
    for col_i, (bm, lbl) in enumerate(zip(bitmaps_m2, bm_labels)):
        ax_bm = fig.add_subplot(gs_m2[0, PAL_COLS + col_i])
        show_bitmap(ax_bm, bm, lbl)
        if lbl == "ODDBALL":
            for spine in ax_bm.spines.values():
                spine.set_edgecolor('crimson')
                spine.set_linewidth(2)

    # ── Separator ─────────────────────────────────────────────────────────────
    ax_sep = fig.add_subplot(gs_outer[base + 3, :])
    ax_sep.axis('off')
    ax_sep.axhline(0.5, color='#aaaaaa', lw=1)

fig.savefig(OUTPUT_PATH, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {OUTPUT_PATH}")
