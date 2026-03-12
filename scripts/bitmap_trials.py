"""
bitmap_trials.py — Generate controlled color-distribution bitmap trials.

Pipeline:
  1. Extract dominant colors from a source image via k-means
  2. Output palette swatch PNG for review
  3. Generate 4AFC bitmap trials at varying difficulty levels
  4. Output original + luminance-controlled versions

Usage:
  python bitmap_trials.py --image PATH --k 5 --extract-palette
  python bitmap_trials.py --image PATH --k 5 --generate-trial --difficulty easy medium hard
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.vq import kmeans2
from pathlib import Path
import os
import sys


def extract_palette(image_path, k=5, seed=42):
    """Extract k dominant colors from an image using k-means clustering.

    Returns:
        colors: (k, 3) array of RGB values, sorted by proportion (largest first)
        proportions: (k,) array of proportions for each color
    """
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img).reshape(-1, 3).astype(np.float64)

    np.random.seed(seed)
    centroids, labels = kmeans2(pixels, k, minit='points', iter=20)

    colors = centroids.astype(np.uint8)
    counts = np.bincount(labels, minlength=k)
    proportions = counts / counts.sum()

    # Sort by proportion, largest first
    order = np.argsort(-proportions)
    colors = colors[order]
    proportions = proportions[order]

    return colors, proportions


def make_palette_png(colors, proportions, output_path, source_name=""):
    """Generate a palette swatch PNG showing colors, RGB values, and proportions."""
    swatch_w = 120
    swatch_h = 100
    padding = 15
    text_h = 60
    n = len(colors)

    total_w = padding + n * (swatch_w + padding)
    total_h = padding + swatch_h + text_h + padding + 30  # extra for title

    img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    title = f"Palette from: {source_name}" if source_name else "Extracted Palette"
    draw.text((padding, padding), title, fill=(0, 0, 0))

    y_top = padding + 25

    for i, (color, prop) in enumerate(zip(colors, proportions)):
        x = padding + i * (swatch_w + padding)

        # Draw color swatch
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        draw.rectangle([x, y_top, x + swatch_w, y_top + swatch_h],
                       fill=(r, g, b), outline=(0, 0, 0))

        # Label below swatch
        rgb_text = f"({r}, {g}, {b})"
        prop_text = f"{prop*100:.1f}%"
        draw.text((x, y_top + swatch_h + 5), rgb_text, fill=(0, 0, 0))
        draw.text((x, y_top + swatch_h + 22), prop_text, fill=(100, 100, 100))

    img.save(output_path)
    print(f"Palette saved to: {output_path}")
    return output_path


def make_bitmap(colors, proportions, grid_size=10, tile_size=20, seed=42):
    """Generate a single bitmap mosaic from colors and proportions.

    Args:
        colors: (k, 3) RGB array
        proportions: (k,) proportion for each color
        grid_size: number of tiles per side
        tile_size: pixel size of each tile
        seed: random seed for tile shuffling

    Returns:
        (H, W, 3) uint8 array
    """
    rng = np.random.RandomState(seed)
    n_tiles = grid_size * grid_size

    # Assign tiles to colors based on proportions
    tile_counts = np.round(proportions * n_tiles).astype(int)
    # Fix rounding so total = n_tiles
    diff = n_tiles - tile_counts.sum()
    if diff > 0:
        # Add extra tiles to the largest color
        tile_counts[0] += diff
    elif diff < 0:
        # Remove excess from the largest color
        tile_counts[0] += diff

    # Build tile color array
    tile_colors = []
    for i, count in enumerate(tile_counts):
        tile_colors.extend([colors[i]] * count)
    tile_colors = np.array(tile_colors)

    # Randomly shuffle
    rng.shuffle(tile_colors)

    # Build image
    img_size = grid_size * tile_size
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for idx in range(n_tiles):
        row = idx // grid_size
        col = idx % grid_size
        y = row * tile_size
        x = col * tile_size
        img[y:y+tile_size, x:x+tile_size] = tile_colors[idx]

    return img


def rgb_to_lab(rgb):
    """Convert RGB (0-255) to CIELAB."""
    # Normalize to 0-1
    rgb_norm = rgb.astype(np.float64) / 255.0

    # sRGB to linear RGB
    mask = rgb_norm > 0.04045
    rgb_lin = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # Linear RGB to XYZ (D65)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb_lin @ mat.T

    # XYZ to Lab
    ref = np.array([0.95047, 1.0, 1.08883])
    xyz_norm = xyz / ref

    mask = xyz_norm > 0.008856
    f = np.where(mask, xyz_norm ** (1/3), (903.3 * xyz_norm + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """Convert CIELAB to RGB (0-255)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    eps = 0.008856
    kappa = 903.3

    x = np.where(fx**3 > eps, fx**3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * eps, ((L + 16) / 116)**3, L / kappa)
    z = np.where(fz**3 > eps, fz**3, (116 * fz - 16) / kappa)

    ref = np.array([0.95047, 1.0, 1.08883])
    xyz = np.stack([x, y, z], axis=-1) * ref

    # XYZ to linear RGB
    mat_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]])
    rgb_lin = xyz @ mat_inv.T

    # Linear RGB to sRGB
    rgb_lin = np.clip(rgb_lin, 0, 1)
    mask = rgb_lin > 0.0031308
    srgb = np.where(mask, 1.055 * rgb_lin ** (1/2.4) - 0.055, 12.92 * rgb_lin)

    return np.clip(srgb * 255, 0, 255).astype(np.uint8)


def equalize_luminance(colors):
    """Set all colors to the same L* (mean L*), preserving a* and b*.

    Args:
        colors: (k, 3) RGB array
    Returns:
        (k, 3) RGB array with equalized L*
    """
    lab = rgb_to_lab(colors.astype(np.float64))
    mean_L = lab[:, 0].mean()
    lab[:, 0] = mean_L
    return lab_to_rgb(lab)


def boost_chroma(colors, factor=1.5):
    """Increase chroma (a*, b*) while keeping L* the same.

    Args:
        colors: (k, 3) RGB array
        factor: multiplier for a* and b* channels (>1 = more saturated)
    Returns:
        (k, 3) RGB array with boosted chroma, same luminance
    """
    lab = rgb_to_lab(colors.astype(np.float64))
    lab[:, 1] *= factor  # boost a*
    lab[:, 2] *= factor  # boost b*
    return lab_to_rgb(lab)


def generate_trial(colors, base_proportions, shift_amount,
                   grid_size=10, tile_size=20, seed=42):
    """Generate a 4AFC trial: 3 distractors + 1 oddball.

    The oddball has shifted proportions. The shift is applied by moving
    probability mass between the two largest-proportion colors.

    Args:
        colors: (k, 3) RGB array
        base_proportions: (k,) distractor proportions
        shift_amount: how much to shift the oddball (fraction, e.g. 0.15)
        grid_size: tiles per side
        tile_size: pixels per tile
        seed: base seed

    Returns:
        distractors: list of 3 image arrays
        oddball: 1 image array
        oddball_proportions: the shifted proportions used
    """
    rng_base = np.random.RandomState(seed)

    # Generate 3 distractors with base proportions, different shuffles
    distractors = []
    for i in range(3):
        s = rng_base.randint(0, 100000)
        distractors.append(make_bitmap(colors, base_proportions,
                                        grid_size, tile_size, seed=s))

    # Create oddball proportions by shifting mass
    oddball_props = base_proportions.copy()
    # Shift from color 0 to color 1
    oddball_props[0] -= shift_amount
    oddball_props[1] += shift_amount
    # Clamp and renormalize
    oddball_props = np.clip(oddball_props, 0.02, 1.0)
    oddball_props /= oddball_props.sum()

    s = rng_base.randint(0, 100000)
    oddball = make_bitmap(colors, oddball_props, grid_size, tile_size, seed=s)

    return distractors, oddball, oddball_props


def main():
    parser = argparse.ArgumentParser(description="Bitmap trial generator")
    parser.add_argument('--image', type=str, required=True,
                       help='Path to source image')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of colors to extract (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory')
    parser.add_argument('--extract-palette', action='store_true',
                       help='Extract and save palette PNG')
    parser.add_argument('--generate-trial', action='store_true',
                       help='Generate trial bitmaps')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Tiles per side (default: 10)')
    parser.add_argument('--tile-size', type=int, default=20,
                       help='Pixels per tile (default: 20)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_name = Path(args.image).stem

    # Extract palette
    colors, proportions = extract_palette(args.image, k=args.k, seed=args.seed)

    print(f"\nExtracted {args.k} colors from {args.image}:")
    for i, (c, p) in enumerate(zip(colors, proportions)):
        print(f"  Color {i+1}: RGB({c[0]}, {c[1]}, {c[2]}) — {p*100:.1f}%")

    if args.extract_palette:
        palette_path = os.path.join(args.output_dir, f"palette_{source_name}_k{args.k}.png")
        make_palette_png(colors, proportions, palette_path, source_name)

    if args.generate_trial:
        difficulties = {
            'easy':   0.30,
            'medium': 0.20,
            'hard':   0.10,
        }

        for diff_name, shift in difficulties.items():
            distractors, oddball, odd_props = generate_trial(
                colors, proportions, shift,
                grid_size=args.grid_size, tile_size=args.tile_size,
                seed=args.seed
            )

            trial_dir = os.path.join(args.output_dir,
                                      f"trial_{source_name}_{diff_name}")
            os.makedirs(trial_dir, exist_ok=True)

            # Save original versions
            Image.fromarray(oddball).save(
                os.path.join(trial_dir, "part1_oddball_original.png"))
            for j, d in enumerate(distractors):
                Image.fromarray(d).save(
                    os.path.join(trial_dir, f"part{j+2}_distractor_original.png"))

            # Save luminance-controlled versions
            lum_colors = equalize_luminance(colors)
            lum_distractors, lum_oddball, _ = generate_trial(
                lum_colors, proportions, shift,
                grid_size=args.grid_size, tile_size=args.tile_size,
                seed=args.seed
            )
            Image.fromarray(lum_oddball).save(
                os.path.join(trial_dir, "part1_oddball_lum_equalized.png"))
            for j, d in enumerate(lum_distractors):
                Image.fromarray(d).save(
                    os.path.join(trial_dir, f"part{j+2}_distractor_lum_equalized.png"))

            # Save luminance palette too
            lum_palette_path = os.path.join(trial_dir, f"palette_lum_equalized.png")
            make_palette_png(lum_colors, proportions, lum_palette_path,
                           f"{source_name} (L*-equalized)")

            # Save chroma-boosted + L*-equalized versions
            chroma_lum_colors = boost_chroma(colors, factor=1.5)
            chroma_lum_colors = equalize_luminance(chroma_lum_colors)
            cl_distractors, cl_oddball, _ = generate_trial(
                chroma_lum_colors, proportions, shift,
                grid_size=args.grid_size, tile_size=args.tile_size,
                seed=args.seed
            )
            Image.fromarray(cl_oddball).save(
                os.path.join(trial_dir, "part1_oddball_chroma_boost_lum_eq.png"))
            for j, d in enumerate(cl_distractors):
                Image.fromarray(d).save(
                    os.path.join(trial_dir, f"part{j+2}_distractor_chroma_boost_lum_eq.png"))

            # Save chroma-boosted + L*-equalized palette
            cl_palette_path = os.path.join(trial_dir, f"palette_chroma_boost_lum_eq.png")
            make_palette_png(chroma_lum_colors, proportions, cl_palette_path,
                           f"{source_name} (chroma 1.5x + L*-eq)")

            # Save trial info
            info_path = os.path.join(trial_dir, "trial_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Source: {args.image}\n")
                f.write(f"Colors (k={args.k}):\n")
                for i, (c, p) in enumerate(zip(colors, proportions)):
                    f.write(f"  Color {i+1}: RGB({c[0]}, {c[1]}, {c[2]}) — {p*100:.1f}%\n")
                f.write(f"\nDifficulty: {diff_name} (shift={shift})\n")
                f.write(f"Distractor proportions: {[f'{p:.3f}' for p in proportions]}\n")
                f.write(f"Oddball proportions:    {[f'{p:.3f}' for p in odd_props]}\n")
                f.write(f"Grid: {args.grid_size}x{args.grid_size}, Tile: {args.tile_size}px\n")
                f.write(f"Seed: {args.seed}\n")

            print(f"\n{diff_name.upper()} trial saved to {trial_dir}/")
            print(f"  Distractor props: {[f'{p:.1%}' for p in proportions]}")
            print(f"  Oddball props:    {[f'{p:.1%}' for p in odd_props]}")


if __name__ == '__main__':
    main()
