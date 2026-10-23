"""Synthetic retinal fundus photograph generator for offline testing and demonstration.

Generates realistic fundus photographs for:
1. Normal healthy retina
2. Cataract (diffuse lens opacification and contrast attenuation)
3. Diabetic Retinopathy (cotton wool spots, microaneurysms, hemorrhages)
4. Glaucoma (enlarged cup-to-disc ratio, optic nerve head excavation)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAMPLE_DIR = Path(__file__).resolve().parent


def draw_base_retina(size: Tuple[int, int] = (512, 512), seed: int = 42) -> Image.Image:
    """Draw a base fundus background with circular ocular mask, optic disc, macula, and vessel tree."""
    rng = np.random.RandomState(seed)
    w, h = size
    cx, cy = w // 2, h // 2
    r_fundus = int(min(w, h) * 0.46)

    # 1. Base dark background
    img = Image.new("RGB", (w, h), color=(10, 8, 8))
    draw = ImageDraw.Draw(img)

    # 2. Fundus orange-red gradient background
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    fundus_mask = dist_from_center <= r_fundus

    bg_arr = np.zeros((h, w, 3), dtype=np.uint8)
    bg_arr[:, :] = [12, 10, 10]

    # Radial color gradient (bright orange-red in center, darker brownish-red at periphery)
    radial_norm = np.clip(dist_from_center / r_fundus, 0.0, 1.0)
    red_ch = (180 - 80 * radial_norm + rng.normal(0, 3, (h, w))).clip(0, 255).astype(np.uint8)
    green_ch = (65 - 35 * radial_norm + rng.normal(0, 2, (h, w))).clip(0, 255).astype(np.uint8)
    blue_ch = (25 - 15 * radial_norm + rng.normal(0, 1, (h, w))).clip(0, 255).astype(np.uint8)

    bg_arr[fundus_mask, 0] = red_ch[fundus_mask]
    bg_arr[fundus_mask, 1] = green_ch[fundus_mask]
    bg_arr[fundus_mask, 2] = blue_ch[fundus_mask]

    img = Image.fromarray(bg_arr)
    draw = ImageDraw.Draw(img)

    # 3. Optic Disc (Nasal side, e.g. x = cx - 120, y = cy)
    disc_cx, disc_cy = cx - int(w * 0.22), cy
    disc_rx, disc_ry = int(w * 0.07), int(h * 0.08)
    draw.ellipse(
        [disc_cx - disc_rx, disc_cy - disc_ry, disc_cx + disc_rx, disc_cy + disc_ry],
        fill=(245, 205, 120),
        outline=(220, 160, 80),
        width=2,
    )
    # Optic cup (inner pale region)
    cup_rx, cup_ry = int(disc_rx * 0.38), int(disc_ry * 0.38)
    draw.ellipse(
        [disc_cx - cup_rx, disc_cy - cup_ry, disc_cx + cup_rx, disc_cy + cup_ry],
        fill=(255, 235, 175),
    )

    # 4. Macula & Fovea (Temporal side, e.g. x = cx + 80, y = cy)
    macula_cx, macula_cy = cx + int(w * 0.15), cy
    macula_r = int(w * 0.08)
    for r_step in range(macula_r, 0, -2):
        alpha = int(40 * (1.0 - r_step / macula_r))
        draw.ellipse(
            [macula_cx - r_step, macula_cy - r_step, macula_cx + r_step, macula_cy + r_step],
            fill=(100 - alpha, 30 - alpha // 2, 10),
        )
    # Foveal reflex dot
    draw.ellipse(
        [macula_cx - 3, macula_cy - 3, macula_cx + 3, macula_cy + 3],
        fill=(60, 15, 5),
    )

    # 5. Major Retinal Blood Vessels (Arcades branching from optic disc)
    vessel_color = (130, 20, 15)
    # Superior temporal arcade
    pts_sup_temp = [
        (disc_cx, disc_cy),
        (disc_cx + 40, disc_cy - 60),
        (disc_cx + 100, disc_cy - 120),
        (disc_cx + 180, disc_cy - 150),
        (disc_cx + 250, disc_cy - 140),
    ]
    draw.line(pts_sup_temp, fill=vessel_color, width=4)

    # Inferior temporal arcade
    pts_inf_temp = [
        (disc_cx, disc_cy),
        (disc_cx + 40, disc_cy + 60),
        (disc_cx + 100, disc_cy + 120),
        (disc_cx + 180, disc_cy + 150),
        (disc_cx + 250, disc_cy + 140),
    ]
    draw.line(pts_inf_temp, fill=vessel_color, width=4)

    # Nasal arcades
    pts_sup_nasal = [(disc_cx, disc_cy), (disc_cx - 50, disc_cy - 80), (disc_cx - 90, disc_cy - 130)]
    pts_inf_nasal = [(disc_cx, disc_cy), (disc_cx - 50, disc_cy + 80), (disc_cx - 90, disc_cy + 130)]
    draw.line(pts_sup_nasal, fill=vessel_color, width=3)
    draw.line(pts_inf_nasal, fill=vessel_color, width=3)

    # Slight Gaussian blur to smooth artificial edges
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def generate_normal_sample() -> Image.Image:
    """Generate healthy normal retinal fundus image."""
    return draw_base_retina(seed=101)


def generate_cataract_sample() -> Image.Image:
    """Generate cataract fundus image characterized by diffuse lens opacification and blur."""
    base = draw_base_retina(seed=202)
    # Heavy blur simulating lens scatter
    blurred = base.filter(ImageFilter.GaussianBlur(radius=4.5))

    # Color shift towards muddy yellowish-brown haze
    arr = np.array(blurred, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.85 + 25, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.75 + 20, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.50 + 5, 0, 255)

    # Contrast attenuation
    mean_val = np.mean(arr)
    arr = np.clip((arr - mean_val) * 0.65 + mean_val, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_diabetic_retinopathy_sample() -> Image.Image:
    """Generate diabetic retinopathy fundus image with exudates and microaneurysms."""
    base = draw_base_retina(seed=303)
    draw = ImageDraw.Draw(base)

    rng = np.random.RandomState(303)
    cx, cy = 256, 256

    # 1. Hard Exudates (bright yellowish lipid deposits)
    for _ in range(25):
        ex_x = cx + rng.randint(-80, 160)
        ex_y = cy + rng.randint(-120, 120)
        ex_r = rng.randint(2, 6)
        draw.ellipse([ex_x - ex_r, ex_y - ex_r, ex_x + ex_r, ex_y + ex_r], fill=(250, 245, 170))

    # 2. Cotton Wool Spots (soft, fuzzy whitish-yellow infarcts)
    for _ in range(6):
        cw_x = cx + rng.randint(-60, 140)
        cw_y = cy + rng.randint(-100, 100)
        cw_r = rng.randint(8, 14)
        draw.ellipse([cw_x - cw_r, cw_y - cw_r, cw_x + cw_r, cw_y + cw_r], fill=(225, 220, 195))

    # 3. Blot & Flame Hemorrhages (dark red irregular spots)
    for _ in range(35):
        h_x = cx + rng.randint(-110, 180)
        h_y = cy + rng.randint(-140, 140)
        h_r = rng.randint(2, 5)
        draw.ellipse([h_x - h_r, h_y - h_r, h_x + h_r, h_y + h_r], fill=(85, 8, 5))

    return base.filter(ImageFilter.GaussianBlur(radius=0.8))


def generate_glaucoma_sample() -> Image.Image:
    """Generate glaucoma fundus image with marked optic nerve cup enlargement."""
    base = draw_base_retina(seed=404)
    draw = ImageDraw.Draw(base)
    cx, cy = 256, 256
    disc_cx, disc_cy = cx - int(512 * 0.22), cy
    disc_rx, disc_ry = int(512 * 0.075), int(512 * 0.085)

    # Significantly enlarged optic cup (Cup-to-disc ratio > 0.8)
    cup_rx = int(disc_rx * 0.82)
    cup_ry = int(disc_ry * 0.82)
    draw.ellipse(
        [disc_cx - disc_rx, disc_cy - disc_ry, disc_cx + disc_rx, disc_cy + disc_ry],
        fill=(245, 210, 130),
        outline=(210, 150, 70),
        width=2,
    )
    draw.ellipse(
        [disc_cx - cup_rx, disc_cy - cup_ry, disc_cx + cup_rx, disc_cy + cup_ry],
        fill=(255, 248, 220),  # Marked pallor of excavation
    )
    # Vessel bending / bayoneting at cup margin
    draw.arc(
        [disc_cx - cup_rx - 4, disc_cy - cup_ry - 4, disc_cx + cup_rx + 4, disc_cy + cup_ry + 4],
        start=45,
        end=315,
        fill=(110, 15, 10),
        width=3,
    )

    return base.filter(ImageFilter.GaussianBlur(radius=0.9))


def create_all_samples(target_dir: Path = SAMPLE_DIR) -> None:
    """Generate and save all 4 representative synthetic fundus images."""
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "normal.png": generate_normal_sample(),
        "cataract.png": generate_cataract_sample(),
        "diabetic_retinopathy.png": generate_diabetic_retinopathy_sample(),
        "glaucoma.png": generate_glaucoma_sample(),
    }

    for filename, img in samples.items():
        out_path = target_dir / filename
        img.save(out_path)
        print(f"Generated sample fundus image: {out_path} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    create_all_samples()
