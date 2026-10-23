"""Training-only background-mask experiments; no mask enabled by default."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.evaluation import encoded, freeze_protocol
from poppypurev2.preprocessing import process_image, resize_rgb


def prototype_masks(image: Image.Image) -> tuple[dict[str, np.ndarray], dict]:
    """Compare intensity and colour; infer polarity from pixels, not folder tags."""
    rgb = np.asarray(image, dtype=np.float64) / 255
    light = rgb2lab(rgb)[..., 0] / 100
    saturation = rgb2hsv(rgb)[..., 1]
    light_threshold = float(threshold_otsu(light))
    saturation_threshold = float(threshold_otsu(saturation))
    # Border medians fail when grains fill the edges. Inspect the dark tail
    # across the whole crop instead, without using folder/background labels.
    dark_tail = float(np.quantile(light, 0.05))
    dark_background = dark_tail < 0.20
    light_mask = light > light_threshold if dark_background else light < light_threshold
    colour_mask = light_mask if dark_background else saturation > saturation_threshold
    masks = {
        "lightness_otsu": closing(light_mask, footprint=disk(1), mode="ignore"),
        "colour_or_lightness": closing(colour_mask, footprint=disk(1), mode="ignore"),
    }
    return masks, {
        "lightness_threshold": light_threshold,
        "saturation_threshold": saturation_threshold,
        "lightness_fifth_percentile": dark_tail,
        "polarity_from_pixels": "dark" if dark_background else "light",
    }


def removal_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Show removed pixels in red without inventing image content."""
    rgb = np.asarray(image, dtype=np.float64)
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Expected matching boolean mask")
    rgb[~mask] = 0.35 * rgb[~mask] + 0.65 * np.array([255, 0, 0])
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))


def mask_review(project_root: Path) -> dict:
    frozen = freeze_protocol(project_root, "v4")
    preprocessing_path = project_root / "configs/preprocessing-v1.json"
    config = json.loads(preprocessing_path.read_text())
    with (project_root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        train = [r for r in csv.DictReader(handle) if r["partition"] == "train"]
    with (project_root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    selected = []
    for label in [0, 10, 20, 30, 40, 50, 100]:
        for background in ("black", "white"):
            for magnification in ("4x", "micro"):
                rows = sorted(
                    [
                        r
                        for r in train
                        if int(r["volume_percent"]) == label
                        and r["background"] == background
                        and r["magnification"] == magnification
                    ],
                    key=lambda r: r["image_id"],
                )
                representative = rows[len(rows) // 2]
                alternatives = [r for r in rows if r != representative]
                low_sharpness = min(
                    alternatives,
                    key=lambda r: float(manifest[r["image_id"]]["sharpness_512"]),
                )
                selected.extend(
                    [
                        (representative, "representative"),
                        (low_sharpness, "low_sharpness"),
                    ]
                )
    destination = project_root / "reports/phase3/masks"
    destination.mkdir(parents=True, exist_ok=True)
    canvases = {}
    records = []
    for row, selection in selected:
        source = manifest[row["image_id"]]
        path = project_root / "data/raw" / source["source_path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_id']}")
        image, _ = process_image(path, config)
        masks, thresholds = prototype_masks(image)
        key = row["background"], row["magnification"], selection
        if key not in canvases:
            canvases[key] = Image.new("RGB", (1080, 7 * 300 + 50), "#eeeeee")
            ImageDraw.Draw(canvases[key]).text(
                (8, 8),
                "TRAIN: crop | lightness Otsu | colour/lightness. RED = removed pixels",
                fill="black",
            )
        index = [0, 10, 20, 30, 40, 50, 100].index(int(row["volume_percent"]))
        for column, preview in enumerate(
            [image, *[removal_overlay(image, mask) for mask in masks.values()]]
        ):
            canvases[key].paste(
                resize_rgb(preview, 260), (column * 360 + 5, index * 300 + 50)
            )
        ImageDraw.Draw(canvases[key]).text(
            (5, index * 300 + 325),
            f"{row['image_id']} {row['volume_percent']}% {selection}",
            fill="black",
        )
        for name, mask in masks.items():
            records.append(
                {
                    "image_id": row["image_id"],
                    "partition": "train",
                    "volume_percent": row["volume_percent"],
                    "background": row["background"],
                    "magnification": row["magnification"],
                    "selection": selection,
                    "method": name,
                    "retained_pixel_fraction": float(mask.mean()),
                    **thresholds,
                }
            )
    for (background, magnification, selection), canvas in canvases.items():
        canvas.save(destination / f"{background}_{magnification}_{selection}.jpg")
    write_csv(destination / "measurements.csv", records)
    summary = {
        "status": "awaiting_visual_review",
        "partition": "train",
        "images": len(selected),
        "methods": ["lightness_otsu", "colour_or_lightness"],
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "preprocessing_config_sha256": sha256_file(preprocessing_path),
        "image_ids": sorted({r["image_id"] for r, _ in selected}),
        "mask_enabled": False,
        "pixel_fraction_meaning": (
            "Visible mask coverage only; not adulteration by volume"
        ),
    }
    (destination / "summary.json").write_bytes(encoded(summary))
    return summary
