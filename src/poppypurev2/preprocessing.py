"""Deterministic, metadata-free RGB inputs and training-only visual QC."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.evaluation import encoded, freeze_protocol


def decode_rgb(path: Path) -> Image.Image:
    """Apply orientation once, then discard EXIF and all ancillary metadata."""
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        clean = Image.new("RGB", oriented.size)
        clean.paste(oriented)
    return clean


def crop_box(size: tuple[int, int], fractions: list[float]) -> tuple[int, ...]:
    if len(fractions) != 4:
        raise ValueError("Expected left/top/right/bottom fractions")
    left, top, right, bottom = fractions
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Invalid crop fractions")
    width, height = size
    box = (
        round(width * left),
        round(height * top),
        round(width * right),
        round(height * bottom),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Empty crop at this image size")
    return box


def resize_rgb(image: Image.Image, long_side: int) -> Image.Image:
    if long_side <= 0:
        raise ValueError("long_side must be positive")
    result = image.copy()
    result.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return result


def square_input(image: Image.Image, side: int) -> tuple[Image.Image, tuple[int, ...]]:
    """Letterbox without distortion; return content bounds to exclude padding."""
    resized = resize_rgb(image, side)
    left, top = (side - resized.width) // 2, (side - resized.height) // 2
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized.width, top + resized.height)


def detail_tiles(image: Image.Image, tile_size: int) -> list[tuple[Image.Image, tuple]]:
    """Five deterministic source regions; caller retains image-level membership."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    width, height = min(tile_size, image.width), min(tile_size, image.height)
    dx, dy = image.width - width, image.height - height
    locations = dict.fromkeys([(0, 0), (dx, 0), (0, dy), (dx, dy), (dx // 2, dy // 2)])
    return [
        (image.crop((x, y, x + width, y + height)), (x, y, x + width, y + height))
        for x, y in locations
    ]


def process_image(
    path: Path, config: dict, variant: str = "crop"
) -> tuple[Image.Image, dict]:
    if variant not in {"whole", "crop"}:
        raise ValueError("Expected whole or crop variant")
    rgb = decode_rgb(path)
    fractions = config["crop_fraction_ltrb"] if variant == "crop" else [0, 0, 1, 1]
    box = crop_box(rgb.size, fractions)
    result = resize_rgb(rgb.crop(box), config["long_side"])
    return result, {
        "original_size": list(rgb.size),
        "crop_box": list(box),
        "output_size": list(result.size),
        "variant": variant,
    }


def training_qc(project_root: Path) -> dict:
    """Process train only; never decode validation/test images for design QC."""
    frozen = freeze_protocol(project_root, "v4")
    config_path = project_root / "configs/preprocessing-v1.json"
    config = json.loads(config_path.read_text())
    with (project_root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        assignments = [r for r in csv.DictReader(handle) if r["partition"] == "train"]
    with (project_root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    strata = defaultdict(list)
    for row in assignments:
        strata[
            (int(row["volume_percent"]), row["background"], row["magnification"])
        ].append(row)
    selected_ids = set()
    for rows in strata.values():
        ordered = sorted(rows, key=lambda r: r["image_id"])
        selected_ids.add(ordered[len(ordered) // 2]["image_id"])
    report = project_root / "reports/phase3"
    report.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for background in ("black", "white"):
        for magnification in ("4x", "micro"):
            canvas = Image.new("RGB", (1200, 7 * 340 + 50), "#eeeeee")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 10),
                f"TRAIN ONLY {background} {magnification}: "
                "original + crop box | cropped 768 | native detail",
                fill="black",
            )
            sheets[(background, magnification)] = canvas
    records = []
    for row in assignments:
        source = manifest[row["image_id"]]
        path = project_root / "data/raw" / source["source_path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_id']}")
        output, details = process_image(path, config)
        if output.mode != "RGB" or output.info:
            raise ValueError("Unexpected colour mode or retained metadata")
        record = {
            "image_id": row["image_id"],
            "partition": "train",
            "preparation_group_id": row["preparation_group_id"],
            "volume_percent": row["volume_percent"],
            "background": row["background"],
            "magnification": row["magnification"],
            **details,
            "output_pixel_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "visually_reviewed": row["image_id"] in selected_ids,
        }
        records.append(record)
        if row["image_id"] not in selected_ids:
            continue
        original = decode_rgb(path)
        marked = original.copy()
        ImageDraw.Draw(marked).rectangle(
            tuple(details["crop_box"]),
            outline="red",
            width=max(3, original.width // 200),
        )
        cropped_native = original.crop(tuple(details["crop_box"]))
        detail = detail_tiles(cropped_native, config["tile_size"])[-1][0]
        canvas = sheets[(row["background"], row["magnification"])]
        index = [0, 10, 20, 30, 40, 50, 100].index(int(row["volume_percent"]))
        for column, image in enumerate((marked, output, detail)):
            preview = resize_rgb(image, 300)
            canvas.paste(preview, (column * 400 + 5, 50 + index * 340))
        ImageDraw.Draw(canvas).text(
            (5, 50 + index * 340 + 305),
            f"{row['image_id']} | {row['volume_percent']}% by volume | TRAIN",
            fill="black",
        )
    for (background, magnification), canvas in sheets.items():
        canvas.save(report / f"crop_review_{background}_{magnification}.jpg")
    write_csv(report / "training_preprocessing.csv", records)
    summary = {
        "status": "initial_processing_candidates_ready_for_review",
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "preprocessing_config_sha256": sha256_file(config_path),
        "images_processed": len(records),
        "partitions_processed": ["train"],
        "visual_review_ids": sorted(selected_ids),
        "input_integrity": "All processed files match frozen manifest SHA-256",
        "mask": "Not enabled; prototype and review separately before selection",
        "metadata": "EXIF used for orientation only, then discarded",
    }
    (report / "summary.json").write_bytes(encoded(summary))
    return summary
