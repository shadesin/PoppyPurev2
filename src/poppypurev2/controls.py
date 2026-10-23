"""Diagnostic inputs for metadata and low-gradient appearance controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv, rgb2lab

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.evaluation import encoded, freeze_protocol
from poppypurev2.masks import removal_overlay
from poppypurev2.preprocessing import process_image, resize_rgb


def metadata_inputs(rows: list[dict]) -> np.ndarray:
    """Only explicitly allowed fields; never image/class paths or preparation IDs."""
    return np.array(
        [
            [
                float(r["width"]),
                float(r["height"]),
                r["capture_date"],
                r["background"],
                r["magnification"],
            ]
            for r in rows
        ],
        dtype=object,
    )


def low_gradient_features(
    image: Image.Image, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Select low-gradient pixels as an appearance control, not a material mask."""
    rgb = np.asarray(image, dtype=np.float64) / 255
    grey = np.asarray(image.convert("L"), dtype=np.float64) / 255
    dy, dx = np.gradient(grey)
    gradient = np.hypot(dx, dy)
    border = config["border_exclusion_pixels"]
    valid = np.zeros(grey.shape, dtype=bool)
    valid[border:-border, border:-border] = True
    threshold = np.quantile(gradient[valid], config["gradient_quantile"])
    selected = valid & (gradient <= threshold)
    features = []
    for values, limits in (
        (rgb2hsv(rgb), [(0, 1)] * 3),
        (rgb2lab(rgb), [(0, 100), (-128, 128), (-128, 128)]),
    ):
        for channel, bounds in enumerate(limits):
            histogram, _ = np.histogram(
                values[..., channel][selected], bins=config["colour_bins"], range=bounds
            )
            if not histogram.sum():
                raise ValueError("Empty low-gradient control selection")
            features.extend(histogram / histogram.sum())
    return np.asarray(features, dtype=np.float32), selected


def prepare_controls(project_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Control output already exists: {output}")
    frozen = freeze_protocol(project_root, "v4")
    config_path = project_root / "configs/phase4-v1.json"
    config = json.loads(config_path.read_text())["low_gradient_control"]
    preprocessing = json.loads(
        (project_root / "configs/preprocessing-v1.json").read_text()
    )
    preprocessing["long_side"] = config["long_side"]
    with (project_root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        assigned = [
            r
            for r in csv.DictReader(handle)
            if r["partition"] in {"train", "validation"}
        ]
    with (project_root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    sample_ids = set(
        json.loads((project_root / "reports/phase3/summary.json").read_text())[
            "visual_review_ids"
        ]
    )
    matrices = {"train": [], "validation": []}
    ids = {"train": [], "validation": []}
    rows = []
    canvases = {}
    for row in assigned:
        source = manifest[row["image_id"]]
        path = project_root / "data/raw" / source["source_path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_id']}")
        image, _ = process_image(path, preprocessing)
        descriptor, mask = low_gradient_features(image, config)
        matrices[row["partition"]].append(descriptor)
        ids[row["partition"]].append(row["image_id"])
        rows.append(
            {
                "image_id": row["image_id"],
                "partition": row["partition"],
                "selected_fraction": float(mask.mean()),
            }
        )
        if row["image_id"] not in sample_ids:
            continue
        key = row["background"], row["magnification"]
        if key not in canvases:
            canvases[key] = Image.new("RGB", (640, 7 * 230 + 40), "#eeeeee")
            ImageDraw.Draw(canvases[key]).text(
                (5, 5),
                "TRAIN: crop | low-gradient selection (red = excluded)",
                fill="black",
            )
        top = 40 + [0, 10, 20, 30, 40, 50, 100].index(int(row["volume_percent"])) * 230
        for column, preview in enumerate((image, removal_overlay(image, mask))):
            canvases[key].paste(resize_rgb(preview, 200), (column * 320 + 5, top))
        ImageDraw.Draw(canvases[key]).text(
            (5, top + 205),
            f"{row['image_id']} {row['volume_percent']}% TRAIN",
            fill="black",
        )
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "low_gradient.npz",
        **{partition: np.stack(matrix) for partition, matrix in matrices.items()},
        **{f"{partition}_ids": np.array(values) for partition, values in ids.items()},
    )
    for (background, magnification), canvas in canvases.items():
        canvas.save(output / f"review_{background}_{magnification}.jpg")
    write_csv(output / "selections.csv", rows)
    summary = {
        "kind": "low_gradient_appearance_control",
        "background_only": False,
        "partitions": {name: len(values) for name, values in ids.items()},
        "feature_width": 96,
        "selection_rule": config,
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "phase4_config_sha256": sha256_file(config_path),
        "artifact_sha256": sha256_file(output / "low_gradient.npz"),
        "note": (
            "Low-gradient pixels can include particle interiors; inspect training "
            "overlays and do not call this pure background"
        ),
    }
    (output / "summary.json").write_bytes(encoded(summary))
    return summary
