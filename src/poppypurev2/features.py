"""Image-level HSV/Lab, uniform LBP and normalized GLCM descriptors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from poppypurev2.acquisition import sha256_file
from poppypurev2.evaluation import encoded, freeze_protocol
from poppypurev2.preprocessing import crop_box, decode_rgb, detail_tiles, resize_rgb


def descriptor_names(config: dict) -> list[str]:
    names = [
        f"{space}_{channel}_bin_{index:02d}"
        for space, channels in (("hsv", "hsv"), ("lab", "Lab"))
        for channel in channels
        for index in range(config["colour"]["bins_per_channel"])
    ]
    for points, radius in config["lbp"]["points_radii"]:
        names.extend(f"lbp_p{points}_r{radius}_bin_{i:02d}" for i in range(points + 2))
    glcm = config["glcm"]
    names.extend(
        f"glcm_{prop}_d{distance}_a{angle}"
        for prop in glcm["properties"]
        for distance in glcm["distances"]
        for angle in glcm["angles_degrees"]
    )
    return names


def image_descriptor(image: Image.Image, config: dict) -> np.ndarray:
    """No padding/masked pixels are introduced; input is the valid RGB region."""
    if image.mode != "RGB" or min(image.size) <= 4:
        raise ValueError("Expected RGB region with both dimensions > 4")
    if config["mask"] != "none":
        raise ValueError("Only visually approved unmasked descriptors are enabled")
    rgb = np.asarray(image, dtype=np.float64) / 255
    hsv, lab = rgb2hsv(rgb), rgb2lab(rgb)
    bins = config["colour"]["bins_per_channel"]
    features = []
    for values, ranges in (
        (hsv, [(0, 1)] * 3),
        (lab, [(0, 100), (-128, 128), (-128, 128)]),
    ):
        for channel, limits in enumerate(ranges):
            histogram, _ = np.histogram(values[..., channel], bins=bins, range=limits)
            if not histogram.sum():
                raise ValueError("Empty colour histogram")
            features.extend(histogram / histogram.sum())
    grey = np.rint(rgb2gray(rgb) * 255).astype(np.uint8)
    for points, radius in config["lbp"]["points_radii"]:
        if 2 * radius >= min(grey.shape):
            raise ValueError("Region too small for configured LBP radius")
        lbp = local_binary_pattern(grey, points, radius, method="uniform")
        interior = lbp[radius:-radius, radius:-radius]
        histogram = np.bincount(interior.astype(np.int64).ravel(), minlength=points + 2)
        if len(histogram) != points + 2:
            raise ValueError("Unexpected LBP bins")
        features.extend(histogram / histogram.sum())
    glcm = config["glcm"]
    quantized = (grey.astype(np.uint16) * glcm["levels"] // 256).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=glcm["distances"],
        angles=np.deg2rad(glcm["angles_degrees"]),
        levels=glcm["levels"],
        symmetric=True,
        normed=True,
    )
    for prop in glcm["properties"]:
        features.extend(graycoprops(matrix, prop).ravel())
    result = np.asarray(features, dtype=np.float32)
    if len(result) != len(descriptor_names(config)) or not np.isfinite(result).all():
        raise ValueError("Invalid descriptor dimensions or values")
    return result


def feature_views(image: Image.Image, preprocessing: dict, config: dict) -> dict:
    cropped = image.crop(crop_box(image.size, preprocessing["crop_fraction_ltrb"]))
    descriptors = np.stack(
        [
            image_descriptor(tile, config)
            for tile, _ in detail_tiles(cropped, preprocessing["tile_size"])
        ]
    )
    return {
        "whole": image_descriptor(
            resize_rgb(image, preprocessing["long_side"]), config
        ),
        "crop": image_descriptor(
            resize_rgb(cropped, preprocessing["long_side"]), config
        ),
        "crop_detail": np.concatenate(
            (descriptors.mean(axis=0), descriptors.std(axis=0))
        ),
    }


def extract_training(project_root: Path, output: Path) -> dict:
    return extract_partition(project_root, output, "train")


def extract_partition(project_root: Path, output: Path, partition: str) -> dict:
    """Development partitions only; final-test extraction is a separate gate."""
    if partition not in {"train", "validation"}:
        raise ValueError("Feature development allows train or validation only")
    if output.exists():
        raise FileExistsError(f"Feature output already exists: {output}")
    frozen = freeze_protocol(project_root, "v4")
    config_path = project_root / "configs/features-v1.json"
    config = json.loads(config_path.read_text())
    preprocessing_path = project_root / config["preprocessing_config"]
    preprocessing = json.loads(preprocessing_path.read_text())
    if partition == "validation":
        training = json.loads(
            (project_root / "data/features/v1/summary.json").read_text()
        )
        if (
            training["feature_config_sha256"] != sha256_file(config_path)
            or training["preprocessing_config_sha256"]
            != sha256_file(preprocessing_path)
            or training["split_sha256"] != frozen["artifacts"]["assignments.csv"]
        ):
            raise ValueError(
                "Validation definitions differ from saved training features"
            )
    with (project_root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        assignments = [r for r in csv.DictReader(handle) if r["partition"] == partition]
    with (project_root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    matrices = {name: [] for name in config["variants"]}
    for index, row in enumerate(assignments, start=1):
        source = manifest[row["image_id"]]
        path = project_root / "data/raw" / source["source_path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_id']}")
        views = feature_views(decode_rgb(path), preprocessing, config)
        for name in matrices:
            matrices[name].append(views[name])
        if index % 32 == 0:
            print(f"{partition} features: {index}/{len(assignments)}", flush=True)
    arrays = {name: np.stack(rows) for name, rows in matrices.items()}
    names = descriptor_names(config)
    schemas = {
        "whole": names,
        "crop": names,
        "crop_detail": [f"mean_{n}" for n in names] + [f"std_{n}" for n in names],
    }
    output.mkdir(parents=True)
    np.savez_compressed(
        output / f"{partition}.npz",
        **arrays,
        image_ids=np.array([r["image_id"] for r in assignments]),
        volume_percent=np.array([int(r["volume_percent"]) for r in assignments]),
        preparation_ids=np.array([r["preparation_group_id"] for r in assignments]),
    )
    (output / "schema.json").write_bytes(encoded(schemas))
    summary = {
        "partition": partition,
        "images": len(assignments),
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "all_finite": all(bool(np.isfinite(array).all()) for array in arrays.values()),
        "mask": "none",
        "scaled": False,
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "feature_config_sha256": sha256_file(config_path),
        "preprocessing_config_sha256": sha256_file(preprocessing_path),
        "artifacts": {
            name: sha256_file(output / name)
            for name in (f"{partition}.npz", "schema.json")
        },
    }
    (output / "summary.json").write_bytes(encoded(summary))
    return summary
