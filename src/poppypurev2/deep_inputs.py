"""Checked RGB inputs for pretrained encoders; development partitions only."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torchvision.models import resnet18, resnet50

from poppypurev2.acquisition import sha256_file
from poppypurev2.evaluation import freeze_protocol
from poppypurev2.preprocessing import crop_box, decode_rgb, detail_tiles

WEIGHTS = {
    "resnet18": ("resnet18-f37072fd.pth", "f37072fd", 512),
    "resnet50": ("resnet50-11ad3fa6.pth", "11ad3fa6", 2048),
}
CLASSES = (0, 10, 20, 30, 40, 50, 100)


def class_indices(labels) -> torch.Tensor:
    mapping = {volume: index for index, volume in enumerate(CLASSES)}
    return torch.tensor([mapping[int(label)] for label in labels], dtype=torch.long)


def development_rows(root: Path) -> tuple[dict, dict, dict]:
    frozen = freeze_protocol(root, "v4")
    with (root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    parts = {
        part: [r for r in rows if r["partition"] == part]
        for part in ("train", "validation")
    }
    assert [len(parts[p]) for p in ("train", "validation")] == [384, 82]
    return parts, manifest, frozen


def letterbox_tensor(image: Image.Image, config: dict) -> torch.Tensor:
    """Preserve aspect ratio; padding equals the exact floating channel mean."""
    side = config["input_size"]
    ratio = side / max(image.size)
    width, height = (max(1, round(v * ratio)) for v in image.size)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    mean = torch.tensor(config["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(config["std"], dtype=torch.float32)[:, None, None]
    canvas = mean.expand(3, side, side).clone()
    left, top = (side - width) // 2, (side - height) // 2
    pixels = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float()
    canvas[:, top : top + height, left : left + width] = pixels / 255
    return (canvas - mean) / std


def augment_image(
    image: Image.Image, rng: random.Random, settings: dict
) -> Image.Image:
    if settings["flips"]:
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        if rng.random() < 0.5:
            image = ImageOps.flip(image)
    if settings["rotation_quarter_turns"]:
        image = image.rotate(90 * rng.randrange(4), expand=True)
    for name, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
    ):
        image = enhancer(image).enhance(
            1 + rng.uniform(-settings[name], settings[name])
        )
    return image


def image_regions(
    image: Image.Image,
    representation: str,
    config: dict,
    source_config: dict,
    rng: random.Random | None = None,
) -> torch.Tensor:
    cropped = image.crop(crop_box(image.size, source_config["crop_fraction_ltrb"]))
    if rng is not None:
        cropped = augment_image(cropped, rng, config["fine_tuning"]["augmentation"])
    if representation == "crop":
        regions = [cropped]
    elif representation == "crop_detail":
        regions = [
            tile for tile, _ in detail_tiles(cropped, source_config["tile_size"])
        ]
        if len(regions) != 5:
            raise ValueError("Expected five distinct native detail regions")
    else:
        raise ValueError("Only crop and crop_detail are enabled")
    return torch.stack(
        [letterbox_tensor(region, config["preprocessing"]) for region in regions]
    )


def checked_image(root: Path, source: dict) -> Image.Image:
    path = root / "data/raw" / source["source_path"]
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"Raw image changed: {source['image_id']}")
    return decode_rgb(path)


def encoder_from_weights(root: Path, backbone: str) -> nn.Module:
    filename, prefix, _ = WEIGHTS[backbone]
    path = root / "models/pretrained" / filename
    if not sha256_file(path).startswith(prefix):
        raise ValueError(f"Pretrained weight checksum failed: {filename}")
    model = {"resnet18": resnet18, "resnet50": resnet50}[backbone](weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.requires_grad_(False)
    return model.eval()


def pooled_embedding(encoder: nn.Module, regions: torch.Tensor) -> torch.Tensor:
    """N images × R regions -> N embeddings, retaining image-level targets."""
    n, regions_per_image = regions.shape[:2]
    features = encoder(regions.flatten(0, 1))
    return features.reshape(n, regions_per_image, -1).mean(dim=1)


def load_settings(root: Path) -> tuple[dict, dict]:
    config = json.loads((root / "configs/phase5-v1.json").read_text())
    source = json.loads((root / config["preprocessing"]["source_config"]).read_text())
    if config["classes_volume_percent"] != list(CLASSES):
        raise ValueError("Volume label mapping changed")
    return config, source
