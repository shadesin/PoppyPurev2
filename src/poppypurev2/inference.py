"""Self-contained, versioned CNN inference using the packaged preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from poppypurev2.acquisition import sha256_file
from poppypurev2.deep import DeepClassifier, EmbeddingHead
from poppypurev2.deep_inputs import CLASSES, image_regions
from poppypurev2.preprocessing import decode_rgb


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Device must be auto, cpu or mps")
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError("Apple MPS is unavailable in this process; use --device cpu")
    return torch.device("mps" if requested != "cpu" and available else "cpu")


def load_bundle(directory: Path, device: str = "auto") -> tuple:
    metadata_path = directory / "bundle.json"
    if not metadata_path.is_file():
        raise ValueError(f"Model bundle is missing bundle.json: {directory}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata["schema_version"] != 1
        or metadata["percentage_basis"] != "volume"
        or metadata["classes_volume_percent"] != list(CLASSES)
    ):
        raise ValueError("Unsupported model bundle or volume class mapping")
    path = directory / "model.pt"
    if sha256_file(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Model checkpoint checksum does not match bundle.json")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint["classes_volume_percent"] != list(CLASSES)
        or checkpoint["model_version"] != metadata["model_version"]
        or checkpoint["specification"]
        != {
            "backbone": "resnet18",
            "head": "mlp",
            "representation": "crop_detail",
            "stage": "frozen",
        }
    ):
        raise ValueError("Checkpoint specification differs from packaged contract")
    state = checkpoint["model_state"]
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Nonfinite checkpoint tensor")
    if not torch.all(state["head.scale"] > 0):
        raise ValueError("Invalid saved embedding scaler")
    # Full weights are in model.pt; no pretrained download or dataset is needed.
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    head = EmbeddingHead(
        512,
        "mlp",
        state["head.mean"],
        state["head.scale"],
        checkpoint["config"]["head"],
    )
    model = DeepClassifier(encoder, head)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    actual_device = resolve_device(device)
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    return model.eval().to(actual_device), checkpoint, metadata, actual_device


def probability_rows(probabilities: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(CLASSES)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid seven-class probabilities")
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    values = [
        {str(label): float(p) for label, p in zip(CLASSES, row, strict=True)}
        for row in probabilities
    ]
    return predictions, values


def predict_image(directory: Path, path: Path, device: str = "auto") -> dict:
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    model, checkpoint, metadata, actual_device = load_bundle(directory, device)
    image = decode_rgb(path)
    try:
        regions = image_regions(
            image,
            "crop_detail",
            checkpoint["config"],
            checkpoint["source_preprocessing"],
        )
    except ValueError as error:
        raise ValueError(
            "Image is too small for five distinct detail regions after cropping; "
            "provide the original close-up image"
        ) from error
    with torch.inference_mode():
        probabilities = model(regions[None].to(actual_device)).softmax(1).cpu().numpy()
    predictions, values = probability_rows(probabilities)
    return {
        "model_version": metadata["model_version"],
        "image": str(path),
        "image_sha256": sha256_file(path),
        "percentage_basis": "volume",
        "predicted_volume_percent": int(predictions[0]),
        "adulterated": bool(predictions[0] > 0),
        "class_probabilities": values[0],
        "device": str(actual_device),
    }
