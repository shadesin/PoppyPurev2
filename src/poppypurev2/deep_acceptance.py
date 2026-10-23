"""Reproduce saved validation predictions and verify Phase 5 fitting boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from poppypurev2.acquisition import sha256_file
from poppypurev2.deep import evaluate_images, image_batch, load_checkpoint
from poppypurev2.deep_inputs import development_rows, load_settings
from poppypurev2.evaluation import encoded


def verify_phase5(root: Path, runs: Path) -> dict:
    config, source = load_settings(root)
    parts, manifest, _ = development_rows(root)
    summary = json.loads((runs / "summary.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    accepted = []
    for result in summary["runs"]:
        directory = runs / result["run_id"]
        for name, digest in result["artifacts"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Run artifact changed: {directory.name}/{name}")
        model, checkpoint = load_checkpoint(root, directory / "checkpoint.pt")
        with (directory / "validation_predictions.csv").open() as handle:
            saved = list(csv.DictReader(handle))
        assert [r["image_id"] for r in saved] == [
            r["image_id"] for r in parts["validation"]
        ]
        assert [int(r["true_volume_percent"]) for r in saved] == [
            int(r["volume_percent"]) for r in parts["validation"]
        ]
        probabilities = np.array(
            [
                [float(row[f"probability_{v}"]) for v in (0, 10, 20, 30, 40, 50, 100)]
                for row in saved
            ]
        )
        if result["stage"] == "frozen":
            cache = root / "data/embeddings/phase5-v1" / result["backbone"]
            info = json.loads((cache / "summary.json").read_text())
            for name, digest in info["artifacts"].items():
                assert sha256_file(cache / name) == digest
            arrays = {}
            for part in ("train", "validation"):
                with np.load(cache / f"{part}.npz", allow_pickle=False) as pack:
                    assert pack["image_ids"].tolist() == [
                        r["image_id"] for r in parts[part]
                    ]
                    assert pack["volume_percent"].tolist() == [
                        int(r["volume_percent"]) for r in parts[part]
                    ]
                    assert pack["preparation_ids"].tolist() == [
                        r["preparation_group_id"] for r in parts[part]
                    ]
                    arrays[part] = pack[result["representation"]].copy()
            scaler = StandardScaler().fit(arrays["train"])
            np.testing.assert_allclose(
                model.head.mean.numpy(), scaler.mean_.astype(np.float32), rtol=0, atol=0
            )
            np.testing.assert_allclose(
                model.head.scale.numpy(),
                scaler.scale_.astype(np.float32),
                rtol=0,
                atol=0,
            )
            with torch.inference_mode():
                reproduced = (
                    model.head(torch.from_numpy(arrays["validation"]))
                    .softmax(1)
                    .numpy()
                )
            np.testing.assert_allclose(reproduced, probabilities, rtol=1e-6, atol=1e-7)
            # Verify real RGB -> full saved encoder/head agrees with cached features.
            model.to(device)
            regions = image_batch(
                root,
                parts["validation"][:3],
                manifest,
                result["representation"],
                config,
                source,
            )
            with torch.inference_mode():
                end_to_end = model(regions.to(device)).softmax(1).cpu().numpy()
            np.testing.assert_allclose(
                end_to_end, probabilities[:3], rtol=1e-4, atol=1e-5
            )
        else:
            model.to(device)
            score = evaluate_images(
                model,
                root,
                parts["validation"],
                manifest,
                result["representation"],
                config,
                source,
                device,
            )
            reproduced = score["probabilities"]
            np.testing.assert_allclose(reproduced, probabilities, rtol=1e-4, atol=1e-5)
            initial = torch.load(
                runs / result["initialized_from"] / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )["model_state"]
            for name, value in checkpoint["model_state"].items():
                if (
                    (
                        name.startswith("encoder.")
                        and not name.startswith("encoder.layer4.")
                    )
                    or name.endswith(
                        ("running_mean", "running_var", "num_batches_tracked")
                    )
                    or name in ("head.mean", "head.scale")
                ):
                    assert torch.equal(value, initial[name]), (
                        f"Frozen value changed: {name}"
                    )
        predictions = np.array((0, 10, 20, 30, 40, 50, 100))[reproduced.argmax(1)]
        assert predictions.tolist() == [
            int(r["predicted_volume_percent"]) for r in saved
        ]
        with (directory / "learning_curve.csv").open() as handle:
            curve = list(csv.DictReader(handle))
        best = max(float(r["validation_macro_f1"]) for r in curve)
        earliest = next(
            int(r["epoch"]) for r in curve if float(r["validation_macro_f1"]) == best
        )
        assert result["selected_epoch"] == earliest == checkpoint["selected_epoch"]
        assert abs(best - result["validation_macro_f1"]) < 1e-12
        accepted.append(result["run_id"])
        print(
            f"Accepted persisted validation predictions: {result['run_id']}", flush=True
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    acceptance = {
        "status": "passed",
        "accepted_runs": accepted,
        "validation_images_per_run": 82,
        "test_used": False,
        "checks": [
            "artifact checksums",
            "IDs/labels/preparation membership",
            "training-only scaler",
            "checkpoint prediction reproduction",
            "real RGB inference agrees with cached embeddings",
            "earliest best epoch",
            "frozen blocks/BatchNorm/scaler unchanged",
        ],
    }
    (runs / "ACCEPTANCE.json").write_bytes(encoded(acceptance))
    return acceptance
