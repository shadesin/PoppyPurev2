"""Bounded Phase 5 transfer learning, checkpointing and validation reports."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.baselines import pick_best, subgroup_metrics
from poppypurev2.deep_inputs import (
    CLASSES,
    WEIGHTS,
    checked_image,
    class_indices,
    development_rows,
    encoder_from_weights,
    image_regions,
    load_settings,
    pooled_embedding,
)
from poppypurev2.evaluation import classification_metrics, encoded
from poppypurev2.records import make_record, source_fingerprint, write_record


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class EmbeddingHead(nn.Module):
    def __init__(self, dimension: int, kind: str, mean, scale, config: dict):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        if kind == "linear":
            self.network = nn.Linear(dimension, 7)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(dimension, config["hidden_units"]),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["hidden_units"], 7),
            )
        else:
            raise ValueError("Unknown head")

    def forward(self, features):
        return self.network((features - self.mean) / self.scale)


class DeepClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, head: EmbeddingHead):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, regions):
        return self.head(pooled_embedding(self.encoder, regions))


def fine_tuning_mode(model: DeepClassifier) -> None:
    model.train()
    model.encoder.requires_grad_(False)
    model.encoder.layer4.requires_grad_(True)
    # eval() fixes every BatchNorm statistic, but leaves layer4 gradients enabled.
    model.encoder.eval()
    model.head.requires_grad_(True)


def cpu_state(model: nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def score_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    logits, labels = logits.detach().cpu(), labels.cpu()
    probabilities = logits.softmax(dim=1).numpy()
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    true = np.asarray(CLASSES)[labels.numpy()]
    metrics = classification_metrics(true.tolist(), predictions.tolist())
    metrics["accuracy"] = float((true == predictions).mean())
    metrics["cross_entropy"] = float(nn.functional.cross_entropy(logits, labels))
    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": predictions,
        "true": true,
    }


def extract_embeddings(
    root: Path,
    backbone: str,
    representations: list[str],
    parts: dict,
    manifest: dict,
    config: dict,
    source: dict,
    device: torch.device,
    frozen: dict,
) -> tuple[dict, dict]:
    directory = root / "data/embeddings/phase5-v1" / backbone
    if directory.exists():
        raise FileExistsError(f"Embedding output exists: {directory}")
    directory.mkdir(parents=True)
    encoder = encoder_from_weights(root, backbone).to(device)
    result = {}
    start = time.perf_counter()
    for partition, rows in parts.items():
        matrices = {representation: [] for representation in representations}
        # Decode each image once; encode its overview and detail views separately.
        step = max(
            1,
            config["embedding_batch_size"]
            // (5 if "crop_detail" in representations else 1),
        )
        for offset in range(0, len(rows), step):
            images = [
                checked_image(root, manifest[r["image_id"]])
                for r in rows[offset : offset + step]
            ]
            for representation in representations:
                regions = torch.stack(
                    [
                        image_regions(image, representation, config, source)
                        for image in images
                    ]
                ).to(device)
                with torch.inference_mode():
                    features = pooled_embedding(encoder, regions).cpu().numpy()
                matrices[representation].append(features)
            if offset % (step * 8) == 0:
                print(
                    f"{backbone} embeddings {partition}: "
                    f"{min(offset + step, len(rows))}/{len(rows)}",
                    flush=True,
                )
        result[partition] = {
            name: np.concatenate(values) for name, values in matrices.items()
        }
        for matrix in result[partition].values():
            if (
                matrix.shape != (len(rows), WEIGHTS[backbone][2])
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Invalid encoder outputs")
        np.savez_compressed(
            directory / f"{partition}.npz",
            **result[partition],
            image_ids=np.array([r["image_id"] for r in rows]),
            volume_percent=np.array([int(r["volume_percent"]) for r in rows]),
            preparation_ids=np.array([r["preparation_group_id"] for r in rows]),
        )
    summary = {
        "backbone": backbone,
        "weights": config["backbones"][backbone]["weights"],
        "weights_sha256": sha256_file(
            root / "models/pretrained" / WEIGHTS[backbone][0]
        ),
        "config_sha256": sha256_file(root / "configs/phase5-v1.json"),
        "source_preprocessing_sha256": sha256_file(
            root / config["preprocessing"]["source_config"]
        ),
        "manifest_sha256": sha256_file(root / "data/manifest.csv"),
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "source_tree_sha256": source_fingerprint(root),
        "device": str(device),
        "representations": representations,
        "encoder_training": False,
        "requires_grad": False,
        "batchnorm_statistics": "frozen",
        "test_used": False,
        "shapes": {
            part: {name: list(a.shape) for name, a in arrays.items()}
            for part, arrays in result.items()
        },
        "seconds": time.perf_counter() - start,
        "artifacts": {
            f"{part}.npz": sha256_file(directory / f"{part}.npz") for part in parts
        },
    }
    (directory / "summary.json").write_bytes(encoded(summary))
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    return result, summary


def fit_head(
    x: np.ndarray,
    labels: torch.Tensor,
    xv: np.ndarray,
    validation_labels: torch.Tensor,
    kind: str,
    config: dict,
) -> tuple:
    seed_everything(config["seed"])
    scaler = StandardScaler().fit(x)
    head = EmbeddingHead(x.shape[1], kind, scaler.mean_, scaler.scale_, config["head"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config["head"]["learning_rate"],
        weight_decay=config["head"]["weight_decay"],
    )
    train_x, val_x = torch.from_numpy(x), torch.from_numpy(xv)
    curve, best_state, best_score, best_epoch = [], None, -1.0, 0
    generator = torch.Generator().manual_seed(config["seed"])
    for epoch in range(1, config["head"]["max_epochs"] + 1):
        head.train()
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(config["head"]["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(train_x[indices]), labels[indices])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            train_score = score_logits(head(train_x), labels)
            val_score = score_logits(head(val_x), validation_labels)
        row = curve_row(epoch, train_score, val_score)
        curve.append(row)
        if row["validation_macro_f1"] > best_score:
            best_score, best_epoch, best_state = (
                row["validation_macro_f1"],
                epoch,
                cpu_state(head),
            )
        if epoch - best_epoch >= config["head"]["patience"]:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        train_score = score_logits(head(train_x), labels)
        val_score = score_logits(head(val_x), validation_labels)
    return head, curve, best_epoch, train_score, val_score


def curve_row(epoch: int, training: dict, validation: dict) -> dict:
    return {
        "epoch": epoch,
        **{
            f"{part}_{name}": result["metrics"][name]
            for part, result in (("training", training), ("validation", validation))
            for name in ("cross_entropy", "macro_f1", "accuracy")
        },
    }


def save_run(
    root: Path,
    directory: Path,
    model: DeepClassifier,
    spec: dict,
    curve: list[dict],
    best_epoch: int,
    train_score: dict,
    val_score: dict,
    rows: list[dict],
    seconds: float,
    config: dict,
    frozen: dict,
    provenance: dict,
) -> dict:
    directory.mkdir()
    checkpoint = {
        "schema_version": 1,
        "specification": spec,
        "model_state": cpu_state(model),
        "config": config,
        "classes_volume_percent": list(CLASSES),
        "selected_epoch": best_epoch,
        "provenance": provenance,
    }
    torch.save(checkpoint, directory / "checkpoint.pt")
    # Verify serialization against every validation probability for the head/full model
    # in the separate acceptance pass; first ensure no tensor changed on save/load.
    loaded = torch.load(
        directory / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    for key, value in checkpoint["model_state"].items():
        if not torch.equal(value, loaded["model_state"][key]):
            raise ValueError("Checkpoint tensor persistence failed")
    predictions = []
    for row, true, predicted, probability in zip(
        rows,
        val_score["true"],
        val_score["predictions"],
        val_score["probabilities"],
        strict=True,
    ):
        predictions.append(
            {
                "image_id": row["image_id"],
                "true_volume_percent": int(true),
                "predicted_volume_percent": int(predicted),
                **{
                    f"probability_{volume}": float(value)
                    for volume, value in zip(CLASSES, probability, strict=True)
                },
            }
        )
    write_csv(directory / "validation_predictions.csv", predictions)
    write_csv(directory / "learning_curve.csv", curve)
    metrics = {
        "training": train_score["metrics"],
        "validation": val_score["metrics"],
        "subgroups": subgroup_metrics(
            rows, val_score["true"], val_score["predictions"]
        ),
    }
    (directory / "metrics.json").write_bytes(encoded(metrics))
    result = {
        "run_id": directory.name,
        "status": "passed",
        **spec,
        "selected_epoch": best_epoch,
        "epochs_executed": len(curve),
        "training_seconds": seconds,
        "fitted_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "validation_macro_f1": val_score["metrics"]["macro_f1"],
        "validation_accuracy": val_score["metrics"]["accuracy"],
        "training_macro_f1": train_score["metrics"]["macro_f1"],
        "validation_volume_mae_pp": val_score["metrics"]["volume_mae_pp"],
        "test_used": False,
        "artifacts": {
            name: sha256_file(directory / name)
            for name in (
                "checkpoint.pt",
                "metrics.json",
                "learning_curve.csv",
                "validation_predictions.csv",
            )
        },
    }
    record = make_record(
        kind="phase5_training",
        project_root=root,
        config_sha256=sha256_file(root / "configs/phase5-v1.json"),
        seed=config["seed"],
        status="passed",
        dataset_manifest_sha256=sha256_file(root / "data/manifest.csv"),
        split_sha256=frozen["artifacts"]["assignments.csv"],
        preprocessing=provenance,
        model=result,
        metrics=metrics,
    )
    write_record(record, directory / "record.json")
    (directory / "result.json").write_bytes(encoded(result))
    print(
        f"{directory.name}: macro F1={result['validation_macro_f1']:.4f}, "
        f"accuracy={result['validation_accuracy']:.2%}, epoch={best_epoch}",
        flush=True,
    )
    return result


def load_checkpoint(root: Path, path: Path) -> tuple[DeepClassifier, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    spec, config = checkpoint["specification"], checkpoint["config"]
    if checkpoint["classes_volume_percent"] != list(CLASSES):
        raise ValueError("Checkpoint volume class mapping changed")
    encoder = encoder_from_weights(root, spec["backbone"])
    state = checkpoint["model_state"]
    head = EmbeddingHead(
        WEIGHTS[spec["backbone"]][2],
        spec["head"],
        state["head.mean"],
        state["head.scale"],
        config["head"],
    )
    model = DeepClassifier(encoder, head)
    model.load_state_dict(state)
    return model.eval(), checkpoint


def image_batch(root, rows, manifest, representation, config, source, rng=None):
    return torch.stack(
        [
            image_regions(
                checked_image(root, manifest[row["image_id"]]),
                representation,
                config,
                source,
                rng,
            )
            for row in rows
        ]
    )


def evaluate_images(
    model, root, rows, manifest, representation, config, source, device
):
    model.eval()
    logits = []
    with torch.inference_mode():
        for offset in range(0, len(rows), config["fine_tuning"]["batch_size"]):
            batch = rows[offset : offset + config["fine_tuning"]["batch_size"]]
            regions = image_batch(root, batch, manifest, representation, config, source)
            logits.append(model(regions.to(device)).cpu())
    return score_logits(
        torch.cat(logits), class_indices([r["volume_percent"] for r in rows])
    )


def fit_images(root, initial, rows, manifest, config, source, device, learning_rate):
    seed_everything(config["seed"])
    model, checkpoint = load_checkpoint(root, initial)
    model.to(device)
    fine_tuning_mode(model)
    settings, spec = config["fine_tuning"], checkpoint["specification"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.layer4.parameters(), "lr": learning_rate},
            {"params": model.head.parameters(), "lr": settings["head_learning_rate"]},
        ],
        weight_decay=settings["weight_decay"],
    )
    generator = torch.Generator().manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    curve, best_state, best_score, best_epoch = [], None, -1.0, 0
    for epoch in range(1, settings["max_epochs"] + 1):
        fine_tuning_mode(model)
        order = torch.randperm(len(rows["train"]), generator=generator).tolist()
        for offset in range(0, len(order), settings["batch_size"]):
            selected = [
                rows["train"][i]
                for i in order[offset : offset + settings["batch_size"]]
            ]
            regions = image_batch(
                root, selected, manifest, spec["representation"], config, source, rng
            )
            targets = class_indices([r["volume_percent"] for r in selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(regions.to(device)), targets)
            loss.backward()
            optimizer.step()
        train_score, val_score = [
            evaluate_images(
                model,
                root,
                rows[part],
                manifest,
                spec["representation"],
                config,
                source,
                device,
            )
            for part in ("train", "validation")
        ]
        row = curve_row(epoch, train_score, val_score)
        curve.append(row)
        print(
            f"fine-tune lr={learning_rate:g} epoch={epoch}: "
            f"train F1={row['training_macro_f1']:.4f}, "
            f"validation F1={row['validation_macro_f1']:.4f}",
            flush=True,
        )
        if row["validation_macro_f1"] > best_score:
            best_score, best_epoch, best_state = (
                row["validation_macro_f1"],
                epoch,
                cpu_state(model),
            )
        if epoch - best_epoch >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    train_score, val_score = [
        evaluate_images(
            model,
            root,
            rows[part],
            manifest,
            spec["representation"],
            config,
            source,
            device,
        )
        for part in ("train", "validation")
    ]
    return model, curve, best_epoch, train_score, val_score


def run_phase5(root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Training destination exists: {destination}")
    config, source = load_settings(root)
    parts, manifest, frozen = development_rows(root)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    destination.mkdir(parents=True)
    provenance = {
        "config_sha256": sha256_file(root / "configs/phase5-v1.json"),
        "source_preprocessing_sha256": sha256_file(
            root / config["preprocessing"]["source_config"]
        ),
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "encoder_device": str(device),
        "frozen_head_device": "cpu",
        "test_used": False,
        "mask": "none",
    }
    (destination / "settings.json").write_bytes(encoded(config))
    (destination / "input_provenance.json").write_bytes(encoded(provenance))
    results, embeddings = [], {}
    labels = {
        part: class_indices([r["volume_percent"] for r in rows])
        for part, rows in parts.items()
    }
    try:
        for backbone in ("resnet18", "resnet50"):
            representations = (
                ["crop", "crop_detail"]
                if backbone == "resnet18"
                else [pick_best(results)["representation"]]
            )
            if backbone == "resnet50":
                (destination / "resnet50_input_selection.json").write_bytes(
                    encoded(
                        {
                            "selected_from": copy.deepcopy(results),
                            "representation": representations[0],
                            "criterion": "macro F1; declared 0.005/size/time tie rule",
                        }
                    )
                )
            embeddings[backbone], summary = extract_embeddings(
                root,
                backbone,
                representations,
                parts,
                manifest,
                config,
                source,
                device,
                frozen,
            )
            for representation in representations:
                for kind in ("linear", "mlp"):
                    run_id = f"{backbone}_{representation}_{kind}_frozen"
                    started = time.perf_counter()
                    head, curve, epoch, train_score, val_score = fit_head(
                        embeddings[backbone]["train"][representation],
                        labels["train"],
                        embeddings[backbone]["validation"][representation],
                        labels["validation"],
                        kind,
                        config,
                    )
                    seconds = time.perf_counter() - started
                    model = DeepClassifier(encoder_from_weights(root, backbone), head)
                    spec = {
                        "backbone": backbone,
                        "representation": representation,
                        "head": kind,
                        "stage": "frozen",
                        "device": "cpu",
                    }
                    results.append(
                        save_run(
                            root,
                            destination / run_id,
                            model,
                            spec,
                            curve,
                            epoch,
                            train_score,
                            val_score,
                            parts["validation"],
                            seconds,
                            config,
                            frozen,
                            {**provenance, "embeddings": summary},
                        )
                    )
                    write_comparison(destination, results)
                    del model
        winner = pick_best(results)
        (destination / "fine_tuning_selection.json").write_bytes(
            encoded(
                {
                    "selected_frozen": winner,
                    "initial_checkpoint_sha256": sha256_file(
                        destination / winner["run_id"] / "checkpoint.pt"
                    ),
                }
            )
        )
        for lr in config["fine_tuning"]["backbone_learning_rates"]:
            started = time.perf_counter()
            model, curve, epoch, train_score, val_score = fit_images(
                root,
                destination / winner["run_id"] / "checkpoint.pt",
                parts,
                manifest,
                config,
                source,
                device,
                lr,
            )
            spec = {key: winner[key] for key in ("backbone", "representation", "head")}
            spec.update(
                stage="fine_tuned",
                device=str(device),
                backbone_learning_rate=lr,
                initialized_from=winner["run_id"],
            )
            run_id = (
                f"{winner['backbone']}_{winner['representation']}_"
                f"{winner['head']}_ft_{lr:g}"
            )
            results.append(
                save_run(
                    root,
                    destination / run_id,
                    model,
                    spec,
                    curve,
                    epoch,
                    train_score,
                    val_score,
                    parts["validation"],
                    time.perf_counter() - started,
                    config,
                    frozen,
                    provenance,
                )
            )
            write_comparison(destination, results)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        summary = {
            "status": "completed",
            "runs": results,
            "selected_deep": pick_best(results),
            "test_used": False,
            "optional_fusion": "not run; bounded eight-run comparison completed",
        }
        (destination / "summary.json").write_bytes(encoded(summary))
        return summary
    except Exception as error:
        (destination / "FAILURE.json").write_bytes(
            encoded(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "completed_runs": [r["run_id"] for r in results],
                    "test_used": False,
                }
            )
        )
        raise


def write_comparison(destination: Path, results: list[dict]) -> None:
    fields = list(dict.fromkeys(key for result in results for key in result))
    write_csv(destination / "comparison.csv", results, fields)
