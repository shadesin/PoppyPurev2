"""Fixed 466-image refit and a sealed, single final-test evaluation."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch import nn

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.baselines import load_feature_pack, subgroup_metrics
from poppypurev2.deep import (
    DeepClassifier,
    EmbeddingHead,
    cpu_state,
    score_logits,
    seed_everything,
)
from poppypurev2.deep_inputs import (
    CLASSES,
    WEIGHTS,
    checked_image,
    class_indices,
    encoder_from_weights,
    image_regions,
    load_settings,
)
from poppypurev2.evaluation import classification_metrics, encoded, freeze_protocol
from poppypurev2.features import image_descriptor
from poppypurev2.inference import load_bundle, probability_rows
from poppypurev2.preprocessing import resize_rgb
from poppypurev2.records import make_record, source_fingerprint, write_record


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def phase6_inputs(root: Path) -> tuple:
    settings_path = root / "configs/phase6-v1.json"
    settings = json.loads(settings_path.read_text())
    selected_path = root / settings["selection_record"]
    if sha256_file(selected_path) != settings["selection_record_sha256"]:
        raise ValueError("Phase 5 selection differs from the approved Phase 6 recipe")
    selected = json.loads(selected_path.read_text())
    config, source = load_settings(root)
    if sha256_file(root / "configs/phase5-v1.json") != selected["config_sha256"]:
        raise ValueError("Deep settings changed after selection")
    if (
        selected["preprocessing"]["source"] != source
        or selected["preprocessing"]["cnn"] != config["preprocessing"]
    ):
        raise ValueError("Preprocessing changed after selection")
    if sha256_file(root / selected["checkpoint"]) != selected["checkpoint_sha256"]:
        raise ValueError("Selected development checkpoint changed")
    spec = {
        key: selected["model"][key]
        for key in ("backbone", "head", "representation", "stage")
    }
    if spec != {
        "backbone": "resnet18",
        "head": "mlp",
        "representation": "crop_detail",
        "stage": "frozen",
    }:
        raise ValueError("This final implementation requires the locked frozen recipe")
    if (
        settings["head_epochs"] != selected["model"]["selected_epoch"]
        or settings["head_epochs"] != 28
        or settings["seed"] != 42
        or settings["fine_tuning"]
        or settings["validation_during_refit"]
    ):
        raise ValueError("Refit schedule differs from the approved fixed 28 epochs")
    frozen = freeze_protocol(root, "v4")
    if frozen["artifacts"]["assignments.csv"] != selected["split_sha256"]:
        raise ValueError("Split differs from model selection")
    rows = read_csv(root / "data/splits/v4/assignments.csv")
    manifest = {r["image_id"]: r for r in read_csv(root / "data/manifest.csv")}
    refit = [r for r in rows if r["partition"] in settings["refit_partitions"]]
    test = [r for r in rows if r["partition"] == "test"]
    if len(refit) != 466 or len(test) != 82:
        raise ValueError("Unexpected final-fit/test membership")
    if {r["preparation_group_id"] for r in refit} & {
        r["preparation_group_id"] for r in test
    }:
        raise ValueError("Preparation units cross final-fit/test partitions")
    return settings, selected, config, source, rows, refit, test, manifest, frozen


def refit_embedding_matrix(root, rows, all_rows, config, source, frozen):
    cache = root / "data/embeddings/phase5-v1/resnet18"
    summary = json.loads((cache / "summary.json").read_text())
    expected = {
        "config_sha256": sha256_file(root / "configs/phase5-v1.json"),
        "source_preprocessing_sha256": sha256_file(
            root / config["preprocessing"]["source_config"]
        ),
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "manifest_sha256": sha256_file(root / "data/manifest.csv"),
        "weights_sha256": sha256_file(
            root / "models/pretrained" / WEIGHTS["resnet18"][0]
        ),
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError("Embedding cache differs from selected recipe")
    if source != json.loads(
        (root / config["preprocessing"]["source_config"]).read_text()
    ):
        raise ValueError("Source preprocessing differs")
    vectors = {}
    for part in ("train", "validation"):
        path = cache / f"{part}.npz"
        if sha256_file(path) != summary["artifacts"][path.name]:
            raise ValueError("Embedding artifact changed")
        expected_rows = [r for r in all_rows if r["partition"] == part]
        with np.load(path, allow_pickle=False) as pack:
            for key, field, cast in (
                ("image_ids", "image_id", str),
                ("volume_percent", "volume_percent", int),
                ("preparation_ids", "preparation_group_id", str),
            ):
                if pack[key].tolist() != [cast(r[field]) for r in expected_rows]:
                    raise ValueError("Embedding membership/target mismatch")
            matrix = pack["crop_detail"]
            if (
                matrix.shape != (len(expected_rows), 512)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Invalid cached embedding matrix")
            vectors.update(zip(pack["image_ids"].tolist(), matrix.copy(), strict=True))
    if set(vectors) != {r["image_id"] for r in rows}:
        raise ValueError("Final-fit embeddings contain missing or extra images")
    return np.stack([vectors[r["image_id"]] for r in rows]), summary


def fit_fixed_head(
    x: np.ndarray, labels: torch.Tensor, config: dict, epochs: int, seed: int
) -> tuple:
    """No validation argument, early stopping or best-epoch selection."""
    if x.ndim != 2 or x.shape[0] != len(labels) or not np.isfinite(x).all():
        raise ValueError("Invalid final-fit examples")
    seed_everything(seed)
    scaler = StandardScaler().fit(x)
    head = EmbeddingHead(x.shape[1], "mlp", scaler.mean_, scaler.scale_, config["head"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config["head"]["learning_rate"],
        weight_decay=config["head"]["weight_decay"],
    )
    features = torch.from_numpy(x)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for epoch in range(1, epochs + 1):
        head.train()
        for indices in torch.randperm(len(x), generator=generator).split(
            config["head"]["batch_size"]
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features[indices]), labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite final-fit loss")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            score = score_logits(head(features), labels)
        curve.append(
            {
                "epoch": epoch,
                **{
                    f"training_{key}": score["metrics"][key]
                    for key in ("cross_entropy", "macro_f1", "accuracy")
                },
            }
        )
        if epoch % 7 == 0 or epoch == epochs:
            print(
                f"Refit epoch {epoch}/{epochs}: training F1="
                f"{score['metrics']['macro_f1']:.4f}",
                flush=True,
            )
    return head, curve, score


def verify_reference_scaler(scaler, examples: np.ndarray) -> None:
    if scaler.n_samples_seen_ != len(examples):
        raise ValueError("Reference scaler was not fitted on all final-fit examples")
    # StandardScaler accumulates in float64 even when input descriptors are float32.
    np.testing.assert_allclose(
        scaler.mean_, examples.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        scaler.var_, examples.var(axis=0, dtype=np.float64), rtol=1e-10, atol=1e-10
    )


def refit_final(root: Path) -> dict:
    bundle, run = root / "models/poppypure-v1", root / "experiments/runs/phase6-v1"
    if bundle.exists() or run.exists():
        raise FileExistsError("Final model/run already exists; refusing to overwrite")
    settings, selected, config, source, all_rows, rows, _, manifest, frozen = (
        phase6_inputs(root)
    )
    torch.set_num_threads(config["cpu_threads"])
    x, embedding_provenance = refit_embedding_matrix(
        root, rows, all_rows, config, source, frozen
    )
    for index, row in enumerate(rows, 1):
        item = manifest[row["image_id"]]
        if sha256_file(root / "data/raw" / item["source_path"]) != item["sha256"]:
            raise ValueError(f"Final-fit raw image changed: {row['image_id']}")
        if index % 128 == 0:
            print(f"Refit raw integrity: {index}/466", flush=True)
    baseline_summary = json.loads(
        (root / "experiments/runs/phase4-v1/summary.json").read_text()
    )
    reference = settings["reference"]
    if baseline_summary["selected_baseline"]["run_id"] != reference["selected_run"]:
        raise ValueError("Reference differs from preselected classical baseline")
    reference_record = json.loads(
        (
            root
            / "experiments/runs/phase4-v1"
            / reference["selected_run"]
            / "record.json"
        ).read_text()
    )["model"]
    if (
        reference["C"] != reference_record["C"]
        or reference["kernel"] != reference_record["estimator_settings"]["kernel"]
        or reference["gamma"] != reference_record["estimator_settings"]["gamma"]
        or reference["representation"] != reference_record["representation"]
        or reference["family"] != reference_record["family"]
    ):
        raise ValueError("Reference hyperparameters differ from the selected SVM")
    handcrafted = {}
    for part, directory in (
        ("train", "data/features/v1"),
        ("validation", "data/features/validation-v1"),
    ):
        pack, _ = load_feature_pack(root, root / directory, part, all_rows, frozen)
        handcrafted.update(zip(pack["image_ids"].tolist(), pack["whole"], strict=True))
    reference_x = np.stack([handcrafted[r["image_id"]] for r in rows])
    labels = class_indices([r["volume_percent"] for r in rows])
    bundle.mkdir(parents=True)
    run.mkdir(parents=True)
    snapshot = run / "source_snapshot"
    shutil.copytree(
        root / "src/poppypurev2",
        snapshot / "src/poppypurev2",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(root / "configs", snapshot / "configs")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(root / name, snapshot / name)
        shutil.copyfile(root / name, bundle / name)
    (snapshot / "SNAPSHOT.json").write_bytes(
        encoded(
            {
                "source_tree_sha256": source_fingerprint(root),
                "files": {
                    str(p.relative_to(snapshot)): sha256_file(p)
                    for p in sorted(snapshot.rglob("*"))
                    if p.is_file()
                },
            }
        )
    )
    (run / "settings.json").write_bytes(encoded(settings))
    write_csv(run / "refit_membership.csv", rows)
    np.savez_compressed(
        run / "refit_embeddings.npz",
        embeddings=x,
        image_ids=np.array([r["image_id"] for r in rows]),
        volume_percent=np.array([int(r["volume_percent"]) for r in rows]),
    )
    started = time.perf_counter()
    head, curve, training = fit_fixed_head(
        x, labels, config, settings["head_epochs"], settings["seed"]
    )
    fitting_seconds = time.perf_counter() - started
    model = DeepClassifier(encoder_from_weights(root, "resnet18"), head).eval()
    checkpoint = {
        "schema_version": 1,
        "model_version": settings["model_version"],
        "classes_volume_percent": list(CLASSES),
        "percentage_basis": "volume",
        "specification": {
            key: selected["model"][key]
            for key in ("backbone", "head", "representation", "stage")
        },
        "config": config,
        "source_preprocessing": source,
        "fixed_epochs": settings["head_epochs"],
        "fit_images": 466,
        "model_state": cpu_state(model),
        "selection_record_sha256": settings["selection_record_sha256"],
    }
    torch.save(checkpoint, bundle / "model.pt")
    write_csv(run / "learning_curve.csv", curve)
    (run / "training_metrics.json").write_bytes(encoded(training["metrics"]))
    y = np.array([int(r["volume_percent"]) for r in rows])
    reference_model = make_pipeline(
        StandardScaler(),
        SVC(
            C=reference["C"],
            kernel=reference["kernel"],
            gamma=reference["gamma"],
            probability=False,
        ),
    )
    with threadpool_limits(limits=1):
        reference_model.fit(reference_x, y)
    joblib.dump(reference_model, bundle / "reference_svm.joblib")
    metadata = {
        "schema_version": 1,
        "model_version": settings["model_version"],
        "percentage_basis": "volume",
        "classes_volume_percent": list(CLASSES),
        "checkpoint_sha256": sha256_file(bundle / "model.pt"),
        "refit_images": 466,
        "refit_preparation_units": len({r["preparation_group_id"] for r in rows}),
        "epochs": 28,
        "validation_during_refit": False,
        "architecture": "frozen ResNet18 + 512/128/7 MLP",
        "input": "five pooled native detail regions",
        "manifest_sha256": sha256_file(root / "data/manifest.csv"),
        "split_sha256": frozen["artifacts"]["assignments.csv"],
    }
    (bundle / "bundle.json").write_bytes(encoded(metadata))
    (bundle / "recipe.json").write_bytes(
        encoded(
            {
                "phase6": settings,
                "source_preprocessing": source,
                "deep": config,
                "selection_record_sha256": settings["selection_record_sha256"],
            }
        )
    )
    # Complete model and inference acceptance happens before any test decoding.
    reloaded, saved, _, device = load_bundle(bundle, "cpu")
    with torch.inference_mode():
        reproduced = reloaded.head(torch.from_numpy(x)).softmax(1).numpy()
    np.testing.assert_allclose(
        reproduced, training["probabilities"], atol=1e-7, rtol=1e-6
    )
    fitted_scaler = StandardScaler().fit(x)
    np.testing.assert_array_equal(
        reloaded.head.mean.numpy(), fitted_scaler.mean_.astype(np.float32)
    )
    np.testing.assert_array_equal(
        reloaded.head.scale.numpy(), fitted_scaler.scale_.astype(np.float32)
    )
    initial_encoder = encoder_from_weights(root, "resnet18").state_dict()
    for name, value in reloaded.encoder.state_dict().items():
        if not torch.equal(value.cpu(), initial_encoder[name]):
            raise ValueError(f"Frozen ImageNet encoder tensor changed: {name}")
    for index in (0, len(rows) // 2, len(rows) - 1):
        image = checked_image(root, manifest[rows[index]["image_id"]])
        regions = image_regions(
            image, "crop_detail", saved["config"], saved["source_preprocessing"]
        )
        with torch.inference_mode():
            observed = reloaded(regions[None].to(device)).softmax(1).cpu().numpy()[0]
        np.testing.assert_allclose(observed, reproduced[index], atol=1e-5, rtol=1e-4)
    loaded_reference = joblib.load(bundle / "reference_svm.joblib")
    np.testing.assert_array_equal(
        loaded_reference.predict(reference_x), reference_model.predict(reference_x)
    )
    verify_reference_scaler(loaded_reference[0], reference_x)
    assert (
        len(curve) == 28
        and curve[-1]["training_macro_f1"] == training["metrics"]["macro_f1"]
    )
    acceptance = {
        "status": "passed",
        "images": 466,
        "epochs": 28,
        "validation_used": False,
        "test_used": False,
        "checks": [
            "train+validation membership only",
            "466-image scaler statistics",
            "final epoch state retained",
            "persisted head probabilities",
            "frozen encoder and BatchNorm match pretrained weights",
            "three full RGB inference checks against cached features",
            "reference persistence and 466-image scaler",
        ],
    }
    (run / "REFIT_ACCEPTANCE.json").write_bytes(encoded(acceptance))
    record = make_record(
        kind="phase6_final_fit",
        project_root=root,
        config_sha256=sha256_file(root / "configs/phase6-v1.json"),
        seed=42,
        status="completed",
        dataset_manifest_sha256=metadata["manifest_sha256"],
        split_sha256=metadata["split_sha256"],
        preprocessing={
            "source": source,
            "cnn": config["preprocessing"],
            "embeddings": embedding_provenance,
        },
        model={**metadata, "head_fit_seconds": fitting_seconds},
        metrics={"training": training["metrics"]},
    )
    record["source_archive_sha256"] = json.loads(
        (root / "data/dataset_freeze.json").read_text()
    )["archive_sha256"]
    record["test_used"] = False
    write_record(record, run / "refit_record.json")
    seal = {
        "schema_version": 1,
        "status": "sealed_before_test",
        "model_version": settings["model_version"],
        "primary_checkpoint_sha256": metadata["checkpoint_sha256"],
        "phase6_config_sha256": sha256_file(root / "configs/phase6-v1.json"),
        "selection_record_sha256": settings["selection_record_sha256"],
        "split_sha256": metadata["split_sha256"],
        "source_tree_sha256": source_fingerprint(root),
        "primary_chosen_before_test": True,
        "test_used": False,
        "bundle_files": {
            name: sha256_file(bundle / name)
            for name in (
                "model.pt",
                "reference_svm.joblib",
                "bundle.json",
                "recipe.json",
                "pyproject.toml",
                "uv.lock",
            )
        },
    }
    (run / "PRETEST_LOCK.json").write_bytes(encoded(seal))
    return {"status": "refit_complete_and_sealed", **metadata, "acceptance": acceptance}


def session_metrics(rows, manifest, true, predicted):
    result = {}
    dates = [manifest[r["image_id"]]["capture_date"] for r in rows]
    for date in sorted(set(dates)):
        indices = [i for i, value in enumerate(dates) if value == date]
        result[date] = {
            "images": len(indices),
            "correct": int((true[indices] == predicted[indices]).sum()),
            "accuracy": float((true[indices] == predicted[indices]).mean()),
            "classes_present": sorted(set(true[indices].tolist())),
        }
    return result


def bootstrap_indices(rows: list[dict], rng) -> np.ndarray:
    groups = {}
    for index, row in enumerate(rows):
        key = (int(row["volume_percent"]), row["preparation_group_id"])
        groups.setdefault(key, []).append(index)
    labels_per_unit = {}
    for label, unit in groups:
        labels_per_unit.setdefault(unit, set()).add(label)
    if any(len(labels) != 1 for labels in labels_per_unit.values()):
        raise ValueError("A preparation unit has conflicting target labels")
    indices = []
    for label in CLASSES:
        keys = [key for key in groups if key[0] == label]
        if not keys:
            raise ValueError("Bootstrap needs all seven true classes")
        for selected in rng.integers(0, len(keys), len(keys)):
            indices.extend(groups[keys[selected]])
    return np.array(indices)


def bootstrap_scores(rows, primary, reference, settings):
    true = class_indices([r["volume_percent"] for r in rows]).numpy()
    predictions = [class_indices(values).numpy() for values in (primary, reference)]
    rng = np.random.default_rng(settings["seed"])
    samples = []
    for replicate in range(settings["replicates"]):
        indices = bootstrap_indices(rows, rng)
        scores = []
        for predicted in predictions:
            matrix = np.bincount(
                7 * true[indices] + predicted[indices], minlength=49
            ).reshape(7, 7)
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = np.divide(
                2 * matrix.diagonal(),
                denominator,
                out=np.zeros(7),
                where=denominator > 0,
            ).mean()
            scores.append((float(np.trace(matrix) / matrix.sum()), float(f1)))
        samples.append(
            {
                "replicate": replicate + 1,
                "primary_accuracy": scores[0][0],
                "primary_macro_f1": scores[0][1],
                "reference_accuracy": scores[1][0],
                "reference_macro_f1": scores[1][1],
                "accuracy_difference": scores[0][0] - scores[1][0],
                "macro_f1_difference": scores[0][1] - scores[1][1],
            }
        )
    intervals = {
        key: np.percentile(
            [r[key] for r in samples], settings["interval_percentiles"]
        ).tolist()
        for key in samples[0]
        if key != "replicate"
    }
    return {
        "method": "percentile bootstrap of preparation units within true class",
        "replicates": len(samples),
        "seed": settings["seed"],
        "interval_percentiles": settings["interval_percentiles"],
        "intervals": intervals,
    }, samples


def claim_test_access(
    run: Path, checkpoint_sha256: str, device: str, images: int
) -> Path:
    output = run / "test"
    output.mkdir(exist_ok=False)
    write_record(
        {
            "status": "started",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "primary_checkpoint_sha256": checkpoint_sha256,
            "pretest_lock_sha256": sha256_file(run / "PRETEST_LOCK.json"),
            "test_images": images,
            "device": device,
        },
        run / "TEST_ACCESS_STARTED.json",
    )
    return output


def evaluate_final(root: Path, requested_device: str = "auto") -> dict:
    bundle, run = root / "models/poppypure-v1", root / "experiments/runs/phase6-v1"
    settings, _, config, source, _, _, rows, manifest, frozen = phase6_inputs(root)
    seal = json.loads((run / "PRETEST_LOCK.json").read_text())
    if seal["phase6_config_sha256"] != sha256_file(root / "configs/phase6-v1.json"):
        raise ValueError("Final-test settings differ from pretest lock")
    if source_fingerprint(root) != seal["source_tree_sha256"]:
        raise ValueError("Runtime code differs from accepted pretest snapshot")
    for name, digest in seal["bundle_files"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError(f"Sealed bundle artifact changed: {name}")
    model, checkpoint, metadata, device = load_bundle(bundle, requested_device)
    reference = joblib.load(bundle / "reference_svm.joblib")
    feature_config = json.loads((root / "configs/features-v1.json").read_text())
    feature_summary = json.loads((root / "data/features/v1/summary.json").read_text())
    if (
        sha256_file(root / "configs/features-v1.json")
        != feature_summary["feature_config_sha256"]
    ):
        raise ValueError("Reference feature configuration changed")
    if len(rows) != settings["test_images"]:
        raise ValueError("Test membership changed")
    # Exclusive directory/ledger blocks a second model-scoring run.
    output = claim_test_access(
        run, metadata["checkpoint_sha256"], str(device), len(rows)
    )
    logits, whole_features = [], []
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), settings["test_batch_size"]):
                batch = rows[offset : offset + settings["test_batch_size"]]
                images = [checked_image(root, manifest[r["image_id"]]) for r in batch]
                inputs = torch.stack(
                    [
                        image_regions(
                            image,
                            "crop_detail",
                            checkpoint["config"],
                            checkpoint["source_preprocessing"],
                        )
                        for image in images
                    ]
                )
                batch_logits = model(inputs.to(device)).cpu()
                if not torch.isfinite(batch_logits).all():
                    raise ValueError("Nonfinite test logits")
                logits.append(batch_logits)
                whole_features.extend(
                    image_descriptor(
                        resize_rgb(image, source["long_side"]), feature_config
                    )
                    for image in images
                )
                print(
                    f"Final test inference: {min(offset + len(batch), len(rows))}/82",
                    flush=True,
                )
        logits = torch.cat(logits)
        whole_features = np.stack(whole_features)
        with threadpool_limits(limits=1):
            reference_pred = reference.predict(whole_features)
            margins = reference.decision_function(whole_features)
        np.testing.assert_array_equal(reference.classes_, CLASSES)
        primary_pred, probabilities = probability_rows(logits.softmax(1).numpy())
        np.savez_compressed(
            output / "inference_arrays.npz",
            cnn_logits=logits.numpy(),
            whole_features=whole_features,
            image_ids=np.array([r["image_id"] for r in rows]),
        )
        for role, predicted in (
            ("primary", primary_pred),
            ("reference", reference_pred),
        ):
            values = []
            for index, row in enumerate(rows):
                values.append(
                    {
                        **row,
                        "capture_date": manifest[row["image_id"]]["capture_date"],
                        "true_volume_percent": int(row["volume_percent"]),
                        "predicted_volume_percent": int(predicted[index]),
                        "score_kind": "probability"
                        if role == "primary"
                        else "decision_margin",
                        **(
                            {
                                f"probability_{v}": probabilities[index][str(v)]
                                for v in CLASSES
                            }
                            if role == "primary"
                            else {
                                f"margin_{v}": float(margins[index, j])
                                for j, v in enumerate(CLASSES)
                            }
                        ),
                    }
                )
            write_csv(output / f"{role}_predictions.csv", values)
        true = np.array([int(r["volume_percent"]) for r in rows])
        primary_metrics = score_logits(logits, class_indices(true))["metrics"]
        reference_metrics = classification_metrics(
            true.tolist(), reference_pred.tolist()
        )
        reference_metrics["accuracy"] = float((true == reference_pred).mean())
        intervals, samples = bootstrap_scores(
            rows, primary_pred, reference_pred, settings["bootstrap"]
        )
        write_csv(output / "bootstrap_samples.csv", samples)
        (output / "bootstrap.json").write_bytes(encoded(intervals))
        metrics = {
            "primary": primary_metrics,
            "reference": reference_metrics,
            "primary_subgroups": subgroup_metrics(rows, true, primary_pred),
            "reference_subgroups": subgroup_metrics(rows, true, reference_pred),
            "capture_dates": session_metrics(rows, manifest, true, primary_pred),
            "test_images": len(rows),
            "preparation_units": len({r["preparation_group_id"] for r in rows}),
            "primary_model_version": metadata["model_version"],
            "primary_checkpoint_sha256": metadata["checkpoint_sha256"],
            "primary_locked_before_test": True,
            "device": str(device),
            "test_used": True,
        }
        (output / "metrics.json").write_bytes(encoded(metrics))
        record = make_record(
            kind="phase6_final_test",
            project_root=root,
            config_sha256=seal["phase6_config_sha256"],
            seed=42,
            status="completed",
            dataset_manifest_sha256=metadata["manifest_sha256"],
            split_sha256=frozen["artifacts"]["assignments.csv"],
            preprocessing={"source": source, "cnn": config["preprocessing"]},
            model={
                "primary": metadata,
                "reference": settings["reference"],
                "pretest_lock": seal,
            },
            metrics=metrics,
        )
        record["test_used"] = True
        record["source_archive_sha256"] = json.loads(
            (root / "data/dataset_freeze.json").read_text()
        )["archive_sha256"]
        write_record(record, output / "record.json")
        completed = {
            "status": "completed",
            "test_images": len(rows),
            "primary_checkpoint_sha256": metadata["checkpoint_sha256"],
            "primary_locked_before_test": True,
            "artifacts": {
                p.name: sha256_file(p) for p in output.iterdir() if p.is_file()
            },
            "test_used": True,
        }
        write_record(completed, run / "TEST_ACCESS_COMPLETED.json")
        return metrics
    except Exception as error:
        write_record(
            {
                "status": "failed",
                "error": type(error).__name__,
                "message": str(error),
                "model_changed": False,
            },
            output / "FAILURE.json",
        )
        raise
