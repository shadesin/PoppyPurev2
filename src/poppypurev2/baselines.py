"""Bounded Phase 4 training and validation, with no final-test input path."""

from __future__ import annotations

import copy
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.config import CLASSES_VOLUME_PERCENT, load_config
from poppypurev2.controls import metadata_inputs
from poppypurev2.evaluation import classification_metrics, encoded, freeze_protocol
from poppypurev2.preprocessing import decode_rgb, resize_rgb
from poppypurev2.records import make_record, write_record


def family_columns(names: list[str], family: str) -> list[int]:
    if family not in {"colour", "texture", "combined"}:
        raise ValueError("Unknown feature family")
    indices = []
    for index, name in enumerate(names):
        base = name.removeprefix("mean_").removeprefix("std_")
        colour = base.startswith(("hsv_", "lab_"))
        texture = base.startswith(("lbp_", "glcm_"))
        if not (colour or texture):
            raise ValueError(f"Unrecognized feature: {name}")
        if (
            family == "combined"
            or (family == "colour" and colour)
            or (family == "texture" and texture)
        ):
            indices.append(index)
    return indices


def load_feature_pack(
    root: Path, directory: Path, partition: str, assigned: list[dict], frozen: dict
) -> tuple[dict, dict]:
    summary = json.loads((directory / "summary.json").read_text())
    schema = json.loads((directory / "schema.json").read_text())
    if (
        summary["partition"] != partition
        or summary["split_sha256"] != frozen["artifacts"]["assignments.csv"]
    ):
        raise ValueError("Feature partition/split provenance differs")
    for name, digest in summary["artifacts"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Feature artifact changed: {name}")
    for key, path in (
        ("feature_config_sha256", "configs/features-v1.json"),
        ("preprocessing_config_sha256", "configs/preprocessing-v1.json"),
    ):
        if summary[key] != sha256_file(root / path):
            raise ValueError(f"Feature definitions changed: {path}")
    with np.load(directory / f"{partition}.npz", allow_pickle=False) as bundle:
        arrays = {name: bundle[name].copy() for name in bundle.files}
    expected = [r for r in assigned if r["partition"] == partition]
    if arrays["image_ids"].tolist() != [r["image_id"] for r in expected]:
        raise ValueError("Feature IDs differ from partition membership/order")
    if arrays["volume_percent"].tolist() != [
        int(r["volume_percent"]) for r in expected
    ]:
        raise ValueError("Feature target labels differ from frozen membership")
    if arrays["preparation_ids"].tolist() != [
        r["preparation_group_id"] for r in expected
    ]:
        raise ValueError("Feature preparation groups differ")
    for variant, names in schema.items():
        matrix = arrays[variant]
        if matrix.shape != (len(expected), len(names)) or not np.isfinite(matrix).all():
            raise ValueError("Invalid feature matrix")
    return arrays, schema


def fitted_parameters(model) -> int:
    """Comparable recorded size convention; structural counts are explicit."""
    if isinstance(model, Pipeline):
        return fitted_parameters(model.steps[-1][1])
    if isinstance(model, LogisticRegression):
        return model.coef_.size + model.intercept_.size
    if isinstance(model, MLPClassifier):
        return sum(a.size for a in [*model.coefs_, *model.intercepts_])
    if isinstance(model, SVC):
        return (
            model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size
        )
    if isinstance(model, RandomForestClassifier):
        return sum(
            2 * int((tree.tree_.children_left != -1).sum())
            + int((tree.tree_.children_left == -1).sum()) * len(model.classes_)
            for tree in model.estimators_
        )
    return 0


def preprocessing_statistics(model) -> int:
    """Stored scaler statistics/category values, recorded separately from the head."""
    if isinstance(model, Pipeline):
        return sum(preprocessing_statistics(step) for _, step in model.steps[:-1])
    if isinstance(model, ColumnTransformer):
        return sum(
            preprocessing_statistics(step)
            for _, step, _ in model.transformers_
            if not isinstance(step, str)
        )
    if isinstance(model, StandardScaler):
        return sum(
            getattr(model, name).size
            for name in ("mean_", "scale_", "var_")
            if getattr(model, name, None) is not None
        )
    if isinstance(model, OneHotEncoder):
        return sum(len(values) for values in model.categories_)
    return 0


def pick_best(results: list[dict], tolerance: float = 0.005) -> dict:
    valid = [r for r in results if r["status"] == "passed"]
    if not valid:
        raise ValueError("No converged successful candidate")
    best_score = max(r["validation_macro_f1"] for r in valid)
    tied = [
        r for r in valid if best_score - r["validation_macro_f1"] <= tolerance + 1e-12
    ]
    return min(
        tied, key=lambda r: (r["fitted_parameters"], r["training_seconds"], r["run_id"])
    )


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    seed: int,
    settings: dict,
) -> tuple[Pipeline, list[dict]]:
    scaler = StandardScaler().fit(x)
    xs, xvs = scaler.transform(x), scaler.transform(xv)
    classifier = MLPClassifier(
        hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]),
        activation=settings["activation"],
        solver=settings["solver"],
        alpha=settings["alpha"],
        learning_rate_init=settings["learning_rate_init"],
        batch_size=settings["batch_size"],
        random_state=seed,
        early_stopping=False,
    )
    best, best_f1, wait = None, -1.0, 0
    curve = []
    for epoch in range(1, settings["max_epochs"] + 1):
        classifier.partial_fit(xs, y, classes=np.array(CLASSES_VOLUME_PERCENT))
        train_predictions = classifier.predict(xs)
        validation_predictions = classifier.predict(xvs)
        train_metrics = classification_metrics(y.tolist(), train_predictions.tolist())
        val_metrics = classification_metrics(
            yv.tolist(), validation_predictions.tolist()
        )
        curve.append(
            {
                "epoch": epoch,
                "loss": float(classifier.loss_),
                "training_log_loss": float(
                    log_loss(
                        y, classifier.predict_proba(xs), labels=CLASSES_VOLUME_PERCENT
                    )
                ),
                "validation_log_loss": float(
                    log_loss(
                        yv, classifier.predict_proba(xvs), labels=CLASSES_VOLUME_PERCENT
                    )
                ),
                "training_macro_f1": train_metrics["macro_f1"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "training_accuracy": float((train_predictions == y).mean()),
                "validation_accuracy": float((validation_predictions == yv).mean()),
            }
        )
        if val_metrics["macro_f1"] > best_f1 + 1e-12:
            best, best_f1, wait = copy.deepcopy(classifier), val_metrics["macro_f1"], 0
            best.selected_epoch_ = epoch
        else:
            wait += 1
        if wait >= settings["patience"]:
            break
    return Pipeline([("standardscaler", scaler), ("mlpclassifier", best)]), curve


def subgroup_metrics(rows: list[dict], true: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for key in ("background", "magnification"):
        for value in sorted({r[key] for r in rows}):
            indices = [
                i
                for i, row in enumerate(rows)
                if row[key] == value
                and (key != "magnification" or row["magnification_conflict"] != "True")
            ]
            if indices:
                result[f"{key}={value}"] = classification_metrics(
                    true[indices].tolist(), predicted[indices].tolist()
                )
    for background in ("black", "white"):
        for magnification in ("4x", "micro"):
            indices = [
                i
                for i, r in enumerate(rows)
                if r["background"] == background
                and r["magnification"] == magnification
                and r["magnification_conflict"] != "True"
            ]
            if indices:
                result[f"condition={background}_{magnification}"] = (
                    classification_metrics(
                        true[indices].tolist(), predicted[indices].tolist()
                    )
                )
    return result


def run_phase4(root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Experiment output already exists: {destination}")
    config_path = root / "configs/phase4-v1.json"
    settings = json.loads(config_path.read_text())
    base, base_sha = load_config(root / "configs/base.toml")
    if settings["seed"] != base["project"]["seed"]:
        raise ValueError("Training and project seeds differ")
    seed = settings["seed"]
    frozen = freeze_protocol(root, "v4", seed)
    with (root / "data/splits/v4/assignments.csv").open(newline="") as handle:
        assigned = list(csv.DictReader(handle))
    with (root / "data/manifest.csv").open(newline="") as handle:
        manifest = {r["image_id"]: r for r in csv.DictReader(handle)}
    train = [r for r in assigned if r["partition"] == "train"]
    validation = [r for r in assigned if r["partition"] == "validation"]
    training_pack, schema = load_feature_pack(
        root, root / settings["training_features"], "train", assigned, frozen
    )
    validation_pack, validation_schema = load_feature_pack(
        root, root / settings["validation_features"], "validation", assigned, frozen
    )
    if schema != validation_schema:
        raise ValueError("Training and validation feature schemas differ")
    if {r["preparation_group_id"] for r in train} & {
        r["preparation_group_id"] for r in validation
    }:
        raise ValueError("Preparation leakage")
    control_path = root / "data/features/controls-v1"
    control_summary = json.loads((control_path / "summary.json").read_text())
    if control_summary["split_sha256"] != frozen["artifacts"][
        "assignments.csv"
    ] or control_summary["phase4_config_sha256"] != sha256_file(config_path):
        raise ValueError("Control input provenance changed")
    if (
        sha256_file(control_path / "low_gradient.npz")
        != control_summary["artifact_sha256"]
    ):
        raise ValueError("Control artifact changed")
    with np.load(control_path / "low_gradient.npz", allow_pickle=False) as bundle:
        control_train, control_val = bundle["train"].copy(), bundle["validation"].copy()
        if (
            bundle["train_ids"].tolist() != training_pack["image_ids"].tolist()
            or bundle["validation_ids"].tolist()
            != validation_pack["image_ids"].tolist()
        ):
            raise ValueError("Control ID order differs")
    y, yv = training_pack["volume_percent"], validation_pack["volume_percent"]
    destination.mkdir(parents=True)
    (destination / "settings.json").write_bytes(encoded(settings))
    provenance = {
        "phase4_config_sha256": sha256_file(config_path),
        "training_feature_summary_sha256": sha256_file(
            root / settings["training_features"] / "summary.json"
        ),
        "validation_feature_summary_sha256": sha256_file(
            root / settings["validation_features"] / "summary.json"
        ),
        "control_summary_sha256": sha256_file(control_path / "summary.json"),
        "partitions": {"train": len(train), "validation": len(validation)},
        "test_used": False,
    }
    (destination / "inputs.json").write_bytes(encoded(provenance))
    results = []

    def execute(
        run_id: str, model, x: np.ndarray, xv: np.ndarray, recipe: dict, role: str
    ) -> dict:
        run = destination / run_id
        run.mkdir()
        record = make_record(
            kind="phase4_training",
            project_root=root,
            config_sha256=base_sha,
            seed=seed,
            status="running",
            dataset_manifest_sha256=frozen["manifest_sha256"],
            split_sha256=frozen["artifacts"]["assignments.csv"],
            preprocessing={**provenance, "input_recipe": recipe},
            model=recipe,
        )
        record["source_archive_sha256"] = json.loads(
            (root / "data/dataset_freeze.json").read_text()
        )["archive_sha256"]
        record["partitions"] = {
            "fit": "train",
            "selection": "validation",
            "test_used": False,
        }
        start = time.perf_counter()
        curve = []
        caught = []
        try:
            with (
                threadpool_limits(limits=settings["thread_limit"]),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                if recipe["model"] == "mlp":
                    model, curve = train_mlp(x, y, xv, yv, seed, settings["mlp"])
                else:
                    model.fit(x, y)
            elapsed = time.perf_counter() - start
            predicted = model.predict(xv)
            metrics = classification_metrics(yv.tolist(), predicted.tolist())
            metrics["accuracy"] = float((yv == predicted).mean())
            training_metrics = classification_metrics(
                y.tolist(), model.predict(x).tolist()
            )
            status = (
                "not_converged"
                if any(issubclass(w.category, ConvergenceWarning) for w in caught)
                else "passed"
            )
            score_kind = (
                "probability" if hasattr(model, "predict_proba") else "decision_margin"
            )
            scores = (
                model.predict_proba(xv)
                if score_kind == "probability"
                else model.decision_function(xv)
            )
            classes = model.classes_.tolist()
            if classes != list(CLASSES_VOLUME_PERCENT):
                raise ValueError("Fitted model class order differs")
            predictions = []
            for i, row in enumerate(validation):
                predictions.append(
                    {
                        "image_id": row["image_id"],
                        "preparation_group_id": row["preparation_group_id"],
                        "partition": "validation",
                        "volume_percent": int(yv[i]),
                        "predicted_volume_percent": int(predicted[i]),
                        "correct": bool(yv[i] == predicted[i]),
                        "background": row["background"],
                        "magnification": row["magnification"],
                        "magnification_conflict": row["magnification_conflict"],
                        "score_kind": score_kind,
                        **{
                            f"score_{label:03d}": float(scores[i, j])
                            for j, label in enumerate(classes)
                        },
                    }
                )
            write_csv(run / "validation_predictions.csv", predictions)
            joblib.dump(
                {
                    "pipeline": model,
                    "input_recipe": recipe,
                    "classes_volume_percent": list(CLASSES_VOLUME_PERCENT),
                    "schema": schema.get(recipe.get("representation"), None),
                    "provenance": provenance,
                },
                run / "model.joblib",
            )
            if curve:
                write_csv(run / "learning_curve.csv", curve)
            detail = {
                "validation": metrics,
                "training": training_metrics,
                "subgroups": subgroup_metrics(validation, yv, predicted),
            }
            (run / "metrics.json").write_bytes(encoded(detail))
            record.update(
                status=status,
                metrics=metrics,
                training_seconds=elapsed,
                fitted_parameters=fitted_parameters(model),
                warnings=[str(w.message) for w in caught],
                training_metrics=training_metrics,
                estimator_parameters=(
                    model.steps[-1][1] if isinstance(model, Pipeline) else model
                ).get_params(deep=False),
                preprocessing_statistic_count=preprocessing_statistics(model),
                validation_preparation_units=len(
                    {r["preparation_group_id"] for r in validation}
                ),
                subgroups=detail["subgroups"],
                selected_epoch=getattr(model.steps[-1][1], "selected_epoch_", None)
                if isinstance(model, Pipeline)
                else None,
                artifact_sha256={
                    name: sha256_file(run / name)
                    for name in (
                        "model.joblib",
                        "validation_predictions.csv",
                        "metrics.json",
                    )
                },
            )
            result = {
                "run_id": run_id,
                "role": role,
                "status": status,
                "representation": recipe.get("representation", "control"),
                "family": recipe.get("family", "control"),
                "model": recipe["model"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "volume_mae_pp": metrics["volume_mae_pp"],
                "training_macro_f1": training_metrics["macro_f1"],
                "fitted_parameters": record["fitted_parameters"],
                "training_seconds": elapsed,
                "selected_epoch": record["selected_epoch"],
            }
        except Exception as error:
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                training_seconds=time.perf_counter() - start,
                warnings=[str(w.message) for w in caught],
            )
            result = {
                "run_id": run_id,
                "role": role,
                "status": "failed",
                "error": record["error"],
            }
        write_record(record, run / "record.json")
        results.append(result)
        (destination / "progress.json").write_bytes(encoded(results))
        print(
            f"{run_id}: {result['status']} "
            f"macro_f1={result.get('validation_macro_f1')}",
            flush=True,
        )
        return result

    def logistic(C=1.0):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, random_state=seed, **settings["logistic"]),
        )

    dummy_x, dummy_xv = np.zeros((len(y), 1)), np.zeros((len(yv), 1))
    execute(
        "control_majority",
        DummyClassifier(strategy="most_frequent", random_state=seed),
        dummy_x,
        dummy_xv,
        {"model": "majority"},
        "control",
    )
    execute(
        "control_random",
        DummyClassifier(strategy="stratified", random_state=seed),
        dummy_x,
        dummy_xv,
        {"model": "stratified_random"},
        "control",
    )
    metadata_model = make_pipeline(
        ColumnTransformer(
            [
                ("dimensions", StandardScaler(), [0, 1]),
                (
                    "capture",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    [2, 3, 4],
                ),
            ]
        ),
        LogisticRegression(C=1.0, random_state=seed, **settings["logistic"]),
    )
    execute(
        "control_metadata",
        metadata_model,
        metadata_inputs([manifest[r["image_id"]] for r in train]),
        metadata_inputs([manifest[r["image_id"]] for r in validation]),
        {
            "model": "metadata_logistic",
            "numeric_fields": ["width", "height"],
            "categorical_fields": ["capture_date", "background", "magnification"],
        },
        "control",
    )
    execute(
        "control_low_gradient",
        logistic(),
        control_train,
        control_val,
        {
            "model": "low_gradient_logistic",
            "selection": settings["low_gradient_control"],
        },
        "control",
    )
    screening = []
    for representation in settings["screening"]["representations"]:
        for family in settings["screening"]["families"]:
            columns = family_columns(schema[representation], family)
            recipe = {
                "model": "logistic",
                "representation": representation,
                "family": family,
                "columns": columns,
                "C": settings["screening"]["C"],
                "estimator_settings": settings["logistic"],
            }
            screening.append(
                execute(
                    f"screen_{representation}_{family}",
                    logistic(recipe["C"]),
                    training_pack[representation][:, columns],
                    validation_pack[representation][:, columns],
                    recipe,
                    "visual",
                )
            )
    first = pick_best(screening, settings["near_tie"])
    second = pick_best(
        [r for r in screening if r["run_id"] != first["run_id"]], settings["near_tie"]
    )
    chosen = [first, second]
    (destination / "stage2_inputs.json").write_bytes(
        encoded(
            {"selected_from_screening": chosen, "selection_rule": settings["selection"]}
        )
    )
    for index, choice in enumerate(chosen, start=1):
        representation, family = choice["representation"], choice["family"]
        columns = family_columns(schema[representation], family)
        x, xv = (
            training_pack[representation][:, columns],
            validation_pack[representation][:, columns],
        )
        common = {
            "representation": representation,
            "family": family,
            "columns": columns,
        }
        for C in settings["svm"]["C_values"]:
            args = {k: v for k, v in settings["svm"].items() if k != "C_values"}
            execute(
                f"svm_input{index}_C{C:g}",
                make_pipeline(StandardScaler(), SVC(C=C, random_state=seed, **args)),
                x,
                xv,
                {**common, "model": "svm", "C": C, "estimator_settings": args},
                "visual",
            )
        for depth in settings["forest"]["max_depth_values"]:
            args = {
                k: v for k, v in settings["forest"].items() if k != "max_depth_values"
            }
            execute(
                f"forest_input{index}_depth{depth}",
                RandomForestClassifier(max_depth=depth, random_state=seed, **args),
                x,
                xv,
                {
                    **common,
                    "model": "forest",
                    "max_depth": depth,
                    "estimator_settings": args,
                },
                "visual",
            )
        execute(
            f"mlp_input{index}",
            None,
            x,
            xv,
            {**common, "model": "mlp", "estimator_settings": settings["mlp"]},
            "visual",
        )
    visual = [r for r in results if r["role"] == "visual"]
    best = pick_best(visual, settings["near_tie"])
    raw_best = max(
        [r for r in visual if r["status"] == "passed"],
        key=lambda r: r["validation_macro_f1"],
    )
    write_csv(
        destination / "comparison.csv",
        results,
        fields=[
            "run_id",
            "role",
            "status",
            "representation",
            "family",
            "model",
            "validation_macro_f1",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "volume_mae_pp",
            "training_macro_f1",
            "fitted_parameters",
            "training_seconds",
            "selected_epoch",
            "error",
        ],
    )
    summary = {
        "status": "completed",
        "runs": len(results),
        "visual_configurations": len(visual),
        "control_configurations": sum(r["role"] == "control" for r in results),
        "failed_or_not_converged": [r for r in results if r["status"] != "passed"],
        "selected_baseline": best,
        "highest_macro_f1": raw_best,
        "test_used": False,
        "input_provenance": provenance,
        "split_sha256": frozen["artifacts"]["assignments.csv"],
    }
    (destination / "summary.json").write_bytes(encoded(summary))
    build_report(root, destination, results, summary, manifest)
    return summary


def build_report(
    root: Path, destination: Path, results: list[dict], summary: dict, manifest: dict
) -> None:
    report = root / "reports/phase4"
    report.mkdir(parents=True, exist_ok=True)
    best = summary["selected_baseline"]
    with (destination / best["run_id"] / "validation_predictions.csv").open(
        newline=""
    ) as handle:
        predictions = list(csv.DictReader(handle))
    errors = sorted(
        [r for r in predictions if r["correct"] == "False"],
        key=lambda r: (
            -abs(int(r["volume_percent"]) - int(r["predicted_volume_percent"])),
            r["image_id"],
        ),
    )
    write_csv(report / "validation_errors.csv", errors, fields=list(predictions[0]))
    if errors:
        canvas = Image.new(
            "RGB", (4 * 280, ((min(len(errors), 24) + 3) // 4) * 260 + 40), "#eeeeee"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (5, 5),
            f"VALIDATION errors: {best['run_id']} | true -> predicted by volume",
            fill="black",
        )
        for i, row in enumerate(errors[:24]):
            image = decode_rgb(
                root / "data/raw" / manifest[row["image_id"]]["source_path"]
            )
            left, top = i % 4 * 280, 40 + i // 4 * 260
            canvas.paste(resize_rgb(image, 220), (left + 5, top))
            draw.text(
                (left + 5, top + 225),
                f"{row['image_id']} {row['volume_percent']} -> "
                f"{row['predicted_volume_percent']}%\n"
                f"{row['background']} {row['magnification']}",
                fill="black",
            )
        canvas.save(report / "validation_error_gallery.jpg")
    (report / "summary.json").write_bytes(encoded(summary))
    write_csv(
        report / "comparison.csv",
        results,
        fields=[
            "run_id",
            "role",
            "status",
            "representation",
            "family",
            "model",
            "validation_macro_f1",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "volume_mae_pp",
            "training_macro_f1",
            "fitted_parameters",
            "training_seconds",
            "selected_epoch",
            "error",
        ],
    )
    experiment_label = (
        destination.relative_to(root)
        if destination.is_relative_to(root)
        else destination
    )
    lines = [
        "# Phase 4 — baseline validation results",
        "",
        "384 training images; 82 validation images. Seven semolina adulteration "
        "classes **by volume**. Final test images were not used.",
        "",
        "## Comparison",
        "",
        "| Run | Role | Validation macro F1 | Accuracy | Training macro F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(results, key=lambda r: -r.get("validation_macro_f1", -1)):
        if "validation_macro_f1" in row:
            lines.append(
                f"| {row['run_id']} | {row['role']} | "
                f"{row['validation_macro_f1']:.4f} | "
                f"{row['validation_accuracy']:.4f} | "
                f"{row['training_macro_f1']:.4f} |"
            )
        else:
            lines.append(f"| {row['run_id']} | {row['role']} | failed | — | — |")
    lines.extend(
        [
            "",
            "## Selected Phase 4 baseline",
            "",
            f"**{best['run_id']}**: macro F1 **{best['validation_macro_f1']:.4f}**, "
            f"accuracy **{best['validation_accuracy']:.4f}**. Selection applies "
            "the frozen 0.005 near-tie rule (fewer fitted parameters, "
            "then measured training time).",
            "",
            "Controls are diagnostic and excluded from winner selection. "
            "The low-gradient control is not verified background-only; "
            "see its training overlays and review decisions.",
            "",
            f"Saved experiment directory: "
            f"`{experiment_label}`. "
            "Each run has a model, settings/provenance, predictions and metrics; "
            "MLP runs also save epoch curves. See `comparison.csv`, `summary.json` "
            "and `validation_errors.csv` here.",
            "",
            "These are validation results used for model selection. Phase 5 "
            "compares frozen ResNet heads and limited fine-tuning before final "
            "recipe selection. No test accuracy has been computed.",
        ]
    )
    (report / "BASELINES.md").write_text("\n".join(lines) + "\n")
