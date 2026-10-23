"""Reproduce MLP diagnostics and render saved Phase 4 validation results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from poppypurev2.acquisition import sha256_file
from poppypurev2.audit import write_csv
from poppypurev2.baselines import train_mlp
from poppypurev2.config import CLASSES_VOLUME_PERCENT, load_config
from poppypurev2.evaluation import encoded, freeze_protocol
from poppypurev2.records import make_record, write_record


def phase4_plots(root: Path, output: Path) -> dict:
    """Keep original runs immutable; refits only add verified loss diagnostics."""
    if output.exists():
        raise FileExistsError(f"Plot output already exists: {output}")
    for variable, name in (
        ("MPLCONFIGDIR", ".matplotlib-cache"),
        ("XDG_CACHE_HOME", ".plot-cache"),
    ):
        cache = (root / name).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    experiment = root / "experiments/runs/phase4-v1"
    summary = json.loads((experiment / "summary.json").read_text())
    settings = json.loads((experiment / "settings.json").read_text())
    frozen = freeze_protocol(root, "v4")
    _, config_hash = load_config(root / "configs/base.toml")
    with (experiment / "comparison.csv").open(newline="") as handle:
        comparison = list(csv.DictReader(handle))
    with np.load(root / "data/features/v1/train.npz", allow_pickle=False) as bundle:
        training = {key: bundle[key].copy() for key in bundle.files}
    with np.load(
        root / "data/features/validation-v1/validation.npz", allow_pickle=False
    ) as bundle:
        validation = {key: bundle[key].copy() for key in bundle.files}
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    files = []
    trace_records = []

    def save(figure, name):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(output / filename, bbox_inches="tight")
            files.append(filename)
        plt.close(figure)

    for run_id in ("mlp_input1", "mlp_input2"):
        original = experiment / run_id
        record = json.loads((original / "record.json").read_text())
        for name, digest in record["artifact_sha256"].items():
            if sha256_file(original / name) != digest:
                raise ValueError(f"Original artifact changed: {run_id}/{name}")
        recipe = record["model"]
        representation, columns = recipe["representation"], recipe["columns"]
        x, xv = (
            training[representation][:, columns],
            validation[representation][:, columns],
        )
        y, yv = training["volume_percent"], validation["volume_percent"]
        with threadpool_limits(limits=settings["thread_limit"]):
            replay, curve = train_mlp(
                x, y, xv, yv, record["seed"], recipe["estimator_settings"]
            )
        original_model = joblib.load(original / "model.joblib")["pipeline"]
        np.testing.assert_array_equal(replay.predict(xv), original_model.predict(xv))
        np.testing.assert_allclose(
            replay.predict_proba(xv), original_model.predict_proba(xv), rtol=0, atol=0
        )
        with (original / "learning_curve.csv").open(newline="") as handle:
            old_curve = list(csv.DictReader(handle))
        if len(curve) != len(old_curve):
            raise ValueError("Diagnostic replay changed epoch count")
        for new, old in zip(curve, old_curve, strict=True):
            for key in ("loss", "training_macro_f1", "validation_macro_f1"):
                if not np.isclose(new[key], float(old[key]), rtol=0, atol=1e-12):
                    raise ValueError(f"Diagnostic replay differs: {run_id}/{key}")
        epoch = replay.steps[-1][1].selected_epoch_
        if epoch != record["selected_epoch"]:
            raise ValueError("Diagnostic replay changed selected checkpoint")
        write_csv(output / f"{run_id}_diagnostic_trace.csv", curve)
        epochs = [r["epoch"] for r in curve]
        figure, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
        for axis, metric, title in zip(
            axes,
            ("log_loss", "macro_f1", "accuracy"),
            ("Cross-entropy loss (lower is better)", "Macro F1", "Accuracy"),
            strict=True,
        ):
            axis.plot(
                epochs,
                [r[f"training_{metric}"] for r in curve],
                label="Training (384)",
                color="#2563eb",
            )
            axis.plot(
                epochs,
                [r[f"validation_{metric}"] for r in curve],
                label="Validation (82)",
                color="#d97706",
            )
            axis.axvline(
                epoch, color="#64748b", linestyle="--", label=f"Selected epoch {epoch}"
            )
            axis.set(xlabel="Epoch", title=title)
            if metric != "log_loss":
                axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        figure.suptitle(
            f"{run_id}: {representation}, {recipe['family']} | "
            "exact diagnostic replay; no test data"
        )
        save(figure, f"{run_id}_learning_curves")
        replay_record = make_record(
            kind="phase4_diagnostic_replay",
            project_root=root,
            config_sha256=config_hash,
            seed=record["seed"],
            status="reproduced",
            dataset_manifest_sha256=frozen["manifest_sha256"],
            split_sha256=frozen["artifacts"]["assignments.csv"],
            model=recipe,
            preprocessing=record["preprocessing"],
            metrics=record["metrics"],
        )
        replay_record.update(
            original_run=run_id,
            original_record_sha256=sha256_file(original / "record.json"),
            selected_epoch=epoch,
            test_used=False,
            predictions_and_probabilities_identical=True,
            original_epoch_traces_reproduced=True,
            diagnostic_trace_sha256=sha256_file(
                output / f"{run_id}_diagnostic_trace.csv"
            ),
        )
        write_record(replay_record, output / f"{run_id}_replay.json")
        trace_records.append(
            {
                "run_id": run_id,
                "selected_epoch": epoch,
                "epochs": len(curve),
                "identical_original_predictions": True,
            }
        )

    best_id = summary["selected_baseline"]["run_id"]
    metrics = json.loads((experiment / best_id / "metrics.json").read_text())
    with (experiment / best_id / "validation_predictions.csv").open(
        newline=""
    ) as handle:
        predictions = list(csv.DictReader(handle))
    true = [int(r["volume_percent"]) for r in predictions]
    predicted = [int(r["predicted_volume_percent"]) for r in predictions]
    matrix = confusion_matrix(true, predicted, labels=CLASSES_VOLUME_PERCENT)
    if matrix.tolist() != metrics["validation"]["confusion_matrix"]:
        raise ValueError("Saved confusion matrix differs from predictions")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [f"{label}%" for label in CLASSES_VOLUME_PERCENT]
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(normalized, display_labels=labels).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=".2f"
    )
    axes[0].set_title("Counts — validation (82 images)")
    axes[1].set_title("Fraction of each true class")
    figure.suptitle(f"{best_id}: seven volume classes")
    save(figure, "selected_svm_confusion_matrix")

    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    positions = np.arange(7)
    for offset, (key, name, colour) in enumerate(
        (
            ("precision", "Precision", "#2563eb"),
            ("recall", "Recall", "#d97706"),
            ("f1", "F1", "#059669"),
        )
    ):
        values = [
            metrics["validation"]["per_class"][str(label)][key]
            for label in CLASSES_VOLUME_PERCENT
        ]
        axis.bar(
            positions + (offset - 1) * 0.24,
            values,
            width=0.24,
            label=name,
            color=colour,
        )
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 1.05),
        xlabel="Semolina adulteration by volume",
        ylabel="Score",
        title=f"Per-class validation performance: {best_id}",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save(figure, "selected_svm_per_class")

    ranked = sorted(comparison, key=lambda r: float(r["validation_macro_f1"]))
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.barh(
        [r["run_id"] for r in ranked],
        [float(r["validation_macro_f1"]) for r in ranked],
        color=["#64748b" if r["role"] == "control" else "#2563eb" for r in ranked],
    )
    axis.set(
        xlim=(0, 1),
        xlabel="Validation macro F1",
        title="All 25 Phase 4 runs — gray bars are diagnostic controls",
    )
    axis.grid(axis="x", alpha=0.2)
    save(figure, "all_run_comparison")

    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    selected_runs = [
        "svm_input1_C10",
        "svm_input2_C10",
        "mlp_input1",
        "control_metadata",
    ]
    for run_id in selected_runs:
        row = next(r for r in comparison if r["run_id"] == run_id)
        axis.scatter(
            float(row["training_macro_f1"]),
            float(row["validation_macro_f1"]),
            s=75,
            label=run_id,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    axis.set(
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        xlabel="Training macro F1",
        ylabel="Validation macro F1",
        title="Training versus validation performance",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    save(figure, "training_validation_gap")

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    condition_names = ["black_4x", "black_micro", "white_4x", "white_micro"]
    axis.bar(
        condition_names,
        [
            metrics["subgroups"][f"condition={name}"]["macro_f1"]
            for name in condition_names
        ],
        color="#2563eb",
    )
    axis.set(
        ylim=(0, 1),
        ylabel="Validation macro F1",
        title="Selected SVM by capture condition (trusted magnification tags)",
    )
    axis.grid(axis="y", alpha=0.2)
    save(figure, "selected_svm_conditions")

    result = {
        "status": "completed",
        "test_used": False,
        "original_experiments": 25,
        "diagnostic_replays": trace_records,
        "figures": files,
        "split_sha256": frozen["artifacts"]["assignments.csv"],
        "artifact_sha256": {name: sha256_file(output / name) for name in files},
    }
    (output / "summary.json").write_bytes(encoded(result))
    return result
