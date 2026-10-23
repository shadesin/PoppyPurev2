"""Select the complete recipe and render a review of accepted Phase 5 results.

Run after train-deep and its automatic checkpoint/plot acceptance finish:
    .venv/bin/python scripts/review_phase5.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from poppypurev2.acquisition import sha256_file
from poppypurev2.baselines import pick_best
from poppypurev2.deep_inputs import (
    CLASSES,
    checked_image,
    development_rows,
    load_settings,
)
from poppypurev2.evaluation import encoded
from poppypurev2.preprocessing import crop_box, detail_tiles, resize_rgb
from poppypurev2.records import make_record, source_fingerprint


def preserve_training_source(root: Path, runs: Path) -> None:
    expected = json.loads(
        (root / "data/embeddings/phase5-v1/resnet18/summary.json").read_text()
    )["source_tree_sha256"]
    snapshot = runs / "source_snapshot"
    if not snapshot.exists():
        assert source_fingerprint(root) == expected
        shutil.copytree(
            root / "src/poppypurev2",
            snapshot / "src/poppypurev2",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(root / "configs", snapshot / "configs")
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(root / name, snapshot / name)
        files = {
            str(p.relative_to(snapshot)): sha256_file(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
        (snapshot / "SNAPSHOT.json").write_bytes(
            encoded(
                {
                    "source_tree_sha256": expected,
                    "files": files,
                    "purpose": "Preserve Phase 5 training source/config/environment",
                }
            )
        )
    assert source_fingerprint(snapshot) == expected
    info = json.loads((snapshot / "SNAPSHOT.json").read_text())
    for name, digest in info["files"].items():
        assert sha256_file(snapshot / name) == digest
    for name, original in (
        ("source_preprocessing.json", "configs/preprocessing-v1.json"),
        ("dataset_provenance.json", "data/dataset_freeze.json"),
        ("preparation_provenance.json", "data/preparation_provenance.json"),
    ):
        path = runs / name
        if not path.exists():
            shutil.copyfile(root / original, path)
        assert json.loads(path.read_text()) == json.loads((root / original).read_text())
    path = runs / "review_phase5.py"
    if not path.exists():
        shutil.copyfile(Path(__file__), path)
    assert sha256_file(path) == sha256_file(Path(__file__))


def main():
    root = Path(__file__).resolve().parents[1]
    runs = root / "experiments/runs/phase5-v1"
    report = root / "reports/phase5"
    selection_path = report / "MODEL_SELECTION.json"
    if selection_path.exists():
        raise FileExistsError(selection_path)
    summary = json.loads((runs / "summary.json").read_text())
    acceptance = json.loads((runs / "ACCEPTANCE.json").read_text())
    assert acceptance["status"] == "passed" and len(acceptance["accepted_runs"]) == 8
    config, source = load_settings(root)
    parts, manifest, frozen = development_rows(root)
    preserve_training_source(root, runs)
    with (root / "experiments/runs/phase4-v1/comparison.csv").open() as handle:
        classical = [
            dict(row) for row in csv.DictReader(handle) if row["role"] == "visual"
        ]
    for row in classical:
        for key in ("validation_macro_f1", "fitted_parameters", "training_seconds"):
            row[key] = float(row[key])
    winner = pick_best(classical + summary["runs"])
    assert winner["run_id"] in acceptance["accepted_runs"]
    directory = runs / winner["run_id"]
    metrics = json.loads((directory / "metrics.json").read_text())
    initial = (
        winner
        if winner["stage"] == "frozen"
        else next(
            row
            for row in summary["runs"]
            if row["run_id"] == winner["initialized_from"]
        )
    )
    selection = make_record(
        kind="phase5_recipe_selection",
        project_root=root,
        config_sha256=sha256_file(root / "configs/phase5-v1.json"),
        seed=42,
        status="locked_for_phase6",
        dataset_manifest_sha256=sha256_file(root / "data/manifest.csv"),
        split_sha256=frozen["artifacts"]["assignments.csv"],
        preprocessing={"source": source, "cnn": config["preprocessing"]},
        model=winner,
        metrics=metrics,
    )
    selection.update(
        {
            "classes_volume_percent": list(CLASSES),
            "percentage_basis": "volume",
            "checkpoint": str((directory / "checkpoint.pt").relative_to(root)),
            "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
            "considered_visual_configurations": len(classical) + len(summary["runs"]),
            "candidate_run_ids": [r["run_id"] for r in classical + summary["runs"]],
            "criterion": (
                "validation macro F1; within 0.005 prefer fewer fitted parameters, "
                "then time"
            ),
            "selection_script_sha256": sha256_file(Path(__file__)),
            "training_source_snapshot_sha256": sha256_file(
                runs / "source_snapshot/SNAPSHOT.json"
            ),
            "preparation_provenance_sha256": sha256_file(
                root / "data/preparation_provenance.json"
            ),
            "optional_fusion_executed": False,
            "confirmation_seed_repeats_executed": False,
            "phase6_proposed_refit": {
                "images": 466,
                "partition": "train+validation",
                "seed": 42,
                "frozen_head_epochs": initial["selected_epoch"],
                "fine_tuning_epochs": winner["selected_epoch"]
                if winner["stage"] == "fine_tuned"
                else 0,
                "scaler": "fit on the 466 refit images only",
                "stopping": "fixed recorded counts; no test-based stopping",
                "executed": False,
            },
            "test_images": 82,
            "test_used": False,
        }
    )
    selection_path.write_bytes(encoded(selection))
    with (directory / "validation_predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    errors = [
        r
        for r in predictions
        if r["true_volume_percent"] != r["predicted_volume_percent"]
    ]
    rows = {r["image_id"]: r for r in parts["validation"]}
    font = ImageFont.load_default(size=16)
    gallery = Image.new("RGB", (1280, max(1, (len(errors) + 1) // 2) * 350), "#eeeeee")
    for index, error in enumerate(errors):
        assert error["image_id"] in rows
        image = checked_image(root, manifest[error["image_id"]])
        crop = image.crop(crop_box(image.size, source["crop_fraction_ltrb"]))
        views = (crop, detail_tiles(crop, source["tile_size"])[-1][0])
        left, top = (index % 2) * 640, (index // 2) * 350
        for col, view in enumerate(views):
            gallery.paste(resize_rgb(view, 300), (left + col * 320, top))
        row = rows[error["image_id"]]
        ImageDraw.Draw(gallery).text(
            (left + 3, top + 305),
            f"{error['image_id']} | true {error['true_volume_percent']}% -> "
            f"predicted {error['predicted_volume_percent']}% by volume\n"
            f"{row['background']} / {row['magnification']} | overview + centre detail",
            fill="black",
            font=font,
        )
    gallery.save(report / "validation_error_gallery.jpg")
    (report / "validation_errors.json").write_bytes(encoded(errors))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".plot-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checksums_path = report / "plots/checksums.json"
    checksums = json.loads(checksums_path.read_text())
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, metric in enumerate(("precision", "recall", "f1")):
        values = [metrics["validation"]["per_class"][str(v)][metric] for v in CLASSES]
        ax.bar(np.arange(7) + (offset - 1) * 0.25, values, 0.25, label=metric)
    ax.set(
        xticks=range(7),
        xticklabels=CLASSES,
        ylim=(0, 1.08),
        xlabel="Semolina % by volume",
        ylabel="Score",
        title="Selected model — validation per-class results",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure_pairs = [(fig, "selected_per_class")]
    conditions = [
        (name.removeprefix("condition="), result)
        for name, result in metrics["subgroups"].items()
        if name.startswith("condition=")
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        [name for name, _ in conditions],
        [result["macro_f1"] for _, result in conditions],
    )
    ax.set(
        ylim=(0, 1.08),
        ylabel="Seven-class macro F1",
        title="Validation capture conditions",
    )
    for index, (_, result) in enumerate(conditions):
        n = sum(sum(row) for row in result["confusion_matrix"])
        ax.text(index, result["macro_f1"] + 0.02, f"n={n}", ha="center")
    figure_pairs.append((fig, "selected_conditions"))
    for fig, name in figure_pairs:
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = report / "plots" / f"{name}.{extension}"
            assert not path.exists()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            checksums[path.name] = sha256_file(path)
        plt.close(fig)
    checksums_path.write_bytes(encoded(checksums))
    report_summary_path = report / "summary.json"
    report_summary = json.loads(report_summary_path.read_text())
    report_summary.update(
        figures=checksums,
        selected_complete_recipe=winner,
        selection_record_sha256=sha256_file(selection_path),
    )
    report_summary_path.write_bytes(encoded(report_summary))
    val = metrics["validation"]
    baseline = json.loads(
        (root / "experiments/runs/phase4-v1/summary.json").read_text()
    )["selected_baseline"]
    baseline_metrics = json.loads(
        (
            root / "experiments/runs/phase4-v1" / baseline["run_id"] / "metrics.json"
        ).read_text()
    )["validation"]
    lines = [
        "# Phase 5 findings",
        "",
        "**Status: complete.** The complete recipe is selected for Phase 6; "
        "no test scoring has occurred.",
        "",
        "## Selected classifier",
        "",
        f"**{winner['backbone']} + {winner['head']}**, "
        f"`{winner['representation']}`, stage **{winner['stage']}**, "
        f"selected epoch **{winner['selected_epoch']}**.",
        "",
        f"Validation: **{val['macro_f1']:.4f} macro F1**, "
        f"**{val['accuracy']:.2%} accuracy ({82 - len(errors)}/82)**, "
        f"balanced accuracy **{val['balanced_accuracy']:.4f}**, "
        f"volume-class MAE **{val['volume_mae_pp']:.3f} percentage points**.",
        "",
        f"Classical reference: {baseline['validation_macro_f1']:.4f} macro F1 / "
        f"{baseline['validation_accuracy']:.2%} accuracy. "
        "The selected model changes these by "
        f"**{val['macro_f1'] - baseline['validation_macro_f1']:+.4f} macro F1 / "
        f"{(val['accuracy'] - baseline['validation_accuracy']) * 100:+.2f} "
        "accuracy percentage points**.",
        "",
        "The task remains seven-class semolina adulteration **by volume**, "
        "with labels 0, 10, 20, 30, 40, 50 and 100.",
        "",
        "## All eight deep configurations",
        "",
        "| Configuration | Validation macro F1 | Accuracy | Training macro F1 "
        "| Best epoch / executed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["runs"]:
        lines.append(
            f"| `{r['run_id']}` | {r['validation_macro_f1']:.4f} | "
            f"{r['validation_accuracy']:.2%} | {r['training_macro_f1']:.4f} | "
            f"{r['selected_epoch']} / {r['epochs_executed']} |"
        )
    lines += [
        "",
        "Selection compares all **21 classical visual configurations and eight "
        "deep configurations**, using validation macro F1 and the predefined "
        "0.005/size/time tie rule. Four Phase 4 diagnostic controls remain "
        "diagnostic. No optional fusion or extra confirmation seeds were run.",
        "",
        "## Per-class validation results",
        "",
        "| Volume class | Precision | Recall | F1 | Classical F1 | Images |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in CLASSES:
        r = val["per_class"][str(v)]
        lines.append(
            f"| {v}% | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | "
            f"{baseline_metrics['per_class'][str(v)]['f1']:.4f} | {r['support']} |"
        )
    decreases = [
        str(v) + "%"
        for v in CLASSES
        if val["per_class"][str(v)]["f1"] < baseline_metrics["per_class"][str(v)]["f1"]
    ]
    if decreases:
        lines += [
            "",
            "Class F1 is lower than the classical reference for "
            + ", ".join(decreases)
            + ". The higher overall macro F1 "
            "does not imply improvement in every class.",
        ]
    lines += [
        "",
        "## Remaining validation errors",
        "",
        f"There are **{len(errors)} errors**. All appear in "
        "[the error gallery](validation_error_gallery.jpg), with original IDs "
        "and true/predicted by-volume classes in "
        "[validation_errors.json](validation_errors.json).",
        "",
    ]
    for error in errors:
        lines.append(
            f"- `{error['image_id']}`: {error['true_volume_percent']}% → "
            f"{error['predicted_volume_percent']}%."
        )
    binary = val["binary"]
    lines += [
        "",
        "Derived pure-versus-adulterated detection: sensitivity "
        f"**{binary['sensitivity']:.2%}**, specificity "
        f"**{binary['specificity']:.2%}**; TP={binary['tp']}, TN={binary['tn']}, "
        f"FP={binary['fp']}, FN={binary['fn']}. "
        "Seven-class macro F1 remains the selection metric.",
        "",
        "## Processing and verification",
        "",
        "The CNN receives RGB crop/detail pixels. EXIF and folder metadata are "
        "excluded; masks and background removal are disabled. Frozen heads use "
        "training-only embedding scaling; fine-tuning preserves it. See "
        "[methods](METHODS.md) for exact weights, normalization, augmentation, "
        "schedules and devices.",
        "",
        "All eight checkpoints passed artifact, membership, scaler, "
        "prediction-reproduction and earliest-best-epoch acceptance. Fine-tuned "
        "checkpoints also preserve early blocks, BatchNorm running statistics "
        "and the scaler. Raw/manifest/split membership stays fixed. All results "
        "above use **384 fitting / 82 validation** images; the **82 test images "
        "are reserved**.",
        "",
        "Saved softmax probabilities are not a calibration measurement. "
        "Condition summaries exclude flagged magnification conflicts only where "
        "trusted magnification is needed; overall metrics retain these images.",
        "",
        "## Artifacts and next step",
        "",
        "- [Complete selected recipe and fixed Phase 6 refit schedule]"
        "(MODEL_SELECTION.json).",
        "- [Learning curves and result charts](plots/README.md).",
        "- All models/predictions/metrics: `experiments/runs/phase5-v1/`.",
        "- Embeddings/provenance: `data/embeddings/phase5-v1/`.",
        "- Phase 6: refit the selected recipe on 466 train+validation images for "
        "the fixed recorded counts, evaluate the reserved 82-image test once, "
        "and package identical CLI inference.",
        "",
    ]
    (report / "FINDINGS.md").write_text("\n".join(lines))
    figure_lines = [
        "# Phase 5 figures",
        "",
        "All figures use training/validation only. Curve markers identify the "
        "earliest best-validation-macro-F1 epoch. Training and validation loss "
        "use the same full-partition cross-entropy definition.",
        "",
        "Each PNG has a matching PDF. The original eight traces live beside "
        "their checkpoints in `experiments/runs/phase5-v1/`.",
        "",
    ]
    for path in sorted((report / "plots").glob("*.png")):
        figure_lines.append(f"- [{path.stem}]({path.name}) · [PDF]({path.stem}.pdf)")
    (report / "plots/README.md").write_text("\n".join(figure_lines) + "\n")
    print(
        json.dumps(
            {
                "selected": winner["run_id"],
                "validation_macro_f1": val["macro_f1"],
                "validation_accuracy": val["accuracy"],
                "errors": len(errors),
                "figure_files": len(checksums),
                "test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
