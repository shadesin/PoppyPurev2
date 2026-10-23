"""Final reports and figures from saved test predictions; no model scoring."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from poppypurev2.acquisition import sha256_file
from poppypurev2.deep_inputs import CLASSES, checked_image
from poppypurev2.evaluation import classification_metrics, encoded
from poppypurev2.final import read_csv
from poppypurev2.preprocessing import crop_box, detail_tiles, resize_rgb


def report_final(root: Path) -> dict:
    run = root / "experiments/runs/phase6-v1"
    test = run / "test"
    bundle = root / "models/poppypure-v1"
    complete = json.loads((run / "TEST_ACCESS_COMPLETED.json").read_text())
    for name, digest in complete["artifacts"].items():
        if sha256_file(test / name) != digest:
            raise ValueError(f"Final-test artifact changed: {name}")
    metrics = json.loads((test / "metrics.json").read_text())
    metadata = json.loads((bundle / "bundle.json").read_text())
    if sha256_file(bundle / "model.pt") != complete["primary_checkpoint_sha256"]:
        raise ValueError("Report model differs from evaluated checkpoint")
    predictions = {
        role: read_csv(test / f"{role}_predictions.csv")
        for role in ("primary", "reference")
    }
    for role, rows in predictions.items():
        if len(rows) != 82 or len({r["image_id"] for r in rows}) != 82:
            raise ValueError("Final prediction coverage is invalid")
        recomputed = classification_metrics(
            [int(r["true_volume_percent"]) for r in rows],
            [int(r["predicted_volume_percent"]) for r in rows],
        )
        for key in ("macro_f1", "confusion_matrix", "volume_mae_pp", "binary"):
            if recomputed[key] != metrics[role][key]:
                raise ValueError(
                    f"Saved final metrics disagree with predictions: {role}/{key}"
                )
    intervals = json.loads((test / "bootstrap.json").read_text())
    report = root / "reports/phase6"
    plots = report / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".plot-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = {}

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "pdf"):
            path = plots / f"{name}.{extension}"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            artifacts[str(path.relative_to(report))] = sha256_file(path)
        plt.close(figure)

    curve = read_csv(run / "learning_curve.csv")
    if len(curve) != 28 or any("validation" in key for row in curve for key in row):
        raise ValueError("Refit trace differs from the fixed training-only schedule")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key, label in zip(
        axes,
        ("cross_entropy", "macro_f1", "accuracy"),
        ("Cross-entropy loss", "Macro F1", "Accuracy"),
        strict=True,
    ):
        ax.plot(
            [int(r["epoch"]) for r in curve],
            [float(r[f"training_{key}"]) for r in curve],
            label="466 fitting images",
        )
        ax.set(xlabel="Epoch", ylabel=label)
        if key != "cross_entropy":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Final refit: 28 fixed epochs — training metrics only")
    save(fig, "refit_learning_curves")
    primary, reference = metrics["primary"], metrics["reference"]
    matrix = np.asarray(primary["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    normalized = matrix / matrix.sum(1, keepdims=True)
    for ax, values, title in zip(
        axes, (matrix, normalized), ("Counts", "Recall within true class"), strict=True
    ):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=values.max())
        for i in range(7):
            for j in range(7):
                text = str(matrix[i, j]) if title == "Counts" else f"{values[i, j]:.0%}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > values.max() / 2 else "black",
                )
        ax.set(
            xticks=range(7),
            yticks=range(7),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicted semolina % by volume",
            ylabel="True semolina % by volume",
            title=title,
        )
    fig.suptitle("Final model — complete 82-image test set")
    save(fig, "test_confusion_matrix")
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, name in enumerate(("precision", "recall", "f1")):
        values = [primary["per_class"][str(v)][name] for v in CLASSES]
        ax.bar(np.arange(7) + (offset - 1) * 0.25, values, 0.25, label=name)
    ax.plot(
        range(7),
        [reference["per_class"][str(v)]["f1"] for v in CLASSES],
        "ko--",
        label="SVM reference F1",
        linewidth=1,
    )
    ax.set(
        xticks=range(7),
        xticklabels=CLASSES,
        ylim=(0, 1.08),
        xlabel="Semolina % by volume",
        ylabel="Score",
        title="Final test: per-class results",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "test_per_class")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name, label in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1"), strict=True
    ):
        values = [metrics[role][name] for role in ("primary", "reference")]
        ax.bar(["ResNet18 + MLP", "SVM reference"], values)
        for index, role in enumerate(("primary", "reference")):
            low, high = intervals["intervals"][f"{role}_{name}"]
            ax.vlines(index, low, high, color="black", linewidth=2)
            ax.text(index, 1.025, f"{values[index]:.4f}", ha="center")
        ax.set(ylim=(0, 1.08), ylabel=label)
    fig.suptitle("Same 466-image fitting set / 82-image test; 95% bootstrap intervals")
    save(fig, "test_reference_comparison")
    conditions = [
        key for key in metrics["primary_subgroups"] if key.startswith("condition=")
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, role in enumerate(("primary", "reference")):
        values = [metrics[f"{role}_subgroups"][key]["macro_f1"] for key in conditions]
        ax.bar(
            np.arange(len(conditions)) + (offset - 0.5) * 0.35,
            values,
            0.35,
            label="ResNet18 + MLP" if role == "primary" else "SVM reference",
        )
    ax.set(
        xticks=range(len(conditions)),
        xticklabels=[k.removeprefix("condition=") for k in conditions],
        ylim=(0, 1.08),
        ylabel="Seven-class macro F1",
        title="Final test capture conditions",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "test_conditions")
    errors = [
        r
        for r in predictions["primary"]
        if r["true_volume_percent"] != r["predicted_volume_percent"]
    ]
    manifest = {r["image_id"]: r for r in read_csv(root / "data/manifest.csv")}
    source = json.loads((bundle / "recipe.json").read_text())["source_preprocessing"]
    gallery = Image.new("RGB", (1280, max(1, (len(errors) + 1) // 2) * 350), "#eeeeee")
    font = ImageFont.load_default(size=16)
    if not errors:
        ImageDraw.Draw(gallery).text(
            (20, 20), "No test classification errors", fill="black", font=font
        )
    for index, row in enumerate(errors):
        image = checked_image(root, manifest[row["image_id"]])
        crop = image.crop(crop_box(image.size, source["crop_fraction_ltrb"]))
        left, top = (index % 2) * 640, (index // 2) * 350
        for col, view in enumerate(
            (crop, detail_tiles(crop, source["tile_size"])[-1][0])
        ):
            gallery.paste(resize_rgb(view, 300), (left + col * 320, top))
        ImageDraw.Draw(gallery).text(
            (left + 3, top + 305),
            f"{row['image_id']} | true {row['true_volume_percent']}% -> "
            f"predicted {row['predicted_volume_percent']}% by volume\n"
            f"{row['background']} / {row['magnification']} | overview + centre detail",
            fill="black",
            font=font,
        )
    gallery.save(report / "test_error_gallery.jpg")
    (report / "test_errors.json").write_bytes(encoded(errors))
    correct = int(np.trace(matrix))
    lines = [
        "# PoppyPure v2 — final evaluation",
        "",
        "**Phase 6 complete.** The selected recipe was refitted on 466 images for "
        "28 fixed epochs, sealed, and evaluated on all 82 reserved test images.",
        "",
        "## Final test results",
        "",
        "| Model | Fitting images | Test images | Accuracy | Macro F1 | "
        "Balanced accuracy | Volume-class MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in (
        ("ResNet18 + MLP (locked primary)", primary),
        ("SVM (fixed reference)", reference),
    ):
        lines.append(
            f"| {name} | 466 | 82 | {values['accuracy']:.2%} | "
            f"{values['macro_f1']:.4f} | {values['balanced_accuracy']:.4f} | "
            f"{values['volume_mae_pp']:.3f} points |"
        )
    lines += [
        "",
        f"The primary model correctly classified **{correct}/82** images, "
        f"with **{len(errors)} errors**. Its version is **poppypure-v1**. "
        "The prediction classes are semolina adulteration percentages **by volume**: "
        "0, 10, 20, 30, 40, 50 and 100.",
        "",
        "The SVM is the previously selected whole-image colour/texture RBF recipe, "
        "refitted on the same 466 images. Both model recipes were fixed before test "
        "access. The test comparison does not select a replacement primary model.",
        "",
        "## Refit and development distinction",
        "",
        "Development selected the 384-image ResNet18 detail + MLP checkpoint at "
        "epoch 28: validation macro F1 0.9249 / accuracy 92.68% (76/82). "
        "The final model is a fresh fit on 384 training + 82 validation images. "
        "The scaler was fitted on these 466 images; the MLP trained for exactly 28 "
        "epochs from seed 42. The ImageNet encoder and BatchNorm remained frozen.",
        "",
        "The refit had **no separate validation set and no early stopping**. Its "
        "learning curves contain training metrics only. The final test was scored "
        "after the checkpoint and reference were sealed. Development and final-test "
        "results refer to different image partitions and fitted checkpoints.",
        "",
        "## Per-class test results",
        "",
        "| Volume class | Precision | Recall | F1 | SVM F1 | Images |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in CLASSES:
        item = primary["per_class"][str(v)]
        lines.append(
            f"| {v}% | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {reference['per_class'][str(v)]['f1']:.4f} | "
            f"{item['support']} |"
        )
    binary = primary["binary"]
    lines += [
        "",
        "## Pure versus adulterated detection",
        "",
        "Derived from the seven-class output: sensitivity "
        f"**{binary['sensitivity']:.2%}**, specificity "
        f"**{binary['specificity']:.2%}**; TP={binary['tp']}, TN={binary['tn']}, "
        f"FP={binary['fp']}, FN={binary['fn']}.",
        "",
        "## Test errors",
        "",
        "See [all error images](test_error_gallery.jpg) and "
        "[their original IDs and probabilities](test_errors.json).",
        "",
    ]
    for row in errors:
        lines.append(
            f"- `{row['image_id']}`: {row['true_volume_percent']}% → "
            f"{row['predicted_volume_percent']}% "
            f"({row['background']}/{row['magnification']})."
        )
    lines += [
        "",
        "## Uncertainty and capture conditions",
        "",
        "The 95% intervals use 2,000 percentile-bootstrap replicates, resampling "
        "preparation units within each true volume class (seed 1042). The test "
        f"contains {metrics['preparation_units']} preparation units across 82 images. "
        "Both models use the same sampled units for paired comparisons.",
        "",
        "| Test metric | Lower 95% bound | Upper 95% bound |",
        "| --- | ---: | ---: |",
    ]
    for key, bounds in intervals["intervals"].items():
        lines.append(f"| {key} | {bounds[0]:.4f} | {bounds[1]:.4f} |")
    lines += [
        "",
        "Condition summaries retain all images in the overall metrics. "
        "Flagged magnification conflicts are excluded only from summaries requiring "
        "trusted magnification.",
        "",
        "| Condition | Images | Primary macro F1 | SVM macro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in conditions:
        values = metrics["primary_subgroups"][key]
        n = sum(sum(row) for row in values["confusion_matrix"])
        lines.append(
            f"| {key.removeprefix('condition=')} | {n} | {values['macro_f1']:.4f} | "
            f"{metrics['reference_subgroups'][key]['macro_f1']:.4f} |"
        )
    lines += [
        "",
        "Capture-date summaries are descriptive; dates are associated with "
        "different class sets.",
        "",
        "| Capture date | Images | Correct | Accuracy | Classes present |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for date, values in metrics["capture_dates"].items():
        lines.append(
            f"| {date} | {values['images']} | {values['correct']} | "
            f"{values['accuracy']:.2%} | {values['classes_present']} |"
        )
    lines += [
        "",
        "## Predictor and reproducibility",
        "",
        "The packaged predictor applies the saved EXIF orientation/RGB conversion, "
        "fractional crop, five native 512-pixel regions, aspect-preserving "
        "224-pixel letterboxing, ImageNet normalization, mean-pooled ResNet18 "
        "embedding, saved scaler and MLP. It receives no EXIF or folder metadata "
        "as classifier features. Background masks remain disabled.",
        "",
        "```sh",
        "uv sync --locked",
        "uv run --locked poppypure predict sample.jpg --device cpu",
        "```",
        "",
        "Use `--model-dir /path/to/poppypure-v1` for a copied model bundle. "
        "Prediction requires the installed package, model bundle and input image; "
        "it does not require Backup Plus, training caches or "
        "pretrained-weight downloads.",
        "",
        "To reproduce Phase 6 from the verified earlier-phase artifacts in a fresh "
        "artifact workspace:",
        "",
        "```sh",
        "uv run --locked poppypure refit-final",
        "uv run --locked poppypure evaluate-final --device auto",
        "uv run --locked poppypure report-final",
        "```",
        "",
        "Refit refuses existing final-model/run directories; evaluation refuses "
        "repeated test access. Reporting reads saved predictions and does not "
        "rescore models. Exact source/config/environment snapshots and checksums "
        "are stored with the run.",
        "",
        f"Evaluated checkpoint SHA-256: `{metadata['checkpoint_sha256']}`.",
        "",
        "## Supporting artifacts",
        "",
        "- [Figures](plots/README.md).",
        "- [Model card](MODEL_CARD.md).",
        "- [Dataset card](DATASET_CARD.md).",
        "- [Saved test metrics](metrics.json).",
        "- Model, scaler and preprocessing: `models/poppypure-v1/`.",
        "- Refit trace, membership, pretest seal, test predictions and records: "
        "`experiments/runs/phase6-v1/`.",
        "",
    ]
    (report / "FINAL_REPORT.md").write_text("\n".join(lines))
    card = [
        "# Model card — poppypure-v1",
        "",
        "## Task",
        "",
        "Seven-class classification of semolina adulteration in poppy seed "
        "close-up images. Labels are percentages **by volume**: 0, 10, 20, 30, 40, "
        "50 and 100. Zero is pure poppy; 100 is pure semolina.",
        "",
        "## Model and fitting",
        "",
        "ImageNet V1 ResNet18 with frozen weights/BatchNorm; mean of five native "
        "detail-region embeddings; a 466-image fitted StandardScaler; MLP "
        "512 → 128 → ReLU → dropout 0.3 → 7 logits. Final fitting uses seed 42, "
        "AdamW lr 0.001, weight decay 0.0001, batch 32 and 28 fixed epochs. "
        "No refit validation, augmentation or fine-tuning.",
        "",
        "## Evaluation",
        "",
        f"The exact packaged checkpoint scored **{primary['accuracy']:.2%} accuracy "
        f"({correct}/82), {primary['macro_f1']:.4f} macro F1** on the reserved test "
        "set after selection and refitting. The checkpoint was sealed before test "
        "access and remains unchanged. See FINAL_REPORT.md for class errors, "
        "binary detection, condition summaries and bootstrap intervals.",
        "",
        "## Input/output",
        "",
        "Input: an original close-up RGB-compatible image with enough resolution "
        "for five distinct 512-pixel detail regions after cropping. EXIF is used "
        "for orientation only. Output: model version, one documented by-volume "
        "class, derived adulteration flag and seven softmax probabilities. "
        "Probabilities are uncalibrated; the output is a discrete class.",
        "",
        "## Provenance",
        "",
        "Dataset: Sarkar, Mukherjee and Chatterjee (2023), Mendeley Data "
        "[10.17632/p2gvvnh8y4.1](https://doi.org/10.17632/p2gvvnh8y4.1), CC BY 4.0. "
        "Related label provenance: Ganguly and Sarkar (2026), "
        "[10.17632/nx8m9kxy26.1](https://doi.org/10.17632/nx8m9kxy26.1). "
        "Source images retain their attribution; crop/feature/model outputs are "
        "PoppyPure v2 additions. Preparation units follow user confirmation, with "
        "the verified same-scene pair kept together.",
        "",
        "Encoder: Torchvision ResNet18, explicit `IMAGENET1K_V1`; weight file "
        "`resnet18-f37072fd.pth`. Runtime package versions are pinned in the "
        "included pyproject.toml/uv.lock; exact fitting source is snapshotted.",
        "",
        f"Checkpoint SHA-256: `{metadata['checkpoint_sha256']}`.",
        "",
    ]
    (report / "MODEL_CARD.md").write_text("\n".join(card))
    shutil.copyfile(report / "MODEL_CARD.md", bundle / "MODEL_CARD.md")
    manifest_rows = list(manifest.values())
    included = [r for r in manifest_rows if r["decision"] == "included"]
    dataset = [
        "# Final dataset card",
        "",
        "## Task and source",
        "",
        "Seven semolina adulteration classes **by volume**: 0, 10, 20, 30, 40, 50 "
        "and 100. Source: Sarkar, Tanmay; Mukherjee, Alok; Chatterjee, Kingshuk "
        "(2023), *Digital Image dataset for Poppy seed adulteration with Semolina*, "
        "Mendeley Data v1, [DOI](https://doi.org/10.17632/p2gvvnh8y4.1), CC BY 4.0.",
        "",
        "The related Ganguly/Sarkar (2026) dataset "
        "[DOI](https://doi.org/10.17632/nx8m9kxy26.1) supplies label provenance; "
        "overlapping images were not added as extra independent observations.",
        "",
        "## Inventory and allocation",
        "",
        f"Original JPEGs: **{len(manifest_rows)}**. Included close-ups: "
        f"**{len(included)}**. Seven extra exact copies and fourteen loose "
        "whole-tray reference images are excluded. The reviewed shifted-view "
        "pair is retained within one preparation group. Raw bytes are preserved "
        "on Backup Plus through `data/raw` and `data/source_archives` symlinks.",
        "",
        "Development allocation: 384 training / 82 validation / 82 test. "
        f"Final fitting: 466 images / {metadata['refit_preparation_units']} "
        f"preparation units; test: 82 images / {metrics['preparation_units']} units. "
        "Preparation identities follow the user's confirmation of individually "
        "prepared mixtures, with the known same-scene pair merged. All seven "
        "classes and all four black/white × 4x/micro conditions appear in every "
        "original partition.",
        "",
        f"Included magnification-conflict flags: "
        f"{sum(r['magnification_conflict'] == 'True' for r in included)}. "
        "These remain in overall fitting/evaluation and are omitted only from "
        "trusted magnification summaries. No labels or split membership were "
        "changed using model results.",
        "",
        "## Integrity and transformations",
        "",
        f"Manifest SHA-256: `{metadata['manifest_sha256']}`.",
        "",
        f"Split SHA-256: `{metadata['split_sha256']}`.",
        "",
        "Image bytes are checked against the frozen manifest. Model processing "
        "uses RGB/orientation normalization, the fixed material crop, five native "
        "detail regions and normalized CNN inputs. No image-area estimate is "
        "converted into a volume label. Labels remain inherited source classes.",
        "",
        "See [the original dataset audit](../phase1/DATASET_CARD.md), "
        "[split protocol](../phase2/TRAIN_VAL_TEST.md) and "
        "[final evaluation](FINAL_REPORT.md).",
        "",
    ]
    (report / "DATASET_CARD.md").write_text("\n".join(dataset))
    shutil.copyfile(test / "metrics.json", report / "metrics.json")
    shutil.copyfile(test / "bootstrap.json", report / "bootstrap.json")
    figure_index = [
        "# Final figures",
        "",
        "PNG and PDF versions are provided. Refit curves contain training-only "
        "metrics; the test set is scored once after fitting, without epoch selection.",
        "",
    ]
    for name in sorted(artifacts):
        if name.endswith(".png"):
            stem = Path(name).stem
            figure_index.append(f"- [{stem}]({stem}.png) · [PDF]({stem}.pdf)")
    (plots / "README.md").write_text("\n".join(figure_index) + "\n")
    (plots / "checksums.json").write_bytes(encoded(artifacts))
    summary = {
        "status": "complete",
        "model_version": metadata["model_version"],
        "refit_images": 466,
        "refit_epochs": 28,
        "validation_during_refit": False,
        "test_images": 82,
        "primary": primary,
        "reference": reference,
        "primary_checkpoint_sha256": metadata["checkpoint_sha256"],
        "test_errors": len(errors),
        "figure_files": len(artifacts),
        "figures": artifacts,
        "source_test_record_sha256": sha256_file(test / "record.json"),
        "test_used": True,
    }
    (report / "summary.json").write_bytes(encoded(summary))
    return summary
