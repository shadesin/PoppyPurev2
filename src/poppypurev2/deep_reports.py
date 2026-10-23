"""Training input previews and saved Phase 5 curves/results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from poppypurev2.acquisition import sha256_file
from poppypurev2.deep_inputs import (
    CLASSES,
    checked_image,
    development_rows,
    image_regions,
    load_settings,
)
from poppypurev2.evaluation import encoded


def training_previews(root: Path) -> None:
    config, source = load_settings(root)
    parts, manifest, _ = development_rows(root)
    output = root / "reports/phase5/inputs"
    output.mkdir(parents=True, exist_ok=False)
    mean = torch.tensor(config["preprocessing"]["mean"])[:, None, None]
    std = torch.tensor(config["preprocessing"]["std"])[:, None, None]
    selected = []
    for background in ("black", "white"):
        for magnification in ("4x", "micro"):
            sheet = Image.new("RGB", (1344, 7 * 254 + 30), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (5, 5),
                f"TRAIN {background} {magnification}: overview | five detail inputs",
                fill="black",
            )
            for index, volume in enumerate(CLASSES):
                rows = [
                    r
                    for r in parts["train"]
                    if int(r["volume_percent"]) == volume
                    and r["background"] == background
                    and r["magnification"] == magnification
                ]
                row = rows[len(rows) // 2]
                selected.append(row["image_id"])
                image = checked_image(root, manifest[row["image_id"]])
                tensors = torch.cat(
                    [
                        image_regions(image, representation, config, source)
                        for representation in ("crop", "crop_detail")
                    ]
                )
                for col, tensor in enumerate(tensors):
                    pixels = (
                        ((tensor * std + mean).permute(1, 2, 0).numpy() * 255)
                        .round()
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    sheet.paste(Image.fromarray(pixels), (col * 224, 30 + index * 254))
                draw.text(
                    (5, 30 + index * 254 + 227),
                    f"{row['image_id']} | {volume}% semolina by volume",
                    fill="black",
                )
            sheet.save(output / f"{background}_{magnification}.jpg")
    (output / "summary.json").write_bytes(
        encoded(
            {
                "partitions": ["train"],
                "review_ids": selected,
                "images": len(selected),
                "mask": "none",
                "config_sha256": sha256_file(root / "configs/phase5-v1.json"),
            }
        )
    )


def render_phase5(root: Path, runs: Path) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".plot-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((runs / "summary.json").read_text())
    output = root / "reports/phase5"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=False)
    artifacts = {}

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "pdf"):
            path = plots / f"{name}.{extension}"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            artifacts[path.name] = sha256_file(path)
        plt.close(figure)

    for result in summary["runs"]:
        with (runs / result["run_id"] / "learning_curve.csv").open() as handle:
            curve = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in curve]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, key, label in zip(
            axes,
            ("cross_entropy", "macro_f1", "accuracy"),
            ("Cross-entropy loss", "Macro F1", "Accuracy"),
            strict=True,
        ):
            for part in ("training", "validation"):
                ax.plot(epochs, [float(r[f"{part}_{key}"]) for r in curve], label=part)
            ax.axvline(
                result["selected_epoch"],
                linestyle="--",
                color="gray",
                label="selected epoch",
            )
            ax.set(xlabel="Epoch", ylabel=label)
            if key != "cross_entropy":
                ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(result["run_id"] + " — validation selects checkpoint", fontsize=10)
        save(fig, result["run_id"] + "_curves")
    winner = summary["selected_deep"]
    metrics = json.loads((runs / winner["run_id"] / "metrics.json").read_text())[
        "validation"
    ]
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set(
        xticks=range(7),
        yticks=range(7),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted semolina % by volume",
        ylabel="True semolina % by volume",
        title="Selected deep model — 82 validation images",
    )
    save(fig, "selected_deep_confusion_matrix")
    classical = json.loads(
        (root / "experiments/runs/phase4-v1/summary.json").read_text()
    )
    # Read selected Phase 4 model metrics directly to avoid assuming summary keys.
    svm = json.loads(
        (root / "experiments/runs/phase4-v1/svm_input1_C10/metrics.json").read_text()
    )
    rows = [(r["run_id"], r["validation_macro_f1"]) for r in summary["runs"]]
    rows += [("Phase 4 whole SVM", svm["validation"]["macro_f1"])]
    rows.sort(key=lambda row: row[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows])
    ax.set(
        xlim=(0, 1),
        xlabel="Validation macro F1 (seven classes)",
        title="Eight deep configurations and classical reference",
    )
    for index, (_, value) in enumerate(rows):
        ax.text(value + 0.005, index, f"{value:.4f}", va="center", fontsize=9)
    save(fig, "model_comparison")
    (plots / "checksums.json").write_bytes(encoded(artifacts))
    (output / "summary.json").write_bytes(
        encoded(
            {
                **summary,
                "classical_reference_macro_f1": svm["validation"]["macro_f1"],
                "phase4_summary_sha256": sha256_file(
                    root / "experiments/runs/phase4-v1/summary.json"
                ),
                "phase4_status": classical.get("status"),
                "figures": artifacts,
            }
        )
    )
    return {"figure_files": len(artifacts), "output": str(output)}
