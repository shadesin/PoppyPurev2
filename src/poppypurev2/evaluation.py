"""Versioned evaluation protocols and preparation provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from poppypurev2.acquisition import sha256_file
from poppypurev2.config import CLASSES_VOLUME_PERCENT


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def assign_conditions(rows: list[dict], version: str = "v1") -> list[dict]:
    """Keep entire class/background/magnification conditions together.

    These are broad evaluation blocks, not inferred physical preparations.
    Dates are deliberately not used to subdivide them.
    """
    if version not in {"v1", "v2", "v3"}:
        raise ValueError("Unknown evaluation version")
    assignments = []
    for row in rows:
        background, magnification = row["background"], row["magnification"]
        if background not in {"black", "white"} or magnification not in {"4x", "micro"}:
            raise ValueError("Unknown capture condition")
        label = int(row["volume_percent"])
        if label not in CLASSES_VOLUME_PERCENT:
            raise ValueError("Unknown volume class")
        holdout = background == "white" and (
            version == "v1" or magnification == "micro"
        )
        fold = magnification if version == "v1" else f"{background}_{magnification}"
        assignments.append(
            {
                "image_id": row["image_id"],
                "volume_percent": label,
                "background": background,
                "magnification": magnification,
                "magnification_conflict": row["magnification_conflict"],
                "preparation_group_id": row["group_id"],
                "capture_group_id": row["capture_group_id"],
                "evaluation_group_id": (
                    f"volume_{label:03d}_{background}_{magnification}"
                ),
                "partition": "holdout" if holdout else "development",
                "validation_fold": "" if holdout else fold,
            }
        )
    return sorted(assignments, key=lambda row: row["image_id"])


def preparation_units(assignments: list[dict], pairs: dict) -> list[dict]:
    """Apply user-confirmed individual preparations, merging verified same scenes."""
    parents = {r["image_id"]: r["image_id"] for r in assignments}

    def find(image_id: str) -> str:
        while parents[image_id] != image_id:
            image_id = parents[image_id]
        return image_id

    for pair in pairs:
        left, right = pair.split("|")
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)
    return [
        {
            **row,
            "legacy_conservative_group_id": row["preparation_group_id"],
            "preparation_group_id": f"preparation_{find(row['image_id'])}",
        }
        for row in assignments
    ]


def validate_assignments(
    rows: list[dict], assignments: list[dict], pairs: dict, version: str = "v1"
) -> None:
    """Check coverage, hashes, condition blocks, class support and known views."""
    by_id = {row["image_id"]: row for row in rows}
    assigned = {row["image_id"]: row for row in assignments}
    if len(by_id) != len(rows) or len(assigned) != len(assignments):
        raise ValueError("Repeated image ID")
    if set(by_id) != set(assigned):
        raise ValueError("Split coverage differs from included manifest")
    expected = assign_conditions(rows, version)
    if version == "v3":
        expected = preparation_units(expected, pairs)
    if assignments != expected:
        raise ValueError("Assignments differ from the frozen condition rule")
    for key in ("sha256", "pixel_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Included images have duplicate {key}")
    for pair, decision in pairs.items():
        if decision != "same_scene_shifted_view_keep_in_same_group":
            raise ValueError("Unresolved or unsupported pair decision")
        left, right = pair.split("|")
        if left in assigned and right in assigned:
            if (
                assigned[left]["evaluation_group_id"]
                != assigned[right]["evaluation_group_id"]
            ):
                raise ValueError("Known related views cross evaluation groups")
        else:
            raise ValueError("Reviewed pair missing from included images")
    for partition in ("development", "holdout"):
        subset = [row for row in assignments if row["partition"] == partition]
        if {row["volume_percent"] for row in subset} != set(CLASSES_VOLUME_PERCENT):
            raise ValueError(f"Missing class in {partition}")
    folds = (
        ("4x", "micro") if version == "v1" else ("black_4x", "black_micro", "white_4x")
    )
    for fold in folds:
        subset = [row for row in assignments if row["validation_fold"] == fold]
        if {row["volume_percent"] for row in subset} != set(CLASSES_VOLUME_PERCENT):
            raise ValueError(f"Missing class in development fold {fold}")
        training = [
            row
            for row in assignments
            if row["partition"] == "development" and row["validation_fold"] != fold
        ]
        if {row["volume_percent"] for row in training} != set(CLASSES_VOLUME_PERCENT):
            raise ValueError(f"Missing training class in fold {fold}")
        if version == "v1" and any(
            row["magnification_conflict"] == "True" for row in subset
        ):
            raise ValueError("Ambiguous magnification in development")


def freeze_protocol(project_root: Path, version: str = "v1", seed: int = 42) -> dict:
    """Create deterministic versioned membership and checksum freeze, or verify it."""
    if version == "v4":
        from poppypurev2.splits import freeze_stratified

        return freeze_stratified(project_root, seed)
    data = project_root / "data"
    freeze = json.loads((data / "dataset_freeze.json").read_text())
    for name, key in (
        ("manifest.csv", "manifest_sha256"),
        ("review_decisions.json", "review_decisions_sha256"),
    ):
        if sha256_file(data / name) != freeze[key]:
            raise ValueError(f"Frozen Phase 1 input changed: {name}")
    with (data / "manifest.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["decision"] == "included"]
    if len(rows) != freeze["included_images"]:
        raise ValueError("Frozen included count differs")
    review = json.loads((data / "review_decisions.json").read_text())
    assignments = assign_conditions(rows, version)
    if version == "v3":
        assignments = preparation_units(assignments, review["pairs"])
    validate_assignments(rows, assignments, review["pairs"], version)
    groups = Counter(row["group_id"] for row in rows)
    preparation_counts = {
        str(label): len(
            {r["group_id"] for r in rows if int(r["volume_percent"]) == label}
        )
        for label in CLASSES_VOLUME_PERCENT
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "black_development_white_holdout_v1",
        "label_meaning": "semolina adulteration percentage by volume",
        "classes_volume_percent": list(CLASSES_VOLUME_PERCENT),
        "manifest_sha256": freeze["manifest_sha256"],
        "review_decisions_sha256": freeze["review_decisions_sha256"],
        "scope": (
            "Internal capture-condition transfer; unseen preparations not evaluated"
        ),
        "preparation_evaluation": {
            "feasible": False,
            "conservative_groups": len(groups),
            "groups_per_class": preparation_counts,
            "reason": (
                "One conservative group per class; holding one out removes its "
                "training class. Independent preparation count is unknown."
            ),
        },
        "membership_rule": (
            "black -> development; white -> final holdout; "
            "never randomly split photographs"
        ),
        "evaluation_group_rule": (
            "volume x background x folder magnification; "
            "broad blocks, not independent preparations"
        ),
        "development_folds": [
            {"fold": "4x", "train": "black micro", "validation": "black 4x"},
            {"fold": "micro", "train": "black 4x", "validation": "black micro"},
        ],
        "selection": {
            "primary_metric": (
                "mean of two development-fold seven-class macro_f1 values"
            ),
            "tie_break": (
                "If within 0.005 macro F1, prefer fewer fitted parameters, "
                "then shorter measured training time"
            ),
            "final_fit": (
                "Refit locked pipeline on all black images; white images never fitted"
            ),
            "early_stopping": (
                "Use a preregistered fixed epoch schedule; do not use outer "
                "validation fold for checkpoint selection"
            ),
        },
        "metrics": {
            "macro_f1": "All seven labels, zero_division=0",
            "balanced_accuracy": (
                "Mean class recall; undefined when any true class missing"
            ),
            "per_class": "precision, recall, F1, support",
            "confusion_matrix": "Rows=true, columns=predicted; fixed class order",
            "volume_mae_pp": (
                "Mean absolute difference between predicted and true volume class; "
                "not continuous concentration estimation"
            ),
            "binary": (
                "Map seven-class argmax to 0=pure, >0=adulterated; "
                "sensitivity=TP/(TP+FN), specificity=TN/(TN+FP); "
                "undefined denominators -> null"
            ),
        },
        "challenge_tests": {
            "background": (
                "One final black-to-white transfer; "
                "no reverse experiment during development"
            ),
            "magnification": (
                "Two development transfer folds; final holdout by folder tag "
                "excluding four ambiguous tags from subgroup scores only"
            ),
            "capture_session": (
                "Not a valid independent challenge: dates confounded with class "
                "and preparation unknown; descriptive counts only"
            ),
            "shortcut_controls": (
                "Phase 4 metadata-only and background-only models "
                "use the same development folds"
            ),
        },
        "uncertainty": (
            "No confidence intervals claiming independent photos or preparations; "
            "report two fold scores/range and condition-block counts descriptively"
        ),
        "holdout_access": (
            "All images visually audited in Phase 1. No white-image feature design, "
            "mask review, tuning, training, calibration, or model selection after "
            "this freeze. Score locked winner once in Phase 6; preserve predictions. "
            "Any later change requires a new exploratory result with disclosed reuse."
        ),
        "leakage_rules": [
            (
                "Fit scalers, PCA, feature selectors, learned preprocessing "
                "and class weights on each training fold only"
            ),
            (
                "Split before tiles/crops/augmentation; descendants inherit source "
                "membership; aggregate tiles to one image prediction"
            ),
            (
                "Known same-scene pair stays together; "
                "unknown overlap across blocks cannot be ruled out"
            ),
            (
                "No date, filename, path, EXIF or background tags "
                "as inputs to the visual classifier"
            ),
            (
                "Do not calibrate probabilities or claim verified confidence "
                "without independent calibration data"
            ),
        ],
        "counts": {
            partition: {
                "images": sum(row["partition"] == partition for row in assignments),
                "by_class": dict(
                    sorted(
                        Counter(
                            str(row["volume_percent"])
                            for row in assignments
                            if row["partition"] == partition
                        ).items()
                    )
                ),
                "condition_blocks": len(
                    {
                        row["evaluation_group_id"]
                        for row in assignments
                        if row["partition"] == partition
                    }
                ),
            }
            for partition in ("development", "holdout")
        },
    }
    if version in {"v2", "v3"}:
        protocol["protocol_id"] = "three_conditions_development_white_micro_holdout_v2"
        protocol["supersedes"] = "black_development_white_holdout_v1"
        protocol["revision_reason"] = (
            "User challenged underuse of training data. Revised before any model "
            "fitting, feature design or test scoring; development now covers both "
            "backgrounds and folder magnifications."
        )
        protocol["membership_rule"] = (
            "white Micro -> final holdout (134); black 4x, black Micro, "
            "white 4x -> development (414); no random photograph split"
        )
        protocol["development_folds"] = [
            {
                "fold": fold,
                "train": "other two development conditions",
                "validation": fold,
            }
            for fold in ("black_4x", "black_micro", "white_4x")
        ]
        protocol["selection"]["primary_metric"] = (
            "mean of three development-fold seven-class macro_f1 values"
        )
        protocol["selection"]["final_fit"] = (
            "Refit locked pipeline on all 414 development images; "
            "134 white Micro holdout images never fitted"
        )
        protocol["challenge_tests"]["background"] = (
            "Development white 4x fold checks black-to-white transfer with "
            "magnification shift mixed in; no isolated full-background final test"
        )
        protocol["challenge_tests"]["magnification"] = (
            "Three development condition-transfer folds; final test is held-out "
            "white Micro combination, not pure magnification transfer. Four "
            "conflicting white 4x tags retained by folder membership; omit from "
            "magnification-specific summaries only, retain all main fold scores."
        )
        protocol["uncertainty"] = (
            "No confidence intervals claiming independent photos or preparations; "
            "report three fold scores/range and condition-block counts descriptively"
        )
        protocol["holdout_access"] = (
            "All images visually audited in Phase 1. No model fitting, processing "
            "design or scoring preceded this revision. White 4x is now development; "
            "white Micro remains held out. Do not use white Micro for mask review, "
            "feature design, tuning, training, calibration or model selection. Score "
            "locked winner once in Phase 6; later reuse is exploratory."
        )
        protocol["post_evaluation_refit"] = (
            "After preserving final evaluation and its 414-image model, a separately "
            "versioned deployment model may refit the unchanged recipe on all 548 "
            "images. It has no untouched test set; never attach the prior score to "
            "it as a direct evaluation."
        )
    if version == "v3":
        provenance_path = data / "preparation_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if provenance["source"] != "user_confirmation":
            raise ValueError("Preparation provenance must identify its source")
        declared_pairs = {
            "|".join(sorted(pair)) for pair in provenance["known_same_scene_pairs"]
        }
        if declared_pairs != set(review["pairs"]):
            raise ValueError("Preparation provenance differs from reviewed scenes")
        protocol["protocol_id"] = "user_confirmed_preparations_condition_holdout_v3"
        protocol["supersedes"] = "three_conditions_development_white_micro_holdout_v2"
        protocol["revision_reason"] = (
            "User confirms images represent individually prepared mixtures; "
            "replace missing-preparation assumption while preserving deduplication."
        )
        protocol["preparation_provenance"] = {
            "source": provenance["source"],
            "path": "data/preparation_provenance.json",
            "sha256": sha256_file(provenance_path),
        }
        protocol["scope"] = (
            "Seven-class by-volume evaluation using user-confirmed individual "
            "preparations, with a held-out white Micro capture condition"
        )
        protocol["preparation_evaluation"] = {
            "feasible": True,
            "source": "user_confirmation",
            "preparation_units": len({r["preparation_group_id"] for r in assignments}),
            "group_rule": (
                "One retained image per unit; confirmed same-scene views merged"
            ),
        }
        protocol["evaluation_group_rule"] = (
            "Class/background/folder condition blocks; preparation units are "
            "separate and derived from user confirmation plus reviewed scenes"
        )
        protocol["challenge_tests"]["capture_session"] = (
            "Capture dates are class-confounded; descriptive date counts only"
        )
        protocol["uncertainty"] = (
            "Report all three fold scores and condition summaries. "
            "If bootstrapping final metrics, resample preparation units."
        )
        protocol["leakage_rules"][2] = (
            "Known same-scene views share one preparation unit and evaluation block"
        )
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(assignments[0]))
    writer.writeheader()
    writer.writerows(assignments)
    files = {
        "assignments.csv": buffer.getvalue().encode(),
        "protocol.json": encoded(protocol),
    }
    frozen = {
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": freeze["manifest_sha256"],
        "artifacts": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    files["freeze.json"] = encoded(frozen)
    destination = data / "splits" / version
    # Validate all existing files first; reruns only verify identical bytes.
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing changed protocol artifact: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(content)
    return frozen


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Score discrete volume classes; missing denominators are explicit nulls."""
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or predicted.shape != true.shape or not len(true):
        raise ValueError("Expected nonempty matching one-dimensional labels")
    if not set(true).union(predicted).issubset(CLASSES_VOLUME_PERCENT):
        raise ValueError("Predictions and targets must be seven-class volume labels")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=CLASSES_VOLUME_PERCENT, zero_division=0
    )
    binary = confusion_matrix(true > 0, predicted > 0, labels=[False, True])
    tn, fp, fn, tp = (int(value) for value in binary.ravel())
    return {
        "classes_volume_percent": list(CLASSES_VOLUME_PERCENT),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()) if all(support) else None,
        "per_class": {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(
                CLASSES_VOLUME_PERCENT, precision, recall, f1, support, strict=True
            )
        },
        "confusion_matrix": confusion_matrix(
            true, predicted, labels=CLASSES_VOLUME_PERCENT
        ).tolist(),
        "volume_mae_pp": float(
            np.abs(true.astype(float) - predicted.astype(float)).mean()
        ),
        "binary": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
        },
    }
