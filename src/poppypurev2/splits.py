"""Stratified train/validation/test allocation with preparation groups intact."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from poppypurev2.acquisition import sha256_file
from poppypurev2.evaluation import assign_conditions, encoded, preparation_units


def stratum(row: dict) -> tuple[int, str, str]:
    return int(row["volume_percent"]), row["background"], row["magnification"]


def quotas(counts: dict, total: int, rng: random.Random) -> dict:
    """Largest-remainder allocation for exact overall counts; seeded tie breaks."""
    size = sum(counts.values())
    result = {key: count * total // size for key, count in counts.items()}
    priorities = [(counts[key] * total % size, rng.random(), key) for key in counts]
    for _, _, key in sorted(priorities, reverse=True)[: total - sum(result.values())]:
        result[key] += 1
    return result


def choose_groups(groups: list[list[dict]], target: int) -> list[list[dict]]:
    """Subset sum preserves whole related-view groups while meeting image counts."""
    possibilities = {0: []}
    for index, group in enumerate(groups):
        for count, selected in list(possibilities.items()):
            new_count = count + len(group)
            if new_count <= target and new_count not in possibilities:
                possibilities[new_count] = [*selected, index]
    if target not in possibilities:
        raise ValueError("Requested allocation cannot preserve preparation groups")
    return [groups[index] for index in possibilities[target]]


def stratified_assignments(rows: list[dict], pairs: dict, seed: int) -> list[dict]:
    units = preparation_units(assign_conditions(rows, "v3"), pairs)
    grouped = defaultdict(list)
    for row in units:
        grouped[row["preparation_group_id"]].append(row)
    strata = defaultdict(list)
    for group in grouped.values():
        if len({stratum(row) for row in group}) != 1:
            raise ValueError("Related views cross class/capture strata")
        strata[stratum(group[0])].append(group)
    rng = random.Random(seed)
    counts = {key: sum(map(len, strata[key])) for key in sorted(strata)}
    held_size = round(len(rows) * 0.15)
    test_quotas = quotas(counts, held_size, rng)
    val_quotas = quotas(counts, held_size, rng)
    assignments = []
    for key in sorted(strata):
        available = sorted(strata[key], key=lambda group: group[0]["image_id"])
        rng.shuffle(available)
        for partition, target in (
            ("test", test_quotas[key]),
            ("validation", val_quotas[key]),
            ("train", None),
        ):
            chosen = available if target is None else choose_groups(available, target)
            selected = {g[0]["preparation_group_id"] for g in chosen}
            for group in chosen:
                for row in group:
                    assignments.append(
                        {
                            **row,
                            "condition_group_id": row["evaluation_group_id"],
                            "evaluation_group_id": row["preparation_group_id"],
                            "partition": partition,
                            "validation_fold": "",
                        }
                    )
            available = [
                g for g in available if g[0]["preparation_group_id"] not in selected
            ]
    return sorted(assignments, key=lambda row: row["image_id"])


def validate_split(rows: list[dict], assignments: list[dict], pairs: dict) -> None:
    source = {row["image_id"]: row for row in rows}
    if len(source) != len(rows) or len(assignments) != len(rows):
        raise ValueError("Duplicate IDs or incorrect coverage")
    if len({r["image_id"] for r in assignments}) != len(rows):
        raise ValueError("Repeated assignment")
    if {r["image_id"] for r in assignments} != set(source):
        raise ValueError("Assignments do not cover the included manifest")
    for key in ("sha256", "pixel_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate included {key}")
    membership = defaultdict(set)
    for row in assignments:
        if stratum(row) != stratum(source[row["image_id"]]):
            raise ValueError("Changed source labels or conditions")
        membership[row["preparation_group_id"]].add(row["partition"])
    if any(len(partitions) != 1 for partitions in membership.values()):
        raise ValueError("Preparation group crosses partitions")
    by_id = {row["image_id"]: row for row in assignments}
    for pair, decision in pairs.items():
        if decision != "same_scene_shifted_view_keep_in_same_group":
            raise ValueError("Unresolved pair review")
        left, right = pair.split("|")
        if by_id[left]["preparation_group_id"] != by_id[right]["preparation_group_id"]:
            raise ValueError("Known same-scene views separated")
    all_strata = {stratum(row) for row in rows}
    if len(all_strata) != 28:
        raise ValueError("Expected all seven classes times four capture conditions")
    if {r["partition"] for r in assignments} != {"train", "validation", "test"}:
        raise ValueError("Unknown partition")
    for partition in ("train", "validation", "test"):
        if {
            stratum(r) for r in assignments if r["partition"] == partition
        } != all_strata:
            raise ValueError(f"Missing class/condition stratum in {partition}")


def freeze_stratified(project_root: Path, seed: int = 42) -> dict:
    data = project_root / "data"
    dataset = json.loads((data / "dataset_freeze.json").read_text())
    for name, key in (
        ("manifest.csv", "manifest_sha256"),
        ("review_decisions.json", "review_decisions_sha256"),
    ):
        if sha256_file(data / name) != dataset[key]:
            raise ValueError(f"Frozen Phase 1 input changed: {name}")
    provenance = json.loads((data / "preparation_provenance.json").read_text())
    if provenance["source"] != "user_confirmation":
        raise ValueError("Expected recorded user preparation confirmation")
    pairs = json.loads((data / "review_decisions.json").read_text())["pairs"]
    if {"|".join(sorted(p)) for p in provenance["known_same_scene_pairs"]} != set(
        pairs
    ):
        raise ValueError("Preparation provenance differs from reviewed pairs")
    with (data / "manifest.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["decision"] == "included"]
    if len(rows) != dataset["included_images"]:
        raise ValueError("Included count differs from data freeze")
    assignments = stratified_assignments(rows, pairs, seed)
    validate_split(rows, assignments, pairs)
    counts = {}
    for partition in ("train", "validation", "test"):
        subset = [r for r in assignments if r["partition"] == partition]
        counts[partition] = {
            "images": len(subset),
            "preparation_units": len({r["preparation_group_id"] for r in subset}),
            "by_class": dict(
                sorted(Counter(str(r["volume_percent"]) for r in subset).items())
            ),
            "by_condition": dict(
                sorted(
                    Counter(
                        f"{r['background']}_{r['magnification']}" for r in subset
                    ).items()
                )
            ),
            "by_class_condition": dict(
                sorted(
                    Counter(
                        f"{int(r['volume_percent']):03d}_{r['background']}_{r['magnification']}"
                        for r in subset
                    ).items()
                )
            ),
        }
    protocol = {
        "schema_version": 1,
        "protocol_id": "stratified_preparation_train_validation_test_v4",
        "supersedes": "user_confirmed_preparations_condition_holdout_v3",
        "seed": seed,
        "label_meaning": "semolina adulteration percentage by volume",
        "classes_volume_percent": [0, 10, 20, 30, 40, 50, 100],
        "manifest_sha256": dataset["manifest_sha256"],
        "review_decisions_sha256": dataset["review_decisions_sha256"],
        "preparation_provenance_sha256": sha256_file(
            data / "preparation_provenance.json"
        ),
        "allocation": (
            "70/15/15, stratified by volume class x background x folder magnification"
        ),
        "algorithm": (
            "Largest-remainder per-stratum image quotas; seeded group "
            "order; subset-sum allocation test then validation then train"
        ),
        "group_rule": (
            "User-confirmed individual preparations; merge confirmed "
            "same-scene views; legacy class-wide groups are historical only"
        ),
        "counts": counts,
        "primary_metric": "seven-class macro_f1 on validation",
        "supporting_metrics": [
            "balanced_accuracy",
            "per_class_precision_recall_f1",
            "confusion_matrix",
            "volume_mae_pp",
            "binary_sensitivity_specificity",
        ],
        "metric_definitions": (
            "poppypurev2.evaluation.classification_metrics; fixed "
            "seven-label order; binary prediction 0 versus >0"
        ),
        "model_selection": (
            "Fit on train only; select processing/model on validation; no "
            "test scoring or visual model-error review before final recipe "
            "lock"
        ),
        "final_fit": (
            "Lock recipe and training schedule, then optionally refit on "
            "train+validation (466 images), score once on test (82 images)"
        ),
        "early_stopping": (
            "Validation may select epoch count; fix that count before "
            "train+validation refit"
        ),
        "tie_break": (
            "Within 0.005 validation macro F1 prefer fewer fitted "
            "parameters, then shorter measured training time"
        ),
        "leakage_rules": [
            (
                "Design crops/masks using training images; fit "
                "scalers/PCA/selectors and other learned transforms on train "
                "only during selection"
            ),
            (
                "Split before augmentation or tiling; descendants inherit "
                "source preparation and partition; one image-level prediction"
            ),
            (
                "No EXIF, camera dates, filenames, paths or condition tags as "
                "visual classifier inputs"
            ),
            (
                "Exclude visible camera marks/borders only using reviewed, "
                "reproducible processing"
            ),
            (
                "Do not fit calibration on test; softmax is not a verified "
                "confidence estimate"
            ),
        ],
        "holdout_access": (
            "All images audited in Phase 1; after this freeze use training "
            "for processing design and validation for selection; test "
            "reserved for final scoring"
        ),
        "challenge_tests": (
            "Report final subgroups by background/magnification; separate "
            "condition-transfer experiments are secondary and cannot tune "
            "on main test images"
        ),
        "magnification_conflicts": (
            "Retain four flagged images by source folder for "
            "stratification; omit only from trusted-magnification summaries"
        ),
        "post_evaluation_refit": (
            "Optional separate deployment version fits unchanged recipe on "
            "all 548 images; preserve evaluated model and its test "
            "predictions"
        ),
    }
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
        "manifest_sha256": dataset["manifest_sha256"],
        "artifacts": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    files["freeze.json"] = encoded(frozen)
    destination = data / "splits/v4"
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
