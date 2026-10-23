"""Inventory original images, flag duplicates, and produce review contact sheets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from poppypurev2.acquisition import sha256_file
from poppypurev2.config import CLASSES_VOLUME_PERCENT


def labels_from_path(name: str) -> dict:
    parts = Path(name).parts
    class_folder = parts[1]
    matched = re.fullmatch(r"(10|20|30|40|50)_", class_folder)
    if matched:
        volume = int(matched[1])
    elif class_folder.startswith("Poppy seed background"):
        volume = 0
    elif class_folder.startswith("Suji background"):
        volume = 100
    else:
        raise ValueError(f"Unrecognized class folder: {name}")
    condition = "/".join(parts[1:-1]).lower()
    if "white" in condition or "white background" in parts[-1].lower():
        background = "white"
    else:
        background = "black"
    magnification = next(
        (x.lower() for x in parts[2:-1] if x.lower() in ("4x", "micro")),
        "unspecified",
    )
    return {
        "volume_percent": volume,
        "background": background,
        "magnification": magnification,
        "layout": "loose" if len(parts) == 3 else "structured",
    }


def perceptual_hashes(rgb: Image.Image) -> tuple[str, str]:
    grey = rgb.convert("L")
    small = np.asarray(grey.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    x = np.arange(32)
    basis = np.cos(np.pi / 32 * (x + 0.5) * x[:, None])
    low = (basis @ small @ basis.T)[:8, :8].flatten()
    # DC brightness is excluded from the median and set to zero in the hash.
    bits = low > np.median(low[1:])
    bits[0] = False
    tiny = np.asarray(grey.resize((9, 8), Image.Resampling.LANCZOS))
    differences = tiny[:, 1:] > tiny[:, :-1]
    return (
        f"{int(''.join(str(int(v)) for v in bits), 2):016x}",
        f"{int(''.join(str(int(v)) for v in differences.flat), 2):016x}",
    )


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        exif = image.getexif()
        extra = exif.get_ifd(34665) if 34665 in exif else {}
        capture = str(extra.get(36867, exif.get(306, ""))).replace("\x00", "").strip()
        camera = " ".join(
            str(exif.get(k, "")).replace("\x00", "").strip() for k in (271, 272)
        ).strip()
        orientation = exif.get(274, 1)
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.load()
        width, height = rgb.size
        pixel_sha = hashlib.sha256(
            f"{width}x{height}:RGB:".encode() + rgb.tobytes()
        ).hexdigest()
        phash, dhash = perceptual_hashes(rgb)
        preview = rgb.copy()
        preview.thumbnail((512, 512))
        grey = np.asarray(preview.convert("L"), dtype=float)
        laplacian = (
            -4 * grey[1:-1, 1:-1]
            + grey[:-2, 1:-1]
            + grey[2:, 1:-1]
            + grey[1:-1, :-2]
            + grey[1:-1, 2:]
        )
        return {
            "decode_status": "ok",
            "decode_error": "",
            "width": width,
            "height": height,
            "exif_orientation": orientation,
            "exif_capture_time": capture,
            "camera": camera,
            "pixel_sha256": pixel_sha,
            "phash": phash,
            "dhash": dhash,
            "sharpness_512": round(float(laplacian.var()), 4),
            "dark_fraction_512": round(float((grey <= 5).mean()), 6),
            "bright_fraction_512": round(float((grey >= 250).mean()), 6),
        }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or (list(rows[0]) if rows else [])
        )
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(
    rows: list[dict],
    raw: Path,
    output: Path,
    title: str,
    columns: int = 8,
    cell_width: int = 190,
    cell_height: int = 165,
) -> None:
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width,
            40 + ((len(rows) + columns - 1) // columns) * cell_height,
        ),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    for index, row in enumerate(rows):
        left = index % columns * cell_width
        top = 40 + index // columns * cell_height
        if row["decode_status"] == "ok":
            with Image.open(raw / row["source_path"]) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((cell_width - 6, cell_height - 40))
                canvas.paste(preview, (left + 3, top))
        caption = (
            f"{row['image_id']} {row['volume_percent']}%\n"
            f"{row['background']} {row['magnification']} {row['layout']}"
        )
        draw.text((left + 3, top + cell_height - 37), caption, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_audit(project_root: Path) -> dict:
    data, report = project_root / "data", project_root / "reports/phase1"
    report.mkdir(parents=True, exist_ok=True)
    raw = data / "raw"
    source = json.loads((data / "sources.json").read_text())["canonical"]
    archive = data / "source_archives" / source["local_filename"]
    if sha256_file(archive) != source["sha256"]:
        raise ValueError("Source archive failed SHA-256 verification")
    overrides_path = data / "review_decisions.json"
    overrides = (
        json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    )
    decisions = overrides.get("images", {})
    entries, rows = [], []
    with zipfile.ZipFile(archive) as zipped:
        for info in sorted(zipped.infolist(), key=lambda x: x.filename):
            is_image = Path(info.filename).suffix.lower() in (".jpg", ".jpeg", ".png")
            entries.append(
                {
                    "source_path": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "entry_type": "directory"
                    if info.is_dir()
                    else ("image" if is_image else "other"),
                }
            )
            if not is_image or info.is_dir():
                continue
            path = raw / info.filename
            labels = labels_from_path(info.filename)
            row = {
                "image_id": f"img_{len(rows) + 1:04d}",
                "source_path": info.filename,
                **labels,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            with zipped.open(info) as stream:
                original_digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    original_digest.update(block)
            if original_digest.hexdigest() != row["sha256"]:
                raise ValueError(
                    f"Extracted image differs from archive: {info.filename}"
                )
            try:
                row.update(inspect_image(path))
            except (OSError, ValueError, SyntaxError) as error:
                row.update(
                    {
                        key: ""
                        for key in (
                            "width",
                            "height",
                            "exif_orientation",
                            "exif_capture_time",
                            "camera",
                            "pixel_sha256",
                            "phash",
                            "dhash",
                            "sharpness_512",
                            "dark_fraction_512",
                            "bright_fraction_512",
                        )
                    }
                )
                row.update(decode_status="error", decode_error=str(error))
            filename_date = re.search(r"(?:IMG_|PXL_)(\d{8})", path.name)
            date = row["exif_capture_time"][:10].replace(":", "-")
            if not date and filename_date:
                digits = filename_date[1]
                date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
            row["capture_date"] = date or "unknown"
            row["capture_date_source"] = (
                "exif"
                if row["exif_capture_time"]
                else ("filename" if filename_date else "unknown")
            )
            # Preparation identities are absent: do not pretend photos are samples.
            row["group_id"] = (
                f"volume_{labels['volume_percent']:03d}_preparation_unknown"
            )
            row["group_evidence"] = (
                "conservative: all views of a volume class; "
                "physical preparation IDs unavailable"
            )
            row["group_confidence"] = "uncertain"
            row["capture_group_id"] = (
                f"volume_{labels['volume_percent']:03d}_{row['capture_date']}"
            )
            row["decision"] = (
                "included" if labels["layout"] == "structured" else "pending_review"
            )
            row["decision_reason"] = (
                "valid structured class label"
                if labels["layout"] == "structured"
                else "loose image requires individual review"
            )
            if row["decode_status"] != "ok":
                row.update(decision="excluded", decision_reason="image decode failed")
            if info.filename in decisions:
                row.update(decisions[info.filename])
            row["quality_flags"] = ""
            row["magnification_conflict"] = False
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"Audited {len(rows)} images", flush=True)

    by_sha, by_pixels = defaultdict(list), defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
        if row["pixel_sha256"]:
            by_pixels[row["pixel_sha256"]].append(row)
    duplicates = []
    for digest, members in by_sha.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (r["layout"] != "structured", r["source_path"]))
        canonical = members[0]
        conflict = len({r["volume_percent"] for r in members}) > 1
        magnification_conflict = len({r["magnification"] for r in members}) > 1
        for row in members:
            row["magnification_conflict"] = magnification_conflict
            duplicates.append(
                {
                    "sha256": digest,
                    "image_id": row["image_id"],
                    "canonical_image_id": canonical["image_id"],
                    "volume_percent": row["volume_percent"],
                    "source_path": row["source_path"],
                    "label_conflict": conflict,
                    "magnification_conflict": magnification_conflict,
                }
            )
            if conflict:
                row.update(
                    decision="pending_review",
                    decision_reason="exact duplicate has conflicting labels",
                )
            elif row is not canonical:
                row.update(
                    decision="excluded",
                    decision_reason=f"exact duplicate of {canonical['image_id']}",
                )
    conditions = defaultdict(list)
    for row in rows:
        if row["decode_status"] == "ok":
            conditions[(row["background"], row["magnification"])].append(row)
    for condition_rows in conditions.values():
        cutoff = float(np.quantile([r["sharpness_512"] for r in condition_rows], 0.05))
        for row in condition_rows:
            flags = []
            if row["sharpness_512"] <= cutoff:
                flags.append("low_sharpness_relative_to_condition")
            if row["bright_fraction_512"] > 0.25:
                flags.append("many_near_white_pixels")
            if row["dark_fraction_512"] > 0.25:
                flags.append("many_near_black_pixels")
            row["quality_flags"] = ";".join(flags)

    unique = [
        members[0] for members in by_sha.values() if members[0]["decode_status"] == "ok"
    ]
    pairs = []
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            phash = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
            dhash = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            pixel_equal = left["pixel_sha256"] == right["pixel_sha256"]
            if phash <= 8 or dhash <= 4 or pixel_equal:
                key = "|".join(sorted((left["image_id"], right["image_id"])))
                review = overrides.get("pairs", {}).get(key, "pending_review")
                pairs.append(
                    {
                        "left_image_id": left["image_id"],
                        "right_image_id": right["image_id"],
                        "phash_distance": phash,
                        "dhash_distance": dhash,
                        "same_pixels": pixel_equal,
                        "same_label": left["volume_percent"] == right["volume_percent"],
                        "review_decision": review,
                    }
                )
    pairs.sort(key=lambda r: (r["phash_distance"], r["dhash_distance"]))
    pixel_duplicates = [
        {"pixel_sha256": digest, "image_ids": ";".join(r["image_id"] for r in members)}
        for digest, members in by_pixels.items()
        if len({r["sha256"] for r in members}) > 1
    ]
    for members in by_pixels.values():
        if len({r["sha256"] for r in members}) <= 1:
            continue
        members.sort(key=lambda r: (r["layout"] != "structured", r["source_path"]))
        conflict = len({r["volume_percent"] for r in members}) > 1
        for row in members if conflict else members[1:]:
            row.update(
                decision="pending_review" if conflict else "excluded",
                decision_reason=(
                    "identical decoded pixels have conflicting labels"
                    if conflict
                    else f"identical decoded pixels to {members[0]['image_id']}"
                ),
            )
    write_csv(data / "archive_inventory.csv", entries)
    write_csv(data / "manifest.csv", rows)
    write_csv(
        data / "image_decisions.csv",
        [
            {
                k: r[k]
                for k in (
                    "image_id",
                    "source_path",
                    "volume_percent",
                    "decision",
                    "decision_reason",
                    "quality_flags",
                )
            }
            for r in rows
        ],
    )
    write_csv(report / "exact_duplicates.csv", duplicates)
    write_csv(
        report / "pixel_duplicates.csv",
        pixel_duplicates,
        ["pixel_sha256", "image_ids"],
    )
    write_csv(report / "near_duplicate_candidates.csv", pairs)
    write_csv(
        report / "groups.csv",
        [
            {
                "group_id": f"volume_{volume:03d}_preparation_unknown",
                "volume_percent": volume,
                "included_images": sum(
                    r["volume_percent"] == volume and r["decision"] == "included"
                    for r in rows
                ),
                "physical_preparations": "unknown; not an independent sample count",
            }
            for volume in CLASSES_VOLUME_PERCENT
        ],
    )
    for volume in CLASSES_VOLUME_PERCENT:
        selected = [r for r in rows if r["volume_percent"] == volume]
        contact_sheet(
            selected,
            raw,
            report / f"class_{volume:03d}.jpg",
            f"Class {volume}% by volume - every original image",
        )
    loose = [r for r in rows if r["layout"] == "loose"]
    contact_sheet(
        loose,
        raw,
        report / "loose_images.jpg",
        "All loose images: individual review",
        columns=4,
        cell_width=400,
        cell_height=300,
    )
    lookup = {r["image_id"]: r for r in rows}
    for start in range(0, len(pairs), 12):
        pair_rows = [
            lookup[p[key]]
            for p in pairs[start : start + 12]
            for key in ("left_image_id", "right_image_id")
        ]
        contact_sheet(
            pair_rows,
            raw,
            report / f"near_pairs_{start // 12 + 1:03d}.jpg",
            f"Near-duplicate candidates {start + 1}-"
            f"{min(start + 12, len(pairs))}; adjacent pairs",
            columns=4,
            cell_width=380,
            cell_height=300,
        )
    summary = {
        "archive_sha256": source["sha256"],
        "manifest_sha256": sha256_file(data / "manifest.csv"),
        "archive_entries": len(entries),
        "image_entries": len(rows),
        "layout_counts": dict(Counter(r["layout"] for r in rows)),
        "unique_byte_images": len(by_sha),
        "exact_duplicate_groups": sum(len(x) > 1 for x in by_sha.values()),
        "images_with_conflicting_magnification_folders": sum(
            r["magnification_conflict"] for r in rows
        ),
        "pixel_duplicate_groups_with_different_bytes": len(pixel_duplicates),
        "near_duplicate_candidate_pairs": len(pairs),
        "near_duplicate_reviews": dict(Counter(p["review_decision"] for p in pairs)),
        "decode_counts": dict(Counter(r["decode_status"] for r in rows)),
        "decision_counts": dict(Counter(r["decision"] for r in rows)),
        "class_counts_original": dict(Counter(r["volume_percent"] for r in rows)),
        "class_counts_included": dict(
            Counter(r["volume_percent"] for r in rows if r["decision"] == "included")
        ),
        "capture_dates": dict(Counter(r["capture_date"] for r in rows)),
        "conservative_groups": len(
            {r["group_id"] for r in rows if r["decision"] == "included"}
        ),
        "grouping_limitation": (
            "Preparation IDs are absent. Class-level conservative groups do not "
            "allow seven-class evaluation on unseen preparations. Capture groups "
            "are metadata only, not proven independent samples."
        ),
    }
    subset_inventory = data / "related_subset_inventory.csv"
    if subset_inventory.exists():
        with subset_inventory.open(newline="", encoding="utf-8") as stream:
            subset_rows = list(csv.DictReader(stream))
        original_signatures = {
            (
                entry["crc32"],
                int(entry["size_bytes"]),
                labels_from_path(entry["source_path"])["volume_percent"],
            )
            for entry in entries
            if entry["entry_type"] == "image"
        }
        subset_signatures = {
            (
                entry["crc32"],
                int(entry["size_bytes"]),
                int(Path(entry["source_path"]).parts[1]),
            )
            for entry in subset_rows
            if Path(entry["source_path"]).suffix.lower() in (".jpg", ".jpeg")
        }
        overlap = {
            "subset_image_entries": len(subset_rows),
            "subset_unique_crc_size_label_signatures": len(subset_signatures),
            "matching_original_signatures": len(
                subset_signatures & original_signatures
            ),
            "unmatched_signatures": len(subset_signatures - original_signatures),
            "method": (
                "ZIP CRC32 + uncompressed file size + volume class; "
                "strong overlap evidence, not a SHA-256 identity proof"
            ),
        }
        (report / "source_overlap.json").write_text(
            json.dumps(overlap, indent=2) + "\n", encoding="utf-8"
        )
    (report / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def freeze_dataset(project_root: Path) -> dict:
    """Freeze a fully reviewed manifest without suggesting independent samples."""
    data = project_root / "data"
    summary = json.loads((project_root / "reports/phase1/summary.json").read_text())
    if summary["decision_counts"].get("pending_review", 0):
        raise ValueError("Image decisions still need review")
    if summary["near_duplicate_reviews"].get("pending_review", 0):
        raise ValueError("Near-duplicate candidates still need review")
    if set(map(int, summary["class_counts_included"])) != set(CLASSES_VOLUME_PERCENT):
        raise ValueError("Included dataset does not contain all seven volume classes")
    manifest_sha = sha256_file(data / "manifest.csv")
    if manifest_sha != summary["manifest_sha256"]:
        raise ValueError("Manifest changed after the last audit")
    result = {
        "schema_version": 1,
        "archive_sha256": summary["archive_sha256"],
        "manifest_sha256": manifest_sha,
        "review_decisions_sha256": sha256_file(data / "review_decisions.json"),
        "included_images": summary["decision_counts"]["included"],
        "classes_volume_percent": list(CLASSES_VOLUME_PERCENT),
        "grouping_limitation": summary["grouping_limitation"],
    }
    output = data / "dataset_freeze.json"
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError(
                "Existing dataset freeze differs; create a new version explicitly"
            )
    else:
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
