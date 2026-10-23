# Phase 1 dataset audit

## Result

The original archive from [Mendeley Data](https://data.mendeley.com/datasets/p2gvvnh8y4/1) was downloaded to Backup Plus and verified against its published SHA-256:

```text
0b29d07414e8e6d85f899e22c3464d61f456336799144ec3ce129d8b0bb3000d
```

It is 1,405,891,275 bytes and contains 569 JPEG files, with no other archive entries. Every extracted file matches its archive member; all 569 decode successfully. The primary modelling set contains **548 close-up images**. No model has been trained and no evaluation split has been made.

| Volume class | Original images | Included close-ups |
| --- | ---: | ---: |
| 0% | 82 | 73 |
| 10% | 83 | 81 |
| 20% | 75 | 73 |
| 30% | 82 | 80 |
| 40% | 83 | 81 |
| 50% | 82 | 80 |
| 100% | 82 | 80 |
| Total | 569 | 548 |

Labels are semolina adulteration **by volume**. They are inherited from the source folders and supported by the related dataset's description; physical preparation notes are unavailable.

## Exclusions and duplicate review

- Fourteen loose files are whole-tray overview photographs. They show grid lines, tray borders, phone watermarks, and a different field of view from the structured close-ups. They are retained as raw references and excluded from the primary close-up task. The individual decisions are in `data/image_decisions.csv`, with a dedicated `loose_images.jpg` contact sheet.
- Seven extra copies are byte-identical duplicates, confirmed by SHA-256. All occur within the 0% class; none has a conflicting volume label. One canonical path per byte-identical group is retained. The archive has 562 unique byte images before removing the 14 references.
- Four duplicate pairs appear in both white-background 4x and Micro folders. All eight entries have `magnification_conflict=True`; four of them remain as canonical images. The folder tag is preserved as source metadata, with no guess about the correct magnification. Three other duplicate groups consist of filename variants within the 4x folder.
- There are no additional groups with identical decoded RGB pixels but different file bytes.
- Perceptual-hash screening found one candidate pair: `img_0181` and `img_0198`, both 30% black-background micro images. Visual review confirms the same grain arrangement with a small view shift. Both remain included and must stay in the same evaluation group. They already share the conservative class group.

Candidate screening uses a 63-bit DCT perceptual hash with Hamming distance at most 8, or a 64-bit difference hash with distance at most 4. It is a screening method, not proof that all remaining images show independent scenes; crops and viewpoint changes can evade these thresholds. Exact duplicate checks compare both original bytes and oriented RGB pixels, including dimensions.

## Quality and capture conditions

Every class contact sheet was inspected, covering all 569 originals. Images vary considerably in grain scale, framing, white balance, background coverage, and focus. Some include grid lines, tray edges, or phone watermarks. These artifacts need attention in Phase 3 because a classifier could use them as shortcuts.

All images identify the camera as TECNO CE7. Oriented dimensions are 4608×3456, 3456×4608, 1600×1200, or 1200×1600. Every image has a capture date. Resizing everything to 128×128 would discard much of the useful grain detail.

Thirty images fall in the lowest 5% of a standardized sharpness score within their background/magnification condition. Twenty-eight are in the included set; two are overview references already excluded for their field of view. No flag caused an exclusion. The score measures Laplacian variance after aspect-preserving reduction to at most 512 pixels per side; it is relative evidence, not a calibrated blur diagnosis. No image exceeds the audit's 25% near-black or near-white pixel flags. This does not prove exposure is ideal. No quality-based exclusions or label changes were made.

| Volume class | Capture dates |
| --- | --- |
| 0% | 2021-11-27, 2021-11-30 |
| 10% | 2021-12-09 |
| 20% | 2021-12-10 |
| 30% | 2021-12-10, 2021-12-11 |
| 40% | 2021-12-11 |
| 50% | 2021-12-13 |
| 100% | 2021-12-06, 2021-12-08 |

Date is strongly associated with the volume label. This does not prove a visual model will use a shortcut, but it makes the planned metadata and background controls necessary. All four background/magnification conditions remain represented in every class, with uneven counts. Magnification conflict flags must be respected in condition-specific tests.

## Grouping and the Phase 2 decision

There are no confirmed physical preparation IDs. The overview trays and close-ups make repeated views of shared material plausible. The manifest therefore conservatively gives every image in a volume class the same `group_id`, with uncertain preparation identity explicitly recorded. These are seven protective groups, not seven verified independent samples. Class/date `capture_group_id` values are provided as metadata and are not sample identities.

These conservative groups cannot support a seven-class evaluation on unseen preparations: a held-out class group leaves that class absent from training. **Phase 2 must resolve the evaluation scope before splitting.** With only these existing images, an experiment on separated scenes or capture conditions may be possible, but its result must be described as performance within this dataset, with possible shared preparations. We cannot establish independent-mixture generalization from the available files. Finding only one perceptual-hash candidate does not remove this limitation.

## Related dataset overlap

The later [Mendeley dataset](https://data.mendeley.com/datasets/nx8m9kxy26/1) lists 474 JPEGs and 468 unique CRC32/file-size/class signatures. All 468 signatures match the original archive. The result is in `source_overlap.json`; the directory snapshot is in `data/related_subset_inventory.csv`. This is strong evidence of a renamed subset, not a SHA-256 proof for its individual images. Its archive was not downloaded and its files are not added to training.

## Outputs and reproduction

Raw archive: `/Volumes/Backup Plus/PoppyPurev2/source_archives/p2gvvnh8y4-v1.zip`.

Extracted source: `/Volumes/Backup Plus/PoppyPurev2/raw/`.

The local `data/source_archives` and `data/raw` paths are symbolic links to those folders. The original project was not modified.

The CSV manifest records source paths, volume classes, conditions, file and pixel checksums, dimensions, capture metadata, hash descriptors, quality flags, group IDs, and decisions. `data/dataset_freeze.json` records the reviewed manifest and decision-file checksums. `summary.json`, duplicate CSVs, group counts, seven class contact sheets, and the loose/pair contact sheets provide review evidence. Source attribution and intended-use limits are in [DATASET_CARD.md](DATASET_CARD.md).

```sh
uv run --locked poppypure source-metadata
uv run --locked poppypure acquire --workers 16
uv run --locked poppypure audit
uv run --locked poppypure freeze-data
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

The audit is deterministic for the frozen raw files and review decisions. Unit checks cover class interpretation, unsafe ZIP paths, duplicate/loose decisions, refusal to freeze pending reviews, and detection of modified raw files. The final audit experiment record is `experiments/runs/phase1-audit.json`; it records dataset counts rather than model performance.
