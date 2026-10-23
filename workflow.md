# PoppyPure v2 workflow log

This is the ongoing record of important work, decisions, checks, and findings. Update it whenever project work changes the data, code, evaluation protocol, model, or conclusions. Record failures and their resolutions when they affect the workflow. Link detailed reports and run records rather than copying every terminal command.

**Last updated:** 24 October 2026, Asia/Kolkata.

This initial log was backfilled from the conversation and the saved project artifacts. Earlier discussion timestamps were not recorded individually. Phase 0 and Phase 1 work occurred on 13 October 2026 local time; their JSON run records use UTC, which places those runs on 12 October 2026 UTC.

## Project decisions established before implementation

1. Reviewed the original final-year project, including its notebooks, image handling, report, and application. The earlier review identified inconsistent results, risks of data leakage, and problems in feature extraction and image resizing. The original project's reported accuracy is not a validated v2 result. This history is backfilled from the discussion; the current verified dataset findings are in the Phase 1 report.
2. Agreed to rebuild the machine-learning project inside `PoppyPurev2/`, preserving the original `PoppyPure/` project.
3. Defined the main task as **seven-class classification of semolina adulteration in poppy seeds by volume**: **0%, 10%, 20%, 30%, 40%, 50%, and 100%**. Zero means pure poppy seeds; 100 means pure semolina. Earlier assumptions about mass were superseded by the user's explicit correction. These labels must not be reinterpreted as mass, seed count, or visible image area.
4. Chose seven-class classification as the primary output. Pure versus adulterated detection will be a secondary metric derived from the seven-class predictions.
5. Established that no new physical samples or photographs are available. Work uses the existing source images.
6. Compared the two Mendeley sources. Selected [p2gvvnh8y4, version 1](https://data.mendeley.com/datasets/p2gvvnh8y4/1) as canonical. The later [nx8m9kxy26, version 1](https://data.mendeley.com/datasets/nx8m9kxy26/1) explicitly describes volume labels and shares file signatures with the original; its images will not be added as independent observations. The overlap was checked again in Phase 1.
7. Created [PLAN.md](PLAN.md) with Phases 0–6: project setup; dataset audit; evaluation protocol; image processing; baselines; classifier improvement; final evaluation and deliverables. The current scope includes the model, reproducible scripts, CLI inference, and reports. The web app is outside the current scope.
8. Planned to compare image crops/masks, colour histograms, LBP, and GLCM features, preserving enough resolution to see grains. Masking and illumination correction require visual review and development evaluation. These are planned modelling techniques, not completed preprocessing.
9. Planned classical baselines and a pretrained CNN. The CNN path starts with a seven-class head on pretrained features, followed by limited fine-tuning if evaluation supports it. A CNN trained from random weights is not the starting point. No classifier has been trained yet.
10. Agreed to store source archives and extracted images on **Backup Plus**, with local symbolic links inside `PoppyPurev2/data/`. Download was deferred until Phase 1 was authorized.

## Phase 0 — completed

### 1. Created the Python project and fixed its environment

- Added [pyproject.toml](pyproject.toml), the `src/poppypurev2/` package, and the `poppypure` command.
- Pinned NumPy 2.4.1, Pillow 12.1.0, scikit-learn 1.8.0, Ruff 0.15.6, and the setuptools build dependency. Saved transitive dependency versions and distribution hashes in [uv.lock](uv.lock).
- Created a clean local `.venv` with `uv sync --locked`. The local verification used Python 3.12.0rc3; the declared supported range is Python 3.11–3.13.
- The initial dependency-resolution attempt failed because sandboxed network access could not resolve PyPI. An approved retry resolved the packages and installed the locked environment.
- Added `.gitignore` rules for environments, caches, raw-data links, generated runs, and large model artifacts. Manifest and split definitions can be versioned.

### 2. Centralized settings and established run records

- Added [configs/base.toml](configs/base.toml) with the seven volume classes, seed 42, source URLs/DOIs, local output paths, macro F1 as the primary metric, and the grouping key.
- Added configuration validation so an incorrect class mapping is rejected.
- Implemented JSON experiment records containing configuration checksum, package/runtime identity, seed, code version, and placeholders for dataset, splits, preprocessing, model, and metrics. Records refuse overwrites. The schema is documented in [experiments/SCHEMA.md](experiments/SCHEMA.md).
- Added `poppypure doctor` to inspect configuration and paths, and `poppypure smoke` to verify configuration loading and record creation. The smoke command does not train or evaluate a model.

### 3. Documented and verified the setup

- Added [README.md](README.md) with setup and reproduction commands.
- Verified the clean installation, CLI, smoke record, unit tests, compilation, formatting, and lint checks. Corrected formatting and lint issues found during the first checks.
- Saved `experiments/runs/phase0-smoke.json`.
- Marked Phase 0 complete in the plan. No dataset download or external-drive links were created in this phase.

## Phase 1 — completed

### 4. Confirmed source metadata and external storage

- Checked that `/Volumes/Backup Plus` was mounted and had sufficient space, approximately 189 GiB free at the initial check.
- Read Mendeley's public metadata to obtain the original archive's URL, size, published SHA-256, contributors, DOI, and CC BY 4.0 licence.
- Saved the source registry in [data/sources.json](data/sources.json) and public metadata snapshots in `data/source_metadata/`.
- Retrieved only the second archive's ZIP directory for overlap verification; did not download a second full image archive.
- Implemented `poppypure source-metadata`. An initial attempt using Python's default HTTP client received HTTP 403; the implementation switched to curl, which the server accepted.

### 5. Downloaded, verified, extracted, and linked the canonical data

- Created `/Volumes/Backup Plus/PoppyPurev2/source_archives/` and `/Volumes/Backup Plus/PoppyPurev2/raw/`.
- A single download connection was too slow. Replaced it with resumable 8 MiB byte-range downloads, initially using eight workers and then 16. Verified the isolated process group before stopping the earlier download; completed chunks were reused.
- Implemented `poppypure acquire`, including published-size/checksum checks, safe extraction paths, comparison of existing extracted files, and protection against replacing existing local data paths.
- Assembled `p2gvvnh8y4-v1.zip`, **1,405,891,275 bytes**, and verified it against Mendeley's published SHA-256 before extraction.
- Extracted the archive without changing source image bytes.
- Created and verified both symbolic links:

  ```text
  PoppyPurev2/data/source_archives -> /Volumes/Backup Plus/PoppyPurev2/source_archives
  PoppyPurev2/data/raw             -> /Volumes/Backup Plus/PoppyPurev2/raw
  ```

- Saved [data/acquisition.json](data/acquisition.json). Download chunks and the interrupted initial download were temporary acquisition files on the external drive; this log does not claim they were cleaned up.

### 6. Built the image inventory and audit pipeline

- Implemented `poppypure audit` in [src/poppypurev2/audit.py](src/poppypurev2/audit.py).
- Accounted for every archive entry in [data/archive_inventory.csv](data/archive_inventory.csv): **569 JPEGs**, consisting of **555 structured images and 14 loose images**.
- Compared every extracted image with the original ZIP member. All 569 images match their archive data and decode successfully.
- Created [data/manifest.csv](data/manifest.csv) with source paths, volume labels, backgrounds, source magnification tags, sizes, SHA-256 values, oriented dimensions, capture metadata, pixel hashes, perceptual hashes, quality flags, groups, and decisions.
- Applied EXIF orientation for inspection and decoded-pixel comparison. Removed null padding from EXIF text fields. Raw files remain unchanged.
- Generated a contact sheet for each of the seven classes, plus dedicated loose-image and near-pair sheets. Inspected every class sheet and the dedicated review sheets.

### 7. Identified exact duplicates and metadata conflicts

- Compared original file SHA-256 values and found **seven duplicate groups**, each with one extra copy. The archive therefore contains **562 unique byte images**.
- All seven duplicate groups are within the 0% class. No exact duplicate has conflicting adulteration labels.
- Retained one canonical structured path per duplicate group and recorded the excluded copy's retained image ID.
- Four duplicate pairs occur in both the white-background 4x and Micro folders. Flagged **eight original entries**, including **four retained canonical images**, with `magnification_conflict=True`. Preserved source folder tags without guessing the correct magnification.
- Three other duplicate groups are filename variants within the white-background 4x folder.
- Compared SHA-256 of oriented RGB pixels and dimensions. Found **zero additional identical-pixel groups with different file bytes**.
- Saved the evidence in `reports/phase1/exact_duplicates.csv` and `reports/phase1/pixel_duplicates.csv`.

### 8. Screened for near duplicates and reviewed the candidate

- Computed a DCT pHash with 63 effective bits and a 64-bit dHash. Compared unique-byte images pairwise.
- Flagged a candidate when **pHash Hamming distance ≤ 8 OR dHash Hamming distance ≤ 4**.
- Found one candidate pair: **`img_0181` and `img_0198`**, both 30% black-background micro images. Their **pHash distance is 12** and **dHash distance is 2**; dHash triggered the review.
- Inspected the pair visually and confirmed the same grain arrangement with a small view shift. Retained both images and recorded `same_scene_shifted_view_keep_in_same_group`.
- Saved the candidate measurements in [reports/phase1/near_duplicate_candidates.csv](reports/phase1/near_duplicate_candidates.csv) and the review in [data/review_decisions.json](data/review_decisions.json).
- Documented that these hashes can miss crops, rotations, overlap, and larger viewpoint changes. One candidate does not establish that every other photograph shows an independent scene.

### 9. Reviewed loose images and quality flags

- Inspected all 14 loose images individually in a contact sheet. They show whole trays with grids, borders, and phone watermarks; they are overview references rather than empty background photographs.
- Retained all 14 in the raw data and excluded them from the primary close-up modelling set, with a reason for each. A provisional review sheet generated during the download was replaced by the verified final review evidence and removed.
- Reviewed sharpness after aspect-preserving reduction to at most 512 pixels per side. Flagged the lowest 5% within each background/magnification condition using Laplacian variance.
- Flagged **30 original images** for relatively low sharpness: 28 remain included; two are overview references excluded for their field of view. No quality flag caused an exclusion.
- Checked near-black and near-white pixel fractions. No image exceeded the 25% flag threshold; this does not prove ideal exposure.
- Observed variation in white balance, focus, grain scale, framing, grid lines, borders, and phone watermarks. Recorded these as image-processing and shortcut concerns for later phases.
- Saved every inclusion/exclusion reason in [data/image_decisions.csv](data/image_decisions.csv). No volume label was changed.

### 10. Assessed capture metadata and grouping limits

- All 569 images identify the camera as TECNO CE7 and provide capture dates across eight dates in November–December 2021.
- Oriented dimensions fall into four sizes: 4608×3456, 3456×4608, 1600×1200, and 1200×1600.
- Found a strong association between dates and volume classes; for example, all 10% images are dated 2021-12-09 and all 50% images 2021-12-13. Recorded the complete date mapping in the audit report.
- Physical preparation IDs are absent. Assigned conservative `group_id` values that keep all images of each volume class together, explicitly marking preparation identity as uncertain. These seven groups are not seven confirmed independent samples.
- Added class/date `capture_group_id` metadata. Ten such groups remain among included images; capture dates do not prove independent preparations.
- Documented the consequence for Phase 2: holding out one conservative class group leaves that class absent from training, so these files cannot establish seven-class performance on unseen preparations. Any weaker evaluation must state its scope and possible shared preparations explicitly.

### 11. Rechecked overlap with the related dataset

- The second dataset's directory lists **474 JPEGs** and **468 unique CRC32/file-size/class signatures**.
- All **468 signatures match the original archive**, with zero unmatched signatures.
- Saved [data/related_subset_inventory.csv](data/related_subset_inventory.csv) and [reports/phase1/source_overlap.json](reports/phase1/source_overlap.json).
- Recorded the method's limitation: matching CRC32, file size, and label is strong overlap evidence, not a SHA-256 identity proof for the second archive. No second-dataset images were added to training.

### 12. Saved the reviewed dataset and reports

After excluding seven duplicate copies and 14 overview references, the primary set contains **548 close-ups**:

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

- Added the findings and reproduction commands in [reports/phase1/AUDIT.md](reports/phase1/AUDIT.md).
- Added source attribution, CC BY 4.0 licensing, scope, decisions, and limitations in [reports/phase1/DATASET_CARD.md](reports/phase1/DATASET_CARD.md).
- Reran the audit with the saved visual-review decisions and wrote `experiments/runs/phase1-audit.json`.
- Added a source-code fingerprint to experiment records so code is identifiable even when `git_revision` is unavailable.
- Implemented and ran `poppypure freeze-data`, which refuses pending reviews and incompatible replacement freezes. Saved [data/dataset_freeze.json](data/dataset_freeze.json).

Frozen SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Original ZIP | `0b29d07414e8e6d85f899e22c3464d61f456336799144ec3ce129d8b0bb3000d` |
| Manifest | `d5358c61f2f42c4dbd5c4efc92e4d62ba54caff998ecbf631f665e7e1fbc5829` |
| Review decisions | `12feae5d7368ed06307288b4071d7afa1bcdcaeea6603f8d655396dbeafc5d01` |

### 13. Verified completion

- All five unit tests passed. They cover configuration/records, label interpretation, unsafe ZIP paths, duplicate and loose-image handling, refusal to freeze pending reviews, and detection of altered raw images.
- Formatting, linting, and compilation passed.
- Final dataset acceptance checks confirmed 569 originals, 548 included images with unique byte and pixel hashes, all seven volume classes, readable structured inputs, resolved pair reviews, matching manifest/review/run-record hashes, and correct external symbolic links.
- Updated the README and plan to mark Phase 1 complete and identify evaluation feasibility as the next task.

## Current state and next work

**Completed:** Phases 0–4. Active protocol **v4** uses **384 training / 82 validation / 82 test images**, with all classes and capture conditions in each partition. Phase 4 completed 21 visual configurations and four controls. Selected whole-image SVM validation macro F1 is **0.8591**, accuracy **85.37%**; best cropped-texture SVM scores **0.8293 / 82.93%**. Saved model bundles and predictions reproduce these results. Background masks remain off. The test set is still reserved; final model selection awaits Phase 5 ResNet comparisons.

**Next:** Phase 5 — frozen ResNet18 linear/MLP-head comparisons on crop/detail inputs, followed by ResNet50 and limited fine-tuning as specified in MODEL_TRAINING_PLAN.md. Keep the v4 split and test reservation. Earlier condition-holdout rules below are historical and superseded by Step 21.

## Phase 2 — completed, limited internal evaluation

### 14. Checked evaluation feasibility and narrowed the claim

- Re-read the phase plan, workflow instructions, frozen manifest, review decisions and duplicate evidence. Used metadata only; did not rerun the image audit, download data or modify raw images.
- Confirmed that the seven conservative preparation groups each contain one entire class. Separating those groups removes a required class from training. No seven-class evaluation on unseen preparations is supported by the available files.
- Kept the original conservative groups and frozen manifest unchanged. Explicitly revised the evaluation scope to internal capture-condition transfer. This supersedes the original plan's stronger requirement to separate all possible views of a physical preparation; that requirement cannot be established here.
- Consulted official scikit-learn guidance on grouped validation, leakage and metric definitions. No dependencies were changed.
- An initial exploratory count used `include` instead of the manifest's actual `included` decision string and returned zero rows. Checked the manifest values, corrected the filter and confirmed all 548 included images before generating artifacts. No outputs were based on the zero-row count.

### 15. Created and froze deterministic split membership

- Added `freeze-evaluation` and [evaluation.py](src/poppypurev2/evaluation.py). The command checks Phase 1 manifest/review checksums before defining splits and refuses incompatible replacement artifacts.
- Reserved all **277 black-background images** for development and all **271 white-background images** for one final background-transfer evaluation. Black development avoids the known magnification-tag conflicts. No random image split or performance-based partition selection was used.
- Defined two development folds: fit 140 black Micro images and validate 137 black 4× images; reverse those roles for the second fold. Each training/validation condition and the final holdout contains all seven volume classes.
- Created broad class/background/folder-magnification `evaluation_group_id` blocks: 14 in development and 14 in holdout. These are capture-condition blocks, not independent physical preparations. Saved original preparation and capture groups alongside them; conservative preparation groups knowingly cross the new partitions, as documented.
- Verified that reviewed near-duplicate images `img_0181` and `img_0198` remain together in black Micro. Other unknown overlapping views cannot be ruled out.
- Kept all four retained images with ambiguous magnification tags in the main white holdout score. Exclude them from magnification-specific subgroup metrics only and report their predictions separately.
- Saved [assignments.csv](data/splits/v1/assignments.csv), [protocol.json](data/splits/v1/protocol.json) and [freeze.json](data/splits/v1/freeze.json). Updated the base configuration to name the evaluation group key and protocol, while preserving the separate preparation group key.

### 16. Fixed metrics, selection and holdout rules before fitting

- Implemented seven-class macro F1, balanced accuracy, per-class precision/recall/F1/support, fixed-order confusion matrix, absolute volume-class error in percentage points, and derived binary sensitivity/specificity. Undefined binary denominators return null; balanced accuracy returns null when a subset lacks a true class. The scorer rejects labels outside the seven by-volume classes.
- Fixed model selection to the mean of two development macro F1 scores. Save both fold scores, range and out-of-fold predictions; near ties within 0.005 prefer fewer fitted parameters and then shorter measured training time.
- Fixed train-fold-only fitting for scaling, PCA, feature selection, class weights and learned processing. Split before tiling/crops/augmentation; preserve source membership and score at image level. CNN schedules must be fixed without outer-fold checkpoint selection. Probability calibration is unsupported with the current independence evidence.
- Defined final evaluation as the locked winner refitted on all black images, scored once on white in Phase 6. No reverse background experiment during development. Capture-date transfer is unsupported because dates are confounded with classes.
- Recorded that all images were visually audited in Phase 1: the holdout is reserved from subsequent development, not unseen by the auditor. Phase 3 processing design and visual QC must now use black development images only. White failures at final evaluation must be reported without retuning the final result.
- Replaced the plan's proposed independent-group uncertainty estimate with descriptive fold/condition variation. No independent-photo or preparation-level confidence interval is justified.
- Added [reports/phase2/EVALUATION.md](reports/phase2/EVALUATION.md), updated the README and plan, and saved `experiments/runs/phase2-evaluation.json`. No model was trained or scored; synthetic metric-test labels are not project performance results.

Frozen Phase 2 SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Split membership | `bf4154ba06cea7f4ea63af9c049efda77c299576c5a254f2a2b2cbf9c8855163` |
| Protocol | `423c02419967d7c2e30e9824801f3ea86a1eadca2ef37ed38b902feba93c422b` |

### 17. Verified the evaluation implementation

- Added meaningful tests for hand-calculated classification errors, volume-point distances, confusion-matrix direction, missing denominators and invalid labels.
- Tested generation against a temporary copy of the actual frozen manifest: 548 assigned IDs, 277 development images, repeatable generation, rejection of a changed manifest or protocol, and rejection of a simulated known near-pair crossing conditions.
- All seven unit tests passed. Initial linting identified overlong protocol-description strings; split the source literals without changing their resulting text, then formatting and lint passed. Compilation passed.
- Confirmed that rerunning the freeze command verifies identical artifacts and that the run record identifies the same manifest and membership hashes. The frozen Phase 1 data decisions were preserved. Raw data remains on Backup Plus through the existing links.
- **Remaining limitation:** Phase 2 enables reproducible internal condition-transfer experiments. It cannot establish generalization to new physical mixtures, cameras or acquisition environments.

## Phase 2 revision — active protocol v2

### 18. Revised the evaluation plan after the user's training-data concern

- The user questioned reserving approximately half the usable data and requested a better plan, including concrete image-processing techniques. Reconsidered the earlier emphasis on whole-background transfer: it prevented training on white-background examples and restricted final fitting to 277 images. A larger, more diverse development set is a better fit for the project's classifier goal; improved accuracy is not presumed.
- Selected **414 development / 134 final holdout** by whole capture conditions. Development now includes black 4× (137), black Micro (140), and white 4× (137). White Micro (134) remains held out. Training has examples from both backgrounds and both folder magnifications. This is a condition-transfer test, not an independent-mixture test or isolated magnification/background effect.
- Defined three validation rounds, each holding out one development condition and fitting the other two (277/274/277 training images respectively). Every development image trains in two rounds and validates once. Final model fitting uses all 414 development images after recipe selection. Selection becomes mean macro F1 across all three folds, preserving supporting metrics, tie-break and train-fold leakage rules.
- White Micro was chosen from metadata before any model results: about one quarter of the data, all seven classes, no known magnification-tag conflicts, while leaving both backgrounds/magnifications in development. The four retained conflicting white 4× tags are now in development, kept in main condition-fold scores and omitted only from summaries claiming trusted magnification. They retain their source folder membership, not inferred optical magnification.
- Added an optional post-evaluation refit of the unchanged recipe on **all 548 usable images**, separately versioned for deployment. Preserve the evaluated 414-image model and predictions first. The all-data refit has no untouched test score; do not transfer the previous score to it as a direct measurement.
- Preserved all v1 split files and the earlier run record; verified their original checksums still match. Created v2 membership/specification/freeze under `data/splits/v2/`, switched the active base configuration to v2, and added CLI `--version` support for explicit historical verification. No model or learned preprocessing was run before the revision, and white Micro remained reserved.
- Saved `experiments/runs/phase2-evaluation-v2.json`. V2 membership SHA-256: `1f7a14e871c99e73c74207be6c2c637c697f225563bbc4a7fa228ee8eb213bbf`; v2 specification SHA-256: `fec2ad27925868e6fc34a0ab3ac9ecced2a9f56b17f68a1d49682237b268dafa`.
- Added [the revised evaluation and processing plan](reports/phase2/REVISED_PLAN.md), marked the old evaluation document as superseded, and updated PLAN/README. Step 18 supersedes Steps 15–16's black-only development, two-fold selection, white-only final holdout and four ambiguous holdout cases; those entries remain as history.

### 19. Specified planned processing and checked the revision

- Planned an RGB/EXIF/aspect-preserving reference pipeline, checked mixture-region crops, and spatially distributed detail tiles pooled at image level if resizing erases grains. Tiles must not be counted as independent samples or assigned verified local volume concentrations.
- Planned optional HSV/Lab material masks with conservative thresholding/morphology, accepted only after visual checks retaining both materials and development evaluation. Mask area is not a volume label. Whole-image/crop alternatives remain available if masking fails, especially for white semolina on white background.
- Planned normalized colour histograms, uniform LBP and normalized GLCM descriptors, keeping artificial mask edges out of texture measurement. Planned RGB inputs with documented pretrained CNN normalization, training-only modest augmentation and optional illumination correction only when comparison supports it. These remain proposals for Phase 3 and later phases, not executed or validated methods.
- Consulted official scikit-learn, scikit-image and Torchvision documentation for validation, texture methods and pretrained input transforms; dependencies remain unchanged.
- Extended the meaningful Phase 2 integrity test to verify 414/134 coverage, three folds, both backgrounds in development, all seven classes in every training/validation side, known near-pair co-location, repeatable v2 generation and preservation of v1 bytes. All seven tests, Ruff formatting/lint and compilation passed. Both version freezes verify unchanged artifacts. The manifest/raw data and source decisions remain unchanged.
- Remaining limitation is unchanged: no verified preparation IDs, possible shared material/overlap across conditions, date/class confounding and no independent-preparation confidence interval. More training images improves data use, not the evidence of physical independence.

### 20. Recorded the user's preparation confirmation

- The user explicitly confirmed that images represent individually prepared mixtures and requested removal of the missing-preparation caveat. Accepted this as project provenance, recording the statement and its source in `data/preparation_provenance.json`. This replaces the earlier assumption based solely on missing source identifiers; it is user-supplied information, not a new image-analysis result.
- Preserved verified audit findings: seven extra exact copies remain excluded, and `img_0181`/`img_0198` remain one confirmed same-scene unit. The 548 included images therefore define 547 preparation units, with 413 development units across 414 development images and 134 test units/images.
- Created active protocol v3 with the same 414/134 image partitions and three development folds. Replaced class-wide preparation groups with image-specific units, merging confirmed same-scene views; retained old class-wide values in `legacy_conservative_group_id`. V1/v2 artifacts and audit history remain unchanged.
- Updated the current README, dataset card, plan and evaluation specification to use user-confirmed preparation provenance and remove the superseded missing-preparation warning. Known duplicate, magnification and capture metadata findings remain recorded as factual audit results. No performance claims or metrics were added.
- Extended integrity checks for repeatable v3 generation, 547 preparation units and no preparation-unit overlap between development and holdout. Seven tests, formatting, lint and compilation passed. Saved `experiments/runs/phase2-evaluation-v3.json`; no model was trained and no image processing was performed.

## 14 October 2026 — mixed-condition split and Phase 3 start

### 21. Replaced the condition holdout with the requested train/validation/test split

- The user requested all image types in training and testing, agreed to approximately 70/15/15, and authorized updating the split and starting Phase 3. Implemented active protocol v4 before any model fitting, preserving v1–v3 files and records as history.
- Added [splits.py](src/poppypurev2/splits.py): largest-remainder quotas across 28 volume-class/background/folder-magnification strata; seeded whole-preparation ordering and subset-sum selection preserve the reviewed two-image scene while meeting exact image quotas. Seed is 42.
- Froze **384 training / 82 validation / 82 test images**, covering all 28 strata in every set. Training contains 383 preparation units; validation and test each contain 82. The known near-duplicate pair is in training; zero preparation-unit overlap across partitions.
- Kept all existing data decisions and raw files unchanged. `evaluation_group_id` now identifies the preparation unit, `condition_group_id` preserves the capture block, and the old class-wide values remain legacy metadata. Four magnification-conflict flags remain by source folder.
- Model selection now uses validation macro F1 rather than condition-fold averages. Fit models/learned processing on training only during selection; optionally refit the locked recipe/schedule on 466 train+validation images before one final 82-image test.
- Saved `data/splits/v4/{assignments.csv,protocol.json,freeze.json}` and `experiments/runs/phase2-evaluation-v4.json`; switched the base configuration to v4. Added [the active evaluation specification](reports/phase2/TRAIN_VAL_TEST.md), updated README/PLAN/dataset card, and marked the old evaluation plan as superseded.
- Membership SHA-256: `cf7e5f8e2ed1f7dd3537bd62c611f54bcc55decc0a5e8ad681f641a4f7db88c0`; specification SHA-256: `0af742424bf95496d8fceee2b116292677f7b6eabbe05919f14a48d05eb4a2c9`.

### 22. Inspected training images and implemented initial processing

- Confirmed Backup Plus is mounted. Generated and inspected four initial training-only sheets, one example per class and capture condition (28 total). Observed lower-edge TECNO markings, varying focus/scale/colour balance and some grid lines. No validation or test images were decoded in this design pass.
- Added [preprocessing.py](src/poppypurev2/preprocessing.py): RGB decoding, EXIF orientation followed by metadata removal, whole-image/crop variants, aspect-preserving resize, optional square padding with content bounds, and five deterministic detail regions inside the cropped image. Raw files remain unchanged.
- Added `configs/preprocessing-v1.json`: initial 768-pixel long side, crop fractions `[0.05,0.05,0.95,0.88]` and 512-pixel detail regions. The crop retains about 74.7% of the area and is a candidate for comparison, not a selected optimum or concentration measurement. No mask, blur or lighting correction is enabled.
- Added `poppypure preprocess-qc`. It verifies v4 and original SHA-256 values, processes only training images, and records crop coordinates, output sizes, pixel hashes and source preparation membership. All **384 training images** passed checks. It does not create independent tile-labelled samples or train a model.
- Saved `reports/phase3/training_preprocessing.csv`, `summary.json`, four crop-review sheets and `experiments/runs/phase3-initial-qc.json`. Inspected all four original/crop/detail sheets and saved [review_decisions.json](reports/phase3/review_decisions.json): observed markings are excluded in the 28 reviewed crops, mixed materials remain visible, and detailed regions show finer texture. Background and some internal grids remain.
- Added [PROCESSING.md](reports/phase3/PROCESSING.md). Phase 3 is in progress; optional mask prototypes, feature extraction and classifier-based comparison remain next work. No claims of improved classification performance were made.

### 23. Verified the new split and processing

- Extended split tests for exact 384/82/82 counts, coverage of all 28 strata in every set, no preparation leakage, repeatable generation and refusal to overwrite altered membership.
- Added meaningful processing tests for EXIF rotation, metadata removal, preservation of original bytes, footer exclusion, aspect ratio/padding bounds, valid crop geometry and deterministic in-bounds detail regions.
- All **10 tests** passed. Formatting, lint and compilation passed. Initial lint findings were unused imports and overlong specification strings; corrected them before final verification.
- Checked run-record hashes against v4, training QC membership against the training split, and preserved historical split artifacts. No dependency downloads, raw-file edits, classifier fitting, validation scoring or test scoring occurred.

### 24. Installed the image-processing dependency and prototyped masks

- The user authorized continuing Phase 3. Read project instructions and prior workflow, verified the external raw-data folder is available, and consulted official scikit-image LBP/GLCM, Otsu and colour-conversion APIs.
- Added pinned **scikit-image 0.26.0** to `pyproject.toml` and updated `uv.lock` with dependency hashes. The sandboxed `uv sync` failed DNS resolution; an approved network retry completed installation. New installed transitives include imageio 2.38.1, lazy-loader 0.6, networkx 3.7, packaging 26.3 and tifffile 2026.9.20. A later `uv sync --locked --offline` verified the environment without further downloads.
- Added [masks.py](src/poppypurev2/masks.py) and `poppypure mask-qc`. Tested Lab-lightness Otsu and a colour/lightness rule using HSV saturation for inferred light backgrounds, with radius-one morphological closing. Background polarity comes from image pixels, not folder/class tags.
- Selected **56 training images**: a representative plus a different low-sharpness example from each of 28 class/capture strata. Generated 112 mask measurements and eight red-removal overlay sheets. Verified source SHA-256 values. No validation/test images were processed.
- The first API choice (`binary_closing`) produced deprecation warnings; switched to supported `closing(..., mode='ignore')`. Initial border-median polarity also failed when grains filled image edges. Visual review identified the error; replaced it with a fifth-percentile lightness rule and added a dense-border synthetic test. Preserved the initial/intermediate run records; the final trial is `experiments/runs/phase3-mask-dark-tail.json`.
- Inspected all **eight corrected sheets / 56 examples**. Corrected masks preserve much material in many black-background examples, but white-background examples lose visible material, including pure semolina; some dense/blurred black images also fail. Saved findings in [masks/review_decisions.json](reports/phase3/masks/review_decisions.json). **Neither mask is enabled for the current classifier inputs.** No segmentation-accuracy or volume estimate was inferred from pixel coverage.

### 25. Implemented and extracted colour/texture features

- Added [features.py](src/poppypurev2/features.py), `configs/features-v1.json` and `poppypure extract-features --output ...`. Output directories and run records refuse overwrite. Experiment records now include the scikit-image version.
- Defined **196 features per region**: 96 HSV/Lab colour histogram values (six channels ×16 bins), 28 uniform LBP bins (P=8/R=1 and P=16/R=2), and 72 GLCM values (32 gray levels; 1/2/4-pixel distances; four directions; contrast, dissimilarity, homogeneity, energy, correlation and ASM).
- Normalize each channel/LBP histogram; use integer grayscale and exclude LBP border samples. GLCMs are symmetric and normalized. Descriptors use valid RGB regions without padding or artificial mask edges. Dataset scaling/PCA/selection are not fitted in this phase.
- Extracted whole-image and crop overviews at the configured 768-pixel long side, plus five native detail regions pooled by descriptor mean and population standard deviation. There is **one row per original image**, not independent tile-labelled examples. Labels and preparation IDs are separate NPZ arrays, not classifier inputs.
- Completed extraction for **all 384 training images**, checking every raw source SHA-256: whole **384×196**, crop **384×196**, detail **384×392**, all finite. Saved `data/features/v1/train.npz`, feature names in `schema.json`, checksums/settings in `summary.json`, and `experiments/runs/phase3-training-features.json`. No validation/test feature extraction or classifier fitting occurred.
- NPZ SHA-256: `1ad5fc7a7d95229c624cd467ed38cdb7c024f799ee8cd36225d9915883423985`; schema SHA-256: `1329a0f699d73dc626567a2b84ba6b96220f8908a343d81a7278d4cb23281fc0`. Both preprocessing and feature config hashes are recorded with the frozen v4 membership hash.
- Added [FEATURES.md](reports/phase3/FEATURES.md), updated README/PLAN and the initial processing report, and marked Phase 3 complete for implementation/QC. Representation comparison and final recipe selection remain in Phases 4–5, as planned. No claim that features improve accuracy has been made.

### 26. Verified feature meaning, numerical outputs and data separation

- Added known-pattern tests: normalized histograms and expected LBP bins for constant regions; zero GLCM contrast and unit energy/correlation for constants; checkerboard contrast equal to 31² at distance one and zero at distance two; deterministic output; correct pooled dimensions/zero variation for identical detail regions; mask/overlay behaviour and dense-border polarity regression.
- All **14 tests** passed without the earlier deprecation warnings. Ruff format/lint, compilation and locked offline environment checks passed.
- Accepted the saved feature artifact only after checking IDs/labels/preparation mapping against v4 training membership, all matrix dimensions/finite values, histogram normalization, schema lengths, checksums and one recomputed real-image descriptor. Mask review IDs are also training-only. Experiment records contain no fitted model or classification metrics. Frozen raw/manifest/split artifacts remain unchanged.

### 27. Planned the classical, MLP and ResNet training comparisons

- The user asked for an appropriate plan covering the ResNet/MLP approach discussed from the original project. Read the current phased plan, frozen split settings and workflow, then inspected the original notebooks read-only.
- Confirmed `PoppyPure/source code/resnet.ipynb` uses pretrained Keras ResNet50 with average pooling, then an MLP with 512/256 hidden units and dropout. Its image/augmentation feature rows are created before splitting, and the test set is reused for early stopping and final scoring. Confirmed the separate handcrafted-feature MLP and PCA-before-split code in the other notebooks. No original files were changed or executed.
- Added [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md), preserving the 384/82/82 v4 split and seven by-volume classes. Phase 4 now has a bounded 21-configuration classical/handcrafted-MLP sequence plus diagnostic controls; validation feature extraction precedes fitting, with all learned transforms fitted on training only.
- Specified frozen ResNet18 linear versus small-MLP heads on crop overview versus pooled detail regions, followed by frozen ResNet50 linear/MLP comparisons so the original architecture family is explicitly evaluated. Planned at most two limited fine-tuning settings and one optional error-motivated fusion experiment, with all trials saved.
- Defined explicit pretrained weight identifiers, RGB/crop/normalization/pooling, frozen BatchNorm handling, starting optimizer/epoch settings, checkpoint rules, train-only augmentation, primary validation ranking/tie rules, and optional confirmation seeds without seed cherry-picking. Planned CPU execution for Phase 4 and an availability check before optional Apple MPS use in Phase 5.
- Described recipe lock, optional 466-image train+validation refit with fixed schedules, one 82-image final test, and versioned CLI/inference deliverables. Consulted official Torchvision model and PyTorch transfer-learning documentation plus scikit-learn leakage guidance; linked sources in the plan.
- Updated PLAN/README links and Phase 4/5 gates to include the concrete comparisons. This step changes planning documents only: no packages or weights were installed, no training/evaluation was run, and the split/model state is unchanged.

### 28. Prepared Phase 4 validation and control inputs

- The user authorized starting work on the training plan. Read project/workflow instructions and the bounded experiment plan; inspected installed scikit-learn APIs and official pipeline/SVC/MLP documentation. Confirmed arm64 hardware with ten reported CPUs. No new dependencies were installed.
- Extended feature extraction to an explicit `train` or `validation` partition; final test is not an allowed development option. Validation extraction checks preprocessing/feature/split hashes against the saved Phase 3 training features before decoding images.
- Extracted all **82 validation images** using exactly the saved whole/crop/detail definitions. Saved `data/features/validation-v1/` and `experiments/runs/phase4-validation-features.json`. Matrices are 82×196, 82×196 and 82×392, all finite; feature schema SHA-256 matches training. Validation NPZ SHA-256: `7353f6cdca635b0ea45cebb4b239e153c5dc43c9e826b4e7ee0109d529b595b2`.
- Added `configs/phase4-v1.json`, [controls.py](src/poppypurev2/controls.py) and `poppypure prepare-controls`. Prepared 384 training and 82 validation low-gradient colour vectors, verifying raw-file hashes. Selection uses the image's lowest-gradient quantile on a 256-pixel cropped overview and excludes the outer two pixels; no class or folder label chooses pixels.
- Inspected four training control sheets (28 examples). Low-gradient pixels include both background and grain interiors, so the control is explicitly **low-gradient appearance**, not background-only. Saved `reports/phase4/CONTROL_REVIEW.json`. Control features/selection measurements are in `data/features/controls-v1/`; no test images were processed.

### 29. Implemented and ran the bounded baseline comparison

- Added [baselines.py](src/poppypurev2/baselines.py) and `poppypure train-baselines --output ...`. Loaders check input artifact hashes, schema, IDs, targets, preparation units and partition membership before fitting. Save complete fitted pipelines, estimator parameters, input/config hashes, training time, explicit fitted-size convention, preprocessing-statistic counts and per-image validation scores.
- Ran the exact plan: **nine logistic screening configurations**, then **six SVM**, **four random forest** and **two MLP** configurations on the two selected screening inputs. Added majority, stratified-random, metadata-only and low-gradient controls. **All 25 runs passed; none failed or produced a convergence warning.**
- Screening chose whole-image combined colour+texture (196 columns) and cropped texture (100 columns). The latter was within 0.005 macro F1 of cropped combined input and had fewer fitted parameters, so the declared near-tie rule selected it. Saved `stage2_inputs.json` before all second-stage runs; no extra hyperparameter trials were added.
- Scaling fitted on training only; random forests use no scaler. MLPs use explicit validation-based epoch selection with `partial_fit`, no automatic internal split, and restore the earliest best-macro-F1 checkpoint. Whole/crop MLP best epochs are 57 and 41. SVM score columns are labelled decision margins rather than probabilities. No PCA, feature selection or calibration was used.
- Saved all models, records, validation predictions, subgroup metrics and MLP curves in `experiments/runs/phase4-v1/`; saved generated comparison/summary/error outputs under `reports/phase4/`. Phase 4 settings SHA-256: `5d801501a386e1b578619a04da145c7a2f3bdae5ee4c697cd012a1d83999b1f8`.
- Aggregate measured fitting/checkpoint time was about **2.16 seconds** on CPU, excluding image-feature extraction and reporting. GPU use was unnecessary. Neither PyTorch nor pretrained ResNet weights were installed in this phase.

### 30. Reviewed the first classification results

- Selected **whole-image colour+texture RBF SVM, C=10, gamma=scale**: validation macro F1 **0.859132**, accuracy **70/82 = 85.37%**, balanced accuracy **0.854978**, volume-class MAE **2.439 points**. Training macro F1 is **0.992412**. The model is `svm_input1_C10`, not the final selected project model.
- Best cropped-input SVM (`svm_input2_C10`, texture features) scored macro F1 **0.829324**, accuracy **68/82 = 82.93%**. Whole and crop winners use different feature families, so their gap does not isolate the effect of cropping or camera markings.
- Metadata-only control scored **85.37% accuracy / 0.797959 macro F1**, despite receiving no image pixels. It missed every 30% validation example, while the visual winner correctly classified 8/12. This records the dataset's predictive capture associations and illustrates why macro F1 is primary. The visual classifier does not receive this metadata.
- Low-gradient control scored **0.642782 macro F1**, stratified random **0.154629**, majority **0.036474**. The appearance control includes grain interiors, so its score is not attributed solely to background.
- Inspected all 12 selected-model validation errors in the saved gallery. Six errors swap 30% and 40%. Weakest class F1 values are 30% (0.64) and 40% (0.72). The largest mistakes include 0→30 and 100→50. Background, scale, packing and focus vary visibly; no specific causal attribution was inferred from the gallery alone.
- Saved [FINDINGS.md](reports/phase4/FINDINGS.md) with metrics, full confusion matrix, class/condition summaries and interpretation, alongside generated [BASELINES.md](reports/phase4/BASELINES.md). Updated README, PLAN, model-training-plan status and record-schema documentation. Phase 4 is complete; Phase 5 compares ResNet models on the same development allocation. **No test accuracy exists.**

### 31. Verified model persistence, leakage rules and artifact integrity

- Added tests for pooled-feature family slicing, near-tie selection and rejection of non-converged candidates, train-only MLP scaling and best-checkpoint restoration, metadata exclusion of labels/class paths/IDs, and valid normalized low-gradient descriptors.
- All **18 tests** passed. Formatting, lint, compilation and locked offline environment checks passed; initial long documentation strings were reformatted before training. No new dependency downloads occurred.
- Reloaded **all 25 model bundles** and reproduced every saved validation prediction, macro F1 and confusion matrix. Verified prediction IDs/labels against the 82-image validation allocation, scaler statistics against training-only columns, selected MLP epochs against their curves, and all model/prediction/metric hashes. Saved `experiments/runs/phase4-v1/ACCEPTANCE.json`.
- Verified unchanged v1–v4 freeze artifacts and original training feature hashes. The 82 test images were not decoded, transformed, fitted or scored. No split/label/raw-file decisions were changed to improve results.

### 32. Added missing rendered curves and result charts

- The user requested loss curves/important graphs and clarification of validation versus test accuracy, code-test counts, experiment counts and macro F1. Checked the saved Phase 4 outputs: numeric results, error gallery and MLP CSV traces existed, but rendered training/result plots had not been generated. Corrected this omission.
- Added pinned Matplotlib **3.10.8** and locked plotting dependencies. Sandboxed dependency access failed DNS resolution; an approved network retry installed them. Plotting caches are project-local and ignored by Git. Initial rendering encountered fontconfig cache warnings but completed; cache defaults were then explicitly configured in the plotting command.
- Extended MLP logging to include full-partition training/validation cross-entropy and accuracy, retaining the original optimizer loss and macro-F1 values. Cross-entropy is computed with the same unpenalized definition for training and validation; the original optimizer loss is recorded separately.
- Added [plots.py](src/poppypurev2/plots.py) and `poppypure plot-phase4 --output ...`. Replayed exactly the two original MLP configurations with the same inputs, scaler fitting, seed, training settings and stopping rules. Verified every original loss/F1 trace, selected epochs (57/41), and identical original validation predictions/probabilities. Original 25-run models, metrics, CSVs and records remain unchanged. The two diagnostic replays are documented separately from the comparison budget.
- Generated **seven figure sets in PNG/PDF (14 files)** under `reports/phase4/plots/`: both MLP learning curves (loss/F1/accuracy), selected SVM count/normalized confusion matrix, per-class precision/recall/F1, all-run comparison, training/validation gap, and condition comparison. Saved loss-trace CSVs, replay records and checksums. Visually inspected the whole-image learning curves, confusion matrix and experiment comparison for legibility and correct labels.
- Added [the figure index and metric explanations](reports/phase4/plots/README.md), linked it from findings/README, and explicitly distinguished **384 fitting images / 82 validation images / 82 reserved test images**, **18 automated software checks**, and **25 experiment configurations**. The 85.37% result is 70/82 correct validation predictions; no complete test-set evaluation has occurred.
- Explained macro F1 as the equally weighted mean of seven class F1 scores, with no universal passing cutoff. The recorded 0.8591 is a useful baseline relative to the 0.1546 random control, while class-30/40 F1 values (0.64/0.72) identify the current improvement targets.
- All 18 existing tests passed after logging changes. Formatting, lint, compilation, figure existence/checksums and unchanged original-run artifacts were verified. No new model family/hyperparameter search, raw-image processing or test scoring occurred.

### 33. Started Phase 5 and prepared the ResNet implementation

- The user authorized Phase 5. Read project instructions, workflow, the frozen allocation and model training plan. Kept the seven **by-volume** labels and 384/82/82 split.
- Installed pinned PyTorch **2.9.1** and Torchvision **0.24.1**, updating the lockfile. Sandboxed installation failed DNS; the approved network retry succeeded. Downloaded the official, explicitly versioned ImageNet weights into `models/pretrained/`: ResNet18 `IMAGENET1K_V1`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; ResNet50 `IMAGENET1K_V2`, SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. Verified published filename hash prefixes before accepting downloads. Raw images remain on Backup Plus.
- The sandbox incorrectly reported MPS unavailable. An approved hardware probe outside the sandbox confirmed **Apple M4, 16 GiB unified memory, working MPS tensor operations**. Plan: MPS for encoders/fine-tuning, CPU for small frozen heads; record actual devices.
- Added `configs/phase5-v1.json`, `deep_inputs.py`, `deep.py` and `deep_reports.py`, plus `train-deep`. Implemented checked RGB/crop inputs, exact neutral mean padding and bilinear letterboxing, five-region mean pooling, fixed label-index conversion, train-only embedding standardization, linear/MLP heads, best-macro-F1 checkpoint restoration, limited layer4 tuning with fixed BatchNorm statistics, and per-run curves/predictions/checksums. Training commands reject existing output directories and save a failure record if interrupted by an exception.
- Added five meaningful CNN tests covering geometry/normalization, image-level pooling, volume class mapping, gradient/BatchNorm boundaries, and train-only scaling/earliest-best checkpoint restoration. **All 23 software checks passed**; formatting and lint passed. These counts describe code checks, not trained classifiers.
- Training-only preview generation and numerical/visual input acceptance precede the actual eight-configuration comparison. This entry records implementation readiness; no deep learning result or test accuracy is claimed yet.

### 34. Accepted CNN inputs and launched the bounded comparison

- Generated four sheets covering **28 training images / all 28 class × background × magnification strata**. Visually inspected all sheets; accepted aspect-preserving overviews and five native detail views. Grain structure remains visible; no camera text was visible in these reviewed crops. Saved `reports/phase5/inputs/REVIEW.json`. Masks and illumination correction remain disabled.
- Benchmarked one real five-region training input on MPS: output **1×512**, finite values; cold forward **1.078 s**, warmed forwards **0.0335 / 0.0146 s**. Saved `hardware_benchmark.json`.
- Added persisted-model acceptance checks: all cached IDs/labels/preparation membership, input/output checksums, training-only scaler, restored validation probabilities, full RGB inference versus cached embeddings, earliest-best checkpoint, and unchanged frozen blocks/BatchNorm/scaler during tuning.
- Launched `poppypure train-deep --output experiments/runs/phase5-v1` with approved local MPS access. Logs go to `reports/phase5/training.log`; settings and input provenance are saved before fitting. The bounded sequence is four ResNet18 heads, two ResNet50 heads on the selected representation, then two layer4 learning rates initialized independently from the best frozen model. This entry records a running experiment, not completed findings.
- Rechecked formatting/lint, compilation and locked offline dependency installation before launch. The 82 test images remain outside the development command.

### 35. Completed the six frozen-encoder comparisons

- ResNet18 extraction completed for **384 training / 82 validation** images, both crop and pooled detail: 512 columns per image, all finite; elapsed **74.70 s** on MPS. Separate NPZ files preserve original IDs/volume targets/preparation groups. Training/validation SHA-256 values are `f4d949b58ef696bd51a29362de562b177c98d9af1a6ffb57df4ae71fee69a2f4` and `c4e29e33b85410915efd7d8cdf5db32a9635d85366dba2d9ff2d09c0ca38d4b6`.
- ResNet18 validation macro F1: crop linear **0.8599**, crop MLP **0.8818**, detail linear **0.9099**, detail MLP **0.9249**. The detail MLP correctly classified **76/82 = 92.68%**, selected epoch 28; its training macro F1 was 1.0. These are validation results; checkpoint acceptance remains pending until the end of the complete sequence.
- Recorded the detail representation selection before ResNet50 extraction. ResNet50 detail linear reached validation macro F1 **0.8608 / 86.59% accuracy**; detail MLP **0.8988 / 90.24%**. Both selected epoch 8. The larger backbone did not win this bounded frozen-feature comparison.
- Saved all six full checkpoints, predictions, metrics, learning traces and records, then recorded the best-frozen selection and checkpoint checksum. Both planned fine-tuning rates now initialize independently from **ResNet18 detail + MLP**, preserving its fitted scaler. No optional fusion/seed search or final-test processing has occurred.

### 36. Prepared the final Phase 5 review while fine-tuning runs

- Snapshotted source preprocessing, the immutable dataset freeze (including archive identity), and the current user-confirmed preparation provenance beside the training settings. Expanded `experiments/SCHEMA.md` to document deep checkpoints, label mapping, scaler buffers, probabilities, timing conventions and acceptance checks.
- Added `scripts/review_phase5.py` as a post-training review command. It requires eight accepted runs, compares all 21 classical visual configurations and eight deep configurations using the declared rule, saves the selected checkpoint/recipe and a proposed fixed Phase 6 refit schedule, generates per-class/condition figures and a validation error gallery, and writes the findings/figure index. It does not fit another model or process test images. Formatting/lint passed. It has not been executed while training is incomplete.
- The first tuning rate's first five validation macro-F1 values were **0.8239, 0.8921, 0.8664, 0.8651, 0.8738**. Best so far is epoch 2, below frozen detail+MLP **0.9249**. Kept the planned patience/schedule and preserved the frozen checkpoint; no settings were adjusted after seeing these values.
- Updated README, PLAN and the model training plan to mark Phase 5 **running**, with setup/method links. The actual training source/settings remain unchanged during execution. Completion, acceptance and final selection are still pending.
- The first fine-tuning trial subsequently stopped after **seven epochs**, restoring epoch **2**: validation macro F1 **0.8921**, accuracy **73/82 = 89.02%**. Saved its checkpoint, metrics, predictions and seven-epoch trace. The second rate (0.00003) then started from the original best frozen checkpoint, as specified; the first fine-tuned weights do not initialize it.
- Preserved the exact Phase 5 training source, configurations, `pyproject.toml` and lockfile under `experiments/runs/phase5-v1/source_snapshot/`, with per-file checksums. Verified its source fingerprint matches both the running source and the recorded encoder provenance: `21b39c7ed1503b1a751f9576f87d318f11becbb6931c3ea01cff769a675b84d9`. This keeps the executed implementation recoverable after future edits.
- The second tuning trial reached a new best **0.8872 validation macro F1 at epoch 10**, resetting its existing patience counter; epoch 11 was lower. Kept the original 20-epoch maximum and patience 5. The frozen model remains ahead. Verified source-snapshot hashes, provenance content and the separately preserved review-script bytes; review generation still waits for all eight accepted runs.

### 37. Completed and verified all eight Phase 5 configurations

- The second tuning rate stopped after **15 epochs**, restoring epoch **10**: validation macro F1 **0.887193**, accuracy **73/82 = 89.02%**. The first rate restored epoch 2 after seven epochs: **0.8921 / 89.02%**. Both initialized independently from the same frozen checkpoint and both scored below it. Actual fitting/evaluation time was approximately **956 s / 1,979 s**, respectively, on MPS; no settings were changed during training.
- All **eight planned deep runs completed successfully**. ResNet50 detail extraction took **75.71 s**, producing 384×2048 / 82×2048 matrices. Training/validation NPZ SHA-256 values: `09ecb2045673073b09b35fd379ec59ab48f4e0cf8887750d86533c032910e6fb` / `1838a0a30901cdd5b4eba60e8776a250255dbb4584afcb88fb518487b472617f`.
- Reloaded every saved checkpoint. Reproduced all 82 saved validation predictions/probabilities for every run, checked frozen-head scaler statistics against training only, verified real RGB inference against cached embeddings, and confirmed earliest-best checkpoint selection. For both fine-tuned models, all earlier blocks, every BatchNorm running statistic and the stored scaler stayed identical to initialization. All checks passed in `experiments/runs/phase5-v1/ACCEPTANCE.json`.
- Ran `scripts/review_phase5.py` after acceptance. Compared **21 classical visual configurations + eight deep configurations** under the declared macro-F1/tie rule. Selected **frozen ResNet18, five native detail regions averaged into a 512-vector, training-fitted scaler, 128-unit ReLU/dropout MLP, seven by-volume classes**, seed 42, selected epoch **28**. No optional fusion or confirmation-seed search was added.
- Selected validation macro F1 **0.9249066**, accuracy **76/82 = 92.68%**, balanced accuracy **0.9253247**, volume-class MAE **1.09756 points**. Compared with the classical SVM, gains are **0.0657743 macro F1 / 7.3171 accuracy percentage points**. Training macro F1 is 1.0. The 20% class F1 decreased from 0.9524 to 0.8421; 30%/40% F1 improved from 0.64/0.72 to 0.8462/0.8696.
- Reviewed the six validation errors: 20→0, 20→30, 20→40, 30→50, and two 40→30. Saved IDs/probabilities and the overview/detail error gallery. Derived binary detection has sensitivity **98.59%**, specificity **100%**, one false negative and no false positives. These remain secondary to seven-class macro F1.
- Locked `reports/phase5/MODEL_SELECTION.json`, identifying the exact selected checkpoint SHA-256 `0903ec0ea94501411fe1377c55ac892ea67f9fc0243785699882754b4bb45ae0`. Proposed Phase 6 refit: **466 train+validation images, fixed 28 frozen-head epochs**, same weights/scaling/input recipe/seed, then one 82-image test evaluation. This refit and test have **not** occurred.

### 38. Delivered Phase 5 graphs, findings and integrity checks

- Generated **12 figure sets / 24 PNG+PDF files**: eight training/validation loss/F1/accuracy curves, selected confusion matrix, all-deep-versus-classical comparison, per-class precision/recall/F1 and four capture-condition scores. Added the figure index, methods, complete findings, selected recipe and validation error gallery in `reports/phase5/`.
- Visually inspected selected frozen and fine-tuned learning curves, model comparison, confusion matrix, class/condition charts and all six error-gallery entries. The initial per-class legend obscured bars; moved it outside the axes. Preserved original figures and corrected-generator bytes under `plot_revisions/`, kept the originally executed review script beside model runs, and recorded both versions/checksums. Updated active figure hashes and report summary; no model, prediction, metric or selected recipe changed.
- Final report checks passed: **24 active figure hashes and file formats**, all eight accepted runs, 29 considered visual candidates, checkpoint/selection identity, 82 validation predictions / 76 correct / six error images, unchanged original manifest and v4 split hashes, and exact training-source snapshot identity. Saved `reports/phase5/REPORT_ACCEPTANCE.json`. Source code remains fingerprint `21b39c7ed1503b1a751f9576f87d318f11becbb6931c3ea01cff769a675b84d9`; final script formatting/lint and compilation passed. The 23 software checks passed earlier with this unchanged training implementation.
- Updated README, PLAN, MODEL_TRAINING_PLAN and Phase 5 methods to mark **Phases 0–5 complete / Phase 6 next**. Phase 5 test use is false throughout. Neither final refitting, final-test evaluation nor inference packaging was started.

### 39. Started Phase 6 with the user-approved final refit

- The user explicitly retained the proposed refit and authorized the next step. Read instructions, current workflow, locked Phase 5 recipe and completion gates. Verified Backup Plus is connected. Preserved **466 train+validation / 82 final test**, seven **by-volume** classes, frozen ResNet18 detail + MLP and **28 fixed head epochs**.
- Added `configs/phase6-v1.json` and a pretest [run protocol](reports/phase6/RUN_PROTOCOL.md). Predeclared a same-466-image refit of the already selected whole-image SVM as a fixed reference. It cannot replace the primary based on test results. Declared 2,000 paired, class-stratified preparation-unit bootstrap replicates (seed 1042) before scoring.
- Implemented `final.py`: verified cache/membership joins, fresh 466-image scaling/head fitting without a validation argument, fixed final-epoch retention, reference fitting, source/environment snapshots, full pretest model acceptance and a seal over checkpoints/settings/code. Added exclusive final-test access records, 82-image inference/prediction persistence, class/condition/date summaries and bootstrap intervals.
- Implemented `inference.py` and `predict`: full weights and source preprocessing/label mapping are packaged; inference builds the architecture without pretrained downloads and does not read project configs or raw data. Added clear missing/corrupt/small-image errors. Added `refit-final`, `evaluate-final` and `report-final` commands plus a report/figure/card generator using saved outputs.
- Added five meaningful tests for fixed epoch/final-state retention, one-time test access, grouped bootstrap resampling, copied-bundle inference without dataset/pretrained/project configs, and invalid mappings/checksums/small inputs. **All 28 software checks passed.** Training and test scoring had not occurred during these checks. Final formatting/lint/compilation and preflight precede execution.

### 40. Corrected a pretest verification precision mismatch

- The initial 466-image refit completed all 28 epochs and saved the CNN/head/scaler and fixed SVM. CNN persistence, full RGB inference, frozen encoder/BatchNorm checks and SVM persistence passed. The next scaler verification failed because its expected mean used float32 accumulation, while StandardScaler uses float64; largest reported discrepancy was approximately **0.0000343**. This was a verification-calculation error, without a change to fitting or model settings.
- Added `verify_reference_scaler` to compare mean/variance in float64 and check the full fitted sample count. Added a regression test using constant float32 descriptors that demonstrate the accumulation difference, including rejection of a scaler fitted on too few rows. All **29 software checks passed** after correction.
- Preserved the complete initial run/model under `phase6-v1-pretest-attempt1` / `poppypure-v1-pretest-attempt1`, including its exact source snapshot, failed verification log and `PRETEST_FAILURE.json`. Verified no test directory or access record existed. The corrected retry keeps all model/training settings and the fixed epoch count unchanged; neither attempt used test images for fitting or verification.

### 41. Completed the 466-image refit and sealed the final predictors

- Corrected pretest verification passed. Refit uses **466 images / 465 preparation units**, in frozen assignment order, fresh scaling on all 466 embeddings, seed 42 and **28 fixed MLP epochs**. The ResNet18 encoder/BatchNorm stays identical to the explicit pretrained weights. No refit validation or early stopping was used; final training macro F1 is 1.0. The fixed SVM reference also fits only these 466 images.
- Compared the retry against the preserved first attempt: **every CNN/head/scaler tensor and the complete serialized checkpoint bytes are identical**. The float64-verification fix changed no trained model or training setting. Kept both attempts' source snapshots and the original failed verification trace.
- Passed final-state/scaler membership, persisted predictions, three full RGB-versus-cached embedding checks, pretrained encoder/BatchNorm identity, and reference-model/scaler persistence. Saved `REFIT_ACCEPTANCE.json`, 28-row training-only curves, membership, fit record and complete runtime bundle.
- Copied the actual model bundle and one fitting image into a temporary project-local directory and invoked the real CLI from `/private/tmp` with a missing project configuration. Prediction succeeded using only the copied bundle/image, without raw-data/pretrained-file access by the predictor. Saved `CLI_PORTABILITY.json`, removed the temporary verification copies and confirmed no test access record existed.
- Saved `PRETEST_LOCK.json` over primary/reference checkpoints, preprocessing/recipe/dependencies, settings and exact source identity. Both predictors and their comparison role are fixed before test inference. Final test evaluation is the next action; no test-driven fitting or selection occurred.

### 42. Evaluated the sealed predictors on the complete final test set

- Ran `poppypure evaluate-final --device mps` after the pretest seal and portable CLI checks. Exclusive `TEST_ACCESS_STARTED.json` / `test/` records were created before decoding test images. Processed **all 82 reserved test images / 82 preparation units** in one final scoring event for the locked primary and predeclared SVM reference. Every source hash matched the frozen manifest. No model settings or selection changed during/after scoring.
- Final **ResNet18 + MLP**: **75/82 = 91.4634% accuracy**, macro F1 **0.9138361**, balanced accuracy **0.9145022**, volume-class MAE **1.3414634 points**. Fixed same-466-image **SVM reference**: **74/82 = 90.2439%**, macro F1 **0.8975868**, balanced accuracy **0.9036797**, MAE **2.9268293 points**. The test result concerns the fresh 466-image model; the earlier 92.68% score concerns development validation of the 384-image model.
- Primary test errors: `img_0080` 10→0, `img_0097` 20→10, `img_0144` 20→30, `img_0170` 30→20, `img_0221` 30→40, `img_0284` 40→30, `img_0337` 50→100. Six errors differ by 10 percentage points; the final error differs by 50 points and its inspected photo has visible blur. No causal attribution, relabeling, post-result filtering or tuning was made.
- Derived binary detection: sensitivity **70/71 = 98.59%**, specificity **11/11 = 100%**, one false negative (10%→pure), no false positives. Weakest test class F1 is 30% (**0.8333**), then 20% (**0.8571**).
- Saved all 82 primary/reference prediction rows with source/preparation/condition/date fields and probabilities/margins, raw inference logits/features, full class/condition/date metrics and completed access record. The one magnification-conflict test image remains in the 82-image overall result; trusted-condition tables cover 81 images.
- Computed the predeclared **2,000 paired preparation-unit bootstrap replicates** from saved predictions only. Primary 95% accuracy interval **0.85366–0.96341**, macro-F1 interval **0.85110–0.96415**. The paired model-difference intervals include zero; the observed CNN advantage is one correct image, without a claim of decisive superiority. Saved every replicate and interval. No additional model inference or test-driven selection was used for interval calculation.
- Exact evaluated checkpoint SHA-256: `520d47d80da1c3d8f5c2d689974a6ecbceee792dba41eae4c162fd072b4d69d1`. It remains the packaged 466-image version; no all-548-image refit was performed.

### 43. Delivered final reports, portable runtime and release verification

- Ran `report-final`, using saved predictions rather than rescoring models. Generated **five figure sets / 10 PNG+PDF files**: training-only 28-epoch refit curves, count/normalized test confusion matrices, per-class results with SVM F1, paired test comparison/intervals and capture-condition comparison. Added the seven-error overview/detail gallery, full technical report, dataset card, model card and metrics/interval copies under `reports/phase6/`.
- Visually inspected all five chart sets and all seven error images. Confirmed refit curves contain no validation/test traces, matrices have fixed seven-class axes, and chart legends do not obscure results. Reconciled confusion counts and all report metrics with original prediction CSVs.
- Created a genuinely fresh temporary source copy and virtual environment, installed pinned dependencies **offline with the lockfile**, then ran the real CLI from `/private/tmp` against a copied model and copied fitting image. No project configs, raw dataset or pretrained files existed in the copied project. Prediction probabilities matched the pretest portability check exactly. Saved `CLEAN_ENV_INSTALL.log` and `CLEAN_ENV_ACCEPTANCE.json`; removed temporary copies/environment.
- Packaged `models/poppypure-v1/` with full checkpoint/scaler/preprocessing, recipe/version/dependencies, model/report/dataset documentation, plots/errors and the exact accepted `runtime_source/`. Added standalone installation/prediction instructions and `RELEASE_MANIFEST.json` with per-file hashes. Copied runtime source matches the sealed source snapshot. Documentation additions did not change any sealed model/runtime artifact.
- Final acceptance passed: all sealed bundle/test/release hashes, unchanged original manifest/v4 split/Phase 5 selection, disjoint 466/82 image and preparation membership, 28 fixed training-only epochs, all 82 saved prediction targets/order, arithmetic replay from saved logits, confusion counts **75/82 and 74/82**, seven error records, all 10 figure formats/hashes, 2,000 bootstrap records and one bootstrap-score check against standard metrics. Saved `reports/phase6/REPORT_ACCEPTANCE.json`. Verification did **not** rerun model inference on test images.
- Final formatting/lint and compilation passed; **29 software checks** passed before the accepted refit with this unchanged implementation. Updated README, PLAN, MODEL_TRAINING_PLAN, execution protocol and record schema to mark **Phases 0–6 complete**, distinguish final test from development validation and document the portable CLI. Original project files, source images, earlier experiments, failed pretest attempt and final evaluated checkpoint are preserved.

### 44. Rechecked prediction readiness — 23 October 2026

- In response to the user's readiness question, verified every release-manifest file checksum and the checkpoint checksum in `bundle.json`. Both the clean-environment and final-report acceptance records remain passed.
- Confirmed the installed `poppypure predict --help` command works and checked the saved final metrics: 75/82 correct, accuracy 91.46%, macro F1 0.9138. The packaged model is ready for CLI image predictions. No retraining, test rescoring or model changes were performed.

### 45. Built the user-requested local prediction app — 23 October 2026

- The user expanded the scope to a minimal local browser app and requested README capture specifications and launch instructions. Added `app/` with paste, drag-and-drop/file selection, image preview, automatic classification, seven by-volume result levels, loading/reset/retry states and readable image errors. Added the Mac `run_app.command` launcher.
- The Python standard-library HTTP server binds to loopback only and loads the existing model once. It uses the release's unchanged bundle loader, RGB/orientation processing, detail-region extraction, classifier and probability mapping. Uploads use temporary files deleted after processing; no image history or cloud service is added. No dependencies, training code, sealed runtime files, checkpoint or test predictions were changed.
- Updated README with setup/run/stop commands, file support and practical phone photo capture specifications. Distinguished input size acceptance from image quality and documented the existing absence of sample localization/background removal/focus checking. App verification is in progress; results will be appended here.

### 46. Verified the local app and documented its use — 23 October 2026

- Six HTTP tests passed in `tests/test_local_app.py`: static/health endpoints and path boundaries, by-volume JSON output, invalid/oversized upload rejection, foreign-origin/host rejection, recovery after an image error and rejection of concurrent inference. Socket creation required an approved sandbox escalation; the app itself remains bound to `127.0.0.1`.
- Verified the real interface in an isolated installed Chrome session: initial state, file-input selection, drag-and-drop, simulated clipboard image paste, clear/reset, removal of a previous result after an invalid format, undersized-image error and successful recovery. Checked a 390-pixel layout without horizontal overflow and no JavaScript exceptions. Inspected desktop/mobile screenshots. Browser check and evidence: `scripts/check_local_app_browser.mjs`, `reports/generated/local-app/BROWSER_ACCEPTANCE.json` and three screenshots. The system clipboard was not accessed; only image-bearing clipboard events were simulated.
- Compared a real photo upload with the unchanged CLI inference path: model version, by-volume class, adulteration flag, input SHA-256 and all seven probabilities agree. Malformed image bytes return a readable HTTP 400 error. Verified all **51 packaged release file hashes** and exact equality of current `src/` Python files with the sealed runtime source. Saved `reports/generated/local-app/API_ACCEPTANCE.json`; reusable check is `scripts/check_local_app_api.py`. This was an app functionality check, not an accuracy study; the reserved test set was not accessed.
- Python formatting/lint, JavaScript syntax, launcher shell syntax and launcher help checks passed. The app adds no dependencies and no changes to the sealed model, runtime source or prior reports. README now explains launch/stop, supported uploads, local processing and photo capture rules. Stopped the temporary verification server/browser after checks; use the documented launcher to start a daily-use session.

### 47. Checked source readiness and prepared Git guidance — 23 October 2026

- The user asked whether the code is ready and which files should be pushed. Checked source/app/test/script formatting and lint: all 34 Python files passed; app JavaScript and Mac launcher syntax also passed. Rechecked the passed app API/browser and final-report acceptance records and all 51 release file hashes. No new model evaluation or training was performed.
- Confirmed `PoppyPurev2` is not currently a Git repository. Measured the final checkpoint at 42.96 MiB and local experiment runs at approximately 461.69 MiB. Added `GIT_GUIDE.md` with a source/metadata/report commit list, local-only artifact exclusions, first-commit commands and a separate release-bundle distribution procedure. Linked it from README and explicitly documented that a source-only checkout needs model weights before predicting.
- Added generated feature/embedding directories, report logs and the optional final-model ZIP to `.gitignore`. Kept the existing exclusions for raw external-drive links, model directories, run archives, environments and caches. These changes preserve the local files; no Git repository, commit, remote, push or release upload was created.
- Verified the ignore rules with Git in a temporary fixture: all 12 representative local-only paths were excluded and all 13 representative source/report/metadata paths remained eligible. Removed the fixture afterward; the actual project still has no `.git` directory.

### 48. Prepared the final-model download assets and local repository — 24 October 2026

- The user created `shadesin/PoppyPurev2`, authorized committing locally and preparing a release ZIP, and explicitly prohibited pushing for now. Verified the repository metadata through the GitHub connector: owner `shadesin`, default branch `main`, size zero. Read the owner's authenticated profile for commit attribution. Initialized Git only inside `PoppyPurev2`, configured the HTTPS origin and repository-local author name/GitHub noreply email. No remote write was made.
- Added `scripts/package_model.py` to build/verify the model asset without training, external-drive access or changes to the sealed model. Packaging uses only the release-listed files plus the release manifest, fixed ZIP metadata and post-build per-file verification. Created `poppypure-v1.zip` (**42,856,775 bytes; 52 files**) and its `.sha256` sidecar. ZIP SHA-256: `adc394eac6a7fcdede9ee146242f389a22100b2548f8068ed567221090069d9c`. The checkpoint SHA-256 remains `520d47d80da1c3d8f5c2d689974a6ecbceee792dba41eae4c162fd072b4d69d1`.
- Added the committed `releases/poppypure-v1/ASSET_MANIFEST.json` and prepared release notes. Ignored the ZIP/checksum sidecar in Git while keeping the small verification metadata. README now explains source download, the separate model asset, verification, correct extraction location, cross-platform startup and the current unpublished status. Updated Git guidance to match the real repository and packaging procedure. Verification of a source checkout plus extracted ZIP is the next check before committing.
- Initial staging identified original CSV CRLF endings as whitespace errors. Preserved all frozen CSV/JSON bytes and added `.gitattributes` with byte-preserving rules for those records, `cr-at-eol` whitespace handling for CSV and LF checkout rules for source. This prevents cross-platform Git normalization from invalidating recorded manifest/split hashes.

### 49. Verified installation from source plus ZIP and finalized the local commit

- Created an isolated checkout from the staged Git files, initially containing no model directory, raw images, pretrained weights, features or embeddings. Installed the locked project into a fresh virtual environment offline using the existing uv cache, verified the downloaded-style ZIP with the documented command, and extracted it into the repository root. All 52 archived file checksums passed and the documented model layout was created.
- Started that isolated checkout's real app on CPU/loopback port 8767. Its actual upload, model/version/volume label and all seven probabilities matched the CLI. Malformed uploads returned a readable error; a deliberately incomplete ZIP was rejected by the download verifier before extraction. All 51 sealed model release files and current training/inference source remained unchanged. No reserved-test images were accessed. Saved the small tracked `releases/poppypure-v1/PACKAGING_ACCEPTANCE.json` and retained API evidence under ignored generated reports.
- Formatting/lint passed for all 35 Python files; JavaScript and Mac launcher syntax passed. Reviewed staging: source, app, configs, tests, metadata, reports, documentation and small release records; no raw-data links, environments, model weights, experiment archives, feature/embedding caches, generated verification screenshots or release ZIP/checksum were included. Preserved byte-level CSV/JSON identities with `.gitattributes` and passed Git's staged whitespace check.
- Stopped the verification server and removed the temporary installed checkout. The initial local commit includes this workflow entry and the complete reviewed source. Remote remains configured as `https://github.com/shadesin/PoppyPurev2.git`; pushing, tags and GitHub Release creation/uploads remain pending under the user's explicit instruction.

### 50. Started the user-authorized GitHub Release publication — 24 October 2026

- The user explicitly asked the assistant to upload/publish the model release. Confirmed the user had already pushed source commit `6726fafc36ff0b64a6581edd9f1afc6f23123c78` to `origin/main`. Authenticated release access with the existing local Git credential, kept exclusively in process memory; no credentials were printed or written to project files. GitHub reported no existing releases.
- Reverified the model ZIP and all 52 packaged files before upload. Updated README, Git guide and release notes with the intended `v1.0.0` release and direct ZIP/checksum download URLs, replacing obsolete unpublished-status instructions. Prepared the documentation commit for publication with the release. Publication and remote-download verification results will follow in the next entry.

## Logging instruction — recorded with this file

The user requested `workflow.md` as the maintained project history. Future important actions must be appended here with their purpose, changes, validation, outcomes, and remaining limitations. Keep proposed techniques distinct from completed experiments, and preserve earlier entries when recording corrections or revised decisions.
