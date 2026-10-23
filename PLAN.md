# PoppyPure v2: seven-class adulteration detection

## Objective and scope

Build a reproducible image classification project that identifies the **percentage of semolina adulteration by volume** in poppy seeds. The classifier predicts exactly one of seven volume-percentage classes: **0%, 10%, 20%, 30%, 40%, 50%, or 100%**. Here 0% is pure poppy seeds and 100% is pure semolina. The seven-class task is primary; pure (0%) versus adulterated (>0%) detection is a secondary evaluation derived from those predictions.

The project includes dataset preparation, image processing, model development, testing, inference code, and a technical report. It does **not** include a web app. Code, manifests, models, reports, and local links stay inside `PoppyPurev2/`. When work begins, source archives and extracted raw images will live on the external `Backup Plus` drive, with links inside `PoppyPurev2/data/`. The original `PoppyPure/` folder remains untouched.

The [original Mendeley dataset](https://data.mendeley.com/datasets/p2gvvnh8y4/1) is the canonical image source. The [later Mendeley dataset](https://data.mendeley.com/datasets/nx8m9kxy26/1) is a renamed subset of the original images; it confirms the labels are **by volume** and provides no independent images. We will never merge the two archives as separate observations.

## Phase 0 — set up a reproducible project

**Objective:** make every later result rebuildable from documented inputs and commands.

**Tasks**

1. Create a Python project with pinned dependencies, configuration files, formatting/linting, and a command-line workflow. Use notebooks only for exploration; final processing and training run from scripts.
2. Record source URLs, DOIs, class mapping, random seeds, and output paths. Add archive checksums when the Phase 1 download is authorized. Keep raw image bytes unchanged and exclude large data and models from Git; version the manifest and split definitions.
3. Set up experiment records that capture the code version, dataset manifest version, split IDs, preprocessing configuration, model configuration, and metrics for every run.

**Deliverables:** project skeleton, environment lock, README with setup/run commands, and experiment-record format.

**Exit gate:** a clean environment can run a small end-to-end smoke test from documented commands.

**Status:** complete. The pinned environment, config, CLI, record schema, tests, and documented smoke command have been verified. Source archive checksums remain a Phase 1 task.

## Phase 1 — audit and prepare the image dataset

**Objective:** establish exactly which images and independent samples are available.

**Tasks**

1. **Only when instructed to begin**, download the original archive to `/Volumes/Backup Plus/PoppyPurev2/source_archives/`, extract it without altering source images to `/Volumes/Backup Plus/PoppyPurev2/raw/`, and create `PoppyPurev2/data/source_archives` and `PoppyPurev2/data/raw` as symbolic links to those folders. Verify the downloaded archive checksum and link targets. Do not use the later subset as additional training data.
2. Build `data/manifest.csv` with the source path, **volume-percentage class**, background, magnification, file size, SHA-256, image dimensions, EXIF capture data when available, and decode status.
3. Account for every archive entry. An initial directory audit found 569 JPEGs: 555 within background/magnification folders and 14 directly in class folders. Review the 14 individually and document whether each is a valid sample or a reference image.
4. Find exact duplicates by SHA-256 and near duplicates by perceptual similarity; review candidates in contact sheets. Check labels, corrupted files, blur, exposure, visible camera markings, and class balance.
5. Assign a conservative `group_id` to views likely captured from the same physical preparation or session. Record the evidence and uncertainty behind each grouping decision. Do not silently discard or relabel images.

**Deliverables:** verified manifest, included/excluded image list with reasons, duplicate report, contact sheets, class/group counts, and dataset card with source attribution.

**Exit gate:** every image is accounted for and the dataset used for modelling is frozen and traceable.

**Status:** complete. The source checksum is verified, external folders are linked, all 569 images are audited, and 548 close-ups are included after seven duplicate-copy and 14 overview-reference exclusions. See `reports/phase1/AUDIT.md`. Subsequent user confirmation establishes individual mixture preparations; active v4 preparation units supersede the earlier class-wide grouping, while preserving verified same-scene views together.

## Phase 2 — lock the evaluation protocol

**Objective:** measure performance on genuinely separated image groups, as far as this dataset permits.

**Tasks**

1. Put all views, crops, and exact/near duplicates of a group on the same side of every split. Save a frozen internal holdout and use group-aware cross-validation on the remaining groups for model selection.
2. Check that all seven classes are represented appropriately in development and holdout data. If grouping makes a balanced holdout impossible, report that limitation and redesign the split before training.
3. Predefine the primary selection metric as **macro F1 across the seven volume-percentage classes**. Supporting metrics are balanced accuracy, per-class precision/recall, confusion matrix, and absolute class-distance error measured in **volume-percentage points**. Separately report pure-versus-adulterated sensitivity and specificity.
4. Predefine challenge tests for black versus white background, 4× versus micro imaging, and capture-session differences where the data supports them. Treat them as checks for shortcuts and domain shift.
5. Keep the holdout unseen during feature design, tuning, threshold selection, and model comparison. Fit any scaler, PCA, feature selector, or calibration step only within training folds. This follows [grouped-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [data-leakage](https://scikit-learn.org/stable/common_pitfalls.html) guidance.

**Deliverables:** versioned split files and a written evaluation specification.

**Exit gate:** split assignments, primary metric, and challenge tests are frozen before model fitting.

**Status:** complete under [active protocol v4](reports/phase2/TRAIN_VAL_TEST.md): **384 training / 82 validation / 82 test images**, each covering all seven classes and all four capture conditions. This supersedes the earlier condition holdouts and grouped condition-fold selection. The user confirms individual preparations; merge confirmed same-scene views and preserve those groups across partitions. Select using validation macro F1; optionally refit the locked recipe on 466 train+validation images before one final test. Earlier protocols remain as history.

## Phase 3 — build and validate image processing

**Objective:** extract visual information from the seeds and semolina without letting background or camera artifacts dominate.

**Tasks**

1. Develop and visually review processing on v4 training images only, spanning all four conditions. Use validation for later processing/model comparison and reserve test images for final evaluation. Decode consistently into RGB, apply EXIF orientation then discard metadata, and record failures/dimensions/quality flags. Do not automatically discard flagged images or smooth away grain texture. See [the initial implementation report](reports/phase3/PROCESSING.md).
2. Inspect development images from every class and available condition to choose useful scales. Preserve aspect ratio when resizing; pad if a fixed input shape is required. Compare whole-image inputs with mixture-region crops. If resizing makes individual grains too small, evaluate detail tiles and pool representations at image level. A tile's local composition need not match the source volume label; do not treat tiles as independently labelled concentration samples. Create tiles only after assigning source-image groups to splits.
3. Prototype an optional foreground mask using colour and brightness in HSV or Lab, with Otsu/adaptive thresholds and only light morphological cleanup where suitable. Inspect overlays across all seven classes and development conditions. Use a mask only if it consistently retains both poppy seeds and semolina; otherwise use a checked crop or full image. Visible area fractions must not be interpreted as volume percentages.
4. For the classical baseline, calculate foreground-normalized colour histograms in HSV or Lab and a small, documented set of texture descriptors: correctly binned uniform local binary patterns (LBP) and grey-level co-occurrence matrix (GLCM) statistics at selected distances and angles. Consider edge or particle-size features only if segmentation passes visual review. Avoid the old negative-entropy calculation and indiscriminate 128×128 whole-image resize.
5. Compare full-image, crop, mask and feature variants on the same v4 training/validation allocation. Fit learned processing on training only. Try colour/illumination normalization as a recorded experiment. Freeze chosen processing and use the same code at training and inference.
6. Add focused checks for decoding, feature dimensions, label mapping, deterministic results, and visual quality of crops or masks.

**Deliverables:** versioned preprocessing package, quality-control images, feature definitions, and checks.

**Exit gate:** representative training images from all seven classes and all four conditions process reproducibly; feature definitions and checks are complete. Test processing uses the locked recipe only at final evaluation.

**Status:** complete. RGB/orientation/metadata handling, whole/crop processing and detail regions are implemented and reviewed. Two background-mask methods were tested on 56 training images and visually rejected for general use; masks remain off. HSV/Lab colour, uniform LBP and normalized GLCM features are implemented and extracted for all 384 training images. See [feature definitions, mask decisions and checks](reports/phase3/FEATURES.md). Classifier-based representation comparison and final recipe selection occur in Phases 4–5.

## Phase 4 — train baselines and diagnose shortcuts

**Detailed experiment sequence:** [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md), including the bounded classical/handcrafted-MLP comparison and Phase 5 ResNet comparisons. Phase 4 training is complete.

**Objective:** learn how much predictive information comes from the mixture itself.

**Tasks**

1. Evaluate majority-class and stratified-random predictors.
2. Train metadata-only and background-only controls. High scores from these controls are evidence that capture conditions can reveal labels without seed analysis.
3. Train simple visual baselines using colour/texture features with regularized linear and tree-based classifiers. Compare whole-image and mixture-region variants on the fixed v4 train/validation split.
4. Save every run's configuration, per-image predictions, group-level results, and errors. Review confusion among neighbouring volume classes, especially 10–50%.

**Deliverables:** baseline comparison report and error gallery, broken down by class, background, magnification, and group.

**Exit gate:** reproducible baseline and shortcut-control evidence, saved validation predictions and checked pipelines. Investigate weak or shortcut-driven results within development, then run the planned frozen ResNet comparisons; those comparisons are not contingent on a classical classifier winning.

**Status:** complete. All 21 planned visual configurations and four controls passed; saved models/predictions reproduce their validation metrics. Selected whole-image SVM: macro F1 0.8591, accuracy 85.37%; best cropped-texture SVM: macro F1 0.8293, accuracy 82.93%. Metadata-only accuracy is also 85.37%, with lower macro F1 0.7980 and zero recall for the 30% class. See [Phase 4 findings](reports/phase4/FINDINGS.md). Test images remain reserved. Proceed to the planned crop/detail ResNet comparisons.

## Phase 5 — improve and select the classifier

**Objective:** improve seven-class performance only where error analysis shows a real opportunity.

**Tasks**

1. Run the planned frozen ResNet18 linear/MLP-head comparisons, then frozen ResNet50 linear/MLP comparisons and limited fine-tuning as specified in [MODEL_TRAINING_PLAN.md](MODEL_TRAINING_PLAN.md). Use the same v4 split, image-level pooling and training-only transforms/augmentation. Keep every region linked to its source group.
2. Consider seed/semolina segmentation or object-level classification only if reviewed errors show that whole-image models miss relevant visual differences. Build and inspect a small annotation set first.
3. Compare candidates on the same v4 validation set using macro F1 and condition-specific summaries. Save all trials; reserve test scoring for the selected recipe.
4. Calibrate predicted probabilities only if enough independent development groups are available. Do not present an uncalibrated softmax maximum as a verified confidence level.

**Deliverables:** selected model, frozen preprocessing, seven-class label mapping, and model-selection record.

**Exit gate:** one complete pipeline is fixed before the final holdout is opened.

**Status:** complete. All eight planned deep configurations passed checkpoint, prediction and fitting-boundary acceptance. Selected **frozen ResNet18 + 128-unit MLP on pooled detail regions**, validation macro F1 **0.9249**, accuracy **76/82 = 92.68%**, selected epoch **28**. Both ResNet50 heads and both tuning rates scored lower. Saved the full [recipe/Phase 6 refit schedule](reports/phase5/MODEL_SELECTION.json), [findings](reports/phase5/FINDINGS.md), curves, class/condition charts and error gallery. Final test remains reserved.

## Phase 6 — final evaluation and deliverables

**Objective:** produce an honest, reproducible result for volume-percentage classification.

**Tasks**

1. Run the selected pipeline once on the frozen internal holdout. Preserve predictions for every image and summaries for every group.
2. Report macro F1, balanced accuracy, per-class precision/recall, confusion matrix, volume-class distance errors, and pure-versus-adulterated sensitivity/specificity. Include image, preparation-unit and capture-condition counts. Resample preparation units if estimating confidence intervals.
3. Report performance by background, magnification, and capture session. Compare with baselines on the same evaluation units and discuss the failures.
4. Package a versioned model and one command-line prediction entry point. It must return one of the seven **by-volume classes**, model version, or a clear input error, using exactly the saved preprocessing pipeline.
5. Write a technical report, dataset card, and model card with attribution, inclusion/exclusion decisions, split protocol, experiments, final metrics, limitations, and reproduction steps.
6. Preserve the final evaluated model (optionally fitted on all 466 train+validation images) and its 82-image test results. Optionally refit the unchanged recipe on all 548 images for a separate deployment version.

**Deliverables:** trained model, inference command, full evaluation report, figures, dataset/model cards, and reproducible commands.

**Status:** complete. The locked recipe refitted on **466 images for 28 fixed epochs**, passed pretest/portable-inference checks, and scored **75/82 = 91.46% test accuracy / 0.9138 macro F1**. The same-466-image SVM reference scored **74/82 = 90.24% / 0.8976**. Saved the evaluated `poppypure-v1` checkpoint, all predictions, intervals, figures, error gallery, technical report and cards. Fresh-environment inference passed without dataset/config/pretrained assets. See [final report](reports/phase6/FINAL_REPORT.md). No test-driven model changes or all-548-image refit occurred.

**Completion gate:** a clean environment can recreate the manifest, splits, training, evaluation, and prediction. The conclusion must match grouped and challenge-test evidence. With no new physical samples, the result is an internal evaluation of this image dataset, not a claim about every future capture setting.

## Planned directory layout

```text
PoppyPurev2/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
├── data/
│   ├── source_archives -> /Volumes/Backup Plus/PoppyPurev2/source_archives/
│   ├── raw -> /Volumes/Backup Plus/PoppyPurev2/raw/
│   ├── manifest.csv
│   └── splits/
├── src/poppypurev2/         # audit, preprocessing, training, evaluation, inference
├── tests/                    # data and inference invariants
├── experiments/              # configurations, predictions, and metrics
├── models/                   # versioned artifacts; large files ignored by Git
└── reports/                  # quality-control figures and final report
```

**Current deliverable:** evaluated **poppypure-v1**, seven-class **by-volume** CLI prediction and complete reports. All planned phases are complete; the final evaluated checkpoint and original test predictions are preserved.
