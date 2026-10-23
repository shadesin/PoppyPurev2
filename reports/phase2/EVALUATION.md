# Phase 2 — frozen evaluation protocol

**Historical v1 — superseded before training.** The active protocol is [v2: revised evaluation and image-processing plan](REVISED_PLAN.md), using 414 development and 134 final-test images. The specification below records the earlier decision; its saved artifacts remain unchanged. The default CLI now verifies v2; use `--version v1` to verify this historical version.

**Status:** complete for limited internal evaluation. No classifier has been fitted or scored.

## What we can measure

The task remains seven-class semolina adulteration classification **by volume**: 0, 10, 20, 30, 40, 50 and 100 percent.

Phase 1 found no physical preparation IDs. Its conservative `group_id` keeps each entire volume class together: seven groups, one per class. A split that separates these groups cannot contain all seven classes on both sides. Capture dates cannot solve this: they are strongly associated with class, and a new date does not prove a new mixture.

Consequently, evaluation on unseen preparations is **infeasible with the available evidence**. We will measure transfer between capture conditions within this dataset. Shared material, unknown overlapping views, camera settings and class/date confounding can still inflate scores. There is no verified count of independent preparations. The number of photographs is not an independent sample count.

## Frozen partitions

Use every one of the 548 included close-ups. Preserve the Phase 1 manifest and all exclusions. Assign entire background/magnification conditions together, without a random image split:

| Role | Condition | Images |
| --- | --- | ---: |
| Development | Black background, 4× folder | 137 |
| Development | Black background, Micro folder | 140 |
| Final holdout | White background, 4× folder | 137 |
| Final holdout | White background, Micro folder | 134 |

**Development total: 277. Holdout total: 271.** Both partitions and both development conditions contain all seven classes.

| Volume class | Black 4× | Black Micro | White 4× | White Micro |
| --- | ---: | ---: | ---: | ---: |
| 0 | 20 | 20 | 17 | 16 |
| 10 | 20 | 20 | 20 | 21 |
| 20 | 16 | 20 | 20 | 17 |
| 30 | 20 | 20 | 20 | 20 |
| 40 | 20 | 20 | 21 | 20 |
| 50 | 20 | 20 | 20 | 20 |
| 100 | 21 | 20 | 19 | 20 |

`evaluation_group_id` is class × background × folder magnification. There are 14 condition blocks in development and 14 in holdout; **these are not independent preparation groups**. The original conservative preparation groups and class/date capture groups are preserved in the split table. Preparation groups knowingly cross these partitions; this is the explicit scope change from the original plan. Unknown related views may cross conditions. Hash screening did not establish their absence.

The reviewed pair `img_0181` / `img_0198` stays in black Micro for every split. Four retained images have conflicting magnification folder evidence; all four are in the white holdout. Keep them in the main final score, but exclude them from magnification-specific subgroup scores and report their predictions separately. No label or manifest decision changes.

Black is development because its folder magnification tags have no known conflicts. White is a deliberately different final background, making this a demanding transfer test. This choice uses audit evidence, not model performance.

## Model selection

Run two fixed development folds:

1. Fit black Micro (140 images); validate black 4× (137).
2. Fit black 4× (137 images); validate black Micro (140).

Select on the **unweighted mean of the two seven-class macro F1 scores**. Save both individual scores, their range, and pooled out-of-fold predictions. Pooled scores are descriptive; they do not replace the selection criterion. If candidates are within 0.005 macro F1, prefer fewer fitted parameters, then shorter measured training time. Record those measurements.

These two folds assess transfer between folder conditions. They are too few, and too dependent, to justify a confidence interval for future mixtures. Restrict the number of trials, save all attempted configurations and do not call the winning development score an unbiased final estimate.

Fit scaling, PCA, selectors, learned normalization, class weights and any other learned operation on the training fold only. Use pipelines for classical models. Augmentation is training-only. Split source images before generating tiles/crops; every descendant inherits its source membership, and tile predictions must aggregate to one source-image prediction. Use fixed, recorded CNN epoch schedules; the outer validation fold must not choose a checkpoint and then score that same checkpoint as an untouched validation result. Do not calibrate probabilities with these validation or holdout labels.

After selection, lock preprocessing, features, architecture, hyperparameters, epoch schedule and label mapping; refit on all 277 black images. The final evaluation model has never fitted white images.

## Metrics

Class order is always `[0, 10, 20, 30, 40, 50, 100]`.

- Primary: seven-class macro F1, equal class weights, undefined class precision/F1 set to zero.
- Balanced accuracy: mean true-class recall; return null if a scored subset lacks any true class.
- Per-class precision, recall, F1 and support; confusion matrix with true rows and predicted columns.
- Mean absolute difference between predicted and true volume class, in **volume-percentage points**. For example, predicting 100 for 50 is a 50-point error. This does not validate continuous concentration estimation.
- Binary sensitivity and specificity, derived from the seven-class prediction: 0 is pure; every other class is adulterated. No independently tuned binary threshold. An absent positive/negative denominator returns null, not a misleading zero.

The scorer accepts class labels rather than encoded indices or arbitrary continuous percentages. Probability columns, when later present, must follow the saved class order. No performance results exist yet; unit tests use synthetic labels only.

## Challenge tests and final holdout access

- **Magnification:** the two development transfer folds above. Final white subgroup results use trusted folder tags only; report the four ambiguous cases separately. Folder tags are not verified optical magnifications.
- **Background:** black-to-white transfer, scored once in Phase 6 for the locked winner. Do not run white-to-black experiments during model selection.
- **Capture sessions:** independent session transfer is unsupported because dates and classes are confounded. Report date counts descriptively; do not substitute singleton date scores for seven-class performance.
- **Shortcuts:** metadata-only and background-only controls in Phase 4 must use these same development folds. Date/EXIF, paths, filenames and condition tags are forbidden visual-classifier inputs. Image artifacts can still encode those conditions, so visual controls remain necessary.

All images were visually inspected for data auditing in Phase 1. The holdout is reserved from subsequent model development, **not previously unseen by the auditor**. Starting with this freeze, Phase 3 mask/crop design and QC use black development images only. Do not use white images to select thresholds, preprocessing or models. At final evaluation, apply the frozen processing unchanged. White preprocessing failures are findings to report, not reasons to retune before reporting the final score.

Save final image predictions and condition summaries. Report descriptive block results and fold variation; do not bootstrap images or claim a preparation-level confidence interval. If final results motivate changes, disclose holdout reuse and label subsequent results exploratory. A later model refitted on white images would require separate versioning and would have no untouched test set here.

## Artifacts and reproduction

- [Membership table](../../data/splits/v1/assignments.csv): all 548 included image IDs and partitions/folds, original groups and condition blocks.
- [Machine-readable specification](../../data/splits/v1/protocol.json).
- [Checksums](../../data/splits/v1/freeze.json): bind split membership and specification to the Phase 1 manifest.
- [Implementation](../../src/poppypurev2/evaluation.py) and [tests](../../tests/test_phase2.py).
- Run record: `experiments/runs/phase2-evaluation.json`.

```sh
uv run --locked poppypure freeze-evaluation
uv run --locked python -m unittest discover -s tests -v
```

The first command creates missing artifacts or verifies identical saved artifacts. It rejects modified frozen inputs or incompatible output files. To create a new run record, add `--record` with a new path; existing records cannot be overwritten. Split generation uses metadata only and does not need to decode raw images or download data. Seed 42 remains the future model seed; these deterministic condition partitions use no random sampling.

## Verification and exit gate

Verified 548 unique included IDs with unique file and pixel hashes, exact split coverage, all-class support in both partitions and both development conditions, no ambiguous development magnification tags, and preservation of the reviewed same-scene pair. Tests check hand-calculated scores, absent-denominator behavior, invalid labels, repeatable generation, altered manifest rejection, altered protocol rejection, and related-view separation rejection. Seven unit tests passed; Ruff formatting/lint and compilation passed.

Split definitions, selection metric, challenges, access rules and limitations are frozen before any model fitting. Phase 2 is complete for this limited scope. It does not solve the missing independent-preparation problem.

Method references: scikit-learn's [cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html), [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), and [metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html). Runtime dependency remains the locked scikit-learn 1.8.0; consulting current documentation did not upgrade it.
