# Active evaluation protocol — v4

**Frozen 14 October 2026.** Seven-class semolina adulteration classification **by volume**: 0, 10, 20, 30, 40, 50, 100. Preparation provenance is the user's recorded confirmation; verified same-scene views share a unit.

## Allocation

| Partition | Images | Preparation units | Role |
| --- | ---: | ---: | --- |
| Training | 384 | 383 | Fit models and learned preprocessing; design crops/masks |
| Validation | 82 | 82 | Compare processing and select model/settings |
| Test | 82 | 82 | Evaluate the locked recipe once at the end |
| Total | 548 | 547 | All included close-ups |

Each partition contains **every one of the 28 class × background × folder-magnification combinations**. The confirmed same-scene pair is together in training. No preparation unit crosses partitions. The original exclusions, raw files, volume labels and manifest are unchanged.

| Capture condition | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Black 4× | 97 | 20 | 20 |
| Black Micro | 98 | 21 | 21 |
| White 4× | 95 | 21 | 21 |
| White Micro | 94 | 20 | 20 |

| Volume class | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| 0 | 51 | 11 | 11 |
| 10 | 57 | 12 | 12 |
| 20 | 51 | 11 | 11 |
| 30 | 56 | 12 | 12 |
| 40 | 57 | 12 | 12 |
| 50 | 56 | 12 | 12 |
| 100 | 56 | 12 | 12 |

## Reproducible construction

Seed **42**. Allocate 82 test and 82 validation images across the 28 strata using largest remainders from proportional quotas, with seeded tie breaks. Shuffle whole preparation units within each stratum, choose complete units to meet test quotas, then validation quotas; use the remainder for training. Subset-sum selection handles the two-image same-scene unit without splitting it. The resulting 384/82/82 allocation is approximately 70/15/15.

`preparation_group_id` and `evaluation_group_id` identify the image-specific units, merging confirmed same-scene views. `condition_group_id` preserves class/background/folder blocks; `legacy_conservative_group_id` preserves the old audit grouping as history. Folder magnification tags are used for stratification; retain existing conflict flags and omit flagged images only from summaries requiring trusted magnification.

## Selection and final evaluation

- Primary selection metric: **validation seven-class macro F1**. Supporting metrics remain balanced accuracy, per-class precision/recall/F1, confusion matrix, volume-percentage-point MAE and derived pure/adulterated sensitivity/specificity.
- During model selection, fit on training only. Fit scaling, PCA, feature selection and learned preprocessing on training only. Use validation for comparison; it must not supply fitted model parameters.
- Split before augmentation or tiling. Descendants inherit source membership. Aggregate regions into one prediction per source image.
- EXIF supplies orientation only; dates, camera metadata, filenames and paths are not visual-model inputs. Apply reviewed crop rules consistently to remove markings/borders where feasible.
- Validation can choose a CNN epoch count. Freeze the recipe and epoch count before optionally refitting on **training + validation (466 images)**. Evaluate the resulting locked model once on the **82 test images**, saving predictions and the evaluated model.
- A later deployment version can refit the unchanged recipe on all 548 images; preserve the evaluated model and report separately.
- Report background/magnification subgroups. Condition-transfer experiments are secondary; they must not expose main test images during model selection.

All photographs were audited in Phase 1. From this freeze onward, test images are reserved from processing design and model selection. Phase 3 visual design uses training images; validation comparisons follow in Phase 4. No model had been fitted or scored when this protocol was frozen.

## Files and commands

- [Membership](../../data/splits/v4/assignments.csv)
- [Machine-readable specification](../../data/splits/v4/protocol.json)
- [Freeze checksums](../../data/splits/v4/freeze.json)
- [Allocation implementation](../../src/poppypurev2/splits.py)
- Run record: `experiments/runs/phase2-evaluation-v4.json`

```sh
uv run --locked poppypure freeze-evaluation
```

The active configuration selects v4. Repeating the command verifies byte-identical artifacts; it rejects altered frozen inputs, incompatible seeds, or modified split/specification files. Earlier v1–v3 artifacts and run records are preserved. V4 replaces their capture-condition holdouts as the project's main protocol.
