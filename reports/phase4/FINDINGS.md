# Phase 4 findings

**Completed 14 October 2026.** All results below are on the fixed **82-image validation set**, after fitting on **384 training images**. No final-test images were decoded, used for tuning, or scored.

## Main results

Rendered figures are now indexed in [plots/README.md](plots/README.md): MLP train/validation curves, SVM confusion matrix and per-class scores, all-run comparison, condition scores and fit/validation differences. The initial completion report contained numeric tables and CSV traces; these figures were added after the user requested them.

Completed the planned **21 visual configurations and four controls**. All 25 runs passed; none failed or produced a convergence warning. No extra hyperparameter configurations were added.

| Model/input | Validation macro F1 | Validation accuracy |
| --- | ---: | ---: |
| **RBF SVM, C=10, whole-image colour+texture** | **0.8591** | **70/82 = 85.37%** |
| RBF SVM, C=10, cropped texture | 0.8293 | 68/82 = 82.93% |
| Small MLP, whole-image colour+texture | 0.8196 | 67/82 = 81.71% |
| Metadata-only logistic control | 0.7980 | 70/82 = 85.37% |
| Low-gradient appearance control | 0.6428 | 53/82 = 64.63% |
| Stratified-random control | 0.1546 | See full table |
| Majority-class control | 0.0365 | See full table |

The selected Phase 4 baseline is **`svm_input1_C10`**, using a training-fitted StandardScaler and RBF SVM with `gamma='scale'`. It is also the highest raw validation macro-F1 result, so the 0.005 near-tie rule does not change the winner.

The strongest cropped-input candidate is **`svm_input2_C10`**. Preserve it alongside the whole-image reference for the next comparisons, since the crop excludes the observed lower-edge camera marks. It uses 100 texture features, while the whole-image model uses 196 colour+texture features: this is not an isolated crop-only comparison.

Full results and artifacts: [baseline comparison](BASELINES.md), [CSV](comparison.csv), [summary](summary.json), and `experiments/runs/phase4-v1/`. Every run has its own model bundle, settings/provenance, training/validation metrics, subgroup summaries and validation predictions.

## What screening selected

The nine logistic-regression runs screened three image representations and three feature families. Whole-image combined features scored 0.8421 macro F1. The second-stage inputs were fixed to:

1. Whole image, combined colour+texture (196 dimensions).
2. Crop, texture (100 dimensions).

Crop combined features had a slightly higher screening score (0.7588) than crop texture (0.7547), but the difference was within 0.005, so the smaller texture input was selected by the saved tie rule. `stage2_inputs.json` was written before SVM/forest/MLP runs. All second-stage families used these same inputs.

## The metadata control is informative

The metadata-only model used capture date, dimensions, background and source magnification tags. It did **not** receive image pixels, volume labels as features, filenames, class-bearing paths or preparation identifiers.

It matched the winning visual model's **85.37% accuracy**, but had lower macro F1 (**0.7980**) because it missed **all 12 examples of the 30% class**. Its predictions follow the dataset's class-associated capture conditions. This demonstrates why our main score gives every class equal weight and why capture metadata stays outside the visual classifier.

The visual SVM classifies 8/12 of the 30% images correctly, and is not fed capture metadata. These results do not identify which particular image cues the SVM uses; they establish that the metadata itself is strongly predictive. Keep the cropped-input comparison and the planned CNN crop/detail experiments.

## Low-gradient control review

The low-gradient rule selects smooth pixels, including background **and grain interiors**. Inspected all four training review sheets (28 examples); saved the conclusion in [CONTROL_REVIEW.json](CONTROL_REVIEW.json). Its 0.6428 macro F1 is an appearance-control result, not a measurement of background-only classification. No valid pure-background mask was asserted.

## Errors of the selected SVM

The model made **12 errors**:

| True volume class | Correct / support | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0 | 10/11 | 0.9091 | 0.9524 |
| 10 | 12/12 | 1.0000 | 0.9231 |
| 20 | 10/11 | 0.9091 | 0.9524 |
| 30 | 8/12 | 0.6667 | 0.6400 |
| 40 | 9/12 | 0.7500 | 0.7200 |
| 50 | 10/12 | 0.8333 | 0.8696 |
| 100 | 11/12 | 0.9167 | 0.9565 |

Six errors are swaps between **30% and 40%** (three in each direction). The largest mistakes are one pure-poppy image classified as 30% and one pure-semolina image classified as 50%. Inspected the saved [validation error gallery](validation_error_gallery.jpg); variable grain scale, focus, packing and illumination remain visible. This inspection does not by itself establish a cause for each prediction.

**Confusion matrix — rows true, columns predicted, by volume:**

| True / predicted | 0 | 10 | 20 | 30 | 40 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 0 | 0 | 1 | 0 | 0 | 0 |
| 10 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| 20 | 0 | 1 | 10 | 0 | 0 | 0 | 0 |
| 30 | 0 | 1 | 0 | 8 | 3 | 0 | 0 |
| 40 | 0 | 0 | 0 | 3 | 9 | 0 | 0 |
| 50 | 0 | 0 | 0 | 1 | 1 | 10 | 0 |
| 100 | 0 | 0 | 0 | 0 | 0 | 1 | 11 |

Balanced accuracy is **0.8550**. Mean absolute volume-class error is **2.439 percentage points**; this scores differences between discrete class labels. Derived binary detection has **71/71 adulterated images detected**, **10/11 pure images correctly rejected** (sensitivity 1.0000, specificity 0.9091).

The whole-image SVM's training macro F1 is **0.9924**, versus validation **0.8591**. The training score is recorded to show fit/generalization differences; it is not the headline result.

## Capture-condition results

| Condition | Selected SVM validation macro F1 |
| --- | ---: |
| Black 4× | 0.7469 |
| Black Micro | 0.9048 |
| White 4× | 0.9071 |
| White Micro | 0.8544 |

These summaries exclude retained magnification-conflict flags; the main metric includes every validation image. Performance is weakest on black 4×. Condition scores use small subsets of the same validation set, so keep the full-set comparison as the primary ranking.

## Training and checkpoint details

- CPU only; one numerical-library thread and one forest worker. Aggregate measured fitting/checkpoint time across 25 runs was approximately **2.16 seconds**, excluding image-feature extraction and report generation. GPU acceleration was unnecessary for these matrices.
- Logistic regression used a 3000-iteration cap; all runs converged.
- Each forest used 300 trees and minimum leaf size two, at the two planned depth limits.
- MLP checkpoints were chosen on explicit validation macro F1, not an internal random split. Best epochs were **57** (whole combined) and **41** (crop texture). The saved models reproduce those checkpoints; curves are stored with the runs.
- No PCA, feature selection or probability calibration was fitted. SVM output scores are decision margins, explicitly labelled rather than treated as probabilities.

## Verification and next work

All **18 unit tests** passed, including train-only scaling, fixed family columns, checkpoint restoration, metadata feature exclusion and near-tie/non-convergence rules. Reloaded saved pipelines and reproduced predictions/metrics; checked artifact checksums, exact validation IDs/labels, feature hashes and unchanged v4 membership. See `experiments/runs/phase4-v1/ACCEPTANCE.json` for the final artifact check.

**Next: Phase 5 — frozen ResNet18 linear/MLP heads, followed by the planned ResNet50 comparison and limited fine-tuning.** Use the saved crop/detail processing, the same 384/82 development allocation and the bounded experiment plan. The 82-image final test remains reserved.

On a clean checkout with the existing training features, create the development inputs at the configured paths, then train into a new experiment directory:

```sh
uv run --locked poppypure extract-features --partition validation --output data/features/validation-v1
uv run --locked poppypure prepare-controls --output data/features/controls-v1
uv run --locked poppypure train-baselines --output experiments/runs/phase4-NEW
```

Skip the first two commands when their saved outputs already exist. Existing feature directories, models and run records refuse overwrite. The preserved original inputs can reproduce training directly with a new experiment output directory; generated summary reports are development reports, not immutable run records. A changed feature/control recipe must be versioned and recorded before additional experiments.
