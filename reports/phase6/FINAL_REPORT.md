# PoppyPure v2 — final evaluation

**Phase 6 complete.** The selected recipe was refitted on 466 images for 28 fixed epochs, sealed, and evaluated on all 82 reserved test images.

## Final test results

| Model | Fitting images | Test images | Accuracy | Macro F1 | Balanced accuracy | Volume-class MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 + MLP (locked primary) | 466 | 82 | 91.46% | 0.9138 | 0.9145 | 1.341 points |
| SVM (fixed reference) | 466 | 82 | 90.24% | 0.8976 | 0.9037 | 2.927 points |

The primary model correctly classified **75/82** images, with **7 errors**. Its version is **poppypure-v1**. The prediction classes are semolina adulteration percentages **by volume**: 0, 10, 20, 30, 40, 50 and 100.

The SVM is the previously selected whole-image colour/texture RBF recipe, refitted on the same 466 images. Both model recipes were fixed before test access. The test comparison does not select a replacement primary model.

## Refit and development distinction

Development selected the 384-image ResNet18 detail + MLP checkpoint at epoch 28: validation macro F1 0.9249 / accuracy 92.68% (76/82). The final model is a fresh fit on 384 training + 82 validation images. The scaler was fitted on these 466 images; the MLP trained for exactly 28 epochs from seed 42. The ImageNet encoder and BatchNorm remained frozen.

The refit had **no separate validation set and no early stopping**. Its learning curves contain training metrics only. The final test was scored after the checkpoint and reference were sealed. Development and final-test results refer to different image partitions and fitted checkpoints.

## Per-class test results

| Volume class | Precision | Recall | F1 | SVM F1 | Images |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.9167 | 1.0000 | 0.9565 | 0.9565 | 11 |
| 10% | 0.9167 | 0.9167 | 0.9167 | 1.0000 | 12 |
| 20% | 0.9000 | 0.8182 | 0.8571 | 0.8333 | 11 |
| 30% | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 12 |
| 40% | 0.9167 | 0.9167 | 0.9167 | 0.7368 | 12 |
| 50% | 1.0000 | 0.9167 | 0.9565 | 1.0000 | 12 |
| 100% | 0.9231 | 1.0000 | 0.9600 | 0.9231 | 12 |

## Pure versus adulterated detection

Derived from the seven-class output: sensitivity **98.59%**, specificity **100.00%**; TP=70, TN=11, FP=0, FN=1.

## Test errors

See [all error images](test_error_gallery.jpg) and [their original IDs and probabilities](test_errors.json).

- `img_0080`: 10% → 0% (white/micro).
- `img_0097`: 20% → 10% (black/4x).
- `img_0144`: 20% → 30% (white/micro).
- `img_0170`: 30% → 20% (black/4x).
- `img_0221`: 30% → 40% (white/micro).
- `img_0284`: 40% → 30% (white/4x).
- `img_0337`: 50% → 100% (black/4x).

## Uncertainty and capture conditions

The 95% intervals use 2,000 percentile-bootstrap replicates, resampling preparation units within each true volume class (seed 1042). The test contains 82 preparation units across 82 images. Both models use the same sampled units for paired comparisons.

| Test metric | Lower 95% bound | Upper 95% bound |
| --- | ---: | ---: |
| accuracy_difference | -0.0610 | 0.0854 |
| macro_f1_difference | -0.0623 | 0.0946 |
| primary_accuracy | 0.8537 | 0.9634 |
| primary_macro_f1 | 0.8511 | 0.9641 |
| reference_accuracy | 0.8537 | 0.9512 |
| reference_macro_f1 | 0.8371 | 0.9511 |

Condition summaries retain all images in the overall metrics. Flagged magnification conflicts are excluded only from summaries requiring trusted magnification.

| Condition | Images | Primary macro F1 | SVM macro F1 |
| --- | ---: | ---: | ---: |
| black_4x | 20 | 0.8306 | 0.8320 |
| black_micro | 21 | 1.0000 | 0.9510 |
| white_4x | 20 | 0.9510 | 0.8031 |
| white_micro | 20 | 0.8463 | 1.0000 |

Capture-date summaries are descriptive; dates are associated with different class sets.

| Capture date | Images | Correct | Accuracy | Classes present |
| --- | ---: | ---: | ---: | --- |
| 2021-11-27 | 6 | 6 | 100.00% | [0] |
| 2021-11-30 | 5 | 5 | 100.00% | [0] |
| 2021-12-06 | 6 | 6 | 100.00% | [100] |
| 2021-12-08 | 6 | 6 | 100.00% | [100] |
| 2021-12-09 | 12 | 11 | 91.67% | [10] |
| 2021-12-10 | 17 | 14 | 82.35% | [20, 30] |
| 2021-12-11 | 18 | 16 | 88.89% | [30, 40] |
| 2021-12-13 | 12 | 11 | 91.67% | [50] |

## Predictor and reproducibility

The packaged predictor applies the saved EXIF orientation/RGB conversion, fractional crop, five native 512-pixel regions, aspect-preserving 224-pixel letterboxing, ImageNet normalization, mean-pooled ResNet18 embedding, saved scaler and MLP. It receives no EXIF or folder metadata as classifier features. Background masks remain disabled.

```sh
uv sync --locked
uv run --locked poppypure predict sample.jpg --device cpu
```

Use `--model-dir /path/to/poppypure-v1` for a copied model bundle. Prediction requires the installed package, model bundle and input image; it does not require Backup Plus, training caches or pretrained-weight downloads.

To reproduce Phase 6 from the verified earlier-phase artifacts in a fresh artifact workspace:

```sh
uv run --locked poppypure refit-final
uv run --locked poppypure evaluate-final --device auto
uv run --locked poppypure report-final
```

Refit refuses existing final-model/run directories; evaluation refuses repeated test access. Reporting reads saved predictions and does not rescore models. Exact source/config/environment snapshots and checksums are stored with the run.

Evaluated checkpoint SHA-256: `520d47d80da1c3d8f5c2d689974a6ecbceee792dba41eae4c162fd072b4d69d1`.

## Supporting artifacts

- [Figures](plots/README.md).
- [Model card](MODEL_CARD.md).
- [Dataset card](DATASET_CARD.md).
- [Saved test metrics](metrics.json).
- Model, scaler and preprocessing: `models/poppypure-v1/`.
- Refit trace, membership, pretest seal, test predictions and records: `experiments/runs/phase6-v1/`.
