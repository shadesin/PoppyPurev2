# Phase 5 findings

**Status: complete.** The complete recipe is selected for Phase 6; no test scoring has occurred.

## Selected classifier

**resnet18 + mlp**, `crop_detail`, stage **frozen**, selected epoch **28**.

Validation: **0.9249 macro F1**, **92.68% accuracy (76/82)**, balanced accuracy **0.9253**, volume-class MAE **1.098 percentage points**.

Classical reference: 0.8591 macro F1 / 85.37% accuracy. The selected model changes these by **+0.0658 macro F1 / +7.32 accuracy percentage points**.

The task remains seven-class semolina adulteration **by volume**, with labels 0, 10, 20, 30, 40, 50 and 100.

## All eight deep configurations

| Configuration | Validation macro F1 | Accuracy | Training macro F1 | Best epoch / executed |
| --- | ---: | ---: | ---: | ---: |
| `resnet18_crop_linear_frozen` | 0.8599 | 85.37% | 0.9671 | 15 / 23 |
| `resnet18_crop_mlp_frozen` | 0.8818 | 87.80% | 0.9975 | 11 / 19 |
| `resnet18_crop_detail_linear_frozen` | 0.9099 | 91.46% | 0.9666 | 20 / 28 |
| `resnet18_crop_detail_mlp_frozen` | 0.9249 | 92.68% | 1.0000 | 28 / 36 |
| `resnet50_crop_detail_linear_frozen` | 0.8608 | 86.59% | 0.9923 | 8 / 16 |
| `resnet50_crop_detail_mlp_frozen` | 0.8988 | 90.24% | 0.9949 | 8 / 16 |
| `resnet18_crop_detail_mlp_ft_1e-05` | 0.8921 | 89.02% | 0.9631 | 2 / 7 |
| `resnet18_crop_detail_mlp_ft_3e-05` | 0.8872 | 89.02% | 0.9762 | 10 / 15 |

Selection compares all **21 classical visual configurations and eight deep configurations**, using validation macro F1 and the predefined 0.005/size/time tie rule. Four Phase 4 diagnostic controls remain diagnostic. No optional fusion or extra confirmation seeds were run.

## Per-class validation results

| Volume class | Precision | Recall | F1 | Classical F1 | Images |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.9167 | 1.0000 | 0.9565 | 0.9524 | 11 |
| 10% | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 12 |
| 20% | 1.0000 | 0.7273 | 0.8421 | 0.9524 | 11 |
| 30% | 0.7857 | 0.9167 | 0.8462 | 0.6400 | 12 |
| 40% | 0.9091 | 0.8333 | 0.8696 | 0.7200 | 12 |
| 50% | 0.9231 | 1.0000 | 0.9600 | 0.8696 | 12 |
| 100% | 1.0000 | 1.0000 | 1.0000 | 0.9565 | 12 |

Class F1 is lower than the classical reference for 20%. The higher overall macro F1 does not imply improvement in every class.

## Remaining validation errors

There are **6 errors**. All appear in [the error gallery](validation_error_gallery.jpg), with original IDs and true/predicted by-volume classes in [validation_errors.json](validation_errors.json).

- `img_0092`: 20% → 0%.
- `img_0136`: 20% → 30%.
- `img_0139`: 20% → 40%.
- `img_0191`: 30% → 50%.
- `img_0269`: 40% → 30%.
- `img_0272`: 40% → 30%.

Derived pure-versus-adulterated detection: sensitivity **98.59%**, specificity **100.00%**; TP=70, TN=11, FP=0, FN=1. Seven-class macro F1 remains the selection metric.

## Processing and verification

The CNN receives RGB crop/detail pixels. EXIF and folder metadata are excluded; masks and background removal are disabled. Frozen heads use training-only embedding scaling; fine-tuning preserves it. See [methods](METHODS.md) for exact weights, normalization, augmentation, schedules and devices.

All eight checkpoints passed artifact, membership, scaler, prediction-reproduction and earliest-best-epoch acceptance. Fine-tuned checkpoints also preserve early blocks, BatchNorm running statistics and the scaler. Raw/manifest/split membership stays fixed. All results above use **384 fitting / 82 validation** images; the **82 test images are reserved**.

Saved softmax probabilities are not a calibration measurement. Condition summaries exclude flagged magnification conflicts only where trusted magnification is needed; overall metrics retain these images.

## Artifacts and next step

- [Complete selected recipe and fixed Phase 6 refit schedule](MODEL_SELECTION.json).
- [Learning curves and result charts](plots/README.md).
- All models/predictions/metrics: `experiments/runs/phase5-v1/`.
- Embeddings/provenance: `data/embeddings/phase5-v1/`.
- Phase 6: refit the selected recipe on 466 train+validation images for the fixed recorded counts, evaluate the reserved 82-image test once, and package identical CLI inference.
