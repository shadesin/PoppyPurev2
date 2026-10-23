# Phase 4 — baseline validation results

384 training images; 82 validation images. Seven semolina adulteration classes **by volume**. Final test images were not used.

## Comparison

| Run | Role | Validation macro F1 | Accuracy | Training macro F1 |
| --- | --- | ---: | ---: | ---: |
| svm_input1_C10 | visual | 0.8591 | 0.8537 | 0.9924 |
| screen_whole_combined | visual | 0.8421 | 0.8415 | 0.9819 |
| svm_input2_C10 | visual | 0.8293 | 0.8293 | 0.9152 |
| mlp_input1 | visual | 0.8196 | 0.8171 | 0.9922 |
| control_metadata | control | 0.7980 | 0.8537 | 0.7982 |
| forest_input1_depthNone | visual | 0.7689 | 0.7683 | 1.0000 |
| mlp_input2 | visual | 0.7678 | 0.7683 | 0.8449 |
| screen_crop_combined | visual | 0.7588 | 0.7561 | 0.9921 |
| screen_crop_texture | visual | 0.7547 | 0.7561 | 0.8162 |
| svm_input1_C1 | visual | 0.7542 | 0.7561 | 0.8570 |
| screen_whole_texture | visual | 0.7528 | 0.7561 | 0.8374 |
| screen_crop_detail_combined | visual | 0.7432 | 0.7439 | 1.0000 |
| forest_input1_depth8 | visual | 0.7316 | 0.7317 | 1.0000 |
| screen_whole_colour | visual | 0.7207 | 0.7195 | 0.8967 |
| screen_crop_detail_colour | visual | 0.6809 | 0.6829 | 0.9923 |
| screen_crop_colour | visual | 0.6558 | 0.6585 | 0.9000 |
| forest_input2_depthNone | visual | 0.6493 | 0.6463 | 0.9949 |
| control_low_gradient | control | 0.6428 | 0.6463 | 0.9158 |
| screen_crop_detail_texture | visual | 0.6115 | 0.6098 | 0.8596 |
| forest_input2_depth8 | visual | 0.5993 | 0.5976 | 0.9644 |
| svm_input2_C1 | visual | 0.5825 | 0.5854 | 0.6740 |
| svm_input1_C0.1 | visual | 0.3957 | 0.4634 | 0.4570 |
| svm_input2_C0.1 | visual | 0.2620 | 0.3293 | 0.3139 |
| control_random | control | 0.1546 | 0.1585 | 0.1588 |
| control_majority | control | 0.0365 | 0.1463 | 0.0369 |

## Selected Phase 4 baseline

**svm_input1_C10**: macro F1 **0.8591**, accuracy **0.8537**. Selection applies the frozen 0.005 near-tie rule (fewer fitted parameters, then measured training time).

Controls are diagnostic and excluded from winner selection. The low-gradient control is not verified background-only; see its training overlays and review decisions.

Saved experiment directory: `experiments/runs/phase4-v1`. Each run has a model, settings/provenance, predictions and metrics; MLP runs also save epoch curves. See `comparison.csv`, `summary.json` and `validation_errors.csv` here.

These are validation results used for model selection. Phase 5 compares frozen ResNet heads and limited fine-tuning before final recipe selection. No test accuracy has been computed.
