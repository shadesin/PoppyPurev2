# Phase 4 figures and how to read them

**All performance plots use training/validation data. The 82-image final test remains reserved.** PNG and PDF versions are provided for each figure.

## Loss, F1 and accuracy curves

- [Whole-image MLP](mlp_input1_learning_curves.png), [PDF](mlp_input1_learning_curves.pdf).
- [Cropped-texture MLP](mlp_input2_learning_curves.png), [PDF](mlp_input2_learning_curves.pdf).

Each has training and validation cross-entropy, macro F1 and accuracy versus epoch. Dashed lines mark the selected macro-F1 checkpoints, epochs 57 and 41 respectively. Cross-entropy is evaluated on the complete corresponding partition at the end of each epoch, with the same definition for both partitions; it excludes the L2 penalty. The diagnostic CSV also retains scikit-learn's original optimizer loss as `loss`, which includes its training objective and is a different measurement.

The original runs saved training optimizer loss and train/validation F1 in CSV, but no rendered figures or validation cross-entropy. To supply both loss curves, the same two MLP recipes were replayed with additional logging. Every original epoch's saved loss/F1 values matched, and the selected checkpoints reproduced the original predictions **and probabilities exactly**. These are diagnostic repetitions of existing configurations, not additional hyperparameter trials. Original models, curves and records remain unchanged; replay records/traces are alongside the figures.

The whole-image MLP's training accuracy approaches 100%, while validation accuracy plateaus around 81.7%. The curves make that difference visible. Checkpoint selection follows validation macro F1 even when cross-entropy continues to improve.

An epoch-loss curve was recorded for the MLP training loop. The SVM/forest training implementation does not expose comparable epoch traces; their main evaluation figures are prediction-based.

## Prediction and comparison figures

| Figure | Purpose |
| --- | --- |
| [SVM confusion matrix](selected_svm_confusion_matrix.png) | Counts and row-normalized errors for seven by-volume classes |
| [SVM per-class precision/recall/F1](selected_svm_per_class.png) | Show which classes need improvement |
| [All 25 experiment configurations](all_run_comparison.png) | Compare validation macro F1; controls are gray |
| [Training versus validation](training_validation_gap.png) | Compare fit and held-out development performance |
| [SVM capture conditions](selected_svm_conditions.png) | Compare black/white and source magnification groups |

The existing [validation error gallery](../validation_error_gallery.jpg) shows all 12 mistakes of the selected SVM. Scalar metrics and full predictions remain in the original experiment directory.

## Terminology

- **384 training images:** used to fit models and learned transforms.
- **82 validation images:** used for model/settings/checkpoint selection; the reported **85.37% = 70/82 correct** comes from this set.
- **82 test images:** reserved for the final locked pipeline in Phase 6.
- **18 software tests:** automated code checks, for example scaler isolation, label handling and model checkpoint restoration.
- **25 experiment configurations:** nine logistic-regression runs, six SVM runs, four forest runs, two MLP runs and four diagnostic controls. These are repeated model/input/settings combinations, not 25 architecture families. Two later diagnostic MLP replays add loss logging without changing that comparison budget.

## Interpreting macro F1

F1 combines a class's precision and recall. Macro F1 averages the seven class F1 values equally. It ranges from zero to one, with one representing perfect predictions. See the official [F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

There is no universal threshold defining a good model. Within this project, **0.8591 is a useful baseline**, well above the recorded random control (0.1546), with clear class-level room for improvement: 30% F1 is 0.64 and 40% F1 is 0.72. A high average should be read alongside those individual results. Macro F1 and accuracy are separate measures, so 0.8591 does not mean 85.91% accuracy.

## Reproduce

```sh
uv run --locked poppypure plot-phase4 --output reports/phase4/plots-NEW
```

The command refuses an existing output directory, verifies original artifacts and replay identity, and uses saved features rather than decoding new photographs. Matplotlib 3.10.8 and its dependencies are locked. Cache paths default to project-local directories. `summary.json` records figure checksums and diagnostic provenance.
