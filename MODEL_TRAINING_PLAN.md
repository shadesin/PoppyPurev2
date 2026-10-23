# Model training plan — Phases 4–6

**Created:** 14 October 2026. **Status:** Phases 4–6 completed. See [baseline findings](reports/phase4/FINDINGS.md), [eight deep comparisons](reports/phase5/FINDINGS.md) and [final evaluation](reports/phase6/FINAL_REPORT.md). Selected frozen ResNet18 detail + MLP: **0.9249 macro F1 / 92.68% development validation accuracy**; final 466-image refit: **0.9138 macro F1 / 91.46% test accuracy (75/82)**. The settings below preserve the declared experiment sequence.

## Goal and fixed data allocation

Classify poppy seed/semolina images into **seven semolina percentages by volume**: `[0, 10, 20, 30, 40, 50, 100]`. Use active protocol v4: **384 training / 82 validation / 82 test images**. All classes and all four capture conditions appear in each partition. User-confirmed individual preparation units and the verified same-scene pair remain as recorded in split membership.

Train on training data, choose models on validation, and evaluate the selected recipe once on test. The existing split, raw images and Phase 3 descriptors are unchanged by this plan.

## What the original project actually used

Read-only inspection of `PoppyPure/source code/resnet.ipynb` found:

- Keras **ResNet50**, ImageNet weights, `include_top=False`, global average pooling.
- ResNet features fed into an MLP: **512 → dropout 0.3 → 256 → dropout 0.3 → 7 softmax outputs**.
- Each original and an augmented version were added before the train/test split (cells 5 and 9). Related versions could cross that split.
- The test set was also supplied as validation for early stopping, then evaluated again (cells 11–13).
- Inputs were resized directly to 224×224. `cv2.imread` provided BGR pixels to the Keras ResNet preprocessing function without a visible BGR-to-RGB conversion in those cells.

The separate `MLP_classifier.ipynb` trained a scikit-learn MLP with hidden layers `(256,128)` on handcrafted features. That is a different pipeline from ResNet50 + MLP. `next.ipynb` also contains PCA fitted before splitting; v2 uses train-only fitted transforms.

We will compare the **ResNet + MLP idea** with smaller alternatives using the rebuilt workflow. The old notebook's scores do not carry over to these experiments, and the exact old head is not assumed optimal. Source notebooks remain untouched.

## Common experiment rules

1. Primary comparison: **seven-class validation macro F1**, seed **42**. Keep balanced accuracy, per-class precision/recall/F1, confusion matrices, volume-class error in percentage points, binary sensitivity/specificity and condition summaries.
2. Each run saves its full configuration, code/environment identity, input/split checksums, fitted pipeline or checkpoint, training time, number of fitted parameters, and per-image validation predictions.
3. Scalers, PCA, feature selection and learned transforms fit on training only. Start without PCA or feature selection. Class labels remain in a separate target array.
4. No test images or labels are used for feature design, augmentation choices, epoch selection, model comparison or error galleries. Test processing occurs in Phase 6.
5. Near ties within **0.005 macro F1** follow the frozen protocol: fewer fitted parameters, then shorter measured training time. Do not choose solely by accuracy or the best-looking condition.
6. Save every planned run, including non-convergence and failed runs. Do not silently keep adding hyperparameter trials. Any added experiment requires a recorded reason before execution.
7. Repeated seeds **43 and 44** may confirm the stability of the two strongest stochastic candidates. Report them all; they are descriptive checks, not extra opportunities to replace seed 42 with a favourable seed.

## Phase 4 — classical baselines and an MLP on handcrafted features

### 4A. Prepare inputs

- Verify `data/features/v1/train.npz`, its schema/config hashes and v4 membership.
- Extend feature extraction to an explicit validation partition, applying the exact saved definitions to 82 validation images. Keep test extraction outside this development command.
- Preserve the three Phase 3 representations: **whole (196)**, **crop (196)**, **pooled crop detail (392)**. Define feature-family column slices from schema names: **colour**, **texture (LBP+GLCM)**, and **combined**. Detail means and standard deviations must both follow the selected family.
- Background masks remain disabled after Phase 3 visual rejection. Whole-image inputs are a reference that may include markings; crop/detail are the material-focused candidates.

### 4B. Establish controls

| Control | Definition |
| --- | --- |
| Majority | Always return the most frequent training class; deterministic tie break by class order |
| Stratified random | Sample the training label distribution using seed 42; no validation-derived probabilities |
| Metadata only | Regularized classifier using capture date, dimensions, background and source magnification tags; exclude labels, class-bearing paths and preparation IDs |
| Background focused | Use a label-independent low-gradient pixel selection rule fitted/designed on training only; save selected-pixel overlays. Call it background-focused unless reviewed annotations establish a background-only region |

Controls use the same train/validation membership. Date/class associations are examined through the metadata control. The controls are diagnostic; they do not become the visual classifier's inputs. If a credible background region cannot be isolated, record that outcome and use a clearly labelled border/low-gradient control without claiming pure background analysis.

### 4C. Bounded model comparison

| Stage | Experiments | Purpose |
| --- | ---: | --- |
| Screening | **9** | StandardScaler + logistic regression, C=1, on 3 representations ×3 feature families |
| SVM | **6** | On the top two distinct screening inputs: StandardScaler + RBF SVM, C ∈ {0.1,1,10}, gamma=`scale` |
| Random forest | **4** | Same two inputs, 300 trees, `min_samples_leaf=2`, maximum depth ∈ {8,None}; no scaler |
| Handcrafted MLP | **2** | Same two inputs: StandardScaler + a single 64-unit ReLU layer, L2 alpha=0.01, Adam lr=0.001, maximum 200 epochs |

**Budget: 21 visual configurations plus four control recipes.** Use one input ranking and the same two inputs for all second-stage classifiers. Prefer crop/detail in input ties under the common parameter/time tie rules. Both selected inputs must be recorded before stage 2 runs.

For the handcrafted MLP, use the explicit 82-image validation set for checkpoint/epoch selection via a controlled training loop; disable scikit-learn's automatic internal early-stopping split. Do not change solver or add hidden layers during the initial comparison. Log convergence and training/validation curves. Logistic regression gets a recorded sufficiently high iteration cap; failure to converge is an implementation finding to resolve, not a usable final model.

### 4D. Review and deliver

Compare model and input choices with trivial and shortcut controls. Save a validation error gallery, including neighbouring-volume errors, and summaries by background and magnification. Keep existing magnification-conflict flags in the main score; exclude them only from summaries requiring trusted magnification.

**Deliverables:** baseline table, saved pipelines, per-image validation predictions, control review, error gallery and Phase 4 report. The strongest classical baseline remains a candidate for the final model; a CNN does not win by default.

**Exit gate:** reproducible baseline evidence and a checked training/evaluation pipeline. Proceed to frozen ResNet comparisons even if classical results are modest. If a control exposes a shortcut, investigate it within development; do not alter the test split to improve results.

## Phase 5 — ResNet, classification heads and limited fine-tuning

### 5A. Prepare the pretrained encoder pipeline

- Use **PyTorch/Torchvision**, with compatible versions pinned when implementation begins. Start with **ResNet18**, explicit `IMAGENET1K_V1` weights; cache weights inside this project and record their checksum. Do not use an unversioned `DEFAULT` alias.
- Apply the saved EXIF/RGB and camera-mark crop processing. Compare two inputs: **crop overview** and **five detail regions pooled at image level**.
- Use aspect-preserving resize and letterboxing to 224×224, followed by the selected weights' documented channel normalization. Fill padding with the normalization mean, producing neutral normalized padding. This is a documented adaptation from Torchvision's default resize/centre-crop transform; save training previews and validate the implementation before extraction.
- Detail regions use the existing native 512-pixel coordinates before resizing. Average the five ResNet embeddings into one image embedding. Keep overview and pooled-detail embedding dimensions equal for head comparisons.
- Remove the original ImageNet classification layer. ResNet18 supplies a **512-dimensional embedding**; ResNet50 supplies **2048 dimensions**. Neither ImageNet classes nor their probabilities are our adulteration labels.
- Frozen backbones run in evaluation mode with gradients disabled, including fixed BatchNorm statistics. Cache deterministic embeddings separately for training and validation. Initial frozen comparisons use no augmentation; source-image grouping remains unchanged.
- Standardize embedding dimensions using training statistics for the frozen-head comparisons and save that transform. Linear and MLP heads receive the same input embeddings/scaler for a fair comparison.

### 5B. Compare frozen ResNet18 heads

```mermaid
flowchart LR
    A[RGB image] --> B[Checked crop or detail regions]
    B --> C[Pretrained ResNet]
    C --> D[One pooled image embedding]
    D --> E[Linear head or small MLP]
    E --> F[Seven by-volume classes]
```

| Candidate | Head | What we learn |
| --- | --- | --- |
| ResNet18 + linear | `512 → 7` logits | How useful frozen pretrained features are with a simple classifier |
| ResNet18 + MLP | `512 → 128 → ReLU → dropout(0.3) → 7` logits | Whether a nonlinear head improves the same representation |

Run each head on overview and pooled detail: **four initial configurations**. Use cross-entropy on logits; apply softmax only for reported prediction probabilities. Argmax returns one of the seven mapped volume classes.

Initial frozen-head schedule: AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, maximum 50 epochs, validation macro-F1 patience 8. Keep the best validation macro-F1 checkpoint; break epoch ties by earlier epoch. Record curves and chosen epoch. These are planned starting settings, not tuned results. Any change gets a new recorded configuration.

### 5C. Compare the original architecture family

Use **frozen ResNet50**, explicit `IMAGENET1K_V2` weights, on the winning ResNet18 input representation. Compare **two heads**: `2048 → 7`, and `2048 → 128 → ReLU → dropout(0.3) → 7`. Use the same frozen-head schedule and train-only embedding scaling.

This directly evaluates **ResNet50 + MLP** while checking whether the MLP actually helps. Start with the small head; do not automatically recreate the original 512/256 head on this small training set. The backbone/weights and head sizes are deliberate experimental choices, not an exact reproduction of the Keras notebook.

### 5D. Limited fine-tuning

Take the best frozen ResNet candidate, initialize from its saved training checkpoint and unfreeze **only `layer4` plus the head**. Keep earlier layers and all BatchNorm running statistics frozen. Retain the same input representation and stored training-only embedding scaler after pooling; use image-level loss for the pooled-detail model.

Compare at most **two** backbone learning rates, 0.00001 and 0.00003, with head learning rate 0.0001, weight decay 0.0001, batch size up to 8 source images, maximum 20 epochs and patience 5 on validation macro F1. Log the actual batch size/device. Use accumulation if detail-region memory requires it; do not count regions as separate labelled images.

Training-only augmentation: horizontal/vertical flips, rotations in multiples of 90°, mild brightness/contrast changes up to ±10%. Apply image-level geometric transforms before detail extraction and use a consistent photometric change across its regions. Keep validation deterministic. Avoid heavy random crops, strong hue shifts or augmentations that remove most of the mixture.

If fine-tuning does not improve validation macro F1 under the existing tie rule, retain the frozen candidate. No randomly initialized ResNet, unrestricted architecture search or larger backbones are part of this initial plan.

### 5E. Optional feature fusion

Only if recorded validation errors show complementary behaviour, compare **one** fusion recipe using the best frozen ResNet embedding plus the selected handcrafted colour/texture vector. Fit each block's scaler on training only, concatenate, then use a regularized linear classifier. This is a new extension, not something found in the original ResNet notebook.

Record the error-based reason before running it. Do not add segmentation, attention or an ensemble without a specific unresolved finding and a revised experiment entry.

### 5F. Select the complete recipe

Compare the strongest classical pipeline, frozen ResNet candidates and fine-tuned candidates on the same 82-image validation set. Use macro F1 and the saved tie rule; inspect all class/condition summaries. Repeat the top stochastic candidates with the declared confirmation seeds if needed, saving every repeat. Keep the original seed-42 comparison as the primary ranking.

**Initial deep budget:** four ResNet18 + two ResNet50 + at most two fine-tuning configurations; optional one fusion configuration. Confirmation seeds are recorded separately. No testing is involved in selecting the winner.

**Deliverables:** model/input/head comparison, cached embedding provenance, learning curves, validation predictions/errors, selected recipe and inference preprocessing specification. The original MLP/ResNet idea receives an explicit comparison even if a simpler model wins.

## Phase 6 — final fit, test and inference package

1. Lock model family, representation, hyperparameters, augmentation, label mapping and epoch/schedule. Save a selection record identifying all validation comparisons.
2. Optionally refit that unchanged recipe on **466 training+validation images**. Reinitialize from the same pretrained weights/seed where applicable; fit transforms on these 466 only. Use the already chosen epoch count/schedule; no test-based stopping. For fine-tuning, reproduce the recorded warm-up then fine-tuning stages with fixed counts.
3. Evaluate the locked model **once on the 82 test images**. Preserve its checkpoint, per-image predictions, full metric report, confusion matrix and condition summaries. The seven-class result is primary; binary detection is derived.
4. Package CLI inference using identical orientation/crop/normalization/region pooling, saved transforms and class order. Save environment, model card, configuration hashes and commands.
5. Optionally create a separately versioned all-548-image deployment refit after preserving the evaluated version. Keep its fitting history distinct from the tested checkpoint.

## Hardware and implementation order

Phase 4 uses the Apple CPU. Before Phase 5, check installed Python/PyTorch compatibility, architecture, memory and whether `torch.backends.mps.is_available()` is true. Use MPS if supported and verified; otherwise use CPU. Record device/fallbacks and benchmark one representative batch. No cloud GPU rental is planned.

PyTorch 2.9.1 and Torchvision 0.24.1 are installed, and both explicit weight files have been downloaded and verified. Apple M4 / 16 GiB memory / working MPS were verified outside the sandbox. All eight planned ResNet configurations completed and passed persisted-prediction acceptance. Frozen-head training used CPU; encoding and tuning used MPS. Phase 6 completed the 466-image refit for 28 fixed head epochs on CPU and the 82-image test evaluation on MPS. See `workflow.md` and final reports for implementation, verification and artifact details.

## References

- [Torchvision ResNet18 and explicit weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Torchvision ResNet50 and explicit weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- [PyTorch transfer-learning workflow](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [scikit-learn train-only transformations and leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html).
