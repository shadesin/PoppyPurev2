# Revised evaluation and image-processing plan

**Superseded evaluation:** [active v4 train/validation/test protocol](TRAIN_VAL_TEST.md) replaces the condition holdout described below. Image-processing ideas below remain relevant, with design now restricted to v4 training images. This document preserves the earlier v2/v3 decisions.

**Active protocol: v3.** Keeps v2's 414/134 split and records the user's confirmation that images represent individually prepared mixtures. Verified same-scene views remain grouped. Earlier protocols are preserved as history; no image-processing implementation, model fitting or scoring has occurred.

## 1. Why revise the split?

The original split was designed around an entire background change. It left training with only black-background images and approximately half the usable data. That was too restrictive as the main protocol for a classifier intended to handle this dataset's different backgrounds and imaging modes.

There are **548 usable close-ups**, not 569 independent training examples: seven duplicate copies and 14 whole-tray overview references were excluded in Phase 1. The second Mendeley archive is a shared subset, not hundreds of extra independent examples. More copies or augmented views do not increase the number of physical mixtures.

Holding back data for testing has a purpose, but the earlier allocation reserved more than necessary for this project. The revised design uses roughly **75.5% for development/final fitting and 24.5% for final testing**. This increases training diversity and image count; better accuracy is not guaranteed.

## 2. Active split

| Condition | Images | Role |
| --- | ---: | --- |
| Black background, 4× folder | 137 | Development |
| Black background, Micro folder | 140 | Development |
| White background, 4× folder | 137 | Development |
| White background, Micro folder | 134 | Final holdout |
| **Total** | **548** | **414 development + 134 holdout** |

This exposes training to both backgrounds and both folder magnifications. It reserves the white/Micro combination as the final condition-transfer test. Both partitions contain all seven classes.

| Volume class | Development | Final holdout |
| --- | ---: | ---: |
| 0 | 57 | 16 |
| 10 | 60 | 21 |
| 20 | 56 | 17 |
| 30 | 60 | 20 |
| 40 | 61 | 20 |
| 50 | 60 | 20 |
| 100 | 60 | 20 |

White/Micro is selected because it is a whole condition of about one quarter of the data, contains every class, has no known magnification-tag conflicts, and leaves both backgrounds/magnifications in development. This is a metadata-based choice made before model results. Preparation provenance is supplied by the user's confirmation in [preparation_provenance.json](../../data/preparation_provenance.json).

### Development: three validation rounds

| Round | Fit | Validate |
| --- | --- | --- |
| 1 | Black Micro + White 4×: 277 | Black 4×: 137 |
| 2 | Black 4× + White 4×: 274 | Black Micro: 140 |
| 3 | Black 4× + Black Micro: 277 | White 4×: 137 |

Every development image is validation data once and training data twice. Select using the **mean of the three seven-class macro F1 scores**, saving each score and out-of-fold predictions. Refitting the chosen recipe uses **all 414 development images**, not just the 274–277 used in an individual validation round. Cross-validation reuse is standard; the final test stays outside that loop. See [scikit-learn's validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

The same-scene pair remains together in black Micro and shares one preparation unit. The other included images each define one preparation unit, giving 547 units across 548 images. Broad class/background/folder blocks remain together. The earlier class-wide groups are preserved only as legacy metadata, not current preparation identities. Four white 4× images retain their source folder condition and magnification-conflict flags.

### Final evaluation and using all data

1. Lock the full recipe using development results only.
2. Fit on 414 development images and score once on the 134 held-out images. Preserve that model, predictions and report.
3. After reporting the final evaluation, optionally refit the **unchanged recipe on all 548 images** as a separately versioned deployment model. This makes all usable data available for fitting the deliverable. That model has no untouched test set; the 414-image model's score must not be presented as a direct test of the 548-image model.

Keep the 414/134 split to test the held-out capture combination. Confirmed related views remain together. If confidence intervals are produced, resample preparation units rather than individual related views.

## 3. Phase 3 — planned image processing

**Not yet implemented.** Design and visual QC will use all three development conditions and every class. White/Micro images remain reserved. All transforms must be reproducible and identical at evaluation/inference.

### A. Establish a simple reference pipeline

- Decode to RGB and apply EXIF orientation consistently.
- Preserve original files and aspect ratio. Use an aspect-preserving resize with padding when the model requires a fixed shape; avoid stretching grains into a square.
- Inspect native-resolution development crops to choose useful processing scales. Candidate overview sizes are 512 and 768 pixels on the long side; these are starting candidates, not validated resolutions. Check whether grains remain visible before choosing.
- Keep a whole-image RGB reference so we can measure whether more processing actually helps. Do not apply blanket blur, sharpening or contrast correction.

### B. Reduce borders, grids and watermark shortcuts

- Compare whole-image input with a deterministic interior/mixture-region crop, chosen and reviewed on development images.
- Check that crops remove visible tray borders/phone markings without systematically removing material. A central crop is a candidate, not an assumed correct foreground region.
- Save crop coordinates and overlays. If a universal crop fails, retain the whole-image reference or use a documented image-derived rule rather than class-specific manual crops.

### C. Preserve grain detail with multiple regions

- If a full-image resize erases grains, extract several spatially distributed high-resolution tiles within the valid mixture region. Start with a reproducible grid and compare tile scales on development only.
- Aggregate region features, or use an image-level pooled CNN representation, to make **one prediction per original image**.
- A small tile may have a different local composition. Do not treat every tile as independently verified at the source image's volume percentage. Train/evaluate at image level, with the source label supervising the pooled representation.
- Tiles inherit source-image split membership. Tiles increase visible detail, not the number of independent samples.

### D. Test an optional material mask

- Compare HSV/Lab brightness and colour separation, with Otsu or adaptive thresholding as candidates where appropriate.
- Use only light, scale-aware morphological cleanup, and inspect overlays for all classes and development conditions. White semolina on white background may merge; dark seeds can be mistaken for shadow. Avoid erosion or speck removal that deletes real small particles.
- Keep a mask only if it consistently retains **both poppy seeds and semolina**, and improves development evaluation. Otherwise use crops/full images. No mask is mandatory.
- A mask describes visible material; its pixel fraction is **not a volume-percentage label or estimate**.

### E. Classical colour and texture features

- HSV or Lab colour histograms, normalized over the chosen valid region.
- Uniform LBP histograms to describe local grain texture, using consistent integer grayscale conversion and correct bins. [LBP reference](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html).
- GLCM statistics such as contrast, dissimilarity, homogeneity, energy and correlation, using normalized matrices. Start with 32 grayscale levels, offsets of 1/2/4 pixels and four directions; freeze definitions after development comparison. [GLCM reference](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html).
- Keep color and texture as separate feature families initially, then compare their combination. Grayscale is for texture computation, not a replacement for every RGB input.
- Compute textures within valid patches; mask edges and blacked-out background must not become artificial texture. Aggregate patch descriptors to image level.
- Pixel distances and apparent grain sizes are not calibrated physical sizes. Avoid particle counting/size-based claims without reliable segmentation and scale evidence.

### F. CNN input and augmentation

- Retain RGB and use the selected pretrained encoder's documented normalization. Compare an overview representation with pooled detail regions if grain visibility requires it. Pretrained weight packages define their input transforms: [Torchvision reference](https://docs.pytorch.org/vision/stable/models.html).
- Training-only rotations/flips and mild brightness/contrast variation are candidates. Avoid severe hue shifts or aggressive random crops that can distort material appearance/composition.
- Compare the frozen pretrained encoder first, then limited fine-tuning in Phase 5. Use the same image-level folds as the classical models.

### G. Optional lighting correction

Compare unchanged lighting with a modest correction such as CLAHE on luminance only, if development failures show a need. Keep it only when it improves the three-fold comparison and preserves material appearance. Do not stack preprocessing methods simply because they are available.

## 4. How we choose processing

Use a small recorded comparison: whole-image reference → interior crop/detail pooling → optional mask → optional lighting correction. Change one component at a time. Check overlays as well as macro F1; reject a visually invalid mask even if a confounded score rises. Limit trials because three conditions are a small development set.

Keep learned scaling, feature selection and all fitted processing inside each training fold. Phase 3 produces candidate processing and QC; Phases 4–5 compare classifiers and select the complete recipe. No preprocessing technique above has been validated yet. No new dependencies were installed for this planning revision.

## 5. Scope and artifacts

The original seven labels remain adulteration **by volume**. Treat images as individually prepared mixtures based on the user's confirmation, merging the verified same-scene pair. Report all three development fold scores and the final held-out condition result.

Active files: [v3 membership](../../data/splits/v3/assignments.csv), [v3 protocol](../../data/splits/v3/protocol.json), [v3 freeze](../../data/splits/v3/freeze.json). Historical v1/v2 files and run records remain unchanged. V3 preserves the split and scoring rules while replacing the superseded missing-preparation assumption with recorded user provenance.

```sh
uv run --locked poppypure freeze-evaluation             # verify active v3
uv run --locked poppypure freeze-evaluation --version v1 # verify historical v1
```

Verification: all seven tests passed with added v2 coverage, all classes present in each training/validation split and holdout, 414/134 counts, known pair kept together, v1 bytes unchanged, and Ruff checks passed. No model or image preprocessing was run.
