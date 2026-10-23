# PoppyPure v2

PoppyPure detects **semolina adulteration in poppy seeds from close-up photos**. Paste, drag and drop, or upload an image into the local app to predict the semolina percentage **by volume**.

The classifier returns one of seven levels: **0%, 10%, 20%, 30%, 40%, 50% or 100%**. Zero means pure poppy seeds; 100% means pure semolina. These are predicted classes, not a continuous measurement.

Built with Python and PyTorch, with a plain HTML, CSS and JavaScript interface. Images are processed on your computer.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) to install Python and the pinned dependencies.
- A modern browser.
- The trained model ZIP from the release linked below.

A GPU is not required. The app was tested on macOS; its dependencies support 64-bit Windows, but Windows execution has not been tested. Internet is needed for the initial downloads and installation; prediction works offline afterward.

## Install and run

### 1. Download the source

```sh
git clone https://github.com/shadesin/PoppyPurev2.git
cd PoppyPurev2
```

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and open a terminal in the folder containing this README.

### 2. Download the trained model

Download both assets from [Release v1.0.0](https://github.com/shadesin/PoppyPurev2/releases/tag/v1.0.0):

- **[poppypure-v1.zip](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip)** — model bundle, 40.9 MiB.
- **[poppypure-v1.zip.sha256](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip.sha256)** — download checksum.

Save both files beside this README. The source download does not include the model weights.

### 3. Install dependencies, verify and extract

Run these commands from the project folder in a macOS terminal or Windows PowerShell:

```sh
uv sync --locked --python 3.12
uv run --locked python scripts/package_model.py verify poppypure-v1.zip
```

Continue only if verification reports **`"status": "passed"`**, then extract:

```sh
uv run --locked python -m zipfile -e poppypure-v1.zip .
```

Extract into the **project root**. The ZIP already includes the `models/poppypure-v1/` folders. The resulting files should include:

```text
PoppyPurev2/
  README.md
  app/server.py
  models/poppypure-v1/model.pt
  models/poppypure-v1/bundle.json
  models/poppypure-v1/RELEASE_MANIFEST.json
```

### 4. Start the app

```sh
uv run --locked python app/server.py --device cpu --open
```

Open **http://127.0.0.1:8765** if the browser does not open automatically. Keep the terminal running; press **Ctrl+C** there to stop the app.

On macOS, you can also double-click **`run_app.command`** after setup. Use `--device auto` instead of `--device cpu` to use Apple MPS acceleration when available.

## Use the app

1. Drag and drop a photo, click **Choose image**, or paste an image using **⌘V** on macOS / **Ctrl+V** on Windows.
2. The app analyses it automatically and displays the predicted adulteration percentage **by volume**.
3. Click the preview to choose another image, or **Clear** to reset.

Supported files: **JPEG, PNG and still WebP**, up to **20 MB**. Export HEIC/HEIF photos as JPEG first. Pasting requires an image on the clipboard, not a filename or URL.

The app listens only on your computer's loopback interface, deletes temporary uploads after processing, and keeps no image history. Neither the training dataset nor an external drive is needed for prediction.

## Photo capture specifications

A phone can capture suitable images. Use the original camera file and follow these guidelines:

| Aspect | How to capture the sample |
| --- | --- |
| Background | Plain, flat, matte **white or black**. Avoid coloured, patterned, glossy or transparent surfaces. |
| Spreading | Gently mix and spread a **thin, fairly even layer**. Grains may touch; avoid heaps, heavy overlap and separate ingredient pockets. |
| Framing | Fill most of the frame with a representative area of material, including near the corners and centre. Avoid large empty borders, hands and tray edges. |
| Angle | Shoot directly down with the camera approximately parallel to the surface. |
| Focus and distance | Move close enough to resolve individual grains while keeping them sharp. Clean the lens, tap to focus and hold the phone steady. No fixed distance works for every phone. |
| Lighting | Use bright, even, diffuse light. Avoid direct flash, glare, harsh shadows and the phone's shadow. |
| Camera settings | Use normal photo mode and full resolution. Avoid digital zoom, portrait blur and filters. Use macro mode only if it preserves sharp detail. |
| File and resolution | Prefer the original image at **1600 × 1200 pixels or larger** (either orientation). Avoid screenshots and reduced-resolution messaging copies. Do not enlarge a small image to bypass the size check. |

The model does not automatically locate the sample, remove the background or check focus. Keep material throughout the frame. Undersized files are rejected, but passing the size check does not establish photo quality.

## Command-line prediction

```sh
uv run --locked poppypure predict "/path/to/sample.jpg" --device cpu
```

The JSON output includes the predicted by-volume class, an adulteration flag and seven class probabilities. Add `--output prediction.json` to save it to a new file. The probabilities are uncalibrated and should not be treated as verified confidence.

## Model and results

The packaged model uses **ResNet18 with an MLP classifier**. It achieved **91.46% accuracy (75/82 images)** and **0.9138 macro F1** on the reserved test set.

Capture conditions can affect predictions, including false adulteration predictions for pure poppy. These test results do not establish accuracy for every phone or daily-use setup.

The training images come from the [Mendeley poppy seed and semolina dataset](https://data.mendeley.com/datasets/p2gvvnh8y4/1). The model download includes its model card and evaluation documentation.

## Troubleshooting

- **Model missing:** extract `poppypure-v1.zip` into the project root and check that `models/poppypure-v1/model.pt` exists.
- **Image too small or unreadable:** use the original close-up photo; export unsupported formats as JPEG.
- **Port already in use:** start with `uv run --locked python app/server.py --device cpu --port 8766 --open`.
- **Apple acceleration unavailable:** use `--device cpu`.
- **Paste does nothing:** copy the actual image rather than its filename or URL, or use file selection.
