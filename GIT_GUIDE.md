# What to put in Git

Use **PoppyPurev2 as the repository root**. The original sibling `PoppyPure/` project is a separate historical project.

## Commit these files

| Files | Purpose |
| --- | --- |
| `src/` | Model training, preprocessing, inference and CLI source |
| `app/` | Python server and HTML/CSS/JavaScript interface |
| `scripts/`, `tests/` | Reproduction, verification and software checks |
| `configs/` | Fixed experiment and preprocessing settings |
| `pyproject.toml`, `uv.lock` | Package configuration and pinned dependencies |
| `run_app.command` | Mac launcher; Windows users can use the Python command |
| `data/` metadata files and `data/splits/` | Source provenance, manifest, review decisions and frozen image allocations |
| `reports/phase1/` through `reports/phase6/` | Findings, metrics, dataset/model cards and figures |
| `experiments/SCHEMA.md` | Experiment record format |
| `releases/poppypure-v1/` | Model asset checksum manifest and prepared release notes |
| README, plans, `workflow.md`, `AGENTS.md`, this guide, `.gitignore`, `.gitattributes` | Usage, history, contributor instructions and stable checkout bytes |

The ignore rules exclude generated caches, raw data links and report logs even when adding these directories.

`.gitattributes` keeps source files at LF line endings and preserves exact CSV/JSON bytes, including the original CSV CRLF endings. These records have frozen SHA-256 identities; changing line endings would invalidate them.

## Keep these out of ordinary Git commits

- `.venv/`, `.uv-cache/`, `__pycache__/`, Ruff/Matplotlib caches and generated package metadata.
- `data/raw`, `data/source_archives`: links to data on Backup Plus, not portable dataset copies.
- `data/features/`, `data/embeddings/`: reproducible feature and embedding caches.
- `experiments/runs/`: intermediate checkpoints, traces and source snapshots. Preserve the local archive.
- `models/`: trained and pretrained weights, earlier attempts and the full runtime bundle. Distribute the selected final bundle separately as described below.
- `reports/generated/`: local app verification screenshots and temporary verification outputs.
- Report `.log` files, `poppypure-v1.zip` and its `.sha256` sidecar.

Do not copy private settings, credentials or personal test photos into the repository.

## Share the trained model

**A source checkout alone cannot predict images:** it also needs the saved model bundle. The checkpoint is about **43 MiB** and is currently ignored by Git.

Recommended distribution: attach the complete **`models/poppypure-v1/`** bundle as a ZIP to a GitHub Release. This retains the model, class mapping, preprocessing recipe, checksums and model documentation together. Large File Storage is another option if you choose to manage weights in the repository.

The prepared local assets are `poppypure-v1.zip` and `poppypure-v1.zip.sha256` in the repository root. The ZIP contains exactly the 51 release-listed files plus `RELEASE_MANIFEST.json`; Python caches and unrelated local files are excluded. The committed [asset manifest](releases/poppypure-v1/ASSET_MANIFEST.json) records the ZIP SHA-256, byte count and checkpoint identity.

To verify the prepared or downloaded ZIP:

```sh
uv run --locked python scripts/package_model.py verify poppypure-v1.zip
```

To rebuild from the existing model bundle, use `uv run --locked python scripts/package_model.py build`. It refuses existing assets; add `--overwrite` only when intentionally replacing them. Packaging verifies the source bundle first, writes fixed ZIP metadata, then verifies every archived file. It writes the ZIP, sidecar and committed asset manifest together. It does not train or modify the model.

The archive's contents start at `models/poppypure-v1/`. Recipients verify it and then extract it into their checkout root with `uv run --locked python -m zipfile -e poppypure-v1.zip .`, preserving this layout:

```text
PoppyPurev2/
  app/
  src/
  models/
    poppypure-v1/
      model.pt
      bundle.json
      recipe.json
      RELEASE_MANIFEST.json
      ...
```

The README links to the repository's Releases page and documents the preparation status. No asset has been uploaded. Prepared [release notes](releases/poppypure-v1/RELEASE_NOTES.md) list the two files to attach to a future release.

## Local Git repository

The repository was initialized locally on `main` at the user's request. Its origin is **`https://github.com/shadesin/PoppyPurev2.git`**. The initial source commit includes the app, source, settings, metadata, reports, release manifest and instructions. Commit authorship uses the owner's GitHub-provided noreply address in repository-local Git configuration.

Review subsequent changes with:

```sh
git status --short
git diff
git diff --cached --stat
```

Stage and commit intended changes following the inclusion/exclusion lists above. **The user requested a local commit only; pushing and release publication are pending.**

When the user authorizes publishing, the configured remote can receive `main`:

```sh
git push -u origin main
```

A future GitHub Release should target the published source commit and attach both prepared assets. Creating the ZIP or committing it locally does not create a GitHub Release. Keep Backup Plus data and earlier experiment outputs locally; pushing source does not back those up.

## Running a checkout

After installing uv and extracting the final model bundle, run:

```sh
uv sync --locked --python 3.12
uv run --locked python app/server.py --device cpu --open
```

These commands work with the intended Windows/macOS setup. Actual app/browser verification was performed on macOS; Windows dependencies were inspected but no Windows machine was tested. See the README for input formats and photo capture specifications.
