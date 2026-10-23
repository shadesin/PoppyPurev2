# Dataset card: PoppyPure v2

## Task and labels

The task is seven-class image classification of semolina adulteration in poppy seeds. Labels mean **semolina percentage by volume**: 0, 10, 20, 30, 40, 50, and 100. Zero means pure poppy seeds; 100 means pure semolina. Folder labels are inherited from the source. The user confirms that the images represent individually prepared mixtures.

## Source and attribution

Canonical source: Sarkar, Tanmay; Mukherjee, Alok; Chatterjee, Kingshuk (2023), *Digital Image dataset for Poppy seed adulteration with Semolina*, Mendeley Data, version 1, DOI [10.17632/p2gvvnh8y4.1](https://doi.org/10.17632/p2gvvnh8y4.1).

Related label provenance: Ganguly, Debayan; Sarkar, Tanmay (2026), *A Rigorous Feature Engineering Framework and Deployment for Poppy Seed Adulteration Detection*, Mendeley Data, version 1, DOI [10.17632/nx8m9kxy26.1](https://doi.org/10.17632/nx8m9kxy26.1). Its description explicitly defines the seven labels by volume. Its images are not added as independent observations. The overlap report compares ZIP CRC32, uncompressed size, and class; this is strong evidence of shared files, not a SHA-256 proof for the second archive.

Both sources use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the contributors, retain attribution, and identify modifications. Raw image bytes remain unchanged; generated thumbnails, metadata, exclusions, and grouping decisions are PoppyPure v2 additions. Public metadata snapshots and the pinned download checksum are saved under `data/`.

## Storage and integrity

The source archive and extracted images are stored on `/Volumes/Backup Plus/PoppyPurev2/`, exposed through local links `data/source_archives` and `data/raw`. The original archive is verified against its published SHA-256. Every extracted image is compared with its archive member during the audit. The manifest identifies image bytes by SHA-256 and decoded, oriented RGB pixels by a separate SHA-256.

## Scope and decisions

The primary modelling set consists of the structured 4x and micro close-up images on black and white backgrounds. Fourteen loose files are whole-tray overview references, with tray borders, grid lines, and phone watermarks. They remain available in the raw dataset but are excluded from the primary close-up task. Byte-identical and pixel-identical copies are excluded with the retained image ID recorded. Quality flags are for review and do not automatically exclude an image. No label is silently changed.

`data/manifest.csv` accounts for every image; `data/archive_inventory.csv` accounts for every ZIP entry; `data/image_decisions.csv` explains every inclusion or exclusion. There are 569 original JPEGs, 562 unique byte images, and 548 included close-ups after removing seven duplicate copies and excluding the 14 overview references. Final counts and review conclusions are recorded in `AUDIT.md` and `summary.json`. Eight entries have conflicting magnification-folder tags and are explicitly flagged.

## Preparation provenance and evaluation

Preparation provenance is recorded in `data/preparation_provenance.json`: the user confirms individually prepared mixtures. Each retained image defines a preparation unit except the visually confirmed same-scene pair, which shares one unit. This yields 547 units across 548 included images. The seven class-wide `group_id` values in the historical Phase 1 manifest are superseded as preparation identities and preserved as legacy metadata in v3 split membership.

Active protocol v4 uses 384 training, 82 validation and 82 test images, with all classes and capture conditions represented in each set and confirmed related views together. See `reports/phase2/TRAIN_VAL_TEST.md`. Earlier missing-preparation conclusions in the frozen audit are historical and superseded by the user's confirmation.

## Intended use

Classification of poppy seeds mixed with semolina into the seven documented by-volume classes. The classifier returns a class rather than treating visible image coverage as a volume measurement.
