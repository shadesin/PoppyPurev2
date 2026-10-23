# Final dataset card

## Task and source

Seven semolina adulteration classes **by volume**: 0, 10, 20, 30, 40, 50 and 100. Source: Sarkar, Tanmay; Mukherjee, Alok; Chatterjee, Kingshuk (2023), *Digital Image dataset for Poppy seed adulteration with Semolina*, Mendeley Data v1, [DOI](https://doi.org/10.17632/p2gvvnh8y4.1), CC BY 4.0.

The related Ganguly/Sarkar (2026) dataset [DOI](https://doi.org/10.17632/nx8m9kxy26.1) supplies label provenance; overlapping images were not added as extra independent observations.

## Inventory and allocation

Original JPEGs: **569**. Included close-ups: **548**. Seven extra exact copies and fourteen loose whole-tray reference images are excluded. The reviewed shifted-view pair is retained within one preparation group. Raw bytes are preserved on Backup Plus through `data/raw` and `data/source_archives` symlinks.

Development allocation: 384 training / 82 validation / 82 test. Final fitting: 466 images / 465 preparation units; test: 82 images / 82 units. Preparation identities follow the user's confirmation of individually prepared mixtures, with the known same-scene pair merged. All seven classes and all four black/white × 4x/micro conditions appear in every original partition.

Included magnification-conflict flags: 4. These remain in overall fitting/evaluation and are omitted only from trusted magnification summaries. No labels or split membership were changed using model results.

## Integrity and transformations

Manifest SHA-256: `d5358c61f2f42c4dbd5c4efc92e4d62ba54caff998ecbf631f665e7e1fbc5829`.

Split SHA-256: `cf7e5f8e2ed1f7dd3537bd62c611f54bcc55decc0a5e8ad681f641a4f7db88c0`.

Image bytes are checked against the frozen manifest. Model processing uses RGB/orientation normalization, the fixed material crop, five native detail regions and normalized CNN inputs. No image-area estimate is converted into a volume label. Labels remain inherited source classes.

See [the original dataset audit](../phase1/DATASET_CARD.md), [split protocol](../phase2/TRAIN_VAL_TEST.md) and [final evaluation](FINAL_REPORT.md).
