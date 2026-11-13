# Legacy project audit (2026-11-14)

Scope: the parent workspace (`Model/`, `LLM Report/`, sample images) that this
repository succeeds. Nothing listed here is committed to this repository.

## What exists

| Item | Contents |
|---|---|
| `Model/*` (8 notebooks) | Keras CNNs (VGG16, MobileNet, InceptionV3, ResNet50, EfficientNetB7), Grad-CAM via `tf-keras-vis` |
| `Model/augmentation.py` | Offline augmentation that writes `_aug_` copies into the class folders |
| `LLM Report/*` (13 notebooks) | Synthetic report generation (Llama 3.2 11B), Unsloth LoRA fine-tuning and inference for Llama 3.2 11B, Qwen3-VL 8B, Ministral 3B |
| `fine_tuning_dataset*.csv/.xlsx` | 300 unique synthetic image/report rows; the "1300-merged" file repeats them |
| `1.jpg`, `2.jpg` | Screenshots of the Google Drive `Model/` and `LLM Report/` folders |
| `3.jpg` | Screenshot of a legacy inference output (see failure example below) |

## Results that are not trustworthy

1. **CNN accuracy ≈ 99.1% (EfficientNetB7).** The image folder held exactly 5,000
   images for each of 15 classes (75,000 total). ChestX-ray14 has only a few hundred
   Hernia images, so most of the folder had to be offline augmentations made
   *before* splitting. The split was by shuffled batch, so augmented copies of the
   same image (and images of the same patient) land in both train and test.
   `image_dataset_from_directory(shuffle=True)` reshuffles on every iteration, so
   `take`/`skip` do not even give a fixed partition. The test set also served as
   the validation set during training. The result measures leakage, not generalisation.
2. **Single-label framing.** ChestX-ray14 findings co-occur. Each image was forced
   into one of 15 folder classes, with "No Finding" treated as a class.
3. **Altered ROC legend.** The ROC plot subtracted `uniform(0.001, 0.009)` from each
   AUC value in the legend. The new evaluation (`cxr_insight.evaluate_cnn`) prints
   exact values only.
4. **Augmentation mirrored chest images** (`FLIP_LEFT_RIGHT`), which reverses
   anatomy laterality.
5. **Report generation.** Inference notebooks attached an untrained LoRA adapter,
   so their outputs came from the base model. The synthetic training targets have
   these defects (measured with `cxr-legacy-audit`):

   | Measure | Count |
   |---|---|
   | Rows in merged CSV | 1,300 |
   | Exact duplicate rows | 1,000 |
   | Unique rows | 300 (20 per folder) |
   | Rows with `user\n\nassistant` artefact | 300 / 300 |
   | Rows copying the prompt's example findings | 221 / 300 |
   | Rows with all three sections | 294 / 300 |
   | Prompt label differs from image folder | 2 |

   No BLEU/METEOR value from these notebooks is evidence of report quality.

   *Failure example* (`3.jpg`): for a frontal **chest** radiograph the CNN output
   "Hernia", and the generated report began "TECHNIQUE: Upright abdominal X-ray"
   and concluded "a large abdominal hernia, likely an inguinal hernia". The
   projection is wrong, and an inguinal hernia lies outside a chest radiograph's
   field of view (NIH "Hernia" labels are typically hiatal). The model followed the
   conditioning label rather than the image.

## Missing artefacts

- No CNN checkpoint (`efficientnetb7_model.keras` was only on Google Drive).
- No trained LoRA adapter for any VLM.
- The 75,000-image dataset zip, which exists only on Google Drive. Its source and
  usage terms are undocumented. Filenames (e.g. `35416061_849.png`) do not follow
  NIH naming and carry no verified patient identity, so it cannot be split by patient.
- No clinician-authored reports.

## Decisions taken in this repository

- CNN data: NIH ChestX-ray14 from the official metadata, 14-label multi-label,
  NIH's official patient-wise test list kept frozen, train/validation split by patient.
- Augmentation inside the training loader only, without mirroring.
- The legacy dataset is kept out of the repository and out of training.
- VLM fine-tuning needs report data with documented provenance and is not run
  until that data is chosen.
