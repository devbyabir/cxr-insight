# Attribution

The MIT license in `LICENSE` covers the code in this repository only. It does not
cover datasets, model weights, or third-party software, which keep their own terms.

## Data

- **NIH ChestX-ray14** — provided by the NIH Clinical Center.
  Download: https://nihcc.app.box.com/v/ChestXray-NIHCC. NIH states there are no
  restrictions on use of the images, and asks users to link to the download site,
  cite the paper below, and acknowledge the NIH Clinical Center as the data provider.
  Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale
  Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
  Localization of Common Thorax Diseases.* IEEE CVPR 2017, pp. 3462–3471.
  The images and metadata were downloaded through the Hugging Face mirror
  `alkzar90/NIH-Chest-X-ray-dataset`, which redistributes the NIH files; the archive
  hash used is recorded in the run metadata. No images are redistributed here.

## Models and methods

- **EfficientNet** — Tan M, Le QV. *EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks.* ICML 2019. ImageNet-1k weights from
  torchvision (BSD-3-Clause).
- **Grad-CAM** — Selvaraju RR et al. *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization.* ICCV 2017. Implemented here from the paper.
- **Unsloth** — https://github.com/unslothai/unsloth. The vision fine-tuning flow in
  `notebooks/finetune_vlm_colab.ipynb` and `cxr_insight.models` is adapted from
  Unsloth's public notebooks (https://github.com/unslothai/notebooks); see their
  repositories for license terms.
- **Qwen2-VL** (default VLM in the Colab notebook) and the models used in the legacy
  experiments (Llama 3.2 Vision, Qwen3-VL, Ministral) are subject to their providers'
  model licenses. No weights are distributed here.

## Software

PyTorch and torchvision (BSD-3-Clause), Streamlit (Apache-2.0), NumPy (BSD-3-Clause),
Pillow (MIT-CMU), Matplotlib (Matplotlib license), Hugging Face Transformers/TRL/PEFT
(Apache-2.0).
