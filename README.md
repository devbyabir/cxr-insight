# CXR Insight

Chest X-ray analysis and vision-language report generation, developed from a
hospital internship research prototype.

**Status: engineering rebuild. No validated checkpoint or deployed inference
service is included yet. Not for diagnosis or clinical use.**

The original experiments explored five CNN architectures, Grad-CAM, and three
vision-language model families. This repository is the reproducible successor;
legacy notebooks and hospital material are deliberately outside its Git root
until provenance and publication rights are established.

## What is being improved

- Patient-grouped train/validation/test separation, before augmentation.
- Exact image deduplication and auditable dataset manifests.
- Held-out evaluation, with no modified or fabricated metrics.
- Explicit distinction between synthetic and clinician-authored reports.
- Versioned model artifacts and an inference UI that reports its actual status.

## Development

Use Python 3.11 or 3.12 for GPU work. The dependency-free data utilities can also
run on newer Python versions.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Evidence and attribution

See [provenance](docs/PROVENANCE.md) and [evaluation protocol](docs/EVALUATION.md).
No legacy accuracy, BLEU, or METEOR value is presented as a validated result.
Actual training and evaluation artifacts must be attached before making claims.

Copyright and licensing of the original work remain to be confirmed. No blanket
license is assigned to third-party code, hospital data, or model weights.
