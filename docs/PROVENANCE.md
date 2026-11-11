# Provenance and publication status

Repository maintainer identity comes from the existing local Git configuration:
Abir Ahmmed (`devbyabir`). The maintainer has requested project credit under this
name and confirmed publication permission on 2026-11-12. This declaration is
recorded from the maintainer; it is not independent verification of historical
sole authorship. Upstream attribution remains separate from project credit.

## Original experiments

The original notebooks remain in the parent workspace. They are not included in
this repository. Their Colab metadata references Unsloth notebooks, and some
cells retain Google copyright notices. Any migrated third-party code must retain
its applicable notices and attribution. This rebuild must not erase collaborator
credit or claim that upstream implementations were authored here.

## Data

The maintainer confirms hospital publication permission. Patient identity
handling and dataset-specific redistribution terms still need documentation.
No original images, report CSVs, screenshots, Drive
identifiers, or model files are committed here.

A training dataset requires a source URL, access/license terms, explicit group
identifiers, and a declaration of whether report targets are synthetic or
clinician-authored. Public availability alone does not establish redistribution
rights. Do not commit patient identifiers; keep manifests in ignored `data/`.

## Results

Legacy saved metrics are not accepted as benchmarks. The original merged CSV
contained 1,300 rows but only 300 distinct image paths and 1,000 exact duplicate
rows. The old evaluation reused training examples; a ROC legend altered AUC
values with random subtraction. New results must be recomputed independently.
