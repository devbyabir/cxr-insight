# Evaluation protocol

1. Record the source, usage terms, report provenance, and inclusion criteria.
2. Remove exact image duplicates before splitting. Conflicting labels/reports for
   an identical image require manual resolution; never select one silently.
3. Split by patient (or a verified equivalent identity group), not by image,
   batch, filename ordering, or augmented variant. Validate group disjointness.
4. Generate augmentation only inside the training loader. Do not mirror chest
   images when laterality is clinically meaningful.
5. Use validation data for checkpoint selection. Keep the test partition frozen
   and evaluate it only after experiment choices have been fixed.
6. State whether classification is single-label or multi-label. Do not coerce
   coexisting findings into one mutually exclusive label without justification.
7. Report class support, per-class precision/recall/F1, macro and micro averages,
   AUROC where defined, and confidence intervals. Preserve raw predictions.
8. Evaluate report generation separately on held-out, preferably independently
   reviewed references. Text similarity is not clinical factual accuracy.
9. Compare a pretrained baseline and the fine-tuned adapter on the same cases.
   Record hallucinations, omissions, incorrect negation/laterality, latency,
   hardware, generation settings, and failure examples.
10. Save dataset/manifest hashes, package versions, seed, Git commit, model ID,
    revision, adapter, and evaluation command alongside every result.

No arbitrary target score is a release criterion. A useful portfolio demonstrates
reproducibility, honest results, and clear limitations even when accuracy is modest.
