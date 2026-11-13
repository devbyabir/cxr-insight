"""Multi-label classification metrics (numpy only).

AUROC values are computed exactly and never adjusted. Classes with no positive
or no negative examples have undefined AUROC and are reported as None, not 0.5.
"""
import numpy as np


def _as_arrays(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch: {y.shape} vs {s.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be 0/1")
    return y, s


def auroc(y_true, scores):
    """Mann-Whitney AUROC with average ranks for ties; None when undefined."""
    y, s = _as_arrays(y_true, scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """False/true positive rates over all distinct thresholds (for plotting)."""
    y, s = _as_arrays(y_true, scores)
    thresholds = np.unique(s)[::-1]
    pos, neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    fpr = [0.0] + [float(((s >= t) & (y == 0)).sum() / neg) for t in thresholds]
    tpr = [0.0] + [float(((s >= t) & (y == 1)).sum() / pos) for t in thresholds]
    return fpr, tpr


def _counts(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def choose_thresholds(y_true, scores, default=0.5):
    """Per-class threshold maximising F1 — fit on validation data only."""
    y, s = _as_arrays(y_true, scores)
    thresholds = []
    for k in range(y.shape[1]):
        if y[:, k].sum() == 0:
            thresholds.append(default)
            continue
        best_t, best_f1 = default, -1.0
        for t in np.unique(s[:, k]):
            f1 = _prf(*_counts(y[:, k], (s[:, k] >= t).astype(int))[:3])[2]
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        thresholds.append(best_t)
    return thresholds


def classification_report(y_true, scores, labels, thresholds):
    """Per-class metrics with confusion counts, plus micro/macro summaries."""
    y, s = _as_arrays(y_true, scores)
    if y.ndim != 2 or y.shape[1] != len(labels) or len(thresholds) != len(labels):
        raise ValueError("labels/thresholds must match the number of columns")
    p = (s >= np.asarray(thresholds)).astype(int)
    per_class, totals = [], np.zeros(4, dtype=int)
    for k, label in enumerate(labels):
        tp, fp, fn, tn = _counts(y[:, k], p[:, k])
        totals += (tp, fp, fn, tn)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class.append({
            "label": label, "support": int(y[:, k].sum()), "threshold": float(thresholds[k]),
            "auroc": auroc(y[:, k], s[:, k]), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    defined = [c["auroc"] for c in per_class if c["auroc"] is not None]
    supported = [c for c in per_class if c["support"] > 0]
    micro_p, micro_r, micro_f1 = _prf(*totals[:3])
    return {
        "n_images": int(len(y)),
        "per_class": per_class,
        "macro_auroc": float(np.mean(defined)) if defined else None,
        "auroc_defined_for": len(defined),
        "macro_f1_supported": float(np.mean([c["f1"] for c in supported])) if supported else None,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
    }


def bootstrap_macro_auroc(y_true, scores, groups, n_resamples=1000, seed=0, alpha=0.05):
    """Percentile CI for macro AUROC, resampling patients (groups), not images."""
    y, s = _as_arrays(y_true, scores)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        rows = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        aucs = [auroc(y[rows, k], s[rows, k]) for k in range(y.shape[1])]
        aucs = [a for a in aucs if a is not None]
        if aucs:
            values.append(np.mean(aucs))
    if not values:
        return None
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"low": float(low), "high": float(high), "resamples": len(values), "unit": "patient"}
