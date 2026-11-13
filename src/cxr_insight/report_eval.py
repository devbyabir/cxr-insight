"""Held-out evaluation of generated reports: baseline vs fine-tuned, same cases.

Measures structure (all sections present), prompt-example copying, and agreement
of *mentioned findings* between generated and reference text, using a small
keyword + negation rule labeller. That labeller is a crude proxy — it is not a
validated clinical labeller (e.g. CheXpert/CheXbert) — and BLEU-style overlap is
lexical similarity only. Neither measures clinical correctness.

Input JSONL rows: {"id": ..., "reference": ..., "generated": ...}
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .inference import parse_report
from .nih import NIH_LABELS

KEYWORDS = {
    "Atelectasis": ("atelectasis", "atelectatic"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement"),
    "Consolidation": ("consolidation",),
    "Edema": ("edema", "oedema"),
    "Effusion": ("effusion",),
    "Emphysema": ("emphysema", "hyperinflation", "hyperinflated"),
    "Fibrosis": ("fibrosis", "fibrotic"),
    "Hernia": ("hernia",),
    "Infiltration": ("infiltrate", "infiltration", "opacity", "opacities"),
    "Mass": ("mass",),
    "Nodule": ("nodule", "nodular"),
    "Pleural_Thickening": ("pleural thickening",),
    "Pneumonia": ("pneumonia",),
    "Pneumothorax": ("pneumothorax",),
}
NEGATIONS = re.compile(r"\b(no|not|without|negative for|free of|absence of|resolved|clear of)\b")


def mentioned_findings(text):
    """Findings asserted (not negated) in text, sentence by sentence."""
    found = set()
    for sentence in re.split(r"[.\n;]+", text.lower()):
        for label, words in KEYWORDS.items():
            for word in words:
                match = re.search(rf"\b{re.escape(word)}", sentence)
                if match and not NEGATIONS.search(sentence[:match.start()]):
                    found.add(label)
    return found


def bleu4(reference, hypothesis):
    """Sentence BLEU-4 with add-one smoothing. Lexical overlap only."""
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    if not hyp:
        return 0.0
    log_precision = 0.0
    for n in range(1, 5):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        overlap = sum(min(c, r[g]) for g, c in h.items())
        log_precision += math.log((overlap + 1) / (sum(h.values()) + 1)) / 4
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(log_precision)


def score_case(reference, generated):
    parsed = parse_report(generated)
    ref, gen = mentioned_findings(reference), mentioned_findings(generated)
    return {
        "complete_sections": parsed.complete,
        "example_leak": bool(parsed.leaked_example_phrases),
        "reference_findings": sorted(ref), "generated_findings": sorted(gen),
        "hallucinated": sorted(gen - ref), "omitted": sorted(ref - gen),
        "finding_set_match": ref == gen,
        "bleu4": bleu4(reference, generated),
    }


def summarise(cases):
    n = len(cases)
    tp = Counter(); fp = Counter(); fn = Counter()
    for c in cases:
        ref, gen = set(c["reference_findings"]), set(c["generated_findings"])
        for label in NIH_LABELS:
            tp[label] += label in ref and label in gen
            fp[label] += label in gen and label not in ref
            fn[label] += label in ref and label not in gen
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    precision = TP / (TP + FP) if TP + FP else None
    recall = TP / (TP + FN) if TP + FN else None
    return {
        "cases": n,
        "complete_sections_rate": sum(c["complete_sections"] for c in cases) / n,
        "example_leak_rate": sum(c["example_leak"] for c in cases) / n,
        "finding_set_exact_match_rate": sum(c["finding_set_match"] for c in cases) / n,
        "finding_mention_micro_precision": precision,
        "finding_mention_micro_recall": recall,
        "cases_with_hallucinated_finding": sum(bool(c["hallucinated"]) for c in cases),
        "cases_with_omitted_finding": sum(bool(c["omitted"]) for c in cases),
        "mean_bleu4_lexical_only": sum(c["bleu4"] for c in cases) / n,
    }


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise ValueError(f"Duplicate id {row['id']} in {path}")
        by_id[row["id"]] = row
    return by_id


def compare(systems):
    """systems: {name: {id: row}} — all must cover exactly the same ids and references."""
    names = list(systems)
    ids = set(systems[names[0]])
    for name in names[1:]:
        if set(systems[name]) != ids:
            raise ValueError("Systems must be evaluated on exactly the same case ids")
        for i in ids:
            if systems[name][i]["reference"] != systems[names[0]][i]["reference"]:
                raise ValueError(f"Reference text differs for case {i}")
    result = {"note": __doc__.split("\n\n")[1].replace("\n", " "), "systems": {}, "failures": {}}
    for name in names:
        cases = {i: score_case(r["reference"], r["generated"]) for i, r in systems[name].items()}
        result["systems"][name] = summarise(list(cases.values()))
        worst = sorted(cases.items(), key=lambda kv: (len(kv[1]["hallucinated"]) +
                                                      len(kv[1]["omitted"])), reverse=True)[:5]
        result["failures"][name] = [{"id": i, **c, "generated": systems[name][i]["generated"]}
                                    for i, c in worst if c["hallucinated"] or c["omitted"]]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("systems", nargs="+", help="name=path.jsonl (e.g. baseline=b.jsonl)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    systems = {}
    for spec in args.systems:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Expected name=path, got {spec}")
        systems[name] = load(path)
    result = compare(systems)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["systems"], indent=2))


if __name__ == "__main__":
    main()
