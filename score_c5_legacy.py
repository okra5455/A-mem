#!/usr/bin/env python3
"""Category-5 rescoring under the fixed and legacy (trap-answer) references.

By default the repository scores category-5 (adversarial) questions against
"Not mentioned in the conversation".  ``--legacy_c5_reference`` restores the
paper-era protocol, which used the adversarial (trap) answer as the reference;
commit b31a1dd states that this is "what the paper's numbers used".

This script recomputes both protocols from the predictions already stored in
``results_*.json`` files, so the two can be compared without re-running any
model.  It also re-derives ``adversarial_correct`` and verifies that the
recomputed fixed-reference metrics match the values stored in each file.

Only F1, BLEU-1 and exact match are computed (the metrics the paper reports).
They use the same helpers as ``utils.calculate_metrics``; pass ``--full`` to
call that function instead (slower: it also computes BERTScore, ROUGE, METEOR
and SBERT, and reloads the BERTScore model on every call).

Usage
-----
    python score_c5_legacy.py                     # every results_*.json in this directory
    python score_c5_legacy.py results_a.json ...  # any result files
    python score_c5_legacy.py --full              # all metrics, via utils.calculate_metrics

Writes ``c5_legacy_scores.json`` next to this script.  The input files are
never modified.
"""

import argparse
import glob
import json
import logging
import os
import warnings

# Quiet the per-call model-loading banners before importing utils/transformers.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from utils import calculate_bleu_scores, is_not_mentioned, simple_tokenize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ("f1", "bleu1", "exact_match")
NOT_MENTIONED = "Not mentioned in the conversation"
OUTPUT_PATH = os.path.join(HERE, "c5_legacy_scores.json")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarise(metrics_list):
    """Mean of the selected metrics over a list of metric dictionaries."""
    if not metrics_list:
        return {}
    return {key: mean([m[key] for m in metrics_list]) for key in METRICS}


def qa_scores(prediction, reference):
    """F1, BLEU-1 and exact match, computed exactly as ``utils.calculate_metrics``."""
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0, "bleu1": 0.0}

    exact_match = int(prediction.lower() == reference.lower())

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "f1": f1,
        "bleu1": calculate_bleu_scores(prediction, reference)["bleu1"],
    }


def score_file(path, full=False):
    """Return the fixed/legacy category-5 summary for one result file."""
    if full:
        from utils import calculate_metrics
        scorer = calculate_metrics
    else:
        scorer = qa_scores

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = [
        row for row in data.get("individual_results", [])
        if int(row.get("category", -1)) == 5
    ]

    fixed, legacy = [], []
    adversarial_correct = []
    counts = {"declines_only": 0, "gives_trap_only": 0, "both": 0, "neither": 0}
    max_diff = 0.0
    missing_trap = 0

    for row in rows:
        prediction = row.get("prediction", "")
        trap = row.get("adversarial_answer") or ""
        fixed_reference = row.get("reference") or NOT_MENTIONED

        # Fixed protocol: the correct answer is that the conversation does not say.
        metrics_fixed = scorer(prediction, fixed_reference)
        fixed.append(metrics_fixed)

        # Fidelity check: the file's stored metrics came from this same metric code.
        stored = row.get("metrics") or {}
        for key in METRICS:
            if key in stored:
                max_diff = max(max_diff, abs(metrics_fixed[key] - stored[key]))

        # Legacy protocol: score against the trap answer offered in the prompt.
        if trap:
            legacy.append(scorer(prediction, trap))
        else:
            missing_trap += 1

        adversarial_correct.append(is_not_mentioned(prediction, trap))

        text = str(prediction).lower()
        declines = ("not mentioned" in text) or ("no information available" in text)
        gives_trap = bool(trap) and (str(trap).lower().strip(" .") in text)
        if declines and gives_trap:
            counts["both"] += 1
        elif declines:
            counts["declines_only"] += 1
        elif gives_trap:
            counts["gives_trap_only"] += 1
        else:
            counts["neither"] += 1

    return {
        "n_c5": len(rows),
        "fixed": summarise(fixed),
        "legacy": summarise(legacy),
        "adversarial_correct_rate": mean(adversarial_correct),
        "prediction_breakdown": counts,
        "legacy_skipped_missing_trap": missing_trap,
        "recompute_max_abs_diff_vs_stored": max_diff,
    }


def print_table(results):
    header = (
        f"{'file':46s} {'n':>4}  {'F1_fix':>7} {'F1_leg':>7}  "
        f"{'B1_fix':>7} {'B1_leg':>7}  {'EM_fix':>7} {'EM_leg':>7}  {'adv_ok':>7}"
    )
    print(header)
    print("-" * len(header))
    for path, r in results.items():
        name = os.path.basename(path)
        f, l = r["fixed"], r["legacy"]
        print(
            f"{name:46s} {r['n_c5']:>4}  "
            f"{f.get('f1', float('nan')) * 100:7.2f} {l.get('f1', float('nan')) * 100:7.2f}  "
            f"{f.get('bleu1', float('nan')) * 100:7.2f} {l.get('bleu1', float('nan')) * 100:7.2f}  "
            f"{f.get('exact_match', float('nan')) * 100:7.2f} {l.get('exact_match', float('nan')) * 100:7.2f}  "
            f"{r['adversarial_correct_rate'] * 100:7.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Rescore category-5 questions under the fixed and legacy references."
    )
    parser.add_argument(
        "results",
        nargs="*",
        help="result JSON files (default: every results_*.json in this directory)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"where to write the JSON summary (default: {os.path.basename(OUTPUT_PATH)})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use utils.calculate_metrics (all metrics, much slower)",
    )
    args = parser.parse_args()

    paths = args.results or sorted(glob.glob(os.path.join(HERE, "results_*.json")))
    paths = [p for p in paths if os.path.exists(p)]

    results = {path: score_file(path, full=args.full) for path in paths}
    print_table(results)

    # Fidelity check: the fixed-protocol recomputation must equal the stored metrics.
    for path, r in results.items():
        diff = r["recompute_max_abs_diff_vs_stored"]
        status = "OK" if diff < 1e-6 else f"MISMATCH ({diff:.2e})"
        print(f"recompute check {os.path.basename(path):46s} {status}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "description": (
                    "Category-5 metrics recomputed from stored predictions under the default "
                    "(fixed) reference and the legacy trap-answer reference. See score_c5_legacy.py."
                ),
                "metrics": list(METRICS),
                "files": {os.path.basename(p): r for p, r in results.items()},
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
