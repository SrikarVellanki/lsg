"""Evaluate the clause classifiers on the held-out validation sentences.

    python -m scripts.evaluate                # prints a markdown report
    python -m scripts.evaluate --write results.md

It uses the SAME stratified split (seed 42) as ``scripts/train_classifier.py``, so the
validation sentences are ones GPT-2 never trained on. The keyword fallback is always evaluated
as a baseline; GPT-2 is evaluated too if a trained model exists in CLASSIFIER_DIR.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from app.config import get_settings
from app.core import ClauseClassifier
from app.corpus import LABELED_CLAUSES, LABELS
from scripts.train_classifier import stratified_split

NO_MODEL = "__keyword_baseline__"


def evaluate(classifier: ClauseClassifier, items) -> dict:
    """Accuracy, per-label accuracy and the most common mistakes for ``(label, sentence)`` pairs."""
    correct = Counter()
    total = Counter()
    mistakes = Counter()
    for label, sentence in items:
        predicted = classifier.classify(sentence).label
        total[label] += 1
        if predicted == label:
            correct[label] += 1
        else:
            mistakes[(label, predicted)] += 1
    n = sum(total.values())
    return {
        "backend": classifier.backend,
        "n": n,
        "accuracy": sum(correct.values()) / n if n else 0.0,
        "per_label": {l: (correct[l], total[l]) for l in LABELS if total[l]},
        "mistakes": mistakes.most_common(5),
    }


def format_report(results: dict[str, dict], n_train: int, n_val: int) -> str:
    lines = [
        "# Evaluation",
        "",
        f"Held-out validation set: {n_val} sentences ({n_train} used for training). "
        f"Random guessing over {len(LABELS)} labels would score about {1 / len(LABELS):.0%}.",
        "",
        "| Classifier | Accuracy on validation set |",
        "|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['accuracy']:.0%} ({round(r['accuracy'] * r['n'])}/{r['n']}) |")
    for name, r in results.items():
        lines += ["", f"## Per-label accuracy: {name}", "", "| Clause type | Correct |", "|---|---|"]
        lines += [f"| {l} | {c}/{t} |" for l, (c, t) in r["per_label"].items()]
        if r["mistakes"]:
            lines += ["", "Most common mistakes (true -> predicted):", ""]
            lines += [f"- {t} -> {p}: {c}x" for (t, p), c in r["mistakes"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=None, help="fine-tuned GPT-2 folder (default: CLASSIFIER_DIR)")
    parser.add_argument("--write", default=None, help="also write the report to this file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    train, val = stratified_split(LABELED_CLAUSES, args.val_fraction, args.seed)
    results = {"Keyword baseline": evaluate(ClauseClassifier(NO_MODEL), val)}

    model_dir = args.model_dir or get_settings().classifier_dir
    if os.path.isdir(model_dir):
        gpt2 = ClauseClassifier(model_dir)
        if gpt2.backend == "gpt2":
            results["Fine-tuned GPT-2"] = evaluate(gpt2, val)
        else:
            print(f"Could not load GPT-2 from {model_dir}: {gpt2.load_error}")
    else:
        print(f"No trained model in {model_dir}; run `python -m scripts.train_classifier` first. "
              "Showing the keyword baseline only.\n")

    report = format_report(results, len(train), len(val))
    print(report)
    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.write}")


if __name__ == "__main__":
    main()
