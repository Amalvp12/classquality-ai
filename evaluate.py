"""Run the assessor over labelled examples and write an evaluation table.

Usage (Windows):  python evaluate.py labels.csv --provider local
labels.csv columns: recording_id, reference_label, transcript_path
  - reference_label: Poor/Weak, Average, Good, Excellent, or "Known quality issues"
  - recording_id is only a display name; the assessor never sees it, and nothing is
    inferred from folder or file names. Keep labels.csv and transcripts OUT of Git.
Gemini requires --allow-external (only if the organiser permits it).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from src.assessment import AssessmentError, assess, load_config
from src.transcript_processing import TranscriptError, parse_transcript
from src.validation import (accuracy_statement, case_row, confusion_matrix, normalize_label, rows_to_csv)
from src.scoring import LABELS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("--provider", default="local", choices=["local", "ollama", "gemini"])
    ap.add_argument("--allow-external", action="store_true")
    ap.add_argument("--out", default="eval_results/evaluation.csv")
    a = ap.parse_args()

    cfg, rows, refs, preds = load_config(), [], [], []
    with open(a.labels_csv, newline="", encoding="utf-8-sig") as f:
        for rec in csv.DictReader(f):
            rid, path = rec["recording_id"], rec["transcript_path"]
            try:
                with open(path, "rb") as t:
                    segs = parse_transcript(t.read(), path)
                report = assess(segs, a.provider, cfg, consent_external=a.allow_external)
            except (OSError, TranscriptError, AssessmentError) as exc:
                print(f"[skip] {rid}: {exc}", file=sys.stderr)
                rows.append({"recording_id": rid, "reference_label": rec["reference_label"],
                             "predicted_label": "ERROR", "match": "error", "limitations": str(exc)})
                continue
            row = case_row(rid, rec["reference_label"], report)
            rows.append(row)
            ref = normalize_label(rec["reference_label"])
            if ref:
                refs.append(ref)
                preds.append(row["predicted_label"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        f.write(rows_to_csv(rows))
    print(f"Wrote {a.out}")
    if refs:
        print(accuracy_statement(len(refs), refs, preds))
        cm = confusion_matrix(refs, preds)
        print("Confusion matrix (rows=reference, cols=predicted):")
        print("            " + " ".join(f"{p:>9}" for p in LABELS))
        for r in LABELS:
            print(f"{r:>11} " + " ".join(f"{cm[r][p]:>9}" for p in LABELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
