"""Quote verification and evaluation helpers (confusion matrix, macro-F1, case table)."""
from __future__ import annotations

import csv
import io

from .scoring import LABELS
from .transcript_processing import Segment, locate_quote, normalize_ws

MIN_QUOTE_CHARS = 12   # shorter strings ("good", "ok") match almost anywhere and prove nothing


def verify_quotes(quotes, segments: list[Segment]) -> tuple[list[dict], int]:
    """Keep only quotations that are exact (whitespace-normalised) substrings of the transcript.

    Returns (verified, rejected_count). Verified items carry the segment index and time.
    Non-string, empty, too-short and duplicate quotes are rejected.
    """
    verified: list[dict] = []
    rejected = 0
    seen: set[str] = set()
    for q in quotes or []:
        if not isinstance(q, str):
            rejected += 1
            continue
        nq = normalize_ws(q)
        if len(nq) < MIN_QUOTE_CHARS or nq in seen:
            rejected += 1
            continue
        loc = locate_quote(nq, segments)
        if loc is None:
            rejected += 1
            continue
        seen.add(nq)
        verified.append({"quote": nq, **loc})
    return verified, rejected


# ---------- evaluation ----------

_LABEL_MAP = {
    "poor": "Poor", "weak": "Poor",
    "average": "Average",
    "good": "Good",
    "excellent": "Excellent",
}


def normalize_label(label: str | None) -> str | None:
    """Map organiser labels to the four output labels. 'Known quality issues' -> None."""
    if not label:
        return None
    return _LABEL_MAP.get(label.strip().lower())


def confusion_matrix(refs: list[str], preds: list[str]) -> dict[str, dict[str, int]]:
    if len(refs) != len(preds):
        raise ValueError("refs and preds must have the same length")
    m = {r: {p: 0 for p in LABELS} for r in LABELS}
    for r, p in zip(refs, preds):
        if r not in m or p not in m[r]:
            raise ValueError(f"Unknown label: {r!r} / {p!r}")
        m[r][p] += 1
    return m


def macro_f1(refs: list[str], preds: list[str]) -> float:
    """Macro-F1 over labels that appear in the references."""
    present = [lab for lab in LABELS if lab in refs]
    if not present:
        return 0.0
    f1s = []
    for lab in present:
        tp = sum(1 for r, p in zip(refs, preds) if r == lab and p == lab)
        fp = sum(1 for r, p in zip(refs, preds) if r != lab and p == lab)
        fn = sum(1 for r, p in zip(refs, preds) if r == lab and p != lab)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def accuracy_statement(n: int, refs: list[str], preds: list[str]) -> str:
    """Refuse to present a headline accuracy from too few labelled examples."""
    if n < 20:
        return (f"Only {n} labelled example(s): too few for a credible accuracy rate. "
                "See the case-by-case table.")
    acc = sum(r == p for r, p in zip(refs, preds)) / n
    return f"Exact-label agreement {acc:.0%} over {n} cases; macro-F1 {macro_f1(refs, preds):.2f}."


EVAL_COLUMNS = ["recording_id", "reference_label", "predicted_label", "score",
                "structure", "clarity", "interaction", "rapport", "support",
                "match", "evidence_notes", "limitations"]


def case_row(recording_id: str, reference: str | None, report: dict) -> dict:
    ref = normalize_label(reference)
    dims = {d["key"]: d["score"] for d in report["dimensions"]}
    pred = report["summary"]["category"]
    return {
        "recording_id": recording_id,
        "reference_label": reference or "",
        "predicted_label": pred,
        "score": report["summary"]["score"],
        **{k: dims.get(k, "") for k in ("structure", "clarity", "interaction", "rapport", "support")},
        "match": ("n/a (no four-way reference)" if ref is None else ("match" if ref == pred else "mismatch")),
        "evidence_notes": "; ".join(f"{d['name']}: {len(d['verified_quotes'])} verified quote(s)"
                                    for d in report["dimensions"]),
        "limitations": " | ".join(report.get("limitations", []))[:500],
    }


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EVAL_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
