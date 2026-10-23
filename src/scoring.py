"""Score calculation, classification and evidence-coverage logic.

Weights and boundaries are PROVISIONAL: the organiser's materials (challenge brief and
feedbacks.pdf) give four labels but no numeric boundaries or weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# key -> (display name, weight). Weights sum to 1.0.
DIMENSIONS: dict[str, tuple[str, float]] = {
    "structure": ("Lesson structure", 0.25),
    "clarity": ("Explanation clarity", 0.20),
    "interaction": ("Interaction", 0.20),
    "rapport": ("Encouragement and rapport", 0.20),
    "support": ("Learning support", 0.15),
}

# (minimum score, label), checked from the top. Provisional.
BOUNDARIES: list[tuple[float, str]] = [(80.0, "Excellent"), (60.0, "Good"), (40.0, "Average"), (0.0, "Poor")]
LABELS = ["Poor", "Average", "Good", "Excellent"]
EVIDENCE_LEVELS = ("strong", "limited", "none")


class ScoringError(ValueError):
    pass


@dataclass
class DimensionResult:
    key: str
    score: int                      # 1-5 after evidence rules
    evidence_level: str             # strong | limited | none
    explanation: str
    verified_quotes: list[dict] = field(default_factory=list)   # {"quote", "segment", "start"}
    rejected_quotes: int = 0
    raw_score: int | None = None    # model score before evidence rules
    note: str = ""

    @property
    def name(self) -> str:
        return DIMENSIONS[self.key][0]

    @property
    def insufficient(self) -> bool:
        return self.evidence_level == "none"


def dimension_to_100(score: int) -> float:
    if not 1 <= score <= 5:
        raise ScoringError(f"Dimension score must be 1-5, got {score!r}")
    return (score - 1) / 4 * 100


def compute_score(scores: dict[str, int]) -> float:
    """Weighted mean of the five dimension scores, mapped from 1-5 to 0-100."""
    missing = set(DIMENSIONS) - set(scores)
    if missing:
        raise ScoringError(f"Missing dimension scores: {sorted(missing)}")
    return sum(DIMENSIONS[k][1] * dimension_to_100(scores[k]) for k in DIMENSIONS)


def classify(score: float) -> str:
    if score < 0 or score > 100:
        raise ScoringError(f"Score out of range: {score}")
    for minimum, label in BOUNDARIES:
        if score >= minimum:
            return label
    return "Poor"


def apply_evidence_rules(result: DimensionResult) -> DimensionResult:
    """High/low scores need at least one verified quotation.

    A score other than 3 with no verified quote cannot be defended from the supplied
    transcript, so it is reported as 3 with evidence level 'none' (insufficient
    evidence). This is an evidence rule, not a penalty: it moves scores toward the
    midpoint in either direction.
    """
    result.raw_score = result.raw_score if result.raw_score is not None else result.score
    if result.evidence_level == "none" or (result.score != 3 and not result.verified_quotes):
        if result.score != 3:
            result.note = (
                f"Model score {result.score} had no verifiable supporting quotation; "
                "reported as 3 (insufficient evidence)."
            )
        result.score = 3
        result.evidence_level = "none"
    return result


def score_range(dims: list[DimensionResult]) -> tuple[float, float]:
    """Plausible range if every insufficient-evidence dimension were really 1 or 5."""
    low = {d.key: (1 if d.insufficient else d.score) for d in dims}
    high = {d.key: (5 if d.insufficient else d.score) for d in dims}
    return compute_score(low), compute_score(high)


def evidence_coverage(dims: list[DimensionResult]) -> float:
    """Share of total weight carried by dimensions that have verified evidence."""
    return sum(DIMENSIONS[d.key][1] for d in dims if not d.insufficient and d.verified_quotes)


def confidence(coverage: float, words: int, transcript_ok: bool = True) -> str:
    if not transcript_ok or coverage < 0.5 or words < 400:
        return "Low"
    if coverage >= 0.8 and words >= 800:
        return "Moderate"
    return "Limited"


def build_score_summary(dims: list[DimensionResult], words: int) -> dict:
    scores = {d.key: d.score for d in dims}
    score = compute_score(scores)
    low, high = score_range(dims)
    cov = evidence_coverage(dims)
    return {
        "score": round(score),
        "score_exact": round(score, 1),
        "category": classify(score),
        "range": [round(low), round(high)],
        "range_categories": [classify(low), classify(high)],
        "evidence_coverage": round(cov, 2),
        "confidence": confidence(cov, words),
        "provisional": True,
    }
