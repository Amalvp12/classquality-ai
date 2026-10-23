import pytest
from src.scoring import (DIMENSIONS, DimensionResult, ScoringError, apply_evidence_rules, classify,
                         compute_score, confidence, score_range, build_score_summary, dimension_to_100)


def all_scores(v):
    return {k: v for k in DIMENSIONS}


def test_weights_sum_to_one():
    assert sum(w for _, w in DIMENSIONS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("v,expected", [(1, 0), (2, 25), (3, 50), (4, 75), (5, 100)])
def test_uniform_scores(v, expected):
    assert compute_score(all_scores(v)) == pytest.approx(expected)


@pytest.mark.parametrize("score,label", [(0, "Poor"), (39.9, "Poor"), (40, "Average"), (59.9, "Average"),
                                         (60, "Good"), (79.9, "Good"), (80, "Excellent"), (100, "Excellent")])
def test_classification_boundaries(score, label):
    assert classify(score) == label


def test_invalid_inputs():
    with pytest.raises(ScoringError):
        classify(101)
    with pytest.raises(ScoringError):
        dimension_to_100(6)
    with pytest.raises(ScoringError):
        compute_score({"structure": 3})


def test_dimensions_are_independent():
    s = all_scores(3)
    s["interaction"] = 5
    assert compute_score(s) == pytest.approx(50 + 0.20 * 50)   # only interaction's weight moves


def test_evidence_rule_moves_unsupported_extremes_to_three():
    r = apply_evidence_rules(DimensionResult("structure", 5, "strong", "x", []))
    assert r.score == 3 and r.evidence_level == "none" and r.raw_score == 5 and "no verifiable" in r.note
    r = apply_evidence_rules(DimensionResult("structure", 1, "strong", "x", []))
    assert r.score == 3
    r = apply_evidence_rules(DimensionResult("structure", 4, "strong", "x", [{"quote": "q" * 20}]))
    assert r.score == 4 and r.evidence_level == "strong"
    r = apply_evidence_rules(DimensionResult("structure", 3, "limited", "x", []))
    assert r.score == 3 and r.evidence_level == "limited"


def test_range_and_summary():
    q = [{"quote": "a quote that is long"}]
    dims = [DimensionResult(k, 4, "strong", "e", q) for k in DIMENSIONS]
    dims[0] = apply_evidence_rules(DimensionResult("structure", 5, "strong", "e", []))
    lo, hi = score_range(dims)
    assert lo < hi
    summ = build_score_summary(dims, words=1000)
    assert summ["provisional"] and summ["evidence_coverage"] == pytest.approx(0.75)
    assert summ["confidence"] == "Limited"


def test_confidence_levels():
    assert confidence(0.9, 1000) == "Moderate"
    assert confidence(0.9, 300) == "Low"
    assert confidence(0.4, 2000) == "Low"
    assert confidence(0.6, 2000) == "Limited"
