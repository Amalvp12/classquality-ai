import json
import pytest
from src import assessment
from src.assessment import (AssessmentError, ConsentError, Config, assess, load_config, normalize_dimensions,
                            parse_model_response)
from src.transcript_processing import parse_transcript
from src.validation import (accuracy_statement, case_row, confusion_matrix, macro_f1, normalize_label,
                            rows_to_csv, verify_quotes)
from src.video_analysis import VideoError, analyze_video

TEXT = ("Good morning everyone, welcome to class. Today we will learn about the fork. "
        "For example, look at this board. Can you tell me which piece is attacked? "
        "That's right, well done! Your homework is to practice three puzzles. See you next class. ") * 8
SEGS = parse_transcript(TEXT, "a.txt")


def test_verify_quotes_exact_only():
    ok, rej = verify_quotes(["welcome to class", "Today we will learn about the fork.",
                             "invented sentence that never occurs", "tiny", 5, None, "welcome to class"], SEGS)
    assert [v["quote"] for v in ok] == ["welcome to class", "Today we will learn about the fork."]
    assert rej == 5   # invented, too short, int, None, duplicate
    ok2, _ = verify_quotes(["Good morning everyone,   welcome to class."], SEGS)
    assert len(ok2) == 1 and ok2[0]["segment"] == 0


def test_verify_quotes_case_sensitive():
    ok, rej = verify_quotes(["good morning everyone, welcome to class."], SEGS)
    assert ok == [] and rej == 1


def good_response(**over):
    q = "Today we will learn about the fork."
    d = {k: {"score": 4, "evidence_level": "strong", "explanation": "e", "quotes": [q]}
         for k in ("structure", "clarity", "interaction", "rapport", "support")}
    d.update(over)
    return {"dimensions": d, "strengths": ["s"], "improvements": ["i"], "limitations": []}


def test_parse_model_response_variants():
    assert parse_model_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_model_response('Here: {"a": 1} thanks') == {"a": 1}
    for bad in ("", None, "no json", "{broken", "[1,2]"):
        with pytest.raises(AssessmentError):
            parse_model_response(bad)


def test_normalize_handles_missing_and_invalid_fields():
    resp = good_response(clarity={"score": "high"}, interaction=None,
                         rapport={"score": 9, "explanation": "x", "quotes": []},
                         support={"score": 5, "explanation": "", "quotes": ["fabricated quote not in text"]})
    del resp["dimensions"]["structure"]
    dims, notes = normalize_dimensions(resp, SEGS)
    by = {d.key: d for d in dims}
    assert all(by[k].score == 3 and by[k].insufficient for k in ("structure", "clarity", "interaction", "rapport"))
    assert by["support"].score == 3 and by["support"].raw_score == 5 and by["support"].rejected_quotes == 1
    assert len(notes) >= 4


def test_normalize_all_invalid_raises():
    with pytest.raises(AssessmentError):
        normalize_dimensions({"dimensions": {}}, SEGS)
    with pytest.raises(AssessmentError):
        normalize_dimensions({"dimensions": []}, SEGS)


def test_gemini_requires_consent_and_key(monkeypatch):
    with pytest.raises(ConsentError):
        assess(SEGS, "gemini", Config(gemini_api_key="x"), consent_external=False)
    with pytest.raises(AssessmentError, match="GEMINI_API_KEY"):
        assess(SEGS, "gemini", Config(gemini_api_key=""), consent_external=True)
    with pytest.raises(ConsentError):
        assess(SEGS, "ollama", Config(ollama_url="http://example.com:11434"))


def test_unknown_provider_and_short_transcript():
    with pytest.raises(AssessmentError):
        assess(SEGS, "nope", Config())
    with pytest.raises(AssessmentError, match="too short"):
        assess(parse_transcript("hello class", "a.txt"), "local", Config())


def test_full_pipeline_with_mocked_model(monkeypatch):
    monkeypatch.setattr(assessment, "_call_gemini", lambda prompt, cfg: json.dumps(good_response()))
    r = assess(SEGS, "gemini", Config(gemini_api_key="k"), consent_external=True)
    assert r["summary"]["score"] == 75 and r["summary"]["category"] == "Good"
    assert r["sent_to_external_service"] and all(d["verified_quotes"] for d in r["dimensions"])


def test_pipeline_malformed_model_output(monkeypatch):
    monkeypatch.setattr(assessment, "_call_gemini", lambda p, c: "I cannot do that")
    with pytest.raises(AssessmentError):
        assess(SEGS, "gemini", Config(gemini_api_key="k"), consent_external=True)


def test_prompt_contains_no_filename_and_marks_excerpts():
    from src.transcript_processing import sample_transcript
    p = assessment.build_prompt(sample_transcript(SEGS))
    assert "E1" in p and "opening" in p and ".txt" not in p


def test_local_baseline_quotes_are_verified_and_never_below_three():
    r = assess(SEGS, "local", Config())
    assert not r["sent_to_external_service"]
    assert all(d["score"] >= 3 for d in r["dimensions"])
    assert any(d["verified_quotes"] for d in r["dimensions"])
    plain = parse_transcript(("The cat sat on the mat and looked at the window. " * 40), "a.txt")
    r2 = assess(plain, "local", Config())
    assert r2["summary"]["category"] == "Average" and r2["summary"]["confidence"] == "Low"
    assert r2["summary"]["range"][0] < r2["summary"]["range"][1]


def test_load_config_from_env_dict():
    c = load_config({"GEMINI_API_KEY": " abc ", "GEMINI_MODEL": "m1"})
    assert c.gemini_api_key == "abc" and c.gemini_model == "m1" and c.ollama_url.startswith("http://localhost")


def test_labels_confusion_and_f1():
    assert normalize_label("Weak") == "Poor" and normalize_label("Known quality issues") is None
    refs, preds = ["Poor", "Good", "Excellent", "Good"], ["Good", "Good", "Average", "Good"]
    cm = confusion_matrix(refs, preds)
    assert cm["Poor"]["Good"] == 1 and cm["Good"]["Good"] == 2
    assert 0 <= macro_f1(refs, preds) <= 1
    assert macro_f1(["Good", "Good"], ["Good", "Good"]) == 1.0
    assert "too few" in accuracy_statement(4, refs, preds)
    with pytest.raises(ValueError):
        confusion_matrix(["Poor"], ["Bad"])


def test_case_row_and_csv():
    rep = assess(SEGS, "local", Config())
    row = case_row("R1", "Weak", rep)
    assert row["match"] in ("match", "mismatch") and row["reference_label"] == "Weak"
    assert case_row("R2", "Known quality issues", rep)["match"].startswith("n/a")
    assert rows_to_csv([row]).splitlines()[0].startswith("recording_id")


def test_video_errors(tmp_path):
    with pytest.raises(VideoError):
        analyze_video(str(tmp_path / "missing.mp4"))
    empty = tmp_path / "e.mp4"; empty.write_bytes(b"")
    with pytest.raises(VideoError):
        analyze_video(str(empty))
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video" * 100)
    with pytest.raises(VideoError):
        analyze_video(str(bad))
    wrong = tmp_path / "x.mkv"; wrong.write_bytes(b"123")
    with pytest.raises(VideoError, match="Unsupported"):
        analyze_video(str(wrong))


def test_video_synthetic(tmp_path):
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    path = str(tmp_path / "s.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 encoder available")
    for i in range(50):
        w.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    w.release()
    info = analyze_video(path, max_frames=10)
    assert info["frames_sampled"] >= 5 and abs(info["duration_seconds"] - 5.0) < 0.5
    assert 115 <= info["mean_brightness"] <= 125 and info["mean_sharpness"] == 0.0
