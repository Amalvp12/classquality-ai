# Architecture and data flow

```
upload (video?, transcript) -> app.py
  video  -> video_analysis.analyze_video  -> technical metrics (separate from score)
  transcript -> transcript_processing.parse_transcript -> Segment[] (text, start, end)
             -> validate_transcript (empty/short/repetitive/[inaudible]/duration mismatch)
             -> sample_transcript -> 5 Excerpt (opening, early, middle, late, ending)
  assessment.assess(provider)
     local  : regex cue baseline over the full transcript (offline)
     ollama / gemini : build_prompt(excerpts) -> model (temperature 0, seed 0, JSON)
              -> parse_model_response -> normalize_dimensions
  validation.verify_quotes   exact-substring check of every quote against the transcript
  scoring.apply_evidence_rules  unsupported non-3 scores -> 3 / "insufficient evidence"
  scoring.build_score_summary   weighted score, category, range, coverage, confidence
  -> report dict -> Streamlit tabs (assessment, evidence, video, limitations) / JSON export
```

## Sampling method
Transcripts up to 1,750 words: every segment is assigned to one of five equal stages by word
position (the model sees everything). Longer transcripts: five 350-word windows - the first
words, windows centred at 25%, 50%, 75%, and the last words - aligned to segment boundaries.
Only these excerpts are sent to an LLM provider. Filenames and folder names are never inputs.

## Reliability rules
- Prompt treats transcript text as data (basic prompt-injection resistance, not a guarantee).
- Missing/invalid model fields become "insufficient evidence"; all-invalid responses raise a clear error.
- API errors (key, model not found, rate limit, 5xx, timeout) map to readable messages; keys are never echoed.
- Quotes: whitespace-normalised, case-sensitive exact substring, >= 12 characters.
- External processing requires explicit consent (`ConsentError`).
