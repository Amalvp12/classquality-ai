# AI interaction history (summary)

Labelled honestly: *Reported* = stated in the project brief; *Done in this session* = performed and
observed; *Proposed* = designed but not tested on real data. Redact personal data before submitting
full exports; attach the raw chat exports from your AI assistants as well.

## Reported earlier work (from the brief, not re-verified)
- Built a Streamlit app (Python 3.12, OpenCV, NumPy, Gemini API, VTT/TXT upload) that sampled six
  transcript excerpts, scored five dimensions and showed brightness/sharpness separately.
- Observed inconsistent scores across versions (B558 Weak: 45/50/54/62; B491 Excellent: 47/59/65;
  one Good recording: 66). Diagnosed risk: plausible-sounding reports with wrong categories.

## Done in this session
1. Read the challenge PDF, `feedbacks.pdf` and the master prompt from the uploads.
2. Tried to reach the Google Drive dataset: no usable response from the sandbox (network
   restrictions), so **no recordings or real transcripts were inspected**.
3. Redesigned the assessment: evidence-gated scoring, exact-quote verification, sampling across
   five stages, consent-gated external processing, separate video metrics.
4. Implemented the app, modules, evaluation script and docs.
5. Ran 48 automated tests (pass), a smoke test of `evaluate.py` on synthetic text, and a headless
   Streamlit start/health check.

## Proposed, not yet tested
- Real-data validation with a calibration/held-out split; trying a local or organiser-approved LLM.
- Visual checks (face visible, background) as a distinct, explicitly permitted feature.
