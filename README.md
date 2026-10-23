# ClassQuality AI

Evidence-based, **preliminary** quality assessment of online class recordings, built for the
Humaniti AI 24-Hour Build Challenge. Upload a class video (optional) and its matching transcript
(VTT/TXT); get a 0-100 score, a category (Poor / Average / Good / Excellent), five dimension
scores with explanations, strengths, improvements, **exact verified transcript quotations**,
confidence/evidence coverage, limitations, and separate video technical measurements.

> **Status:** unit-tested on synthetic data only. It has **not** been validated against the
> organiser's recordings (see `docs/evaluation.md`). It is **not deployed** until you deploy it.

## What it actually does
- Parses VTT/TXT transcripts, samples five stages of the lesson (opening, early, middle, late, ending).
- Scores five dimensions (structure, clarity, interaction, rapport, learning support) from 1-5.
- Verifies every quotation as an exact substring of your transcript; unverifiable quotes are dropped.
- A dimension scored other than 3 without a verified quote is reported as 3 ("insufficient evidence").
- Computes a weighted score with **provisional** weights/boundaries and shows a plausible range.
- Measures video duration, frames sampled, mean brightness, sharpness (OpenCV) - shown separately,
  never part of the teaching score. **No audio or visual understanding is claimed or implemented.**

## Engines (privacy first)
| Engine | Where data goes | Notes |
|---|---|---|
| `local` (default) | nowhere | Keyword baseline. Cannot detect weak teaching (never scores a dimension below 3). Pipeline test only. |
| `ollama` | your machine | Local LLM; you must run Ollama and pull a model. |
| `gemini` | **Google API** | Sends sampled transcript excerpts (never video). Blocked until the user ticks a consent box. Use only if the organiser permits it. |

Recordings may show identifiable children. Do not upload recordings/transcripts to GitHub, and
do not host a public app that stores them. Uploaded videos are written to a temp file for
measurement and deleted immediately; nothing is persisted by the app.

## Setup on Windows (Python 3.12)
```bat
cd classquality-ai
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
(PowerShell: `.venv\Scripts\Activate.ps1`.) Edit `.env` and set `GEMINI_API_KEY` only if you will use
Gemini. `.env` is git-ignored; never share or print it.

## Run
```bat
streamlit run app.py
```
## Test
```bat
python -m pytest -q
```
## Check available Gemini models (do not assume the default ID exists)
```bat
python -c "from src.assessment import load_config, list_gemini_models; print(list_gemini_models(load_config()))"
```
Then set `GEMINI_MODEL` in `.env`.

## Evaluate on labelled examples
Create `labels.csv` (kept out of Git) with columns `recording_id,reference_label,transcript_path`:
```bat
python evaluate.py labels.csv --provider local
python evaluate.py labels.csv --provider gemini --allow-external
```
Writes `eval_results/evaluation.csv` and prints a confusion matrix. A headline accuracy is
suppressed below 20 labelled cases.

## Deployment (not done by this project)
1. Push the repo (no recordings, no `.env`) to GitHub.
2. Streamlit Community Cloud -> New app -> select repo, main file `app.py`, Python 3.12.
3. App settings -> Secrets: `GEMINI_API_KEY = "..."` and optionally `GEMINI_MODEL`. Never commit them.
4. Before using Gemini on a hosted app, confirm the organiser permits external processing, and
   remember uploaded recordings reach the host's server. Consider access restrictions.
5. Open the hosted URL, test with a sample, and only then submit it.

## Project layout
`app.py` UI · `src/transcript_processing.py` parsing/sampling · `src/assessment.py` providers,
prompt, response validation · `src/scoring.py` weights/boundaries/evidence rules ·
`src/validation.py` quote verification and evaluation metrics · `src/video_analysis.py` frame
measurements · `evaluate.py` batch evaluation · `tests/` · `docs/` · `submission/`.

## Known limitations
Transcript only; no teacher/student speaker labels in many transcripts; automatic captions drop
words; keyword baseline is weak; provisional weights/boundaries; unvalidated on real data.
See `docs/evaluation.md`.
