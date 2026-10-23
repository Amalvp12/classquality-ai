# Evaluation

## Rubric (1-5 per dimension)
5 repeated concrete positive evidence · 4 clear positive evidence · 3 mixed **or insufficient**
evidence · 2 clear negative evidence · 1 repeated concrete negative evidence. Dimensions:
lesson structure, explanation clarity, interaction, encouragement/rapport, learning support.
These follow the organiser's feedback themes (intro/conclusion, rapport, individual questions,
resources, homework). They are criteria to assess, not hard-coded answers.

## Score formula (PROVISIONAL)
`score = sum(weight_i * (dim_i - 1) / 4 * 100)` with weights structure 0.25, clarity 0.20,
interaction 0.20, rapport 0.20, support 0.15. Boundaries: Poor < 40 <= Average < 60 <= Good < 80
<= Excellent. Rationale: a uniform 3 (mixed evidence) maps to 50 = Average; uniform 4 = 75 = Good;
Excellent needs mostly 5s. The organiser's PDFs specify labels but **no numeric weights or
boundaries**, so these are my design choices, not official, and were not tuned to any recording.

## Evidence rule
A non-3 score with no verified quote is reported as 3 / "none". It is symmetric (pulls high and
low scores to the midpoint) and is not tuned to any label. A plausible score range treats
insufficient dimensions as anywhere from 1 to 5.

## Reference cases from `feedbacks.pdf` (as read from the uploaded file)
Predictions were **not run**: the Drive recordings could not be accessed from the build
environment, so no real transcript has been scored.

| Recording | Organiser label | Organiser notes (summary) | Predicted | Score |
|---|---|---|---|---|
| B491 | Excellent | one-to-one interaction, background, topic videos, homework | not run | - |
| B472 | Excellent | rapport, everyone considered, individual questions, homework | not run | - |
| B552 | Good | good intro, 8x8 background, homework; needs pleasanter interaction | not run | - |
| B435 (25/07) | Good | good intro, everyone considered; should give homework | not run | - |
| B516 | Average | background kept, individual attention, puzzles, homework; some clarity issues | not run | - |
| B435 (28/08) | Average | needs more energetic/pleasant interaction; no homework | not run | - |
| B558 | Weak | needs better intro/conclusion, time for new joiners; needs more pleasant interaction | not run | - |
| B482 | Weak | no rapport before revision, no self-intro time, camera off at times | not run | - |
| B392 | Known quality issues | no background, needs happier interaction | n/a (no 4-way label) | - |
| B441 | Known quality issues | face not visible, no rapport building | n/a (no 4-way label) | - |

Data oddities in the PDF: "B435" appears twice with different dates and labels, and one date
reads "02/09/203326". Confirm IDs before building `labels.csv`. Several organiser notes concern
background, face visibility and camera state - **visual** signals this app does not assess.

## Earlier observations (from the project brief, not a controlled benchmark)
Previous app versions scored B558 (Weak) 45-62, once Good, and B491 (Excellent) 47-65, once
Average. Different code versions were involved, so these are anecdotes, not a model comparison.

## Validation plan
1. Obtain transcripts for the labelled recordings (matching VTT/TXT).
2. Split into a calibration set (to adjust the rubric wording) and a held-out set (final report).
3. Run `evaluate.py`; report the case table; add a confusion matrix and macro-F1 only with enough cases
   (the tool suppresses headline accuracy below 20).
4. Report mismatches honestly; do not add per-recording rules.

## Limitations
Transcript-only; the local baseline cannot find weak teaching; LLM scores can vary and may
over-credit fluent speech; ASR errors and unlabeled speakers; provisional weights; no real-data
validation yet; with 8 usable labels in the PDF, accuracy cannot be credible.
