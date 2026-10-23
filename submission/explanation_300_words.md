# ClassQuality AI - approach (under 300 words)

**Problem.** Rate an online children's class as Poor, Average, Good or Excellent, with reasons a
reviewer can check. An earlier version of my app produced fluent reports but unstable scores and
wrong categories.

**Approach.** I redesigned around evidence. The transcript is parsed (VTT/TXT) and sampled across
the opening, early, middle, late and ending of the lesson. A model scores five teaching dimensions
(structure, clarity, interaction, rapport, learning support) from 1-5 at temperature 0, and must
quote the transcript. Every quote is verified as an exact substring; unverifiable quotes are
dropped, and a score with no verified evidence is reported as 3, "insufficient evidence". The
final score is a weighted mean with provisional boundaries, shown with a plausible range,
confidence and limitations. Video brightness/sharpness are shown separately and never affect the score.

**Privacy.** Recordings may show identifiable children. Videos are never uploaded to an AI
service. The default engine is offline. Sending transcript excerpts to Gemini is blocked until
the user confirms the organiser permits it. Secrets stay in a git-ignored `.env`.

**What worked.** Quote verification, evidence gating and input validation are covered by 48
passing automated tests.

**What did not / limitations.** I could not access the Drive dataset from my build environment,
so the app has **not** been validated on real recordings, and I report no accuracy figure.
Weights and boundaries are my own, since the organiser gave none. The offline baseline is only a
keyword matcher and cannot detect weak teaching. The app reads only transcripts: it cannot judge
camera visibility, background, audio or tone, which the organiser notes matter. Transcript gaps
are treated as limits, not poor teaching.
