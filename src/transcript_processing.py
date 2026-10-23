"""Transcript parsing, validation, sampling and quote location.

Supports WebVTT (.vtt) and plain text (.txt, optionally with [hh:mm:ss] stamps).
All text the rest of the app works with comes from `full_text(segments)`, so every
quotation can be traced back to a segment (and a timestamp when one exists).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

MIN_WORDS_ERROR = 80      # below this there is not enough text to assess anything
MIN_WORDS_WARNING = 400   # below this the assessment is flagged as thin
WINDOW_WORDS = 350        # words per excerpt for long transcripts
LONG_SPLIT_WORDS = 120    # very long paragraphs are split into sentence chunks


class TranscriptError(ValueError):
    """Raised when a transcript cannot be used at all."""


@dataclass
class Segment:
    index: int
    start: float | None
    end: float | None
    text: str


@dataclass
class Excerpt:
    id: str
    stage: str
    start: float | None
    end: float | None
    text: str
    first_segment: int
    last_segment: int

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class TranscriptCheck:
    words: int
    segments: int
    has_timestamps: bool
    duration_covered: float | None
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


_TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_CUE_RE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}")
_TXT_STAMP_RE = re.compile(
    r"^\s*[\[(]?(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d{1,3})?[\])]?\s*[-–:]?\s*(.*)$"
)
_VOICE_RE = re.compile(r"<v(?:\.[^ >]+)?\s+([^>]+)>")
_TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int((ms or "0").ljust(3, "0")) / 1000


def decode_bytes(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("cp1252", errors="replace")


def _clean_cue_text(lines: list[str]) -> str:
    text = " ".join(lines)
    voice = _VOICE_RE.search(text)
    text = _VOICE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if voice and text:
        text = f"{voice.group(1).strip()}: {text}"
    return text


def parse_vtt(text: str) -> list[Segment]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.split("\n")
        if lines and lines[0].strip().startswith("WEBVTT"):
            lines = lines[1:]
        lines = [ln for ln in lines if ln.strip()]
        if not lines or lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        timing_idx = next((i for i, ln in enumerate(lines) if _CUE_RE.match(ln)), None)
        if timing_idx is None:
            continue  # malformed block without a timing line
        m = _CUE_RE.match(lines[timing_idx])
        start = _to_seconds(*m.groups()[0:4])
        end = _to_seconds(*m.groups()[4:8])
        cue_text = _clean_cue_text(lines[timing_idx + 1:])
        if cue_text:
            segments.append(Segment(len(segments), start, end, cue_text))
    return segments


def parse_txt(text: str) -> list[Segment]:
    segments: list[Segment] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        start = None
        m = _TXT_STAMP_RE.match(line)
        if m and m.group(4):
            start = _to_seconds(m.group(1), m.group(2), m.group(3), None)
            line = m.group(4).strip()
        line = re.sub(r"\s+", " ", _TAG_RE.sub("", html.unescape(line))).strip()
        if line:
            segments.append(Segment(len(segments), start, None, line))
    return segments


def _split_long(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        if len(seg.text.split()) <= LONG_SPLIT_WORDS:
            out.append(seg)
            continue
        buf: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", seg.text):
            buf.append(sentence)
            if len(" ".join(buf).split()) >= 60:
                out.append(Segment(0, seg.start, seg.end, " ".join(buf)))
                buf = []
        if buf:
            out.append(Segment(0, seg.start, seg.end, " ".join(buf)))
    return out


def _dedupe_and_index(segments: list[Segment]) -> list[Segment]:
    result: list[Segment] = []
    for seg in segments:
        if result and result[-1].text == seg.text:
            if seg.end is not None:
                result[-1].end = seg.end
            continue  # rolling-caption repeat
        result.append(seg)
    for i, seg in enumerate(result):
        seg.index = i
    return result


def parse_transcript(raw: bytes | str, filename: str = "") -> list[Segment]:
    """Parse a .vtt or .txt transcript. Raises TranscriptError for unusable input."""
    name = (filename or "").lower()
    if name and not name.endswith((".vtt", ".txt")):
        raise TranscriptError("Transcript must be a .vtt or .txt file.")
    if raw is None or len(raw) == 0:
        raise TranscriptError("The transcript file is empty.")
    text = decode_bytes(raw)
    if not text.strip():
        raise TranscriptError("The transcript file is empty.")
    is_vtt = name.endswith(".vtt") or "-->" in text[:5000]
    segments = parse_vtt(text) if is_vtt else parse_txt(text)
    segments = _dedupe_and_index(_split_long(segments))
    if not segments:
        raise TranscriptError(
            "No spoken text could be extracted. Check that the file is a valid "
            "WebVTT/TXT transcript."
        )
    return segments


def full_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)


def total_words(segments: list[Segment]) -> int:
    return len(full_text(segments).split())


_ARTIFACT_RE = re.compile(r"\[(?:inaudible|music|noise|silence|crosstalk)[^\]]*\]", re.I)


def validate_transcript(segments: list[Segment], video_duration: float | None = None) -> TranscriptCheck:
    errors: list[str] = []
    warnings: list[str] = []
    words = total_words(segments)
    ends = [s.end if s.end is not None else s.start for s in segments
            if (s.end if s.end is not None else s.start) is not None]
    has_ts = bool(ends)
    covered = max(ends) if ends else None

    if words == 0:
        errors.append("Transcript contains no words.")
    elif words < MIN_WORDS_ERROR:
        errors.append(
            f"Transcript has only {words} words - too short to assess a class "
            f"(minimum {MIN_WORDS_ERROR})."
        )
    elif words < MIN_WORDS_WARNING:
        warnings.append(f"Transcript is short ({words} words); the assessment will be thin.")

    if len(segments) > 10 and len({s.text for s in segments}) / len(segments) < 0.3:
        warnings.append("Transcript is highly repetitive; it may be malformed or a caption loop.")

    artifacts = sum(1 for s in segments if _ARTIFACT_RE.search(s.text))
    if segments and artifacts / len(segments) > 0.05:
        warnings.append(
            "Many segments contain markers such as [inaudible]; gaps are treated as a "
            "limitation, not as poor teaching."
        )

    if has_ts and video_duration and video_duration > 0 and covered is not None:
        if covered > video_duration * 1.2 + 30:
            warnings.append(
                f"Transcript runs to {covered/60:.1f} min but the video is "
                f"{video_duration/60:.1f} min - the files may not match."
            )
        elif covered < video_duration * 0.5:
            warnings.append(
                f"Transcript covers only {covered/60:.1f} of {video_duration/60:.1f} min of "
                "video - evidence from the missing part is unavailable."
            )
    return TranscriptCheck(words, len(segments), has_ts, covered, errors, warnings)


STAGES = ["opening", "early", "middle", "late", "ending"]


def sample_transcript(segments: list[Segment], window_words: int = WINDOW_WORDS) -> list[Excerpt]:
    """Return excerpts covering the opening, early, middle, late and ending of the lesson.

    * Transcripts of <= 5 windows: every segment is assigned to one of five equal
      stages by word position, so the model sees the whole transcript.
    * Longer transcripts: five windows of `window_words` words - the first words,
      centred windows at 25%, 50% and 75%, and the last words - aligned to segment
      boundaries. The opening and the ending always have a window.
    """
    offsets, running = [], 0
    for seg in segments:
        offsets.append(running)
        running += len(seg.text.split())
    total = running
    if total == 0:
        return []

    groups: list[list[int]] = [[] for _ in STAGES]
    if total <= window_words * len(STAGES):
        for i, off in enumerate(offsets):
            groups[min(len(STAGES) - 1, off * len(STAGES) // total)].append(i)
    else:
        centres = [None, 0.25, 0.50, 0.75, None]
        for k, c in enumerate(centres):
            if k == 0:
                start_w = 0
            elif k == len(STAGES) - 1:
                start_w = total - window_words
            else:
                start_w = int(total * c - window_words / 2)
            start_w = max(0, min(start_w, total - window_words))
            first = max(i for i, off in enumerate(offsets) if off <= start_w)
            acc, i = 0, first
            while i < len(segments) and acc < window_words:
                groups[k].append(i)
                acc += len(segments[i].text.split())
                i += 1

    excerpts: list[Excerpt] = []
    for k, idxs in enumerate(groups):
        if not idxs:
            continue
        segs = [segments[i] for i in idxs]
        starts = [s.start for s in segs if s.start is not None]
        ends = [s.end if s.end is not None else s.start for s in segs
                if (s.end if s.end is not None else s.start) is not None]
        excerpts.append(Excerpt(
            id=f"E{len(excerpts) + 1}", stage=STAGES[k],
            start=min(starts) if starts else None, end=max(ends) if ends else None,
            text=" ".join(s.text for s in segs),
            first_segment=idxs[0], last_segment=idxs[-1],
        ))
    return excerpts


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, segments: list[Segment]) -> dict | None:
    """Find an exact (whitespace-normalised) quote. Returns segment index and time, or None."""
    q = normalize_ws(quote)
    if not q:
        return None
    text = full_text(segments)
    pos = text.find(q)
    if pos < 0:
        return None
    running = 0
    for seg in segments:
        end = running + len(seg.text)
        if pos < end + 1:
            return {"segment": seg.index, "start": seg.start}
        running = end + 1
    return {"segment": segments[-1].index, "start": segments[-1].start}
