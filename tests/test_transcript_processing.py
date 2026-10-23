import pytest
from src.transcript_processing import (TranscriptError, parse_transcript, sample_transcript,
                                       validate_transcript, full_text, locate_quote, total_words)

VTT = """WEBVTT

NOTE this is a comment

1
00:00:01.000 --> 00:00:04.000
<v Teacher>Good morning everyone, welcome to class.</v>

2
00:00:04.500 --> 00:00:08.000
Today we will learn about <c.yellow>the fork</c> &amp; the pin.

bad block with no timing

00:01:00.000 --> 00:01:03.000
Well done!
"""


def test_vtt_removes_markup_and_keeps_timestamps():
    segs = parse_transcript(VTT, "x.vtt")
    assert [s.text for s in segs] == [
        "Teacher: Good morning everyone, welcome to class.",
        "Today we will learn about the fork & the pin.",
        "Well done!"]
    assert segs[0].start == 1.0 and segs[2].start == 60.0
    assert "WEBVTT" not in full_text(segs) and "-->" not in full_text(segs)


def test_vtt_without_hours_and_rolling_duplicates():
    raw = "WEBVTT\n\n00:01.000 --> 00:02.000\nHello there\n\n00:02.000 --> 00:03.000\nHello there\n"
    segs = parse_transcript(raw, "a.vtt")
    assert len(segs) == 1 and segs[0].end == 3.0


def test_txt_with_and_without_timestamps():
    segs = parse_transcript("[00:00:05] Hello class\nHow are you today?\n", "a.txt")
    assert segs[0].start == 5 and segs[0].text == "Hello class"
    assert segs[1].start is None


def test_txt_long_paragraph_is_split():
    para = " ".join(f"Sentence number {i} is here." for i in range(80))
    assert len(parse_transcript(para, "a.txt")) > 1


@pytest.mark.parametrize("raw,name", [("", "a.txt"), ("   \n ", "a.vtt"), (b"", "a.vtt")])
def test_empty_transcripts_rejected(raw, name):
    with pytest.raises(TranscriptError):
        parse_transcript(raw, name)


def test_wrong_extension_and_no_text():
    with pytest.raises(TranscriptError):
        parse_transcript("hello", "a.docx")
    with pytest.raises(TranscriptError):
        parse_transcript("WEBVTT\n\nno cues here\n", "a.vtt")


def test_bytes_with_bom_and_cp1252():
    assert parse_transcript("\ufeffHello there".encode("utf-8"), "a.txt")[0].text == "Hello there"
    assert "café" in parse_transcript("café time".encode("cp1252"), "a.txt")[0].text


def _long_segments(n_words):
    lines = [f"[{i//60:02d}:{i%60:02d}:00] " + " ".join(f"w{i*10+j}" for j in range(10)) for i in range(n_words // 10)]
    return parse_transcript("\n".join(lines), "a.txt")


def test_validation_short_and_mismatch():
    short = parse_transcript("hello class today we learn", "a.txt")
    assert not validate_transcript(short).ok
    segs = parse_transcript("\n".join(f"[00:{i:02d}:00] " + " ".join(f"w{i}x{j}" for j in range(10)) for i in range(0, 50)), "a.txt")
    chk = validate_transcript(segs, video_duration=600)
    assert chk.ok and any("may not match" in w for w in chk.warnings)
    chk2 = validate_transcript(segs, video_duration=100000)
    assert any("covers only" in w for w in chk2.warnings)


def test_sampling_short_covers_everything_in_five_stages():
    segs = _long_segments(500)
    ex = sample_transcript(segs)
    assert [e.stage for e in ex] == ["opening", "early", "middle", "late", "ending"]
    assert sum(e.words for e in ex) == total_words(segs)


def test_sampling_long_has_opening_and_ending_windows():
    segs = _long_segments(6000)
    ex = sample_transcript(segs, window_words=300)
    assert [e.stage for e in ex] == ["opening", "early", "middle", "late", "ending"]
    assert ex[0].first_segment == 0 and ex[-1].last_segment == len(segs) - 1
    assert all(250 <= e.words <= 330 for e in ex)
    assert ex[1].start < ex[2].start < ex[3].start


def test_locate_quote():
    segs = parse_transcript(VTT, "x.vtt")
    assert locate_quote("welcome   to class", segs)["start"] == 1.0
    assert locate_quote("Well done!", segs)["segment"] == 2
    assert locate_quote("never said this", segs) is None
