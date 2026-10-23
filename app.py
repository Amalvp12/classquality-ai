"""ClassQuality AI - Streamlit app. Run with: streamlit run app.py"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace

import streamlit as st

from src.assessment import (AssessmentError, ConsentError, PROVIDERS, assess, load_config)
from src.transcript_processing import TranscriptError, format_time, parse_transcript
from src.video_analysis import VideoError, analyze_video

st.set_page_config(page_title="ClassQuality AI", page_icon="🎓", layout="wide")

# Hosted deployments: copy secrets into the environment (never displayed).
try:
    for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "OLLAMA_URL", "OLLAMA_MODEL"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally - that is fine

cfg = load_config()
st.session_state.setdefault("results", [])

PROVIDER_LABELS = {
    "local": "Local keyword baseline (offline, no data leaves this machine)",
    "ollama": "Local LLM via Ollama (offline)",
    "gemini": "Google Gemini (sends transcript excerpts to Google)",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Assessment engine")
    provider = st.radio("Provider", PROVIDERS, format_func=PROVIDER_LABELS.get)
    consent = False
    if provider == "gemini":
        cfg = replace(cfg, gemini_model=st.text_input("Gemini model ID", cfg.gemini_model))
        st.warning("Sampled transcript excerpts (which may contain identifiable children's speech) "
                   "are sent to Google's API. Videos are never uploaded.")
        consent = st.checkbox("The Humaniti AI organiser has permitted this processing.")
        st.caption("API key: " + ("configured ✔" if cfg.gemini_api_key else "not set ✘ (see README)"))
    elif provider == "ollama":
        cfg = replace(cfg, ollama_url=st.text_input("Ollama URL", cfg.ollama_url),
                      ollama_model=st.text_input("Ollama model", cfg.ollama_model))
        st.caption("Local models are not verified by this app; pull the model first.")
        consent = st.checkbox("The Ollama URL is a remote server and the organiser permits this.",
                              help="Only needed if the URL is not localhost.")
    else:
        st.info("The baseline matches keyword cues. It cannot detect weak teaching - use it to test "
                "the pipeline offline, not as a quality judgement.")
    st.divider()
    st.caption("Scores, weights and category boundaries are provisional. Only the transcript is "
               "assessed; video measurements are shown separately.")

# --------------------------------------------------------------------- main
st.title("🎓 ClassQuality AI")
st.write("Upload a class recording and its matching transcript to get an evidence-based, "
         "preliminary quality assessment with verified transcript quotations.")

with st.form("upload", clear_on_submit=False):
    label = st.text_input("Label for this recording (optional - do not use children's names)")
    c1, c2 = st.columns(2)
    video_file = c1.file_uploader("Class recording (MP4, MOV, AVI) - optional for technical checks",
                                  type=["mp4", "mov", "avi"])
    transcript_file = c2.file_uploader("Matching transcript (VTT or TXT) - required", type=["vtt", "txt"])
    submitted = st.form_submit_button("Assess recording", type="primary")

if submitted:
    if transcript_file is None:
        st.error("Please upload the matching transcript (.vtt or .txt).")
    else:
        name = label.strip() or f"Recording {len(st.session_state.results) + 1}"
        video_info, video_error, duration = None, None, None
        tmp_path = None
        try:
            if video_file is not None:
                suffix = os.path.splitext(video_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.getbuffer())
                    tmp_path = tmp.name
                with st.spinner("Measuring video frames..."):
                    try:
                        video_info = analyze_video(tmp_path)
                        duration = video_info["duration_seconds"]
                    except VideoError as exc:
                        video_error = str(exc)
            segments = parse_transcript(transcript_file.getvalue(), transcript_file.name)
            with st.spinner("Assessing transcript..."):
                report = assess(segments, provider, cfg, consent_external=consent, video_duration=duration)
            report.update(name=name, video=video_info, video_error=video_error)
            st.session_state.results.append(report)
            st.success(f"Assessed '{name}'.")
        except TranscriptError as exc:
            st.error(f"Transcript problem: {exc}")
        except ConsentError as exc:
            st.error(str(exc))
        except AssessmentError as exc:
            st.error(f"Assessment failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# ------------------------------------------------------------------ results
results = st.session_state.results
if results:
    st.subheader("Session results")
    st.dataframe([{"Recording": r["name"], "Score /100": r["summary"]["score"],
                   "Category": r["summary"]["category"], "Confidence": r["summary"]["confidence"],
                   "Engine": r["provider"]} for r in results], use_container_width=True, hide_index=True)
    if st.button("Clear results"):
        st.session_state.results = []
        st.rerun()

for i, r in enumerate(reversed(results)):
    s = r["summary"]
    with st.expander(f"{r['name']} - {s['score']}/100 ({s['category']})", expanded=(i == 0)):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Preliminary score", f"{s['score']}/100")
        m2.metric("Category", s["category"])
        m3.metric("Confidence", s["confidence"])
        m4.metric("Evidence coverage", f"{s['evidence_coverage']:.0%}")
        if s["range"][0] != s["range"][1]:
            st.caption(f"Plausible range given insufficient evidence in some dimensions: "
                       f"{s['range'][0]}-{s['range'][1]} ({s['range_categories'][0]} to {s['range_categories'][1]}). "
                       "Weights and boundaries are provisional.")
        if r["provider"] == "local":
            st.warning("Local keyword baseline: indicates which cues appear, not how well the class was taught.")

        tab_a, tab_e, tab_v, tab_l = st.tabs(["Assessment", "Evidence (observed text)",
                                              "Video technical checks", "Limitations & export"])
        with tab_a:
            st.caption("Model interpretation of the transcript, by dimension (1-5).")
            for d in r["dimensions"]:
                st.markdown(f"**{d['name']}** - {d['score']}/5 · evidence: {d['evidence_level']} "
                            f"· weight {d['weight']:.0%}")
                st.progress(d["score"] / 5)
                st.write(d["explanation"])
                if d["note"]:
                    st.caption(d["note"])
            cs, ci = st.columns(2)
            cs.markdown("**Strengths**")
            for t in r["strengths"] or ["None reported."]:
                cs.write(f"- {t}")
            ci.markdown("**Areas for improvement**")
            for t in r["improvements"] or ["None reported."]:
                ci.write(f"- {t}")
        with tab_e:
            st.caption("Exact quotations, each verified as a substring of the supplied transcript. "
                       "Unverifiable quotations are omitted.")
            for d in r["dimensions"]:
                st.markdown(f"**{d['name']}**")
                if not d["verified_quotes"]:
                    st.write("_No verified quotation._")
                for q in d["verified_quotes"]:
                    st.markdown(f"> {q['quote']}  \n> <sub>segment {q['segment']} · {format_time(q['start'])}</sub>",
                                unsafe_allow_html=True)
            st.markdown("**Transcript sampling**")
            st.dataframe([{"Excerpt": e["id"], "Stage": e["stage"], "From": format_time(e["start"]),
                           "To": format_time(e["end"]), "Words": e["words"]} for e in r["excerpts"]],
                         hide_index=True)
        with tab_v:
            st.caption("Measurements from sampled frames. Not part of the teaching score.")
            if r["video"]:
                v = r["video"]
                a, b, c, d_ = st.columns(4)
                a.metric("Duration", format_time(v["duration_seconds"]))
                b.metric("Frames sampled", v["frames_sampled"])
                c.metric("Mean brightness (0-255)", v["mean_brightness"])
                d_.metric("Mean sharpness", v["mean_sharpness"])
                st.write(f"Resolution {v['resolution']} · brightness range {v['brightness_range']} · "
                         f"lowest sharpness {v['min_sharpness']}")
                st.info(v["note"])
            elif r["video_error"]:
                st.error(f"Video check failed: {r['video_error']}")
            else:
                st.write("No video uploaded.")
        with tab_l:
            st.markdown("**Unavailable evidence and limitations**")
            for t in r["limitations"]:
                st.write(f"- {t}")
            st.write(f"Data sent to an external service: **{'yes' if r['sent_to_external_service'] else 'no'}**. "
                     f"Engine: {r['provider']} ({r['model']}).")
            st.download_button("Download report (JSON)", json.dumps(r, indent=2, default=str),
                               file_name="classquality_report.json", mime="application/json",
                               key=f"dl_{len(results) - 1 - i}")
