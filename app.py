import datetime, os, tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import core

st.set_page_config(page_title="Speaking Confidence Detection", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading the model (first start only)...")
def get_assets():
    return core.load_assets()


def show(fig):
    st.pyplot(fig, clear_figure=True); plt.close(fig)


A = get_assets()
ss = st.session_state
ss.setdefault("history", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### About this estimate")
    st.write("The selected model is a support vector regression (SVR), chosen from five tuned models "
             "(Ridge, Random Forest, SVR, XGBoost and a neural network). It produces a 0-10 confidence estimate "
             "from 29 vocal features.")
    st.caption("This is a research estimate, not a human expert rating, formal language certification "
               "or clinical assessment.")
    st.caption("Recordings are processed on this hosted app and are not saved as project data.")

st.title("🎙️ Speaking Confidence Detection")
st.write("Record up to 1 minute of English speech (or upload a file), press **Analyse**, then click through the tabs. "
         "Record again and press **Analyse** again to track your progress.")

# ------------------------------------------------------------------ input
c1, c2 = st.columns(2)
rec = c1.audio_input("Record your speech (up to 1 minute)")
up = c2.file_uploader("...or upload a recording", type=["wav", "mp3", "m4a", "ogg", "opus", "flac"])
src = rec if rec is not None else up

b1, b2 = st.columns(2)
go = b1.button("Analyse", type="primary", width="stretch")
if b2.button("Clear history", width="stretch"):
    ss.history = []; ss.pop("result", None)

if go:
    if src is None:
        st.warning("Please record or upload audio first.")
    else:
        try:
            with st.spinner("Analysing your voice... this can take up to a minute."):
                suffix = os.path.splitext(getattr(src, "name", "") or "")[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(src.getvalue()); path = f.name
                try:
                    out = core.analyse(path, A)
                finally:
                    os.remove(path)
            r = out["res"]
            ss.result = out; ss.play_once = True
            ss.history.append({"Attempt": len(ss.history) + 1, "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                               "Score": r["confidence_score"], "Level": r["level"],
                               **{k: round(v, 1) for k, v in out["fs"].items()}})
        except Exception as e:
            st.error(f"Could not analyse this recording: {e}")

# ------------------------------------------------------------------ results
out = ss.get("result")
if out is None:
    st.info("Your results will appear here after you press Analyse.")
    st.stop()

res, row, S, fs = out["res"], out["row"], out["S"], out["fs"]
t_sum, t_feat, t_det, t_fb, t_prog, t_info = st.tabs(
    ["📊 Summary", "🔍 Feature Analysis", "📋 Detailed Metrics", "💡 Feedback", "📈 Progress", "🧠 Model Info"])

with t_sum:
    st.subheader("Your speech profile")
    m1, m2 = st.columns(2)
    m1.metric("Confidence score", f"{res['confidence_score']} / 10")
    m2.metric("Level", res["level"])
    show(core.gauge(res["confidence_score"]))
    st.markdown("#### Key speech measures")
    k = st.columns(5)
    k[0].metric("Duration", f"{row['duration_s']:.0f} s")
    k[1].metric("Pauses / minute", f"{row['pauses_per_min']:.1f}")
    k[2].metric("Average pause", f"{row['mean_pause_s']:.2f} s")
    k[3].metric("Speaking share", f"{(1 - row['silence_ratio']) * 100:.0f}%")
    k[4].metric("Syllables / s", f"{row['syll_rate']:.1f}")
    show(core.radar(fs))
    st.caption("Feature scores compare you with the training recordings: 5 means average for that dataset.")

with t_feat:
    st.subheader("Click a feature to explore it")
    name = st.radio("Feature", list(core.FEATS), horizontal=True, label_visibility="collapsed", key="feat_choice")
    m = core.FEATS[name]
    st.markdown(f"### {name}: {fs[name]:.1f} / 10")
    st.write(m["why"]); st.caption(m["good"])
    cols = st.columns(len(m["show"]))
    for col, (c, lab) in zip(cols, m["show"]):
        v = row.get(c); col.metric(lab, f"{v:.2f}" if v == v else "n/a", f"dataset avg {A['ref'][c].mean():.2f}", delta_color="off")
    show(core.main_plot(m["kind"], S, row))
    show(core.dist_plot(A["ref"], m["main"], row.get(m["main"]), m["unit"]))

with t_det:
    st.subheader("All measured values")
    det = pd.DataFrame({"Measure": A["cols"], "Your value": [row.get(c) for c in A["cols"]],
                        "Dataset average": [A["ref"][c].mean() for c in A["cols"]]}).round(3)
    st.dataframe(det, width="stretch", hide_index=True)

with t_fb:
    st.subheader("Personalised feedback")
    for f in res["feedback"]:
        st.markdown(f"- {f}")
    if out.get("voice"):
        auto = ss.pop("play_once", False)
        try:
            st.audio(out["voice"], format="audio/mp3", autoplay=auto)
        except TypeError:
            st.audio(out["voice"], format="audio/mp3")
    else:
        st.caption("Spoken feedback is unavailable right now.")
    show(core.impact_bar(res["factor_impact"]))

with t_prog:
    st.subheader("Your attempts")
    st.dataframe(pd.DataFrame(ss.history), width="stretch", hide_index=True)
    show(core.history_plot(ss.history))

with t_info:
    st.markdown("""**Model:** SVR (support vector regression), chosen from five tuned models.
**Inputs:** 29 audio features (pitch, loudness, pauses, rhythm, stability, MFCCs) from speech cleaned with noise reduction, filtering and a peak limiter. Silences are kept because pauses are part of the signal.
**Output:** a 0-10 confidence estimate with Low (below 4), Medium and High (7 and above) levels.
**Explanation:** SHAP values show which feature groups raised or lowered your score.
**Important:** the training labels are rubric-based scores calculated from the features, so this is a research estimate that reproduces the rubric. It is not a human expert rating or a clinical assessment.""")
