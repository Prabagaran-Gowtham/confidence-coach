"""Core analysis for the Speaking Confidence Detection app (Component 2)."""
import io, os, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

HERE = Path(__file__).parent
SR, HOP = 16000, 320
MAX_SECONDS = 60
FILTERS = ("highpass=f=60:p=2,lowpass=f=7600:p=2,afftdn=nr=12:nf=-45:tn=1:ad=0.5:gs=6,"
           "alimiter=limit=0.891251:level=0:latency=1")
THRESH, LOW, HIGH = -0.15, 4, 7

GROUPS = {
    "Pitch variation": ["f0_std_st", "f0_range_st"],
    "Pause patterns": ["pauses_per_min", "mean_pause_s", "silence_ratio"],
    "Loudness & energy": ["loud_mean_db", "rms_mean", "centroid_hz"],
    "Speaking rhythm": ["syll_rate", "rhythm_cv"],
    "Voice stability": ["jitter_proxy", "loud_std_db", "voiced_ratio"],
    "Expressiveness": ["loud_range_db"],
    "Voice quality (MFCC)": ["mfcc_dyn"] + [f"mfcc{i}_mean" for i in range(1, 14)],
}
TIPS = {
    "Pitch variation": "Vary your pitch more. Raise and lower your voice to stress key words.",
    "Pause patterns": "Reduce long or frequent pauses. Plan your next sentence while finishing the current one.",
    "Loudness & energy": "Speak louder and with more energy. Project your voice as if talking to the back of a room.",
    "Speaking rhythm": "Keep a steadier pace. Avoid rushing or dragging, and practise with a timer.",
    "Voice stability": "Steady your voice. Breathe slowly before you start and keep your volume consistent.",
    "Expressiveness": "Add more emphasis and emotion. Use louder and softer moments to highlight important points.",
    "Voice quality (MFCC)": "Speak more clearly. Open your mouth and articulate each word.",
}
FEATS = {
    "Pitch variation": dict(kind="pitch", main="f0_std_st", unit="pitch variation (semitones)",
        why="How much your pitch moves around your own average. Lively movement sounds engaged and sure.",
        good="Higher is better, up to natural speech.",
        show=[("f0_std_st", "Pitch spread (semitones)"), ("f0_range_st", "Pitch range (semitones)")]),
    "Pause patterns": dict(kind="pause", main="pauses_per_min", unit="pauses per minute",
        why="How often and how long you stop. Many long pauses sound hesitant. Red areas are pauses of 0.25 s or more.",
        good="Fewer and shorter pauses are better.",
        show=[("pauses_per_min", "Pauses per minute"), ("mean_pause_s", "Average pause (s)"), ("silence_ratio", "Silence share")]),
    "Loudness": dict(kind="loud", main="loud_mean_db", unit="average loudness (dB)",
        why="How strongly you project while speaking. Very soft speech sounds unsure.",
        good="Higher is better, while staying comfortable.",
        show=[("loud_mean_db", "Average loudness (dB)")]),
    "Speaking rhythm": dict(kind="rhythm", main="syll_rate", unit="syllable-like events per second",
        why="Your pace and how even it is. Rushed or choppy delivery lowers the score.",
        good="A steady, moderate pace is best.",
        show=[("syll_rate", "Syllable rate (per s)"), ("rhythm_cv", "Rhythm variability")]),
    "Vocal energy": dict(kind="energy", main="rms_mean", unit="average energy",
        why="The strength and brightness of your voice. Weak, dull speech sounds low in energy.",
        good="Higher is better.",
        show=[("rms_mean", "Average energy"), ("centroid_hz", "Brightness (Hz)")]),
    "Voice stability": dict(kind="stab", main="jitter_proxy", unit="pitch wobble (semitones)",
        why="How steady your voice is. Shakiness shows as jumpy pitch and uneven volume.",
        good="Lower wobble is better.",
        show=[("jitter_proxy", "Pitch wobble"), ("loud_std_db", "Volume variation (dB)"), ("voiced_ratio", "Voiced share")]),
    "Expressiveness": dict(kind="expr", main="loud_range_db", unit="loudness range (dB)",
        why="How much emphasis and emotion you add through pitch and volume. The band is your normal pitch range.",
        good="More variation is better, up to natural speech.",
        show=[("f0_range_st", "Pitch range (semitones)"), ("loud_range_db", "Loudness range (dB)")]),
    "Voice quality (MFCC)": dict(kind="mfcc", main="mfcc_dyn", unit="spectral change (MFCC)",
        why="MFCCs describe the tone and clarity of your voice. The heatmap shows how it changes while you speak.",
        good="More change usually means clearer articulation.",
        show=[("mfcc_dyn", "Spectral change")]),
}


# ---------------------------------------------------------------- model assets
def load_assets():
    import shap
    art = joblib.load(HERE / "confidence_model.joblib")
    ref = pd.read_csv(HERE / "confidence_dataset.csv")
    return {"model": art["model"], "cols": art["columns"], "med": art["median"], "ref": ref,
            "name": art.get("name", ""), "explainer": shap.Explainer(art["model"].predict, art["background"])}


# ---------------------------------------------------------------- audio + features
def clean_audio(src, dst):
    """Same cleaning as the training audio (16 kHz mono, band-pass, light denoise, peak limiter)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-t", str(MAX_SECONDS),
           "-ac", "1", "-ar", str(SR), "-af", FILTERS, str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or "Could not read this audio file.")


def extract_full(path):
    """Identical feature computation to the training run, plus the signals used for the graphs."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = y[:SR * 90]; dur = len(y) / sr
    if dur < 2:
        raise ValueError("The recording is too short. Please speak for at least a few seconds.")
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
    iv = librosa.effects.split(y, top_db=30, frame_length=640, hop_length=HOP)
    speech = sum(e - s for s, e in iv) / sr
    gaps = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
    pauses = [g for g in gaps if g >= 0.25]
    mask = np.zeros(len(rms), bool)
    for s, e in iv:
        mask[s // HOP: e // HOP + 1] = True
    spdb = db[mask] if mask.sum() > 5 else db
    sprms = rms[mask] if mask.sum() > 5 else rms
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    ok = ~np.isnan(f0)
    f0_std = f0_rng = jit = np.nan
    if ok.sum() > 10:
        st = 12 * np.log2(f0[ok] / np.median(f0[ok]))
        f0_std = st.std(); f0_rng = np.percentile(st, 95) - np.percentile(st, 5)
        idx = np.where(ok)[0]; d = np.abs(np.diff(st))[np.diff(idx) == 1]
        jit = d.mean() if len(d) else np.nan
    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    ioi = np.diff(on); ioi = ioi[ioi < 1.0]
    cv = ioi.std() / ioi.mean() if len(ioi) > 5 else np.nan
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=640, hop_length=HOP)[0]
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=640, hop_length=HOP, n_mels=40)
    row = {"duration_s": round(dur, 2),
           "f0_std_st": f0_std, "f0_range_st": f0_rng, "jitter_proxy": jit, "voiced_ratio": ok.mean(),
           "loud_mean_db": spdb.mean(), "loud_std_db": spdb.std(),
           "loud_range_db": np.percentile(spdb, 95) - np.percentile(spdb, 5),
           "rms_mean": sprms.mean(),
           "centroid_hz": cent[:len(mask)][mask[:len(cent)]].mean() if mask.sum() > 5 else cent.mean(),
           "pauses_per_min": len(pauses) / dur * 60, "mean_pause_s": np.mean(pauses) if pauses else 0.0,
           "silence_ratio": 1 - speech / dur, "syll_rate": len(on) / speech if speech > 0 else np.nan,
           "rhythm_cv": cv, "mfcc_dyn": float(np.mean(mf[1:].std(axis=1)))}
    for i in range(13):
        row[f"mfcc{i + 1}_mean"] = mf[i].mean()
    S = dict(y=y, rms=rms, db=db, cent=cent, mf=mf, on=on, f0=f0,
             t_r=librosa.times_like(rms, sr=SR, hop_length=HOP), t_f=librosa.times_like(f0, sr=SR, hop_length=HOP),
             pz=[(iv[i][1] / SR, iv[i + 1][0] / SR) for i in range(len(iv) - 1)
                 if (iv[i + 1][0] - iv[i][1]) / SR >= 0.25])
    return row, S


# ---------------------------------------------------------------- scoring
def predict(row, A):
    x = pd.DataFrame([row]).reindex(columns=A["cols"]).fillna(A["med"])
    score = float(np.clip(A["model"].predict(x)[0], 0, 10))
    contrib = dict(zip(A["cols"], A["explainer"](x).values[0]))
    impact = {g: float(sum(contrib[c] for c in cs if c in contrib)) for g, cs in GROUPS.items()}
    weak = sorted([(g, v) for g, v in impact.items() if v < THRESH], key=lambda t: t[1])[:2]
    return {"confidence_score": round(score, 1),
            "level": "Low" if score < LOW else "High" if score >= HIGH else "Medium",
            "feedback": [TIPS[g] for g, _ in weak] or ["Good delivery. No major weaknesses found."],
            "factor_impact": {g: round(v, 2) for g, v in impact.items()}}


def _pct(c, v, ref):
    return 5.0 if v is None or pd.isna(v) else percentileofscore(ref[c].dropna().values, v, kind="mean") / 10


def feature_scores(r, ref):
    hi = lambda c: _pct(c, r.get(c), ref)
    lo = lambda c: 10 - _pct(c, r.get(c), ref)
    mid = lambda c: (1 - abs(_pct(c, r.get(c), ref) / 10 - 0.5) * 2) * 10
    m = lambda *v: float(np.mean(v))
    return {"Pitch variation": m(hi("f0_std_st"), hi("f0_range_st")),
            "Pause patterns": m(lo("pauses_per_min"), lo("silence_ratio"), lo("mean_pause_s")),
            "Loudness": hi("loud_mean_db"),
            "Speaking rhythm": m(mid("syll_rate"), lo("rhythm_cv")),
            "Vocal energy": m(hi("rms_mean"), hi("centroid_hz")),
            "Voice stability": m(lo("jitter_proxy"), lo("loud_std_db"), hi("voiced_ratio")),
            "Expressiveness": m(hi("f0_range_st"), hi("loud_range_db")),
            "Voice quality (MFCC)": hi("mfcc_dyn")}


def spoken_text(res):
    imp = res["factor_impact"]; best = max(imp, key=imp.get)
    t = f"Your confidence score is {res['confidence_score']} out of 10, which is a {res['level']} level. "
    if imp[best] > 0.15:
        t += f"Your strongest area is {best.lower()}. "
    return (t + "Here is my feedback. " + " ".join(res["feedback"])).replace(" (MFCC)", "")


def make_voice(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO(); gTTS(text, lang="en").write_to_fp(buf); return buf.getvalue()
    except Exception:
        return None


def analyse(src_path, A):
    with tempfile.TemporaryDirectory() as d:
        clean = os.path.join(d, "clean.wav")
        clean_audio(src_path, clean)
        row, S = extract_full(clean)
    res = predict(row, A)
    return {"res": res, "row": row, "S": S, "fs": feature_scores(row, A["ref"]), "voice": make_voice(spoken_text(res))}


# ---------------------------------------------------------------- figures
def gauge(score):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    for a, b, c in [(0, 4, "#e74c3c"), (4, 7, "#f1c40f"), (7, 10, "#2ecc71")]:
        ax.barh(0, b - a, left=a, color=c, height=.5)
    ax.plot([score, score], [-.45, .45], color="black", lw=4)
    ax.text(score, .6, f"{score:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(-.6, 1.1); ax.set_yticks([]); ax.set_xticks([0, 4, 7, 10])
    ax.set_title("Confidence score (Low | Medium | High)")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    fig.tight_layout(); return fig


def radar(fs):
    lab, v = list(fs), list(fs.values())
    ang = np.linspace(0, 2 * np.pi, len(lab), endpoint=False).tolist(); v += v[:1]; ang += ang[:1]
    fig = plt.figure(figsize=(5.5, 5.5)); ax = plt.subplot(111, polar=True)
    ax.plot(ang, [5] * len(ang), "--", c="gray", lw=1, label="Dataset average")
    ax.plot(ang, v, c="#2c7be5", lw=2); ax.fill(ang, v, c="#2c7be5", alpha=.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0, 10)
    ax.set_title("Feature scores (0-10)", y=1.1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -.05), fontsize=8); fig.tight_layout(); return fig


def impact_bar(imp):
    it = sorted(imp.items(), key=lambda kv: kv[1]); fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([k for k, _ in it], [v for _, v in it], color=["#e74c3c" if v < 0 else "#2ecc71" for _, v in it])
    ax.axvline(0, c="black", lw=1); ax.set_xlabel("Effect on your score (points, vs an average speaker)")
    ax.set_title("Why you got this score"); fig.tight_layout(); return fig


def dist_plot(ref, col, val, label):
    fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(ref[col].dropna(), bins=30, color="#cfd8e3")
    if val is not None and np.isfinite(val):
        ax.axvline(val, c="#2c7be5", lw=3, label="You"); ax.legend()
    ax.set_title(f"You vs the {len(ref)} training recordings"); ax.set_xlabel(label); fig.tight_layout(); return fig


def main_plot(kind, S, row):
    fig, ax = plt.subplots(figsize=(8, 3)); y, tr, tf, f0 = S["y"], S["t_r"], S["t_f"], S["f0"]
    ok = np.isfinite(f0).sum() > 5
    if kind == "pitch":
        if ok: ax.plot(tf, f0, ".", ms=3); ax.axhline(np.nanmedian(f0), ls="--", c="gray"); ax.set_ylabel("Pitch (Hz)")
        ax.set_title("Your pitch contour" if ok else "Pitch could not be tracked in this recording")
    elif kind == "pause":
        ax.plot(np.arange(len(y)) / SR, y, lw=.4)
        for a, b in S["pz"]: ax.axvspan(a, b, color="red", alpha=.3)
        ax.set_title(f"Waveform, {len(S['pz'])} pauses of 0.25 s or more in red")
    elif kind == "loud":
        ax.plot(tr, S["db"], lw=.8); ax.axhline(row["loud_mean_db"], ls="--", c="gray", label="average while speaking")
        ax.legend(); ax.set_ylabel("dB"); ax.set_title("Your loudness over time")
    elif kind == "rhythm":
        ioi = np.diff(S["on"]); ioi = ioi[ioi < 1]; ax.hist(ioi, bins=25, color="#2c7be5")
        ax.set_xlabel("Time between speech events (s)"); ax.set_title("Pace pattern (narrow = even rhythm)")
    elif kind == "energy":
        ax.plot(tr, S["rms"], lw=.8); ax2 = ax.twinx(); ax2.plot(tr, S["cent"], c="orange", lw=.6)
        ax.set_title("Energy (blue) and brightness (orange)")
    elif kind in ("stab", "expr"):
        if ok:
            st = 12 * np.log2(f0 / np.nanmedian(f0)); ax.plot(tf, st, ".-", lw=.5, ms=3); ax.set_ylabel("Semitones from your average")
            if kind == "expr": ax.axhspan(np.nanpercentile(st, 5), np.nanpercentile(st, 95), alpha=.2)
        ax.set_title("Pitch steadiness" if kind == "stab" else "Pitch range (band = 5th to 95th percentile)")
    else:
        im = ax.imshow(S["mf"], aspect="auto", origin="lower", extent=[0, tr[-1], 1, 13])
        fig.colorbar(im); ax.set_ylabel("MFCC no."); ax.set_title("MFCC heatmap")
    ax.set_xlabel(ax.get_xlabel() or "Time (s)"); fig.tight_layout(); return fig


def history_plot(hist):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhspan(0, 4, color="#e74c3c", alpha=.1); ax.axhspan(7, 10, color="#2ecc71", alpha=.1)
    ax.plot([h["Attempt"] for h in hist], [h["Score"] for h in hist], "o-")
    ax.set_ylim(0, 10); ax.set_xlabel("Attempt"); ax.set_ylabel("Score"); ax.set_title("Your progress")
    ax.set_xticks([h["Attempt"] for h in hist]); fig.tight_layout(); return fig
