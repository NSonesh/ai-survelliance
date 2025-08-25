import os
import io
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import cv2

# ─────────────────────────── Config ───────────────────────────
OUTPUTS_DIR = Path("outputs")
FPS_DEFAULT = 25  # change in sidebar if needed
ACCENT = "#60a5fa"  # sky-500

# ─────────────────────────── Page & CSS ───────────────────────
st.set_page_config(
    page_title="🚨 AI Surveillance Dashboard",
    layout="wide",
    page_icon="🚨",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
:root {{
  --accent: {ACCENT};
}}
.block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
h1, h2, h3 {{ letter-spacing: .2px; }}
.small {{ color:#9ca3af; font-size:.85rem; }}
.badge {{ display:inline-block; padding:8px 12px; border-radius:12px; background:#111827; color:#e5e7eb; border:1px solid #1f2937; }}
.metric {{ display:inline-grid; gap:2px; padding:10px 14px; border-radius:14px; background:#0b1220; border:1px solid #111827; }}
.metric b {{ color:#d1d5db; }}
.metric span {{ color:#9ca3af; font-size:.8rem; }}
.card {{
  border:1px solid #1f2937; border-radius:16px; padding:14px; background:linear-gradient(180deg,#0b1220 0%, #0a0f1a 100%);
  box-shadow: 0 8px 30px rgba(0,0,0,.35);
}}
.video-wrap {{ border:1px solid #1f2937; border-radius:14px; overflow:hidden; }}
.thumb {{ border:1px solid #1f2937; border-radius:8px; overflow:hidden; }}
.btn-download {{
  display:inline-block; padding:8px 12px; border-radius:10px; border:1px solid #1f2937; background:#0b1220; color:#e5e7eb;
}}
.hl {{ background:rgba(96,165,250,.2); padding:0 6px; border-radius:6px; }}
.stSlider > div [data-baseweb="slider"]>div>div {{ background: var(--accent) !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("🚨 AI‑Powered Surveillance Dashboard")
st.caption("Explore anomalies, timelines, and annotated videos from Avenue / UCSD evaluations.")

# ─────────────────────────── Helpers ──────────────────────────
def list_alert_csvs() -> List[str]:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    return sorted([p.name for p in OUTPUTS_DIR.glob("*_alerts.csv")])

def load_alerts(csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / csv_name)
    # normalize
    cols = {c.lower(): c for c in df.columns}
    if "clip" not in df.columns:
        for alt in ("video","file","clip_name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "clip"})
                break
    if "clip" not in df.columns:
        df["clip"] = "unknown"

    # unify frame_idx & score
    df = df.rename(columns={
        cols.get("frame_idx","frame_idx"): "frame_idx",
        cols.get("mean_error","mean_error"): "mean_error"
    })

    # boolean alert
    if "alert" in df.columns:
        df["alert_bool"] = df["alert"].astype(str).str.lower().eq("anomaly")
    elif "pred_alert" in df.columns:
        df["alert_bool"] = df["pred_alert"].astype(int) == 1
    else:
        df["alert_bool"] = False

    df["_source_csv"] = csv_name
    return df

def timestamp_from_frame(frame_idx: int, fps: int) -> str:
    secs = frame_idx / max(fps, 1)
    m = int(secs // 60)
    s = secs - 60*m
    return f"{m:02d}:{int(s):02d}.{int((s-int(s))*1000):03d}"

def find_video_for_clip(clip: str) -> Optional[Path]:
    """Find *_annot.mp4 in outputs/ that contains the clip name."""
    c = str(clip).lower()
    for p in OUTPUTS_DIR.glob("*_annot.mp4"):
        if c in p.stem.lower():
            return p
    # fallback: first mp4
    vids = list(OUTPUTS_DIR.glob("*.mp4"))
    return vids[0] if vids else None

def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ensure_h264(input_mp4: Path) -> Path:
    """
    Make H.264/AAC copy if necessary for HTML5 playback.
    Writes outputs/h264_<name>.mp4 and returns that path on success.
    If ffmpeg is missing or transcode fails, returns the original path.
    """
    out = input_mp4.with_name(f"h264_{input_mp4.name}")
    if out.exists():
        return out
    if not ffmpeg_exists():
        return input_mp4
    try:
        cmd = [
            "ffmpeg","-y","-i",str(input_mp4),
            "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",
            "-c:a","aac","-movflags","+faststart", str(out)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except Exception:
        return input_mp4

def grab_thumbnails(video_path: Path, times_s: List[float], size: Optional[Tuple[int,int]]=None) -> List[np.ndarray]:
    """
    Extract thumbnails (BGR) at given second offsets. Uses OpenCV only.
    """
    thumbs = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return thumbs
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    for t in times_s:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            thumbs.append(None)
            continue
        if size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        thumbs.append(frame)
    cap.release()
    return thumbs

def to_png_bytes(bgr_img: np.ndarray) -> bytes:
    if bgr_img is None:
        return b""
    ok, buf = cv2.imencode(".png", bgr_img)
    return buf.tobytes() if ok else b""

# ─────────────────────────── Sidebar ──────────────────────────
csv_files = list_alert_csvs()
if not csv_files:
    st.error("No *_alerts.csv found in outputs/. Run an evaluation first.")
    st.stop()

dataset = st.sidebar.selectbox("Results CSV", csv_files, index=0)
df = load_alerts(dataset)

clips = sorted(df["clip"].astype(str).unique())
clip = st.sidebar.selectbox("Clip", clips, index=0)

fps = st.sidebar.number_input("Playback FPS", 1, 120, FPS_DEFAULT, step=1)
only_anoms = st.sidebar.checkbox("Show only anomalies", value=True)

if df.empty:
    st.warning("Selected CSV is empty.")
    st.stop()

# per-clip slice
clip_df_all = df[df["clip"].astype(str) == clip].copy()
f_min, f_max = int(clip_df_all["frame_idx"].min()), int(clip_df_all["frame_idx"].max())

score_min = float(clip_df_all["mean_error"].min())
score_max = float(clip_df_all["mean_error"].max())
th_default = float(np.quantile(clip_df_all["mean_error"], 0.95)) if len(clip_df_all) else score_min

th = st.sidebar.slider("Score threshold", min_value=score_min, max_value=score_max,
                       value=th_default, step=float((score_max-score_min)/100 or 0.001))

f_range = st.sidebar.slider("Frame range", min_value=f_min, max_value=f_max, value=(f_min, f_max), step=1)

# ─────────────────────────── Metrics Row ──────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='metric'><span>Rows (clip)</span><b>{len(clip_df_all)}</b></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric'><span>Anomalies</span><b>{int(clip_df_all['alert_bool'].sum())}</b></div>", unsafe_allow_html=True)

clip_df = clip_df_all[(clip_df_all["frame_idx"].between(f_range[0], f_range[1])) &
                      (clip_df_all["mean_error"] >= th)]
if only_anoms:
    clip_df = clip_df[clip_df["alert_bool"] == True]

m3.markdown(f"<div class='metric'><span>Filtered</span><b>{len(clip_df)}</b></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric'><span>File</span><b>{dataset}</b></div>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────── Timeline & Histogram ─────────────
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader(f"Timeline — {clip}")
    temp = clip_df_all.set_index("frame_idx").sort_index()
    st.line_chart(temp["mean_error"], height=220)
    st.caption("Tip: raise/lower the score threshold in the sidebar to focus on top‑scoring frames.")

    # score histogram
    st.subheader("Score distribution (clip)")
    hist_counts, bin_edges = np.histogram(clip_df_all["mean_error"].values, bins=40)
    hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "count": hist_counts})
    st.bar_chart(hist_df.set_index("bin_left"))

    # alerts table
    st.subheader("Alerts")
    if not clip_df.empty:
        clip_df = clip_df.assign(timestamp=[timestamp_from_frame(int(i), fps) for i in clip_df["frame_idx"]])
    show_cols = ["timestamp", "frame_idx", "mean_error"]
    if "gt_flag" in clip_df.columns:
        show_cols.append("gt_flag")
    st.dataframe(
        clip_df.sort_values("frame_idx")[show_cols].rename(columns={"mean_error": "score","gt_flag":"gt"}),
        use_container_width=True, height=280
    )

with right:
    st.subheader("Video Player")
    raw_video = find_video_for_clip(clip)
    if not raw_video:
        st.warning("No annotated video found in outputs/. Generate *_annot.mp4 with your eval script.")
    else:
        # ensure H.264 copy (for browser compatibility)
        play_video = ensure_h264(raw_video)
        st.markdown(f"<div class='small'>Playing: <b>{play_video.name}</b></div>", unsafe_allow_html=True)

        st.markdown("<div class='video-wrap'>", unsafe_allow_html=True)
        st.video(str(play_video))
        st.markdown("</div>", unsafe_allow_html=True)

        # Download buttons
        c1, c2 = st.columns(2)
        with c1:
            with open(OUTPUTS_DIR / dataset, "rb") as f:
                st.download_button("⬇️ Download CSV", data=f, file_name=dataset, mime="text/csv")
        with c2:
            try:
                with open(play_video, "rb") as fv:
                    st.download_button("⬇️ Download Video", data=fv, file_name=play_video.name, mime="video/mp4")
            except Exception:
                st.info("Open the outputs/ folder to download the video file directly.")

        # Jump helper
        if len(clip_df_all):
            jump_frame = int(clip_df["frame_idx"].iloc[0]) if len(clip_df) else int(clip_df_all["frame_idx"].iloc[0])
            st.markdown(
                f"Jump target → frame <b>{jump_frame}</b> ≈ "
                f"<span class='hl'>{timestamp_from_frame(jump_frame, fps)}</span> at {fps} FPS.",
                unsafe_allow_html=True
            )
        st.caption("Use the player's scrubber to approximate the target time.")

# ─────────────────────────── Thumbnail Strip ───────────────────
st.markdown("---")
st.subheader("Frame thumbnails")

if raw_video:
    # pick 10 evenly spaced times inside selected frame range
    total_frames = int(clip_df_all["frame_idx"].max()) + 1
    start_s = f_range[0] / float(fps)
    end_s   = max(f_range[0]+1, f_range[1]) / float(fps)
    times = np.linspace(start_s, end_s, num=10, endpoint=True)
    thumbs = grab_thumbnails(ensure_h264(raw_video), times, size=(224, 126))  # 16:9ish

    cols = st.columns(10)
    for i, (t, img) in enumerate(zip(times, thumbs)):
        with cols[i]:
            if img is None:
                st.write("—")
                continue
            st.image(img[:, :, ::-1], use_column_width=True, caption=timestamp_from_frame(int(t*fps), fps))
else:
    st.info("No video available to render thumbnails.")

st.markdown("---")
st.subheader("All annotated videos")
# gallery
vids = sorted(OUTPUTS_DIR.glob("*_annot.mp4"))
if not vids:
    st.info("No *_annot.mp4 files in outputs/.")
else:
    cols = st.columns(2)
    for i, p in enumerate(vids):
        with cols[i % 2]:
            st.markdown(f"<div class='card'><b>{p.name}</b></div>", unsafe_allow_html=True)
            st.video(str(ensure_h264(p)))
