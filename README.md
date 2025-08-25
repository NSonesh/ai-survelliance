# 🚨 AI-Powered Surveillance System

Anomaly detection for surveillance video using **3D Convolutional Autoencoders (CAE)** and an interactive **Streamlit dashboard**. Supports **Avenue** and **UCSD Ped1/Ped2** datasets.

---

## ✨ Features
- Learns “normal” video motion with a lightweight 3D-CAE
- Detects anomalies by reconstruction error (unsupervised)
- Supports **Avenue** (pixel-level masks) & **UCSD** (frame flags + masks)
- Generates annotated videos + per-frame CSV alerts
- Rich **Streamlit dashboard**:
  - Timeline chart
  - Score histograms
  - Alerts table with timestamps
  - Thumbnail strip with frame info
  - Inline H.264 video playback
  - One-click CSV/Video download

---

## ⚡ Quickstart

```bash
# 1) Create venv
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

# 2) Prepare datasets (after downloading)
python -m src.avenue.prepare_avenue
python -m src.ucsd.prepare_ucsd

# 3) Train model (CPU friendly demo)
python -m src.avenue.train_avenue_cae --epochs 3 --bs 4 --size 112x160

# 4) Evaluate
python -m src.avenue.eval_avenue
python -m src.ucsd.eval_ucsd

# 5) Launch dashboard
streamlit run src/dashboard.py
