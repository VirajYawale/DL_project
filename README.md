# 🎓 Smart Attendance System v2

**ArcFace + InsightFace · Video-based Model Training · Anti-Spoofing · Flask**

A production-ready smart attendance system where each student's face model is trained
directly from a 5–10 second webcam video — no pre-built dataset needed.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## ✨ What's New in v2

| Feature | Detail |
|---|---|
| **Video-based training** | 5–10s webcam recording → full ArcFace model per student |
| **9-type augmentation** | Glasses, low-light, distance, hat, noise, blur, flip, brightness, partial cover |
| **Low-light robustness** | CLAHE + auto gamma correction handles dim/night CCTV |
| **Distant face support** | Bicubic upscale for small faces; trained on distance-simulated variants |
| **Glasses + accessories** | Enrolled with augmented glasses/hat variants — no need to remove |
| **Live training UI** | 6-step animated pipeline with real-time progress |
| **Frame quality scorer** | Discards blurry/dark frames automatically during training |
| **Anti-spoofing** | SilentFace (MiniFASNet) or LBP+FFT texture fallback |

---

## 🧠 Training Pipeline (Enrollment)

```
5–10s Video
    │
    ▼
Frame Extraction (adaptive 0.38s sampling → 15–25 frames)
    │
    ▼
Enhancement Pipeline per frame:
  • CLAHE — contrast-limited adaptive histogram (low-light fix)
  • Gamma correction — auto-detected for dark scenes (γ = 1.7–2.3)
  • NL-Means denoising — removes CCTV sensor noise
  • Bicubic upscale — min 480px width for distant faces
    │
    ▼
Face Detection (InsightFace SCRFD) + Quality Filter
  • Sharpness (Laplacian variance) + brightness + size + contrast
  • Reject frames below quality threshold (0.22)
    │
    ▼
ArcFace Embedding (InsightFace buffalo_l — 512-dim, L2 normalized)
    │
    ▼
Augmentation × 10 variants per accepted frame:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Augmentation        │ What it trains for                   │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Glasses simulation  │ Tinted/dark-lens glasses             │
  │ Low-light + noise   │ Night/dim CCTV, 15–40% brightness   │
  │ Distance simulate   │ Far students (20–40% scale factor)   │
  │ Hat/cap shadow      │ Top-of-head darkening               │
  │ Partial face cover  │ Chin/mouth occlusion                │
  │ Brightness jitter   │ Variable room lighting              │
  │ Horizontal flip     │ Face symmetry coverage              │
  │ Gaussian noise      │ Camera sensor artifacts             │
  │ Motion blur         │ Camera shake / fast movement        │
  └─────────────────────┴──────────────────────────────────────┘
    │
    ▼
Stack + L2 normalize all embeddings → save {roll_no}.pkl
Reload in-memory cosine search index (instant, no restart)

Result: ~120–160 total embeddings from a 5s video
```

---

## 🔍 Recognition (Live Attendance)

```
Webcam Frame
  → Enhancement (same pipeline as enrollment)
  → InsightFace SCRFD detection
  → ArcFace embedding (512-dim)
  → Cosine similarity search across all stored embeddings
  → Threshold 0.38 → match or unknown
  → Anti-spoof check
  → Mark attendance (30s cooldown debounce)
```

---

## 🛡️ Anti-Spoofing

**Layer 1 — Silent-Face (MiniFASNet)** *(install separately for production)*
```bash
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
cp -r Silent-Face-Anti-Spoofing/src models/silent_face/src
cp -r Silent-Face-Anti-Spoofing/resources/anti_spoof_models models/anti_spoof_models
```

**Layer 2 — Texture Analysis fallback** *(auto-used if SilentFace not found)*
- LBP (Local Binary Pattern) uniformity
- FFT mid-frequency energy ratio
- Sobel gradient variance

Detects: printed photos, phone/tablet screens, video replay attacks.

---

## 📁 Project Structure

```
smart_attendance_v2/
├── app.py                      # Flask + enrollment API + training trigger
├── requirements.txt
├── database/
│   └── models.py               # Student, Subject, Enrollment, Attendance, FaceEmbedding
├── utils/
│   ├── face_processor.py       # ★ Training pipeline + ArcFace recognition
│   ├── anti_spoof.py           # SilentFace + texture fallback
│   └── attendance_engine.py    # Frame orchestration + debounce
├── templates/
│   ├── base.html               # Sidebar layout + toast + API helper
│   ├── dashboard.html          # Stats + system status + enrollment overview
│   ├── enroll.html             # ★ Video recording + training UI
│   ├── attendance.html         # Live session + annotations + CSV export
│   ├── students.html           # Add / delete students
│   └── subjects.html           # Add subjects
└── data/                       # Auto-created
    ├── embeddings/             # {roll_no}.pkl — ArcFace embedding arrays
    ├── frames/                 # Saved frames per student (audit)
    └── videos/                 # Raw enrollment videos
```

---

## ⚙️ Key Configuration (utils/face_processor.py)

| Parameter | Default | Effect |
|---|---|---|
| `THRESHOLD` | `0.38` | Cosine similarity cutoff. Lower = stricter identity check |
| `MIN_QUALITY` | `0.22` | Frame quality floor. Raise to 0.35 for strict mode |
| `sample_interval` | `0.38s` | Frame sampling rate. Lower = more frames, slower training |
| `n_augments` | `5` | Albumentations augments per frame (+ 5 domain-specific fixed) |

---

## 💡 Tips

- **Glasses**: Keep them on during enrollment — the model is trained with glasses simulation
- **Low light**: CLAHE handles it, but at least one light source on the face recommended
- **Distance**: Works to ~4–5m with HD camera; ensure face is >30px after upscale
- **Bad enrollment**: Use "Retake" button — frame quality bar in the UI guides you
- **Production anti-spoof**: Always install Silent-Face; texture fallback is for dev only
- **GPU**: InsightFace uses ONNX and runs on CPU — GPU not required but faster with CUDA

---

## 📦 Dependencies

```bash
pip install flask flask-sqlalchemy insightface onnxruntime \
            deepface opencv-python numpy albumentations requests
```

Optional for better augmentation:
```bash
pip install albumentations  # Already in requirements.txt
```
