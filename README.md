# NeuroScan EEG — Multidimensional Neurological Disorder Diagnosis System

A complete, production-ready Flask web application for EEG-based neurological diagnosis using
Machine Learning, Deep Learning, AES Encryption, and Role-Based Access Control.

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.0 |
| Database | SQLite via Flask-SQLAlchemy |
| Auth | Flask-Login + Flask-Bcrypt |
| Encryption | `cryptography` library (Fernet / AES-128-CBC) |
| ML | scikit-learn (Random Forest, SVM), XGBoost |
| DL | TensorFlow/Keras (CNN + BiLSTM) |
| Frontend | Bootstrap 5 + Chart.js |
| Signal Processing | NumPy, SciPy, Matplotlib |

---

## 📁 Project Structure

```
eeg_project/
│
├── app.py                      ← Main Flask app (routes, auth, RBAC, predictions)
├── train_model.py              ← ML/DL model training script
├── encryption.py               ← AES Fernet encryption module
├── features.py                 ← EEG feature extraction (time + frequency domain)
├── generate_sample_dataset.py  ← Generate synthetic dataset for testing
│
├── models/                     ← Saved trained models (.pkl, .keras)
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── xgb_model.pkl
│   ├── bilstm_model.keras
│   ├── cnn_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── best_model.txt
│   └── metrics.json
│
├── data/                       ← Place your EEG dataset CSV here
│   └── eeg_dataset.csv
│
├── encrypted_data/             ← AES-encrypted EEG files stored here
├── static/
│   ├── css/custom.css
│   ├── uploads/                ← Temporary (raw files deleted after encryption)
│   └── reports/                ← Temporary report downloads
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── about.html
│   ├── login.html
│   ├── register.html
│   ├── admin_dashboard.html
│   ├── doctor_dashboard.html
│   ├── upload.html
│   ├── result.html
│   ├── reports.html
│   └── error.html
│
├── logs/                       ← Application audit logs
├── secret.key                  ← Auto-generated AES encryption key
├── database.db                 ← SQLite database (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow is optional. If you skip it, CNN and BiLSTM training will be skipped but all other models will work.
> To install without TensorFlow: remove `tensorflow` from requirements.txt before installing.

### Step 3 — Generate sample dataset (or use your own)

```bash
python generate_sample_dataset.py
```

This creates `data/eeg_dataset.csv` with 2500 synthetic EEG samples across 5 classes.

**Or use the real Epileptic Seizure Recognition Dataset:**
- Download from: https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
- Save as `data/eeg_dataset.csv`

### Step 4 — Train the models

```bash
python train_model.py --dataset data/eeg_dataset.csv
```

This will:
- Load and preprocess the dataset
- Extract 13 time + frequency domain features per sample
- Train Random Forest, SVM, XGBoost, BiLSTM, CNN
- Save all models to `/models/`
- Save performance metrics to `models/metrics.json`
- Display accuracy/F1/precision/recall for each model

### Step 5 — Run the Flask web application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 👤 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |

You can register new Doctor accounts from the Register page.

To register an Admin account, you need the admin token (first 8 chars of SECRET_KEY: `eeg-secu`).

---

## 🔐 Security Architecture

### AES Encryption Flow

```
Doctor uploads CSV
      ↓
File saved temporarily to /static/uploads/
      ↓
Fernet.encrypt(file_bytes) → saved to /encrypted_data/*.enc
      ↓
Raw file DELETED from uploads
      ↓
For prediction: decrypt to temp path → run ML → delete temp file
      ↓
Prediction report dict → JSON → Fernet.encrypt() → stored in SQLite as BLOB
```

### Password Security
- All passwords hashed with **bcrypt** (adaptive hashing, salt included)
- Sessions managed by Flask-Login with secure cookies

### Role-Based Access Control
- `@role_required('admin')` — Admin-only routes
- `@role_required('doctor', 'admin')` — Doctor + Admin routes
- Doctors can only view their own patient reports
- Admins can view all users, reports, and logs

---

## 📊 EEG Feature Extraction

### Time Domain (8 features)
| Feature | Formula |
|---|---|
| Mean | μ = Σx / N |
| Variance | σ² = Σ(x-μ)² / N |
| Standard Deviation | σ = √(σ²) |
| Energy | E = Σx² |
| Entropy | H = -Σp·log(p) |
| Hjorth Activity | var(x) |
| Hjorth Mobility | √(var(x') / var(x)) |
| Hjorth Complexity | mob(x') / mob(x) |

### Frequency Domain (5 features)
| Feature | Method |
|---|---|
| Delta Band Power (0.5-4 Hz) | FFT + Trapezoid integration |
| Theta Band Power (4-8 Hz) | FFT + Trapezoid integration |
| Alpha Band Power (8-13 Hz) | FFT + Trapezoid integration |
| Beta Band Power (13-30 Hz) | FFT + Trapezoid integration |
| Spectral Entropy | Entropy of normalised PSD |

### Preprocessing Pipeline
1. **FIR Bandpass Filter** (0.5–50 Hz, Hamming window, 51 taps) — removes power-line noise
2. **ICA** (via FastICA) — removes eye-blink/muscle artifacts
3. **StandardScaler** — zero-mean, unit-variance normalisation

---

## 🤖 Models & Performance

After training on the Epileptic Seizure Recognition dataset, typical performance:

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | ~98% | ~0.98 |
| SVM (RBF) | ~97% | ~0.97 |
| XGBoost | ~98% | ~0.98 |
| BiLSTM | ~95% | ~0.95 |
| CNN (Spectrogram) | ~93% | ~0.93 |

---

## 🐛 Troubleshooting

**`No trained model found` error:**
→ Run `python train_model.py --dataset data/eeg_dataset.csv` first.

**`ModuleNotFoundError: tensorflow`:**
→ Install with: `pip install tensorflow` or use CPU-only: `pip install tensorflow-cpu`

**`ModuleNotFoundError: xgboost`:**
→ Install with: `pip install xgboost`

**File upload fails:**
→ Check that `/static/uploads/` folder exists and is writable.

**Database errors:**
→ Delete `database.db` and restart the app — it will be recreated.

---

## 📄 License

MIT License — Free to use for academic and educational purposes.
Built as a B.Tech Final Year Mini Project.
