"""
train_model.py - EEG Model Training Script
Trains Random Forest, SVM, XGBoost, CNN (spectrogram), and BiLSTM models.
Saves best model using joblib. Run this BEFORE starting the Flask app.

Usage:
    python train_model.py --dataset data/eeg_dataset.csv

The CSV must have EEG columns (numeric) and a final 'label' column.
"""
# pyright: reportMissingImports=false
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Skipping.")

from features import extract_all_features, get_feature_names

warnings.filterwarnings('ignore')

MODEL_DIR   = "models"
STATIC_DIR  = "static"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str):
    """
    Load EEG CSV. Expected format:
    - Each row = one EEG sample with signal columns + 'label' column.
    - If dataset is raw signals (many cols), extract features per row.
    """
    print(f"[+] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Separate label
    if 'label' in df.columns:
        y_raw = df['label'].values
        X_raw = df.drop(columns=['label']).values
    elif 'y' in df.columns:
        y_raw = df['y'].values
        X_raw = df.drop(columns=['y']).values
    else:
        # Assume last column is label
        y_raw = df.iloc[:, -1].values
        X_raw = df.iloc[:, :-1].values

    print(f"[+] Dataset shape: {X_raw.shape}, Classes: {np.unique(y_raw)}")

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Extract features from each row (each row = a signal segment)
    print("[+] Extracting features from EEG signal rows...")
    X_feats = np.array([extract_all_features(row) for row in X_raw])
    print(f"[+] Feature matrix shape: {X_feats.shape}")

    return X_feats, y, le

# ─────────────────────────────────────────────────────────────
# 2. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, classes):
    """Compute and print metrics. Returns metrics dict."""
    avg = 'binary' if len(classes) == 2 else 'macro'
    y_pred = model.predict(X_test)

    metrics = {
        'name':      name,
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        'cm':        confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n[{name}] Accuracy: {metrics['accuracy']:.4f} | "
          f"F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0))
    return metrics

# ─────────────────────────────────────────────────────────────
# 3. PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def save_confusion_matrix(cm, classes, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, f'cm_{model_name.replace(" ","_")}.png')
    plt.savefig(path, dpi=100)
    plt.close()
    return path

# ─────────────────────────────────────────────────────────────
# 4. DEEP LEARNING: BiLSTM
# ─────────────────────────────────────────────────────────────

def train_bilstm(X_train, X_test, y_train, y_test, n_classes):
    """Train a BiLSTM model on feature sequences."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential 
        from tensorflow.keras.layers import (Bidirectional, LSTM, Dense,
                                             Dropout, BatchNormalization)
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping

        n_features = X_train.shape[1]
        # Reshape for LSTM: (samples, timesteps, features)
        X_tr = X_train.reshape(-1, 1, n_features)
        X_te = X_test.reshape(-1, 1, n_features)

        if n_classes > 2:
            y_tr = to_categorical(y_train, n_classes)
            y_te = to_categorical(y_test, n_classes)
            loss, out_act, out_units = 'categorical_crossentropy', 'softmax', n_classes
        else:
            y_tr, y_te = y_train, y_test
            loss, out_act, out_units = 'binary_crossentropy', 'sigmoid', 1

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, n_features)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(out_units, activation=out_act)
        ])
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=30, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=0)

        # Predict
        raw = model.predict(X_te, verbose=0)
        if n_classes > 2:
            y_pred = np.argmax(raw, axis=1)
        else:
            y_pred = (raw.squeeze() > 0.5).astype(int)

        # Save keras model
        model.save(os.path.join(MODEL_DIR, 'bilstm_model.keras'))
        print("[+] BiLSTM model saved.")
        return y_pred
    except Exception as e:
        print(f"[!] BiLSTM training failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 5. DEEP LEARNING: CNN on Spectrogram
# ─────────────────────────────────────────────────────────────

def train_cnn(X_raw_train, X_raw_test, y_train, y_test, n_classes, fs=256.0):
    """
    Generate spectrograms from raw signals and train a CNN.
    X_raw: 2D array (samples x signal_length)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                             Dense, Dropout, BatchNormalization)
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping

        IMG_H, IMG_W = 64, 64

        def to_spectrogram(sig):
            """Convert 1D signal to a fixed-size spectrogram image."""
            f, t, Sxx = matplotlib.mlab.specgram(sig, Fs=fs, NFFT=64, noverlap=32)
            # Resize to IMG_H x IMG_W using simple interpolation
            from PIL import Image
            img = np.log1p(Sxx)
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((IMG_W, IMG_H)))
            return img / 255.0

        print("[+] Generating spectrograms for CNN...")
        X_tr_spec = np.array([to_spectrogram(row) for row in X_raw_train])[..., np.newaxis]
        X_te_spec = np.array([to_spectrogram(row) for row in X_raw_test])[..., np.newaxis]

        if n_classes > 2:
            y_tr = to_categorical(y_train, n_classes)
            y_te = to_categorical(y_test, n_classes)
            loss, out_act, out_units = 'categorical_crossentropy', 'softmax', n_classes
        else:
            y_tr, y_te = y_train, y_test
            loss, out_act, out_units = 'binary_crossentropy', 'sigmoid', 1

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(IMG_H, IMG_W, 1)),
            BatchNormalization(), MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(), MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'), Dropout(0.4),
            Dense(out_units, activation=out_act)
        ])
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_tr_spec, y_tr, epochs=20, batch_size=16,
                  validation_split=0.1, callbacks=[es], verbose=0)

        raw = model.predict(X_te_spec, verbose=0)
        y_pred = np.argmax(raw, axis=1) if n_classes > 2 else (raw.squeeze() > 0.5).astype(int)

        model.save(os.path.join(MODEL_DIR, 'cnn_model.keras'))
        print("[+] CNN model saved.")
        return y_pred
    except Exception as e:
        print(f"[!] CNN training failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 6. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG Model Trainer")
    parser.add_argument('--dataset', type=str, default='data/eeg_dataset.csv',
                        help='Path to EEG CSV dataset')
    args = parser.parse_args()

    # Load data and extract features
    X, y, le = load_and_prepare(args.dataset)
    n_classes = len(le.classes_)
    classes   = le.classes_

    # Save label encoder
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    all_metrics = []

    # ── Random Forest ──
    print("\n[+] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m = evaluate_model("Random Forest", rf, X_test, y_test, classes)
    all_metrics.append(m)
    save_confusion_matrix(np.array(m['cm']), classes, "Random Forest")
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))

    # ── SVM ──
    print("\n[+] Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    m = evaluate_model("SVM", svm, X_test, y_test, classes)
    all_metrics.append(m)
    save_confusion_matrix(np.array(m['cm']), classes, "SVM")
    joblib.dump(svm, os.path.join(MODEL_DIR, 'svm_model.pkl'))

    # ── XGBoost ──
    if XGBOOST_AVAILABLE:
        print("\n[+] Training XGBoost...")
        xgb = XGBClassifier(n_estimators=200, use_label_encoder=False,
                             eval_metric='mlogloss', random_state=42)
        xgb.fit(X_train, y_train)
        m = evaluate_model("XGBoost", xgb, X_test, y_test, classes)
        all_metrics.append(m)
        save_confusion_matrix(np.array(m['cm']), classes, "XGBoost")
        joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

    # ── BiLSTM ──
    print("\n[+] Training BiLSTM...")
    bilstm_pred = train_bilstm(X_train, X_test, y_train, y_test, n_classes)
    if bilstm_pred is not None:
        avg = 'binary' if n_classes == 2 else 'macro'
        m = {
            'name':      'BiLSTM',
            'accuracy':  round(float(accuracy_score(y_test, bilstm_pred)), 4),
            'precision': round(float(precision_score(y_test, bilstm_pred, average=avg, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_test, bilstm_pred, average=avg, zero_division=0)), 4),
            'f1':        round(float(f1_score(y_test, bilstm_pred, average=avg, zero_division=0)), 4),
            'cm':        confusion_matrix(y_test, bilstm_pred).tolist(),
        }
        all_metrics.append(m)

    # ── Pick best model by F1 ──
    best = max(all_metrics, key=lambda x: x['f1'])
    print(f"\n[✓] Best model: {best['name']} | F1: {best['f1']}")

    # Save metrics JSON (used by Flask dashboard)
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Save best model name
    with open(os.path.join(MODEL_DIR, 'best_model.txt'), 'w') as f:
        f.write(best['name'])

    # Save scaler separately for prediction
    print("\n[✓] Training complete. All models saved to /models/")
    print("[✓] Model metrics saved to models/metrics.json")

if __name__ == '__main__':
    main()
