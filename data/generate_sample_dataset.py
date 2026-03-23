"""
generate_sample_dataset.py
Generates a synthetic EEG CSV dataset for testing.
Usage: python generate_sample_dataset.py

Creates: data/eeg_dataset.csv

Classes:
  1 = Epileptic Seizure
  2 = Tumor-Related
  3 = Healthy / Normal
  4 = Alzheimer's Disease
  5 = Parkinson's Disease
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs('data', exist_ok=True)

N_SAMPLES  = 500   # samples per class
N_FEATURES = 178   # signal points per sample
FS         = 256   # simulated sampling frequency

# Class label names (used for display in the web app)
CLASS_NAMES = {
    1: 'Epileptic',
    2: 'Tumor',
    3: 'Normal',
    4: 'Alzheimers',
    5: 'Parkinsons',
}

def generate_eeg_signal(label: int, n_points: int) -> np.ndarray:
    """
    Simulate EEG signals for different neurological conditions.

    label 1: Epileptic Seizure  — high amplitude, high freq chaotic bursts
    label 2: Tumor-Related      — irregular moderate amplitude
    label 3: Healthy / Normal   — low amplitude, regular slow waves
    label 4: Alzheimer's        — reduced alpha, increased theta/delta slowing
    label 5: Parkinson's        — tremor-related beta band oscillations (~20Hz)
    """
    t = np.linspace(0, n_points / FS, n_points)

    if label == 1:  # Epileptic — chaotic high amplitude spikes
        sig = (80 * np.sin(2*np.pi*25*t) +
               50 * np.sin(2*np.pi*40*t) +
               30 * np.random.randn(n_points))

    elif label == 2:  # Tumor — irregular moderate activity
        sig = (40 * np.sin(2*np.pi*8*t) +
               20 * np.sin(2*np.pi*15*t) +
               25 * np.random.randn(n_points))

    elif label == 3:  # Normal — clean low amplitude
        sig = (10 * np.sin(2*np.pi*10*t) +   # dominant alpha
               5  * np.sin(2*np.pi*2*t)  +   # small delta
               5  * np.random.randn(n_points))

    elif label == 4:  # Alzheimer's — cortical slowing: strong theta/delta, weak alpha
        sig = (35 * np.sin(2*np.pi*3*t)  +   # strong delta (0.5-4 Hz)
               25 * np.sin(2*np.pi*6*t)  +   # strong theta (4-8 Hz)
               5  * np.sin(2*np.pi*10*t) +   # weak alpha (reduced)
               15 * np.random.randn(n_points))

    else:  # Parkinson's — beta band oscillations (~20 Hz tremor)
        sig = (30 * np.sin(2*np.pi*20*t) +   # strong beta (tremor)
               15 * np.sin(2*np.pi*6*t)  +   # theta component
               10 * np.random.randn(n_points))

    return sig

# Generate dataset
rows = []
for label in range(1, 6):
    for _ in range(N_SAMPLES):
        sig = generate_eeg_signal(label, N_FEATURES)
        rows.append(np.append(sig, label))

cols = [f'X{i}' for i in range(1, N_FEATURES+1)] + ['label']
df = pd.DataFrame(rows, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = 'data/eeg_dataset.csv'
df.to_csv(out, index=False)

print(f"[+] Dataset generated: {out}")
print(f"    Shape : {df.shape}")
print(f"    Classes:")
for lbl, name in CLASS_NAMES.items():
    count = (df['label'] == lbl).sum()
    print(f"      {lbl} = {name:15s} → {count} samples")