"""
generate_sample_dataset.py
Generates a synthetic EEG CSV dataset for testing.
Usage: python generate_sample_dataset.py

Creates: data/eeg_dataset.csv
Compatible with: Epileptic Seizure Recognition Dataset format (178 signal features + label)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs('data', exist_ok=True)

N_SAMPLES   = 500   # samples per class (increase for better accuracy)
N_FEATURES  = 178   # signal points per sample (like the Bonn/Epileptic dataset)
FS          = 256   # simulated sampling frequency

def generate_eeg_signal(label: int, n_points: int) -> np.ndarray:
    """
    Simulate EEG signals for different neurological states.
    label 1: Epileptic seizure — high amplitude, high freq
    label 2: Tumor-related   — irregular, moderate amp
    label 3: Healthy control — low amplitude, regular
    label 4: Eyes open       — alpha waves (~10 Hz)
    label 5: Eyes closed     — stronger alpha
    """
    t = np.linspace(0, n_points / FS, n_points)

    if label == 1:  # Epileptic — chaotic high amplitude
        sig = (80 * np.sin(2*np.pi*25*t) +
               50 * np.sin(2*np.pi*40*t) +
               30 * np.random.randn(n_points))
    elif label == 2:
        sig = (40 * np.sin(2*np.pi*8*t) +
               20 * np.sin(2*np.pi*15*t) +
               25 * np.random.randn(n_points))
    elif label == 3:
        sig = (10 * np.sin(2*np.pi*1*t) +
               5  * np.random.randn(n_points))
    elif label == 4:  # Eyes open — alpha
        sig = (20 * np.sin(2*np.pi*10*t) +
               8  * np.random.randn(n_points))
    else:             # Eyes closed — stronger alpha
        sig = (30 * np.sin(2*np.pi*9*t) +
               10 * np.random.randn(n_points))

    return sig

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
print(f"[+] Sample dataset generated: {out}")
print(f"    Shape: {df.shape} | Classes: {df['label'].value_counts().to_dict()}")
