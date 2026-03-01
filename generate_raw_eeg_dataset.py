import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1000      # total rows
n_features = 178     # EEG time points per segment

def generate_eeg_segment(label):
    if label == 1:  # Normal EEG
        signal = np.random.normal(0, 50, n_features)
    else:  # Epileptic EEG (higher spikes)
        signal = np.random.normal(0, 100, n_features)
        spike_indices = np.random.choice(n_features, size=10, replace=False)
        signal[spike_indices] += np.random.normal(300, 50, 10)

    return signal

data = []

for i in range(n_samples):
    label = 1 if i < n_samples // 2 else 2
    segment = generate_eeg_segment(label)
    row = list(segment) + [label]
    data.append(row)

columns = [f"col{i+1}" for i in range(n_features)] + ["label"]

df = pd.DataFrame(data, columns=columns)

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("data/eeg_raw_dataset.csv", index=False)

print("Raw EEG dataset generated successfully!")