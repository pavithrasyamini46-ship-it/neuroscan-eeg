import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 500       # number of EEG segments
n_points = 178        # time points per segment

def generate_normal_eeg():
    t = np.linspace(0, 1, n_points)
    alpha_wave = 50 * np.sin(2 * np.pi * 10 * t)   # 10 Hz alpha rhythm
    noise = np.random.normal(0, 10, n_points)
    return alpha_wave + noise

def generate_epileptic_eeg():
    t = np.linspace(0, 1, n_points)
    base_wave = 30 * np.sin(2 * np.pi * 5 * t)  # slower rhythm
    noise = np.random.normal(0, 20, n_points)

    # Add seizure spikes
    spikes = np.zeros(n_points)
    spike_positions = np.random.choice(range(20, 160), size=8, replace=False)
    for pos in spike_positions:
        spikes[pos:pos+3] += np.random.uniform(200, 400)

    return base_wave + noise + spikes

data = []

# Generate normal samples (label = 1)
for _ in range(n_samples):
    signal = generate_normal_eeg()
    row = list(signal) + [1]
    data.append(row)

# Generate epileptic samples (label = 2)
for _ in range(n_samples):
    signal = generate_epileptic_eeg()
    row = list(signal) + [2]
    data.append(row)

columns = [f"col{i+1}" for i in range(n_points)] + ["label"]

df = pd.DataFrame(data, columns=columns)
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("data/epileptic_eeg_dataset.csv", index=False)

print("Epileptic EEG dataset generated successfully!")