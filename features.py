"""
features.py - EEG Feature Extraction Module
Extracts Time Domain and Frequency Domain features from EEG signals.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy as scipy_entropy

# ─────────────────────────────────────────────────────────────
# FIR BANDPASS FILTER
# ─────────────────────────────────────────────────────────────

def apply_fir_filter(data: np.ndarray, fs: float = 256.0,
                     lowcut: float = 0.5, highcut: float = 50.0,
                     numtaps: int = 51) -> np.ndarray:
    """
    Apply FIR bandpass filter to remove noise from EEG signal.
    fs: sampling frequency (Hz)
    lowcut/highcut: passband edges in Hz
    """
    nyq = 0.5 * fs
    taps = signal.firwin(numtaps, [lowcut / nyq, highcut / nyq],
                         pass_zero=False, window='hamming')
    filtered = signal.lfilter(taps, 1.0, data)
    return filtered

# ─────────────────────────────────────────────────────────────
# TIME DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────

def compute_hjorth_parameters(signal_arr: np.ndarray):
    """
    Compute Hjorth parameters: Activity, Mobility, Complexity.
    Activity  = variance of signal
    Mobility  = sqrt(var(1st derivative) / var(signal))
    Complexity = Mobility(1st derivative) / Mobility(signal)
    """
    activity = np.var(signal_arr)
    diff1 = np.diff(signal_arr)
    diff2 = np.diff(diff1)

    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))) / (mobility + 1e-10)

    return activity, mobility, complexity

def extract_time_domain_features(signal_arr: np.ndarray) -> dict:
    """Extract time domain statistical features from a 1D EEG segment."""
    feats = {}
    feats['mean']     = np.mean(signal_arr)
    feats['variance'] = np.var(signal_arr)
    feats['std']      = np.std(signal_arr)
    feats['energy']   = np.sum(signal_arr ** 2)

    # Entropy using histogram-based probability
    hist, _ = np.histogram(signal_arr, bins=10, density=True)
    hist = hist + 1e-10  # avoid log(0)
    feats['entropy'] = scipy_entropy(hist)

    # Hjorth Parameters
    activity, mobility, complexity = compute_hjorth_parameters(signal_arr)
    feats['hjorth_activity']   = activity
    feats['hjorth_mobility']   = mobility
    feats['hjorth_complexity'] = complexity

    return feats

# ─────────────────────────────────────────────────────────────
# FREQUENCY DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────

EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}

def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Compute band power for a given frequency range using trapezoidal integration."""
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])

def spectral_entropy(psd: np.ndarray) -> float:
    """Compute spectral entropy from PSD."""
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return scipy_entropy(psd_norm + 1e-10)

def extract_frequency_domain_features(signal_arr: np.ndarray,
                                       fs: float = 256.0) -> dict:
    """
    Extract frequency domain features using FFT.
    Returns band power for Delta, Theta, Alpha, Beta and spectral entropy.
    """
    n = len(signal_arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal_arr)) ** 2  # Power spectrum

    feats = {}
    for band_name, (low, high) in EEG_BANDS.items():
        feats[f'bp_{band_name}'] = band_power(freqs, fft_vals, low, high)

    feats['spectral_entropy'] = spectral_entropy(fft_vals)
    return feats

# ─────────────────────────────────────────────────────────────
# COMBINED FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_all_features(signal_arr: np.ndarray, fs: float = 256.0,
                          apply_filter: bool = True) -> np.ndarray:
    """
    Apply FIR filter, then extract all time + frequency domain features.
    Returns a 1D numpy array of features.
    """
    if apply_filter:
        signal_arr = apply_fir_filter(signal_arr, fs=fs)

    td = extract_time_domain_features(signal_arr)
    fd = extract_frequency_domain_features(signal_arr, fs=fs)

    # Merge and convert to array (preserving consistent order)
    all_feats = {**td, **fd}
    return np.array(list(all_feats.values()), dtype=np.float32)

def get_feature_names() -> list:
    """Return list of feature names (must match extract_all_features order)."""
    time_names = ['mean', 'variance', 'std', 'energy', 'entropy',
                  'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
    freq_names = ['bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'spectral_entropy']
    return time_names + freq_names
