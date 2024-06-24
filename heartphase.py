"""
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load ECG data from CSV
ecg_data = pd.read_csv('ecg-sample.csv')
ecg_trace = ecg_data.iloc[:, 0].values

# Parameters
sampling_rate = 512  # Hz

def find_heartbeats(ecg_trace, sampling_rate):
    distance = sampling_rate // 2 
    peaks = []
    threshold = np.max(ecg_trace) * 0.5
    for i in range(1, len(ecg_trace) - 1):
        if ecg_trace[i] > threshold and ecg_trace[i] > ecg_trace[i - 1] and ecg_trace[i] > ecg_trace[i + 1]:
            if not peaks or i - peaks[-1] > distance:
                peaks.append(i)
    return peaks

def segment_heartbeats(ecg_trace, peaks, window_size):
    return [ecg_trace[p - window_size//2:p + window_size//2] for p in peaks if p - window_size//2 >= 0 and p + window_size//2 < len(ecg_trace)]

# Compute pairwise phase locking value (PLV) of analytic signals of segmented heartbeat (after Hilbert transform)
def compute_phase_diff(signal1, signal2):
    analytic_signal1 = signal.hilbert(signal1)
    analytic_signal2 = signal.hilbert(signal2)
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)
    min_length = min(len(phase1), len(phase2))
    phase_diff = np.unwrap(phase1[:min_length] - phase2[:min_length])
    return phase_diff

def raw_plv(phase_diff):
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def normalised_plv(phase_diff, n):
    raw = raw_plv(phase_diff)
    expected_plv = 1 / np.sqrt(n)
    return (raw - expected_plv) / (1 - expected_plv)

def phase_space_reconstruct(signal, dim, tau):
    n = len(signal)
    reconstructed = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        reconstructed[:, i] = signal[i * tau:n - (dim - 1) * tau + i * tau]
    return reconstructed

def create_phase_space_plot(ax, reconstructed_data, elev, azim, add_text=True):
    ax.plot(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], lw=1, color='black')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=elev, azim=azim)

"""
# Plot Hilbert transform
def plot_heartbeat_and_analytic_signal(heartbeat):
    analytic_signal = signal.hilbert(heartbeat)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    amplitude_envelope = np.abs(analytic_signal)
    t = np.arange(len(heartbeat)) / sampling_rate
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Raw signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, heartbeat, label='Raw signal', color='black')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Raw ECG Signal')
    ax1.legend()
    
    # Analytic signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.real(analytic_signal), label='Real part', color='black')
    ax2.plot(t, np.imag(analytic_signal), label='Imaginary part', color='green')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Analytic Signal')
    ax2.legend()
    
    # Instantaneous phase
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, instantaneous_phase, label='Instantaneous phase', color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase')
    ax3.set_title('Instantaneous Phase')
    ax3.legend()
    
    # Amplitude envelope
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, amplitude_envelope, label='Amplitude envelope', color='red')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Amplitude Envelope')
    ax4.legend()
    
    # Polar plot
    ax5 = fig.add_subplot(gs[:, 2], projection='polar')
    theta = instantaneous_phase
    r = amplitude_envelope
    ax5.plot(theta, r, color='black')
    ax5.set_title('Polar Plot of Analytic Signal')
    ax5.set_rticks([])  # Remove radial ticks
    
    # Remove top and right spines, set bottom and left spines to gray
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.tick_params(axis='both', colors='gray')
    
    plt.tight_layout()
    return fig
"""

# Find peaks in the ECG data
peaks = find_heartbeats(ecg_trace, sampling_rate)

# Print the number of detected heartbeats
num_heartbeats = len(peaks)
print(f"Number of detected heartbeats: {num_heartbeats}")

# Segment the data into heartbeats
heartbeat_segments = segment_heartbeats(ecg_trace, peaks, 2 * sampling_rate // 2)  # 1-second window

# Calculate raw and normalised PLV for all pairs
all_raw_plvs = []
all_norm_plvs = []
for i in range(len(heartbeat_segments)):
    for j in range(i + 1, len(heartbeat_segments)):
        phase_diff = compute_phase_diff(heartbeat_segments[i], heartbeat_segments[j])
        raw = raw_plv(phase_diff)
        norm = normalised_plv(phase_diff, 2)
        all_raw_plvs.append(raw)
        all_norm_plvs.append(norm)

# Compute overall raw and normalised PLV
overall_phase_diff = []
for i in range(len(heartbeat_segments) - 1):
    for j in range(i + 1, len(heartbeat_segments)):
        phase_diff = compute_phase_diff(heartbeat_segments[i], heartbeat_segments[j])
        overall_phase_diff.extend(phase_diff)

overall_phase_diff = np.array(overall_phase_diff)
raw_plv_all = raw_plv(overall_phase_diff)
norm_plv_all = normalised_plv(overall_phase_diff, num_heartbeats)

# Print the raw and normalised PLVs
print(f"Raw PLV for each pair of heartbeats: {all_raw_plvs}")
print(f"Normalised PLV for each pair of heartbeats: {all_norm_plvs}")
print(f"Overall raw PLV for all heartbeats: {raw_plv_all}")
print(f"Overall normalised PLV for all heartbeats: {norm_plv_all}")

# Parameters for phase space reconstruction
dimension = 3
time_delay = 20

# Perform phase space reconstruction
reconstructed_data = phase_space_reconstruct(ecg_trace, dimension, time_delay)

# Create the first figure with a single phase portrait
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
create_phase_space_plot(ax1, reconstructed_data, elev=20, azim=45)
fig1.text(0.5, 0.94, "Heartphase Version 0.0.1", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig1.text(0.9, 0.9, f"Coherence score: {norm_plv_all:.18f}\nNumber of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
fig1.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=344, color='gray', alpha=0.2, zorder=0)
fig1.text(0.5, 0.2, "@Heartphase is Made for Life in Great Britain.", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
plt.tight_layout()
plt.savefig('heartphase-single.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Image 'heartphase-single.png' has been saved.")

# Create the second figure with three phase portraits
fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(20, 8), subplot_kw={'projection': '3d'})
create_phase_space_plot(ax2, reconstructed_data, elev=20, azim=0, add_text=False)
create_phase_space_plot(ax3, reconstructed_data, elev=20, azim=45, add_text=False)
create_phase_space_plot(ax4, reconstructed_data, elev=20, azim=90, add_text=False)
fig2.text(0.5, 0.94, "Heartphase Version 0.0.1", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig2.text(0.9, 0.9, f"Coherence score: {norm_plv_all:.18f}\nNumber of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
fig2.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=344, color='gray', alpha=0.2, zorder=0)
fig2.text(0.5, 0.2, "@Heartphase is Made for Life in Great Britain.", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
plt.tight_layout()
plt.savefig('heartphase-triple.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Image 'heartphase-triple.png' has been saved.")

"""
# Plot the first heartbeat and its analytic signal
if heartbeat_segments:
    first_heartbeat = heartbeat_segments[0]
    heartbeat_fig = plot_heartbeat_and_analytic_signal(first_heartbeat)
    heartbeat_fig.savefig('hilbert-transform.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Image 'hilbert-transform.png' has been saved.")
"""
plt.show()