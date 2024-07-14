"""
Heartphase 0.0.6
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

ecg_data = pd.read_csv('ecg-sample.csv')
ecg_trace = ecg_data.iloc[:, 0].values

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
    segments = [ecg_trace[p - window_size//2:p + window_size//2] for p in peaks if p - window_size//2 >= 0 and p + window_size//2 < len(ecg_trace)]
    reversed_segments = [seg[::-1] for seg in segments] 
    return segments, reversed_segments

# Compute pairwise phase locking value (PLV) of analytic signals of segmented heartbeats (after Hilbert transform)
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
def plot_heartbeat_and_analytic_signal(heartbeat, title_prefix=""):
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
    ax1.set_title(f'{title_prefix}Raw ECG Signal')
    ax1.legend()
    
    # Analytic signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.real(analytic_signal), label='Real part', color='black')
    ax2.plot(t, np.imag(analytic_signal), label='Imaginary part', color='green')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'{title_prefix}Analytic Signal')
    ax2.legend()
    
    # Instantaneous phase
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, instantaneous_phase, label='Instantaneous phase', color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase')
    ax3.set_title(f'{title_prefix}Instantaneous Phase')
    ax3.legend()
    
    # Amplitude envelope
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, amplitude_envelope, label='Amplitude envelope', color='red')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title(f'{title_prefix}Amplitude Envelope')
    ax4.legend()
    
    # Polar plot
    ax5 = fig.add_subplot(gs[:, 2], projection='polar')
    theta = instantaneous_phase
    r = amplitude_envelope
    ax5.plot(theta, r, color='black')
    ax5.set_title(f'{title_prefix}Polar Plot of Analytic Signal')
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

# Segment the data into heartbeats and create time-reversed versions
heartbeat_segments, reversed_heartbeat_segments = segment_heartbeats(ecg_trace, peaks, 2 * sampling_rate // 2)  # 1-second window

# Calculate raw and normalised PLV for all pairs of normal heartbeats
all_raw_plvs = []
all_norm_plvs = []
for i in range(len(heartbeat_segments)):
    for j in range(i + 1, len(heartbeat_segments)):
        phase_diff = compute_phase_diff(heartbeat_segments[i], heartbeat_segments[j])
        raw = raw_plv(phase_diff)
        norm = normalised_plv(phase_diff, 2)
        all_raw_plvs.append(raw)
        all_norm_plvs.append(norm)

# Compute overall raw and normalised PLV for normal heartbeats
overall_phase_diff = []
for i in range(len(heartbeat_segments) - 1):
    for j in range(i + 1, len(heartbeat_segments)):
        phase_diff = compute_phase_diff(heartbeat_segments[i], heartbeat_segments[j])
        overall_phase_diff.extend(phase_diff)

overall_phase_diff = np.array(overall_phase_diff)
raw_plv_all = raw_plv(overall_phase_diff)
norm_plv_all = normalised_plv(overall_phase_diff, num_heartbeats)

# Print the raw and normalised PLVs for normal heartbeats
print("Normal Heartbeats:")
print(f"Raw PLV for each pair of heartbeats: {all_raw_plvs}")
print(f"Normalised PLV for each pair of heartbeats: {all_norm_plvs}")
print(f"Overall raw PLV for all heartbeats: {raw_plv_all}")
print(f"Overall normalised PLV for all heartbeats: {norm_plv_all}")

# Calculate raw and normalised PLV for all pairs of time-reversed heartbeats
reversed_all_raw_plvs = []
reversed_all_norm_plvs = []
for i in range(len(reversed_heartbeat_segments)):
    for j in range(i + 1, len(reversed_heartbeat_segments)):
        phase_diff = compute_phase_diff(reversed_heartbeat_segments[i], reversed_heartbeat_segments[j])
        raw = raw_plv(phase_diff)
        norm = normalised_plv(phase_diff, 2)
        reversed_all_raw_plvs.append(raw)
        reversed_all_norm_plvs.append(norm)

# Compute overall raw and normalised PLV for time-reversed heartbeats
reversed_overall_phase_diff = []
for i in range(len(reversed_heartbeat_segments) - 1):
    for j in range(i + 1, len(reversed_heartbeat_segments)):
        phase_diff = compute_phase_diff(reversed_heartbeat_segments[i], reversed_heartbeat_segments[j])
        reversed_overall_phase_diff.extend(phase_diff)

reversed_overall_phase_diff = np.array(reversed_overall_phase_diff)
reversed_raw_plv_all = raw_plv(reversed_overall_phase_diff)
reversed_norm_plv_all = normalised_plv(reversed_overall_phase_diff, num_heartbeats)

# Print the raw and normalised PLVs for time-reversed heartbeats
print("\nTime-Reversed Heartbeats:")
print(f"Raw PLV for each pair of time-reversed heartbeats: {reversed_all_raw_plvs}")
print(f"Normalised PLV for each pair of time-reversed heartbeats: {reversed_all_norm_plvs}")
print(f"Overall raw PLV for all time-reversed heartbeats: {reversed_raw_plv_all}")
print(f"Overall normalised PLV for all time-reversed heartbeats: {reversed_norm_plv_all}")

def are_plvs_equal(plv1, plv2, tolerance=1e-12):
    return abs(plv1 - plv2) < tolerance

print("\nComparing PLVs:")
different_pairs = []
for i in range(len(all_raw_plvs)):
    if not are_plvs_equal(all_raw_plvs[i], reversed_all_raw_plvs[i]):
        different_pairs.append((i, all_raw_plvs[i], reversed_all_raw_plvs[i]))

if different_pairs:
    print(f"Found {len(different_pairs)} pairs with different PLVs:")
    for pair, normal_plv, reversed_plv in different_pairs:
        print(f"Pair {pair}: Normal PLV = {normal_plv:.15f}, Reversed PLV = {reversed_plv:.15f}")
else:
    print("All pairs have effectively identical PLVs.")

print("\nOverall comparison:")
if are_plvs_equal(raw_plv_all, reversed_raw_plv_all):
    print("Overall raw PLVs are effectively identical")
else:
    print(f"Overall raw PLVs differ: Normal = {raw_plv_all:.15f}, Reversed = {reversed_raw_plv_all:.15f}")

if are_plvs_equal(norm_plv_all, reversed_norm_plv_all):
    print("Overall normalized PLVs are effectively identical")
else:
    print(f"Overall normalized PLVs differ: Normal = {norm_plv_all:.15f}, Reversed = {reversed_norm_plv_all:.15f}")

# Parameters for phase space reconstruction
dimension = 3
time_delay = 20

# Perform phase space reconstruction
reconstructed_data = phase_space_reconstruct(ecg_trace, dimension, time_delay)

# Create the first figure with a single phase portrait
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
create_phase_space_plot(ax1, reconstructed_data, elev=20, azim=45)
fig1.text(0.5, 0.94, "Heartphase Version 0.0.6", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig1.text(0.9, 0.9, f"Coherence score: {norm_plv_all:.18f}\nTime-reversed coherence score: {reversed_norm_plv_all:.18f}\nNumber of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
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
fig2.text(0.5, 0.94, "Heartphase Version 0.0.6", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig2.text(0.9, 0.9, f"Coherence score: {norm_plv_all:.18f}\nTime-reversed coherence score: {reversed_norm_plv_all:.18f}\nNumber of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
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

    # Plot the time-reversed version of the first heartbeat
    first_reversed_heartbeat = reversed_heartbeat_segments[0]
    reversed_heartbeat_fig = plot_heartbeat_and_analytic_signal(first_reversed_heartbeat, title_prefix="Time-Reversed ")
    reversed_heartbeat_fig.savefig('time-reversed-hilbert-transform.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Image 'time-reversed-hilbert-transform.png' has been saved.")
"""

plt.show()