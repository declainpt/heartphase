"""
Heartphase 0.0.6
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt

def load_labeled_ecg_data(file_path):
    ecg_data = pd.read_csv(file_path)
    ecg_trace = ecg_data['ECG Signal'].values
    q_points = np.where(ecg_data['Q-Point'] == 1)[0]
    r_peaks = np.where(ecg_data['R-Peak'] == 1)[0]
    s_points = np.where(ecg_data['S-Point'] == 1)[0]
    p_starts = np.where(ecg_data['P-Wave Start'] == 1)[0]
    p_peaks = np.where(ecg_data['P-Wave Peak'] == 1)[0]
    p_ends = np.where(ecg_data['P-Wave End'] == 1)[0]
    t_starts = np.where(ecg_data['T-Wave Start'] == 1)[0]
    t_peaks = np.where(ecg_data['T-Wave Peak'] == 1)[0]
    t_ends = np.where(ecg_data['T-Wave End'] == 1)[0]
    
    return ecg_trace, q_points, r_peaks, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends

sampling_rate = 512  # Hz

"""
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
"""

def segment_heartbeats(ecg_trace, r_peaks, window_size, q_points, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends):
    segments = []
    qrs_complexes = []
    p_waves = []
    t_waves = []
    for i, r_peak in enumerate(r_peaks):
        if r_peak - window_size//2 >= 0 and r_peak + window_size//2 < len(ecg_trace):
            segment = ecg_trace[r_peak - window_size//2:r_peak + window_size//2]
            segments.append(segment)
            
            q = q_points[i] - (r_peak - window_size//2)
            s = s_points[i] - (r_peak - window_size//2)
            qrs_complexes.append((q, window_size//2, s))
            
            p_start = p_starts[i] - (r_peak - window_size//2)
            p_peak = p_peaks[i] - (r_peak - window_size//2)
            p_end = p_ends[i] - (r_peak - window_size//2)
            p_waves.append((p_start, p_peak, p_end))
            
            t_start = t_starts[i] - (r_peak - window_size//2)
            t_peak = t_peaks[i] - (r_peak - window_size//2)
            t_end = t_ends[i] - (r_peak - window_size//2)
            t_waves.append((t_start, t_peak, t_end))
    
    return segments, qrs_complexes, p_waves, t_waves

def phase_space_reconstruct(signal, dim, tau):
    n = len(signal)
    reconstructed = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        reconstructed[:, i] = signal[i * tau:n - (dim - 1) * tau + i * tau]
    return reconstructed

def create_interactive_plots(fig, reconstructed_data, ecg_trace, r_peaks, qrs_complexes, p_waves, t_waves, elev, azim):
    ax1 = fig.add_subplot(111, projection='3d')
    
    ax2 = fig.add_axes([0.18, 0.6, 0.25, 0.15])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.tick_params(axis='both', which='both', length=0)
    
    lines = []
    for r_peak in r_peaks:
        start = max(0, r_peak - sampling_rate // 2)
        end = min(len(reconstructed_data), r_peak + sampling_rate // 2)
        line, = ax1.plot(reconstructed_data[start:end, 0], 
                         reconstructed_data[start:end, 1], 
                         reconstructed_data[start:end, 2],
                         lw=1, color='black', alpha=1.0, picker=5)
        lines.append(line)

    highlight_line, = ax1.plot([], [], [], lw=2, color='black', alpha=1.0)

    # Create lines for QRS, P, and T waves in 3D
    qrs_3d_line, = ax1.plot([], [], [], lw=2, color='#ff4500', visible=False)
    p_3d_line, = ax1.plot([], [], [], lw=2, color='#4169e1', visible=False)
    t_3d_line, = ax1.plot([], [], [], lw=2, color='#00A86B', visible=False)

    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('none')
    ax1.yaxis.pane.set_edgecolor('none')
    ax1.zaxis.pane.set_edgecolor('none')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.view_init(elev=elev, azim=azim)

    ax2.set_visible(False)

    selected_line = None
    animation = None
    is_animating = False
    duration_text = None

    def animate(frame, highlight_line, qrs_3d_line, p_3d_line, t_3d_line, ecg_line, qrs_2d_line, p_2d_line, t_2d_line, start, end, qrs, p_wave, t_wave):
        nonlocal is_animating
        highlight_line.set_data(reconstructed_data[start:start+frame, 0], reconstructed_data[start:start+frame, 1])
        highlight_line.set_3d_properties(reconstructed_data[start:start+frame, 2])
        
        q, r, s = qrs
        p_start, p_peak, p_end = p_wave
        t_start, t_peak, t_end = t_wave
        
        highlight_line.set_color('black')
        
        if frame > q:
            qrs_end = min(frame, s)
            qrs_3d_line.set_data(reconstructed_data[start+q:start+qrs_end, 0], reconstructed_data[start+q:start+qrs_end, 1])
            qrs_3d_line.set_3d_properties(reconstructed_data[start+q:start+qrs_end, 2])
            qrs_3d_line.set_visible(True)
        else:
            qrs_3d_line.set_visible(False)
        
        if frame > p_start:
            p_end = min(frame, p_end)
            p_3d_line.set_data(reconstructed_data[start+p_start:start+p_end, 0], reconstructed_data[start+p_start:start+p_end, 1])
            p_3d_line.set_3d_properties(reconstructed_data[start+p_start:start+p_end, 2])
            p_3d_line.set_visible(True)
        else:
            p_3d_line.set_visible(False)
        
        if frame > t_start:
            t_end = min(frame, t_end)
            t_3d_line.set_data(reconstructed_data[start+t_start:start+t_end, 0], reconstructed_data[start+t_start:start+t_end, 1])
            t_3d_line.set_3d_properties(reconstructed_data[start+t_start:start+t_end, 2])
            t_3d_line.set_visible(True)
        else:
            t_3d_line.set_visible(False)
        
        time = np.arange(start, start+frame) / sampling_rate
        ecg_line.set_data(time, ecg_trace[start:start+frame])
        
        if frame > q:
            qrs_end = min(frame, s)
            qrs_2d_line.set_data(time[q:qrs_end], ecg_trace[start+q:start+qrs_end])
        else:
            qrs_2d_line.set_data([], [])
        
        if frame > p_start:
            p_end = min(frame, p_end)
            p_2d_line.set_data(time[p_start:p_end], ecg_trace[start+p_start:start+p_end])
        else:
            p_2d_line.set_data([], [])
        
        if frame > t_start:
            t_end = min(frame, t_end)
            t_2d_line.set_data(time[t_start:t_end], ecg_trace[start+t_start:start+t_end])
        else:
            t_2d_line.set_data([], [])
        
        ax2.relim()
        ax2.autoscale_view()
        
        if frame == end - start - 1:
            is_animating = False
        
        return highlight_line, qrs_3d_line, p_3d_line, t_3d_line, ecg_line, qrs_2d_line, p_2d_line, t_2d_line

    def on_pick(event):
        nonlocal selected_line, animation, is_animating, duration_text
        line = event.artist
        if line in lines and not is_animating:
            # Reset opacity for all lines
            for l in lines:
                l.set_alpha(0.04)
            
            line.set_alpha(1.0)
            
            heartbeat_index = lines.index(line)
            
            r_peak = r_peaks[heartbeat_index]
            start = max(0, r_peak - sampling_rate // 2)
            end = min(len(reconstructed_data), r_peak + sampling_rate // 2)
            highlight_line.set_data(reconstructed_data[start:end, 0], reconstructed_data[start:end, 1])
            highlight_line.set_3d_properties(reconstructed_data[start:end, 2])
            
            time = np.arange(start, end) / sampling_rate
            ax2.clear()
            ecg_line, = ax2.plot([], [], color='black', linewidth=0.8)
            qrs_2d_line, = ax2.plot([], [], color='#ff4500', linewidth=1.2)
            p_2d_line, = ax2.plot([], [], color='#4169e1', linewidth=1.2)
            t_2d_line, = ax2.plot([], [], color='#00A86B', linewidth=1.2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title(f'Heartbeat {heartbeat_index + 1}', fontsize=10)
            ax2.tick_params(axis='both', which='both', length=0)
            ax2.set_xlim(time[0], time[-1])
            ax2.set_ylim(min(ecg_trace[start:end]), max(ecg_trace[start:end]))
            ax2.set_visible(True)
            
            p_start, _, p_end = p_waves[heartbeat_index]
            q, _, s = qrs_complexes[heartbeat_index]
            t_start, _, t_end = t_waves[heartbeat_index]
            
            p_duration = (p_end - p_start) / sampling_rate * 1000
            qrs_duration = (s - q) / sampling_rate * 1000
            t_duration = (t_end - t_start) / sampling_rate * 1000
            
            duration_info = f"P-wave: {p_duration:.1f} ms\nQRS: {qrs_duration:.1f} ms\nT-wave: {t_duration:.1f} ms"
            
            duration_text = ax2.text(0.02, 0.95, duration_info, transform=ax2.transAxes, 
                                     verticalalignment='top', fontsize=8, 
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
            
            frames = end - start
            qrs = qrs_complexes[heartbeat_index]
            p_wave = p_waves[heartbeat_index]
            t_wave = t_waves[heartbeat_index]
            is_animating = True
            animation = FuncAnimation(fig, animate, frames=frames, 
                                      fargs=(highlight_line, qrs_3d_line, p_3d_line, t_3d_line,
                                             ecg_line, qrs_2d_line, p_2d_line, t_2d_line, 
                                             start, end, qrs, p_wave, t_wave),
                                      interval=1000/sampling_rate, blit=True, repeat=False)
            
            fig.canvas.draw_idle()

    def reset_view():
        nonlocal selected_line, animation, is_animating, duration_text
        if animation and hasattr(animation, 'event_source') and animation.event_source:
            animation.event_source.stop()
        animation = None
        
        for line in lines:
            line.set_alpha(1.0)
        
        highlight_line.set_data([], [])
        highlight_line.set_3d_properties([])
        qrs_3d_line.set_data([], [])
        qrs_3d_line.set_3d_properties([])
        p_3d_line.set_data([], [])
        p_3d_line.set_3d_properties([])
        t_3d_line.set_data([], [])
        t_3d_line.set_3d_properties([])
        
        selected_line = None
        ax2.set_visible(False)
        ax2.clear()
        is_animating = False
        
        if duration_text:
            duration_text.remove()
            duration_text = None
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_release_event', lambda event: reset_view())

    return lines

# Load the labeled ECG data
file_path = 'ecg-sample-pqrst.csv'
ecg_trace, q_points, r_peaks, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends = load_labeled_ecg_data(file_path)

# Print the number of detected heartbeats
num_heartbeats = len(r_peaks)
print(f"Number of detected heartbeats: {num_heartbeats}")

# Segment the data into heartbeats and use the labeled QRS complexes, P waves, and T waves
heartbeat_segments, qrs_complexes, p_waves, t_waves = segment_heartbeats(ecg_trace, r_peaks, sampling_rate, q_points, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends)

# Parameters for phase space reconstruction
dimension = 3
time_delay = 20

reconstructed_data = phase_space_reconstruct(ecg_trace, dimension, time_delay)

fig = plt.figure(figsize=(12, 10))
create_interactive_plots(fig, reconstructed_data, ecg_trace, r_peaks, qrs_complexes, p_waves, t_waves, elev=20, azim=45)

fig.text(0.5, 0.94, "Heartphase Version 0.0.6", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig.text(0.9, 0.9, f"Number of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
fig.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=344, color='gray', alpha=0.2, zorder=0)
fig.text(0.5, 0.2, "@Heartphase is Made for Life in Great Britain.", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)

plt.show()