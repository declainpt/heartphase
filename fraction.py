"""
Heartphase 0.0.4
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load ECG data
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

def phase_space_reconstruct(signal, dim, tau):
    n = len(signal)
    reconstructed = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        reconstructed[:, i] = signal[i * tau:n - (dim - 1) * tau + i * tau]
    return reconstructed

def create_interactive_plots(fig, reconstructed_data, ecg_trace, peaks, elev, azim):
    # Main 3D phase space plot
    ax1 = fig.add_subplot(111, projection='3d')
    
    # Small 1D ECG plot in bottom right
    ax2 = fig.add_axes([0.16, 0.6, 0.25, 0.15])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.tick_params(axis='both', which='both', length=0)
    
    lines = []
    for i, peak in enumerate(peaks):
        start = max(0, peak - sampling_rate // 2)
        end = min(len(reconstructed_data), peak + sampling_rate // 2)
        line, = ax1.plot(reconstructed_data[start:end, 0], 
                         reconstructed_data[start:end, 1], 
                         reconstructed_data[start:end, 2], 
                         lw=1, color='black', picker=5)
        lines.append(line)

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

    label = ax1.text2D(0.05, 0.95, "", transform=ax1.transAxes)

    selected_line = None
    animation = None
    is_animating = False

    def animate(frame, line, ecg_line, start, end):
        nonlocal is_animating
        line.set_data(reconstructed_data[start:start+frame, 0], reconstructed_data[start:start+frame, 1])
        line.set_3d_properties(reconstructed_data[start:start+frame, 2])
        
        time = np.arange(start, start+frame) / sampling_rate
        ecg_line.set_data(time, ecg_trace[start:start+frame])
        
        ax2.relim()
        ax2.autoscale_view()
        
        if frame == end - start - 1:
            is_animating = False
        
        return line, ecg_line

    def on_pick(event):
        nonlocal selected_line, animation, is_animating
        line = event.artist
        if line in lines and not is_animating:
            heartbeat_index = lines.index(line)
            if line == selected_line:
                # Deselect if clicking the same line
                reset_view()
            else:
                # Select the new line and hide others
                for l in lines:
                    if l != line:
                        l.set_visible(False)
                line.set_color('#ff4500')
                line.set_visible(True)
                selected_line = line
                
                # Plot the corresponding 1D ECG segment
                peak = peaks[heartbeat_index]
                start = max(0, peak - sampling_rate // 2)
                end = min(len(ecg_trace), peak + sampling_rate // 2)
                time = np.arange(start, end) / sampling_rate
                ax2.clear()
                ecg_line, = ax2.plot([], [], color='#ff4500', linewidth=0.8)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude')
                ax2.set_title(f'Heartbeat {heartbeat_index + 1}', fontsize=10)
                ax2.tick_params(axis='both', which='both', length=0)
                ax2.set_xlim(time[0], time[-1])
                ax2.set_ylim(min(ecg_trace[start:end]), max(ecg_trace[start:end]))
                ax2.set_visible(True)
                
                # Create animation
                frames = end - start
                is_animating = True
                animation = FuncAnimation(fig, animate, frames=frames, fargs=(line, ecg_line, start, end),
                                          interval=1000/sampling_rate, blit=True, repeat=False)
            
            fig.canvas.draw_idle()

    def on_release(event):
        nonlocal selected_line, is_animating
        if selected_line and not is_animating:
            reset_view()

    def reset_view():
        nonlocal selected_line, animation, is_animating
        if animation and hasattr(animation, 'event_source') and animation.event_source:
            animation.event_source.stop()
        animation = None
        
        for line in lines:
            line.set_visible(True)
            line.set_color('black')
        
        selected_line = None
        ax2.set_visible(False)
        label.set_text("")
        is_animating = False
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_release_event', on_release)

# Find peaks in the ECG data
peaks = find_heartbeats(ecg_trace, sampling_rate)

# Print the number of detected heartbeats
num_heartbeats = len(peaks)
print(f"Number of detected heartbeats: {num_heartbeats}")

# Segment the data into heartbeats and create time-reversed versions
heartbeat_segments, reversed_heartbeat_segments = segment_heartbeats(ecg_trace, peaks, 2 * sampling_rate // 2)  # 1-second window

# Parameters for phase space reconstruction
dimension = 3
time_delay = 20

# Perform phase space reconstruction
reconstructed_data = phase_space_reconstruct(ecg_trace, dimension, time_delay)

# Create the interactive plot
fig = plt.figure(figsize=(12, 10))
create_interactive_plots(fig, reconstructed_data, ecg_trace, peaks, elev=20, azim=45)
fig.text(0.5, 0.94, "Heartphase Version 0.0.4", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
# fig.text(0.98, 0.98, f"Number of Heartbeats: {num_heartbeats}", ha='right', va='top', fontsize=14, color='gray', alpha=0.2)
fig.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=344, color='gray', alpha=0.2, zorder=0)
fig.text(0.5, 0.2, "@Heartphase is Made for Life in Great Britain.", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)

plt.show()