"""
Heartphase 0.0.5
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

df1 = pd.read_csv('ecg-sample.csv')
df2 = pd.read_csv('ecg-sample-2.csv')
ecg_data1 = df1.iloc[:, 0].values
ecg_data2 = df2.iloc[:, 0].values

min_length = min(len(ecg_data1), len(ecg_data2))
ecg_data1 = ecg_data1[:min_length]
ecg_data2 = ecg_data2[:min_length]

# Single or dual
while True:
    choice = input("Enter '1' for ECG1, '2' for ECG2, or 'both' for both ECGs: ").lower()
    if choice in ['1', '2', 'both']:
        break
    print("Invalid input. Please try again.")

# Phase space reconstruction function
def phase_space_reconstruct(signal, dim, tau):
    n = len(signal)
    reconstructed = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        reconstructed[:, i] = signal[i * tau:n - (dim - 1) * tau + i * tau]
    return reconstructed

# Parameters for phase space reconstruction
dimension = 3
time_delay = 20

# Perform phase space reconstruction for selected dataset(s)
if choice == '1' or choice == 'both':
    reconstructed_data1 = phase_space_reconstruct(ecg_data1, dimension, time_delay)
if choice == '2' or choice == 'both':
    reconstructed_data2 = phase_space_reconstruct(ecg_data2, dimension, time_delay)

# Create a figure for plotting
fig = plt.figure(figsize=(20, 10))
plt.rcParams.update({'font.size': 8})
fig.text(0.5, 0.94, "Heartphase Version 0.0.5", ha='center', va='bottom', fontsize=14, color='gray', alpha=0.2)
fig.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=444, color='gray', alpha=0.2, zorder=0)
fig.text(0.5, 0.04, "@Heartphase is Made for Life in Great Britain", ha='center', va='top', fontsize=14, color='gray', alpha=0.2)

# 3D phase space plots
ax2_left = fig.add_subplot(131, projection='3d')
ax2_center = fig.add_subplot(132, projection='3d')
ax2_right = fig.add_subplot(133, projection='3d')

# Set positions and sizes
ax2_left.set_position([0.02, 0.1, 0.23, 0.6])
ax2_center.set_position([0.28, 0.05, 0.44, 0.8])
ax2_right.set_position([0.75, 0.1, 0.23, 0.6])

# Set different view angles
ax2_left.view_init(elev=20, azim=110)
ax2_center.view_init(elev=25, azim=42)
ax2_right.view_init(elev=20, azim=-110)

# Create lines and dots for each 3D plot
lines_dots = []
for ax in [ax2_left, ax2_center, ax2_right]:
    line1, = ax.plot([], [], [], lw=1, color='black')
    line2, = ax.plot([], [], [], lw=1, color='#4169E1')
    dot1, = ax.plot([], [], [], 'o', color='white')
    dot2, = ax.plot([], [], [], 'o', color='white')
    lines_dots.append((line1, line2, dot1, dot2))
    
    # Remove axis lines, ticks, and labels
    ax.set_axis_off()
    
    # Remove panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Set axis limits for 3D plots
for ax in [ax2_left, ax2_center, ax2_right]:
    if choice == 'both':
        ax.set_xlim(min(reconstructed_data1[:, 0].min(), reconstructed_data2[:, 0].min()),
                    max(reconstructed_data1[:, 0].max(), reconstructed_data2[:, 0].max()))
        ax.set_ylim(min(reconstructed_data1[:, 1].min(), reconstructed_data2[:, 1].min()),
                    max(reconstructed_data1[:, 1].max(), reconstructed_data2[:, 1].max()))
        ax.set_zlim(min(reconstructed_data1[:, 2].min(), reconstructed_data2[:, 2].min()),
                    max(reconstructed_data1[:, 2].max(), reconstructed_data2[:, 2].max()))
    elif choice == '1':
        ax.set_xlim(reconstructed_data1[:, 0].min(), reconstructed_data1[:, 0].max())
        ax.set_ylim(reconstructed_data1[:, 1].min(), reconstructed_data1[:, 1].max())
        ax.set_zlim(reconstructed_data1[:, 2].min(), reconstructed_data1[:, 2].max())
    else:
        ax.set_xlim(reconstructed_data2[:, 0].min(), reconstructed_data2[:, 0].max())
        ax.set_ylim(reconstructed_data2[:, 1].min(), reconstructed_data2[:, 1].max())
        ax.set_zlim(reconstructed_data2[:, 2].min(), reconstructed_data2[:, 2].max())

# 1D ECG plot (top left)
ax1 = fig.add_axes([0.05, 0.8, 0.25, 0.15])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ecg_line1, = ax1.plot([], [], color='black', label='ECG 1')
ecg_line2, = ax1.plot([], [], color='#4169E1', label='ECG 2')
ecg_dot1, = ax1.plot([], [], 'o', markersize=5, color='black')
ecg_dot2, = ax1.plot([], [], 'o', markersize=5, color='#4169E1')
ax1.tick_params(axis='both', which='both', length=0)
if choice == 'both':
    ax1.legend(loc='upper right', fontsize='x-small')
elif choice == '1':
    ax1.legend([ecg_line1], ['ECG 1'], loc='upper right', fontsize='x-small')
else:
    ax1.legend([ecg_line2], ['ECG 2'], loc='upper right', fontsize='x-small')

# Text area for amplitude values (top right)
ax3 = fig.add_axes([0.7, 0.8, 0.25, 0.15])
text = ax3.text(0.05, 0.95, '', fontsize=7, verticalalignment='top', transform=ax3.transAxes)
ax3.axis('off')

sampling_rate = 512
fps = 30
window_size = 4 * sampling_rate
time_per_sample = 1 / sampling_rate
time_array = np.arange(0, len(ecg_data1)) * time_per_sample

# Set initial axis limits for 1D plot
ax1.set_xlim(0, 4.5)
if choice == 'both':
    ax1.set_ylim(min(ecg_data1.min(), ecg_data2.min()), max(ecg_data1.max(), ecg_data2.max()))
elif choice == '1':
    ax1.set_ylim(ecg_data1.min(), ecg_data1.max())
else:
    ax1.set_ylim(ecg_data2.min(), ecg_data2.max())


total_ecg_frames = len(ecg_data1) * fps // sampling_rate
rotation_frames = 360
total_frames = total_ecg_frames + rotation_frames

# Update function
def update(frame):
    if frame < total_ecg_frames:
        # ECG plotting phase
        num = frame * (sampling_rate // fps)
        start_idx = max(0, num - window_size)
        end_idx = num

        # Update 1D ECG plot
        plot_time = time_array[start_idx:end_idx]
        if choice == '1' or choice == 'both':
            plot_data1 = ecg_data1[start_idx:end_idx]
            ecg_line1.set_data(plot_time, plot_data1)
            if num > 0:
                ecg_dot1.set_data([time_array[num-1]], [ecg_data1[num-1]])
            else:
                ecg_dot1.set_data([], [])
        if choice == '2' or choice == 'both':
            plot_data2 = ecg_data2[start_idx:end_idx]
            ecg_line2.set_data(plot_time, plot_data2)
            if num > 0:
                ecg_dot2.set_data([time_array[num-1]], [ecg_data2[num-1]])
            else:
                ecg_dot2.set_data([], [])
        
        if num > window_size:
            ax1.set_xlim(time_array[num - window_size], time_array[num - 1] + 0.5)

        # Update 3D phase space plots
        for line1, line2, dot1, dot2 in lines_dots:
            if choice == '1' or choice == 'both':
                line1.set_data(reconstructed_data1[:num, :2].T)
                line1.set_3d_properties(reconstructed_data1[:num, 2])
                if num > 0:
                    dot1.set_data([reconstructed_data1[num-1, 0]], [reconstructed_data1[num-1, 1]])
                    dot1.set_3d_properties([reconstructed_data1[num-1, 2]])
                else:
                    dot1.set_data([], [])
                    dot1.set_3d_properties([])
            if choice == '2' or choice == 'both':
                line2.set_data(reconstructed_data2[:num, :2].T)
                line2.set_3d_properties(reconstructed_data2[:num, 2])
                if num > 0:
                    dot2.set_data([reconstructed_data2[num-1, 0]], [reconstructed_data2[num-1, 1]])
                    dot2.set_3d_properties([reconstructed_data2[num-1, 2]])
                else:
                    dot2.set_data([], [])
                    dot2.set_3d_properties([])
        
        """
        # Update amplitude values text
        values = []
        if choice == '1' or choice == 'both':
            values1 = [f"ECG1 {time_array[j]:.3f}s: {v:.2f}" for j, v in enumerate(ecg_data1[start_idx:end_idx:sampling_rate//fps], start=start_idx)]
            if values1:
                current_value1 = f"ECG 1 {time_array[end_idx-1]:.3f}s: {ecg_data1[end_idx-1]:.2f}"
                values1[-1] = f"-->{current_value1}"
            values.extend(values1[-6:])
        if choice == '2' or choice == 'both':
            values2 = [f"ECG2 {time_array[j]:.3f}s: {v:.2f}" for j, v in enumerate(ecg_data2[start_idx:end_idx:sampling_rate//fps], start=start_idx)]
            if values2:
                current_value2 = f"ECG 2 {time_array[end_idx-1]:.3f}s: {ecg_data2[end_idx-1]:.2f}"
                values2[-1] = f"-->{current_value2}"
            values.extend(values2[-6:])
        text.set_text('\n'.join(values))
        """

    else:
       # Rotation phase
        rotation_frame = frame - total_ecg_frames
        rotation_angle = rotation_frame * (360 / rotation_frames)
        
        # Rotate each 3D plot around a different axis
        ax2_left.view_init(elev=20, azim=110 + rotation_angle)
        ax2_center.view_init(elev=25, azim=42 + rotation_angle)
        ax2_right.view_init(elev=20, azim=-110 - rotation_angle)

    return (ecg_line1, ecg_line2, ecg_dot1, ecg_dot2, text, 
            *[item for sublist in lines_dots for item in sublist])

progress_bar = tqdm(total=total_frames, unit='frames', desc="Rendering Heartphase")

class FFMpegWriterWithProgress(FFMpegWriter):
    def __init__(self, progress_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = progress_callback

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self.progress_callback()

ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True, cache_frame_data=False)

# Adjust these to change video quality
writer = FFMpegWriterWithProgress(
    progress_bar.update,
    fps=fps,
    metadata=dict(artist='Heartphase'),
    bitrate=1200,
    codec='libx264',
    extra_args=[
        '-vf', 'scale=1280:720',
        '-preset', 'slower',
        '-crf', '18'
    ]
)

ani.save('heartphase-animation.mp4', writer=writer)

progress_bar.close()

print("Heartphase rendered!")