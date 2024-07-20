"""
Heartphase 0.0.7
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

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

def calculate_all_durations(p_starts, p_ends, q_points, s_points, t_starts, t_ends, sampling_rate):
    durations = []
    for p_start, p_end, q, s, t_start, t_end in zip(p_starts, p_ends, q_points, s_points, t_starts, t_ends):
        p_duration = (p_end - p_start) / sampling_rate * 1000  # Convert to milliseconds
        qrs_duration = (s - q) / sampling_rate * 1000
        t_duration = (t_end - t_start) / sampling_rate * 1000
        
        durations.append({
            'P-wave (ms)': round(p_duration, 1),
            'QRS (ms)': round(qrs_duration, 1),
            'T-wave (ms)': round(t_duration, 1)
        })
    
    return pd.DataFrame(durations)

def calculate_wave_amplitudes(ecg_trace, p_peaks, r_peaks, q_points, s_points):
    amplitudes = []
    for p, r, q, s in zip(p_peaks, r_peaks, q_points, s_points):
        amplitudes.append({
            'P-wave peak (µV)': round(ecg_trace[p], 3),
            'R-wave peak (µV)': round(ecg_trace[r], 3),
            'Q-wave trough (µV)': round(ecg_trace[q], 3),
            'S-wave trough (µV)': round(ecg_trace[s], 3)
        })
    return pd.DataFrame(amplitudes)

def calculate_amplitude_differences(ecg_trace, p_peaks, r_peaks, q_points, s_points, t_peaks):
    differences = []
    for p, r, q, s, t in zip(p_peaks, r_peaks, q_points, s_points, t_peaks):
        differences.append({
            'PR difference (µV)': round(ecg_trace[r] - ecg_trace[p], 3),
            'QR difference (µV)': round(ecg_trace[r] - ecg_trace[q], 3),
            'SR difference (µV)': round(ecg_trace[r] - ecg_trace[s], 3),
            'TR difference (µV)': round(ecg_trace[r] - ecg_trace[t], 3)
        })
    return pd.DataFrame(differences)

def calculate_tp_segment_median_and_differences(ecg_trace, t_ends, p_starts, p_peaks, r_peaks, q_points, s_points, t_peaks):
    tp_segment_values = []
    for t_end, p_start in zip(t_ends[:-1], p_starts[1:]):
        tp_segment_values.extend(ecg_trace[t_end:p_start])
    
    tp_segment_median = np.median(tp_segment_values)
    
    differences = []
    for p, r, q, s, t in zip(p_peaks, r_peaks, q_points, s_points, t_peaks):
        differences.append({
            'P-peak to TP-median (µV)': round(ecg_trace[p] - tp_segment_median, 3),
            'R-peak to TP-median (µV)': round(ecg_trace[r] - tp_segment_median, 3),
            'Q-trough to TP-median (µV)': round(ecg_trace[q] - tp_segment_median, 3),
            'S-trough to TP-median (µV)': round(ecg_trace[s] - tp_segment_median, 3),
            'T-peak to TP-median (µV)': round(ecg_trace[t] - tp_segment_median, 3)
        })
    
    return tp_segment_median, pd.DataFrame(differences)

def calculate_segments(ecg_data, sampling_rate):
    segments = []
    t_ends = np.where(ecg_data['T-Wave End'] == 1)[0]
    p_starts = np.where(ecg_data['P-Wave Start'] == 1)[0]
    q_points = np.where(ecg_data['Q-Point'] == 1)[0]
    s_points = np.where(ecg_data['S-Point'] == 1)[0]
    p_ends = np.where(ecg_data['P-Wave End'] == 1)[0]
    t_starts = np.where(ecg_data['T-Wave Start'] == 1)[0]

    first_t_end_index = 0
    while first_t_end_index < len(t_ends) and t_ends[first_t_end_index] < p_starts[0]:
        first_t_end_index += 1

    for i in range(first_t_end_index, len(t_ends) - 1):
        tp_segment = (p_starts[i+1] - t_ends[i]) / sampling_rate * 1000
        
        q_point = next((q for q in q_points if q > p_starts[i+1]), None)
        s_point = next((s for s in s_points if s > p_starts[i+1]), None)
        
        if q_point is not None and s_point is not None:
            p_end = next((p for p in p_ends if p > p_starts[i+1] and p < q_point), None)
            pr_segment = (q_point - p_end) / sampling_rate * 1000 if p_end is not None else None
            
            t_start = next((t for t in t_starts if t > s_point), None)
            st_segment = (t_start - s_point) / sampling_rate * 1000 if t_start is not None else None
            
            segments.append({
                'TP segment (ms)': round(tp_segment, 1),
                'PR segment (ms)': round(pr_segment, 1) if pr_segment is not None else None,
                'ST segment (ms)': round(st_segment, 1) if st_segment is not None else None
            })
        
    return pd.DataFrame(segments)

def calculate_ecg_intervals(p_starts, q_points, t_ends, r_peaks, sampling_rate):
    intervals = []
    for i in range(len(q_points)):
        pr_interval = (q_points[i] - p_starts[i]) / sampling_rate * 1000
        qt_interval = (t_ends[i] - q_points[i]) / sampling_rate * 1000
        
        rr_interval = (r_peaks[i+1] - r_peaks[i]) / sampling_rate * 1000 if i < len(r_peaks) - 1 else None
        
        intervals.append({
            'PR interval (ms)': round(pr_interval, 1),
            'QT interval (ms)': round(qt_interval, 1),
            'RR interval (ms)': round(rr_interval, 1) if rr_interval is not None else None
        })
    
    return pd.DataFrame(intervals)

file_path = 'ecg-sample-pqrst.csv'
ecg_trace, q_points, r_peaks, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends = load_labeled_ecg_data(file_path)

num_heartbeats = len(r_peaks)
print(f"Number of detected heartbeats: {num_heartbeats}")

durations_df = calculate_all_durations(p_starts, p_ends, q_points, s_points, t_starts, t_ends, sampling_rate)

print("\nDurations for all heartbeats:")
print(durations_df.to_string(index=True))

duration_summary_stats = durations_df.describe()
print("\nSummary Statistics for Durations:")
print(duration_summary_stats.to_string())

amplitudes_df = calculate_wave_amplitudes(ecg_trace, p_peaks, r_peaks, q_points, s_points)

print("\nAmplitudes for all heartbeats:")
print(amplitudes_df.to_string(index=True))

amplitude_summary_stats = amplitudes_df.describe()
print("\nSummary Statistics for Amplitudes:")
print(amplitude_summary_stats.to_string())

amplitude_differences_df = calculate_amplitude_differences(ecg_trace, p_peaks, r_peaks, q_points, s_points, t_peaks)

print("\nAmplitude Differences for all heartbeats:")
print(amplitude_differences_df.to_string(index=True))

amplitude_differences_summary_stats = amplitude_differences_df.describe()
print("\nSummary Statistics for Amplitude Differences:")
print(amplitude_differences_summary_stats.to_string())

tp_segment_median, tp_differences_df = calculate_tp_segment_median_and_differences(
    ecg_trace, t_ends, p_starts, p_peaks, r_peaks, q_points, s_points, t_peaks
)

print(f"\nMedian TP-segment amplitude: {tp_segment_median:.3f} µV")

print("\nDifferences between wave peaks/troughs and TP-segment median:")
print(tp_differences_df.to_string(index=True))

tp_differences_summary_stats = tp_differences_df.describe()
print("\nSummary Statistics for TP-segment Differences:")
print(tp_differences_summary_stats.to_string())

ecg_data = pd.DataFrame({
    'ECG Signal': ecg_trace,
    'Q-Point': np.zeros(len(ecg_trace)),
    'R-Peak': np.zeros(len(ecg_trace)),
    'S-Point': np.zeros(len(ecg_trace)),
    'P-Wave Start': np.zeros(len(ecg_trace)),
    'P-Wave Peak': np.zeros(len(ecg_trace)),
    'P-Wave End': np.zeros(len(ecg_trace)),
    'T-Wave Start': np.zeros(len(ecg_trace)),
    'T-Wave Peak': np.zeros(len(ecg_trace)),
    'T-Wave End': np.zeros(len(ecg_trace))
})

ecg_data.loc[q_points, 'Q-Point'] = 1
ecg_data.loc[r_peaks, 'R-Peak'] = 1
ecg_data.loc[s_points, 'S-Point'] = 1
ecg_data.loc[p_starts, 'P-Wave Start'] = 1
ecg_data.loc[p_peaks, 'P-Wave Peak'] = 1
ecg_data.loc[p_ends, 'P-Wave End'] = 1
ecg_data.loc[t_starts, 'T-Wave Start'] = 1
ecg_data.loc[t_peaks, 'T-Wave Peak'] = 1
ecg_data.loc[t_ends, 'T-Wave End'] = 1

segments_df = calculate_segments(ecg_data, sampling_rate)

print("\nSegments for all heartbeats:")
print(segments_df.to_string(index=True))

segments_summary_stats = segments_df.describe()
print("\nSummary Statistics for Segments:")
print(segments_summary_stats.to_string())

intervals_df = calculate_ecg_intervals(p_starts, q_points, t_ends, r_peaks, sampling_rate)

print("\nECG Intervals for all heartbeats:")
print(intervals_df.to_string(index=True))

intervals_summary_stats = intervals_df.describe()
print("\nSummary Statistics for ECG Intervals:")
print(intervals_summary_stats.to_string())

"""
plt.figure(figsize=(15, 5))
plt.plot(ecg_trace)
plt.title('ECG')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
"""