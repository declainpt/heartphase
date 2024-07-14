"""
Heartphase 0.0.6
Copyright (c) 2024 Declain P. Thomas
Distributed under the MIT software license.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load ECG data from CSV
file_path = 'ecg-sample.csv'
ecg_data = pd.read_csv(file_path)
ecg_trace = ecg_data.iloc[:, 0].values

# Parameters
sampling_rate = 512  # Hz
window_size = 3 * sampling_rate  # 3-second windows

# Lists to store the manually labeled points
q_points, r_peaks, s_points = [], [], []
p_starts, p_peaks, p_ends = [], [], []
t_starts, t_peaks, t_ends = [], [], []

# Dictionary to store the plot objects for each marker
markers = {
    'q': [], 'r': [], 's': [],
    'p_start': [], 'p_peak': [], 'p_end': [],
    't_start': [], 't_peak': [], 't_end': []
}

# Variable to store the current labeling mode
current_mode = 'r'  # Default to R-peak labeling

# Global dictionary to store running totals
global_running_totals = {
    'q': 0, 'r': 0, 's': 0,
    'p_start': 0, 'p_peak': 0, 'p_end': 0,
    't_start': 0, 't_peak': 0, 't_end': 0
}

def update_running_totals():
    global global_running_totals
    global_running_totals = {
        'q': len(q_points),
        'r': len(r_peaks),
        's': len(s_points),
        'p_start': len(p_starts),
        'p_peak': len(p_peaks),
        'p_end': len(p_ends),
        't_start': len(t_starts),
        't_peak': len(t_peaks),
        't_end': len(t_ends)
    }

def onclick(event, window_start, window_end):
    global global_running_totals
    if event.inaxes == ax:
        x_local = int(event.xdata)
        x_global = x_local + window_start
        y = ecg_trace[x_global]
        if current_mode in ['q', 'r', 's']:
            handle_qrs_click(x_global, x_local, y)
        elif current_mode.startswith('p_'):
            handle_p_wave_click(x_global, x_local, y)
        elif current_mode.startswith('t_'):
            handle_t_wave_click(x_global, x_local, y)
        global_running_totals[current_mode] += 1
        display_running_totals(fig, ax)
        plt.draw()

def handle_qrs_click(x_global, x_local, y):
    global q_points, r_peaks, s_points
    if current_mode == 'r' and x_global not in r_peaks:
        r_peaks.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['r']['marker'], color=point_styles['r']['color'], markersize=8)
        markers['r'].append(marker)
        print(f"R-peak at index: {x_global}")
    elif current_mode == 'q' and x_global not in q_points:
        q_points.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['q']['marker'], color=point_styles['q']['color'], markersize=8)
        markers['q'].append(marker)
        print(f"Q-point at index: {x_global}")
    elif current_mode == 's' and x_global not in s_points:
        s_points.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['s']['marker'], color=point_styles['s']['color'], markersize=8)
        markers['s'].append(marker)
        print(f"S-point at index: {x_global}")

def handle_p_wave_click(x_global, x_local, y):
    global p_starts, p_peaks, p_ends
    if current_mode == 'p_start' and x_global not in p_starts:
        p_starts.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['p_start']['marker'], color=point_styles['p_start']['color'], markersize=8)
        markers['p_start'].append(marker)
        print(f"P-wave start at index: {x_global}")
    elif current_mode == 'p_peak' and x_global not in p_peaks:
        p_peaks.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['p_peak']['marker'], color=point_styles['p_peak']['color'], markersize=8)
        markers['p_peak'].append(marker)
        print(f"P-wave peak at index: {x_global}")
    elif current_mode == 'p_end' and x_global not in p_ends:
        p_ends.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['p_end']['marker'], color=point_styles['p_end']['color'], markersize=8)
        markers['p_end'].append(marker)
        print(f"P-wave end at index: {x_global}")

def handle_t_wave_click(x_global, x_local, y):
    global t_starts, t_peaks, t_ends
    if current_mode == 't_start' and x_global not in t_starts:
        t_starts.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['t_start']['marker'], color=point_styles['t_start']['color'], markersize=8)
        markers['t_start'].append(marker)
        print(f"T-wave start at index: {x_global}")
    elif current_mode == 't_peak' and x_global not in t_peaks:
        t_peaks.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['t_peak']['marker'], color=point_styles['t_peak']['color'], markersize=8)
        markers['t_peak'].append(marker)
        print(f"T-wave peak at index: {x_global}")
    elif current_mode == 't_end' and x_global not in t_ends:
        t_ends.append(x_global)
        marker, = ax.plot(x_local, y, marker=point_styles['t_end']['marker'], color=point_styles['t_end']['color'], markersize=8)
        markers['t_end'].append(marker)
        print(f"T-wave end at index: {x_global}")

def onmove(event, window_start, window_end):
    if event.inaxes == ax:
        x_local = int(event.xdata)
        x_global = x_local + window_start
        if 0 <= x_local < (window_end - window_start):
            y = ecg_trace[x_global]
            
            y_before = ecg_trace[x_global-1] if x_global > 0 else None
            y_after = ecg_trace[x_global+1] if x_global < len(ecg_trace) - 1 else None
            
            window_width = window_end - window_start
            on_right_side = x_local > window_width / 2

            if on_right_side:
                annot.set_position((-15, 0))
                annot.xyann = (-15, 0)
                annot.set_ha('right')
            else:
                annot.set_position((15, 0))
                annot.xyann = (15, 0)
                annot.set_ha('left')

            annot.xy = (x_local, y)
            annotation_text = f"Index: {x_global}\n"
            annotation_text += f"Before: {y_before:.2f}\n" if y_before is not None else "Before: N/A\n"
            annotation_text += f"Current: {y:.2f}\n"
            annotation_text += f"After: {y_after:.2f}" if y_after is not None else "After: N/A"
            
            annot.set_text(annotation_text)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            annot.set_visible(False)

def undo(event):
    global q_points, r_peaks, s_points, p_starts, p_peaks, p_ends, t_starts, t_peaks, t_ends, global_running_totals
    if current_mode in markers and markers[current_mode]:
        try:
            if current_mode == 'r':
                r_peaks.pop()
            elif current_mode == 'q':
                q_points.pop()
            elif current_mode == 's':
                s_points.pop()
            elif current_mode == 'p_start':
                p_starts.pop()
            elif current_mode == 'p_peak':
                p_peaks.pop()
            elif current_mode == 'p_end':
                p_ends.pop()
            elif current_mode == 't_start':
                t_starts.pop()
            elif current_mode == 't_peak':
                t_peaks.pop()
            elif current_mode == 't_end':
                t_ends.pop()
            
            global_running_totals[current_mode] -= 1
            
            marker = markers[current_mode].pop()
            if marker in ax.lines:
                marker.remove()
            print(f"Removed last {current_mode} point")
        except IndexError:
            print(f"No {current_mode} points to remove")
        except ValueError as e:
            print(f"Error removing marker: {e}")
        
        display_running_totals(fig, ax)
        plt.draw()

def set_mode(mode):
    def mode_setter(event):
        global current_mode
        current_mode = mode
        print(f"Switched to {mode} labeling mode.")
        update_button_appearances() 
    return mode_setter

def update_button_appearances():
    for btn, mode in zip(buttons, button_modes):
        if mode == current_mode:
            btn.color = point_styles[mode]['color']
            btn.label.set_color('white')
        else:
            btn.color = 'white'
            btn.label.set_color(point_styles[mode]['color'] if mode != 'undo' else 'gray')
        btn.ax.set_facecolor(btn.color)
    plt.draw()

def add_watermark(fig):
    fig.text(0.5, 0.5, "♡Φ", ha='center', va='center', fontsize=444, 
             color='gray', alpha=0.2, zorder=0)

point_styles = {
    'q': {'color': '#4169E1', 'marker': 'o'},
    'r': {'color': '#ff4500', 'marker': 'o'},
    's': {'color': '#00A86B', 'marker': 'o'},
    'p_start': {'color': '#9A5FAB', 'marker': '^'},
    'p_peak': {'color': '#9A5FAB', 'marker': 'o'},
    'p_end': {'color': '#9A5FAB', 'marker': 'v'},
    't_start': {'color': '#ffdc73', 'marker': '^'},
    't_peak': {'color': '#ffdc73', 'marker': 'o'},
    't_end': {'color': '#ffdc73', 'marker': 'v'}
}

def display_running_totals(fig, ax):
    for txt in fig.texts:
        if hasattr(txt, 'is_running_total') and txt.is_running_total:
            txt.remove()
    
    totals_text = "Running Totals:"
    
    title_text = fig.text(0.98, 0.98, totals_text, 
             horizontalalignment='right',
             verticalalignment='top',
             transform=fig.transFigure,
             fontsize=10,
             fontweight='bold',
             color='black')
    title_text.is_running_total = True
    
    label_groups = [
        ['q', 'r', 's'],
        ['p_start', 'p_peak', 'p_end'],
        ['t_start', 't_peak', 't_end']
    ]
    
    y_offset = 0.02
    for group in label_groups:
        x_offset = 0
        for label in group:
            color = point_styles[label]['color']
            text = f"{label.replace('_', ' ').title()}: {global_running_totals[label]}"
            txt = fig.text(0.98 - x_offset, 0.98 - y_offset, text, 
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=fig.transFigure,
                     fontsize=10,
                     color=color)
            txt.is_running_total = True
            x_offset += 0.01 * len(text)
        y_offset += 0.02
    
    fig.canvas.draw_idle()

# Iterate over the ECG signal in windows
for start in range(0, len(ecg_trace), window_size):
    end = min(start + window_size, len(ecg_trace))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)  # Adjust margins

    ax.plot(range(end - start), ecg_trace[start:end], color='black')
    ax.set_title(f'Mark the PQRST points (Window {start}-{end})')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')

    ax.set_xlim(0, end - start)

    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    fig.canvas.mpl_connect('button_press_event', lambda event, start=start, end=end: onclick(event, start, end))
    fig.canvas.mpl_connect('motion_notify_event', lambda event, start=start, end=end: onmove(event, start, end))
    
    button_labels = ['Undo', 'R', 'Q', 'S', 'P Start', 'P Peak', 'P End', 'T Start', 'T Peak', 'T End']
    button_modes = ['undo', 'r', 'q', 's', 'p_start', 'p_peak', 'p_end', 't_start', 't_peak', 't_end']
    
    button_width = 0.08
    button_height = 0.05
    button_spacing = (1 - (len(button_labels) * button_width)) / (len(button_labels) + 1)
    
    buttons = []
    
    for i, (label, mode) in enumerate(zip(button_labels, button_modes)):
        button_pos = [button_spacing + i * (button_width + button_spacing), 0.05, button_width, button_height]
        btn_ax = fig.add_axes(button_pos)
        
        if mode == 'undo':
            color = 'gray'
        else:
            color = point_styles[mode]['color']
        
        btn = Button(btn_ax, label, color='white', hovercolor=color)
        btn.label.set_color(color)
        btn.ax.set_facecolor('none')
        for spine in btn.ax.spines.values():
            spine.set_color(color)
        
        def hover(event, btn=btn, color=color, mode=mode):
            if event.inaxes == btn.ax:
                if mode != current_mode:
                    btn.label.set_color('white')
                    btn.ax.set_facecolor(color)
            else:
                if mode != current_mode:
                    btn.label.set_color(color)
                    btn.ax.set_facecolor('none')
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', hover)
        
        if mode == 'undo':
            btn.on_clicked(undo)
        else:
            btn.on_clicked(set_mode(mode))
        buttons.append(btn)

    update_running_totals()
    display_running_totals(fig, ax)

    def update_display(event):
        update_running_totals()
        display_running_totals(fig, ax)
        fig.canvas.draw_idle()

    for btn in buttons:
        btn.on_clicked(update_display)

    update_button_appearances()

    plt.show()

results = pd.DataFrame({'ECG Signal': ecg_trace})

for col in ['Q-Point', 'R-Peak', 'S-Point', 
            'P-Wave Start', 'P-Wave Peak', 'P-Wave End',
            'T-Wave Start', 'T-Wave Peak', 'T-Wave End']:
    results[col] = 0

results.loc[q_points, 'Q-Point'] = 1
results.loc[r_peaks, 'R-Peak'] = 1
results.loc[s_points, 'S-Point'] = 1
results.loc[p_starts, 'P-Wave Start'] = 1
results.loc[p_peaks, 'P-Wave Peak'] = 1
results.loc[p_ends, 'P-Wave End'] = 1
results.loc[t_starts, 'T-Wave Start'] = 1
results.loc[t_peaks, 'T-Wave Peak'] = 1
results.loc[t_ends, 'T-Wave End'] = 1

# Save the results to a new CSV file
results.to_csv('ecg-labelled-pqrst.csv', index=False)

print("All points have been labeled manually and saved to 'ecg-labelled-pqrst.csv'.")

def plot_final_results(ecg_data, q_points, r_peaks, s_points, 
                       p_starts, p_peaks, p_ends, 
                       t_starts, t_peaks, t_ends):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.plot(ecg_data, color='black', label='ECG Signal', zorder=1)
    
    point_data = {
        'Q Points': {'points': q_points, 'color': '#4169E1', 'marker': 'o'},
        'R Peaks': {'points': r_peaks, 'color': '#ff4500', 'marker': 'o'},
        'S Points': {'points': s_points, 'color': '#00A86B', 'marker': 'o'},
        'P-Wave Start': {'points': p_starts, 'color': '#9A5FAB', 'marker': '^'},
        'P-Wave Peak': {'points': p_peaks, 'color': '#9A5FAB', 'marker': 'o'},
        'P-Wave End': {'points': p_ends, 'color': '#9A5FAB', 'marker': 'v'},
        'T-Wave Start': {'points': t_starts, 'color': '#ffdc73', 'marker': '^'},
        'T-Wave Peak': {'points': t_peaks, 'color': '#ffdc73', 'marker': 'o'},
        'T-Wave End': {'points': t_ends, 'color': '#ffdc73', 'marker': 'v'}
    }
    
    for label, data in point_data.items():
        points = data['points']
        count = len(points)
        ax.scatter(points, ecg_data[points], color=data['color'], marker=data['marker'], 
                   label=f'{label} ({count})', zorder=2)
    
    ax.set_title('Heartphase ♡Φ Selection')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    add_watermark(fig)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    plt.savefig('ecg-labelled-pqrst.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_final_results(ecg_trace, q_points, r_peaks, s_points, 
                   p_starts, p_peaks, p_ends, 
                   t_starts, t_peaks, t_ends)

print("Final plot with labeled points has been saved as 'ecg-labelled-pqrst.png'.")
    