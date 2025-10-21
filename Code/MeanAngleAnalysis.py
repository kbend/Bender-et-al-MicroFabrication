import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile

# Load the CSV file
def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df

# Process data by removing integer values
def process_data(df):
    df = df[~df[1].astype(float).apply(lambda x: x.is_integer())]
    return df

# Convert angles to new frame where:
# 0° = top, + clockwise, - counterclockwise
def convert_to_clock_frame(angles_deg):
    # Convert angle to range (-180, 180] with 0° = top
    converted = (90 - angles_deg) % 360
    converted[converted > 180] -= 360
    return converted

def plot_rose(df, output_dir, file_name, bins=36):
    angles_deg = df[1].to_numpy()
    angles_clock_deg = convert_to_clock_frame(angles_deg)
    
    # For axial symmetry, wrap angles to [0, 180)
    angles_axial_deg = np.abs(angles_clock_deg) % 180
    angles_axial_rad = np.deg2rad(angles_axial_deg)

    # Duplicate for symmetry
    symmetric_angles_rad = np.concatenate([
        angles_axial_rad,
        np.mod(angles_axial_rad + np.pi, 2 * np.pi)
    ])

    # Histogram
    hist, bin_edges = np.histogram(symmetric_angles_rad, bins=bins, range=(0, 2 * np.pi))
    theta = bin_edges[:-1] + np.diff(bin_edges) / 2
    width = np.diff(bin_edges)

    # Mean axis
    doubled = 2 * angles_axial_rad
    mean_doubled_angle = np.arctan2(np.sum(np.sin(doubled)), np.sum(np.cos(doubled)))
    best_fit_angle_rad = (mean_doubled_angle / 2) % np.pi
    best_fit_angle_deg = np.rad2deg(best_fit_angle_rad)

    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # 0° at top
    ax.set_theta_direction(-1)       # Clockwise angles

    ax.bar(theta, hist, width=width, color='b', alpha=0.6, edgecolor='black')
    ax.set_title(f"{file_name}")

    n_points = len(angles_deg)
    ax.text(-0.2, 1.05, f"Cell Count = {n_points}", transform=ax.transAxes,
            fontsize=10, ha='left', va='center')
    ax.text(-0.2, 1.00, f"Average Chirality = {best_fit_angle_deg:.2f}°", transform=ax.transAxes,
            fontsize=10, ha='left', va='center')

    # Red dotted symmetry line
    ax.plot([best_fit_angle_rad, best_fit_angle_rad], [0, max(hist)], linestyle='--', color='red', linewidth=2)
    ax.plot([best_fit_angle_rad + np.pi, best_fit_angle_rad + np.pi], [0, max(hist)], linestyle='--', color='red', linewidth=2)

    # Save
    output_path = os.path.join(output_dir, f"{file_name}_rose_plot.png")
    plt.savefig(output_path)
    plt.close()

    return angles_clock_deg.tolist()

def process_csv_files(csv_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_angles = {}
    
    for csv_path in csv_paths:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {csv_path}...")
        df = load_csv(csv_path)
        df = process_data(df)
        all_angles[file_name] = plot_rose(df, output_dir, file_name)
    
    angles_df = pd.DataFrame.from_dict(all_angles, orient='index').transpose()
    angles_csv_path = os.path.join(output_dir, "processed_angles.csv")
    angles_df.to_csv(angles_csv_path, index=False)
    
    zip_path = os.path.join(output_dir, "rose_plots.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, file)
    
    print("All CSV files processed and plots saved!")
    return zip_path
