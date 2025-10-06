import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.cluster import KMeans


# Load the CSV file
def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)  # Read CSV without column names
    return df

# Process data by removing integer values
def process_data(df):
    df = df[~df[1].astype(float).apply(lambda x: x.is_integer())]  # Remove rows where value is an integer as those are edge cases that are not actual cells
    return df

def convert_to_clock_frame(angles_deg):
    # Convert angle to range (-180, 180] with 0° = top
    converted = (90 - angles_deg) % 360
    converted[converted > 180] -= 360
    return converted


# Plot the data as a rose plot and save it
# Plot combined x-position histogram and rose plot
def plot_rose(df, output_dir, file_name, bins=36):
    x_positions = df[0].astype(float)
    angles_deg = df[1].to_numpy()
    angles_clock_deg = convert_to_clock_frame(angles_deg)

    # Histogram of x-positions (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    hist_x, bin_edges_x = np.histogram(x_positions, bins=20)
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2

    # KMeans clustering where K = 2 to separate high vs low count bins (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    hist_x_reshaped = hist_x.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(hist_x_reshaped)

    # Identify which cluster label corresponds to high counts (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    high_label = np.argmax(kmeans.cluster_centers_)

    # Keep bins with high counts only (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    keep_bins = [i for i, label in enumerate(labels) if label == high_label]
    keep_mask = np.zeros(len(df), dtype=bool)

    # Assign each data point to a bin, keep if in a high-count bin (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    for i in keep_bins:
        bin_min = bin_edges_x[i]
        bin_max = bin_edges_x[i + 1]
        in_bin = (x_positions >= bin_min) & (x_positions < bin_max)
        keep_mask |= in_bin

    # Filter data (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    df_filtered = df[keep_mask]
    x_positions_filtered = df_filtered[0].astype(float)
    angles_filtered = df_filtered[1].to_numpy()
    angles_clock_filtered = convert_to_clock_frame(angles_filtered)

    angles_axial_deg = np.abs(angles_clock_filtered) % 180
    angles_axial_rad = np.deg2rad(angles_axial_deg)

    symmetric_angles_rad = np.concatenate([
        angles_axial_rad,
        np.mod(angles_axial_rad + np.pi, 2 * np.pi)
    ])

    hist, bin_edges = np.histogram(symmetric_angles_rad, bins=bins, range=(0, 2 * np.pi))
    theta = bin_edges[:-1] + np.diff(bin_edges) / 2
    width = np.diff(bin_edges)

    doubled = 2 * angles_axial_rad
    mean_doubled_angle = np.arctan2(np.sum(np.sin(doubled)), np.sum(np.cos(doubled)))
    best_fit_angle_rad = (mean_doubled_angle / 2) % np.pi
    best_fit_angle_deg = np.rad2deg(best_fit_angle_rad)

    fig = plt.figure(figsize=(12, 6))

    # Histogram subplot of the x-position (this part of the code is specific to images of vertical running stripes, and would not apply to other patterns)
    ax1 = fig.add_subplot(1, 2, 1)
    shown_labels = set()
    for i in range(len(hist_x)):
        bin_min = bin_edges_x[i]
        bin_max = bin_edges_x[i + 1]
        height = hist_x[i]
        color = 'skyblue' if i in keep_bins else 'lightgray'
        label = None
        if i not in keep_bins and "Filtered out" not in shown_labels:
            label = "Filtered out"
            shown_labels.add(label)
        elif i in keep_bins and "Kept" not in shown_labels:
            label = "Kept"
            shown_labels.add(label)
        ax1.bar((bin_min + bin_max) / 2, height, width=bin_max - bin_min,
                color=color, edgecolor='black', align='center',
                label=label)
    ax1.legend()
    ax1.set_title(f"X-Position Histogram: {file_name}")
    ax1.set_xlabel("X-Position")
    ax1.set_ylabel("Count")

    # Rose plot subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.bar(theta, hist, width=width, color='b', alpha=0.6, edgecolor='black')
    ax2.set_title(f"Rose Plot: {file_name}")

    n_points = len(angles_filtered)
    ax2.text(-0.2, 1.05, f"Cell Count = {n_points}", transform=ax2.transAxes,
             fontsize=10, ha='left', va='center')
    ax2.text(-0.2, 1.00, f"Average Chirality = {best_fit_angle_deg:.2f}°", transform=ax2.transAxes,
             fontsize=10, ha='left', va='center')

    ax2.plot([best_fit_angle_rad, best_fit_angle_rad], [0, max(hist)], linestyle='--', color='red', linewidth=2)
    ax2.plot([best_fit_angle_rad + np.pi, best_fit_angle_rad + np.pi], [0, max(hist)], linestyle='--', color='red', linewidth=2)

    output_path = os.path.join(output_dir, f"{file_name}_combined_plot.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return angles_axial_deg.tolist(), angles_clock_filtered.tolist()


def process_csv_files(csv_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_angles = {}
    
    for csv_path in csv_paths:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {csv_path}...")
        df = load_csv(csv_path)
        df = process_data(df)
        
        angles_axial_deg, angles_clock_filtered_deg = plot_rose(df, output_dir, file_name)
        
        # Save filtered angles (clock frame, degrees) to CSV
        filtered_angles_df = pd.DataFrame({'Filtered_Angles_Clock_Deg': angles_clock_filtered_deg})
        filtered_csv_path = os.path.join(output_dir, f"{file_name}_filtered_angles.csv")
        filtered_angles_df.to_csv(filtered_csv_path, index=False)
        
        all_angles[file_name] = angles_axial_deg  # For axial angle summary CSV
    
    # Save axial angles summary across files
    angles_df = pd.DataFrame.from_dict(all_angles, orient='index').transpose()
    angles_csv_path = os.path.join(output_dir, "processed_axial_angles.csv")
    angles_df.to_csv(angles_csv_path, index=False)
    
    # Zip everything
    zip_path = os.path.join(output_dir, "rose_plots.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, file)
    
    print("All CSV files processed and plots saved!")
    return zip_path


# Example usage
# output_directory = os.path.expanduser("C:/Users/your_username/Downloads/Rose_Plots")
# csv_files = ["data1.csv", "data2.csv", "data3.csv"]
# zip_file_path = process_csv_files(csv_files, output_directory)
# print(f"Analysis complete. Results saved and zipped at: {zip_file_path}")
