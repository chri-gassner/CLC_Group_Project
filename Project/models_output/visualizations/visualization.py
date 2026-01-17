import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASETS = {
    'OpenPose': 'Project/openpose_pose_extraction/output/openpose_fitness_dataset.csv',
    'MediaPipe': 'Project/mediapipe_pose_extraction/output/mediapipe_pose_dataset.csv'
}

OUTPUT_DIR = 'Project/models_output/visualizations/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# DEFINITIONS: Colors & Connections
# ---------------------------------------------------------

# Define skeleton connections (lines)
SKELETON_CONNECTIONS = [
    ('nose', 'l_shoulder'), ('nose', 'r_shoulder'),
    ('l_shoulder', 'r_shoulder'),
    ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
    ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
    ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'),
    ('l_hip', 'r_hip'),
    ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
    ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
]

# Color schema and legend
# Format: 'Legend Label': ([List of Joints], 'Color', 'Marker Symbol')
JOINT_GROUPS = {
    'Nose (Head)': (['nose'], 'gold', 'o'),
    'Left Arm': (['l_shoulder', 'l_elbow', 'l_wrist'], 'cyan', 'o'),
    'Right Arm': (['r_shoulder', 'r_elbow', 'r_wrist'], 'magenta', 'o'),
    'Left Leg': (['l_hip', 'l_knee', 'l_ankle'], 'blue', 's'), # Square
    'Right Leg': (['r_hip', 'r_knee', 'r_ankle'], 'red', 's'),  # Square
}

# ---------------------------------------------------------
# FUNCTION: Create Plot
# ---------------------------------------------------------
def create_average_pose_plot(csv_path, dataset_name):
    print(f"Creating visualization for: {dataset_name}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    unique_labels = sorted(df['label'].unique())
    
    # Setup Plot Grid (e.g., 2 rows, 3 cols)
    cols = 3
    rows = (len(unique_labels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # For legend handles (create legend only once)
    legend_handles = []
    added_labels = set()

    for i, label in enumerate(unique_labels):
        ax = axes[i]
        subset = df[df['label'] == label]
        
        # Calculate average
        avg_pose = subset.filter(regex='_x$|_y$').mean()

        # 1. Draw lines (Skeleton)
        for joint1, joint2 in SKELETON_CONNECTIONS:
            try:
                x1, y1 = avg_pose[f"{joint1}_x"], avg_pose[f"{joint1}_y"]
                x2, y2 = avg_pose[f"{joint2}_x"], avg_pose[f"{joint2}_y"]
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=2, alpha=0.5, zorder=1)
            except KeyError:
                continue

        # 2. Draw points (using colored groups)
        for group_label, (joints, color, marker) in JOINT_GROUPS.items():
            for joint in joints:
                try:
                    x, y = avg_pose[f"{joint}_x"], avg_pose[f"{joint}_y"]
                    ax.scatter(x, y, c=color, marker=marker, s=80, zorder=2, edgecolors='black')
                    
                    # Save for legend (once per group)
                    if group_label not in added_labels:
                        legend_handles.append(mlines.Line2D([], [], color=color, marker=marker, 
                                                            linestyle='None', markersize=10, label=group_label))
                        added_labels.add(group_label)
                except KeyError:
                    continue

        # Plot Styling
        ax.set_title(label.upper(), fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # IMPORTANT: Image coordinates (0,0 is top-left)
        ax.axis('off')     # Hide axes

    # Hide empty subplots (if exercise count is odd)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add global legend
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(JOINT_GROUPS), frameon=False)

    plt.suptitle(f"Average Poses ({dataset_name})", fontsize=20, y=1.1)
    plt.tight_layout()
    
    # Save
    outfile = os.path.join(OUTPUT_DIR, f'viz_{dataset_name}.png')
    plt.savefig(outfile, bbox_inches='tight')
    print(f" -> Saved to: {outfile}")
    plt.close()

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    for name, path in DATASETS.items():
        create_average_pose_plot(path, name)
    print("\nDone! Images are in:", OUTPUT_DIR)