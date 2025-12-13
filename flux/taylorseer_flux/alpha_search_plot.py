import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_alpha_curve(csv_path, output_path):
    # 1. Read the CSV file
    try:
        # Using skipinitialspace=True to handle potential spaces after commas
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Separate Baseline and Numeric Data
    # Identify the baseline row
    baseline_row = df[df['ts_exp_smooth_alpha'].astype(str).str.contains('baseline', case=False)]
    
    if not baseline_row.empty:
        baseline_value = float(baseline_row['ImageReward'].values[0])
    else:
        baseline_value = None
        print("Warning: No 'baseline' row found.")

    # Filter out the baseline row to get the main numeric data
    data_df = df[df['ts_exp_smooth_alpha'].astype(str) != 'baseline'].copy()
    
    # Convert columns to numeric
    data_df['ts_exp_smooth_alpha'] = pd.to_numeric(data_df['ts_exp_smooth_alpha'])
    data_df['ImageReward'] = pd.to_numeric(data_df['ImageReward'])
    
    # Sort by alpha just in case
    data_df = data_df.sort_values('ts_exp_smooth_alpha')

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Color Scheme: Blue
    main_color = '#1E90FF'  # DodgerBlue
    baseline_color = '#000080'  # Navy Blue
    
    # Plot the main curve
    plt.plot(data_df['ts_exp_smooth_alpha'], data_df['ImageReward'], 
             marker='o', linestyle='-', linewidth=2, color=main_color, 
             label='Exponential Smoothing')

    # Plot the baseline
    if baseline_value is not None:
        plt.axhline(y=baseline_value, color=baseline_color, linestyle='--', linewidth=2, 
                    label=f'Baseline (Naive TS): {baseline_value:.4f}')

    # Find and annotate the maximum point
    max_point = data_df.loc[data_df['ImageReward'].idxmax()]
    plt.plot(max_point['ts_exp_smooth_alpha'], max_point['ImageReward'], 
             marker='*', markersize=15, color='gold', markeredgecolor='orange',
             label=f'Best Alpha: {max_point["ts_exp_smooth_alpha"]} ({max_point["ImageReward"]:.4f})')

    # 4. Formatting
    plt.title('Impact of Smoothing Alpha on ImageReward', fontsize=14, fontweight='bold')
    plt.xlabel('Smoothing Alpha', fontsize=12)
    plt.ylabel('ImageReward Score', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # 5. Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    base_dir = "/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth"
    csv_file = os.path.join(base_dir, "alpha_search.csv")
    output_file = os.path.join(base_dir, "alpha_curve.png")
    
    if os.path.exists(csv_file):
        plot_alpha_curve(csv_file, output_file)
    else:
        print(f"File not found: {csv_file}")