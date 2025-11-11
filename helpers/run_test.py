"""
Generate metrics visualization showing how different configurations affect clustering quality μ′.

Creates a facet grid with:
- Rows: num_colors
- Columns: fill_percentage
- Each subplot: Time vs Quality with different lines for different NUM_ANTS values
- Averages over 5 runs for each configuration
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import sys

# Add parent directory to path so we can import from objects and config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import tqdm for progress bar, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm class if not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args):
            pass

# Suppress matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

from objects.simulation import Simulation
import config


def run_simulation(height, width, num_colors, fill_percentage, num_ants, num_steps, track_interval=100):
    """
    Run a single simulation and return quality history.
    
    Args:
        height (int): Grid height
        width (int): Grid width
        num_colors (int): Number of colors
        fill_percentage (float): Fill percentage
        num_ants (int): Number of ants
        num_steps (int): Number of steps
        track_interval (int): How often to track quality
        
    Returns:
        list: Quality history at tracked intervals
        list: Step numbers corresponding to quality history
    """
    sim = Simulation(height, width, num_colors, fill_percentage, num_ants, num_steps)
    sim.run(verbose=False, track_interval=track_interval)
    
    # Get quality history
    # Simulation tracks: initial (step 0), then after steps where (step+1) % track_interval == 0
    # So if track_interval=100 and num_steps=5000, we track at steps: 0, 100, 200, ..., 5000
    quality_history = sim.quality_history
    
    # Create step history corresponding to quality history
    # First entry is step 0 (initial), then track_interval, 2*track_interval, ..., num_steps
    num_points = len(quality_history)
    step_history = [0] + [track_interval * i for i in range(1, num_points)]
    # Ensure the last point is exactly num_steps
    if step_history[-1] != num_steps:
        step_history[-1] = num_steps
    
    return quality_history, step_history


def average_runs(height, width, num_colors, fill_percentage, num_ants, num_steps, 
                 num_runs=5, track_interval=100):
    """
    Run multiple simulations and average the results.
    
    Args:
        height (int): Grid height
        width (int): Grid width
        num_colors (int): Number of colors
        fill_percentage (float): Fill percentage
        num_ants (int): Number of ants
        num_steps (int): Number of steps
        num_runs (int): Number of runs to average
        track_interval (int): How often to track quality
        
    Returns:
        np.array: Average quality history
        list: Step numbers
    """
    all_quality_histories = []
    step_history = None
    
    for run in range(num_runs):
        quality_history, steps = run_simulation(
            height, width, num_colors, fill_percentage, num_ants, num_steps, track_interval
        )
        all_quality_histories.append(quality_history)
        if step_history is None:
            step_history = steps
        else:
            # Verify step_history is consistent across runs
            if len(quality_history) != len(step_history):
                raise ValueError(f"Inconsistent quality history length: {len(quality_history)} vs {len(step_history)}")
    
    # Average across runs
    avg_quality = np.mean(all_quality_histories, axis=0)
    
    return avg_quality, step_history


def generate_metrics():
    """
    Generate facet grid visualization of metrics.
    """
    # Configuration parameters
    HEIGHT = config.HEIGHT
    WIDTH = config.WIDTH
    NUM_STEPS = 5000  # Max steps as specified
    NUM_RUNS = 5  # Average over 5 runs
    TRACK_INTERVAL = 100  # Track quality every 100 steps
    
    # Parameter ranges to test
    FILL_PERCENTAGES = [20, 30, 40, 50]  # x-axis (columns)
    NUM_COLORS_LIST = [2, 3, 4]  # y-axis (rows)
    NUM_ANTS_LIST = [25, 50, 100]  # Different lines in each subplot
    
    OUTPUT_DIR = config.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Generating Metrics Visualization")
    print("=" * 60)
    print(f"Grid size: {HEIGHT} × {WIDTH}")
    print(f"Steps: {NUM_STEPS}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Fill percentages: {FILL_PERCENTAGES}")
    print(f"Number of colors: {NUM_COLORS_LIST}")
    print(f"Number of ants: {NUM_ANTS_LIST}")
    print("=" * 60)
    
    # Store all results
    # Structure: results[(num_colors, fill_percentage, num_ants)] = (avg_quality, steps)
    results = {}
    
    # Calculate total number of configurations
    total_configs = len(FILL_PERCENTAGES) * len(NUM_COLORS_LIST) * len(NUM_ANTS_LIST)
    
    # Run all simulations
    print("\nRunning simulations...")
    config_count = 0
    with tqdm(total=total_configs * NUM_RUNS, desc="Progress") as pbar:
        for num_colors in NUM_COLORS_LIST:
            for fill_percentage in FILL_PERCENTAGES:
                for num_ants in NUM_ANTS_LIST:
                    config_count += 1
                    if not HAS_TQDM:
                        print(f"Running config {config_count}/{total_configs}: "
                              f"Colors={num_colors}, Fill={fill_percentage}%, Ants={num_ants}")
                    
                    # Run and average
                    avg_quality, steps = average_runs(
                        HEIGHT, WIDTH, num_colors, fill_percentage, num_ants, 
                        NUM_STEPS, NUM_RUNS, TRACK_INTERVAL
                    )
                    results[(num_colors, fill_percentage, num_ants)] = (avg_quality, steps)
                    pbar.update(NUM_RUNS)
    
    # Create facet grid plot
    print("\nGenerating visualization...")
    n_rows = len(NUM_COLORS_LIST)
    n_cols = len(FILL_PERCENTAGES)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Handle case where there's only one subplot or one row/column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Color map for different num_ants values
    colors = plt.cm.tab10(np.linspace(0, 1, len(NUM_ANTS_LIST)))
    
    # Plot each subplot
    for i, num_colors in enumerate(NUM_COLORS_LIST):
        for j, fill_percentage in enumerate(FILL_PERCENTAGES):
            ax = axes[i, j]
            
            # Plot lines for each num_ants value
            for k, num_ants in enumerate(NUM_ANTS_LIST):
                avg_quality, steps = results[(num_colors, fill_percentage, num_ants)]
                ax.plot(steps, avg_quality, color=colors[k], linewidth=2, 
                       label=f'{num_ants} ants', alpha=0.8)
            
            # Set labels and title
            if i == n_rows - 1:  # Bottom row
                ax.set_xlabel('Time Step', fontsize=10)
            if j == 0:  # Left column
                ax.set_ylabel('Clustering Quality μ′', fontsize=10)
            
            # Title shows fill_percentage (column) and num_colors (row)
            ax.set_title(f'Colors: {num_colors}, Fill: {fill_percentage}%', 
                        fontsize=11, fontweight='bold')
            
            # Set limits and grid
            ax.set_xlim([0, NUM_STEPS])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
    
    # Add overall title
    fig.suptitle('Clustering Quality Over Time by Configuration', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(OUTPUT_DIR, 'metrics_facet_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to '{output_path}'")
    
    plt.close()


if __name__ == "__main__":
    generate_metrics()

