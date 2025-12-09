"""
Generate a matrix visualization showing clustering quality over time
for different combinations of k1, k2, and number of ants.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Suppress matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# Use non-interactive backend for batch processing
matplotlib.use("Agg")

from objects.grid import Grid
from objects.ant import RuleBasedAnt
from helpers.metrics import calculate_clustering_score
import config


def run_simulation(
    width,
    height,
    num_colors,
    fill_percentage,
    num_ants,
    num_steps,
    k1,
    k2,
    track_interval=100,
):
    """
    Run a single simulation and return clustering scores over time.

    Args:
        width (int): Grid width
        height (int): Grid height
        num_colors (int): Number of colors
        fill_percentage (float): Fill percentage (0.0 to 1.0)
        num_ants (int): Number of ants
        num_steps (int): Number of simulation steps
        k1 (float): Pick threshold parameter
        k2 (float): Drop threshold parameter
        track_interval (int): How often to track scores (every N steps)

    Returns:
        list: Clustering scores at tracked intervals
        list: Step numbers corresponding to scores
    """
    # Initialize Grid and Ants
    grid = Grid(width, height, num_colors, fill_percentage)
    ants = []
    for i in range(num_ants):
        ants.append(RuleBasedAnt(grid, k1, k2))

    # Track clustering scores
    scores = []
    steps = []

    # Initial score
    initial_score = calculate_clustering_score(grid)
    scores.append(initial_score)
    steps.append(0)

    # Run simulation
    for step in range(1, num_steps + 1):
        # Step all ants
        for ant in ants:
            ant.step()

        # Track score at intervals
        if step % track_interval == 0 or step == num_steps:
            score = calculate_clustering_score(grid)
            scores.append(score)
            steps.append(step)

    return scores, steps


def generate_matrix():
    """
    Generate a matrix visualization of clustering quality over time
    for different combinations of k1, k2, and number of ants.
    """
    # Configuration parameters
    WIDTH = config.WIDTH
    HEIGHT = config.HEIGHT
    NUM_COLORS = config.NUM_COLORS
    FILL_PERCENTAGE = config.FILL_PERCENTAGE / 100.0  # Convert to probability
    NUM_STEPS = config.NUM_STEPS
    OUTPUT_DIR = config.OUTPUT_DIR

    # Parameter ranges to test
    K1_VALUES = [0.1, 0.3, 0.5, 0.7]  # x-axis (columns) - pick threshold
    K2_VALUES = [0.05, 0.15, 0.25, 0.35]  # y-axis (rows) - drop threshold
    NUM_ANTS_LIST = [5, 10, 20]  # Different lines in each subplot

    TRACK_INTERVAL = 100  # Track quality every 100 steps

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Clustering Quality Matrix Visualization")
    print("=" * 60)
    print(f"Grid size: {HEIGHT} × {WIDTH}")
    print(f"Colors: {NUM_COLORS}")
    print(f"Fill percentage: {FILL_PERCENTAGE * 100:.1f}%")
    print(f"Steps: {NUM_STEPS}")
    print(f"K1 values (pick threshold): {K1_VALUES}")
    print(f"K2 values (drop threshold): {K2_VALUES}")
    print(f"Number of ants: {NUM_ANTS_LIST}")
    print("=" * 60)

    # Store all results
    # Structure: results[(k2, k1, num_ants)] = (scores, steps)
    results = {}

    # Calculate total number of configurations
    total_configs = len(K1_VALUES) * len(K2_VALUES) * len(NUM_ANTS_LIST)

    # Run all simulations
    print("\nRunning simulations...")
    config_count = 0
    for k2 in K2_VALUES:
        for k1 in K1_VALUES:
            for num_ants in NUM_ANTS_LIST:
                config_count += 1
                print(
                    f"Running config {config_count}/{total_configs}: "
                    f"K1={k1}, K2={k2}, Ants={num_ants}"
                )

                # Run simulation
                scores, steps = run_simulation(
                    WIDTH,
                    HEIGHT,
                    NUM_COLORS,
                    FILL_PERCENTAGE,
                    num_ants,
                    NUM_STEPS,
                    k1,
                    k2,
                    TRACK_INTERVAL,
                )
                results[(k2, k1, num_ants)] = (scores, steps)

    # Create facet grid plot
    print("\nGenerating visualization...")
    n_rows = len(K2_VALUES)
    n_cols = len(K1_VALUES)

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
    for i, k2 in enumerate(K2_VALUES):
        for j, k1 in enumerate(K1_VALUES):
            ax = axes[i, j]

            # Plot lines for each num_ants value
            for k, num_ants in enumerate(NUM_ANTS_LIST):
                scores, steps = results[(k2, k1, num_ants)]
                ax.plot(
                    steps,
                    scores,
                    color=colors[k],
                    linewidth=2,
                    label=f"{num_ants} ants",
                    alpha=0.8,
                )

            # Set labels and title
            if i == n_rows - 1:  # Bottom row
                ax.set_xlabel("Time Step", fontsize=10)
            if j == 0:  # Left column
                ax.set_ylabel("Clustering Quality", fontsize=10)

            # Title shows k1 (column) and k2 (row)
            ax.set_title(f"K1={k1}, K2={k2}", fontsize=11, fontweight="bold")

            # Set limits and grid
            ax.set_xlim([0, NUM_STEPS])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)

    # Add overall title and axis labels
    fig.suptitle(
        f"Clustering Quality Over Time (Colors={NUM_COLORS}, Fill={FILL_PERCENTAGE*100:.0f}%)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Add row and column labels
    fig.text(
        0.5, 0.02, "K1 (Pick Threshold)", ha="center", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.02,
        0.5,
        "K2 (Drop Threshold)",
        va="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08)

    # Save plot
    output_path = os.path.join(OUTPUT_DIR, "clustering_quality_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to '{output_path}'")

    plt.close()


if __name__ == "__main__":
    generate_matrix()
