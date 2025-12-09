"""
Main script to run the rule-based ant sorting simulation.
"""

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Suppress matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# Try to use interactive backend, fallback to default if needed
try:
    matplotlib.use("TkAgg")
except:
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass  # Use default backend

from objects.grid import Grid
from objects.ant import RuleBasedAnt
from helpers.metrics import calculate_clustering_score
import config


def main():
    """
    Run the rule-based ant sorting simulation.
    Default mode is headless. Use --interface to enable visualization.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Ant Sorting Simulation - Rule-Based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run in headless mode (default)
  python main.py --interface         # Run with visualization
  python main.py --interface --step-by-step  # Start in step-by-step mode
  
Note: Configuration parameters can be modified in config.py
        """,
    )
    parser.add_argument(
        "--interface",
        action="store_true",
        help="Enable visualization interface (default: headless)",
    )
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Start in step-by-step mode (requires --interface)",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="Pick threshold parameter (overrides config)",
    )
    parser.add_argument(
        "--k2",
        type=float,
        default=None,
        help="Drop threshold parameter (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps (overrides config)",
    )

    args = parser.parse_args()

    # Configuration parameters from config file
    WIDTH = config.WIDTH
    HEIGHT = config.HEIGHT
    NUM_COLORS = config.NUM_COLORS
    FILL_PERCENTAGE = (
        config.FILL_PERCENTAGE / 100.0
    )  # Convert percentage to probability
    NUM_ANTS = config.NUM_ANTS
    MAX_STEPS = args.max_steps if args.max_steps is not None else config.NUM_STEPS
    K1 = args.k1 if args.k1 is not None else config.K1_PICK_THRESHOLD
    K2 = args.k2 if args.k2 is not None else config.K2_DROP_THRESHOLD
    OUTPUT_DIR = config.OUTPUT_DIR

    # Mode flags
    HEADLESS_MODE = not args.interface  # Default is headless
    STEP_BY_STEP_MODE = args.step_by_step

    # Validate step-by-step requires interface
    if STEP_BY_STEP_MODE and HEADLESS_MODE:
        print("Warning: --step-by-step requires --interface. Enabling interface mode.")
        HEADLESS_MODE = False

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Ant Sorting Simulation - Rule-Based")
    print("=" * 60)
    print(f"Grid size: {HEIGHT} × {WIDTH}")
    print(f"Colors: {NUM_COLORS}")
    print(f"Fill percentage: {FILL_PERCENTAGE * 100:.1f}%")
    print(f"Number of ants: {NUM_ANTS}")
    print(f"Time steps: {MAX_STEPS}")
    print(f"K1 (pick threshold): {K1}")
    print(f"K2 (drop threshold): {K2}")
    print("=" * 60)

    # Initialize Grid and Ants
    grid = Grid(WIDTH, HEIGHT, NUM_COLORS, FILL_PERCENTAGE)
    ants = []
    for i in range(NUM_ANTS):
        ants.append(RuleBasedAnt(grid, K1, K2))

    # For tracking clustering scores
    scores = []
    steps = []

    # Initialize visualization based on mode
    if not HEADLESS_MODE:
        plt.ion()  # interactive mode ON
        fig, ax = plt.subplots(figsize=(10, 6))
        (line,) = ax.plot([], [], "b-", linewidth=2, label="Clustering Score")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Clustering Score", fontsize=12)
        ax.set_title("Rule-Based Ant Clustering Over Time", fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        print("\nRunning simulation with live visualization...")
        print("Press Ctrl+C to stop early.\n")
    else:
        print("\nRunning simulation in headless mode (no visualization)...")
        print("Press Ctrl+C to stop early.\n")

    # Run simulation
    try:
        for step in range(1, MAX_STEPS + 1):
            # Step all ants
            for ant in ants:
                ant.step()

            # Compute clustering score
            score = calculate_clustering_score(grid)
            scores.append(score)
            steps.append(step)

            # Update visualization
            if not HEADLESS_MODE:
                if step % 100 == 0:  # update every 100 steps
                    line.set_data(steps, scores)
                    ax.set_xlim([0, max(steps)])
                    if len(scores) > 0:
                        ax.set_ylim([0, max(1.0, max(scores) * 1.1)])
                    plt.draw()
                    plt.pause(0.001)

            # Print progress
            if step % 1000 == 0:
                print(f"Step {step}/{MAX_STEPS}: Clustering Score = {score:.4f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    if not HEADLESS_MODE:
        plt.ioff()

    # Results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    initial_score = scores[0] if scores else 0.0
    final_score = scores[-1] if scores else 0.0
    improvement = final_score - initial_score

    print(f"Initial clustering score: {initial_score:.4f}")
    print(f"Final clustering score: {final_score:.4f}")
    print(f"Improvement: {improvement:+.4f}")

    # Save final plot
    plt.figure(figsize=(10, 6))
    if len(scores) > 0:
        plt.plot(steps, scores, "b-", linewidth=2, label="Clustering Score")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Clustering Score", fontsize=12)
    plt.title("Clustering Score Over Time (Rule-Based Ants)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, max(1.0, max(scores) * 1.1) if scores else 1.0])
    plt.legend()
    plt.tight_layout()
    quality_path = os.path.join(OUTPUT_DIR, "clustering_quality.png")
    plt.savefig(quality_path, dpi=150)
    print(f"\nQuality plot saved to '{quality_path}'")
    plt.close()

    if not HEADLESS_MODE:
        print("\nPress any key in the visualization window to close...")
        plt.show()


if __name__ == "__main__":
    main()
