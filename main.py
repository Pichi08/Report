"""
Main script to run the ant sorting simulation with rule-based ants.
"""
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Suppress matplotlib deprecation warnings for get_cmap
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
# Try to use interactive backend, fallback to default if needed
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend
from objects.simulation import Simulation
from helpers.metrics import calculate_clustering_quality
from helpers.visualizer import LiveVisualizer
import config


def main():
    """
    Run the baseline rule-based ant sorting simulation.
    Default mode is headless. Use --interface to enable visualization.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Ant Sorting Simulation - Rule-Based Baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run in headless mode (default)
  python main.py --interface         # Run with visualization
  python main.py --interface --step-by-step  # Start in step-by-step mode
  
Note: Configuration parameters can be modified in config.py
        """
    )
    parser.add_argument('--interface', action='store_true',
                       help='Enable visualization interface (default: headless)')
    parser.add_argument('--step-by-step', action='store_true',
                       help='Start in step-by-step mode (requires --interface)')
    
    args = parser.parse_args()
    
    # Configuration parameters from config file
    HEIGHT = config.HEIGHT
    WIDTH = config.WIDTH
    NUM_COLORS = config.NUM_COLORS
    FILL_PERCENTAGE = config.FILL_PERCENTAGE
    NUM_ANTS = config.NUM_ANTS
    NUM_STEPS = config.NUM_STEPS
    VIS_UPDATE_INTERVAL = config.VIS_UPDATE_INTERVAL
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
    print("Ant Sorting Simulation - Rule-Based Baseline")
    print("=" * 60)
    print(f"Grid size: {HEIGHT} × {WIDTH}")
    print(f"Colors: {NUM_COLORS}")
    print(f"Fill percentage: {FILL_PERCENTAGE}%")
    print(f"Number of ants: {NUM_ANTS}")
    print(f"Time steps: {NUM_STEPS}")
    print(f"Visualization update: every {VIS_UPDATE_INTERVAL} steps")
    print("=" * 60)
    
    # Create simulation
    sim = Simulation(
        height=HEIGHT,
        width=WIDTH,
        num_colors=NUM_COLORS,
        fill_percentage=FILL_PERCENTAGE,
        num_ants=NUM_ANTS,
        num_steps=NUM_STEPS
    )
    
    # Initialize visualization based on mode
    visualizer = None
    if not HEADLESS_MODE:
        # Enable interactive mode
        plt.ion()
        
        # Create visualizer
        visualizer = LiveVisualizer(sim.grid, NUM_COLORS, update_interval=VIS_UPDATE_INTERVAL)
        
        # Set initial mode
        if STEP_BY_STEP_MODE:
            visualizer.step_by_step_mode = True
            visualizer.mode_text.set_text('Mode: Step-by-Step (Press SPACE) | Press M to toggle, D for headless')
        
        # Initial visualization
        initial_quality = calculate_clustering_quality(sim.grid)
        visualizer.update(0, sim.grid, initial_quality, sim.ants)
        plt.pause(0.1)  # Brief pause to show initial state
        
        print("\nRunning simulation with live visualization...")
        print("Controls:")
        print("  M - Toggle step-by-step mode")
        print("  SPACE - Advance one step (in step-by-step mode)")
        print("  D - Switch to headless mode (hide visualization)")
        print("  Close window or press Ctrl+C to stop early.\n")
    else:
        print("\nRunning simulation in headless mode (no visualization)...")
        print("Press Ctrl+C to stop early.\n")
    
    # Define callback for visualization
    def update_vis(step, grid, quality, ants):
        if visualizer is None or visualizer.headless_mode:
            return
        
        # Always update in step-by-step mode, or at intervals in auto mode
        should_vis_update = visualizer.step_by_step_mode or visualizer.should_update(step)
        
        if should_vis_update:
            visualizer.update(step, grid, quality, ants)
            
            # Wait for space in step-by-step mode
            if visualizer.step_by_step_mode:
                visualizer.wait_for_step()
            else:
                plt.pause(0.001)  # Small pause to allow GUI updates
    
    # Run simulation
    try:
        if HEADLESS_MODE:
            # Headless mode - no visualization
            for step in range(NUM_STEPS):
                # Each ant takes one step
                for ant in sim.ants:
                    ant.step()
                
                # Calculate quality for this step
                quality = calculate_clustering_quality(sim.grid)
                
                # Track quality at intervals
                if (step + 1) % 1000 == 0 or step == NUM_STEPS - 1:
                    sim.quality_history.append(quality)
                    if (step + 1) % 1000 == 0:
                        print(f"Step {step + 1}/{NUM_STEPS}: Quality μ′ = {quality:.4f}")
        else:
            # Visualization mode
            for step in range(NUM_STEPS):
                # Check if headless mode was activated
                if visualizer.headless_mode:
                    print("\nSwitching to headless mode...")
                    visualizer.close()
                    visualizer = None
                    # Continue running without visualization
                    for remaining_step in range(step, NUM_STEPS):
                        for ant in sim.ants:
                            ant.step()
                        quality = calculate_clustering_quality(sim.grid)
                        if (remaining_step + 1) % 1000 == 0 or remaining_step == NUM_STEPS - 1:
                            sim.quality_history.append(quality)
                            if (remaining_step + 1) % 1000 == 0:
                                print(f"Step {remaining_step + 1}/{NUM_STEPS}: Quality μ′ = {quality:.4f}")
                    break
                
                # Each ant takes one step
                for ant in sim.ants:
                    ant.step()
                
                # Calculate quality for this step
                quality = calculate_clustering_quality(sim.grid)
                
                # Track quality at intervals
                if (step + 1) % 1000 == 0 or step == NUM_STEPS - 1:
                    sim.quality_history.append(quality)
                
                # Call step callback for visualization
                update_vis(step + 1, sim.grid, quality, sim.ants)
                
                # Check if window was closed
                if visualizer and not plt.fignum_exists(visualizer.fig.number):
                    print("\nVisualization window closed. Continuing in headless mode...")
                    visualizer.headless_mode = True
                    continue
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    # Final update if visualization still exists
    if visualizer and not visualizer.headless_mode and plt.fignum_exists(visualizer.fig.number):
        final_quality = calculate_clustering_quality(sim.grid)
        visualizer.update(NUM_STEPS, sim.grid, final_quality, sim.ants)
    
    # Results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    initial_quality = sim.quality_history[0] if sim.quality_history else initial_quality
    final_quality = sim.get_final_quality()
    improvement = final_quality - initial_quality
    
    print(f"Initial clustering quality μ′: {initial_quality:.4f}")
    print(f"Final clustering quality μ′: {final_quality:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    
    # Save final plots
    plt.figure(figsize=(10, 6))
    if len(sim.quality_history) > 0:
        steps_tracked = np.linspace(0, NUM_STEPS, len(sim.quality_history))
        plt.plot(steps_tracked, sim.quality_history, 'b-', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Clustering Quality μ′', fontsize=12)
    plt.title('Clustering Quality Over Time (Rule-Based Ants)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    quality_path = os.path.join(OUTPUT_DIR, 'clustering_quality.png')
    plt.savefig(quality_path, dpi=150)
    print(f"\nQuality plot saved to '{quality_path}'")
    
    # Visualize final grid state
    grid_path = os.path.join(OUTPUT_DIR, 'final_grid.png')
    visualize_grid(sim.grid, NUM_COLORS, grid_path)
    print(f"Final grid visualization saved to '{grid_path}'")
    
    if visualizer and not visualizer.headless_mode and plt.fignum_exists(visualizer.fig.number):
        print("\nPress any key in the visualization window to close...")
        plt.show()


def visualize_grid(grid, num_colors, filename):
    """
    Visualize the grid state with different colors for different object types.
    
    Args:
        grid (GridWorld): The grid to visualize
        num_colors (int): Number of colors
        filename (str): Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a colored visualization
    # Use a colormap: 0 = white (empty), 1-K = distinct colors
    cmap = plt.cm.get_cmap('tab10', num_colors + 1)
    
    # Create visualization array (shift colors by 1 so 0 maps to first color)
    vis_grid = grid.grid.copy()
    
    # Plot
    im = ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=num_colors, interpolation='nearest')
    ax.set_title(f'Final Grid State (Quality μ′ = {calculate_clustering_quality(grid):.4f})', 
                 fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(num_colors + 1))
    cbar.set_label('Color (0 = Empty)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

