"""
Live visualization for the ant sorting simulation.
"""
import warnings
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np
from helpers.metrics import calculate_clustering_quality

# Suppress matplotlib deprecation warnings for get_cmap
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)


class LiveVisualizer:
    """
    Provides live visualization of the ant sorting simulation.
    """
    
    def __init__(self, grid, num_colors, update_interval=1):
        """
        Initialize the visualizer.
        
        Args:
            grid (GridWorld): The grid to visualize
            num_colors (int): Number of colors
            update_interval (int): Update every N steps (default 1 for every step)
        """
        self.grid = grid
        self.num_colors = num_colors
        self.update_interval = update_interval
        self.current_step = 0
        self.current_quality = 0.0
        
        # Setup figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Setup grid visualization
        self.cmap = plt.cm.get_cmap('tab10', num_colors + 1)
        self.im = self.ax1.imshow(
            grid.grid, 
            cmap=self.cmap, 
            vmin=0, 
            vmax=num_colors, 
            interpolation='nearest'
        )
        self.ax1.set_title('Grid State', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Column', fontsize=12)
        self.ax1.set_ylabel('Row', fontsize=12)
        
        # Add colorbar
        self.cbar = plt.colorbar(self.im, ax=self.ax1, ticks=range(num_colors + 1))
        self.cbar.set_label('Color (0 = Empty)', fontsize=10)
        
        # For drawing ants - we'll use scatter plots
        self.ant_scatter = None  # For ants not carrying
        self.carrying_scatter = None  # For ants carrying objects
        
        # Setup quality plot
        self.quality_history = []
        self.step_history = []
        self.line, = self.ax2.plot([], [], 'b-', linewidth=2)
        self.ax2.set_xlabel('Time Step', fontsize=12)
        self.ax2.set_ylabel('Clustering Quality μ′', fontsize=12)
        self.ax2.set_title('Clustering Quality Over Time', fontsize=14, fontweight='bold')
        self.ax2.set_ylim([0, 1])
        self.ax2.grid(True, alpha=0.3)
        
        # Text for step and quality info
        self.info_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
        
        # Mode indicator text
        self.mode_text = self.fig.text(0.5, 0.05, 'Mode: Auto | Press M for step-by-step, D for headless', 
                                       ha='center', fontsize=10, style='italic')
        
        # Mode tracking
        self.step_by_step_mode = False
        self.waiting_for_space = False
        self.headless_mode = False
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
    
    def update(self, step, grid, quality, ants):
        """
        Update the visualization.
        
        Args:
            step (int): Current step number
            grid (GridWorld): Current grid state
            quality (float): Current clustering quality
            ants (list): List of ant agents
        """
        self.current_step = step
        self.current_quality = quality
        
        # Update grid visualization
        self.im.set_array(grid.grid)
        
        # Remove old ant markers
        if self.ant_scatter is not None:
            try:
                self.ant_scatter.remove()
            except:
                pass
            self.ant_scatter = None
        if self.carrying_scatter is not None:
            try:
                self.carrying_scatter.remove()
            except:
                pass
            self.carrying_scatter = None
        
        # Separate ants by carrying status
        ants_not_carrying = []
        ants_carrying = []
        carrying_colors = []
        
        for ant in ants:
            if ant.carrying == 0:
                ants_not_carrying.append((ant.row, ant.col))
            else:
                ants_carrying.append((ant.row, ant.col))
                carrying_colors.append(ant.carrying)
        
        # Draw ants not carrying (black squares)
        if ants_not_carrying:
            rows, cols = zip(*ants_not_carrying)
            self.ant_scatter = self.ax1.scatter(
                cols, rows, 
                c='black', 
                s=100,  # Size of the square
                marker='s',  # Square marker
                edgecolors='white',
                linewidths=0.5,
                zorder=10  # Draw on top
            )
        
        # Draw ants carrying objects (colored squares)
        if ants_carrying:
            rows, cols = zip(*ants_carrying)
            # Get colors from colormap
            colors = [self.cmap(color) for color in carrying_colors]
            self.carrying_scatter = self.ax1.scatter(
                cols, rows,
                c=colors,
                s=150,  # Slightly larger to show it's carrying
                marker='s',  # Square marker
                edgecolors='black',
                linewidths=1.5,
                zorder=11  # Draw on top of regular ants
            )
        
        # Update quality plot
        self.quality_history.append(quality)
        self.step_history.append(step)
        
        if len(self.quality_history) > 0:
            self.line.set_data(self.step_history, self.quality_history)
            if len(self.step_history) > 1:
                self.ax2.set_xlim([0, max(self.step_history)])
        
        # Update info text
        carrying_count = sum(1 for ant in ants if ant.carrying != 0)
        mode_str = 'Step-by-Step (Press SPACE)' if self.step_by_step_mode else 'Auto'
        self.info_text.set_text(
            f'Step: {step} | Quality μ′: {quality:.4f} | Ants carrying: {carrying_count}/{len(ants)} | Mode: {mode_str}'
        )
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def should_update(self, step):
        """Check if we should update at this step."""
        return step % self.update_interval == 0
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'm' or event.key == 'M':
            # Toggle step-by-step mode
            self.step_by_step_mode = not self.step_by_step_mode
            if not self.step_by_step_mode:
                self.waiting_for_space = False  # Release if switching to auto
            mode_str = 'Step-by-Step (Press SPACE)' if self.step_by_step_mode else 'Auto'
            self.mode_text.set_text(f'Mode: {mode_str} | Press M to toggle, D for headless')
            self.fig.canvas.draw()
        elif (event.key == ' ' or event.key == 'space') and self.step_by_step_mode:
            # Advance one step in step-by-step mode
            self.waiting_for_space = False
        elif event.key == 'd' or event.key == 'D':
            # Headless mode - signal to close visualization
            self.headless_mode = True
            self.close()
    
    def wait_for_step(self):
        """Wait for space key press in step-by-step mode."""
        if self.step_by_step_mode:
            self.waiting_for_space = True
            while self.waiting_for_space:
                plt.pause(0.1)
                if not plt.fignum_exists(self.fig.number):
                    break
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)

