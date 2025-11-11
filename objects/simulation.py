"""
Simulation runner for the ant sorting experiment.
"""
import numpy as np
from objects.grid import GridWorld
from objects.ant import Ant
from helpers.metrics import calculate_clustering_quality


class Simulation:
    """
    Manages the simulation of ants sorting objects on a grid.
    
    Attributes:
        grid (GridWorld): The grid world
        ants (list): List of Ant agents
        num_steps (int): Number of time steps to run
        quality_history (list): History of clustering quality scores over time
    """
    
    def __init__(self, height, width, num_colors, fill_percentage, num_ants, num_steps):
        """
        Initialize the simulation.
        
        Args:
            height (int): Grid height
            width (int): Grid width
            num_colors (int): Number of colors (K)
            fill_percentage (float): Percentage of cells to fill with objects
            num_ants (int): Number of ants in the simulation
            num_steps (int): Number of time steps (T) to run
        """
        self.grid = GridWorld(height, width, num_colors, fill_percentage)
        self.num_steps = num_steps
        
        # Create ants
        self.ants = []
        for _ in range(num_ants):
            ant = Ant(self.grid)
            self.ants.append(ant)
        
        # Track quality over time
        self.quality_history = []
        initial_quality = calculate_clustering_quality(self.grid)
        self.quality_history.append(initial_quality)
    
    def run(self, verbose=False, track_interval=100, step_callback=None):
        """
        Run the simulation for the specified number of steps.
        
        Args:
            verbose (bool): Whether to print progress
            track_interval (int): How often to track quality (every N steps)
            step_callback (callable): Optional callback function(step, grid, quality) called each step
        """
        for step in range(self.num_steps):
            # Each ant takes one step
            for ant in self.ants:
                ant.step()
            
            # Calculate quality for this step
            quality = calculate_clustering_quality(self.grid)
            
            # Track quality at intervals
            if (step + 1) % track_interval == 0 or step == self.num_steps - 1:
                self.quality_history.append(quality)
                
                if verbose:
                    print(f"Step {step + 1}/{self.num_steps}: Quality μ′ = {quality:.4f}")
            
            # Call step callback for visualization
            if step_callback is not None:
                step_callback(step + 1, self.grid, quality, self.ants)
    
    def get_final_quality(self):
        """Get the final clustering quality score."""
        return self.quality_history[-1] if self.quality_history else 0.0
    
    def get_quality_history(self):
        """Get the full history of quality scores."""
        return self.quality_history.copy()

