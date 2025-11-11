"""
2D Bounded Grid World for Ant Sorting Simulation
"""
import numpy as np
import random


class GridWorld:
    """
    A 2D bounded grid world where ants can move and sort colored objects.
    Edges do not wrap around - borders are boundaries.
    
    Attributes:
        height (int): Grid height
        width (int): Grid width
        num_colors (int): Number of different colors (K)
        grid (np.ndarray): 2D array where 0 = empty, 1-K = colored objects
        empty_cells (set): Set of (row, col) tuples for empty cells
    """
    
    def __init__(self, height, width, num_colors, fill_percentage):
        """
        Initialize the grid world.
        
        Args:
            height (int): Grid height (H)
            width (int): Grid width (W)
            num_colors (int): Number of colors (K) to scatter
            fill_percentage (float): Percentage of blocks (p%) to populate (0-100)
        """
        self.height = height
        self.width = width
        self.num_colors = num_colors
        self.grid = np.zeros((height, width), dtype=int)
        
        # Calculate number of objects to place
        total_cells = height * width
        num_objects = int(total_cells * fill_percentage / 100.0)
        
        # Distribute objects roughly evenly across colors
        objects_per_color = num_objects // num_colors
        remainder = num_objects % num_colors
        
        # Place objects
        all_positions = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(all_positions)
        
        pos_idx = 0
        for color in range(1, num_colors + 1):
            count = objects_per_color + (1 if color <= remainder else 0)
            for _ in range(count):
                if pos_idx < len(all_positions):
                    r, c = all_positions[pos_idx]
                    self.grid[r, c] = color
                    pos_idx += 1
        
        # Track empty cells
        self.empty_cells = set()
        for r in range(height):
            for c in range(width):
                if self.grid[r, c] == 0:
                    self.empty_cells.add((r, c))
    
    def is_valid(self, row, col):
        """Check if a position is within grid bounds."""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def get(self, row, col):
        """
        Get the value at a position (bounded, no wrapping).
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            int: Grid value (0 = empty, 1-K = colored object), or 0 if out of bounds
        """
        if not self.is_valid(row, col):
            return 0
        return self.grid[row, col]
    
    def set(self, row, col, value):
        """
        Set the value at a position (bounded, no wrapping).
        
        Args:
            row (int): Row index
            col (int): Column index
            value (int): Value to set (0 = empty, 1-K = colored object)
        """
        if not self.is_valid(row, col):
            return
        old_value = self.grid[row, col]
        self.grid[row, col] = value
        
        # Update empty cells set
        if old_value != 0 and value == 0:
            self.empty_cells.add((row, col))
        elif old_value == 0 and value != 0:
            self.empty_cells.discard((row, col))
    
    def is_empty(self, row, col):
        """Check if a cell is empty."""
        return self.get(row, col) == 0
    
    def get_neighbors(self, row, col, radius=1):
        """
        Get neighboring cells (bounded, no wrapping).
        Only returns neighbors that are within grid bounds.
        Corner cells have 3 neighbors, edge cells have 5, interior cells have 8.
        
        Args:
            row (int): Row index
            col (int): Column index
            radius (int): Neighbor radius (default 1 for 8 neighbors)
            
        Returns:
            list: List of (row, col, value) tuples for valid neighbors
        """
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r = row + dr
                c = col + dc
                # Only add if within bounds
                if self.is_valid(r, c):
                    neighbors.append((r, c, self.grid[r, c]))
        return neighbors
    
    def get_local_similarity(self, row, col, color):
        """
        Calculate local similarity: fraction of neighbors with the same color.
        
        Args:
            row (int): Row index
            col (int): Column index
            color (int): Color to check similarity for
            
        Returns:
            float: Fraction of neighbors with matching color (0.0 to 1.0)
        """
        neighbors = self.get_neighbors(row, col)
        if not neighbors:
            return 0.0
        
        matching = sum(1 for _, _, val in neighbors if val == color)
        return matching / len(neighbors)

