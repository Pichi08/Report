"""
Metrics for tracking clustering quality in the ant sorting simulation.
"""
import numpy as np


def calculate_clustering_quality(grid):
    """
    Calculate clustering quality score μ′.
    
    The score measures how well objects of the same color are clustered together.
    Higher values indicate better clustering.
    
    Formula: For each colored object, calculate the fraction of its neighbors
    that have the same color. Average this across all objects.
    
    Args:
        grid (GridWorld): The grid world to evaluate
        
    Returns:
        float: Clustering quality score μ′ (0.0 to 1.0)
    """
    similarities = []
    
    for r in range(grid.height):
        for c in range(grid.width):
            color = grid.get(r, c)
            if color != 0:  # If cell has an object
                similarity = grid.get_local_similarity(r, c, color)
                similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    return np.mean(similarities)


def calculate_color_distribution(grid):
    """
    Calculate the distribution of colors across the grid.
    
    Args:
        grid (GridWorld): The grid world to evaluate
        
    Returns:
        dict: Dictionary mapping color -> count
    """
    distribution = {}
    for r in range(grid.height):
        for c in range(grid.width):
            color = grid.get(r, c)
            if color != 0:
                distribution[color] = distribution.get(color, 0) + 1
    return distribution

