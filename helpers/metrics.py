"""
Metrics for tracking clustering quality in the ant sorting simulation.
"""


def calculate_clustering_score(grid):
    """
    Calculate how well the items are clustered on a given grid.

    Formula: Sum of adjacent matching pairs / Total items
    This provides the Global Reward signal.

    Args:
        grid (Grid): The grid world to evaluate

    Returns:
        float: Clustering score (0.0 to 1.0), where higher values indicate better clustering
    """
    score = 0
    h, w = grid.cells.shape
    total_items = 0

    for y in range(h):
        for x in range(w):
            val = grid.cells[y, x]
            if val == 0:
                continue
            total_items += 1

            # Check right neighbor
            if x + 1 < w and grid.cells[y, x + 1] == val:
                score += 1
            # Check down neighbor
            if y + 1 < h and grid.cells[y + 1, x] == val:
                score += 1

    if total_items == 0:
        return 0.0

    # Normalize by total items to keep reward magnitude consistent
    # regardless of how full the grid is.
    return score / total_items
