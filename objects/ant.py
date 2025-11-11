"""
Rule-Based Ant Agent for Sorting Colored Objects
"""
import random
import numpy as np


class Ant:
    """
    A simple ant agent that follows fixed pick/drop rules based on local similarity.
    
    Attributes:
        row (int): Current row position
        col (int): Current column position
        carrying (int): Currently carried object color (0 = not carrying)
        grid (GridWorld): Reference to the grid world
        random_move_prob (float): Probability of random movement component
    """
    
    # Directions: up, down, left, right, and diagonals
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    def __init__(self, grid, start_row=None, start_col=None, random_move_prob=0.1):
        """
        Initialize an ant.
        
        Args:
            grid (GridWorld): The grid world to operate on
            start_row (int): Starting row (random if None)
            start_col (int): Starting column (random if None)
            random_move_prob (float): Probability of random movement (default 0.1)
        """
        self.grid = grid
        self.random_move_prob = random_move_prob
        
        # Random starting position if not specified
        if start_row is None:
            start_row = random.randint(0, grid.height - 1)
        if start_col is None:
            start_col = random.randint(0, grid.width - 1)
        
        self.row = start_row
        self.col = start_col
        self.carrying = 0  # 0 = not carrying, 1-K = carrying color
    
    def step(self):
        """
        Execute one step: decide to pick/drop, then move.
        """
        current_cell_value = self.grid.get(self.row, self.col)
        
        # Decision: pick or drop
        if self.carrying == 0 and current_cell_value != 0:
            # Not carrying and cell has object: decide whether to pick
            self._decide_pick(current_cell_value)
        elif self.carrying != 0 and current_cell_value == 0:
            # Carrying and cell is empty: decide whether to drop
            self._decide_drop()
        
        # Move to adjacent cell
        self._move()
    
    def _decide_pick(self, color):
        """
        Decide whether to pick up an object based on local similarity.
        Rule: Pick probability increases when local match is low (object doesn't fit well here).
        Uses probabilistic decision based on Deneubourg's model.
        
        Args:
            color (int): Color of the object to potentially pick
        """
        similarity = self.grid.get_local_similarity(self.row, self.col, color)
        
        # Pick probability: higher when similarity is lower
        # Formula: P_pick = (k1 / (k1 + similarity))^2
        # This makes picking more likely when similarity is low
        k1 = 0.1  # Threshold parameter
        pick_prob = (k1 / (k1 + similarity)) ** 2
        
        if random.random() < pick_prob:
            self.carrying = color
            self.grid.set(self.row, self.col, 0)
    
    def _decide_drop(self):
        """
        Decide whether to drop the carried object based on local similarity.
        Rule: Drop probability increases when local match is high (object fits well here).
        Uses probabilistic decision based on Deneubourg's model.
        """
        similarity = self.grid.get_local_similarity(self.row, self.col, self.carrying)
        
        # Drop probability: higher when similarity is higher
        # Formula: P_drop = (similarity / (k2 + similarity))^2
        # This makes dropping more likely when similarity is high
        k2 = 0.3  # Threshold parameter
        drop_prob = (similarity / (k2 + similarity)) ** 2
        
        if random.random() < drop_prob:
            self.grid.set(self.row, self.col, self.carrying)
            self.carrying = 0
    
    def _move(self):
        """
        Move to an adjacent cell with a small random component.
        Stays within grid bounds (no wrapping).
        """
        # Get valid directions (only those that stay within bounds)
        valid_directions = []
        for dr, dc in self.DIRECTIONS:
            new_row = self.row + dr
            new_col = self.col + dc
            if self.grid.is_valid(new_row, new_col):
                valid_directions.append((dr, dc))
        
        # If no valid directions (shouldn't happen), don't move
        if not valid_directions:
            return
        
        # Choose a direction from valid ones
        dr, dc = random.choice(valid_directions)
        
        # Update position (stays within bounds)
        self.row += dr
        self.col += dc

