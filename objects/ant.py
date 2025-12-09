"""
Ant classes for the ant sorting simulation.
Contains the base Ant class and RuleBasedAnt implementation.
"""

import random


class Ant:
    """
    Base ant class representing an agent on the grid.

    Attributes:
        x (int): Current X position of the ant
        y (int): Current Y position of the ant
        carrying (int or None): Currently carried object color (None = not carrying)
    """

    def __init__(self, x, y):
        """
        Initialize an ant.

        Args:
            x (int): Starting X position
            y (int): Starting Y position
        """
        self.x = x  # current X position of the ant
        self.y = y  # current Y position of the ant
        self.carrying = None  # None or color_id (int)


class RuleBasedAnt(Ant):
    """
    Rule-based ant that follows classical behavior rules from ant clustering algorithms.
    Extends the base Ant class by adding interaction with the grid and probabilistic pick/drop decisions.

    Key Behaviors:
    - Pick rule: If the ant is empty-handed and stands on an item, it may pick it up.
                 The probability decreases when many similar items are nearby.
    - Drop rule: If the ant is carrying an item and stands on an empty cell, it may drop it.
                 The probability increases when many similar items are nearby.
    - Movement: The ant randomly moves to any valid neighboring cell (8-direction movement).
    """

    DIRECTIONS = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]  # possible movement directions

    def __init__(self, grid, k1, k2, start_x=None, start_y=None):
        """
        Initialize a rule-based ant.

        Args:
            grid (Grid): The grid world to operate on
            k1 (float): Pick threshold parameter
            k2 (float): Drop threshold parameter
            start_x (int, optional): Starting X position (random if None)
            start_y (int, optional): Starting Y position (random if None)
        """
        # Initialize base Ant with coordinates
        x = start_x if start_x is not None else random.randint(0, grid.width - 1)
        y = start_y if start_y is not None else random.randint(0, grid.height - 1)
        super().__init__(x, y)

        self.grid = grid
        self.k1 = k1
        self.k2 = k2

    def step(self):
        """
        Execute one time step for the ant.
        Applies pickup rule, drop rule, and random movement.
        """
        cell_value = self.grid.get(self.x, self.y)
        self._pickup_rule(cell_value)
        self._drop_rule(cell_value)
        self._random_movement()

    def _pickup_rule(self, cell_value):
        """
        Implementation of the pick rule.
        Can only pick up if hands are empty AND cell has an item.
        Pick probability increases when similarity is low (object doesn't fit well here).

        Args:
            cell_value (int): Value of the current cell
        """
        if self.carrying is None and cell_value != 0:
            sim = self.grid.get_local_similarity(
                self.x, self.y, cell_value
            )  # compute similarity locally
            pick_prob = (self.k1 / (self.k1 + sim)) ** 2

            # Pick item stochastically based on similarity
            if random.random() < pick_prob:
                self.carrying = cell_value
                self.grid.cells[self.y, self.x] = 0  # remove item from grid

    def _drop_rule(self, cell_value):
        """
        Implementation of the drop rule.
        Can only drop if carrying an item AND current cell is empty.
        Drop probability increases when similarity is high (object fits well here).

        Args:
            cell_value (int): Value of the current cell
        """
        if self.carrying is not None and cell_value == 0:
            sim = self.grid.get_local_similarity(self.x, self.y, self.carrying)
            drop_prob = (sim / (self.k2 + sim)) ** 2

            # Drop item stochastically based on similarity
            if random.random() < drop_prob:
                self.grid.cells[self.y, self.x] = self.carrying
                self.carrying = None

    def _random_movement(self):
        """
        Choose a random move from the possible moves.
        Moves in one of 8 directions (including diagonals) if valid.
        """
        valid_moves = []
        # Check all possible movement directions
        for dx, dy in self.DIRECTIONS:
            nx = self.x + dx
            ny = self.y + dy
            # Check whether the next position is inside the grid boundaries
            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                valid_moves.append((dx, dy))
        # Perform a random valid movement
        if valid_moves:
            dx, dy = random.choice(valid_moves)
            self.x += dx
            self.y += dy
