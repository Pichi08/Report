"""
Grid class for the ant sorting simulation.
Represents the 2D environment where ants move and sort colored items.
"""

import numpy as np
import random


class Grid:
    """
    Represents the 2D environment of items.

    Attributes:
        width (int): Grid width
        height (int): Grid height
        cells (np.ndarray): 2D array where 0 = empty, 1-K = colored objects
        num_colors (int): Number of different colored object types
    """

    def __init__(self, width, height, num_colors, fill_percentage=0.4):
        """
        Initialize the grid.

        Args:
            width (int): Grid width
            height (int): Grid height
            num_colors (int): Number of colors
            fill_percentage (float): Probability of filling each cell (0.0 to 1.0)
        """
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width), dtype=int)  # Empty grid
        self.num_colors = num_colors
        self._populate(fill_percentage)  # Populate the empty grid

    def get(self, x, y):
        """
        Get the value at a position.

        Args:
            x (int): Column index
            y (int): Row index

        Returns:
            int: Grid value (0 = empty, 1-K = colored object)
        """
        return self.cells[y, x]

    def _populate(self, fill_percentage):
        """
        Populate the grid with a certain fill percentage.

        Args:
            fill_percentage (float): Probability of filling each cell (0.0 to 1.0)
        """
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < fill_percentage:
                    self.cells[y, x] = random.randint(1, self.num_colors)

    def move_ant(self, ant, dx, dy):
        """
        Move an ant if possible given the grid width and height.

        Args:
            ant (Ant): The ant to move
            dx (int): Change in x direction
            dy (int): Change in y direction

        Returns:
            bool: True if move was successful, False otherwise
        """
        nx, ny = ant.x + dx, ant.y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            ant.x, ant.y = nx, ny
            return True
        return False

    def pick_item(self, ant):
        """
        Let an ant pick up an item if possible.
        Can only pick if: Cell has item AND Ant hands are empty.

        Args:
            ant (Ant): The ant attempting to pick

        Returns:
            bool: True if item was picked, False otherwise
        """
        if self.cells[ant.y, ant.x] != 0 and ant.carrying is None:
            ant.carrying = self.cells[ant.y, ant.x]
            self.cells[ant.y, ant.x] = 0
            return True
        return False

    def drop_item(self, ant):
        """
        Let an ant drop an item if possible.
        Can only drop if: Cell is empty AND Ant has item.

        Args:
            ant (Ant): The ant attempting to drop

        Returns:
            bool: True if item was dropped, False otherwise
        """
        if self.cells[ant.y, ant.x] == 0 and ant.carrying is not None:
            self.cells[ant.y, ant.x] = ant.carrying
            ant.carrying = None
            return True
        return False

    def get_local_similarity(self, x, y, value, radius=1):
        """
        Get the similarity of a value and the neighborhood of one coordinate.
        Calculates the fraction of neighbors with the same color.

        Args:
            x (int): Column index
            y (int): Row index
            value (int): Color value to check similarity for
            radius (int): Neighbor radius (default 1 for 8 neighbors)

        Returns:
            float: Fraction of neighbors with matching color (0.0 to 1.0)
        """
        count_same = 0
        count_total = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # skip center cell
                nx = x + dx
                ny = y + dy
                # Bounds check
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_val = self.get(nx, ny)
                    if neighbor_val != 0:
                        count_total += 1
                        if neighbor_val == value:
                            count_same += 1
        if count_total == 0:
            return 0
        return count_same / count_total
