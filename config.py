"""
Configuration parameters for the ant sorting simulation.
"""

# Grid dimensions
HEIGHT = 20
WIDTH = 20

# Simulation parameters
NUM_COLORS = 2  # Number of different colored object types
FILL_PERCENTAGE = 20  # Percentage of grid cells filled with objects (0-100)
NUM_ANTS = 10  # Number of ants in the simulation
NUM_STEPS = 5000  # Number of simulation steps to run

# Visualization parameters
VIS_UPDATE_INTERVAL = 1  # Update visualization every N steps

# Output directory
OUTPUT_DIR = "results"

# Ant behavior parameters (Deneubourg model thresholds)
K1_PICK_THRESHOLD = 0.3  # Pick threshold parameter (lower = more likely to pick)
K2_DROP_THRESHOLD = 0.15  # Drop threshold parameter (lower = more likely to drop)
