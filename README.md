# Ant Sorting Simulation

A 2D simulation where rule-based "ants" sort colored items on a grid. This project implements a classical ant clustering algorithm using probabilistic pick/drop rules based on local similarity.

## Features

### Rule-Based Ants

- **Simple pick/drop rules** based on local similarity (Deneubourg model)
- **Pick rule**: Ants pick up items when local similarity is low (item doesn't fit well)
- **Drop rule**: Ants drop items when local similarity is high (item fits well)
- **Random movement**: Ants move randomly in 8 directions (including diagonals)

### Grid World

- 2D bounded grid (no wrapping)
- Configurable parameters:
  - Grid size (H × W)
  - Number of colors (K)
  - Fill percentage (p%)
- Random object placement with even color distribution

### Metrics

- **Clustering score**: Measures how well similar-colored items are clustered together
- Tracks improvement in sorting over time
- Visualizes clustering quality over simulation steps

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the rule-based simulation:

```bash
# Headless mode (default)
python main.py

# With visualization
python main.py --interface

# Step-by-step mode (requires --interface)
python main.py --interface --step-by-step

# Override configuration parameters
python main.py --k1 0.1 --k2 0.3 --max-steps 2000
```

The simulation will:

- Create a grid with configurable size, colors, and fill percentage
- Run multiple ants for a specified number of steps
- Display progress and final results
- Generate plots:
  - `results/clustering_quality.png`: Clustering score over time

## Configuration

Edit `config.py` to adjust parameters:

```python
# Grid dimensions
HEIGHT = 20
WIDTH = 50

# Simulation parameters
NUM_COLORS = 2        # Number of different colored object types
FILL_PERCENTAGE = 30  # Percentage of grid cells filled (0-100)
NUM_ANTS = 10         # Number of ants in the simulation
NUM_STEPS = 5000      # Number of simulation steps to run

# Ant behavior parameters (Deneubourg model thresholds)
K1_PICK_THRESHOLD = 0.3   # Pick threshold (lower = more likely to pick)
K2_DROP_THRESHOLD = 0.15  # Drop threshold (lower = more likely to drop)
```

## Project Structure

```
.
├── main.py              # Main entry point
├── config.py            # Configuration parameters
├── objects/             # Core simulation objects
│   ├── __init__.py
│   ├── grid.py          # Grid world implementation
│   └── ant.py           # Ant classes (base and rule-based)
├── helpers/             # Utility functions
│   ├── __init__.py
│   └── metrics.py       # Clustering quality metrics
├── results/             # Output directory for plots
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Algorithm

The rule-based ants use the Deneubourg model for probabilistic pick/drop decisions:

- **Pick probability**: `P_pick = (k1 / (k1 + similarity))²`
  - Higher when similarity is low (item doesn't fit well)
- **Drop probability**: `P_drop = (similarity / (k2 + similarity))²`
  - Higher when similarity is high (item fits well)

Where `similarity` is the fraction of neighboring cells with the same color.

## Examples

### Basic run (headless)

```bash
python main.py
```

### With live visualization

```bash
python main.py --interface
```

### Custom parameters

```bash
python main.py --k1 0.1 --k2 0.3 --max-steps 2000 --interface
```

## Output

The simulation generates:

- Console output showing progress and final results
- `results/clustering_quality.png`: Plot of clustering score over time

## Authors

Daniel Escobar & Noah van Potten  
ELTE University Budapest, 2025 Fall Semester, Computational Intelligence
