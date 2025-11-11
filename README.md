# Ant Sorting Simulation

A 2D simulation where simple "ants" sort colored items on a toroidal grid. This project compares different approaches to ant-based sorting algorithms.

## Features

### Current Implementation (Baseline)

1. **Grid World** (`grid_world.py`)
   - 2D toroidal (wrap-around) grid
   - Configurable parameters:
     - Grid size (H × W)
     - Number of colors (K)
     - Fill percentage (p%)
   - Random object placement with even color distribution

2. **Rule-Based Ants** (`ant.py`)
   - Simple pick/drop rules based on local similarity
   - Pick rule: Pick object if local match is low (< 0.5)
   - Drop rule: Drop object if local match is high (> 0.5)
   - Movement with small random component

3. **Simulation** (`simulation.py`)
   - Coordinates multiple ants over T time steps
   - Tracks clustering quality score μ′ over time

4. **Metrics** (`metrics.py`)
   - Clustering quality score: average local similarity across all objects
   - Tracks improvement in sorting over time

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the baseline rule-based simulation:

```bash
python main.py
```

The simulation will:
- Create a 50×50 grid with 3 colors and 30% fill
- Run 20 ants for 10,000 steps
- Display progress and final results
- Generate plots:
  - `clustering_quality.png`: Quality score over time
  - `final_grid.png`: Visual representation of the final grid state

## Configuration

Edit `main.py` to adjust parameters:

```python
HEIGHT = 50           # Grid height
WIDTH = 50            # Grid width
NUM_COLORS = 3        # Number of colors (K)
FILL_PERCENTAGE = 30  # Percentage of cells filled (p%)
NUM_ANTS = 20         # Number of ants
NUM_STEPS = 10000     # Number of time steps (T)
```

## Project Structure

```
.
├── grid_world.py    # Toroidal grid implementation
├── ant.py           # Rule-based ant agent
├── simulation.py    # Simulation runner
├── metrics.py       # Clustering quality metrics
├── main.py          # Main script
└── requirements.txt # Dependencies
```

## Next Steps

Future implementations will include:
- Neural network policy (learning-based approach)
- Q-learning baseline (optional)

