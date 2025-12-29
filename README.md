# Data Analyst Forecasting Project

Fashion sales forecasting using NHITS neural network model.

## Project Structure

```
.
├── data/
│   ├── generate_data.py      # Data generation module
│   └── files/                 # Data files
├── src/
│   ├── data_processing.py     # Data processing utilities
│   ├── models.py              # Model configurations
│   └── pipeline.py            # ML pipeline orchestration
├── visualization/
│   └── charts.py              # Visualization utilities
├── reports/                   # Generated reports and charts
├── main.py                    # Entry point
├── pyproject.toml             # Project dependencies and config
└── .python-version            # Python version specification
```

## Quick Start

Run the setup script:

```bash
./scripts/setup.sh
```

Or manually install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
```

## Usage

Run the forecasting pipeline:

```bash
uv run python main.py
```

**Note for macOS users**: If you encounter MPS device errors, the code automatically enables CPU fallback. Alternatively, you can set:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python main.py
```

Or activate the virtual environment:

```bash
source .venv/bin/activate
python main.py
```

### Using Makefile

```bash
make install    # Install dependencies
make run        # Run the pipeline
make test       # Run tests
make clean      # Clean up generated files
make dev        # Start IPython shell
```

## Modules

### data/generate_data.py
Generates synthetic fashion sales data with trend, seasonality, and noise.

### src/data_processing.py
Functions for splitting train/test data and merging forecasts.

### src/models.py
NHITS model configuration with customizable hyperparameters.

### src/pipeline.py
End-to-end pipeline orchestrating data loading, training, prediction, and visualization.

### visualization/charts.py
Creates comparison charts between actual and predicted values.

## Configuration

Edit `src/models.py` to modify:
- Forecast horizon
- Input size
- Confidence levels
- Training parameters

