# WARNING: UNDER HEAVY CONSTRUCTION!

This code is currently just a basis for a future software system for using PINOs with reservoir simulation models. Most source files are placeholders while some can be used independently. 



# PINO Surrogate for Reservoir Simulation

A physics-informed neural operator (PINO) surrogate model for reservoir simulation.

## Overview

This project provides a tool to create physics-informed neural operator (PINO) surrogates for reservoir simulators. The surrogate model combines deep learning with physical constraints to provide fast and accurate predictions of reservoir behavior.

### Features

- Physics-informed neural operator architecture
- Support for both NVIDIA CUDA and AMD ROCm GPUs
- Graphical user interface for easy model setup and training
- Interactive 2D and 3D visualization of reservoir properties and grid
- Comprehensive visualization and analysis tools
- Integration with Eclipse simulation files (.EGRID, .INIT, .UNRST, .SMSPEC, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pino_surrogate_for_reservoir_simulation.git
cd pino_surrogate_for_reservoir_simulation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Use the graphical interface to:
   - Load simulation files (EGRID, INIT, UNRST, SMSPEC/UNSMRY)
   - Configure model parameters
   - Train the surrogate model
   - Visualize results
   - Explore grid and property data in 2D and 3D

### Visualization Features

The application includes comprehensive visualization capabilities:

- **Interactive Grid Visualization**: View reservoir grid structure in both 2D and 3D
- **Property Visualization**: Visualize properties from INIT and restart files with interactive slicing
- **Summary Data Plots**: Analyze time-series summary data with multi-axis plots
- **Model Results**: Compare original simulation results with PINO model predictions

## Project Structure

```
pino_surrogate_for_reservoir_simulation/
├── src/
│   ├── core/
│   │   ├── pino_model.py      # PINO model implementation
│   │   ├── physics.py         # Physics constraints
│   │   ├── visualization.py   # Visualization tools
│   │   ├── data_processing.py # Data processing utilities
│   │   └── eclipse_reader.py  # Eclipse file reader
│   ├── gui/
│   │   └── main_window.py     # Main GUI window
│   └── main.py                # Application entry point
├── tests/                     # Test suite
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Development

### Code Style

The project follows PEP 8 guidelines and uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

### Testing

Run the test suite:
```bash
pytest tests/
```

## Dependencies

- **PyQt6**: GUI framework
- **PyQt6-WebEngine**: Required for interactive visualizations
- **plotly**: Interactive 2D and 3D plots
- **resdata**: Eclipse file reader (replacement for libecl)
- **numpy/pandas**: Data processing
- **torch**: Deep learning framework
- **matplotlib/seaborn**: Static visualization

## License

This project is proprietary and confidential. All rights reserved.

## Author

- Bjørn Egil Ludvigsen, Aker BP 
