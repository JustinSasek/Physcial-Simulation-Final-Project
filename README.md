# pygoo
By Justin Sasek

A Python mass-spring physics simulator with PyTorch-based differentiable physics and a pygame UI. Simulate soft body dynamics and optimize parameters through gradient-based learning.

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JustinSasek/Physcial-Simulation-Final-Project.git
cd goo
```

2. Create and activate a conda environment:
```bash
conda create -n sim python=3.11.15 -y
conda activate sim
```

3. Install in development mode with dependencies:
```bash
python -m pip install -e '.[dev]'
```

This installs pygoo and all required dependencies including PyTorch, numpy, matplotlib, pygame, and testing tools.

## Quick Start

### Run the Interactive Simulator

```bash
# Start the simulator with an empty scene
python -m pygoo.app

# Reproduce all figures and videos from my presentation
# This may take several hours to run
sh scripts/run_spring_oscillation.sh
```

## Using the Interactive GUI

When you run `python -m pygoo.app`, a window opens with a simulation viewport on the right and a control panel on the left. This section explains all available controls.

### Control Panel Buttons

#### Simulation Control

- **Run / Pause** - Toggle simulation playback. Click to start or pause the physics simulation.
- **Reset All** - Clear all particles and springs, revert to a blank scene.
- **Clear Particles Only** - Remove all particles but keep the simulation parameters and floor.

#### Scene Management

- **Random Scene** - Generate a random configuration of particles with random positions and connectivity.
- **Import Scene** - Load a complete scene from JSON (both particles/springs and parameters).
- **Import Observables Only** - Load only target trajectory/video without importing parameters. Useful for testing if nonobservable parameters can be recovered through optimization.
- **Export Scene + Video** - Save current state as:
  - `scene.json` - Full scene configuration
  - `scene.traj.npz` - Recorded trajectory of particle positions
  - `scene_preview.avi` - Video preview

#### Mouse Interaction Mode

- **Click: Add** / **Click: Select** - Toggle between two modes:
  - **Add Mode** (green) - Left-click in the viewport to add new particles
  - **Select Mode** (orange) - Left-click particles to select them for editing; left-click springs to select them
  
- **New Particle: Free** / **New Particle: Fixed** - When in Add mode, toggle whether new particles are free-moving (green) or fixed in place (orange)

#### Rendering

- **Render: Normal** / **Render: Pyramid** - Switch between visualization modes:
  - **Normal** - Direct rendering of particle positions
  - **Pyramid** - Gaussian pyramid rendering for smoother/blurred preview

### Parameter Sliders and Fields

Below the buttons, you'll find adjustable simulation parameters organized as:

| Parameter | Range | Function |
|-----------|-------|----------|
| **time step** | 0.0001-0.01 | Physics timestep size (smaller = more accurate, slower) |
| **gravity g** | -18 to -0.5 | Gravitational acceleration (more negative = stronger downward) |
| **new mass (fixed)** | 0.08-12 | Mass of newly added particles |
| **spring stiffness** | 25-850 | Spring constant for newly added springs |
| **damping stiffness** | 0-12 | Velocity damping (higher = more friction) |
| **max spring dist (fixed)** | 0.02-1 | Maximum distance for automatic spring creation |
| **floor y (fixed)** | -0.95 to 0.2 | Vertical position of the floor |
| **floor bounce** | 0.05-1 | Coefficient of restitution on floor collision |

For each parameter:
- **Slider** - Drag to adjust the value continuously
- **Text Field** - Type exact numerical values
- **Lock Button** (for optimizable parameters) - Lock/unlock this parameter during optimization (red = locked, green = trainable)

### Particle and Spring Editing

When you select a particle or spring:

- **Selected Particle** section:
  - Shows which particle is selected
  - Text field to edit particle mass
  - Lock button to freeze/train mass during optimization
    - The lock button here applies to all particles!
  - **Apply** button to save changes

- **Selected Spring** section:
  - Shows which spring is selected  
  - Text field to edit spring stiffness
  - Lock button to freeze/train stiffness during optimization
    - The lock button here applies to all springs!
  - **Apply** button to save changes

### Optimization

- **Load Target Video** - Load a target video (MP4, AVI, etc.) to match against during optimization
- **Start Optimization** - Begin gradient-based parameter optimization toward the target trajectory

## Repository Structure

```
goo/
├── src/pygoo/                    # Main package source code
│   ├── __init__.py
│   ├── app.py                    # Entry point; main UI loop and CLI handling
│   ├── config.py                 # Simulation configuration (parameters, defaults)
│   ├── state.py                  # SimulationState: particle/spring management
│   ├── optimize.py               # DifferentiableFitter: gradient-based optimization
│   ├── trajectory.py             # Trajectory recording, comparison, I/O
│   ├── physics/                  # Physics engine
│   │   ├── __init__.py
│   │   ├── forces.py             # Force calculations (gravity, springs, damping)
│   │   ├── constraints.py        # Constraint enforcement
│   │   └── stepper.py            # Numerical integration (Stepper class)
│   ├── render/                   # Visualization and rendering
│   │   ├── __init__.py
│   │   ├── diff.py               # Differentiable soft particle rendering
│   │   ├── gaussian.py           # Gaussian pyramid for preview rendering
│   │   └── viewport.py           # World-to-screen coordinate transformation
│   ├── io/                       # I/O and serialization
│   │   ├── __init__.py
│   │   ├── scene.py              # Scene loading/saving (JSON format)
│   │   └── presets.py            # Built-in scene presets
│   └── ui/                       # Interactive UI
│       ├── __init__.py
│       ├── control_panel.py      # Parameter editing and optimization UI
│       └── viewport.py           # UI viewport management
│
├── tests/                        # Test suite
│   ├── test_integration.py       # End-to-end integration tests
│   ├── test_differentiable_fitter.py  # Optimizer tests
│   ├── test_differentiable_pipeline.py # Full pipeline tests
│   ├── test_forces.py            # Physics force tests
│   ├── test_scene_io.py          # Scene I/O tests
│   └── test_trajectory_tools.py  # Trajectory handling tests
│
├── data/                         # Pre-configured demo scenarios
│   ├── floor_bounce/
│   │   ├── orig.json             # Initial scene configuration
|   |   ├── orig_video.avi        # Target video for trajectory matching
│   │   └── orig.traj.npz         # Target trajectory
│   ├── spring_oscillate/
│   │   ├── orig.json
|   |   ├── orig_video.avi
│   │   └── orig.traj.npz
│   └── random/
│       ├── orig.json
|   |   ├── orig_video.avi
│       └── orig.traj.npz
│
├── logs/                         # Optimization output (generated by scripts/run_spring_oscillation.sh)
│   ├── floor_bounce_floor_bounce/
│   │   ├── optim_log.csv         # Loss progression
│   │   └── frames/               # State snapshots per iteration
│   ├── spring_oscillate_spring_k/
│   │   ├── optim_log.csv
│   │   └── frames/
│   └── ...
│
├── scripts/                      # Utility scripts
│   └── run_spring_oscillation.sh # Batch optimization script
│
├── make_timelapse.py             # Generate timelapse from optimization frames
├── plot_optimization.py          # Visualize optimization metrics
├── pyproject.toml                # Package metadata and dependencies
└── README.md                     # This file
```

### Key Files and Directories

**Core Modules** (`src/pygoo/`)
- `app.py` - Entry point; handles CLI, pygame loop, optimization dispatch
- `config.py` - All simulation parameters with sensible defaults
- `state.py` - Manages particles and springs; PyTorch tensors for GPU compatibility
- `optimize.py` - Implements `DifferentiableFitter` for parameter learning

**Physics Engine** (`src/pygoo/physics/`)
- `stepper.py` - Integrates forces over time; supports PyTorch JIT compilation
- `forces.py` - Gravity, spring forces, collision, damping
- `constraints.py` - Hard constraints (fixed particles, collisions)

**Visualization** (`src/pygoo/render/`)
- `diff.py` - Differentiable soft particle rendering (main visualization)
- `gaussian.py` - Gaussian pyramid for smooth preview videos
- `viewport.py` - Coordinate transformation (world ↔ screen space)

**Data I/O** (`src/pygoo/io/`)
- `scene.py` - JSON serialization and loading; parameter randomization
- `presets.py` - Built-in demo configurations

**Demo Data** (`data/`)
- Each subfolder contains `orig.json` (scene), `orig_video.avi` (target video), and `orig.traj.npz` (target trajectory)

**Optimization Logs** (`logs/`)
- Auto-created during optimization runs

## CLI Reference

### Main Application

`python -m pygoo.app [OPTIONS]`

#### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--device` | `{cpu,mps,cuda}` | `cpu` | Compute device (MPS for Apple Silicon) |
| `--headless` | flag | | Run without GUI |
| `--frames` | int | 0 | Stop after N frames (0 = unlimited) |
| `--compile` | flag | | Force PyTorch compilation |
| `--no-compile` | flag | | Disable PyTorch compilation |

#### Scene Setup

| Option | Type | Description |
|--------|------|-------------|
| `--scene-json` | path | Load scene configuration from JSON |
| `--observable-json` | path | Load target trajectory from JSON |
| `--target-video` | path | Load target video for trajectory matching |

#### Optimization

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--optimize-start` | flag | | Enable optimization mode at startup |
| `--iterations` | int | 0 | Stop after N iterations (0 = unlimited) |
| `--optimize-params` | list | all | Comma-separated params to optimize (e.g., `mass,edge_k`) |
| `--freeze-params` | list | | Comma-separated params to freeze/lock |
| `--curriculum-advance-threshold` | float | | Loss threshold for curriculum advancement |

#### Configuration

| Option | Type | Description |
|--------|------|-------------|
| `--set` | list | Override parameters (repeatable). Use dot-path syntax: `particle.2.mass=5.0` or `time_step=0.001` |
| `--log_dir` | path | Directory to save optimization logs and frames |

#### Rendering

| Command | Description |
|---------|-------------|
| `render-video --json JSON --output VIDEO` | Render scene to preview video without GUI |

### Example Commands

```bash
# Load custom scene and optimize
python -m pygoo.app --scene-json my_scene.json --optimize-start

# Modify parameters and run
python -m pygoo.app --scene-json my_scene.json --set time_step=0.003 --optimize-start 

# Modify parameters, set optimization params, and run
python -m pygoo.app --scene-json my_scene.json --set time_step=0.003 --freeze-params mass,spring_k --optimize-params time_step --optimize-start 

# Only load observables from custom scene (no mass, stiffness, etc) and optimize
python -m pygoo.app --observable-json my_scene.json --optimize-start

```

## Additional Scripts

- `make_timelapse.py` - Create timelapse from optimization frames
- `plot_optimization.py` - Visualize optimization metrics
