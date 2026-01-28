## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management and requires specific system-level libraries for 3D processing.

### 1. Install `uv`

If you haven't already, install the `uv` package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

```

### 2. Install System Dependencies

This project relies on C++ headers (like `sparsehash`) that must be installed at the OS level before Python packages can compile.

```bash
# Make the helper script executable
chmod +x install_sys_deps.sh

# Install required apt packages (requires sudo)
./install_sys_deps.sh

```

### 3. Install Python Environment

Initialize the virtual environment and compile all dependencies (including `torchsparse`).

> **Note:** The first run may take **10-20 minutes** to compile custom CUDA kernels. Do not interrupt the process.

```bash
uv sync

```

### 4. Activate & Run

You can run commands directly using `uv run` without manually activating the environment:

```bash
# Run the API server
uv run python api.py

```