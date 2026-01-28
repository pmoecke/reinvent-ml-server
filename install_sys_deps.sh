#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Updating apt repositories..."
sudo apt-get update -y

echo "Installing Compilation Tools..."
# Required for building any C++ extension (like torchsparse)
sudo apt-get install -y build-essential cmake git ninja-build python3-dev

echo "Installing Google SparseHash (CRITICAL for torchsparse)..."
# This is the specific header-only library that torchsparse looks for
sudo apt-get install -y libsparsehash-dev

echo "Installing Project Aria / 3D Helper Libraries..."
# These are often required by Facebook's 3D tools (projectaria, efm3d) if wheels fallback to source
sudo apt-get install -y \
    liblz4-dev \
    libzstd-dev \
    libxxhash-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libgflags-dev

echo "Installing Graphics/Rendering Backend Support..."
# Necessary if your code ever touches headless rendering (Open3D, Plotly, etc.)
sudo apt-get install -y libgl1 libglib2.0-0

echo "✅ System dependencies installed successfully."