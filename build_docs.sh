#!/bin/bash

# MAPLE Documentation Build Script
# This script activates the maple environment and builds the documentation

echo "Building MAPLE Documentation..."
echo "================================"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate maple

# Set environment variable to avoid OpenMP conflicts
export KMP_DUPLICATE_LIB_OK=TRUE

# Change to docs directory and build
cd docs/
make html

echo ""
echo "Documentation built successfully!"
echo "Open docs/_build/html/index.html to view the documentation."

# Optional: Open in browser (uncomment the line below)
# open _build/html/index.html
