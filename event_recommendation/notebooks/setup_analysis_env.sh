#!/bin/bash
# Setup script for analysis notebook dependencies

echo "Setting up analysis environment..."

# Check if virtual environment exists
if [ -d "../activate" ]; then
    echo "Found 'activate' virtual environment"
    source ../activate/bin/activate
elif [ -d "../venv" ]; then
    echo "Found 'venv' virtual environment"
    source ../venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
fi

# Install analysis packages
echo "Installing analysis packages..."
pip install pandas numpy matplotlib seaborn

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import pandas; import numpy; import matplotlib; import seaborn; print('✓ All packages installed successfully')"

echo ""
echo "Setup complete! You can now run the notebook."

