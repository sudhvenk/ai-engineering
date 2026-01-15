#!/bin/bash
# Install all requirements from requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing packages from requirements.txt..."
echo "Using Python: $(which python3)"

# Activate virtual environment if it exists
if [ -d "activate" ]; then
    echo "Activating virtual environment: activate"
    source activate/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment: venv"
    source venv/bin/activate
fi

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "Verifying installation..."
python3 -c "import pandas; import numpy; import matplotlib; import seaborn; print('✓ All packages verified!')"

