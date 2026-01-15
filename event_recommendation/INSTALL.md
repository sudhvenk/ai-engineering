# Installation Instructions

## Quick Install

Run the installation script:

```bash
./install_requirements.sh
```

Or manually install:

```bash
# Activate your virtual environment
source activate/bin/activate

# Install all requirements
pip install -r requirements.txt
```

## Running the Application

**Important**: Always use the virtual environment's Python, not the system Python.

### Option 1: Use the run script (Recommended)
```bash
./run.sh
```

### Option 2: Activate virtual environment first
```bash
source activate/bin/activate
python main.py
```

### Option 3: Use virtual environment Python directly
```bash
./activate/bin/python main.py
```

## Verify Installation

After installation, verify packages are installed:

```bash
# Using virtual environment Python
./activate/bin/python -c "import langchain_chroma; import dotenv; print('✓ All packages installed!')"
```

## For Jupyter Notebooks

After installing packages:
1. Restart your Jupyter kernel
2. Ensure your IDE is using the correct Python interpreter: `activate/bin/python3`

