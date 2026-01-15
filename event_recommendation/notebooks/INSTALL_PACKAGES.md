# Installing Analysis Packages

## Quick Fix

Run these commands in your terminal:

```bash
# Navigate to project root
cd /Users/adipole/github/ai_portfolio/event_recommendation

# Activate the virtual environment
source activate/bin/activate

# Install the analysis packages
pip install pandas numpy matplotlib seaborn

# Verify installation
python -c "import pandas; import numpy; import matplotlib; import seaborn; print('✓ All packages installed!')"
```

## After Installation

1. **Restart your Jupyter kernel:**
   - In Jupyter: Kernel → Restart
   - In VS Code/Cursor: Click the kernel name → Restart Kernel

2. **Verify the correct Python interpreter is selected:**
   - Path should be: `/Users/adipole/github/ai_portfolio/event_recommendation/activate/bin/python3`

## Troubleshooting

If you still see import errors after installing:

1. Check which Python you're using:
   ```bash
   which python
   # Should show: .../activate/bin/python
   ```

2. Verify packages are in the right environment:
   ```bash
   source activate/bin/activate
   pip list | grep -E "(pandas|numpy|matplotlib|seaborn)"
   ```

3. If packages are installed but still not found:
   - Restart your IDE/Jupyter
   - Check that your IDE is using the `activate` environment
   - Try running the notebook cell again after restart

