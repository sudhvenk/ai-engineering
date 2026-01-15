# Notebooks

## Reviews Analysis Notebook

The `reviews_analysis.ipynb` notebook analyzes review data from the CSV file.

### Setup

Before running the notebook, install the required dependencies:

**Option 1: Using the setup script (recommended)**
```bash
cd notebooks
./setup_analysis_env.sh
```

**Option 2: Manual installation**
```bash
# Activate your virtual environment (if using one)
source activate/bin/activate  # or: source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt

# Or install analysis packages individually
pip install pandas numpy matplotlib seaborn
```

**Important:** After installing, make sure your IDE/Jupyter is using the same Python interpreter where you installed the packages. Restart your Jupyter kernel if needed.

### Running the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `reviews_analysis.ipynb`

3. Run all cells to perform the analysis

### Output

Analysis results are exported to `notebooks/analysis_output/`:
- `reviews_processed.csv` - Processed data with extracted fields
- `event_type_summary.csv` - Event type statistics
- `location_summary.csv` - Location statistics

