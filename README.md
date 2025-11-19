# Solarflare SVM Classifier

## Overview
This repository contains the implementation for Assignment 2 of SFWENG‑4AL3 (Applied Machine Learning).  
The project loads the provided datasets, constructs feature sets (FS I to FS IV), trains SVM baseline models, evaluates them using True Skill Score (TSS), and generates summary CSV files and plots.

## Project Components
- **main.py**: Driver script that loads datasets, builds feature sets, runs SVM experiments, saves results, and produces figures.
- **enviroment.yml**: Conda environment specification.
- **results/**: Directory automatically created to store output CSVs and figures.

## Python Version
Tested with Python 3.12.

## Setup Instructions
Create and activate the conda environment:

```
conda env create -f enviroment.yml
conda activate assignment2
```

## Dataset Configuration
Update the dataset paths inside `main.py` before running the script. Example:

```python
dataset_2010_path = "/full/path/to/data-2010-15"
dataset_2020_path = "/full/path/to/data-2020-24"
```

The datasets must contain the structure expected by the assignment specification.

## Running the Project
Execute the main driver script:

```
python main.py
```

## Output
The script automatically generates:
- CSV files containing model performance metrics.
- Bar plots and summary figures saved inside the `results/` directory.

## Notes and Conventions
- Matplotlib is configured to use a noninteractive backend (Agg) so that figures can render on headless systems.
- Code structure follows the guidance provided in the assignment handout.

## AI Tools Disclosure
AI assistants, including ChatGPT and GitHub Copilot, were used for help with:
- Drafting code snippets.
- Understanding scikit-learn API patterns.
- Writing and refining this README.

Final implementation, design decisions, and verification were performed by the author.

## Estimated Carbon Footprint
Assumption:
- Approximate usage: 5 AI prompts.
- Estimated CO2 emissions per prompt: roughly 4.32 g.

Calculation:
- Total CO2 ≈ 5 × 4.32 g = 21.6 g.

## Author
S. Pathmanathan  
Software Engineering, McMaster University
