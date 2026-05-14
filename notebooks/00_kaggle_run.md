# Kaggle Run Instructions (Notebook Replacement)

This document provides cells and commands you can paste into a Kaggle notebook. It acts as a lightweight replacement for a full .ipynb and contains exactly the steps to run the smoke/subset experiments on Kaggle.

## 1) Environment and imports

```python
# Check Python and device
import sys
import platform
print('Python', sys.version)
try:
    import torch
    print('Torch available, CUDA:', torch.cuda.is_available())
except Exception:
    print('Torch not available')

# Minimal imports to ensure project is visible
import os
print('CWD', os.getcwd())
```

## 2) Install minimal packages (optional)

Only run installs if Kaggle environment is missing a dependency. Prefer enabling internet on your Kaggle notebook or pre-build a dataset with artifacts.

```bash
# In a Kaggle notebook cell (bash):
# pip install -r /kaggle/working/requirements.txt
# or install specific packages if needed
pip install sentence-transformers hnswlib transformers
```

## 3) Set dataset paths and config

In Kaggle, datasets are mounted under `/kaggle/input/<dataset-name>/...`.
Open `config.yaml` in the repo and set the paths accordingly, for example:

```yaml
data:
  dataset_name: your_dataset_name
  dataset_paths:
    nq: /kaggle/input/your-dataset/nq.jsonl
    triviaqa: /kaggle/input/your-dataset/triviaqa.jsonl
    hotpotqa: /kaggle/input/your-dataset/hotpot.jsonl
```

You can edit `config.yaml` in a notebook cell:

```python
from pathlib import Path
cfg = Path('config.yaml').read_text()
print(cfg)
# Optionally write a new config with dataset paths
```

## 4) Run smoke test

```bash
python run_experiment.py --mode smoke
```

The script will produce outputs under `outputs/predictions` and `outputs/metrics`.

## 5) Inspect outputs

```python
import pandas as pd
print('Predictions:')
print(pd.read_csv('outputs/predictions/predictions.csv').head())
import json
print('Metrics:')
print(json.load(open('outputs/metrics/metrics.json')))
```

## 6) Notes on packaging Kaggle dataset

- Put all corpus files and any required model artifacts under a single folder.
- Zip and upload as a Kaggle Dataset following Kaggle's dataset UI. The repo `config.yaml` should point to files under `/kaggle/input/<your-dataset>/*`.
- Keep generation disabled for initial runs to avoid heavy model installs.

---

If you prefer, I can also generate a proper `.ipynb` file; tell me if you want that and I will create it next.