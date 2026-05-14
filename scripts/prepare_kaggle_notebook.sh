#!/bin/bash
# Helper commands to run in a Kaggle notebook cell (prefix with !)

# Show Python / device
python -c "import sys, torch; print(sys.version); print('cuda', getattr(torch, 'cuda', None) and torch.cuda.is_available())"

# Install optional packages if internet access is enabled (use sparingly)
pip install -q sentence-transformers hnswlib transformers

# Run smoke test
python run_experiment.py --mode smoke

# Show outputs
ls -la outputs/predictions
cat outputs/metrics/metrics.json
