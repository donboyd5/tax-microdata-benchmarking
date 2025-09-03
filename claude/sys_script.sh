#!/bin/bash
echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"
echo ""

echo "=== Python Information ==="
python --version
echo "Python path: $(which python)"
echo ""

echo "=== Package Versions ==="
python -c "
import pandas as pd
import numpy as np
import sys

print(f'Python: {sys.version}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CoW mode: {pd.options.mode.copy_on_write}')
"

echo ""
echo "=== Warning Configuration ==="
python -c "
import warnings
print('Warning filters (first 3):')
for i, f in enumerate(warnings.filters[:3]):
    print(f'  {i}: {f}')
"

echo ""
echo "=== Environment Variables ==="
echo "PYTHONWARNINGS: ${PYTHONWARNINGS:-'not set'}"
echo "PYTHONDONTWRITEBYTECODE: ${PYTHONDONTWRITEBYTECODE:-'not set'}"