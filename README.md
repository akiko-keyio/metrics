# Metrics

A small library for residual analysis based on vectorised `numpy` and
`pandas` operations.  It provides a set of metric functions and a
high-level `analyze` helper to compute them on arrays, matrices or
DataFrames.

## Installation

```bash
uv pip install -e .
```

## Quick start

Compute statistics for two arrays:

```python
from metrics import analyze
import numpy as np

pred = np.array([1.0, 2.0, 3.0])
true = np.array([1.1, 1.9, 3.2])
stats = analyze(pred, true)
print(stats.rms)
```

DataFrame usage with grouping:

```python
import pandas as pd
from metrics.analyzer import DataFrameAnalyzer

df = pd.read_parquet("data/train.parquet")
ana = DataFrameAnalyzer(df, "ztd_nwm", "ztd_gnss_sigma")
report = ana.summary(group=["site", "time:D"], metrics=("rms", "bias", "std"))
print(report.head())
```
