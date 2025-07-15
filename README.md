# Metrics

A small library for residual analysis based on vectorised `numpy` and
`pandas` operations.  It provides a set of built-in metric functions and
a high-level `analyze` helper to compute them on arrays, as well as a
`DataFrameAnalyzer` class for grouped statistics on DataFrames.

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
print(stats.rms, stats.bias, stats.std)
```

DataFrame usage with grouping:

```python
import pandas as pd
from metrics import DataFrameAnalyzer

df = pd.read_parquet("data/train.parquet")
analyzer = DataFrameAnalyzer(df, ["ztd_nwm"], "ztd_gnss")
report = analyzer.summary(group=["site", "season"], metrics=("rms", "bias", "std"))
print(report.head())
```

## Custom metrics

New metrics can be registered via `register_metric`:

```python
from metrics import register_metric

@register_metric("mae")
def mean_abs_error(res):
    return np.nanmean(np.abs(res))
```
