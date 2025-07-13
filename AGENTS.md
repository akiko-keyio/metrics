# **resan — 残差分析开发手册（修订版）**

> **说明**
> 本文是 *唯一* 面向开发者的规范文档。开发者在阅读前不必了解任何先前讨论。本手册阐述 **目标、模块结构、核心概念、公共 API、扩展要点、质量保障**。
> 依照下列要求实现即可；任何偏离都会带来维护风险。

---

## 1. 愿景与设计原则

| 维度         | 目标                                                               |
| ---------- | ---------------------------------------------------------------- |
| **准确**     | 所有统计量严格符合公认定义；置信区间基于 *标准误差*（或自助法可选）。                             |
| **向量化**    | 100 % 利用 `numpy`, `pandas`, `scipy` 的原生向量化运算；**禁止显式 Python 循环**。 |
| **简洁 API** | 一行获得常用指标；复杂场景通过一致的 `summary()` 接口。                               |
| **分组灵活**   | 支持 **任意时间粒度**（小时、日、月、季）与自定义列/多列分组。                               |
| **零并行**    | 不使用多线程/多进程；依赖单核向量化即可满足性能需求。                                      |
| **最小重复**   | 若 `numpy/pandas/scipy` 已提供功能，**直接调用**，不要重复实现。                    |

---

## 2. 依赖与版本

```
numpy      >=1.24
pandas     >=2.0
scipy      >=1.11
scikit-learn  # 仅用于 R²
```

*无* CLI、*无* joblib、*无* typer 等并行/命令行依赖。

---

## 3. 目录结构

```
resan/
├── metrics.py        # 原子指标函数 + 注册器
├── stats.py          # dataclass ResidualStats（immutable）
├── grouping.py       # 时间/普通分组助手
├── analyzer/
│   ├── base.py       # BaseAnalyzer: prepare→group→reduce
│   ├── array.py      # ArrayAnalyzer: 1-D
│   ├── matrix.py     # MatrixAnalyzer: 2-D(time×site)
│   └── dataframe.py  # DataFrameAnalyzer: 通用 DataFrame
└── __init__.py       # 公共入口函数 analyze(...)
```

---

## 4. 核心概念

### 4.1 `ResidualStats`

```python
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass(frozen=True)
class ResidualStats:
    rms:   float
    bias:  float
    std:   float
    r2:    Optional[float] = None
    max:   Optional[float] = None
    min:   Optional[float] = None
    median: Optional[float] = None
    p025:  Optional[float] = None
    p975:  Optional[float] = None
    ks_p:  Optional[float] = None
    ci_low:  Optional[float] = None
    ci_high: Optional[float] = None

    # 便于序列化
    def as_dict(self):
        return asdict(self)
```

* 保持 **不可变**，以防无意修改。
* 字段为 *可选*，方便插件动态插入新指标。

### 4.2 指标插件 (`metrics.py`)

```python
METRICS: dict[str, callable] = {}

def register_metric(name: str):
    def decorator(fn):
        METRICS[name] = fn
        return fn
    return decorator

# 内置指标 (示例)
@register_metric("rms")
def rms(res):           # res: np.ndarray
    return np.sqrt(np.nanmean(res ** 2))

@register_metric("bias")
def bias(res):
    return np.nanmean(res)

@register_metric("std")
def std(res):
    return np.nanstd(res, ddof=0)

@register_metric("r2")
def r2(res, y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)
```

* **原则**：若 `numpy/pandas/scipy` 已有实现（如 `np.nanmean`、`scipy.stats.kstest`），必须直接调用。
* 所有指标须原生向量化；输入/输出均为 `np.ndarray` 或标量。

### 4.3 分组助手 (`grouping.py`)

```python
import pandas as pd

def make_time_grouper(index: pd.DatetimeIndex, rule: str):
    """
    rule 示例：
      'H'   → 按小时
      '3H'  → 每 3 小时
      'M'   → 月
      'Q'   → 季 (北半球, DJF=冬)
    返回 Series[group_key]
    """
    if rule.upper() in {"Q", "QS", "Q-DEC"}:
        # pandas 自带季节, 默认 DJF-MAM-JJA-SON
        return index.to_period("Q")
    else:
        return index.floor(rule)
```

* 对季节分组使用 **pandas 自带“季度”**（`to_period("Q")`），无需自定义。
* 普通列分组直接调用 `df.groupby([...])`。

---

## 5. 分析器层

### 5.1 统一接口

所有分析器都继承 `BaseAnalyzer` 并暴露同一签名：

```python
summary(
    group: str | list[str] | None = "total",
    metrics: list[str] | None = None,
    ci: float | None = 0.95,
    quantiles: tuple[float, float] = (0.025, 0.975),
    na_policy: Literal["warn", "fail", "ignore"] = "warn",
) -> pd.DataFrame
```

* `group="total"` —— 整体统计；
* `group="time:3H"` —— 时间滚动分组，用冒号语法委托 `grouping.make_time_grouper`;
* `group=["site", "time:M"]` —— 多层分组：先以列 `site`，再按月份。

### 5.2 `ArrayAnalyzer`

```python
class ArrayAnalyzer(BaseAnalyzer):
    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray | float = 0.0):
        ...
```

* 1-D 向量场景。
* `y_true` 既可为同维向量，也可为标量（零偏实验）。

### 5.3 `MatrixAnalyzer`

```python
class MatrixAnalyzer(BaseAnalyzer):
    """
    约定:
        行 = 时间 (pd.DatetimeIndex)
        列 = 站点
        values = residuals (预测-真值)
    """
    def __init__(self, mat: pd.DataFrame): ...
```

* 支持 `group="site"` / `"time:Q"` / `["site","time:H"]` 等。
* 向量化聚合示例（季节 × 站点）：

```python
stats = mat.stack().groupby([mat.columns.get_level_values(0), mat.index.to_period("Q")]).agg(metric_funcs)
```

### 5.4 `DataFrameAnalyzer`

```python
class DataFrameAnalyzer(BaseAnalyzer):
    def __init__(self, df: pd.DataFrame, pred_col: str, true_col: str):
        ...
```

* 传统机器学习对比场景；残差 = `df[pred_col] - df[true_col]`。
* 时间分组自动检测 `df.index` 若为 `DatetimeIndex`，否则要求显式指定时间列。

---

## 6. 顶层快捷函数

在 `resan.__init__` 中暴露：

```python
from resan.analyzer import ArrayAnalyzer, DataFrameAnalyzer, MatrixAnalyzer

def analyze(
    y_pred,
    y_true=None,
    *,
    group="total",
    metrics=None,
    ci=0.95,
    quantiles=(0.025, 0.975),
    na_policy="warn",
):
    """
    调度到合适的 Analyzer；一行调用。
    """
    # 自动类型判定 -> 实例化相应 Analyzer -> summary(...)
```

* 判定逻辑：

  * `pandas.DataFrame & pred_col in kwargs` → `DataFrameAnalyzer`
  * `pandas.DataFrame | pd.Series` (含 datetime index / multi-cols) → `MatrixAnalyzer`
  * `np.ndarray` → `ArrayAnalyzer`

---

## 7. 扩展方法

### 7.1 新指标

```python
from resan.metrics import register_metric
import numpy as np

@register_metric("mae")
def mae(res: np.ndarray):
    return np.nanmean(np.abs(res))
```

### 7.2 自定义时间粒度

```python
# 每 6 小时滚动
dfa.summary(group="time:6H")

# 每 10 天
dfa.summary(group="time:10D")
```

底层调用 `pandas.DatetimeIndex.floor(rule)`，无需自行实现。

---

## 8. 质量保障

| 阶段       | 要求                                                 |
| -------- | -------------------------------------------------- |
| **单元测试** | 每个指标函数 <1e-12 误差；分组路径（多列 + 时间）全覆盖；缺失值场景。           |
| **静态检查** | `ruff` / `flake8` + `mypy --strict`.               |
| **持续集成** | Linux/macOS/Windows Python 3.9–3.12。               |
| **文档**   | docstring + sphinx-autosummary；示例 notebook 演示常见用法。 |

---

## 9. 开发里程碑

1. **MVP**

   * `metrics.py`（rms, bias, std, r2, ks, ci）
   * `ArrayAnalyzer`, `analyze` 快捷函数
2. **DataFrame & Matrix 支持**

   * 分组助手实现
   * `DataFrameAnalyzer`, `MatrixAnalyzer`
3. **高级分组**

   * `time:<rule>` 语法
   * 多索引聚合
4. **文档 & 单测完善**

   * sphinx / GitHub Pages
   * pytest 覆盖率 > 90 %


