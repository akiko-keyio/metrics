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


以下是通用指南：
> 本指南为 Codex 生成 Python 软件包时的“行为约束文档”，强调低耦合、高内聚、遵循 SOLID 且避免过度设计，并在整个生命周期内贯彻 YAGNI 与极简主义。([digitalocean.com][1], [geeksforgeeks.org][2])

## 摘要

* **Ruff 一站式静态分析+格式化**：以 `ruff check` + `ruff format` 统一 lint 与代码风格，速度远超 Black/isort/flake8 组合。([docs.astral.sh][3], [github.com][4])
* **uv / pip** 负责环境与依赖；两者均可 **直接驱动 `pyproject.toml`** 工作流，且 uv 与 pip 完全兼容。([docs.astral.sh][5], [github.com][6])
* 设计层面坚持 **SOLID**，同时用 **YAGNI** 限制前期膨胀，保证实现仅含最小必要元素。([digitalocean.com][1], [reddit.com][7])
* 测试首选 **pytest + coverage**；CI 中设置 `--fail-under` 阈值确保质量门槛。([docs.pytest.org][8], [stackoverflow.com][9], [coverage.readthedocs.io][10])

---

## 1. 设计原则

| 原则      | 在包中的落地方式                                                      |
| ------- | ------------------------------------------------------------- |
| **S**RP | 每个模块/函数只承担单一责任，变更原因唯一。([digitalocean.com][1])                 |
| **O**CP | 通过抽象基类或 `typing.Protocol` 开放扩展、封闭修改。                          |
| **L**SP | 子类必须可无痛替换父类；优先组合而非继承。                                         |
| **I**SP | 将大接口拆分为细粒度协议，避免“胖接口”。                                         |
| **D**IP | 高层依赖抽象而非具体，实现依赖注入或 `functools.partial`。([arjancodes.com][11]) |

> **关键提醒：** 若某原则实施后导致模板代码急剧增加且用户价值不变，即视为过度设计，应回退。([geeksforgeeks.org][2])

---

## 2. 编码规范

1. **Ruff 为唯一格式与 Lint 工具**

   * `ruff format .` 保证代码自动对齐与换行。
   * `ruff check .` 开启默认规则；必要时在 `pyproject.toml` 内微调。([docs.astral.sh][3], [docs.astral.sh][12])
2. **类型标注**

   * 所有公共 API 必须完整注解，CI 触发 `mypy --strict`。([packaging.python.org][13])
3. **函数体 ≤ 20 行**，最多一级嵌套；复杂逻辑拆分为私有辅助函数。
4. **异常与日志**

   * 核心层直接抛出异常；边界层捕获并记录。
5. **Docstring**

   * 模块级说明“是什么 & 为什么”；函数级说明“做什么 & 参数/返回”。

---

## 3. 依赖管理

| 工具      | 适用场景         | 关键命令                                                                       |
| ------- | ------------ | -------------------------------------------------------------------------- |
| **uv**  | 高速、可复现的构建与安装 | `uv pip install -r requirements.txt`([docs.astral.sh][5], [astral.sh][14]) |
| **pip** | 经典、稳定、广泛支持   | `python -m pip install .`([reddit.com][15], [packaging.python.org][13])    |

* **最小依赖集合**：能用标准库解决的场景禁止引入第三方库（YAGNI）。([geeksforgeeks.org][2])
* 所有依赖声明于 `project.dependencies`（运行时）与 `project.optional-dependencies.dev`（开发时）段落内。([packaging.python.org][13])

---

## 4. 测试与质量保障

1. **pytest**：覆盖全部公共功能路径；保持测试代码同样简洁。([docs.pytest.org][8], [emimartin.me][16])
3. **property-based 测试**：对纯函数可选用 Hypothesis 提升稳健性。

---

## 5. 自动化与 CI

* **pre-commit**：配置 `ruff`, `mypy`, `pytest` 钩子，在本地即阻止劣质提交。
* **GitHub Actions**：

  ```yaml
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: uv pip install -e .[dev]  # 或 python -m pip ...
    - run: ruff format --check .
    - run: ruff check .
    - run: mypy .
    - run: pytest --cov=src
    - run: coverage report --fail-under=90
  ```

([docs.astral.sh][5], [stackoverflow.com][9], [emimartin.me][16])

---

## 6. 生成策略

> 在编写任何代码前先自问：**“如果现在删掉这段实现，用户功能是否受损？”**——若答案为否，立即移除。

* **先写测试，再写实现**（TDD-light）。
* **小步提交**：每次 commit 仅覆盖单一逻辑变更，信息格式 `feat|fix|docs: <scope>`。
* **显式导出**：所有对外符号必须写入 `__all__`，避免 API 泄漏。
* **文档即代码的一部分**：更新 Public API 时同步更新 README 与 docstring。

---


[1]: https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design?utm_source=chatgpt.com "SOLID Design Principles Explained: Building Better Software ..."
[2]: https://www.geeksforgeeks.org/what-is-yagni-principle-you-arent-gonna-need-it/?utm_source=chatgpt.com "What is YAGNI principle (You Aren't Gonna Need It)? - GeeksforGeeks"
[3]: https://docs.astral.sh/ruff/formatter/?utm_source=chatgpt.com "The Ruff Formatter - Astral Docs"
[4]: https://github.com/astral-sh/ruff?utm_source=chatgpt.com "astral-sh/ruff: An extremely fast Python linter and code formatter ... - GitHub"
[5]: https://docs.astral.sh/uv/pip/compatibility/?utm_source=chatgpt.com "Compatibility with pip | uv - Astral Docs"
[6]: https://github.com/astral-sh/uv?utm_source=chatgpt.com "astral-sh/uv: An extremely fast Python package and project ... - GitHub"
[7]: https://www.reddit.com/r/ExperiencedDevs/comments/11vonwg/yagni_is_a_good_principle_but_many_devs_miss_the/?utm_source=chatgpt.com "\"YAGNI\" is a good principle, but many devs miss the point and conflate it ..."
[8]: https://docs.pytest.org/en/stable/explanation/goodpractices.html?utm_source=chatgpt.com "Good Integration Practices - pytest documentation"
[9]: https://stackoverflow.com/questions/59420123/is-there-a-standard-way-to-fail-pytest-if-test-coverage-falls-under-x?utm_source=chatgpt.com "Is there a standard way to fail pytest if test coverage falls under x%"
[10]: https://coverage.readthedocs.io/?utm_source=chatgpt.com "Coverage.py — Coverage.py 7.9.1 documentation"
[11]: https://arjancodes.com/blog/dependency-inversion-principle-in-python-programming/?utm_source=chatgpt.com "Mastering Dependency Inversion in Python Coding | ArjanCodes"
[12]: https://docs.astral.sh/ruff/?utm_source=chatgpt.com "Ruff - Astral Docs"
[13]: https://packaging.python.org/tutorials/managing-dependencies/?utm_source=chatgpt.com "Managing Application Dependencies - Python Packaging User Guide"
[14]: https://astral.sh/blog/uv?utm_source=chatgpt.com "uv: Python packaging in Rust - Astral"
[15]: https://www.reddit.com/r/Python/comments/1gphzn2/a_completeish_guide_to_dependency_management_in/?utm_source=chatgpt.com "A complete-ish guide to dependency management in Python - Reddit"
[16]: https://emimartin.me/pytest_best_practices?utm_source=chatgpt.com "Pytest best practices - Emiliano Martin"


