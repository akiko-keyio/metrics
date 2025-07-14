from __future__ import annotations

from collections.abc import Iterable
from inspect import signature
from typing import List, cast

import numpy as np
import pandas as pd

from ..metrics import METRICS                # 见下方说明
from .base import BaseAnalyzer


class DataFrameAnalyzer(BaseAnalyzer):
    """
    Analyzer operating on a ``pandas.DataFrame`` with *pandas* / *NumPy*
    dual-engine support.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.  若需按 ``year / season / month / hour`` 分组，须包含
        ``time``（datetime64[ns]）列。
    pred_col : str | list[str]
        预测列名。
    true_col : str
        真值列名。
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, df: pd.DataFrame, pred_col: str | list[str], true_col: str) -> None:
        self.df = df
        self.true_col = true_col
        self.pred_cols: List[str] = [pred_col] if isinstance(pred_col, str) else list(pred_col)

        # backward-compat
        self.pred_col = pred_col

        # init parent with第1列（占位，用不到）
        res_0 = df[self.pred_cols[0]] - df[true_col]
        super().__init__(res_0.to_numpy(),
                         df[true_col].to_numpy(),
                         df[self.pred_cols[0]].to_numpy())

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def summary(
        self,
        group: str | list[str] | None = "total",
        metrics: Iterable[str] | None = None,
        *,
        engine: str = "auto",      # "auto" | "pandas" | "numpy"
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        group   : 分组键，可含 ``year/season/month/hour``。
        metrics : 额外指标名，须已在 ``METRICS`` 注册。
        engine  : "numpy" 时走纯 NumPy 路径；"pandas" 固定用 pandas；
                  "auto" 若所有指标仅依赖 ``res``/``y_true``/``y_pred``
                  且已实现向量化，则自动选用 NumPy。

        Returns
        -------
        pd.DataFrame
        """
        base_metrics = {"rms", "bias", "std"}
        requested = set(base_metrics).union(metrics or [])

        if engine not in {"auto", "pandas", "numpy"}:
            raise ValueError("engine must be 'auto', 'pandas', or 'numpy'.")

        if engine == "numpy" or (
            engine == "auto"
            and requested <= {"rms", "bias", "std", "r2"}     # 目前 NumPy 支持到 r2
        ):
            return self._summary_numpy(group, requested)

        return self._summary_pandas(group, requested)

    # ------------------------------------------------------------------ #
    # pandas 实现（保持不变，略）
    # ------------------------------------------------------------------ #
    def _prepare_long(self, group_list: List[str]) -> pd.DataFrame:
        # …同上一版，无改动…
        df = self.df.copy()
        temporal = {"year", "season", "month", "hour"}
        need_time = [g for g in group_list if g in temporal]
        if need_time:
            if "time" not in df:           # 必须已有
                raise KeyError("Temporal grouping requires a 'time' column.")
            idx = df["time"].dt
            if "year" in need_time:   df["year"] = idx.year
            if "month" in need_time:  df["month"] = idx.month
            if "hour" in need_time:   df["hour"] = idx.hour
            if "season" in need_time:
                df["season"] = idx.month.map(
                    lambda m: ("Winter", "Spring", "Summer", "Autumn")[(m - 1) // 3]
                )

        id_vars = [self.true_col] + group_list
        df_long = df[id_vars + self.pred_cols].melt(
            id_vars=id_vars,
            value_vars=self.pred_cols,
            var_name="var",
            value_name="pred",
        )
        df_long["res"] = df_long["pred"] - df_long[self.true_col]
        return df_long

    def _summary_pandas(self, group, metrics: set[str]) -> pd.DataFrame:
        # …同上一版，无改动…
        if group in (None, "total"):
            group_list: List[str] = []
        else:
            group_list = (
                [group] if isinstance(group, str) else list(cast(list[str], group))
            )

        df_long = self._prepare_long(group_list)
        funcs = [METRICS[m] for m in metrics]
        keys = group_list + ["var"] if group_list else ["var"]

        out = df_long.groupby(keys, observed=True)["res"].agg(funcs)
        out = out.rename(columns={f.__name__: m for f, m in zip(funcs, metrics)}).reset_index()
        return out[keys + sorted(metrics)]

    # ------------------------------------------------------------------ #
    # NumPy 超极速实现（支持 rms / bias / std / r2）
    # ------------------------------------------------------------------ #
    def _summary_numpy(self, group, metrics: set[str]) -> pd.DataFrame:
        # ---- 校验 -----------------------------------------------------
        supported = {"rms", "bias", "std", "r2"}
        if set(metrics) - supported:
            raise NotImplementedError(f"NumPy engine currently supports {supported}")

        # ---- 解析分组列 ----------------------------------------------
        if group in (None, "total"):
            group_list: List[str] = []
        else:
            group_list = (
                [group] if isinstance(group, str) else list(cast(list[str], group))
            )

        # ---- 衍生时间列（如需） --------------------------------------
        temporal = {"year", "season", "month", "hour"}
        need_time = [g for g in group_list if g in temporal]
        if need_time:
            if "time" not in self.df:
                raise KeyError("Temporal keys require a 'time' column.")
            idx = self.df["time"].dt
            if "year" in need_time and "year" not in self.df:
                self.df["year"] = idx.year
            if "month" in need_time and "month" not in self.df:
                self.df["month"] = idx.month
            if "hour" in need_time and "hour" not in self.df:
                self.df["hour"] = idx.hour
            if "season" in need_time and "season" not in self.df:
                self.df["season"] = idx.month.map(
                    lambda m: ("Winter", "Spring", "Summer", "Autumn")[(m - 1) // 3]
                )

        # ---- 编码分组 → base_gid -------------------------------------
        if not group_list:
            base_gid = np.zeros(len(self.df), dtype=np.int64)
            levels, shapes = [], []
        else:
            codes, levels, shapes = [], [], []
            for col in group_list:
                code, uniq = pd.factorize(self.df[col].to_numpy(), sort=True)
                codes.append(code.astype(np.int64, copy=False))
                levels.append(uniq)
                shapes.append(len(uniq))
            base_gid = np.ravel_multi_index(codes, shapes)

        # 共用真值
        y_true_all = self.df[self.true_col].to_numpy()

        frames: list[pd.DataFrame] = []

        # ---- 遍历每个预测列 ------------------------------------------
        for pred in self.pred_cols:
            y_pred_all = self.df[pred].to_numpy()
            res_all = y_pred_all - y_true_all

            # mask 掉 NaN（残差或真值 / 预测为 NaN 时）
            mask = ~np.isnan(res_all) & ~np.isnan(y_true_all) & ~np.isnan(y_pred_all)
            gid = base_gid[mask]
            res = res_all[mask]
            y_true = y_true_all[mask]

            # 最小 length 防御
            size = gid.max() + 1 if gid.size else 0

            # ---- 聚合量：一次 bincount 就够 -------------------------
            n         = np.bincount(gid, minlength=size)
            sum_res   = np.bincount(gid, weights=res, minlength=size)
            sumsq_res = np.bincount(gid, weights=res * res, minlength=size)

            # r2 需要额外聚合
            if "r2" in metrics:
                sum_y   = np.bincount(gid, weights=y_true, minlength=size)
                sumsq_y = np.bincount(gid, weights=y_true * y_true, minlength=size)

            # ---- 公式计算 ------------------------------------------
            with np.errstate(divide="ignore", invalid="ignore"):
                bias = sum_res / n
                rms  = np.sqrt(sumsq_res / n)
                std  = np.sqrt(np.maximum(sumsq_res / n - bias**2, 0.0))

            cols = {
                "bias": bias,
                "rms":  rms,
                "std":  std,
            }

            if "r2" in metrics:
                mean_y = sum_y / n
                ss_tot = sumsq_y - n * mean_y**2
                ss_res = sumsq_res
                r2 = 1.0 - ss_res / ss_tot
                r2[(ss_tot == 0) | (n == 0)] = np.nan
                cols["r2"] = r2

            # ---- 组键 DataFrame ------------------------------------
            if levels:       # 有显式分组
                mindex = pd.MultiIndex.from_product(levels, names=group_list)
                key_df = mindex.to_frame(index=False)
            else:
                key_df = pd.DataFrame(index=np.arange(len(n)))

            key_df["var"] = pred
            for k in sorted(metrics):
                key_df[k] = cols[k]

            frames.append(key_df)

        out = pd.concat(frames, ignore_index=True)

        if group in (None, "total") and not group_list:
            return out[["var"] + sorted(metrics)]

        return out[group_list + ["var"] + sorted(metrics)]
