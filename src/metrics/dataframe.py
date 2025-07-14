from __future__ import annotations

from collections.abc import Iterable
from inspect import signature
from typing import List, cast, Tuple, Dict, Any

import numpy as np
import pandas as pd

from metrics.metrics_factory import METRICS


class DataFrameAnalyzer():
    """
    Analyzer for computing residual statistics on a pandas.DataFrame.

    Offers multiple computation engines for performance and flexibility.
    The standard deviation ('std') metric respects the `ddof` parameter,
    which defaults to 1 for unbiased estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Source data. Must contain a 'time' column of datetime-like objects
        if temporal grouping (year, season, month, hour) is used.
    pred_col : str | list[str]
        Name(s) of the prediction column(s).
    true_col : str
        Name of the ground-truth column.
    """

    def __init__(self, df: pd.DataFrame, pred_col: str | list[str], true_col: str) -> None:
        self.df = df
        self.true_col = true_col
        self.pred_cols: List[str] = [pred_col] if isinstance(pred_col, str) else list(pred_col)
        self.pred_col = pred_col

    def summary(
            self,
            group: str | list[str] | None = "total",
            metrics: Iterable[str] | None = None,
            *,
            engine: str = "auto",
            ddof: int = 1,
    ) -> pd.DataFrame:
        """
        Compute error metrics, optionally grouped by specified columns.

        (文档与之前版本一致)
        """
        if not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer.")

        base_metrics = {"rms", "bias", "std"}
        requested = set(base_metrics).union(metrics or [])

        if engine not in {"auto", "pandas", "numpy"}:
            raise ValueError("engine must be 'auto', 'pandas', or 'numpy'.")

        bincount_supported = {"rms", "bias", "std", "r2"}
        use_bincount = requested.issubset(bincount_supported)

        if engine == "numpy" or (engine == "auto"):
            if use_bincount:
                return self._summary_numpy_bincount(group, requested, ddof=ddof)
            else:
                return self._summary_numpy_generic(group, requested, ddof=ddof)

        return self._summary_pandas(group, requested, ddof=ddof)

    # ------------------------------------------------------------------ #
    # Private Helper Methods for Pre-processing
    # ------------------------------------------------------------------ #
    def _prepare_temporal_cols(self, df: pd.DataFrame, group_list: List[str]) -> pd.DataFrame:
        """Adds temporal columns (year, season, etc.) to the DataFrame if needed."""
        temporal_keys = {"year", "season", "month", "hour"}
        needed_keys = [g for g in group_list if g in temporal_keys]

        if not needed_keys:
            return df

        if "time" not in df:
            raise KeyError("Temporal grouping requires a 'time' column.")

        df_out = df if df is not self.df else df.copy()

        time_idx = pd.to_datetime(df_out["time"]).dt

        if "year" in needed_keys and "year" not in df_out.columns:
            df_out["year"] = time_idx.year
        if "month" in needed_keys and "month" not in df_out.columns:
            df_out["month"] = time_idx.month  # <--- 直接使用 time_idx.month
        if "hour" in needed_keys and "hour" not in df_out.columns:
            df_out["hour"] = time_idx.hour
        if "season" in needed_keys and "season" not in df_out.columns:
            # 使用月份直接映射，代码更清晰
            season_map = {
                1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn',
                11: 'Autumn', 12: 'Winter'
            }
            # 直接使用 month 列进行映射
            df_out["season"] = time_idx.month.map(season_map)

        return df_out

    def _prepare_numpy_groups(self, group: str | list[str] | None) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
        """
        Shared preparation logic for both NumPy engines.
        Parses group keys, prepares temporal columns, and encodes groups.
        """
        if group in (None, "total"):
            group_list: List[str] = []
        else:
            group_list = [group] if isinstance(group, str) else list(cast(list[str], group))

        # 准备带时间列的 DataFrame
        df_work = self._prepare_temporal_cols(self.df, group_list)

        group_info = {"group_list": group_list}

        if group_list:
            codes_list = []
            levels_list = []
            shapes_list = []
            for col in group_list:
                codes, levels = pd.factorize(df_work[col], sort=True)
                codes_list.append(codes.astype(np.int64, copy=False))
                levels_list.append(levels)
                shapes_list.append(len(levels))

            group_info["base_gid"] = np.ravel_multi_index(codes_list, shapes_list)
            group_info["levels"] = levels_list
            group_info["shapes"] = shapes_list
        else:
            group_info["base_gid"] = np.zeros(len(df_work), dtype=np.int64)
            group_info["levels"] = []
            group_info["shapes"] = []

        return group_list, df_work, group_info

    # ------------------------------------------------------------------ #
    # Pandas Fallback Implementation
    # ------------------------------------------------------------------ #
    def _summary_pandas(self, group, metrics: set[str], ddof: int) -> pd.DataFrame:
        if group in (None, "total"):
            group_list: List[str] = []
        else:
            group_list = [group] if isinstance(group, str) else list(cast(list[str], group))

        df = self._prepare_temporal_cols(self.df, group_list)
        id_vars = [self.true_col] + group_list
        df_long = df.reindex(columns=id_vars + self.pred_cols).melt(
            id_vars=id_vars, value_vars=self.pred_cols, var_name="var", value_name="pred",
        )
        df_long["res"] = df_long["pred"] - df_long[self.true_col]

        def agg_fn(df_group):
            results = {}
            for m in metrics:
                fn = METRICS[m]
                sig = signature(fn)
                kwargs = {}
                if 'res' in sig.parameters: kwargs['res'] = df_group['res']
                if 'y_true' in sig.parameters: kwargs['y_true'] = df_group[self.true_col]
                if 'y_pred' in sig.parameters: kwargs['y_pred'] = df_group['pred']
                if 'ddof' in sig.parameters: kwargs['ddof'] = ddof
                results[m] = fn(**kwargs)
            return pd.Series(results)

        keys = group_list + ["var"] if group_list else ["var"]
        out = (df_long.groupby(keys, observed=True)
               .apply(agg_fn, include_groups=False)
               .reset_index())
        return out[keys + sorted(list(metrics))]

    # ------------------------------------------------------------------ #
    # NumPy Engines
    # ------------------------------------------------------------------ #
    def _summary_numpy_bincount(self, group, metrics: set[str], ddof: int) -> pd.DataFrame:
        # ---- 1. 使用辅助方法准备分组信息 ----
        group_list, df_work, group_info = self._prepare_numpy_groups(group)
        base_gid = group_info["base_gid"]
        base_levels = group_info["levels"]
        base_shapes = group_info["shapes"]

        # ---- 2. 转换数据为NumPy数组 (此部分特定于 bincount) ----
        y_true_all = df_work[self.true_col].to_numpy(dtype=float)
        n_pred = len(self.pred_cols)
        y_pred_stacked = np.stack([df_work[col].to_numpy(dtype=float) for col in self.pred_cols], axis=0)
        res_stacked = y_pred_stacked - y_true_all

        # ---- 3. 编码 'var' 维度并计算完整 gid ----
        var_codes = np.arange(n_pred, dtype=np.int64)
        var_levels = np.array(self.pred_cols)
        full_shapes = tuple(base_shapes + [n_pred])
        full_levels = base_levels + [var_levels]
        full_group_names = group_list + ["var"]
        gid_full = (base_gid * n_pred + var_codes[:, None]).ravel()

        # ---- 4. 准备 bincount 的 weights 并处理 NaN ----
        res_flat, y_pred_flat = res_stacked.ravel(), y_pred_stacked.ravel()
        y_true_tiled = np.tile(y_true_all, n_pred)
        mask = ~np.isnan(res_flat) & ~np.isnan(y_true_tiled) & ~np.isnan(y_pred_flat)
        gid_masked = gid_full[mask]
        final_size = np.prod(full_shapes, dtype=np.int64) if full_shapes else 1

        # ---- 5. 一次 bincount 完成所有计算 ----
        # (此逻辑块与之前版本相同, 已高度优化)
        n = np.bincount(gid_masked, minlength=final_size)
        sum_res = np.bincount(gid_masked, weights=res_flat[mask], minlength=final_size)
        sumsq_res = np.bincount(gid_masked, weights=res_flat[mask] ** 2, minlength=final_size)
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = sum_res / n
            rms = np.sqrt(sumsq_res / n)
            pop_var = np.maximum(sumsq_res / n - bias ** 2, 0.0)
            if ddof > 0:
                correction_factor = np.divide(n, n - ddof, where=(n > ddof), out=np.full_like(n, np.nan, dtype=float))
                std = np.sqrt(pop_var * correction_factor)
            else:
                std = np.sqrt(pop_var)
        cols = {"bias": bias, "rms": rms, "std": std}
        if "r2" in metrics:
            sum_y = np.bincount(gid_masked, weights=y_true_tiled[mask], minlength=final_size)
            sumsq_y = np.bincount(gid_masked, weights=y_true_tiled[mask] ** 2, minlength=final_size)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_y = sum_y / n
                ss_tot = sumsq_y - n * mean_y ** 2
                r2 = 1.0 - sumsq_res / ss_tot
                r2[(ss_tot == 0) | (n == 0)] = np.nan
            cols["r2"] = r2

        # ---- 6. 构建结果DataFrame ----
        if not full_group_names:
            out = pd.DataFrame({'var': self.pred_cols})
        else:
            mindex = pd.MultiIndex.from_product(full_levels, names=full_group_names)
            out = mindex.to_frame(index=False)
        for k in sorted(metrics):
            out[k] = cols[k]
        out = out.dropna(subset=list(metrics), how='all').reset_index(drop=True)
        return out[full_group_names + sorted(list(metrics))]

    def _summary_numpy_generic(self, group, metrics: set[str], ddof: int) -> pd.DataFrame:
        # ---- 1. 使用辅助方法准备分组信息 ----
        group_list, df_work, group_info = self._prepare_numpy_groups(group)
        base_gid = group_info["base_gid"]
        group_levels = group_info["levels"]
        group_shapes = group_info["shapes"]

        # ---- 2. 准备数据 (此部分特定于 generic) ----
        y_true_all = df_work[self.true_col].to_numpy(dtype=float)
        all_pred_arrays = {pc: df_work[pc].to_numpy(dtype=float) for pc in self.pred_cols}

        # ---- 3. 循环每个预测列并计算指标 ----
        results_frames = []
        for pred_col, y_pred_all in all_pred_arrays.items():
            res_all = y_pred_all - y_true_all
            mask = ~np.isnan(res_all) & ~np.isnan(y_true_all) & ~np.isnan(y_pred_all)
            gid, res, y_true, y_pred = base_gid[mask], res_all[mask], y_true_all[mask], y_pred_all[mask]
            if gid.size == 0: continue

            sorter = np.argsort(gid, kind='mergesort')
            sorted_gid = gid[sorter]
            unique_groups, group_starts = np.unique(sorted_gid, return_index=True)

            metric_results = {}
            for metric_name in sorted(list(metrics)):
                fn = METRICS[metric_name]
                sig = signature(fn)
                kwargs_template = {}
                if 'res' in sig.parameters: kwargs_template['res'] = np.split(res[sorter], group_starts[1:])
                if 'y_true' in sig.parameters: kwargs_template['y_true'] = np.split(y_true[sorter], group_starts[1:])
                if 'y_pred' in sig.parameters: kwargs_template['y_pred'] = np.split(y_pred[sorter], group_starts[1:])

                metric_values = []
                for i in range(len(unique_groups)):
                    call_kwargs = {k: v[i] for k, v in kwargs_template.items()}
                    if 'ddof' in sig.parameters: call_kwargs['ddof'] = ddof
                    metric_values.append(fn(**call_kwargs))
                metric_results[metric_name] = np.array(metric_values)

            # ---- 4. 构建当前预测列的结果DataFrame ----
            if group_list:
                group_codes_unraveled = np.unravel_index(unique_groups, group_shapes)
                group_df_data = {group_list[i]: group_levels[i][group_codes_unraveled[i]] for i in
                                 range(len(group_list))}
            else:
                group_df_data = {}
            res_df = pd.DataFrame(group_df_data)
            res_df["var"] = pred_col
            for metric_name, values in metric_results.items():
                res_df[metric_name] = values
            results_frames.append(res_df)

        if not results_frames: return pd.DataFrame()
        out = pd.concat(results_frames, ignore_index=True)
        return out[group_list + ["var"] + sorted(list(metrics))]