# -*- coding: utf-8 -*-
""" 
自定义特征工程预处理组件
实现适用于 sklearn.Pipeline 的转换器：
1. Winsorizer: 按列分位数截断极端值，避免异常点干扰。
2. CrossSectionZScore: 按列执行 Z-score 标准化，支持在 `.fit` 阶段计算训练集均值/方差，在 `.transform` 阶段复用。

后续可扩展更多横截面相关变换（如行业中性化等）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union

class Winsorizer(BaseEstimator, TransformerMixin):
    """按列分位数截断极端值。

    Parameters
    ----------
    lower_quantile : float, default 0.01
        下侧分位数 (0-1)。所有小于该分位数的值将被截断到该阈值。
    upper_quantile : float, default 0.99
        上侧分位数 (0-1)。所有大于该分位数的值将被截断到该阈值。
    inclusive : bool, default True
        是否包含等于阈值的样本。
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, inclusive: bool = True):
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("lower_quantile 必须 < upper_quantile 且位于 0-1 之间")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.inclusive = inclusive
        # 在 fit 阶段保存阈值
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):  # noqa: N803
        """计算每列的截断阈值。"""
        X_df = self._to_dataframe(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):  # noqa: N803
        """应用截断阈值。"""
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Winsorizer 尚未 fit")
        X_df = self._to_dataframe(X)
        if self.inclusive:
            X_clipped = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        else:
            # 排除等于阈值的情况
            X_clipped = X_df.mask(X_df < self.lower_bounds_, self.lower_bounds_, axis=1)
            X_clipped = X_clipped.mask(X_clipped > self.upper_bounds_, self.upper_bounds_, axis=1)
        return X_clipped.values if isinstance(X, np.ndarray) else X_clipped

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError("Winsorizer 仅支持 DataFrame 或 ndarray 输入")

class CrossSectionZScore(BaseEstimator, TransformerMixin):
    """按列 Z-score 标准化 ( (x-mean)/std )。"""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):  # noqa: N803
        X_df = self._to_dataframe(X)
        self.mean_ = X_df.mean()
        self.std_ = X_df.std().replace(0, self.eps)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):  # noqa: N803
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CrossSectionZScore 尚未 fit")
        X_df = self._to_dataframe(X)
        X_scaled = (X_df - self.mean_) / self.std_
        return X_scaled.values if isinstance(X, np.ndarray) else X_scaled

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError("CrossSectionZScore 仅支持 DataFrame 或 ndarray 输入")