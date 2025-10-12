# -*- coding: utf-8 -*-
"""
Winsorizer 预处理器
用于剪尾处理，兼容 sklearn Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizer 剪尾处理器
    
    将极端值剪切到指定分位数，减少异常值对模型的影响
    
    Parameters
    ----------
    quantile_range : tuple, default=(0.01, 0.99)
        剪尾的分位数范围 (lower, upper)
    
    clip_method : str, default='quantile'
        剪尾方法:
        - 'quantile': 基于分位数
        - 'std': 基于标准差 (mean ± n*std)
    
    n_std : float, default=3.0
        当 clip_method='std' 时，使用的标准差倍数
    
    copy : bool, default=True
        是否复制数据
    
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> 
    >>> pipe = Pipeline([
    ...     ('winsorizer', Winsorizer(quantile_range=(0.01, 0.99))),
    ...     ('scaler', StandardScaler())
    ... ])
    >>> X_transformed = pipe.fit_transform(X)
    """
    
    def __init__(self, 
                 quantile_range: tuple = (0.01, 0.99),
                 clip_method: str = 'quantile',
                 n_std: float = 3.0,
                 copy: bool = True):
        """初始化 Winsorizer"""
        self.quantile_range = quantile_range
        self.clip_method = clip_method
        self.n_std = n_std
        self.copy = copy
        
        # 存储每列的剪尾界限
        self.clip_values_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        """
        计算剪尾界限
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        y : Ignored
            不使用，保持接口一致性
        
        Returns
        -------
        self : object
            返回自身
        """
        X = self._validate_data(X)
        
        self.n_features_in_ = X.shape[1]
        
        # 存储特征名（如果输入是DataFrame）
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        
        # 计算每列的剪尾界限
        self.clip_values_ = []
        
        for i in range(X.shape[1]):
            col_data = X[:, i] if not isinstance(X, pd.DataFrame) else X.iloc[:, i]
            
            # 过滤NaN
            valid_data = col_data[~np.isnan(col_data)] if hasattr(col_data, '__iter__') else col_data
            
            if len(valid_data) == 0:
                # 如果全是NaN，使用默认值
                self.clip_values_.append((-np.inf, np.inf))
                continue
            
            if self.clip_method == 'quantile':
                # 基于分位数
                lower = np.quantile(valid_data, self.quantile_range[0])
                upper = np.quantile(valid_data, self.quantile_range[1])
            elif self.clip_method == 'std':
                # 基于标准差
                mean = np.mean(valid_data)
                std = np.std(valid_data)
                lower = mean - self.n_std * std
                upper = mean + self.n_std * std
            else:
                raise ValueError(f"Unknown clip_method: {self.clip_method}")
            
            self.clip_values_.append((lower, upper))
        
        return self
    
    def transform(self, X):
        """
        应用剪尾
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            要转换的数据
        
        Returns
        -------
        X_transformed : ndarray or DataFrame
            剪尾后的数据
        """
        # 检查是否已拟合
        if self.clip_values_ is None:
            raise RuntimeError("Winsorizer must be fitted before transform")
        
        X = self._validate_data(X, reset=False)
        
        if self.copy:
            X = X.copy()
        
        # 应用剪尾
        is_dataframe = isinstance(X, pd.DataFrame)
        
        for i, (lower, upper) in enumerate(self.clip_values_):
            if is_dataframe:
                X.iloc[:, i] = X.iloc[:, i].clip(lower=lower, upper=upper)
            else:
                X[:, i] = np.clip(X[:, i], lower, upper)
        
        return X
    
    def _validate_data(self, X, reset=True):
        """验证和转换输入数据"""
        # 转换为numpy数组（如果需要）
        if isinstance(X, pd.DataFrame):
            # 保持DataFrame格式以便后续处理
            X_array = X
        elif isinstance(X, (list, tuple)):
            X_array = np.array(X)
        else:
            X_array = X
        
        # 确保是2D
        if isinstance(X_array, np.ndarray) and X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # 检查特征数量
        if not reset and self.n_features_in_ is not None:
            n_features = X_array.shape[1]
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but Winsorizer "
                    f"is expecting {self.n_features_in_} features"
                )
        
        return X_array
    
    def get_feature_names_out(self, input_features=None):
        """
        获取输出特征名称
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            输入特征名称
        
        Returns
        -------
        feature_names_out : ndarray of str
            输出特征名称
        """
        if input_features is None:
            if self.feature_names_in_ is not None:
                return self.feature_names_in_
            else:
                return np.array([f"x{i}" for i in range(self.n_features_in_)])
        else:
            return np.asarray(input_features, dtype=object)


class RobustWinsorizer(Winsorizer):
    """
    鲁棒 Winsorizer
    
    使用中位数和MAD (Median Absolute Deviation) 进行剪尾，
    对异常值更鲁棒
    
    Parameters
    ----------
    n_mad : float, default=3.0
        MAD 的倍数
    
    copy : bool, default=True
        是否复制数据
    """
    
    def __init__(self, n_mad: float = 3.0, copy: bool = True):
        super().__init__(
            quantile_range=(0.01, 0.99),  # 不使用
            clip_method='mad',
            n_std=n_mad,
            copy=copy
        )
        self.n_mad = n_mad
    
    def fit(self, X, y=None):
        """计算基于 MAD 的剪尾界限"""
        X = self._validate_data(X)
        
        self.n_features_in_ = X.shape[1]
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        
        self.clip_values_ = []
        
        for i in range(X.shape[1]):
            col_data = X[:, i] if not isinstance(X, pd.DataFrame) else X.iloc[:, i]
            valid_data = col_data[~np.isnan(col_data)] if hasattr(col_data, '__iter__') else col_data
            
            if len(valid_data) == 0:
                self.clip_values_.append((-np.inf, np.inf))
                continue
            
            # 计算中位数和 MAD
            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))
            
            # 使用 MAD 的倍数作为界限
            # 注：1.4826 是将 MAD 转换为标准差的常数
            lower = median - self.n_mad * mad * 1.4826
            upper = median + self.n_mad * mad * 1.4826
            
            self.clip_values_.append((lower, upper))
        
        return self


# 便捷函数
def create_winsorizer(method='quantile', **kwargs):
    """
    创建 Winsorizer 的工厂函数
    
    Parameters
    ----------
    method : str, default='quantile'
        剪尾方法: 'quantile', 'std', 'mad'
    **kwargs : dict
        传递给 Winsorizer 的参数
    
    Returns
    -------
    winsorizer : Winsorizer or RobustWinsorizer
        Winsorizer 实例
    """
    if method == 'mad':
        return RobustWinsorizer(**kwargs)
    else:
        return Winsorizer(clip_method=method, **kwargs)
