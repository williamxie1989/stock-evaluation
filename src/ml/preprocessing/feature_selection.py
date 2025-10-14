# -*- coding: utf-8 -*-
"""
特征选择模块
支持分类和回归任务的独立特征选择
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ModelBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    基于模型的特征选择器
    
    使用树模型（LightGBM/XGBoost）的特征重要性进行选择
    
    Parameters
    ----------
    estimator : object
        带有 feature_importances_ 属性的估计器
    
    threshold : float or str, default='median'
        特征重要性阈值
        - float: 绝对阈值
        - 'median': 使用中位数
        - 'mean': 使用平均值
    
    min_features : int, default=10
        最少保留的特征数量
    
    max_features : int or None, default=None
        最多保留的特征数量
    """
    
    def __init__(self,
                 estimator,
                 threshold: float = 'median',
                 min_features: int = 10,
                 max_features: Optional[int] = None):
        self.estimator = estimator
        self.threshold = threshold
        self.min_features = min_features
        self.max_features = max_features
        
        self.selected_features_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y):
        """
        拟合估计器并选择特征
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        y : array-like of shape (n_samples,)
            目标变量
        
        Returns
        -------
        self : object
        """
        # 训练估计器
        self.estimator.fit(X, y)
        
        # 获取特征重要性
        if not hasattr(self.estimator, 'feature_importances_'):
            raise AttributeError("Estimator must have feature_importances_ attribute")
        
        importances = self.estimator.feature_importances_
        self.feature_importances_ = importances
        self.n_features_in_ = len(importances)
        
        # 计算阈值
        if isinstance(self.threshold, str):
            if self.threshold == 'median':
                thresh_value = np.median(importances)
            elif self.threshold == 'mean':
                thresh_value = np.mean(importances)
            else:
                raise ValueError(f"Unknown threshold: {self.threshold}")
        else:
            thresh_value = self.threshold
        
        # 选择特征
        mask = importances >= thresh_value
        selected_indices = np.where(mask)[0]
        
        # 确保最小和最大数量
        if len(selected_indices) < self.min_features:
            # 选择top min_features
            selected_indices = np.argsort(importances)[-self.min_features:]
        
        if self.max_features is not None and len(selected_indices) > self.max_features:
            # 选择top max_features
            top_indices = np.argsort(importances)[-self.max_features:]
            selected_indices = top_indices
        
        self.selected_features_ = sorted(selected_indices)
        
        logger.info(f"特征选择完成: {len(self.selected_features_)}/{self.n_features_in_} 特征保留")
        
        return self
    
    def transform(self, X):
        """
        转换数据，只保留选定的特征
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入数据
        
        Returns
        -------
        X_selected : array-like of shape (n_samples, n_selected_features)
            选定特征的数据
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        else:
            return X[:, self.selected_features_]
    
    def get_support(self, indices=False):
        """
        获取选定特征的掩码或索引
        
        Parameters
        ----------
        indices : bool, default=False
            如果为True，返回索引；否则返回布尔掩码
        
        Returns
        -------
        support : array
            特征掩码或索引
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted")
        
        if indices:
            return np.array(self.selected_features_)
        else:
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
            return mask
    
    def get_feature_names_out(self, input_features=None):
        """
        获取输出特征名称
        
        Parameters
        ----------
        input_features : array-like of str or None
            输入特征名称
        
        Returns
        -------
        feature_names_out : ndarray of str
            选定的特征名称
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted")
        
        if input_features is None:
            return np.array([f"x{i}" for i in self.selected_features_])
        else:
            input_features = np.asarray(input_features, dtype=object)
            return input_features[self.selected_features_]


def select_features_for_task(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    method: str = 'lightgbm',
    threshold: str = 'median',
    min_features: int = 10,
    max_features: Optional[int] = None,
    **estimator_params
) -> Tuple[List[str], np.ndarray]:
    """
    为特定任务选择特征
    
    Parameters
    ----------
    X : DataFrame
        特征数据
    y : Series
        目标变量
    task : str
        任务类型: 'classification' 或 'regression'
    method : str, default='lightgbm'
        选择方法: 'lightgbm', 'xgboost', 'random_forest'
    threshold : str or float, default='median'
        重要性阈值
    min_features : int, default=10
        最少保留特征数
    max_features : int or None, default=None
        最多保留特征数
    **estimator_params : dict
        传递给估计器的参数
    
    Returns
    -------
    selected_features : List[str]
        选定的特征名称列表
    importances : ndarray
        特征重要性
    """
    # 创建估计器
    if method == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        if task == 'classification':
            estimator = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
                **estimator_params
            )
        else:
            estimator = LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
                **estimator_params
            )
    elif method == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor
        if task == 'classification':
            estimator = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                **estimator_params
            )
        else:
            estimator = XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                **estimator_params
            )
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **estimator_params
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **estimator_params
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 创建选择器
    selector = ModelBasedFeatureSelector(
        estimator=estimator,
        threshold=threshold,
        min_features=min_features,
        max_features=max_features
    )
    
    # 拟合
    selector.fit(X, y)
    
    # 获取选定特征
    feature_names = X.columns.tolist()
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    logger.info(f"{task} 任务特征选择完成:")
    logger.info(f"  原始特征数: {len(feature_names)}")
    logger.info(f"  选定特征数: {len(selected_features)}")
    
    # 显示top特征
    importances = selector.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    logger.info(f"  Top 10 特征:")
    for i in top_indices:
        logger.info(f"    {feature_names[i]}: {importances[i]:.4f}")
    
    return selected_features, selector.feature_importances_


def select_features_separately(
    X: pd.DataFrame,
    y_cls: pd.Series,
    y_reg: pd.Series,
    method: str = 'lightgbm',
    **kwargs
) -> Dict[str, List[str]]:
    """
    分别为分类和回归任务选择特征
    
    Parameters
    ----------
    X : DataFrame
        特征数据
    y_cls : Series
        分类标签
    y_reg : Series
        回归目标
    method : str, default='lightgbm'
        选择方法
    **kwargs : dict
        传递给 select_features_for_task 的参数
    
    Returns
    -------
    result : dict
        包含 'classification' 和 'regression' 的特征列表
    """
    logger.info("="*80)
    logger.info("开始分类和回归任务的独立特征选择")
    logger.info("="*80)
    
    # 分类任务特征选择
    logger.info("\n分类任务特征选择...")
    selected_cls, importances_cls = select_features_for_task(
        X, y_cls, task='classification', method=method, **kwargs
    )
    
    # 回归任务特征选择
    logger.info("\n回归任务特征选择...")
    selected_reg, importances_reg = select_features_for_task(
        X, y_reg, task='regression', method=method, **kwargs
    )
    
    # 汇总
    logger.info("\n"+"="*80)
    logger.info("特征选择完成")
    logger.info("="*80)
    logger.info(f"分类特征: {len(selected_cls)} 个")
    logger.info(f"回归特征: {len(selected_reg)} 个")
    logger.info(f"共同特征: {len(set(selected_cls) & set(selected_reg))} 个")
    logger.info(f"独有特征: 分类 {len(set(selected_cls) - set(selected_reg))}, 回归 {len(set(selected_reg) - set(selected_cls))}")
    
    return {
        'classification': selected_cls,
        'regression': selected_reg,
        'importances_cls': importances_cls,
        'importances_reg': importances_reg
    }
