# -*- coding: utf-8 -*-
"""时间序列切分相关工具。"""

import logging
from typing import Tuple
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def improved_time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_size_ratio: float = 0.2,
    embargo_days: int = 40,  # 🔧 修复: 从5天改为40天 (预测期30天+10天缓冲)
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """执行带 embargo 期的时间序列切分。"""
    dates_dt = pd.to_datetime(dates)
    order = dates_dt.argsort()

    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    dates_sorted = dates_dt.iloc[order].reset_index(drop=True)

    total_size = len(X_sorted)
    test_size = int(total_size * test_size_ratio)
    train_size = total_size - test_size - embargo_days

    train_end_idx = train_size
    embargo_end_idx = train_end_idx + embargo_days

    X_train = X_sorted.iloc[:train_end_idx].reset_index(drop=True)
    y_train = y_sorted.iloc[:train_end_idx].reset_index(drop=True)
    X_val = X_sorted.iloc[embargo_end_idx:].reset_index(drop=True)
    y_val = y_sorted.iloc[embargo_end_idx:].reset_index(drop=True)

    if verbose:
        logger.info("时序切分统计:")
        logger.info("  训练集: %d 样本", len(X_train))
        logger.info(
            "    时间范围: %s ~ %s",
            dates_sorted.iloc[0].strftime('%Y-%m-%d'),
            dates_sorted.iloc[train_end_idx - 1].strftime('%Y-%m-%d')
        )
        logger.info("    正样本率: %.2f%%", y_train.mean() * 100)
        logger.info("  Embargo期: %d 天", embargo_days)
        logger.info(
            "    时间范围: %s ~ %s",
            dates_sorted.iloc[train_end_idx].strftime('%Y-%m-%d'),
            dates_sorted.iloc[embargo_end_idx - 1].strftime('%Y-%m-%d')
        )
        logger.info("  验证集: %d 样本", len(X_val))
        logger.info(
            "    时间范围: %s ~ %s",
            dates_sorted.iloc[embargo_end_idx].strftime('%Y-%m-%d'),
            dates_sorted.iloc[-1].strftime('%Y-%m-%d')
        )
        logger.info("    正样本率: %.2f%%", y_val.mean() * 100)
        logger.info(
            "  标签分布差异: %.2f%%",
            abs(y_train.mean() - y_val.mean()) * 100
        )

    return X_train, X_val, y_train, y_val


def rolling_window_time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_window_years: float = 3.0,
    test_size_ratio: float = 0.2,
    embargo_days: int = 40,  # 🔧 修复: 从5天改为40天 (预测期30天+10天缓冲)
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    使用滚动窗口的时间序列切分（仅保留最近N年训练数据）
    
    Parameters
    ----------
    train_window_years : float
        训练窗口长度（年），例如 3.0 表示只用最近3年数据训练
    """
    dates_dt = pd.to_datetime(dates)
    order = dates_dt.argsort()

    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    dates_sorted = dates_dt.iloc[order].reset_index(drop=True)

    # 计算验证集起始点
    total_size = len(X_sorted)
    test_size = int(total_size * test_size_ratio)
    val_start_idx = total_size - test_size
    
    # 计算训练集起始点（从验证集往前推 train_window_years 年）
    val_start_date = dates_sorted.iloc[val_start_idx]
    train_start_date = val_start_date - pd.DateOffset(years=train_window_years)
    
    # 找到训练集起始索引
    train_start_idx = dates_sorted.searchsorted(train_start_date, side='left')
    train_end_idx = val_start_idx - embargo_days
    
    # 确保训练集不为空
    if train_end_idx <= train_start_idx:
        logger.warning("滚动窗口过小，回退到全量训练")
        train_start_idx = 0
    
    X_train = X_sorted.iloc[train_start_idx:train_end_idx].reset_index(drop=True)
    y_train = y_sorted.iloc[train_start_idx:train_end_idx].reset_index(drop=True)
    X_val = X_sorted.iloc[val_start_idx:].reset_index(drop=True)
    y_val = y_sorted.iloc[val_start_idx:].reset_index(drop=True)

    if verbose:
        logger.info("时序切分统计（滚动窗口 %.1f 年）:", train_window_years)
        logger.info("  训练集: %d 样本", len(X_train))
        logger.info(
            "    时间范围: %s ~ %s",
            dates_sorted.iloc[train_start_idx].strftime('%Y-%m-%d'),
            dates_sorted.iloc[train_end_idx - 1].strftime('%Y-%m-%d')
        )
        logger.info("    正样本率: %.2f%%", y_train.mean() * 100)
        logger.info("  Embargo期: %d 天", embargo_days)
        logger.info("  验证集: %d 样本", len(X_val))
        logger.info(
            "    时间范围: %s ~ %s",
            dates_sorted.iloc[val_start_idx].strftime('%Y-%m-%d'),
            dates_sorted.iloc[-1].strftime('%Y-%m-%d')
        )
        logger.info("    正样本率: %.2f%%", y_val.mean() * 100)

    return X_train, X_val, y_train, y_val


def get_time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_years: float = 3.0,
    val_years: float = 1.0,
    embargo_days: int = 40,  # 🔧 修复: 从5天改为40天 (预测期30天+10天缓冲)
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    使用滚动窗口进行时间序列切分
    
    关键改进：使用最近 N 年数据训练，而非全部历史
    避免早期数据（如2015-2018）对近期预测造成负面影响
    
    Parameters
    ----------
    X : DataFrame
        特征数据
    y : Series
        标签
    dates : Series
        日期序列
    train_years : float
        训练窗口年数（默认3年）
    val_years : float  
        验证窗口年数（默认1年）
    embargo_days : int
        禁用期天数
    verbose : bool
        是否打印日志
        
    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    dates_dt = pd.to_datetime(dates)
    order = dates_dt.argsort()

    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    dates_sorted = dates_dt.iloc[order].reset_index(drop=True)

    # 计算验证集结束日期（最新日期）
    val_end_date = dates_sorted.iloc[-1]
    
    # 计算验证集开始日期（向前推 val_years 年）
    val_start_date = val_end_date - timedelta(days=int(365.25 * val_years))
    
    # 计算 embargo 开始日期
    embargo_start_date = val_start_date - timedelta(days=embargo_days)
    
    # 计算训练集结束日期
    train_end_date = embargo_start_date
    
    # 计算训练集开始日期（向前推 train_years 年）
    train_start_date = train_end_date - timedelta(days=int(365.25 * train_years))

    # 找到对应的索引
    train_mask = (dates_sorted >= train_start_date) & (dates_sorted < train_end_date)
    val_mask = dates_sorted >= val_start_date

    X_train = X_sorted[train_mask].reset_index(drop=True)
    y_train = y_sorted[train_mask].reset_index(drop=True)
    X_val = X_sorted[val_mask].reset_index(drop=True)
    y_val = y_sorted[val_mask].reset_index(drop=True)

    if len(X_train) == 0 or len(X_val) == 0:
        fallback_ratio = val_years / max(train_years + val_years, 1e-6)
        fallback_ratio = max(0.1, min(0.5, fallback_ratio))
        if verbose:
            logger.warning(
                "滚动窗口切分样本不足(训练=%d, 验证=%d)，回退到改进时间切分 (test_ratio=%.2f)",
                len(X_train),
                len(X_val),
                fallback_ratio
            )
        return improved_time_series_split(
            X,
            y,
            dates,
            test_size_ratio=fallback_ratio,
            embargo_days=embargo_days,
            verbose=verbose
        )

    if verbose:
        logger.info("滚动窗口切分统计:")
        logger.info("  训练窗口: %.1f 年", train_years)
        logger.info("  训练集: %d 样本", len(X_train))
        if len(X_train) > 0:
            logger.info(
                "    时间范围: %s ~ %s",
                dates_sorted[train_mask].iloc[0].strftime('%Y-%m-%d'),
                dates_sorted[train_mask].iloc[-1].strftime('%Y-%m-%d')
            )
            logger.info("    正样本率: %.2f%%", y_train.mean() * 100)
        logger.info("  Embargo期: %d 天", embargo_days)
        logger.info("  验证窗口: %.1f 年", val_years)
        logger.info("  验证集: %d 样本", len(X_val))
        if len(X_val) > 0:
            logger.info(
                "    时间范围: %s ~ %s",
                dates_sorted[val_mask].iloc[0].strftime('%Y-%m-%d'),
                dates_sorted[val_mask].iloc[-1].strftime('%Y-%m-%d')
            )
            logger.info("    正样本率: %.2f%%", y_val.mean() * 100)
        
        # 计算标签分布差异
        if len(X_train) > 0 and len(X_val) > 0:
            label_diff = abs(y_train.mean() - y_val.mean()) * 100
            logger.info("  标签分布差异: %.2f%%", label_diff)

    return X_train, X_val, y_train, y_val


# 🔧 兼容性别名：rolling_window_split = get_time_series_split
# 为了兼容旧代码（enhanced_trainer_v2.py），提供别名
def rolling_window_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_years: float = 3.0,
    val_years: float = 1.0,
    embargo_days: int = 40,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    滚动窗口时间序列切分（兼容性包装器）
    
    这是 get_time_series_split 的别名函数，用于兼容旧代码。
    参数名称适配：train_years, val_years → 传递给 get_time_series_split
    
    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    y : pd.Series
        标签数据
    dates : pd.Series
        日期序列
    train_years : float, default=3.0
        训练集窗口长度（年）
    val_years : float, default=1.0
        验证集窗口长度（年）
    embargo_days : int, default=40
        禁用期天数（防止标签泄漏）
    verbose : bool, default=True
        是否打印详细信息
        
    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    return get_time_series_split(
        X=X,
        y=y,
        dates=dates,
        train_years=train_years,
        val_years=val_years,
        embargo_days=embargo_days,
        verbose=verbose
    )
