"""时间序列金融评估指标实现
包括：
1. RankIC（秩相关信息系数）
2. Top Decile Spread（十分位差）
3. Hit Rate（方向命中率）

这些指标主要服务于横截面选股或因子回归等场景。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def _to_numpy(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """确保输入被转换为一维 numpy 数组"""
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return y_true, y_pred


def rank_ic(y_true, y_pred) -> float:
    """Spearman 秩相关 IC。

    参数
    ----
    y_true: array-like
        真实收益或标签。
    y_pred: array-like
        模型预测值（打分或因子暴露）。
    返回
    ----
    float
        RankIC，取值范围 [-1, 1]，越高表示预测排名与真实收益越一致。
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    # 如果 scipy 不可用，使用 pandas 计算秩相关
    try:
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_pred, y_true)
        return float(ic)
    except Exception:
        # 转秩排名
        rank_true = pd.Series(y_true).rank(pct=True).values
        rank_pred = pd.Series(y_pred).rank(pct=True).values
        cov = np.cov(rank_true, rank_pred, ddof=0)[0, 1]
        std_prod = np.std(rank_true) * np.std(rank_pred)
        return float(cov / std_prod) if std_prod != 0 else np.nan


def top_decile_spread(y_true, y_pred, quantile: float = 0.1) -> float:
    """Top/Bottom 分位收益差。

    将预测值排序，取前 quantile 与后 quantile 的 y_true 平均值之差。
    默认 quantile=0.1，即十分位差。
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df_sorted = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    k = int(np.floor(n * quantile))
    if k == 0:
        return np.nan
    top_mean = df_sorted.loc[: k - 1, "y_true"].mean()
    bottom_mean = df_sorted.loc[n - k :, "y_true"].mean()
    return float(top_mean - bottom_mean)


def hit_rate(y_true, y_pred) -> float:
    """方向命中率。

    计算预测值与真实值方向（正负号）一致的比例。
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    # 避免 0 值影响方向判断
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    hits = sign_true == sign_pred
    return float(np.mean(hits))


def sharpe_ratio(y_true, y_pred, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """计算夏普比率 (Sharpe Ratio)。

    这里默认将 *预测收益* y_pred 视为投资组合的日度收益序列，
    不依赖 y_true，因此保持与信息系数等评分函数的签名一致。
    若标准差为 0，返回 0 以避免数值错误。
    """
    _y = np.asarray(y_pred).flatten()
    if _y.size == 0:
        return 0.0
    excess_ret = _y - risk_free_rate / periods_per_year
    std = np.std(excess_ret)
    return float(np.mean(excess_ret) / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def information_ratio(y_true, y_pred) -> float:
    """计算信息比率 (Information Ratio)。

    定义为主动收益的平均值除以主动收益的标准差，
    其中主动收益 = y_pred - y_true。
    若标准差为 0，返回 0 以避免数值错误。
    """
    y_true_arr, y_pred_arr = _to_numpy(y_true, y_pred)
    active_ret = y_pred_arr - y_true_arr
    std_active = np.std(active_ret)
    return float(np.mean(active_ret) / std_active) if std_active > 0 else 0.0