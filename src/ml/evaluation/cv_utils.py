# -*- coding: utf-8 -*-
"""cv_utils.py
时间序列交叉验证工具。

当前仅实现 PurgedKFoldWithEmbargo，用于避免经典 KFold 在时间序列上产生的泄漏问题。
核心思想源于 Marcos López de Prado 在《Advances in Financial Machine Learning》中提出的 Purged K-Fold CV。

功能要点
--------
1. **Purging**: 删除训练集中与验证集时间区间重叠的样本，防止标签重叠导致的先知信息泄漏。
2. **Embargo**: 在验证集结束后留出一定“禁区”样本，使得未来数据不会泄漏到训练集。
3. **等宽折**: 为简化实现，将所有折划分为等宽（最后一折包含剩余样本）。

使用方式
--------
>>> cv = PurgedKFoldWithEmbargo(n_splits=5, embargo=10)
>>> for train_idx, test_idx in cv.split(X):
...     clf.fit(X[train_idx], y[train_idx])
...     y_pred = clf.predict(X[test_idx])

参数
----
- n_splits : 折数，默认为 5。
- embargo  : 禁区样本数量，整型，单位为“行”。典型设置为预测期 horizon 的长度。
- purge_window : 训练/测试分割时要剔除的重叠窗口长度，整型，默认为 0（全部由 embargo 处理）。
"""
from typing import Tuple, Iterator
import numpy as np
from sklearn.model_selection import BaseCrossValidator

__all__ = ["PurgedKFoldWithEmbargo"]


class PurgedKFoldWithEmbargo(BaseCrossValidator):
    """Purged K-Fold 交叉验证，带 Embargo。

    本实现假设输入样本已按时间顺序排序（递增）。当样本存在重复时间戳时，只要排序正确即可。
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0, purge_window: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")
        if embargo < 0 or purge_window < 0:
            raise ValueError("embargo 和 purge_window 必须 >= 0")
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge_window = purge_window

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: D401
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:  # noqa: D401
        n_samples = len(X)
        indices = np.arange(n_samples)
        # 等宽折，最后一折吸收余数
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            test_start = current
            test_end = current + fold_size  # exclusive
            current = test_end

            test_indices = indices[test_start:test_end]

            # 训练集左侧边界
            train_left_end = max(0, test_start - self.purge_window)
            train_left = indices[:train_left_end]

            # 训练集右侧开始 = 测试集结束 + embargo
            train_right_start = min(n_samples, test_end + self.embargo)
            train_right = indices[train_right_start:]

            train_indices = np.concatenate((train_left, train_right))
            yield train_indices, test_indices

    def __repr__(self):
        return (
            f"PurgedKFoldWithEmbargo(n_splits={self.n_splits}, "
            f"embargo={self.embargo}, purge_window={self.purge_window})"
        )