"""stacking_utils.py
自定义 Stacking 辅助函数：
1. generate_oof_predictions: 生成基学习器的 OOF 预测矩阵并返回已训练好的基学习器。
2. train_meta_learner: 使用 OOF 预测作为特征训练元学习器。

设计目标：
• 避免 sklearn.cross_val_predict 对 partition 的严格要求。
• 完全控制时间序列交叉验证逻辑，保证无未来泄漏。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_X_y

__all__ = [
    "generate_oof_predictions",
    "train_meta_learner",
    "SimpleStackingRegressor",
]


class SimpleStackingRegressor:
    """简化版可序列化 Stacking 回归器 (已训练基模型 + 元学习器)。

    仅实现 predict / get_params / set_params 以便与 sklearn 兼容并可被 pickle。
    """

    def __init__(self, base_estimators, meta_learner):
        self.base_estimators = base_estimators  # List[(name, estimator)]
        self.meta_learner = meta_learner

    def predict(self, X):
        import numpy as np
        meta_features = np.column_stack([est.predict(X) for _, est in self.base_estimators])
        return self.meta_learner.predict(meta_features)

    # 兼容 sklearn 的接口（部分）
    def get_params(self, deep: bool = True):
        params = {f"base_estimator_{i}": est for i, (_, est) in enumerate(self.base_estimators)}
        params["meta_learner"] = self.meta_learner
        return params

    def set_params(self, **params):
        # 仅允许替换 meta_learner
        if "meta_learner" in params:
            self.meta_learner = params["meta_learner"]
        return self


def generate_oof_predictions(
    estimators: List[Tuple[str, object]],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    fit_params: dict | None = None,
):
    """为每个基学习器生成 OOF 预测。

    参数
    ------
    estimators: [(name, estimator)] 列表。
    X, y: 训练数据。
    cv: 任意支持 split(X) 的交叉验证分割器。
    fit_params: 传递给 estimator.fit 的额外参数 dict，可为 None。

    返回
    ------
    oof_preds: ndarray, shape=(n_samples, n_estimators)
    fitted_estimators: [(name, fitted_estimator)] 与 estimators 顺序对应。
    """
    X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
    n_samples = X.shape[0]
    n_estimators = len(estimators)

    oof_preds = np.zeros((n_samples, n_estimators), dtype=float)
    fitted_estimators: list[Tuple[str, object]] = []

    # 先为每个 estimator 复制一份用于不同折训练
    for est_idx, (name, estimator) in enumerate(estimators):
        est_oof = np.zeros(n_samples, dtype=float)
        for train_idx, val_idx in cv.split(X):
            # 克隆新模型，避免不同折之间权重污染
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx], **(fit_params or {}))
            est_oof[val_idx] = model.predict(X[val_idx])
        oof_preds[:, est_idx] = est_oof
        # 最终使用 full 数据重新训练该基学习器，便于后续预测
        final_model = clone(estimator)
        final_model.fit(X, y, **(fit_params or {}))
        fitted_estimators.append((name, final_model))
    return oof_preds, fitted_estimators


def train_meta_learner(meta_estimator, oof_preds: np.ndarray, y: pd.Series):
    """使用 OOF 预测矩阵训练元学习器。

    返回已 fit 的 meta_estimator。"""
    meta_estimator = clone(meta_estimator)
    meta_estimator.fit(oof_preds, y)
    return meta_estimator


def run_oof_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None,
    estimators: List[Tuple[str, object]],
    meta_estimator,
    cv,
    fit_params: dict | None = None,
):
    """完整执行一次 OOF-Stacking 流程。

    返回
    ------
    stacking_model: SimpleStackingRegressor
    train_pred: ndarray，对应 X_train 的预测
    test_pred: ndarray 或 None，对应 X_test 的预测
    """
    # 生成 OOF 预测并训练基学习器
    oof_preds, fitted_estimators = generate_oof_predictions(
        estimators, X_train.values, y_train.values, cv, fit_params
    )
    # 训练元学习器
    meta_learner = train_meta_learner(meta_estimator, oof_preds, y_train.values)
    # 组装最终模型
    stacking_model = SimpleStackingRegressor(fitted_estimators, meta_learner)
    # 预测
    train_pred = stacking_model.predict(X_train.values)
    test_pred = stacking_model.predict(X_test.values) if X_test is not None else None
    return stacking_model, train_pred, test_pred