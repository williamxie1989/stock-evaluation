# -*- coding: utf-8 -*-
"""metrics_io.py
用于将交叉验证各折指标持久化为 CSV。

提供 save_cv_metrics 函数，可接受任意指标列表，自动生成表头并写入文件。

示例
----
>>> from ml.evaluation.metrics_io import save_cv_metrics
>>> metrics = {
...     'accuracy': [0.85, 0.83, 0.88],
...     'precision': [0.8, 0.82, 0.85]
... }
>>> save_cv_metrics(metrics, 'metrics/xgb_classification_cv_scores.csv')
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

__all__ = ["save_cv_metrics"]


def save_cv_metrics(metrics_per_fold: Dict[str, List[float]], csv_path: str) -> None:
    """保存交叉验证各折指标到 CSV。

    Parameters
    ----------
    metrics_per_fold : Dict[str, List[float]]
        键为指标名称，值为该指标在各折上的取值列表。所有列表长度需一致。
    csv_path : str
        目标 CSV 文件路径。如上层目录不存在将自动创建。
    """
    if not metrics_per_fold:
        raise ValueError("metrics_per_fold 不能为空")

    # 校验所有指标长度一致
    lengths = {len(v) for v in metrics_per_fold.values()}
    if len(lengths) != 1:
        raise ValueError("所有指标列表长度必须一致")

    n_folds = lengths.pop()
    header = ["fold"] + list(metrics_per_fold.keys())

    # 创建目录
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_folds):
            row = [i + 1] + [metrics_per_fold[m][i] for m in metrics_per_fold]
            writer.writerow(row)