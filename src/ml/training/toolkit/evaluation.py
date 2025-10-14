# -*- coding: utf-8 -*-
"""评估相关工具。"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_by_month(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """按月份输出分类指标表现。"""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    dates_dt = pd.to_datetime(dates)
    thresholds = thresholds or {'0.5': 0.5}
    results: List[Dict[str, float]] = []

    for month in dates_dt.dt.to_period('M').unique():
        mask = dates_dt.dt.to_period('M') == month
        sample_count = int(mask.sum())
        if sample_count < 10:
            continue

        y_true_month = y_true[mask]
        y_pred_month = y_pred[mask]

        try:
            auc = float(roc_auc_score(y_true_month, y_pred_month)) if y_true_month.nunique() > 1 else np.nan
        except Exception as exc:
            logger.warning("月度评估 AUC 计算失败 %s: %s", month, exc)
            auc = np.nan

        row: Dict[str, float] = {
            'month': str(month),
            'samples': sample_count,
            'pos_rate': float(y_true_month.mean()),
            'auc': float(auc)
        }

        for name, thr in thresholds.items():
            y_pred_binary = (y_pred_month >= thr).astype(int)
            precision = precision_score(y_true_month, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_month, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_month, y_pred_binary, zero_division=0)
            row[f'precision@{name}'] = float(precision)
            row[f'recall@{name}'] = float(recall)
            row[f'f1@{name}'] = float(f1)

        results.append(row)

    df_results = pd.DataFrame(results)

    if len(df_results) > 0:
        logger.info("\n按月份评估:")
        threshold_labels = list(thresholds.keys())
        for _, row in df_results.iterrows():
            metrics_desc = [f"AUC{row['auc']:.3f}" if not np.isnan(row['auc']) else "AUCnan"]
            for name in threshold_labels:
                metrics_desc.append(
                    "P@%s=%.3f R@%s=%.3f F1@%s=%.3f" % (
                        name,
                        row[f'precision@{name}'],
                        name,
                        row[f'recall@{name}'],
                        name,
                        row[f'f1@{name}']
                    )
                )

            logger.info(
                "  %s: 样本%4d, 正样本率%5.1f%%, %s",
                row['month'],
                int(row['samples']),
                row['pos_rate'] * 100,
                ' | '.join(metrics_desc)
            )

        auc_std = df_results['auc'].std()
        logger.info(
            "\n  AUC标准差: %.4f %s",
            auc_std,
            '(稳定)' if auc_std < 0.05 else '(不稳定)'
        )

    return df_results
