# -*- coding: utf-8 -*-
"""
特征选择模块
支持分类和回归任务的独立特征选择
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import logging
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)


logger = logging.getLogger(__name__)


class SafeSimpleImputer(SimpleImputer):
    """在 sklearn SimpleImputer 基础上增加兜底值，避免全缺失列触发警告。"""

    def __init__(self, fallback_value: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.fallback_value = fallback_value

    def fit(self, X, y=None):  # noqa: N803
        fitted = super().fit(X, y)
        stats = getattr(self, 'statistics_', None)
        if stats is not None:
            stats = np.array(stats, copy=True)
            if np.issubdtype(stats.dtype, np.number):
                mask = np.isnan(stats)
                if mask.any():
                    fallback = self.fill_value if self.strategy == 'constant' else self.fallback_value
                    stats[mask] = fallback
                    self.statistics_ = stats
        return fitted

try:
    from config.prediction_config import (
        FEATURE_SELECTION_GROUP_LIMIT,
        FEATURE_SELECTION_CORR_THRESHOLD
    )
except Exception:  # pragma: no cover
    FEATURE_SELECTION_GROUP_LIMIT = 4
    FEATURE_SELECTION_CORR_THRESHOLD = 0.95


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
    cv_splits: int = 5,
    candidate_step: Optional[int] = None,
    scoring: Optional[str] = None,
    cv_n_jobs: int = 1,
    random_state: int = 42,
    **estimator_params
) -> Tuple[List[str], np.ndarray, List[str]]:
    """多策略融合 + 时间序列交叉验证的特征选择。

    Returns
    -------
    selected_features : List[str]
        最终保留的特征名称。
    importances : ndarray
        与 `importance_feature_names` 对齐的特征重要性。
    importance_feature_names : List[str]
        参与重要性计算的基础特征顺序（仅包含数值特征）。
    """

    if task not in {'classification', 'regression'}:
        raise ValueError(f"Unsupported task: {task}")

    method = (method or 'lightgbm').lower()
    estimator_params = dict(estimator_params)
    use_shap = method == 'shap'

    shap_sample_size: Optional[int] = None
    shap_base_method = 'lightgbm'
    if method == 'model_based':
        # 历史别名，保持与原有模型重要性方法一致
        estimator_method = 'lightgbm'
    elif use_shap:
        shap_sample_size = estimator_params.pop('shap_sample_size', 2000)
        shap_background_size = estimator_params.pop('shap_background_size', 256)
        shap_tree_limit = estimator_params.pop('shap_tree_limit', None)
        shap_base_method = estimator_params.pop('shap_base_method', 'lightgbm')
        if shap_base_method not in {'lightgbm', 'xgboost', 'random_forest'}:
            logger.warning("未知的 SHAP 基模型 %s，回退至 lightgbm", shap_base_method)
            shap_base_method = 'lightgbm'
        estimator_method = shap_base_method
    else:
        estimator_method = method

    feature_names = X.columns.tolist()
    if not feature_names:
        return [], np.array([]), []

    categorical_cols = [col for col in feature_names if X[col].dtype == 'object' or str(X[col].dtype).startswith('category')]
    if categorical_cols:
        logger.info("特征选择跳过非数值列: %s", categorical_cols)

    X_numeric = X.drop(columns=categorical_cols, errors='ignore').copy()
    feature_names = X_numeric.columns.tolist()
    if not feature_names:
        return [], np.array([]), []

    empty_cols = [col for col in feature_names if not X_numeric[col].notna().any()]
    if empty_cols:
        logger.info("特征选择移除全缺失列: %s", empty_cols)
        X_numeric = X_numeric.drop(columns=empty_cols)
        feature_names = X_numeric.columns.tolist()
        if not feature_names:
            return [], np.array([]), []

    for col in feature_names:
        if X_numeric[col].dtype == 'object':
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    medians = X_numeric.median()
    X_filled = X_numeric.fillna(medians)

    X_matrix = X_filled.to_numpy(dtype=float)
    y_array = np.asarray(y)
    n_features = X_matrix.shape[1]

    if max_features is None or max_features > n_features:
        max_features = n_features
    min_features = max(1, min(min_features, n_features))
    if min_features > max_features:
        min_features = max_features

    def _safe_normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(scores, 0):
            return np.zeros_like(scores)
        mn, mx = scores.min(), scores.max()
        if np.isclose(mx, mn):
            return np.ones_like(scores)
        return (scores - mn) / (mx - mn)

    # 单变量
    try:
        if task == 'classification' and len(np.unique(y_array)) > 1:
            uni_scores, _ = f_classif(X_matrix, y_array)
        elif task == 'regression':
            uni_scores, _ = f_regression(X_matrix, y_array)
        else:
            uni_scores = np.zeros(n_features)
    except Exception as exc:
        logger.debug("单变量打分失败: %s", exc)
        uni_scores = np.zeros(n_features)
    uni_scores = _safe_normalize(uni_scores)

    # 互信息
    try:
        if task == 'classification' and len(np.unique(y_array)) > 1:
            mi_scores = mutual_info_classif(X_matrix, y_array, random_state=random_state)
        elif task == 'regression':
            mi_scores = mutual_info_regression(X_matrix, y_array, random_state=random_state)
        else:
            mi_scores = np.zeros(n_features)
    except Exception as exc:
        logger.debug("互信息打分失败: %s", exc)
        mi_scores = np.zeros(n_features)
    mi_scores = _safe_normalize(mi_scores)

    # 模型重要性
    if estimator_method == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        base_estimator = (LGBMClassifier if task == 'classification' else LGBMRegressor)(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            n_jobs=1,
            random_state=random_state,
            verbosity=-1,
            **estimator_params
        )
    elif estimator_method == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor
        base_estimator = (XGBClassifier if task == 'classification' else XGBRegressor)(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=random_state,
            verbosity=0,
            **estimator_params
        )
    elif estimator_method == 'random_forest':
        base_estimator = (RandomForestClassifier if task == 'classification' else RandomForestRegressor)(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=1,
            **estimator_params
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    try:
        base_estimator.fit(X_matrix, y_array)
        if use_shap:
            model_scores = _safe_normalize(_compute_shap_importance(
                base_estimator,
                X_matrix,
                task,
                random_state,
                shap_sample_size,
                shap_background_size,
                shap_tree_limit
            ))
            if np.allclose(model_scores, 0):
                logger.warning("SHAP 重要性为空，回退至模型特征重要性")
                model_scores = getattr(base_estimator, 'feature_importances_', np.zeros(n_features))
        else:
            model_scores = getattr(base_estimator, 'feature_importances_', np.zeros(n_features))
    except Exception as exc:
        logger.debug("模型重要性计算失败: %s", exc)
        model_scores = np.zeros(n_features)
    model_scores = _safe_normalize(model_scores)

    importance_label = 'shap' if use_shap else 'model'
    scores_df = pd.DataFrame({
        'feature': feature_names,
        'univariate': uni_scores,
        'mutual_info': mi_scores,
        'importance': model_scores
    })
    scores_df['combined'] = scores_df[['univariate', 'mutual_info', 'importance']].mean(axis=1)
    scores_df.sort_values(by='combined', ascending=False, inplace=True)

    logger.info("特征打分汇总 (Top 10) [%s]", importance_label)
    for _, row in scores_df.head(10).iterrows():
        logger.info(
            "  %s | combined=%.4f uni=%.4f mi=%.4f %s=%.4f",
            row['feature'], row['combined'], row['univariate'], row['mutual_info'], importance_label, row['importance']
        )

    ordered_features = scores_df['feature'].tolist()

    def _infer_group(name: str) -> str:
        if not isinstance(name, str):
            return 'unknown'
        if name.startswith('fund_'):
            parts = name.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:2])
        if '_' in name:
            return name.split('_')[0]
        return name

    # 先按照相关性去重，避免高度共线特征重复入选
    corr_filtered_features: List[str] = []
    if ordered_features:
        try:
            corr_matrix = X_filled[ordered_features].corr().abs().fillna(0.0)
            for feat in ordered_features:
                if not corr_filtered_features:
                    corr_filtered_features.append(feat)
                    continue
                if feat not in corr_matrix.index:
                    corr_filtered_features.append(feat)
                    continue
                corr_with_selected = corr_matrix.loc[feat, corr_filtered_features]
                if corr_with_selected.empty or (corr_with_selected < FEATURE_SELECTION_CORR_THRESHOLD).all():
                    corr_filtered_features.append(feat)
        except Exception as exc:  # pragma: no cover
            logger.debug("相关性去重失败，保留原始排序: %s", exc)
            corr_filtered_features = ordered_features
    else:
        corr_filtered_features = ordered_features

    if len(corr_filtered_features) < len(ordered_features):
        logger.info("相关性去重: %d → %d", len(ordered_features), len(corr_filtered_features))

    # 控制每个主题的特征数量，避免同类特征集中
    group_limit = max(1, FEATURE_SELECTION_GROUP_LIMIT)
    group_counts: Counter[str] = Counter()
    balanced_features: List[str] = []
    for feat in corr_filtered_features:
        group = _infer_group(feat)
        if group_counts[group] >= group_limit:
            continue
        group_counts[group] += 1
        balanced_features.append(feat)

    if len(balanced_features) < len(corr_filtered_features):
        logger.info("主题配额裁剪: %d → %d (每组上限=%d)", len(corr_filtered_features), len(balanced_features), group_limit)

    # 若筛后不足最小特征数，则按原排序补齐
    if len(balanced_features) < min_features:
        existing = set(balanced_features)
        for feat in ordered_features:
            if feat in existing:
                continue
            balanced_features.append(feat)
            existing.add(feat)
            if len(balanced_features) >= min_features:
                break

    ordered_features = balanced_features

    if candidate_step is None:
        candidate_step = max(1, (max_features - min_features) // 5)
    candidate_sizes = sorted(set([min_features, max_features] + list(range(min_features, max_features + 1, candidate_step))))

    max_possible_splits = len(X_numeric) - 1
    if max_possible_splits < 2:
        logger.warning("样本量过小，跳过CV评估，直接返回Top特征")
        selected_features = ordered_features[:max_features]
        combined_scores_aligned = scores_df.set_index('feature')['combined'].reindex(feature_names).fillna(0.0)
        importances = combined_scores_aligned.to_numpy(dtype=float)
        return selected_features, importances, feature_names

    n_splits = min(cv_splits, max_possible_splits)
    n_splits = max(2, n_splits)
    cv = TimeSeriesSplit(n_splits=n_splits)
    if scoring is None:
        scoring = 'roc_auc' if task == 'classification' else 'r2'

    best_score = -np.inf
    best_size = candidate_sizes[0]
    best_cv_scores: Optional[np.ndarray] = None

    for size in candidate_sizes:
        subset_features = ordered_features[:size]
        subset_data = X_numeric[subset_features]

        cv_model = clone(base_estimator)
        pipeline_steps = [('imputer', SafeSimpleImputer(strategy='median', fallback_value=0.0))]
        pipeline_steps.append(('model', cv_model))
        cv_pipeline = Pipeline(pipeline_steps)

        try:
            cv_scores = cross_val_score(
                cv_pipeline,
                subset_data,
                y_array,
                cv=cv,
                scoring=scoring,
                n_jobs=cv_n_jobs
            )
        except ValueError:
            fallback = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'
            cv_scores = cross_val_score(
                cv_pipeline,
                subset_data,
                y_array,
                cv=cv,
                scoring=fallback,
                n_jobs=cv_n_jobs
            )
            cv_scores = np.nan_to_num(cv_scores, nan=0.0)

        mean_score = float(np.nanmean(cv_scores))
        logger.info(
            "候选特征数 %d | CV(%s)=%.4f ± %.4f",
            size, scoring, mean_score, float(np.nanstd(cv_scores))
        )

        if mean_score > best_score:
            best_score = mean_score
            best_size = size
            best_cv_scores = cv_scores

    selected_features = ordered_features[:best_size]
    logger.info("✅ 最终保留 %d 个特征 (CV最佳 %.4f)", len(selected_features), best_score)
    if best_cv_scores is not None:
        logger.info("  折别得分: %s", np.array2string(np.asarray(best_cv_scores), precision=4))

    combined_scores_aligned = scores_df.set_index('feature')['combined'].reindex(feature_names).fillna(0.0)
    importances = combined_scores_aligned.to_numpy(dtype=float)

    return selected_features, importances, feature_names


def _compute_shap_importance(
    estimator,
    X_matrix: np.ndarray,
    task: str,
    random_state: int,
    shap_sample_size: Optional[int],
    shap_background_size: Optional[int],
    shap_tree_limit: Optional[int]
) -> np.ndarray:
    """
    计算基于 SHAP 的特征重要性（平均绝对 SHAP 值）。
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        logger.warning("未安装 shap 库，无法使用 SHAP 特征选择: %s", exc)
        return np.zeros(X_matrix.shape[1], dtype=float)

    n_samples = X_matrix.shape[0]
    rng = np.random.default_rng(random_state)

    try:
        requested_sample = int(shap_sample_size) if shap_sample_size else n_samples
    except (TypeError, ValueError):
        logger.warning("无效的 shap_sample_size=%s，使用全部样本", shap_sample_size)
        requested_sample = n_samples
    sample_size = min(n_samples, max(200, requested_sample))
    if sample_size < n_samples:
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        sample_data = X_matrix[sample_indices]
    else:
        sample_data = X_matrix

    try:
        requested_background = int(shap_background_size) if shap_background_size else 256
    except (TypeError, ValueError):
        logger.warning("无效的 shap_background_size=%s，使用默认背景样本数", shap_background_size)
        requested_background = 256
    background_size = min(sample_data.shape[0], max(50, requested_background))
    if background_size < sample_data.shape[0]:
        background_indices = rng.choice(sample_data.shape[0], size=background_size, replace=False)
        background = sample_data[background_indices]
    else:
        background = sample_data

    try:
        explainer = shap.TreeExplainer(estimator, data=background)
    except Exception as exc:
        logger.warning("构建 SHAP 解释器失败，回退至模型重要性: %s", exc)
        return np.zeros(X_matrix.shape[1], dtype=float)

    shap_kwargs = {}
    if shap_tree_limit is not None:
        try:
            shap_kwargs['tree_limit'] = int(shap_tree_limit)
        except (TypeError, ValueError):
            logger.warning("无效的 shap_tree_limit=%s，忽略该限制", shap_tree_limit)
    if task == 'regression':
        shap_kwargs.setdefault('check_additivity', False)

    try:
        shap_values = explainer.shap_values(sample_data, **shap_kwargs)
    except Exception as exc:
        logger.warning("计算 SHAP 值失败，回退至模型重要性: %s", exc)
        return np.zeros(X_matrix.shape[1], dtype=float)

    if isinstance(shap_values, list):
        shap_abs = [np.abs(values) for values in shap_values]
        stacked = np.stack(shap_abs, axis=0)
        shap_scores = stacked.mean(axis=(0, 1))
    else:
        shap_array = np.abs(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array.mean(axis=0)
        shap_scores = shap_array.mean(axis=0)

    if shap_scores.shape[0] != X_matrix.shape[1]:
        logger.warning("SHAP 重要性维度不匹配，回退至模型重要性")
        return np.zeros(X_matrix.shape[1], dtype=float)

    return shap_scores.astype(float)


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
    selected_cls, importances_cls, _ = select_features_for_task(
        X, y_cls, task='classification', method=method, **kwargs
    )
    
    # 回归任务特征选择
    logger.info("\n回归任务特征选择...")
    selected_reg, importances_reg, _ = select_features_for_task(
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
