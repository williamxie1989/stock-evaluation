# -*- coding: utf-8 -*-
"""
增强版统一训练器 V2
整合新特征、预处理管道、特征选择、概率校准
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta, timezone
import joblib
import os
import json
import hashlib
from pathlib import Path
import math

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score

# 导入配置
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import *
from src.ml.preprocessing.winsorizer import Winsorizer
from src.ml.preprocessing.feature_selection import select_features_for_task, SafeSimpleImputer
from pandas.util import hash_pandas_object

logger = logging.getLogger(__name__)


class PreprocessedModel(BaseEstimator):
    """Wraps a fitted preprocessor and estimator for safe reuse."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        estimator: Any,
        task: str,
        feature_columns: Optional[List[str]] = None
    ):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.task = task
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.feature_names_in_ = (np.array(self.feature_columns)
                                  if self.feature_columns is not None
                                  else getattr(preprocessor, 'feature_names_in_', None))
        if hasattr(estimator, 'classes_'):
            self.classes_ = estimator.classes_
        if hasattr(estimator, 'n_features_in_'):
            self.n_features_in_ = estimator.n_features_in_
        self._has_decision_function = hasattr(estimator, 'decision_function')

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if self.feature_columns is not None:
                # 按训练列顺序对齐，避免列缺失或乱序
                return X.loc[:, self.feature_columns]
            return X
        if self.feature_columns is not None:
            return pd.DataFrame(X, columns=self.feature_columns)
        # 无列信息时退化为范围索引
        return pd.DataFrame(X)

    def _transform(self, X: pd.DataFrame) -> Any:
        df = self._ensure_dataframe(X)
        return self.preprocessor.transform(df)

    def fit(self, X: Any, y: Any):
        if isinstance(X, pd.DataFrame):
            self.feature_columns = list(X.columns)
        elif self.feature_columns is None and isinstance(X, np.ndarray):
            self.feature_columns = list(range(X.shape[1]))
        self.preprocessor.fit(X)
        Xt = self.preprocessor.transform(X)
        self.estimator.fit(Xt, y)
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        if hasattr(self.estimator, 'n_features_in_'):
            self.n_features_in_ = self.estimator.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_t = self._transform(X)
        return self.estimator.predict(X_t)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self.estimator, 'predict_proba'):
            raise AttributeError("Estimator does not support predict_proba")
        X_t = self._transform(X)
        return self.estimator.predict_proba(X_t)

    def decision_function(self, X: Any) -> np.ndarray:
        X_t = self._transform(X)
        if self._has_decision_function:
            try:
                return self.estimator.decision_function(X_t)
            except AttributeError:
                pass
        if hasattr(self.estimator, 'predict_proba'):
            proba = self.estimator.predict_proba(X_t)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        raise AttributeError("Estimator does not support decision_function or predict_proba")

    @property
    def named_steps(self) -> Dict[str, Any]:
        return {'preprocessor': self.preprocessor, 'estimator': self.estimator}

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {'preprocessor': self.preprocessor, 'estimator': self.estimator, 'task': self.task}


class XGBoostClassifierWrapper(BaseEstimator):
    """封装原生 XGBoost Booster 以满足 sklearn 接口需求"""

    def __init__(
        self,
        booster: Any,
        best_iteration: int,
        feature_count: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ):
        import numpy as _np  # 局部导入避免全局依赖

        self.booster = booster
        self.best_iteration = best_iteration
        self.n_features_in_ = feature_count if feature_count is not None else 0
        self.feature_names = feature_names
        self.classes_ = _np.array([0, 1])

    def predict_proba(self, X: Any) -> np.ndarray:
        iteration_range = (0, self.best_iteration + 1) if self.best_iteration is not None else None
        proba = self.booster.inplace_predict(
            X,
            iteration_range=iteration_range
        )
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class XGBoostRegressorWrapper(BaseEstimator):
    """封装回归 Booster"""

    def __init__(self, booster: Any, best_iteration: int, feature_count: Optional[int] = None):
        self.booster = booster
        self.best_iteration = best_iteration
        self.n_features_in_ = feature_count if feature_count is not None else 0

    def predict(self, X: Any) -> np.ndarray:
        iteration_range = (0, self.best_iteration + 1) if self.best_iteration is not None else None
        return self.booster.inplace_predict(X, iteration_range=iteration_range)

class EnhancedTrainerV2:
    """
    增强版训练器 V2
    
    核心改进：
    1. 统一预处理管道（ColumnTransformer）
    2. 分类/回归独立特征选择
    3. 概率校准（Isotonic/Platt）
    4. 新模型持久化格式
    """
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        config: Optional[Dict] = None
    ):
        """
        初始化训练器
        
        Parameters
        ----------
        numerical_features : List[str]
            数值特征列表
        categorical_features : List[str]
            类别特征列表
        config : dict or None
            配置字典（若为None则使用默认配置）
        """
        # 保存特征列表副本，避免外部列表被原地修改
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        # 记录初始特征集，便于在不同折中按需筛选
        self._base_numerical_features = list(self.numerical_features)
        self._base_categorical_features = list(self.categorical_features)
        base_config = self._get_default_config()
        if config:
            base_config.update(config)
        self.config = base_config
        
        logger.info("初始化 EnhancedTrainerV2")
        logger.info(f"  数值特征: {len(numerical_features)}")
        logger.info(f"  类别特征: {len(categorical_features)}")
        
        # Stage 5: 特征选择配置
        self.enable_feature_selection = self.config.get('enable_feature_selection', False)
        if self.enable_feature_selection:
            logger.info("  ✅ 特征选择: 已启用")
            logger.info(f"     方法: {self.config.get('feature_selection_method', 'lightgbm')}")
            logger.info(f"     范围: {self.config.get('min_features', 15)}-{self.config.get('max_features', 30)} 特征")
            if self.config.get('feature_selection_method', 'lightgbm') == 'shap':
                logger.info("     SHAP 抽样: %s, 背景: %s",
                            self.config.get('shap_sample_size'),
                            self.config.get('shap_background_size'))

        self.enable_sample_weighting = bool(self.config.get('enable_sample_weighting', False))
        self.sample_weight_halflife = float(self.config.get('sample_weight_halflife', SAMPLE_WEIGHT_HALFLIFE_YEARS))
        if self.enable_sample_weighting:
            logger.info("  ✅ 样本加权: 已启用 (半衰期=%.2f年)", self.sample_weight_halflife)
        
        # Stage 5: Optuna超参数优化配置
        self.enable_optuna = self.config.get('enable_optuna', False)
        if self.enable_optuna:
            logger.info("  ✅ Optuna优化: 已启用")
            logger.info(f"     试验次数: {self.config.get('optuna_trials', 100)}")
            logger.info(f"     采样器: {self.config.get('optuna_sampler', 'tpe')}")

        # 数据驱动参数调整配置
        self.enable_data_driven_param_adjustment = self.config.get('enable_data_driven_param_adjustment', ENABLE_DATA_DRIVEN_PARAM_ADJUSTMENT)
        self.data_driven_adjustment_mode = self.config.get('data_driven_adjustment_mode', DATA_DRIVEN_ADJUSTMENT_MODE)
        if self.enable_data_driven_param_adjustment:
            logger.info("  ✅ 数据驱动参数调整: 已启用")
            logger.info(f"     模式: {self.data_driven_adjustment_mode}")
            logger.info(f"     小样本阈值: {self.config.get('small_sample_threshold', SMALL_SAMPLE_THRESHOLD)}")
            logger.info(f"     大样本阈值: {self.config.get('large_sample_threshold', LARGE_SAMPLE_THRESHOLD)}")
            logger.info(f"     高维特征阈值: {self.config.get('high_dimension_threshold', HIGH_DIMENSION_THRESHOLD)}")
            logger.info(f"     高复杂度阈值: {self.config.get('high_complexity_threshold', HIGH_COMPLEXITY_THRESHOLD)}")

        self.cv_fold_info: List[Dict[str, Any]] = []
        self._cv_sorted_frames: Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]] = None
        self._cv_pairs: List[Dict[str, Any]] = []
        self._cv_classification_records: List[Dict[str, Any]] = []
        self._cv_regression_records: List[Dict[str, Any]] = []
        self._last_split_dates: Dict[str, Optional[pd.Series]] = {'train': None, 'val': None}
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'winsor_clip_quantile': WINSOR_CLIP_QUANTILE,
            'calibration_method': CALIBRATION_METHOD,
            'calibration_cv': CALIBRATION_CV,
            'enable_feature_selection': ENABLE_FEATURE_SELECTION,
            'min_features': FEATURE_SELECTION_MIN_FEATURES if 'FEATURE_SELECTION_MIN_FEATURES' in globals() else 10,
            'max_features': FEATURE_SELECTION_MAX_FEATURES if 'FEATURE_SELECTION_MAX_FEATURES' in globals() else None,
            'shap_sample_size': FEATURE_SELECTION_SHAP_SAMPLE_SIZE if 'FEATURE_SELECTION_SHAP_SAMPLE_SIZE' in globals() else None,
            'shap_background_size': FEATURE_SELECTION_SHAP_BACKGROUND_SIZE if 'FEATURE_SELECTION_SHAP_BACKGROUND_SIZE' in globals() else None,
            'shap_tree_limit': FEATURE_SELECTION_SHAP_TREE_LIMIT if 'FEATURE_SELECTION_SHAP_TREE_LIMIT' in globals() else None,
            'cls_threshold': CLS_PRODUCTION_THRESHOLD if 'CLS_PRODUCTION_THRESHOLD' in globals() else CLS_THRESHOLD,
            'prediction_period': PREDICTION_PERIOD_DAYS,
            'enable_time_series_split': ENABLE_TIME_SERIES_SPLIT,
            'cv_n_splits': CV_N_SPLITS,
            'cv_embargo': CV_EMBARGO,
            'cv_allow_future': CV_ALLOW_FUTURE,
            'use_rolling_cv': False,
            'enable_sample_weighting': ENABLE_SAMPLE_WEIGHTING,
            'sample_weight_halflife': SAMPLE_WEIGHT_HALFLIFE_YEARS,
            'classification_metrics': CLASSIFICATION_METRICS,
            'regression_metrics': REGRESSION_METRICS,
            'top_k_values': TOP_K_VALUES,
            'reuse_feature_selection': FEATURE_SELECTION_CACHE_ENABLED,
            'force_refresh_feature_selection': FEATURE_SELECTION_CACHE_FORCE_REFRESH,
            'feature_selection_cache_dir': FEATURE_SELECTION_CACHE_DIR,
            'feature_selection_cache_ttl_days': FEATURE_SELECTION_CACHE_TTL_DAYS,
            'feature_selection_cache_version': FEATURE_SELECTION_CACHE_VERSION,
            'enable_regression_task': ENABLE_REGRESSION_TASK if 'ENABLE_REGRESSION_TASK' in globals() else True
        }

    def _get_feature_selection_cache_settings(self, task: str) -> Dict[str, Any]:
        """组装特征选择缓存设置。"""
        cache_dir = self.config.get('feature_selection_cache_dir')
        enabled = bool(self.config.get('reuse_feature_selection', False))
        if not cache_dir:
            enabled = False
        return {
            'enabled': enabled,
            'force_refresh': bool(self.config.get('force_refresh_feature_selection', False)),
            'cache_dir': cache_dir,
            'ttl_days': self.config.get('feature_selection_cache_ttl_days'),
            'version': self.config.get('feature_selection_cache_version', 'v1'),
            'method': self.config.get('feature_selection_method', 'lightgbm'),
            'threshold': self.config.get('feature_selection_threshold', 'median'),
            'min_features': self.config.get('min_features'),
            'max_features': self.config.get('max_features'),
            'task': task
        }

    def _build_feature_selection_cache_signature(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        settings: Dict[str, Any]
    ) -> str:
        """根据数据与配置生成缓存签名。"""
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y).reset_index(drop=True)
        else:
            y_series = y.reset_index(drop=True)
        row_fingerprint = hash_pandas_object(X, index=True).values.tobytes()
        target_fingerprint = hash_pandas_object(y_series, index=True).values.tobytes()
        meta = {
            'task': settings.get('task'),
            'method': settings.get('method'),
            'threshold': settings.get('threshold'),
            'min_features': settings.get('min_features'),
            'max_features': settings.get('max_features'),
            'version': settings.get('version'),
            'columns': list(X.columns)
        }
        hasher = hashlib.blake2b(digest_size=20)
        hasher.update(row_fingerprint)
        hasher.update(target_fingerprint)
        hasher.update(json.dumps(meta, sort_keys=True).encode('utf-8'))
        return hasher.hexdigest()

    def _load_feature_selection_cache(
        self,
        settings: Dict[str, Any],
        signature: str,
        X: pd.DataFrame
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """尝试从磁盘读取缓存的特征选择结果。"""
        cache_dir = settings.get('cache_dir')
        if not cache_dir:
            return None
        cache_path = Path(cache_dir) / f"{signature}.json"
        if not cache_path.exists():
            return None
        try:
            with cache_path.open('r', encoding='utf-8') as fp:
                payload = json.load(fp)
        except Exception as exc:
            logger.warning("特征选择缓存读取失败（%s）: %s", cache_path, exc)
            return None

        expected_version = settings.get('version')
        if expected_version and payload.get('version') != expected_version:
            logger.info("特征选择缓存版本不匹配（%s ≠ %s），忽略缓存", payload.get('version'), expected_version)
            return None

        cached_columns = payload.get('columns')
        if cached_columns != list(X.columns):
            logger.info("特征选择缓存列集合已变化，自动失效")
            return None

        ttl_days = settings.get('ttl_days')
        if ttl_days and ttl_days > 0:
            created_raw = payload.get('created_at')
            if created_raw:
                try:
                    created_dt = datetime.fromisoformat(created_raw.replace('Z', '+00:00'))
                except ValueError:
                    created_dt = None
                if created_dt:
                    if created_dt.tzinfo is not None:
                        created_dt_utc = created_dt.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        created_dt_utc = created_dt
                    if datetime.utcnow() - created_dt_utc > timedelta(days=int(ttl_days)):
                        logger.info("特征选择缓存已超过 %d 天，重新计算", ttl_days)
                        return None

        selected_features = payload.get('selected_features') or []
        importances_map = payload.get('importances') or {}
        importances = np.array([float(importances_map.get(col, 0.0)) for col in X.columns], dtype=float)

        logger.info("♻️ 复用特征选择缓存（%s）: %d → %d", signature[:12], len(X.columns), len(selected_features))
        return selected_features, importances

    def _save_feature_selection_cache(
        self,
        settings: Dict[str, Any],
        signature: str,
        X: pd.DataFrame,
        selected_features: List[str],
        importances: np.ndarray,
        importance_columns: Optional[List[str]] = None
    ) -> None:
        """将特征选择结果写入磁盘缓存。"""
        cache_dir = settings.get('cache_dir')
        if not cache_dir:
            return
        path = Path(cache_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("无法创建特征选择缓存目录 %s: %s", path, exc)
            return
        cache_path = path / f"{signature}.json"
        if importance_columns is None:
            importance_columns = list(X.columns[:len(importances)])
        else:
            importance_columns = list(importance_columns)

        if len(importance_columns) != len(importances):
            min_len = min(len(importance_columns), len(importances))
            logger.debug(
                "特征重要性列数与数值不一致，按最小长度对齐: columns=%d, importances=%d",
                len(importance_columns),
                len(importances)
            )
            importance_map = {
                col: float(importances[idx])
                for idx, col in enumerate(importance_columns[:min_len])
            }
        else:
            importance_map = {
                col: float(importances[idx])
                for idx, col in enumerate(importance_columns)
            }

        payload = {
            'version': settings.get('version'),
            'signature': signature,
            'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'columns': list(X.columns),
            'row_count': int(len(X)),
            'selected_features': list(selected_features),
            'importance_columns': importance_columns,
            'importances': importance_map,
            'meta': {
                'task': settings.get('task'),
                'method': settings.get('method'),
                'threshold': settings.get('threshold'),
                'min_features': settings.get('min_features'),
                'max_features': settings.get('max_features')
            }
        }
        tmp_path = cache_path.with_suffix('.json.tmp')
        try:
            with tmp_path.open('w', encoding='utf-8') as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            os.replace(tmp_path, cache_path)
            logger.info("💾 已缓存特征选择结果（%s）: %d → %d", signature[:12], len(X.columns), len(selected_features))
        except Exception as exc:
            logger.warning("写入特征选择缓存失败（%s）: %s", cache_path, exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    def _prepare_timeseries_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """按时间序列切分训练/验证集 (使用改进的时间分割策略)"""
        self.cv_fold_info = []
        self._cv_pairs = []
        self._cv_sorted_frames = None
        self._last_split_dates = {'train': None, 'val': None}
        
        if dates is None or not self.config.get('enable_time_series_split', False):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
            )
            self._last_split_dates = {'train': None, 'val': None}
            return X_train, X_val, y_train, y_val

        # 🔧 使用改进的时间序列分割 (5天禁用期, 连续时间窗口)
        from src.ml.training.toolkit import improved_time_series_split, rolling_window_split
        from config.prediction_config import (
            USE_ROLLING_WINDOW, 
            ROLLING_TRAIN_YEARS, 
            ROLLING_VAL_YEARS,
            ROLLING_EMBARGO_DAYS
        )
        
        try:
            # 根据配置选择切分方式
            if USE_ROLLING_WINDOW:
                X_train, X_val, y_train, y_val = rolling_window_split(
                    X=X,
                    y=y,
                    dates=dates,
                    train_years=ROLLING_TRAIN_YEARS,
                    val_years=ROLLING_VAL_YEARS,
                    embargo_days=ROLLING_EMBARGO_DAYS,
                    verbose=True
                )
            else:
                X_train, X_val, y_train, y_val = improved_time_series_split(
                    X=X,
                    y=y, 
                    dates=dates,
                    test_size_ratio=0.2,
                    embargo_days=5,  # 添加5天禁用期防止信息泄露
                    verbose=True
                )
            
            # 保存折叠信息用于日志
            dates_dt = pd.to_datetime(dates)
            self.cv_fold_info = [{
                'fold_id': 0,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'train_start': dates_dt.min(),
                'train_end': dates_dt.max(),
                'val_start': dates_dt.min(),
                'val_end': dates_dt.max(),
                'train_pos_rate': float(y_train.mean()),
                'val_pos_rate': float(y_val.mean())
            }]
            train_dates = None
            val_dates = None
            if isinstance(X_train, pd.DataFrame) and '_date' in X_train.columns:
                train_dates = pd.to_datetime(X_train['_date'])
                X_train = X_train.drop(columns=['_date'])
            if isinstance(X_val, pd.DataFrame) and '_date' in X_val.columns:
                val_dates = pd.to_datetime(X_val['_date'])
                X_val = X_val.drop(columns=['_date'])
            self._last_split_dates = {'train': train_dates, 'val': val_dates}
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.warning(f"改进的时间分割失败,回退到常规分割: {e}")
            # 回退到常规切分
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if (y.nunique() == 2 and len(y.unique()) > 1) else None
            )
            self._cv_pairs = []
            if isinstance(X_train, pd.DataFrame) and '_date' in X_train.columns:
                X_train = X_train.drop(columns=['_date'])
            if isinstance(X_val, pd.DataFrame) and '_date' in X_val.columns:
                X_val = X_val.drop(columns=['_date'])
            self._last_split_dates = {'train': None, 'val': None}
            return X_train, X_val, y_train, y_val

    @staticmethod
    def _compute_ks_score(y_true: pd.Series, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(np.abs(tpr - fpr))) if len(fpr) > 0 else np.nan

    @staticmethod
    def _compute_top_k_hit_rates(y_true: pd.Series, y_score: np.ndarray, ks: List[int]) -> Dict[int, float]:
        if len(y_true) == 0:
            return {k: np.nan for k in ks}
        order = np.argsort(-y_score)
        hits = {}
        y_array = y_true.to_numpy()
        for k in ks:
            k_eff = min(k, len(order))
            if k_eff == 0:
                hits[k] = np.nan
                continue
            top_hit = y_array[order[:k_eff]].mean()
            hits[k] = float(top_hit)
        return hits

    def _compute_time_decay_weights(self, dates: Optional[pd.Series]) -> Optional[np.ndarray]:
        if not self.enable_sample_weighting:
            return None
        if dates is None or len(dates) == 0:
            logger.warning("样本加权跳过：缺少日期信息")
            return None
        dates = pd.to_datetime(dates)
        if dates.isna().all():
            logger.warning("样本加权跳过：日期均为NaT")
            return None
        halflife_years = max(self.sample_weight_halflife, 1e-6)
        decay_rate = np.log(2) / (halflife_years * 365.0)
        max_date = dates.max()
        delta_days = (max_date - dates).dt.days.astype(float)
        weights = np.exp(-decay_rate * np.clip(delta_days, a_min=0.0, a_max=None))
        return weights.to_numpy(dtype=float)

    def _log_recent_window_metrics(
        self,
        y_true: pd.Series,
        scores: pd.Series,
        dates: Optional[pd.Series],
        task: str,
        window_days: int = 180
    ) -> None:
        if dates is None:
            logger.info("近期窗口评估跳过：缺少日期信息")
            return
        dates = pd.to_datetime(dates)
        if dates.empty:
            logger.info("近期窗口评估跳过：日期序列为空")
            return
        cutoff = dates.max() - pd.Timedelta(days=window_days)
        mask = dates >= cutoff
        if mask.sum() == 0:
            logger.info("近期窗口评估跳过：最近%d天无样本", window_days)
            return

        y_recent = y_true.reset_index(drop=True)[mask.values]
        scores_recent = scores.reset_index(drop=True)[mask.values]
        dates_recent = dates[mask]
        logger.info("近期窗口(%d天)样本: %d 日期范围: %s ~ %s",
                    window_days,
                    len(y_recent),
                    dates_recent.min().strftime('%Y-%m-%d'),
                    dates_recent.max().strftime('%Y-%m-%d'))

        if task == 'classification':
            from sklearn.metrics import roc_auc_score, brier_score_loss
            if y_recent.nunique() < 2:
                logger.info("  近期窗口分类样本不足以计算AUC")
            else:
                auc_recent = roc_auc_score(y_recent, scores_recent)
                brier_recent = brier_score_loss(y_recent, scores_recent)
                logger.info("  近期AUC=%.4f Brier=%.4f", auc_recent, brier_recent)
            top_k_values = self.config.get('top_k_values', [])
            if top_k_values:
                top_hits = self._compute_top_k_hit_rates(y_recent, scores_recent.to_numpy(), top_k_values)
                for k, hit in top_hits.items():
                    logger.info("  Top-%d 命中率=%.2f%%", k, hit * 100 if not np.isnan(hit) else float('nan'))
        else:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2_recent = float('nan')
            if y_recent.nunique() > 1:
                r2_recent = r2_score(y_recent, scores_recent)
            mae_recent = mean_absolute_error(y_recent, scores_recent)
            rmse_recent = np.sqrt(mean_squared_error(y_recent, scores_recent))
            logger.info("  近期R²=%.4f MAE=%.4f RMSE=%.4f", r2_recent, mae_recent, rmse_recent)

    @staticmethod
    def _find_optimal_threshold(y_true: pd.Series, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
        # 覆盖更细颗粒度并强制包含配置阈值，避免遗漏0.03等业务设定
        base_grid = np.linspace(0.01, 0.99, 99)
        thresholds = np.unique(np.concatenate([base_grid, np.array([CLS_PRODUCTION_THRESHOLD if 'CLS_PRODUCTION_THRESHOLD' in globals() else CLS_THRESHOLD])]))
        best_threshold = 0.5
        best_f1 = -1.0
        best_metrics = {'precision': np.nan, 'recall': np.nan, 'f1': np.nan}

        for thresh in thresholds:
            preds = (y_score >= thresh).astype(int)
            if preds.sum() == 0:
                continue
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)
                best_metrics = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }

        return best_threshold, best_metrics

    @staticmethod
    def _evaluate_threshold_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
        preds = (y_score >= threshold).astype(int)
        positive_rate = float(preds.mean())
        if preds.sum() == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'positive_rate': positive_rate
            }
        return {
            'precision': float(precision_score(y_true, preds, zero_division=0)),
            'recall': float(recall_score(y_true, preds, zero_division=0)),
            'f1': float(f1_score(y_true, preds, zero_division=0)),
            'positive_rate': positive_rate
        }

    @staticmethod
    def _derive_dynamic_threshold(
        scores: np.ndarray,
        target_rate: Optional[float],
        fallback: float,
        clip_range: Tuple[float, float] = (0.01, 0.99)
    ) -> Optional[float]:
        if scores is None or len(scores) == 0:
            return None
        if target_rate is None or not (0 < target_rate < 1):
            return None

        quantile_level = float(np.clip(1.0 - target_rate, 0.0, 1.0))
        threshold = float(np.quantile(scores, quantile_level))

        if np.isnan(threshold) or not math.isfinite(threshold):
            return None

        lo, hi = clip_range
        threshold = float(np.clip(threshold, lo, hi))

        return threshold if not math.isclose(threshold, fallback, rel_tol=1e-5, abs_tol=1e-5) else fallback

    def _log_fold_classification_diagnostics(
        self,
        model_wrapper: PreprocessedModel,
        calibrator: Optional[CalibratedClassifierCV],
        reference_threshold: float
    ) -> None:
        if not self._cv_pairs or self._cv_sorted_frames is None:
            return

        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

        X_sorted, y_sorted, dates_sorted = self._cv_sorted_frames
        fold_meta_map = {info['fold_id']: info for info in self.cv_fold_info} if self.cv_fold_info else {}
        record_map = {rec['fold_id']: rec for rec in self._cv_classification_records} if self._cv_classification_records else {}

        logger.info("折别回放诊断(分类):")
        for pair in self._cv_pairs:
            val_idx = pair['val_idx']
            if len(val_idx) == 0:
                continue
            fold_id = pair['fold_id']
            record = record_map.get(fold_id)
            if record is not None:
                y_true = record['y_true']
                proba_raw = record['y_pred'].to_numpy()
                proba_cal = None
            else:
                y_true = y_sorted.iloc[val_idx].reset_index(drop=True)
                X_val_fold = X_sorted.iloc[val_idx].reset_index(drop=True)
                try:
                    proba_raw = model_wrapper.predict_proba(X_val_fold)[:, 1]
                except Exception as exc:  # pragma: no cover - 诊断日志
                    logger.warning("  折%02d 预测失败: %s", fold_id, exc)
                    continue
                proba_cal = None
                if calibrator is not None:
                    try:
                        proba_cal = calibrator.predict_proba(X_val_fold)[:, 1]
                    except Exception as exc:  # pragma: no cover
                        logger.warning("  折%02d 校准预测失败: %s", fold_id, exc)
                        proba_cal = None
            start_dt = dates_sorted.iloc[val_idx[0]].strftime('%Y-%m-%d')
            end_dt = dates_sorted.iloc[val_idx[-1]].strftime('%Y-%m-%d')

            auc_raw = float('nan')
            auc_cal = float('nan')
            if y_true.nunique() > 1:
                auc_raw = float(roc_auc_score(y_true, proba_raw))
                if proba_cal is not None:
                    auc_cal = float(roc_auc_score(y_true, proba_cal))

            brier_raw = float(brier_score_loss(y_true, proba_raw))
            brier_cal = float('nan')
            if proba_cal is not None:
                brier_cal = float(brier_score_loss(y_true, proba_cal))

            cfg_threshold = self.config.get('cls_threshold', 0.5)
            preds_cfg = (proba_raw >= cfg_threshold).astype(int)
            precision_cfg = precision_score(y_true, preds_cfg, zero_division=0)
            recall_cfg = recall_score(y_true, preds_cfg, zero_division=0)
            f1_cfg = f1_score(y_true, preds_cfg, zero_division=0)
            pos_rate_cfg = float(preds_cfg.mean())

            preds_ref = (proba_raw >= reference_threshold).astype(int)
            precision_ref = precision_score(y_true, preds_ref, zero_division=0)
            recall_ref = recall_score(y_true, preds_ref, zero_division=0)
            f1_ref = f1_score(y_true, preds_ref, zero_division=0)
            pos_rate_ref = float(preds_ref.mean())

            meta = fold_meta_map.get(fold_id, {})
            logger.info(
                "  折%02d 验证[%s ~ %s] 样本%5d 正样本率%.2f%% | AUC(raw)=%.4f AUC(cal)=%.4f | Brier(raw)=%.4f Brier(cal)=%.4f | F1@阈值(%.3f)=%.4f(预测占比%.2f%%) F1@最优%.3f=%.4f(预测占比%.2f%%)",
                fold_id,
                start_dt,
                end_dt,
                len(val_idx),
                meta.get('val_pos_rate', y_true.mean()) * 100,
                auc_raw,
                auc_cal,
                brier_raw,
                brier_cal,
                cfg_threshold,
                f1_cfg,
                pos_rate_cfg * 100,
                reference_threshold,
                f1_ref,
                pos_rate_ref * 100
            )

    def _log_fold_regression_diagnostics(
        self,
        model_wrapper: PreprocessedModel
    ) -> None:
        if not self._cv_pairs or self._cv_sorted_frames is None:
            return

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import math

        X_sorted, y_sorted, dates_sorted = self._cv_sorted_frames
        fold_meta_map = {info['fold_id']: info for info in self.cv_fold_info} if self.cv_fold_info else {}
        record_map = {rec['fold_id']: rec for rec in self._cv_regression_records} if self._cv_regression_records else {}

        logger.info("折别回放诊断(回归):")
        for pair in self._cv_pairs:
            val_idx = pair['val_idx']
            if len(val_idx) == 0:
                continue
            fold_id = pair['fold_id']
            record = record_map.get(fold_id)
            if record is not None:
                y_true = record['y_true']
                preds = record['y_pred'].to_numpy()
            else:
                y_true = y_sorted.iloc[val_idx].reset_index(drop=True)
                X_val_fold = X_sorted.iloc[val_idx].reset_index(drop=True)
                try:
                    preds = model_wrapper.predict(X_val_fold)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  折%02d 回归预测失败: %s", fold_id, exc)
                    continue
            start_dt = dates_sorted.iloc[val_idx[0]].strftime('%Y-%m-%d')
            end_dt = dates_sorted.iloc[val_idx[-1]].strftime('%Y-%m-%d')

            r2 = float('nan')
            if y_true.nunique() > 1:
                r2 = float(r2_score(y_true, preds))
            mae = float(mean_absolute_error(y_true, preds))
            rmse = float(math.sqrt(mean_squared_error(y_true, preds)))

            meta = fold_meta_map.get(fold_id, {})
            logger.info(
                "  折%02d 验证[%s ~ %s] 样本%5d 均值%.4f | R²=%.4f MAE=%.4f RMSE=%.4f",
                fold_id,
                start_dt,
                end_dt,
                len(val_idx),
                y_true.mean(),
                r2,
                mae,
                rmse
            )

    def _train_xgboost_classifier(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        early_stopping_rounds: Optional[int],
        model_params: Dict[str, Any],
        sample_weight: Optional[np.ndarray] = None
    ) -> XGBoostClassifierWrapper:
        import xgboost as xgb

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': model_params.get('learning_rate', 0.05),
            'max_depth': model_params.get('max_depth', 5),
            'subsample': model_params.get('subsample', 0.8),
            'colsample_bytree': model_params.get('colsample_bytree', 0.8),
            'reg_alpha': model_params.get('reg_alpha', 0.1),
            'reg_lambda': model_params.get('reg_lambda', 1.0),
            'tree_method': model_params.get('tree_method', 'hist'),
            'seed': 42
        }

        dtrain = xgb.DMatrix(X_train, label=y_train.to_numpy(), weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val.to_numpy())
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                maximize=True,
                data_name='validation',
                metric_name='auc',
                save_best=True
            ))

        num_boost_round = model_params.get('n_estimators', 800)
        evals = [(dtrain, 'train'), (dval, 'validation')]
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False
        )

        best_iteration = booster.best_iteration
        if best_iteration is None:
            best_iteration = num_boost_round - 1

        logger.info(
            "XGBoost 分类训练完成: best_iteration=%d, best_score=%.4f",
            best_iteration,
            float(booster.best_score) if booster.best_score is not None else float('nan')
        )

        estimator = XGBoostClassifierWrapper(booster, best_iteration, X_train.shape[1])
        return estimator

    def _train_xgboost_regressor(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        early_stopping_rounds: Optional[int],
        model_params: Dict[str, Any]
    ) -> XGBoostRegressorWrapper:
        import xgboost as xgb

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': model_params.get('learning_rate', 0.05),
            'max_depth': model_params.get('max_depth', 5),
            'subsample': model_params.get('subsample', 0.8),
            'colsample_bytree': model_params.get('colsample_bytree', 0.8),
            'reg_alpha': model_params.get('reg_alpha', 0.1),
            'reg_lambda': model_params.get('reg_lambda', 1.0),
            'tree_method': model_params.get('tree_method', 'hist'),
            'seed': 42
        }

        dtrain = xgb.DMatrix(X_train, label=y_train.to_numpy())
        dval = xgb.DMatrix(X_val, label=y_val.to_numpy())
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                maximize=False,
                data_name='validation',
                metric_name='rmse',
                save_best=True
            ))

        num_boost_round = model_params.get('n_estimators', 800)
        evals = [(dtrain, 'train'), (dval, 'validation')]
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False
        )

        best_iteration = booster.best_iteration
        if best_iteration is None:
            best_iteration = num_boost_round - 1

        logger.info(
            "XGBoost 回归训练完成: best_iteration=%d, best_score=%.4f",
            best_iteration,
            float(booster.best_score) if booster.best_score is not None else float('nan')
        )

        estimator = XGBoostRegressorWrapper(booster, best_iteration, X_train.shape[1])
        return estimator
    
    def create_preprocessing_pipeline(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> ColumnTransformer:
        """
        创建统一预处理管道

        Parameters
        ----------
        numerical_features : List[str], optional
            本次训练使用的数值特征列表
        categorical_features : List[str], optional
            本次训练使用的类别特征列表

        Returns
        -------
        preprocessor : ColumnTransformer
            预处理器
        """

        if numerical_features is None:
            numerical_features = list(self._base_numerical_features)
        else:
            numerical_features = list(numerical_features)

        if categorical_features is None:
            categorical_features = list(self._base_categorical_features)
        else:
            categorical_features = list(categorical_features)

        # 数值特征 pipeline
        numerical_pipe = Pipeline([
            ('imputer', SafeSimpleImputer(strategy='median', fallback_value=0.0)),
            ('winsorizer', Winsorizer(
                quantile_range=(
                    self.config['winsor_clip_quantile'],
                    1 - self.config['winsor_clip_quantile']
                )
            )),
            ('scaler', RobustScaler(quantile_range=(5, 95)))
        ])

        # 类别特征 pipeline
        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        transformers: List[Tuple[str, Pipeline, List[str]]] = []
        if numerical_features:
            transformers.append(('num', numerical_pipe, numerical_features))
        if categorical_features:
            transformers.append(('cat', categorical_pipe, categorical_features))

        if not transformers:
            preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
        else:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        logger.info("创建预处理管道:")
        if numerical_features:
            logger.info("  数值管道: Imputer → Winsorizer → Scaler (%d 列)", len(numerical_features))
        if categorical_features:
            logger.info("  类别管道: Imputer → OneHot (%d 列)", len(categorical_features))
        if not categorical_features:
            logger.info("  类别特征为空，跳过OneHot")
        logger.info("  提示: 行业与板块在OneHot后会作为稠密列输入模型，跳过的是原始列级特征选择")

        return preprocessor
    
    def _select_features_for_task(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification'
    ) -> Tuple[List[str], np.ndarray]:
        """
        执行特征选择
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            标签数据
        task : str
            任务类型 ('classification' 或 'regression')
        
        Returns
        -------
        selected_features : List[str]
            选定的特征名列表
        importances : ndarray
            特征重要性
        """
        logger.info(f"执行{task}特征选择...")
        # 🔧 关键修复: 确保数据类型正确
        X_converted = X.copy()
        for col in X_converted.columns:
            if X_converted[col].dtype == 'object':
                # 仅在实际存在数值时才转换，避免行业等纯分类列被强制转换为全NaN
                converted = pd.to_numeric(X_converted[col], errors='coerce')
                if converted.notna().any():
                    X_converted[col] = converted
                else:
                    logger.debug("特征选择保留原始字符串列 %s（未检测到可转换的数值数据）", col)

        logger.debug("特征数据类型: %s", {name: str(dtype) for name, dtype in X_converted.dtypes.items()})

        cache_settings = self._get_feature_selection_cache_settings(task)
        cache_signature: Optional[str] = None
        if cache_settings.get('enabled'):
            try:
                cache_signature = self._build_feature_selection_cache_signature(X_converted, y, cache_settings)
            except Exception as exc:
                logger.warning("生成特征选择缓存签名失败，改为重新计算: %s", exc)
                cache_signature = None
            if cache_signature and cache_settings.get('force_refresh'):
                logger.info("⟳ 忽略历史特征选择缓存并强制重算（%s）", cache_signature[:12])
            if cache_signature and not cache_settings.get('force_refresh'):
                cached = self._load_feature_selection_cache(cache_settings, cache_signature, X_converted)
                if cached is not None:
                    return cached

        shap_kwargs: Dict[str, Any] = {}
        if self.config.get('feature_selection_method', 'lightgbm') == 'shap':
            if self.config.get('shap_sample_size'):
                shap_kwargs['shap_sample_size'] = self.config.get('shap_sample_size')
            if self.config.get('shap_background_size'):
                shap_kwargs['shap_background_size'] = self.config.get('shap_background_size')
            if self.config.get('shap_tree_limit') is not None:
                shap_kwargs['shap_tree_limit'] = self.config.get('shap_tree_limit')

        selected_features, importances, importance_feature_names = select_features_for_task(
            X_converted, y,
            task=task,
            method=self.config.get('feature_selection_method', 'lightgbm'),
            threshold=self.config.get('feature_selection_threshold', 'median'),
            min_features=self.config.get('min_features', FEATURE_SELECTION_MIN_FEATURES if 'FEATURE_SELECTION_MIN_FEATURES' in globals() else 15),
            max_features=self.config.get('max_features', FEATURE_SELECTION_MAX_FEATURES if 'FEATURE_SELECTION_MAX_FEATURES' in globals() else 30),
            cv_n_jobs=self.config.get('cv_n_jobs', -1),
            **shap_kwargs
        )

        if task == 'classification':
            selected_features = self._append_categorical_features(selected_features, X, context="特征选择")

        logger.info(f"✅ 特征选择完成: {len(X.columns)} → {len(selected_features)}")

        importances_array = np.asarray(importances, dtype=float)
        if cache_settings.get('enabled') and cache_signature:
            self._save_feature_selection_cache(
                cache_settings,
                cache_signature,
                X_converted,
                selected_features,
                importances_array,
                list(importance_feature_names)
            )

        return selected_features, importances_array

    def _split_feature_types(self, selected_features: List[str]) -> Tuple[List[str], List[str]]:
        """根据初始特征集，将选定特征划分为数值和类别两类。"""
        numeric = [f for f in self._base_numerical_features if f in selected_features]
        categorical = [f for f in self._base_categorical_features if f in selected_features]
        return numeric, categorical

    def _append_categorical_features(self, selected: List[str], X: pd.DataFrame, context: str = "") -> List[str]:
        appended: List[str] = []
        for cat in self._base_categorical_features:
            if cat in X.columns and cat not in selected:
                selected.append(cat)
                appended.append(cat)
        if appended:
            msg = f"附加类别特征({context})" if context else "附加类别特征"
            logger.info("%s: %s", msg, appended)
        # 去重保持顺序
        selected = list(dict.fromkeys(selected))
        return selected

    def _analyze_data_complexity(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        分析数据复杂度，为参数调整提供依据
        
        Parameters
        ----------
        X : DataFrame or ndarray
            特征数据
        y : Series or ndarray
            标签数据
            
        Returns
        -------
        metrics : dict
            包含以下指标的字典：
            - n_samples: 样本数量
            - n_features: 特征数量
            - feature_density: 特征密度 (非零特征比例)
            - target_variance: 目标变量方差
            - complexity_score: 综合复杂度评分
        """
        # 从配置获取阈值
        small_sample_threshold = self.config.get('small_sample_threshold', SMALL_SAMPLE_THRESHOLD)
        large_sample_threshold = self.config.get('large_sample_threshold', LARGE_SAMPLE_THRESHOLD)
        high_dimension_threshold = self.config.get('high_dimension_threshold', HIGH_DIMENSION_THRESHOLD)
        high_complexity_threshold = self.config.get('high_complexity_threshold', HIGH_COMPLEXITY_THRESHOLD)
        
        # 处理不同的输入类型
        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
        else:
            n_samples, n_features = X.shape
        
        # 计算特征密度 (非零特征比例)
        if hasattr(X, 'sparse'):
            feature_density = (X != 0).sum().sum() / (n_samples * n_features)
        else:
            feature_density = 1.0  # 稠密矩阵
        
        # 目标变量方差
        if isinstance(y, pd.Series):
            target_variance = y.var() if len(y) > 1 else 0.0
        else:
            target_variance = np.var(y) if len(y) > 1 else 0.0
        
        # 综合复杂度评分 (0-1范围)，使用配置阈值
        complexity_score = min(1.0, 
            (n_samples / large_sample_threshold) * 0.3 +  # 样本规模影响
            (n_features / high_dimension_threshold) * 0.3 +   # 特征规模影响
            feature_density * 0.2 +       # 特征密度影响
            (target_variance / 10) * 0.2  # 目标复杂度影响
        )
        
        metrics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_density': feature_density,
            'target_variance': target_variance,
            'complexity_score': complexity_score,
            'small_sample_threshold': small_sample_threshold,
            'large_sample_threshold': large_sample_threshold,
            'high_dimension_threshold': high_dimension_threshold,
            'high_complexity_threshold': high_complexity_threshold
        }
        
        logger.info(f"📊 数据复杂度分析: {n_samples}样本, {n_features}特征, "
                   f"密度{feature_density:.3f}, 复杂度评分{complexity_score:.3f}")
        
        return metrics

    def _adjust_parameter_ranges(self, complexity_metrics: Dict[str, float], task: str, model_type: str) -> Dict[str, Any]:
        """
        基于数据复杂度动态调整参数范围
        
        Parameters
        ----------
        complexity_metrics : dict
            数据复杂度指标
        task : str
            任务类型 ('classification' 或 'regression')
        model_type : str
            模型类型 ('xgboost', 'lightgbm', 'logistic', 'random_forest')
            
        Returns
        -------
        adjusted_params : dict
            调整后的参数范围
        """
        n_samples = complexity_metrics['n_samples']
        n_features = complexity_metrics['n_features']
        complexity_score = complexity_metrics['complexity_score']
        small_sample_threshold = complexity_metrics['small_sample_threshold']
        large_sample_threshold = complexity_metrics['large_sample_threshold']
        high_dimension_threshold = complexity_metrics['high_dimension_threshold']
        high_complexity_threshold = complexity_metrics['high_complexity_threshold']
        
        adjusted_params = {}
        
        # 根据调整模式选择策略
        if self.data_driven_adjustment_mode == 'conservative':
            # 保守模式：使用更窄的参数范围
            return self._get_conservative_params(model_type, task)
        
        # 自适应模式：基于数据复杂度动态调整
        # 根据模型类型和任务类型调整参数
        if model_type == 'xgboost':
            if task == 'classification':
                # 基础参数范围
                base_params = {
                    'n_estimators': (50, 1000),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_weight': (1, 10),
                    'gamma': (0, 1.0),
                    'reg_alpha': (0, 1.0),
                    'reg_lambda': (0, 1.0)
                }
                
                # 基于数据复杂度调整
                if n_samples < small_sample_threshold:
                    # 小样本：减少树数量和深度
                    base_params['n_estimators'] = (30, 300)
                    base_params['max_depth'] = (2, 6)
                    base_params['learning_rate'] = (0.05, 0.2)
                elif n_samples > large_sample_threshold:
                    # 大样本：增加树数量和深度
                    base_params['n_estimators'] = (200, 2000)
                    base_params['max_depth'] = (5, 15)
                
                if n_features >= high_dimension_threshold:
                    # 高维特征：加强正则化
                    base_params['reg_alpha'] = (0.1, 2.0)
                    base_params['reg_lambda'] = (0.1, 2.0)
                    base_params['colsample_bytree'] = (0.3, 0.8)
                
                # 基于复杂度评分微调
                if complexity_score > high_complexity_threshold:
                    # 高复杂度数据：加强正则化
                    base_params['reg_alpha'] = (base_params['reg_alpha'][0] * 1.5, base_params['reg_alpha'][1] * 1.5)
                    base_params['reg_lambda'] = (base_params['reg_lambda'][0] * 1.5, base_params['reg_lambda'][1] * 1.5)
                
                adjusted_params = base_params
                
            elif task == 'regression':
                # 回归任务参数调整
                base_params = {
                    'n_estimators': (50, 1000),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.2),
                    'subsample': (0.7, 1.0),
                    'colsample_bytree': (0.7, 1.0),
                    'min_child_weight': (1, 20),
                    'gamma': (0, 0.5),
                    'reg_alpha': (0, 2.0),
                    'reg_lambda': (0, 2.0)
                }
                
                # 回归任务特有的调整
                if n_samples < small_sample_threshold // 2:  # 回归任务对样本量更敏感
                    base_params['n_estimators'] = (30, 200)
                    base_params['max_depth'] = (2, 5)
                
                if n_features > high_dimension_threshold * 0.6:  # 回归任务对特征维度更敏感
                    base_params['colsample_bytree'] = (0.5, 0.9)
                
                adjusted_params = base_params
        
        elif model_type == 'lightgbm':
            # LightGBM参数调整
            base_params = {
                'n_estimators': (50, 1000),
                'num_leaves': (20, 150),
                'max_depth': (3, 12),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_samples': (10, 100),
                'reg_alpha': (0, 1.0),
                'reg_lambda': (0, 1.0)
            }
            
            # 基于数据复杂度调整
            if n_samples < small_sample_threshold:
                base_params['num_leaves'] = (10, 50)
                base_params['max_depth'] = (2, 6)
            
            if n_features > high_dimension_threshold:
                base_params['colsample_bytree'] = (0.4, 0.8)
            
            adjusted_params = base_params
        
        logger.info(f"🔧 参数范围调整: {model_type} {task}, "
                   f"样本{n_samples}, 特征{n_features}, 复杂度{complexity_score:.3f}")
        
        return adjusted_params
    
    def _get_conservative_params(self, model_type: str, task: str) -> Dict[str, Any]:
        """
        获取保守模式的参数范围
        
        Parameters
        ----------
        model_type : str
            模型类型
        task : str
            任务类型
            
        Returns
        -------
        params : dict
            保守参数范围
        """
        if model_type == 'xgboost':
            if task == 'classification':
                return {
                    'n_estimators': (100, 500),
                    'max_depth': (3, 8),
                    'learning_rate': (0.05, 0.2),
                    'subsample': (0.7, 1.0),
                    'colsample_bytree': (0.7, 1.0),
                    'min_child_weight': (1, 5),
                    'gamma': (0, 0.5),
                    'reg_alpha': (0, 0.5),
                    'reg_lambda': (0, 0.5)
                }
            else:  # regression
                return {
                    'n_estimators': (100, 500),
                    'max_depth': (3, 8),
                    'learning_rate': (0.05, 0.15),
                    'subsample': (0.8, 1.0),
                    'colsample_bytree': (0.8, 1.0),
                    'min_child_weight': (1, 10),
                    'gamma': (0, 0.3),
                    'reg_alpha': (0, 1.0),
                    'reg_lambda': (0, 1.0)
                }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': (100, 500),
                'num_leaves': (31, 100),
                'max_depth': (3, 8),
                'learning_rate': (0.05, 0.2),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'min_child_samples': (20, 50),
                'reg_alpha': (0, 0.5),
                'reg_lambda': (0, 0.5)
            }
        else:
            # 对于其他模型类型，返回空字典
            return {}

    def _build_cv_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[Dict[str, Any]]]:
        """对齐时间顺序并生成时间序列交叉验证折。"""

        if dates is not None:
            dates_series = pd.to_datetime(dates)
        else:
            dates_series = pd.Series(np.arange(len(X)), name='date')

        order = dates_series.argsort()
        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)
        dates_sorted = dates_series.iloc[order].reset_index(drop=True)

        n_splits = self.config.get('cv_n_splits', 5)
        n_splits = max(2, min(n_splits, len(X_sorted) - 1))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds: List[Dict[str, Any]] = []

        embargo_days = self.config.get('cv_embargo', 0)
        use_rolling = self.config.get('use_rolling_cv', False)

        if use_rolling:
            from config.prediction_config import ROLLING_TRAIN_YEARS
            rolling_years = ROLLING_TRAIN_YEARS
        else:
            rolling_years = None

        for fold_id, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
            train_idx = np.asarray(train_idx, dtype=int)
            val_idx = np.asarray(val_idx, dtype=int)

            if len(val_idx) == 0:
                continue

            # 应用embargo，移除距离验证集过近的训练样本
            if embargo_days and len(train_idx) > 0:
                cutoff_date = dates_sorted.iloc[val_idx[0]] - pd.Timedelta(days=embargo_days)
                mask = dates_sorted.iloc[train_idx] <= cutoff_date
                train_idx = train_idx[mask.to_numpy()]

            # 滚动窗口：限定训练数据仅使用最近窗口
            if rolling_years is not None and len(train_idx) > 0:
                train_start_cut = dates_sorted.iloc[val_idx[0]] - pd.DateOffset(years=rolling_years)
                mask = dates_sorted.iloc[train_idx] >= train_start_cut
                train_idx = train_idx[mask.to_numpy()]

            if len(train_idx) == 0:
                logger.warning("CV折%d 训练样本为空，跳过该折", fold_id)
                continue

            fold = {
                'fold_id': fold_id,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_start': dates_sorted.iloc[train_idx[0]],
                'train_end': dates_sorted.iloc[train_idx[-1]],
                'val_start': dates_sorted.iloc[val_idx[0]],
                'val_end': dates_sorted.iloc[val_idx[-1]]
            }

            folds.append(fold)

        if not folds:
            raise ValueError("时间序列交叉验证折生成失败，请检查样本量或配置")

        return X_sorted, y_sorted, dates_sorted, folds

    def _store_cv_metadata(
        self,
        X_sorted: pd.DataFrame,
        y_sorted: pd.Series,
        dates_sorted: pd.Series,
        folds: List[Dict[str, Any]],
        task: str
    ) -> None:
        """记录CV数据用于后续诊断日志。"""

        self._cv_sorted_frames = (X_sorted, y_sorted, dates_sorted)
        self._cv_pairs = [
            {
                'fold_id': fold['fold_id'],
                'train_idx': fold['train_idx'],
                'val_idx': fold['val_idx']
            }
            for fold in folds
        ]

        fold_infos: List[Dict[str, Any]] = []
        is_classification = (task == 'classification')

        for fold in folds:
            train_idx = fold['train_idx']
            val_idx = fold['val_idx']
            info = {
                'fold_id': fold['fold_id'],
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_start': fold['train_start'],
                'train_end': fold['train_end'],
                'val_start': fold['val_start'],
                'val_end': fold['val_end']
            }

            if is_classification:
                info['train_pos_rate'] = float(y_sorted.iloc[train_idx].mean())
                info['val_pos_rate'] = float(y_sorted.iloc[val_idx].mean())

            fold_infos.append(info)

        self.cv_fold_info = fold_infos
    
    def _optuna_optimize(
        self,
        X_sorted: pd.DataFrame,
        y_sorted: pd.Series,
        dates_sorted: pd.Series,
        folds: List[Dict[str, Any]],
        task: str,
        model_type: str
    ) -> Dict[str, Any]:
        """在给定的时间序列折上执行Optuna调参（含折内特征选择）。"""

        if not self.enable_optuna:
            return {}

        supported_cls = {'xgboost', 'lightgbm'}
        supported_reg = {'xgboost'}
        if task == 'classification' and model_type not in supported_cls:
            logger.warning("Optuna暂不支持 %s 分类模型，跳过", model_type)
            return {}
        if task == 'regression' and model_type not in supported_reg:
            logger.warning("Optuna暂不支持 %s 回归模型，跳过", model_type)
            return {}

        logger.info("执行%s Optuna超参数优化 (模型=%s)...", task, model_type)

        import optuna
        from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
        from optuna.pruners import MedianPruner

        sampler_type = self.config.get('optuna_sampler', 'tpe').lower()
        if sampler_type == 'tpe':
            sampler = TPESampler(seed=42)
        elif sampler_type == 'random':
            sampler = RandomSampler(seed=42)
        elif sampler_type == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            logger.warning("未知的Optuna采样器 %s，回退至TPE", sampler_type)
            sampler = TPESampler(seed=42)

        pruner = MedianPruner(
            n_startup_trials=self.config.get('optuna_patience', 10),
            n_warmup_steps=5
        )

        direction = 'maximize'
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        # 数据驱动参数调整：分析训练数据的复杂度
        complexity_metrics = self._analyze_data_complexity(X_sorted, y_sorted)
        adjusted_param_ranges = self._adjust_parameter_ranges(complexity_metrics, task, model_type)

        def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
            # 使用数据驱动调整后的参数范围
            if model_type == 'xgboost':
                # 获取调整后的参数范围，如果未调整则使用默认范围
                n_estimators_range = adjusted_param_ranges.get('n_estimators', (50, 1000))
                max_depth_range = adjusted_param_ranges.get('max_depth', (3, 10))
                learning_rate_range = adjusted_param_ranges.get('learning_rate', (0.01, 0.3))
                subsample_range = adjusted_param_ranges.get('subsample', (0.6, 1.0))
                colsample_bytree_range = adjusted_param_ranges.get('colsample_bytree', (0.6, 1.0))
                min_child_weight_range = adjusted_param_ranges.get('min_child_weight', (1, 10))
                gamma_range = adjusted_param_ranges.get('gamma', (0, 1.0))
                reg_alpha_range = adjusted_param_ranges.get('reg_alpha', (0, 1.0))
                reg_lambda_range = adjusted_param_ranges.get('reg_lambda', (0, 1.0))
                
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1]),
                    'max_depth': trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1]),
                    'learning_rate': trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1], log=True),
                    'subsample': trial.suggest_float('subsample', subsample_range[0], subsample_range[1]),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', colsample_bytree_range[0], colsample_bytree_range[1]),
                    'min_child_weight': trial.suggest_int('min_child_weight', min_child_weight_range[0], min_child_weight_range[1]),
                    'gamma': trial.suggest_float('gamma', gamma_range[0], gamma_range[1]),
                    'reg_alpha': trial.suggest_float('reg_alpha', reg_alpha_range[0], reg_alpha_range[1]),
                    'reg_lambda': trial.suggest_float('reg_lambda', reg_lambda_range[0], reg_lambda_range[1]),
                    'max_leaves': trial.suggest_int('max_leaves', 15, 255),
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbosity': 0
                }
                if task == 'classification':
                    params['eval_metric'] = 'auc'
                    params['use_label_encoder'] = False
                else:
                    params['eval_metric'] = 'rmse'
                return params

            # LightGBM 分类
            elif model_type == 'lightgbm':
                # 获取调整后的参数范围
                n_estimators_range = adjusted_param_ranges.get('n_estimators', (50, 1000))
                num_leaves_range = adjusted_param_ranges.get('num_leaves', (20, 150))
                max_depth_range = adjusted_param_ranges.get('max_depth', (3, 12))
                learning_rate_range = adjusted_param_ranges.get('learning_rate', (0.01, 0.3))
                subsample_range = adjusted_param_ranges.get('subsample', (0.6, 1.0))
                colsample_bytree_range = adjusted_param_ranges.get('colsample_bytree', (0.6, 1.0))
                min_child_samples_range = adjusted_param_ranges.get('min_child_samples', (10, 100))
                reg_alpha_range = adjusted_param_ranges.get('reg_alpha', (0, 1.0))
                reg_lambda_range = adjusted_param_ranges.get('reg_lambda', (0, 1.0))
                
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1]),
                    'num_leaves': trial.suggest_int('num_leaves', num_leaves_range[0], num_leaves_range[1]),
                    'max_depth': trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1]),
                    'learning_rate': trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1], log=True),
                    'subsample': trial.suggest_float('subsample', subsample_range[0], subsample_range[1]),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', colsample_bytree_range[0], colsample_bytree_range[1]),
                    'min_child_samples': trial.suggest_int('min_child_samples', min_child_samples_range[0], min_child_samples_range[1]),
                    'reg_alpha': trial.suggest_float('reg_alpha', reg_alpha_range[0], reg_alpha_range[1]),
                    'reg_lambda': trial.suggest_float('reg_lambda', reg_lambda_range[0], reg_lambda_range[1]),
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbosity': -1
                }
                return params
            
            else:
                # 对于不支持的模型类型，返回空参数
                return {}

        metric_name = 'auc' if task == 'classification' else 'r2'

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            scores: List[float] = []

            for fold in folds:
                train_idx = fold['train_idx']
                val_idx = fold['val_idx']

                X_train = X_sorted.iloc[train_idx]
                y_train = y_sorted.iloc[train_idx]
                X_val = X_sorted.iloc[val_idx]
                y_val = y_sorted.iloc[val_idx]

                if task == 'classification' and y_train.nunique() < 2:
                    continue

                selected_features, _ = self._select_features_for_task(X_train, y_train, task)
                if not selected_features:
                    continue

                num_feats, cat_feats = self._split_feature_types(selected_features)
                preprocessor = self.create_preprocessing_pipeline(num_feats, cat_feats)
                preprocessor.fit(X_train[selected_features])

                X_train_trans = preprocessor.transform(X_train[selected_features])
                X_val_trans = preprocessor.transform(X_val[selected_features])

                estimator = self._create_estimator(model_type, task, params)
                estimator.fit(X_train_trans, y_train)

                if task == 'classification':
                    if y_val.nunique() < 2:
                        continue
                    if hasattr(estimator, 'predict_proba'):
                        y_pred = estimator.predict_proba(X_val_trans)[:, 1]
                    else:
                        y_pred = estimator.decision_function(X_val_trans)
                    from sklearn.metrics import roc_auc_score
                    try:
                        score = roc_auc_score(y_val, y_pred)
                    except ValueError:
                        continue
                else:
                    preds = estimator.predict(X_val_trans)
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, preds)

                scores.append(float(score))
                trial.report(float(score), fold['fold_id'])
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not scores:
                return float('-inf')

            return float(np.mean(scores))

        study.optimize(
            objective,
            n_trials=self.config.get('optuna_trials', 100),
            timeout=self.config.get('optuna_timeout', 3600),
            n_jobs=1,
            show_progress_bar=False
        )

        if study.best_trial is None:
            logger.warning("Optuna未找到有效试验，返回空参数")
            return {}

        logger.info("✅ Optuna优化完成: 最优%s=%.4f", metric_name.upper(), study.best_value)
        logger.info("  试验次数: %d", len(study.trials))

        return study.best_params

    def _create_estimator(self, model_type: str, task: str, params: Dict[str, Any]):
        """根据任务类型创建基础模型实例。"""
        params = params.copy()

        early_stopping_rounds = params.pop('early_stopping_rounds', 30)
        estimator = None

        if model_type == 'xgboost':
            import xgboost as xgb
            params.setdefault('random_state', 42)
            params.setdefault('n_jobs', 1)
            params.setdefault('verbosity', 0)
            if task == 'classification':
                params.setdefault('objective', 'binary:logistic')
                params.setdefault('eval_metric', 'auc')
                params.setdefault('use_label_encoder', False)
                estimator = xgb.XGBClassifier(**params)
            else:
                params.setdefault('objective', 'reg:squarederror')
                params.setdefault('eval_metric', 'rmse')
                estimator = xgb.XGBRegressor(**params)
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            params.setdefault('random_state', 42)
            params.setdefault('n_jobs', 1)
            params.setdefault('verbosity', -1)
            if task == 'classification':
                estimator = lgb.LGBMClassifier(**params)
            else:
                estimator = lgb.LGBMRegressor(**params)
        elif model_type == 'logistic' and task == 'classification':
            from sklearn.linear_model import LogisticRegression

            allowed_keys = {'C', 'max_iter', 'penalty', 'solver', 'l1_ratio', 'fit_intercept', 'class_weight', 'tol'}
            lr_params = {k: params.pop(k) for k in list(params.keys()) if k in allowed_keys}
            lr_params.setdefault('max_iter', 1000)
            lr_params.setdefault('solver', 'lbfgs')
            lr_params.setdefault('penalty', 'l2')
            lr_params.setdefault('C', 1.0)
            lr_params.setdefault('class_weight', 'balanced')
            estimator = LogisticRegression(random_state=42, **lr_params)
            early_stopping_rounds = None
        elif model_type == 'ridge' and task == 'regression':
            from sklearn.linear_model import Ridge
            allowed_keys = {'alpha', 'solver', 'tol', 'max_iter'}
            ridge_params = {k: params.pop(k) for k in list(params.keys()) if k in allowed_keys}
            ridge_params.setdefault('alpha', 1.0)
            estimator = Ridge(**ridge_params)
            early_stopping_rounds = None
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        if early_stopping_rounds and hasattr(estimator, 'set_params'):
            setattr(estimator, '_codex_early_stopping_rounds', early_stopping_rounds)
        else:
            setattr(estimator, '_codex_early_stopping_rounds', None)

        return estimator
        

    def _evaluate_cv_performance(
        self,
        X_sorted: pd.DataFrame,
        y_sorted: pd.Series,
        dates_sorted: pd.Series,
        folds: List[Dict[str, Any]],
        task: str,
        model_type: str,
        model_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """使用最终超参数在时间序列折上评估性能。"""

        scores: List[float] = []
        metric_name = 'auc' if task == 'classification' else 'r2'

        if task == 'classification':
            self._cv_classification_records = []
        else:
            self._cv_regression_records = []

        for fold in folds:
            train_idx = fold['train_idx']
            val_idx = fold['val_idx']

            X_train = X_sorted.iloc[train_idx]
            y_train = y_sorted.iloc[train_idx]
            X_val = X_sorted.iloc[val_idx]
            y_val = y_sorted.iloc[val_idx]

            if task == 'classification' and y_train.nunique() < 2:
                logger.debug("折%d 训练标签单一，跳过该折的CV评估", fold['fold_id'])
                continue

            selected_features, _ = self._select_features_for_task(X_train, y_train, task)
            if not selected_features:
                logger.debug("折%d 无可用特征，跳过", fold['fold_id'])
                continue

            if task == 'classification':
                selected_features = self._append_categorical_features(
                    selected_features,
                    X_train,
                    context=f"CV折{fold['fold_id']}"
                )

            num_feats, cat_feats = self._split_feature_types(selected_features)
            preprocessor = self.create_preprocessing_pipeline(num_feats, cat_feats)
            preprocessor.fit(X_train[selected_features])

            X_train_trans = preprocessor.transform(X_train[selected_features])
            X_val_trans = preprocessor.transform(X_val[selected_features])

            estimator = self._create_estimator(model_type, task, model_params.copy())

            fit_kwargs: Dict[str, Any] = {}
            callbacks: List[Any] = []
            early_rounds = getattr(estimator, '_codex_early_stopping_rounds', None)

            if model_type == 'lightgbm':
                import lightgbm as lgb
                metric = 'auc' if task == 'classification' else 'rmse'
                fit_kwargs['eval_set'] = [(X_val_trans, y_val)]
                fit_kwargs['eval_metric'] = metric
                if early_rounds:
                    callbacks.append(lgb.early_stopping(early_rounds, verbose=False))
            elif model_type == 'xgboost':
                fit_kwargs['eval_set'] = [(X_val_trans, y_val)]
                if early_rounds:
                    fit_kwargs['early_stopping_rounds'] = early_rounds

            if callbacks:
                existing = fit_kwargs.get('callbacks')
                if existing:
                    fit_kwargs['callbacks'] = list(existing) + callbacks
                else:
                    fit_kwargs['callbacks'] = callbacks

            try:
                estimator.fit(X_train_trans, y_train, **fit_kwargs)
            except TypeError as exc:
                removed = False
                if 'early_stopping_rounds' in fit_kwargs:
                    fit_kwargs.pop('early_stopping_rounds', None)
                    removed = True
                if 'callbacks' in fit_kwargs:
                    fit_kwargs.pop('callbacks', None)
                    removed = True
                if removed:
                    try:
                        estimator.fit(X_train_trans, y_train, **fit_kwargs)
                    except Exception as exc_inner:
                        logger.debug("折%d 训练失败: %s", fold['fold_id'], exc_inner)
                        continue
                else:
                    logger.debug("折%d 训练失败: %s", fold['fold_id'], exc)
                    continue
            except ValueError as exc:
                logger.debug("折%d 训练失败: %s", fold['fold_id'], exc)
                continue

            if task == 'classification':
                if y_val.nunique() < 2:
                    continue
                if hasattr(estimator, 'predict_proba'):
                    y_pred = estimator.predict_proba(X_val_trans)[:, 1]
                else:
                    y_pred = estimator.decision_function(X_val_trans)
                from sklearn.metrics import roc_auc_score
                try:
                    score = roc_auc_score(y_val, y_pred)
                except ValueError:
                    continue
            else:
                preds = estimator.predict(X_val_trans)
                from sklearn.metrics import r2_score
                score = r2_score(y_val, preds)

            scores.append(float(score))

            if task == 'classification':
                self._cv_classification_records.append({
                    'fold_id': fold['fold_id'],
                    'y_true': y_val.reset_index(drop=True),
                    'y_pred': pd.Series(y_pred).reset_index(drop=True)
                })
            else:
                self._cv_regression_records.append({
                    'fold_id': fold['fold_id'],
                    'y_true': y_val.reset_index(drop=True),
                    'y_pred': pd.Series(preds).reset_index(drop=True)
                })

        metrics: Dict[str, float] = {}
        if scores:
            metrics[f'cv_{metric_name}_mean'] = float(np.mean(scores))
            metrics[f'cv_{metric_name}_std'] = float(np.std(scores))
        else:
            logger.warning("时间序列交叉验证未得到有效评分，可能是标签或样本不足")

        # 记录元数据供后续日志分析
        self._store_cv_metadata(X_sorted, y_sorted, dates_sorted, folds, task)

        return metrics
    
    def train_classification_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'lightgbm',
        dates: Optional[pd.Series] = None,
        **model_params
    ) -> Dict:
        """
        训练分类模型
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            分类标签
        model_type : str, default='lightgbm'
            模型类型
        **model_params : dict
            模型参数
        
        Returns
        -------
        result : dict
            包含 pipeline, calibrator, metrics 等
        """
        logger.info("="*80)
        logger.info(f"开始训练分类模型: {model_type}")
        logger.info("="*80)

        # 1) 构建时间序列交叉验证数据
        X_cv_sorted, y_cv_sorted, dates_cv_sorted, cv_folds = self._build_cv_dataset(X, y, dates)

        # 2) Optuna调参（折内特征选择）
        if self.enable_optuna and model_type in ['xgboost', 'lightgbm']:
            best_params = self._optuna_optimize(
                X_cv_sorted,
                y_cv_sorted,
                dates_cv_sorted,
                cv_folds,
                task='classification',
                model_type=model_type
            )
            if best_params:
                logger.info("应用Optuna优化参数")
                model_params.update(best_params)
                logger.info("  调整后参数: %s", {k: model_params[k] for k in best_params.keys()})

        # 3) 多折评估
        cv_metrics = self._evaluate_cv_performance(
            X_cv_sorted,
            y_cv_sorted,
            dates_cv_sorted,
            cv_folds,
            task='classification',
            model_type=model_type,
            model_params=model_params
        )
        for name, value in cv_metrics.items():
            logger.info("CV指标 %s = %.4f", name, value)

        # 4) 最终训练/验证切分（最近窗口）
        cv_fold_info_backup = [info.copy() for info in self.cv_fold_info]
        cv_pairs_backup = list(self._cv_pairs) if self._cv_pairs else []
        cv_sorted_backup = self._cv_sorted_frames

        X_train, X_val, y_train, y_val = self._prepare_timeseries_split(X, y, dates)
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"正样本比例: 训练 {y_train.mean():.2%}, 验证 {y_val.mean():.2%}")

        # 恢复CV元信息供后续折别诊断使用
        if cv_fold_info_backup:
            self.cv_fold_info = cv_fold_info_backup
        if cv_pairs_backup:
            self._cv_pairs = cv_pairs_backup
        if cv_sorted_backup is not None:
            self._cv_sorted_frames = cv_sorted_backup

        # 5) 训练集特征选择（避免泄漏）
        if self.enable_feature_selection:
            selected_features, feature_importances = self._select_features_for_task(
                X_train,
                y_train,
                task='classification'
            )
            selected_features = self._append_categorical_features(selected_features, X_train, context="训练集")
            logger.info("训练集特征选择: %d → %d", X_train.shape[1], len(selected_features))
        else:
            selected_features = list(X_train.columns)
            feature_importances = None

        num_feats, cat_feats = self._split_feature_types(selected_features)
        preprocessor = self.create_preprocessing_pipeline(num_feats, cat_feats)
        preprocessor.fit(X_train[selected_features])

        X_train_trans = preprocessor.transform(X_train[selected_features])
        X_val_trans = preprocessor.transform(X_val[selected_features])

        train_dates = self._last_split_dates.get('train')
        sample_weight_train = self._compute_time_decay_weights(train_dates)
        if sample_weight_train is not None and len(sample_weight_train) != len(X_train_trans):
            logger.warning("样本加权长度(%d)与训练集(%d)不一致，忽略", len(sample_weight_train), len(X_train_trans))
            sample_weight_train = None

        # 6) 拟合最终模型
        early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
        if model_type == 'lightgbm':
            import lightgbm as lgb
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(
                n_estimators=model_params.get('n_estimators', 800),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.05),
                subsample=model_params.get('subsample', 0.8),
                colsample_bytree=model_params.get('colsample_bytree', 0.8),
                reg_alpha=model_params.get('reg_alpha', 0.1),
                reg_lambda=model_params.get('reg_lambda', 1.0),
                random_state=42,
                verbosity=-1
            )
            estimator.fit(
                X_train_trans,
                y_train,
                sample_weight=sample_weight_train,
                eval_set=[(X_val_trans, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        elif model_type == 'xgboost':
            estimator = self._train_xgboost_classifier(
                X_train_trans,
                y_train,
                X_val_trans,
                y_val,
                early_stopping_rounds,
                model_params,
                sample_weight=sample_weight_train
            )
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            penalty = model_params.get('penalty', 'l2')
            solver = model_params.get('solver', 'lbfgs')
            lr_params = {
                'max_iter': model_params.get('max_iter', 1000),
                'C': model_params.get('C', 1.0),
                'penalty': penalty,
                'solver': solver,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            if penalty == 'elasticnet':
                lr_params['l1_ratio'] = model_params.get('l1_ratio', 0.5)
            estimator = LogisticRegression(**lr_params)
            estimator.fit(X_train_trans, y_train, sample_weight=sample_weight_train)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_wrapper = PreprocessedModel(
            preprocessor=preprocessor,
            estimator=estimator,
            task='classification',
            feature_columns=list(selected_features)
        )

        # 7) 评估与校准
        from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

        y_pred_train = model_wrapper.predict_proba(X_train)[:, 1]
        y_pred_val = model_wrapper.predict_proba(X_val)[:, 1]

        metrics = {
            'train_auc': roc_auc_score(y_train, y_pred_train),
            'val_auc': roc_auc_score(y_val, y_pred_val),
            'train_f1_default': f1_score(y_train, (y_pred_train > 0.5).astype(int)),
            'val_f1_default': f1_score(y_val, (y_pred_val > 0.5).astype(int)),
            'train_brier': brier_score_loss(y_train, y_pred_train),
            'val_brier': brier_score_loss(y_val, y_pred_val)
        }
        metrics.update(cv_metrics)

        if 'ks' in self.config.get('classification_metrics', []):
            metrics['val_ks'] = self._compute_ks_score(y_val, y_pred_val)

        if 'top_k_hit_rate' in self.config.get('classification_metrics', []):
            top_k_hits = self._compute_top_k_hit_rates(y_val, y_pred_val, self.config.get('top_k_values', []))
            for k, hit in top_k_hits.items():
                metrics[f'top_{k}_hit_rate'] = hit

        logger.info("训练集评估: AUC %.4f / F1@0.5 %.4f / Brier %.4f",
                    metrics['train_auc'], metrics['train_f1_default'], metrics['train_brier'])
        logger.info("验证集评估: AUC %.4f / F1@0.5 %.4f / Brier %.4f",
                    metrics['val_auc'], metrics['val_f1_default'], metrics['val_brier'])

        calibrator = None
        calibrated_scores = y_pred_val
        if ENABLE_CALIBRATION:
            logger.info("应用概率校准: %s", CALIBRATION_METHOD)
            calibrator = CalibratedClassifierCV(estimator=model_wrapper, method=CALIBRATION_METHOD, cv='prefit')
            calibrator.fit(X_val, y_val)
            calibrated_scores = calibrator.predict_proba(X_val)[:, 1]
            cal_brier = brier_score_loss(y_val, calibrated_scores)
            metrics['val_brier_calibrated'] = cal_brier
            logger.info("校准后 Brier: %.4f (改善 %.4f)", cal_brier, metrics['val_brier'] - cal_brier)

        baseline_threshold = float(self.config.get('cls_threshold', CLS_THRESHOLD))
        production_threshold = baseline_threshold

        target_positive_rate = self.config.get('target_positive_rate')
        dynamic_threshold = None
        dynamic_metrics: Dict[str, float] = {}

        if self.config.get('auto_adjust_threshold', False):
            dynamic_threshold = self._derive_dynamic_threshold(
                calibrated_scores,
                target_positive_rate,
                fallback=baseline_threshold
            )
            if dynamic_threshold is not None and not math.isclose(dynamic_threshold, production_threshold, rel_tol=1e-3, abs_tol=1e-4):
                logger.info(
                    "动态阈值调整: %.3f → %.3f (目标正样本率 %.2f%%)",
                    production_threshold,
                    dynamic_threshold,
                    target_positive_rate * 100 if target_positive_rate else float('nan')
                )
                production_threshold = dynamic_threshold
            elif dynamic_threshold is not None:
                logger.info("动态阈值与初始阈值一致: %.3f", dynamic_threshold)

        prod_metrics = self._evaluate_threshold_metrics(y_val, calibrated_scores, production_threshold)
        if dynamic_threshold is not None:
            dynamic_metrics = prod_metrics.copy()

        optimal_threshold, threshold_metrics = self._find_optimal_threshold(y_val, calibrated_scores)
        high_threshold = min(0.6, max(production_threshold + 0.05, optimal_threshold + 0.03))
        high_metrics = self._evaluate_threshold_metrics(y_val, calibrated_scores, high_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        metrics.update({f"optimal_{k}": v for k, v in threshold_metrics.items()})
        metrics['baseline_threshold'] = baseline_threshold
        metrics['production_threshold'] = production_threshold
        metrics.update({f"production_{k}": v for k, v in prod_metrics.items()})
        metrics['dynamic_threshold'] = dynamic_threshold
        if dynamic_metrics:
            metrics.update({f"dynamic_{k}": v for k, v in dynamic_metrics.items()})
        metrics['high_threshold'] = high_threshold
        metrics.update({f"high_{k}": v for k, v in high_metrics.items()})

        # 将最终阈值写回配置，供后续流程和模型持久化使用
        self.config['cls_threshold'] = production_threshold

        logger.info(
            "最佳阈值 %.3f → 验证 Precision %.4f / Recall %.4f / F1 %.4f",
            optimal_threshold,
            threshold_metrics.get('precision', np.nan),
            threshold_metrics.get('recall', np.nan),
            threshold_metrics.get('f1', np.nan)
        )
        logger.info(
            "生产阈值 %.3f → Precision %.4f / Recall %.4f / F1 %.4f",
            production_threshold,
            prod_metrics.get('precision', np.nan),
            prod_metrics.get('recall', np.nan),
            prod_metrics.get('f1', np.nan)
        )
        if dynamic_threshold is not None and not math.isclose(dynamic_threshold, baseline_threshold, rel_tol=1e-3, abs_tol=1e-4):
            logger.info(
                "动态阈值 %.3f → Precision %.4f / Recall %.4f / F1 %.4f (预测占比 %.2f%%)",
                dynamic_threshold,
                dynamic_metrics.get('precision', np.nan),
                dynamic_metrics.get('recall', np.nan),
                dynamic_metrics.get('f1', np.nan),
                dynamic_metrics.get('positive_rate', 0.0) * 100 if 'positive_rate' in dynamic_metrics else float('nan')
            )

        val_dates = None
        if isinstance(X_val, pd.DataFrame) and '_date' in X_val.columns:
            val_dates = pd.to_datetime(X_val['_date'])
        elif self._last_split_dates.get('val') is not None:
            val_dates = self._last_split_dates['val']
        scores_series = pd.Series(calibrated_scores if calibrator is not None else y_pred_val)
        try:
            self._log_recent_window_metrics(
                y_true=y_val.reset_index(drop=True),
                scores=scores_series,
                dates=val_dates.reset_index(drop=True) if isinstance(val_dates, pd.Series) else val_dates,
                task='classification'
            )
        except Exception as exc:
            logger.debug("近期窗口评估失败: %s", exc)

        if self._cv_pairs and self._cv_sorted_frames is not None:
            self._log_fold_classification_diagnostics(model_wrapper, calibrator, optimal_threshold)

        # 8) 结果封装
        result = {
            'task': 'classification',
            'model_type': model_type,
            'pipeline': model_wrapper,
            'calibrator': calibrator,
            'selected_features': list(selected_features),
            'metrics': metrics,
            'threshold': production_threshold,
            'optimal_threshold': optimal_threshold,
            'dynamic_threshold': dynamic_threshold,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy(),
            'stage5_metadata': {
                'feature_selection_enabled': self.enable_feature_selection,
                'feature_importances': feature_importances.tolist() if feature_importances is not None else None,
                'original_feature_count': len(self._base_numerical_features) + len(self._base_categorical_features),
                'selected_feature_count': len(selected_features),
                'optuna_enabled': self.enable_optuna,
                'optuna_params': {k: model_params[k] for k in model_params if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']} if self.enable_optuna else None
            }
        }

        logger.info("✅ 分类模型训练完成\n")
        return result
    
    def train_regression_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'lightgbm',
        dates: Optional[pd.Series] = None,
        **model_params
    ) -> Dict:
        """
        训练回归模型
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            回归目标
        model_type : str, default='lightgbm'
            模型类型
        **model_params : dict
            模型参数
        
        Returns
        -------
        result : dict
            包含 pipeline, metrics 等
        """
        logger.info("="*80)
        logger.info(f"开始训练回归模型: {model_type}")
        logger.info("="*80)

        # 1) 构建时间序列交叉验证数据
        X_cv_sorted, y_cv_sorted, dates_cv_sorted, cv_folds = self._build_cv_dataset(X, y, dates)

        # 2) Optuna调参
        if self.enable_optuna and model_type == 'xgboost':
            best_params = self._optuna_optimize(
                X_cv_sorted,
                y_cv_sorted,
                dates_cv_sorted,
                cv_folds,
                task='regression',
                model_type=model_type
            )
            if best_params:
                logger.info("应用Optuna优化参数")
                model_params.update(best_params)
                logger.info("  调整后参数: %s", {k: model_params[k] for k in best_params.keys()})

        # 3) 多折评估
        cv_metrics = self._evaluate_cv_performance(
            X_cv_sorted,
            y_cv_sorted,
            dates_cv_sorted,
            cv_folds,
            task='regression',
            model_type=model_type,
            model_params=model_params
        )
        for name, value in cv_metrics.items():
            logger.info("CV指标 %s = %.4f", name, value)

        # 4) 最终训练/验证切分
        cv_fold_info_backup = [info.copy() for info in self.cv_fold_info]
        cv_pairs_backup = list(self._cv_pairs) if self._cv_pairs else []
        cv_sorted_backup = self._cv_sorted_frames

        X_train, X_val, y_train, y_val = self._prepare_timeseries_split(X, y, dates)
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"目标统计（标准化前）: 均值 {y_train.mean():.4f}, 标准差 {y_train.std():.4f}")

        if cv_fold_info_backup:
            self.cv_fold_info = cv_fold_info_backup
        if cv_pairs_backup:
            self._cv_pairs = cv_pairs_backup
        if cv_sorted_backup is not None:
            self._cv_sorted_frames = cv_sorted_backup

        # 5) 标签标准化（可选）
        y_scaler = None
        if self.config.get('normalize_regression_labels', False):
            from sklearn.preprocessing import StandardScaler
            logger.info("✅ 启用回归标签标准化（StandardScaler）")
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
            y_train_original = y_train.copy()
            y_val_original = y_val.copy()
            y_train = pd.Series(y_train_scaled, index=y_train.index)
            y_val = pd.Series(y_val_scaled, index=y_val.index)
        else:
            y_train_original = y_train
            y_val_original = y_val

        # 6) 训练集特征选择
        if self.enable_feature_selection:
            selected_features, feature_importances = self._select_features_for_task(
                X_train,
                y_train,
                task='regression'
            )
            logger.info("训练集特征选择: %d → %d", X_train.shape[1], len(selected_features))
        else:
            selected_features = list(X_train.columns)
            feature_importances = None

        num_feats, cat_feats = self._split_feature_types(selected_features)
        preprocessor = self.create_preprocessing_pipeline(num_feats, cat_feats)
        preprocessor.fit(X_train[selected_features])

        X_train_trans = preprocessor.transform(X_train[selected_features])
        X_val_trans = preprocessor.transform(X_val[selected_features])

        # 7) 拟合最终模型
        early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
        if model_type == 'lightgbm':
            import lightgbm as lgb
            from lightgbm import LGBMRegressor
            from src.ml.training.toolkit.params import get_optimized_lgbm_regression_params
            optimized_params = get_optimized_lgbm_regression_params()
            estimator = LGBMRegressor(
                n_estimators=model_params.get('n_estimators', 800),
                num_leaves=model_params.get('num_leaves', optimized_params.get('num_leaves', 63)),
                max_depth=model_params.get('max_depth', optimized_params.get('max_depth', 8)),
                learning_rate=model_params.get('learning_rate', optimized_params.get('learning_rate', 0.02)),
                subsample=model_params.get('subsample', optimized_params.get('bagging_fraction', 0.8)),
                colsample_bytree=model_params.get('colsample_bytree', optimized_params.get('feature_fraction', 0.9)),
                min_child_samples=model_params.get('min_child_samples', optimized_params.get('min_data_in_leaf', 50)),
                reg_alpha=model_params.get('reg_alpha', optimized_params.get('lambda_l1', 0.1)),
                reg_lambda=model_params.get('reg_lambda', optimized_params.get('lambda_l2', 0.1)),
                random_state=42,
                verbosity=-1
            )
            estimator.fit(
                X_train_trans,
                y_train,
                eval_set=[(X_val_trans, y_val)],
                eval_metric='rmse',
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        elif model_type == 'xgboost':
            estimator = self._train_xgboost_regressor(
                X_train_trans,
                y_train,
                X_val_trans,
                y_val,
                early_stopping_rounds,
                model_params
            )
        elif model_type == 'ridge':
            from sklearn.linear_model import Ridge
            estimator = Ridge(alpha=model_params.get('alpha', 1.0))
            estimator.fit(X_train_trans, y_train)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_wrapper = PreprocessedModel(
            preprocessor=preprocessor,
            estimator=estimator,
            task='regression',
            feature_columns=list(selected_features)
        )

        # 8) 评估
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        y_pred_train = model_wrapper.predict(X_train)
        y_pred_val = model_wrapper.predict(X_val)

        if y_scaler is not None:
            logger.info("反标准化预测结果用于评估")
            y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
            y_pred_val = y_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            y_train = y_train_original
            y_val = y_val_original

        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'val_mae': mean_absolute_error(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val))
        }
        metrics.update(cv_metrics)

        logger.info("训练集评估: R² %.4f / MAE %.4f / RMSE %.4f",
                    metrics['train_r2'], metrics['train_mae'], metrics['train_rmse'])
        logger.info("验证集评估: R² %.4f / MAE %.4f / RMSE %.4f",
                    metrics['val_r2'], metrics['val_mae'], metrics['val_rmse'])

        val_dates = None
        if isinstance(X_val, pd.DataFrame) and '_date' in X_val.columns:
            val_dates = pd.to_datetime(X_val['_date'])
        elif self._last_split_dates.get('val') is not None:
            val_dates = self._last_split_dates['val']
        try:
            self._log_recent_window_metrics(
                y_true=y_val.reset_index(drop=True),
                scores=pd.Series(y_pred_val),
                dates=val_dates.reset_index(drop=True) if isinstance(val_dates, pd.Series) else val_dates,
                task='regression'
            )
        except Exception as exc:
            logger.debug("近期窗口评估(回归)失败: %s", exc)

        if self._cv_pairs and self._cv_sorted_frames is not None:
            self._log_fold_regression_diagnostics(model_wrapper)

        # 9) 结果封装
        result = {
            'task': 'regression',
            'model_type': model_type,
            'pipeline': model_wrapper,
            'y_scaler': y_scaler,
            'calibrator': None,
            'selected_features': list(selected_features),
            'metrics': metrics,
            'threshold': None,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy(),
            'stage5_metadata': {
                'feature_selection_enabled': self.enable_feature_selection,
                'feature_importances': feature_importances.tolist() if feature_importances is not None else None,
                'original_feature_count': len(self._base_numerical_features) + len(self._base_categorical_features),
                'selected_feature_count': len(selected_features),
                'optuna_enabled': self.enable_optuna,
                'optuna_params': {k: model_params[k] for k in model_params if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']} if (self.enable_optuna and model_type == 'xgboost') else None,
                'label_normalization_enabled': y_scaler is not None
            }
        }

        logger.info("✅ 回归模型训练完成\n")
        return result
    
    def save_model(self, model_artifact: Dict, filepath: str, is_best: bool = False):
        """
        保存模型（新格式）
        
        Parameters
        ----------
        model_artifact : dict
            模型artifact
        filepath : str
            保存路径
        is_best : bool, default=False
            是否为最优模型
        """
        # 添加is_best标记
        model_artifact['is_best'] = is_best
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存
        joblib.dump(model_artifact, filepath)
        
        logger.info(f"模型已保存: {filepath}")
        logger.info(f"  任务: {model_artifact['task']}")
        logger.info(f"  类型: {model_artifact['model_type']}")
        logger.info(f"  最优: {is_best}")
    
    @staticmethod
    def load_model(filepath: str) -> Dict:
        """
        加载模型（兼容新格式）
        
        Parameters
        ----------
        filepath : str
            模型文件路径
        
        Returns
        -------
        model_artifact : dict
            模型artifact
        """
        model_artifact = joblib.load(filepath)
        
        # 验证格式
        required_keys = ['task', 'pipeline']
        for key in required_keys:
            if key not in model_artifact:
                logger.warning(f"模型文件缺少字段: {key}")
        
        logger.info(f"模型已加载: {filepath}")
        logger.info(f"  任务: {model_artifact.get('task', 'unknown')}")
        logger.info(f"  训练日期: {model_artifact.get('training_date', 'unknown')}")
        
        return model_artifact
