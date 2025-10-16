# -*- coding: utf-8 -*-
"""
增强版统一训练器 V2
整合新特征、预处理管道、特征选择、概率校准
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import joblib
import os

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
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
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
        
        # Stage 5: Optuna超参数优化配置
        self.enable_optuna = self.config.get('enable_optuna', False)
        if self.enable_optuna:
            logger.info("  ✅ Optuna优化: 已启用")
            logger.info(f"     试验次数: {self.config.get('optuna_trials', 100)}")
            logger.info(f"     采样器: {self.config.get('optuna_sampler', 'tpe')}")

        self.cv_fold_info: List[Dict[str, Any]] = []
        self._cv_sorted_frames: Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]] = None
        self._cv_pairs: List[Dict[str, Any]] = []
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'winsor_clip_quantile': WINSOR_CLIP_QUANTILE,
            'calibration_method': CALIBRATION_METHOD,
            'calibration_cv': CALIBRATION_CV,
            'enable_feature_selection': ENABLE_FEATURE_SELECTION,
            'min_features': 10,
            'max_features': None,
            'cls_threshold': CLS_THRESHOLD,
            'prediction_period': PREDICTION_PERIOD_DAYS,
            'enable_time_series_split': ENABLE_TIME_SERIES_SPLIT,
            'cv_n_splits': CV_N_SPLITS,
            'cv_embargo': CV_EMBARGO,
            'cv_allow_future': CV_ALLOW_FUTURE,
            'use_rolling_cv': False,
            'classification_metrics': CLASSIFICATION_METRICS,
            'regression_metrics': REGRESSION_METRICS,
            'top_k_values': TOP_K_VALUES
        }

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
        
        if dates is None or not self.config.get('enable_time_series_split', False):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
            )
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
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.warning(f"改进的时间分割失败,回退到常规分割: {e}")
            # 回退到常规切分
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if (y.nunique() == 2 and len(y.unique()) > 1) else None
            )
            self._cv_pairs = []
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

    @staticmethod
    def _find_optimal_threshold(y_true: pd.Series, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
        # 覆盖更细颗粒度并强制包含配置阈值，避免遗漏0.03等业务设定
        base_grid = np.linspace(0.01, 0.99, 99)
        thresholds = np.unique(np.concatenate([base_grid, np.array([CLS_THRESHOLD])]))
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

        logger.info("折别回放诊断(分类):")
        for pair in self._cv_pairs:
            val_idx = pair['val_idx']
            if len(val_idx) == 0:
                continue
            fold_id = pair['fold_id']
            y_true = y_sorted.iloc[val_idx].reset_index(drop=True)
            X_val_fold = X_sorted.iloc[val_idx].reset_index(drop=True)
            start_dt = dates_sorted.iloc[val_idx[0]].strftime('%Y-%m-%d')
            end_dt = dates_sorted.iloc[val_idx[-1]].strftime('%Y-%m-%d')

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

        logger.info("折别回放诊断(回归):")
        for pair in self._cv_pairs:
            val_idx = pair['val_idx']
            if len(val_idx) == 0:
                continue
            fold_id = pair['fold_id']
            y_true = y_sorted.iloc[val_idx].reset_index(drop=True)
            X_val_fold = X_sorted.iloc[val_idx].reset_index(drop=True)
            start_dt = dates_sorted.iloc[val_idx[0]].strftime('%Y-%m-%d')
            end_dt = dates_sorted.iloc[val_idx[-1]].strftime('%Y-%m-%d')

            try:
                preds = model_wrapper.predict(X_val_fold)
            except Exception as exc:  # pragma: no cover
                logger.warning("  折%02d 回归预测失败: %s", fold_id, exc)
                continue

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
        model_params: Dict[str, Any]
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

        dtrain = xgb.DMatrix(X_train, label=y_train.to_numpy())
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
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        创建统一预处理管道
        
        Returns
        -------
        preprocessor : ColumnTransformer
            预处理器
        """
        # 数值特征 pipeline
        numerical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
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
        
        # 组合
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipe, self.numerical_features),
                ('cat', categorical_pipe, self.categorical_features)
            ],
            remainder='drop'
        )
        
        logger.info("创建预处理管道:")
        logger.info(f"  数值管道: Imputer → Winsorizer → Scaler")
        logger.info(f"  类别管道: Imputer → OneHot")
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
        from src.ml.preprocessing.feature_selection import select_features_for_task
        
        logger.info(f"执行{task}特征选择...")
        
        # 🔧 关键修复: 确保数据类型正确
        X_converted = X.copy()
        for col in X_converted.columns:
            if X_converted[col].dtype == 'object':
                X_converted[col] = pd.to_numeric(X_converted[col], errors='coerce')
        
        # 检查是否有缺失值
        if X_converted.isna().any().any():
            logger.warning("特征选择前检测到缺失值，将进行填充")
            X_converted = X_converted.fillna(X_converted.median())
        
        logger.debug(f"特征数据类型: {X_converted.dtypes.to_dict()}")
        
        selected_features, importances = select_features_for_task(
            X_converted, y,
            task=task,
            method=self.config.get('feature_selection_method', 'lightgbm'),
            threshold=self.config.get('feature_selection_threshold', 'median'),
            min_features=self.config.get('min_features', 15),
            max_features=self.config.get('max_features', 30)
        )
        
        logger.info(f"✅ 特征选择完成: {len(X.columns)} → {len(selected_features)}")
        
        return selected_features, importances
    
    def _optuna_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series],
        task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        执行Optuna超参数优化
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            标签数据
        dates : Series or None
            日期序列
        task : str
            任务类型 ('classification' 或 'regression')
        
        Returns
        -------
        best_params : dict
            最优超参数
        """
        from src.ml.optimization.optuna_optimizer import OptunaOptimizer
        
        logger.info(f"执行{task} Optuna超参数优化...")
        
        # 🔧 关键修复: Optuna只支持数值特征，需要过滤类别特征
        numerical_cols = [col for col in X.columns if col not in self.categorical_features]
        if len(numerical_cols) < len(X.columns):
            logger.info(f"Optuna优化将使用 {len(numerical_cols)} 个数值特征（排除 {len(self.categorical_features)} 个类别特征）")
            X_for_optuna = X[numerical_cols].copy()
        else:
            X_for_optuna = X.copy()
        
        # 确保数据类型正确
        for col in X_for_optuna.columns:
            if X_for_optuna[col].dtype == 'object':
                X_for_optuna[col] = pd.to_numeric(X_for_optuna[col], errors='coerce')
        
        # 填充缺失值
        if X_for_optuna.isna().any().any():
            logger.warning("Optuna优化前检测到缺失值，将进行填充")
            X_for_optuna = X_for_optuna.fillna(X_for_optuna.median())
        
        optimizer_config = {
            'n_trials': self.config.get('optuna_trials', 100),
            'timeout': self.config.get('optuna_timeout', 3600),
            'sampler': self.config.get('optuna_sampler', 'tpe'),
            'cv_folds': self.config.get('optuna_cv_folds', 5),
            'n_jobs': 1  # 避免嵌套并行
        }
        
        optimizer = OptunaOptimizer(optimizer_config)
        
        # 根据任务类型和模型类型选择优化方法
        model_type = self.config.get('optuna_model_type', 'xgboost')
        
        if task == 'classification':
            if model_type == 'xgboost':
                result = optimizer.optimize_xgboost_classification(X_for_optuna, y, dates)
            elif model_type == 'lightgbm':
                result = optimizer.optimize_lightgbm_classification(X_for_optuna, y, dates)
            else:
                logger.warning(f"不支持的模型类型: {model_type}，跳过Optuna优化")
                return {}
        else:  # regression
            if model_type == 'xgboost':
                result = optimizer.optimize_xgboost_regression(X_for_optuna, y, dates)
            else:
                logger.warning(f"回归任务仅支持XGBoost优化，跳过")
                return {}
        
        logger.info(f"✅ Optuna优化完成")
        logger.info(f"  最优得分: {result['best_score']:.4f}")
        logger.info(f"  试验次数: {result['n_trials']}")
        
        return result['best_params']
    
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
        
        # Stage 5: 特征选择 (在数据切分前执行)
        selected_features = None
        feature_importances = None
        original_numerical_features = self.numerical_features.copy()
        original_categorical_features = self.categorical_features.copy()
        
        if self.enable_feature_selection:
            selected_features, feature_importances = self._select_features_for_task(
                X, y, task='classification'
            )
            # 只保留选定的特征
            X = X[selected_features]
            logger.info(f"特征选择后数据形状: {X.shape}")
            
            # 🔧 关键修复: 更新numerical_features和categorical_features
            # 只保留仍然存在的特征
            self.numerical_features = [f for f in self.numerical_features if f in selected_features]
            self.categorical_features = [f for f in self.categorical_features if f in selected_features]
            logger.info(f"更新特征集合: 数值特征 {len(self.numerical_features)}, 类别特征 {len(self.categorical_features)}")
        
        # Stage 5: Optuna超参数优化 (在数据切分前执行)
        if self.enable_optuna and model_type in ['xgboost', 'lightgbm']:
            best_params = self._optuna_optimize(X, y, dates, task='classification')
            # 🔧 Stage5修复: Optuna优化结果优先（覆盖用户传入的参数）
            if best_params:
                logger.info("应用Optuna优化参数（Optuna优先覆盖）")
                model_params.update(best_params)  # ✅ 使用update覆盖，而非只填充缺失项
                logger.info(f"  优化后参数: {list(best_params.keys())}")
        
        # 切分数据
        X_train, X_val, y_train, y_val = self._prepare_timeseries_split(
            X, y, dates
        )
        
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"正样本比例: 训练集 {y_train.mean():.2%}, 验证集 {y_val.mean():.2%}")

        if self.config.get('use_rolling_cv', False) and self.cv_fold_info:
            logger.info("时间序列折统计:")
            for fold in self.cv_fold_info:
                logger.info(
                    "  折%02d 训练[%s ~ %s] %4d 条(正样本 %.2f%%) | 验证[%s ~ %s] %4d 条(正样本 %.2f%%)",
                    fold['fold_id'],
                    fold['train_start'].strftime('%Y-%m-%d'),
                    fold['train_end'].strftime('%Y-%m-%d'),
                    fold['train_size'],
                    fold['train_pos_rate'] * 100,
                    fold['val_start'].strftime('%Y-%m-%d'),
                    fold['val_end'].strftime('%Y-%m-%d'),
                    fold['val_size'],
                    fold['val_pos_rate'] * 100
                )
        
        # 创建预处理器并仅在训练集上拟合
        preprocessor = self.create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        feature_columns = list(X_train.columns)
        X_train_trans = preprocessor.transform(X_train)
        X_val_trans = preprocessor.transform(X_val)

        # 创建模型
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
                model_params
            )
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            
            # 获取参数
            penalty = model_params.get('penalty', 'l2')
            solver = model_params.get('solver', 'lbfgs')
            
            # 构建LogisticRegression参数
            lr_params = {
                'max_iter': model_params.get('max_iter', 1000),
                'C': model_params.get('C', 1.0),
                'penalty': penalty,
                'solver': solver,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            
            # 如果使用elasticnet惩罚，需要添加l1_ratio参数
            if penalty == 'elasticnet':
                lr_params['l1_ratio'] = model_params.get('l1_ratio', 0.5)
            
            estimator = LogisticRegression(**lr_params)
            estimator.fit(X_train_trans, y_train)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_wrapper = PreprocessedModel(
            preprocessor=preprocessor,
            estimator=estimator,
            task='classification',
            feature_columns=feature_columns
        )

        # 预测（用于评估和校准）
        y_pred_train = model_wrapper.predict_proba(X_train)[:, 1]
        y_pred_val = model_wrapper.predict_proba(X_val)[:, 1]
        
        # 评估
        from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
        
        train_auc = roc_auc_score(y_train, y_pred_train)
        val_auc = roc_auc_score(y_val, y_pred_val)
        metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_f1_default': f1_score(y_train, (y_pred_train > 0.5).astype(int)),
            'val_f1_default': f1_score(y_val, (y_pred_val > 0.5).astype(int)),
            'train_brier': brier_score_loss(y_train, y_pred_train),
            'val_brier': brier_score_loss(y_val, y_pred_val)
        }

        if 'ks' in self.config.get('classification_metrics', []):
            metrics['val_ks'] = self._compute_ks_score(y_val, y_pred_val)

        if 'top_k_hit_rate' in self.config.get('classification_metrics', []):
            top_k_hits = self._compute_top_k_hit_rates(y_val, y_pred_val, self.config.get('top_k_values', []))
            for k, hit in top_k_hits.items():
                metrics[f'top_{k}_hit_rate'] = hit
        
        logger.info("训练集评估:")
        logger.info(f"  AUC: {metrics['train_auc']:.4f}")
        logger.info(f"  F1(0.5): {metrics['train_f1_default']:.4f}")
        logger.info(f"  Brier: {metrics['train_brier']:.4f}")
        
        logger.info("验证集评估:")
        logger.info(f"  AUC: {metrics['val_auc']:.4f}")
        logger.info(f"  F1(0.5): {metrics['val_f1_default']:.4f}")
        logger.info(f"  Brier: {metrics['val_brier']:.4f}")
        
        # 概率校准
        calibrator = None
        calibrated_scores = y_pred_val
        if ENABLE_CALIBRATION:
            logger.info(f"应用概率校准: {CALIBRATION_METHOD}")
            calibrator = CalibratedClassifierCV(
                estimator=model_wrapper,
                method=CALIBRATION_METHOD,
                cv='prefit'
            )
            calibrator.fit(X_val, y_val)
            
            # 评估校准后的性能
            y_pred_cal = calibrator.predict_proba(X_val)[:, 1]
            calibrated_scores = y_pred_cal
            cal_brier = brier_score_loss(y_val, y_pred_cal)
            
            logger.info(f"校准后 Brier: {cal_brier:.4f} (改善: {metrics['val_brier'] - cal_brier:.4f})")
            metrics['val_brier_calibrated'] = cal_brier
        
        optimal_threshold, threshold_metrics = self._find_optimal_threshold(y_val, calibrated_scores)
        metrics['optimal_threshold'] = optimal_threshold
        metrics.update({f"optimal_{k}": v for k, v in threshold_metrics.items()})
        metrics['val_f1_optimal'] = threshold_metrics.get('f1', np.nan)
        metrics['val_precision_optimal'] = threshold_metrics.get('precision', np.nan)
        metrics['val_recall_optimal'] = threshold_metrics.get('recall', np.nan)

        logger.info(
            "最佳阈值 %.3f → 验证集 Precision %.4f / Recall %.4f / F1 %.4f",
            optimal_threshold,
            metrics['val_precision_optimal'],
            metrics['val_recall_optimal'],
            metrics['val_f1_optimal']
        )

        if self.config.get('use_rolling_cv', False) and self._cv_pairs and self._cv_sorted_frames is not None:
            self._log_fold_classification_diagnostics(model_wrapper, calibrator, optimal_threshold)

        # 🔧 恢复原始特征列表（如果进行了特征选择）
        if self.enable_feature_selection:
            self.numerical_features = original_numerical_features
            self.categorical_features = original_categorical_features

        # 特征选择（如果需要）
        selected_features_legacy = None
        if self.config['enable_feature_selection']:
            if self.categorical_features:
                logger.info("存在类别特征，跳过原始空间特征选择以避免object类型问题")
            else:
                logger.info("执行特征选择...")
                from src.ml.preprocessing.feature_selection import select_features_for_task
                
                selected_features_legacy, _ = select_features_for_task(
                    X_train[self.numerical_features],
                    y_train,
                    task='classification',
                    min_features=self.config['min_features'],
                    max_features=self.config['max_features']
                )
        
        # 返回结果
        result = {
            'task': 'classification',
            'model_type': model_type,
            'pipeline': model_wrapper,
            'calibrator': calibrator,
            'selected_features': selected_features if selected_features else selected_features_legacy,
            'metrics': metrics,
            'threshold': optimal_threshold,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy(),
            # Stage 5: 添加元数据
            'stage5_metadata': {
                'feature_selection_enabled': self.enable_feature_selection,
                'feature_importances': feature_importances.tolist() if feature_importances is not None else None,
                'original_feature_count': len(original_numerical_features) + len(original_categorical_features) if self.enable_feature_selection else None,
                'selected_feature_count': len(selected_features) if selected_features else None,
                'optuna_enabled': self.enable_optuna,
                'optuna_params': {k: v for k, v in model_params.items() if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']} if self.enable_optuna else None
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
        
        # Stage 5: 特征选择 (在数据切分前执行)
        selected_features = None
        feature_importances = None
        original_numerical_features = self.numerical_features.copy()
        original_categorical_features = self.categorical_features.copy()
        
        if self.enable_feature_selection:
            selected_features, feature_importances = self._select_features_for_task(
                X, y, task='regression'
            )
            # 只保留选定的特征
            X = X[selected_features]
            logger.info(f"特征选择后数据形状: {X.shape}")
            
            # 🔧 关键修复: 更新numerical_features和categorical_features
            # 只保留仍然存在的特征
            self.numerical_features = [f for f in self.numerical_features if f in selected_features]
            self.categorical_features = [f for f in self.categorical_features if f in selected_features]
            logger.info(f"更新特征集合: 数值特征 {len(self.numerical_features)}, 类别特征 {len(self.categorical_features)}")
        
        # Stage 5: Optuna超参数优化 (在数据切分前执行)
        if self.enable_optuna and model_type == 'xgboost':  # 回归仅支持XGBoost
            best_params = self._optuna_optimize(X, y, dates, task='regression')
            # 🔧 Stage5修复: Optuna优化结果优先（覆盖用户传入的参数）
            if best_params:
                logger.info("应用Optuna优化参数（Optuna优先覆盖）")
                model_params.update(best_params)  # ✅ 使用update覆盖，而非只填充缺失项
                logger.info(f"  优化后参数: {list(best_params.keys())}")
        
        # 切分数据
        X_train, X_val, y_train, y_val = self._prepare_timeseries_split(
            X, y, dates
        )
        
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"目标统计（标准化前）: 均值 {y_train.mean():.4f}, 标准差 {y_train.std():.4f}")
        
        # 🔴 方案C2: 回归标签标准化
        y_scaler = None
        if self.config.get('normalize_regression_labels', False):
            from sklearn.preprocessing import StandardScaler
            logger.info("✅ 启用回归标签标准化（StandardScaler）")
            
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
            
            logger.info(f"  标准化后均值: {y_train_scaled.mean():.4f}")
            logger.info(f"  标准化后标准差: {y_train_scaled.std():.4f}")
            
            # 替换原始标签
            y_train_original = y_train.copy()
            y_val_original = y_val.copy()
            y_train = pd.Series(y_train_scaled, index=y_train.index)
            y_val = pd.Series(y_val_scaled, index=y_val.index)
        else:
            logger.info("回归标签未标准化")

        if self.config.get('use_rolling_cv', False) and self.cv_fold_info:
            logger.info("时间序列折统计:")
            for fold in self.cv_fold_info:
                logger.info(
                    "  折%02d 训练[%s ~ %s] %4d 条 | 验证[%s ~ %s] %4d 条",
                    fold['fold_id'],
                    fold['train_start'].strftime('%Y-%m-%d'),
                    fold['train_end'].strftime('%Y-%m-%d'),
                    fold['train_size'],
                    fold['val_start'].strftime('%Y-%m-%d'),
                    fold['val_end'].strftime('%Y-%m-%d'),
                    fold['val_size']
                )
        
        # 创建预处理器并仅在训练集上拟合
        preprocessor = self.create_preprocessing_pipeline()
        preprocessor.fit(X_train)
        feature_columns = list(X_train.columns)
        X_train_trans = preprocessor.transform(X_train)
        X_val_trans = preprocessor.transform(X_val)

        # 创建模型
        early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
        if model_type == 'lightgbm':
            import lightgbm as lgb
            from lightgbm import LGBMRegressor
            
            # 使用优化的回归参数（如果未提供自定义参数）
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
            feature_columns=feature_columns
        )

        # 预测
        y_pred_train = model_wrapper.predict(X_train)
        y_pred_val = model_wrapper.predict(X_val)
        
        # 🔴 方案C2: 如果标签被标准化，需要反标准化预测结果
        if y_scaler is not None:
            logger.info("反标准化预测结果...")
            y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
            y_pred_val = y_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            
            # 使用原始标签进行评估
            y_train = y_train_original
            y_val = y_val_original
            logger.info("使用原始标签尺度评估性能")
        
        # 评估
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'val_mae': mean_absolute_error(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val))
        }
        
        logger.info("训练集评估:")
        logger.info(f"  R²:  {metrics['train_r2']:.4f}")
        logger.info(f"  MAE: {metrics['train_mae']:.4f}")
        logger.info(f"  RMSE: {metrics['train_rmse']:.4f}")
        
        logger.info("验证集评估:")
        logger.info(f"  R²:  {metrics['val_r2']:.4f}")
        logger.info(f"  MAE: {metrics['val_mae']:.4f}")
        logger.info(f"  RMSE: {metrics['val_rmse']:.4f}")

        if self.config.get('use_rolling_cv', False) and self._cv_pairs and self._cv_sorted_frames is not None:
            self._log_fold_regression_diagnostics(model_wrapper)
        
        # 🔧 恢复原始特征列表（如果进行了特征选择）
        if self.enable_feature_selection:
            self.numerical_features = original_numerical_features
            self.categorical_features = original_categorical_features
        
        # 特征选择（如果需要）
        selected_features_legacy = None
        if self.config['enable_feature_selection']:
            if self.categorical_features:
                logger.info("存在类别特征，跳过原始空间特征选择以避免object类型问题")
            else:
                logger.info("执行特征选择...")
                from src.ml.preprocessing.feature_selection import select_features_for_task
                
                selected_features_legacy, _ = select_features_for_task(
                    X_train[self.numerical_features],
                    y_train,
                    task='regression',
                    min_features=self.config['min_features'],
                    max_features=self.config['max_features']
                )
        
        # 返回结果
        result = {
            'task': 'regression',
            'model_type': model_type,
            'pipeline': model_wrapper,
            'y_scaler': y_scaler,  # 🔴 方案C2: 保存标签标准化器
            'calibrator': None,  # 回归不需要校准
            'selected_features': selected_features if selected_features else selected_features_legacy,
            'metrics': metrics,
            'threshold': None,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy(),
            # Stage 5: 添加元数据
            'stage5_metadata': {
                'feature_selection_enabled': self.enable_feature_selection,
                'feature_importances': feature_importances.tolist() if feature_importances is not None else None,
                'original_feature_count': len(original_numerical_features) + len(original_categorical_features) if self.enable_feature_selection else None,
                'selected_feature_count': len(selected_features) if selected_features else None,
                'optuna_enabled': self.enable_optuna,
                'optuna_params': {k: v for k, v in model_params.items() if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']} if self.enable_optuna else None,
                'label_normalization_enabled': y_scaler is not None  # 🔴 方案C2: 记录是否使用了标签标准化
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
