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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# 导入配置
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import *
from src.ml.preprocessing.winsorizer import Winsorizer
from src.ml.preprocessing.feature_selection import select_features_separately

logger = logging.getLogger(__name__)


class EnhancedTrainerV2:
    """
    增强版训练器 V2
    
    核心改进：
    1. 统一预处理管道（ColumnTransformer）
    2. 分类/回归独立特征选择
    3. 概率校准（Isotonic/Platt）
    4. 新模型持久化格式
    """
    
    def __init__(self,
                 numerical_features: List[str],
                 categorical_features: List[str],
                 config: Optional[Dict] = None):
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
        self.config = config or self._get_default_config()
        
        logger.info("初始化 EnhancedTrainerV2")
        logger.info(f"  数值特征: {len(numerical_features)}")
        logger.info(f"  类别特征: {len(categorical_features)}")
    
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
            'prediction_period': PREDICTION_PERIOD_DAYS
        }
    
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
            ('scaler', StandardScaler())
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
        
        return preprocessor
    
    def train_classification_model(self,
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     model_type: str = 'lightgbm',
                                     **model_params) -> Dict:
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
        
        # 切分数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"正样本比例: 训练集 {y_train.mean():.2%}, 验证集 {y_val.mean():.2%}")
        
        # 创建预处理器
        preprocessor = self.create_preprocessing_pipeline()
        
        # 创建模型
        if model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42,
                verbosity=-1
            )
        elif model_type == 'xgboost':
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42,
                verbosity=0
            )
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 创建完整 pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        
        # 训练
        logger.info("训练模型...")
        pipeline.fit(X_train, y_train)
        
        # 预测（用于评估和校准）
        y_pred_train = pipeline.predict_proba(X_train)[:, 1]
        y_pred_val = pipeline.predict_proba(X_val)[:, 1]
        
        # 评估
        from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
        
        metrics = {
            'train_auc': roc_auc_score(y_train, y_pred_train),
            'val_auc': roc_auc_score(y_val, y_pred_val),
            'train_f1': f1_score(y_train, (y_pred_train > 0.5).astype(int)),
            'val_f1': f1_score(y_val, (y_pred_val > 0.5).astype(int)),
            'train_brier': brier_score_loss(y_train, y_pred_train),
            'val_brier': brier_score_loss(y_val, y_pred_val)
        }
        
        logger.info("训练集评估:")
        logger.info(f"  AUC: {metrics['train_auc']:.4f}")
        logger.info(f"  F1:  {metrics['train_f1']:.4f}")
        logger.info(f"  Brier: {metrics['train_brier']:.4f}")
        
        logger.info("验证集评估:")
        logger.info(f"  AUC: {metrics['val_auc']:.4f}")
        logger.info(f"  F1:  {metrics['val_f1']:.4f}")
        logger.info(f"  Brier: {metrics['val_brier']:.4f}")
        
        # 概率校准
        calibrator = None
        if ENABLE_CALIBRATION:
            logger.info(f"应用概率校准: {CALIBRATION_METHOD}")
            calibrator = CalibratedClassifierCV(
                estimator=pipeline,
                method=CALIBRATION_METHOD,
                cv='prefit'
            )
            calibrator.fit(X_val, y_val)
            
            # 评估校准后的性能
            y_pred_cal = calibrator.predict_proba(X_val)[:, 1]
            cal_brier = brier_score_loss(y_val, y_pred_cal)
            
            logger.info(f"校准后 Brier: {cal_brier:.4f} (改善: {metrics['val_brier'] - cal_brier:.4f})")
            metrics['val_brier_calibrated'] = cal_brier
        
        # 特征选择（如果需要）
        selected_features = None
        if self.config['enable_feature_selection']:
            logger.info("执行特征选择...")
            from src.ml.preprocessing.feature_selection import select_features_for_task
            
            selected_features, _ = select_features_for_task(
                X_train[self.numerical_features + self.categorical_features],
                y_train,
                task='classification',
                min_features=self.config['min_features'],
                max_features=self.config['max_features']
            )
        
        # 返回结果
        result = {
            'task': 'classification',
            'model_type': model_type,
            'pipeline': pipeline,
            'calibrator': calibrator,
            'selected_features': selected_features,
            'metrics': metrics,
            'threshold': self.config['cls_threshold'],
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy()
        }
        
        logger.info("✅ 分类模型训练完成\n")
        
        return result
    
    def train_regression_model(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                model_type: str = 'lightgbm',
                                **model_params) -> Dict:
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
        
        # 切分数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"数据切分: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
        logger.info(f"目标统计: 均值 {y_train.mean():.4f}, 标准差 {y_train.std():.4f}")
        
        # 创建预处理器
        preprocessor = self.create_preprocessing_pipeline()
        
        # 创建模型
        if model_type == 'lightgbm':
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42,
                verbosity=-1
            )
        elif model_type == 'xgboost':
            from xgboost import XGBRegressor
            estimator = XGBRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 5),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42,
                verbosity=0
            )
        elif model_type == 'ridge':
            from sklearn.linear_model import Ridge
            estimator = Ridge(alpha=model_params.get('alpha', 1.0))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 创建完整 pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ])
        
        # 训练
        logger.info("训练模型...")
        pipeline.fit(X_train, y_train)
        
        # 预测
        y_pred_train = pipeline.predict(X_train)
        y_pred_val = pipeline.predict(X_val)
        
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
        
        # 特征选择（如果需要）
        selected_features = None
        if self.config['enable_feature_selection']:
            logger.info("执行特征选择...")
            from src.ml.preprocessing.feature_selection import select_features_for_task
            
            selected_features, _ = select_features_for_task(
                X_train[self.numerical_features + self.categorical_features],
                y_train,
                task='regression',
                min_features=self.config['min_features'],
                max_features=self.config['max_features']
            )
        
        # 返回结果
        result = {
            'task': 'regression',
            'model_type': model_type,
            'pipeline': pipeline,
            'calibrator': None,  # 回归不需要校准
            'selected_features': selected_features,
            'metrics': metrics,
            'threshold': None,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'config': self.config.copy()
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
