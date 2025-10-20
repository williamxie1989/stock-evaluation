# -*- coding: utf-8 -*-
"""
Optuna超参数优化器
自动搜索最优模型超参数
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
import logging
import warnings

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

# Model imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Optuna超参数优化器
    
    支持XGBoost和LightGBM的分类/回归任务
    使用时间序列交叉验证确保无数据泄漏
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化器
        
        Parameters
        ----------
        config : dict, optional
            配置参数:
            - n_trials: 优化试验次数 (default: 100)
            - timeout: 超时时间秒数 (default: 3600)
            - sampler: 采样器类型 ('tpe'/'random'/'cmaes', default: 'tpe')
            - cv_folds: 交叉验证折数 (default: 5)
            - n_jobs: 并行任务数 (default: 1)
            - direction: 优化方向 ('maximize'/'minimize', default: 'maximize')
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for optimization. Install with: pip install optuna")
        
        config = config or {}
        self.n_trials = config.get('n_trials', 100)
        self.timeout = config.get('timeout', 3600)
        self.sampler_type = config.get('sampler', 'tpe')
        self.cv_folds = config.get('cv_folds', 5)
        self.n_jobs = config.get('n_jobs', 1)
        self.direction = config.get('direction', 'maximize')
        
        # 创建采样器
        if self.sampler_type == 'tpe':
            self.sampler = TPESampler(seed=42)
        elif self.sampler_type == 'random':
            self.sampler = RandomSampler(seed=42)
        elif self.sampler_type == 'cmaes':
            self.sampler = CmaEsSampler(seed=42)
        else:
            raise ValueError(f"未知采样器: {self.sampler_type}")
        
        # 创建pruner (提前终止表现差的试验)
        self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        logger.info(f"Optuna优化器初始化: {self.n_trials} trials, {self.sampler_type} sampler")
    
    def _analyze_data_complexity(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        分析数据复杂度，为参数调整提供依据
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            目标变量
            
        Returns
        -------
        complexity_metrics : dict
            数据复杂度指标:
            - n_samples: 样本数量
            - n_features: 特征数量
            - feature_density: 特征密度 (非零特征比例)
            - target_variance: 目标变量方差
            - complexity_score: 综合复杂度评分
        """
        n_samples, n_features = X.shape
        
        # 计算特征密度 (非零特征比例)
        if hasattr(X, 'sparse'):
            feature_density = (X != 0).sum().sum() / (n_samples * n_features)
        else:
            feature_density = 1.0  # 稠密矩阵
        
        # 目标变量方差
        target_variance = y.var() if len(y) > 1 else 0.0
        
        # 综合复杂度评分 (0-1范围)
        complexity_score = min(1.0, 
            (n_samples / 10000) * 0.3 +  # 样本规模影响
            (n_features / 100) * 0.3 +   # 特征规模影响
            feature_density * 0.2 +       # 特征密度影响
            (target_variance / 10) * 0.2  # 目标复杂度影响
        )
        
        metrics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_density': feature_density,
            'target_variance': target_variance,
            'complexity_score': complexity_score
        }
        
        logger.info(f"📊 数据复杂度分析: {n_samples}样本, {n_features}特征, "
                   f"密度{feature_density:.3f}, 复杂度评分{complexity_score:.3f}")
        
        return metrics
    
    def _adjust_parameter_ranges(self, base_params: Dict[str, Any], 
                                complexity_metrics: Dict[str, float],
                                model_type: str) -> Dict[str, Any]:
        """
        基于数据复杂度动态调整参数搜索范围
        
        Parameters
        ----------
        base_params : dict
            基础参数搜索范围
        complexity_metrics : dict
            数据复杂度指标
        model_type : str
            模型类型 ('xgboost_classification', 'xgboost_regression', 'lightgbm_classification')
            
        Returns
        -------
        adjusted_params : dict
            调整后的参数搜索范围
        """
        complexity_score = complexity_metrics['complexity_score']
        n_samples = complexity_metrics['n_samples']
        n_features = complexity_metrics['n_features']
        
        adjusted_params = base_params.copy()
        
        # 基于样本量调整参数
        if n_samples < 1000:
            # 小样本数据：减少模型复杂度
            if 'n_estimators' in adjusted_params:
                adjusted_params['n_estimators'] = (50, 300)  # 减少树的数量
            if 'max_depth' in adjusted_params:
                adjusted_params['max_depth'] = (2, 6)  # 降低树深度
            if 'learning_rate' in adjusted_params:
                adjusted_params['learning_rate'] = (0.01, 0.3)  # 提高学习率
        elif n_samples > 10000:
            # 大样本数据：增加模型复杂度
            if 'n_estimators' in adjusted_params:
                adjusted_params['n_estimators'] = (200, 1000)  # 增加树的数量
            if 'max_depth' in adjusted_params:
                adjusted_params['max_depth'] = (5, 12)  # 增加树深度
            if 'learning_rate' in adjusted_params:
                adjusted_params['learning_rate'] = (0.001, 0.2)  # 降低学习率
        
        # 基于特征数量调整正则化参数
        if n_features >= 50:
            # 高维特征：加强正则化
            if 'reg_alpha' in adjusted_params:
                adjusted_params['reg_alpha'] = (0.1, 20.0)  # 增加L1正则化
            if 'reg_lambda' in adjusted_params:
                adjusted_params['reg_lambda'] = (0.1, 20.0)  # 增加L2正则化
            if 'colsample_bytree' in adjusted_params:
                adjusted_params['colsample_bytree'] = (0.3, 0.8)  # 降低特征采样率
        
        # 基于复杂度评分微调
        if complexity_score > 0.7:
            # 高复杂度数据：增加正则化
            if 'gamma' in adjusted_params:
                adjusted_params['gamma'] = (1.0, 20.0)  # 增加分裂阈值
            if 'min_child_weight' in adjusted_params:
                adjusted_params['min_child_weight'] = (5, 30)  # 增加叶子节点最小样本
        
        logger.info(f"🔧 基于数据复杂度调整参数范围: {model_type}")
        logger.info(f"   样本量: {n_samples}, 特征数: {n_features}, 复杂度: {complexity_score:.3f}")
        
        return adjusted_params
    
    def optimize_xgboost_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        metric: str = 'auc'
    ) -> Dict[str, Any]:
        """
        优化XGBoost分类器超参数
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            分类标签
        dates : Series, optional
            日期序列（用于时间序列切分）
        metric : str, default='auc'
            优化指标 ('auc', 'f1')
        
        Returns
        -------
        result : dict
            包含 best_params, best_score, n_trials, study
        """
        logger.info("="*80)
        logger.info("开始XGBoost分类器超参数优化")
        logger.info("="*80)
        
        # 分析数据复杂度
        complexity_metrics = self._analyze_data_complexity(X, y)
        
        # 基础参数搜索范围
        base_params = {
            'n_estimators': (100, 800),  # 🔧 扩展到100-800
            'max_depth': (3, 5),
            'learning_rate': (0.01, 0.2),
            'subsample': (0.5, 0.75),
            'colsample_bytree': (0.5, 0.8),
            'min_child_weight': (5, 40),
            'gamma': (0.5, 10.0),
            'reg_alpha': (2, 20.0),
            'reg_lambda': (2, 20.0),
        }
        
        # 基于数据复杂度动态调整参数范围
        adjusted_params = self._adjust_parameter_ranges(base_params, complexity_metrics, 'xgboost_classification')
        
        def objective(trial):
            # 使用调整后的参数范围 - 🔧 第一阶段优化: 数据驱动参数调整
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *adjusted_params['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *adjusted_params['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *adjusted_params['learning_rate'], log=True),
                'subsample': trial.suggest_float('subsample', *adjusted_params['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *adjusted_params['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *adjusted_params['min_child_weight']),
                'gamma': trial.suggest_float('gamma', *adjusted_params['gamma']),
                'reg_alpha': trial.suggest_float('reg_alpha', *adjusted_params['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *adjusted_params['reg_lambda']),
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': 1  # 避免嵌套并行
            }
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = []
            
            # 处理numpy数组和pandas DataFrame
            if hasattr(X, 'iloc'):  # pandas DataFrame
                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:  # numpy数组
                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                
                # 训练模型
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # 评估
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                if metric == 'auc':
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    from sklearn.metrics import f1_score
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_val, y_pred)
                
                cv_scores.append(score)
                
                # 报告中间结果（用于pruning）
                trial.report(score, fold_idx)
                
                # 检查是否应该提前终止
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        # 创建study并优化
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        logger.info(f"✅ XGBoost分类优化完成:")
        logger.info(f"  最优{metric.upper()}: {study.best_value:.4f}")
        logger.info(f"  最优参数:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        logger.info(f"  总试验次数: {len(study.trials)}")
        logger.info(f"  剪枝试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'study': study
        }
    
    def optimize_xgboost_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        metric: str = 'r2'
    ) -> Dict[str, Any]:
        """
        优化XGBoost回归器超参数
        
        Parameters
        ----------
        X : DataFrame
            特征数据
        y : Series
            回归目标
        dates : Series, optional
            日期序列
        metric : str, default='r2'
            优化指标 ('r2', 'mse', 'mae')
        
        Returns
        -------
        result : dict
            包含 best_params, best_score, n_trials, study
        """
        logger.info("="*80)
        logger.info("开始XGBoost回归器超参数优化")
        logger.info("="*80)
        
        # 分析数据复杂度
        complexity_metrics = self._analyze_data_complexity(X, y)
        
        # 基础参数搜索范围
        base_params = {
            'n_estimators': (100, 800),  # 🔧 扩展到100-800
            'max_depth': (3, 8),  # 🔧 从3-4扩展到3-8
            'learning_rate': (0.005, 0.5),  # 🔧 扩展到0.005-0.5
            'subsample': (0.5, 0.9),  # 🔧 从0.6-0.8扩展到0.5-0.9
            'colsample_bytree': (0.5, 0.9),  # 🔧 从0.6-0.8扩展到0.5-0.9
            'min_child_weight': (1, 20),  # 🔧 扩展到1-20
            'gamma': (0.0, 15.0),  # 🔧 扩展到0-15
            'reg_alpha': (0.0, 15.0),  # 🔧 从3-10扩展到0-15
            'reg_lambda': (0.0, 15.0),  # 🔧 从5-10扩展到0-15
        }
        
        # 基于数据复杂度动态调整参数范围
        adjusted_params = self._adjust_parameter_ranges(base_params, complexity_metrics, 'xgboost_regression')
        
        def objective(trial):
            # 使用调整后的参数范围 - 🔧 第一阶段优化: 数据驱动参数调整
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *adjusted_params['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *adjusted_params['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *adjusted_params['learning_rate'], log=True),
                'subsample': trial.suggest_float('subsample', *adjusted_params['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *adjusted_params['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *adjusted_params['min_child_weight']),
                'gamma': trial.suggest_float('gamma', *adjusted_params['gamma']),
                'reg_alpha': trial.suggest_float('reg_alpha', *adjusted_params['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *adjusted_params['reg_lambda']),
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': 1
            }
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = []
            
            # 处理numpy数组和pandas DataFrame
            if hasattr(X, 'iloc'):  # pandas DataFrame
                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:  # numpy数组
                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                
                if metric == 'r2':
                    score = r2_score(y_val, y_pred)
                elif metric == 'mse':
                    score = -mean_squared_error(y_val, y_pred)  # 负值因为要最大化
                else:  # mae
                    from sklearn.metrics import mean_absolute_error
                    score = -mean_absolute_error(y_val, y_pred)
                
                cv_scores.append(score)
                trial.report(score, fold_idx)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        logger.info(f"✅ XGBoost回归优化完成:")
        logger.info(f"  最优{metric.upper()}: {study.best_value:.4f}")
        logger.info(f"  最优参数:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        logger.info(f"  总试验次数: {len(study.trials)}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'study': study
        }
    
    def optimize_lightgbm_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        metric: str = 'auc'
    ) -> Dict[str, Any]:
        """
        优化LightGBM分类器超参数
        """
        logger.info("="*80)
        logger.info("开始LightGBM分类器超参数优化")
        logger.info("="*80)
        
        # 分析数据复杂度
        complexity_metrics = self._analyze_data_complexity(X, y)
        
        # 基础参数搜索范围
        base_params = {
            'n_estimators': (100, 800),  # 🔧 扩展到100-800
            'max_depth': (3, 8),  # 🔧 从3-4扩展到3-8
            'learning_rate': (0.005, 0.5),  # 🔧 扩展到0.005-0.5
            'subsample': (0.5, 0.9),  # 🔧 从0.6-0.8扩展到0.5-0.9
            'colsample_bytree': (0.5, 0.9),  # 🔧 从0.6-0.8扩展到0.5-0.9
            'min_child_samples': (10, 150),  # 🔧 扩展到10-150
            'reg_alpha': (0.0, 15.0),  # 🔧 从3-10扩展到0-15
            'reg_lambda': (0.0, 15.0),  # 🔧 从5-10扩展到0-15
        }
        
        # 基于数据复杂度动态调整参数范围
        adjusted_params = self._adjust_parameter_ranges(base_params, complexity_metrics, 'lightgbm_classification')
        
        def objective(trial):
            # 使用调整后的参数范围 - 🔧 第一阶段优化: 数据驱动参数调整
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *adjusted_params['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *adjusted_params['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *adjusted_params['learning_rate'], log=True),
                'subsample': trial.suggest_float('subsample', *adjusted_params['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *adjusted_params['colsample_bytree']),
                'min_child_samples': trial.suggest_int('min_child_samples', *adjusted_params['min_child_samples']),
                'reg_alpha': trial.suggest_float('reg_alpha', *adjusted_params['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *adjusted_params['reg_lambda']),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': 1
            }
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                if metric == 'auc':
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    from sklearn.metrics import f1_score
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_val, y_pred)
                
                cv_scores.append(score)
                trial.report(score, fold_idx)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        logger.info(f"✅ LightGBM分类优化完成:")
        logger.info(f"  最优{metric.upper()}: {study.best_value:.4f}")
        logger.info(f"  最优参数: {study.best_params}")
        logger.info(f"  总试验次数: {len(study.trials)}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'study': study
        }
    
    def visualize_optimization(self, study: optuna.Study, save_path: Optional[str] = None):
        """
        可视化优化过程
        
        Parameters
        ----------
        study : optuna.Study
            完成的study对象
        save_path : str, optional
            保存路径前缀
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice
            )
            
            logger.info("生成Optuna可视化图表...")
            
            # 优化历史
            fig1 = plot_optimization_history(study)
            if save_path:
                fig1.write_html(f"{save_path}_history.html")
            
            # 参数重要性
            fig2 = plot_param_importances(study)
            if save_path:
                fig2.write_html(f"{save_path}_importance.html")
            
            # 平行坐标图
            fig3 = plot_parallel_coordinate(study)
            if save_path:
                fig3.write_html(f"{save_path}_parallel.html")
            
            logger.info(f"✅ 可视化完成，保存至: {save_path}_*.html")
            
        except Exception as e:
            logger.warning(f"可视化失败: {e}")
