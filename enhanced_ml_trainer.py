#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型机器学习训练器 - 第一阶段优化
集成RandomForest、XGBoost、LightGBM等先进模型
支持贝叶斯超参数优化和模型集成
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
import joblib

# 导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM未安装，将跳过LightGBM模型")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna未安装，将使用网格搜索替代贝叶斯优化")

from db import DatabaseManager
from enhanced_features import EnhancedFeatureGenerator
from features import FeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """
    增强型机器学习训练器
    支持多种先进模型和优化技术
    """
    
    def __init__(self, db_manager: DatabaseManager = None, model_dir: str = "models", 
                 use_enhanced_features: bool = True, use_bayesian_optimization: bool = True):
        self.db_manager = db_manager or DatabaseManager()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.use_bayesian_optimization = use_bayesian_optimization and OPTUNA_AVAILABLE
        
        # 根据参数选择特征生成器
        if use_enhanced_features:
            self.feature_generator = EnhancedFeatureGenerator()
            logger.info("使用增强特征生成器")
        else:
            self.feature_generator = FeatureGenerator()
            logger.info("使用基础特征生成器")
        
        # 模型配置
        self.model_configs = {
            'logistic': {
                'model_class': LogisticRegression,
                'default_params': {
                    'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear',
                    'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42
                },
                'tuning_params': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'class_weight': ['balanced', None]
                }
            },
            'randomforest': {
                'model_class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
                    'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1
                },
                'tuning_params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        # 添加XGBoost配置
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model_class': xgb.XGBClassifier,
                'default_params': {
                    'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                    'eval_metric': 'logloss'
                },
                'tuning_params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            }
        
        # 添加LightGBM配置
        if LIGHTGBM_AVAILABLE:
            self.model_configs['lightgbm'] = {
                'model_class': lgb.LGBMClassifier,
                'default_params': {
                    'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                    'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'random_state': 42, 'verbose': -1
                },
                'tuning_params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [20, 31, 50],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            }
    
    def load_samples_from_db(self, symbols: List[str] = None, 
                           period: str = '10d',
                           start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
        """从数据库加载训练样本"""
        try:
            query = """
            SELECT symbol, date, period, label, forward_return, features
            FROM samples
            WHERE period = ?
            """
            params = [period]
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY symbol, date"
            
            with self.db_manager.get_conn() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if df.empty:
                logger.warning("未找到匹配的样本数据")
                return pd.DataFrame()
                
            # 解析特征JSON
            features_list = []
            for _, row in df.iterrows():
                try:
                    features = json.loads(row['features'])
                    features['symbol'] = row['symbol']
                    features['date'] = row['date']
                    features['label'] = row['label']
                    features['forward_return'] = row['forward_return']
                    features_list.append(features)
                except Exception as e:
                    logger.warning(f"解析特征失败: {e}")
                    continue
                    
            if not features_list:
                logger.warning("没有有效的特征数据")
                return pd.DataFrame()
                
            result_df = pd.DataFrame(features_list)
            logger.info(f"加载样本数据: {len(result_df)} 条, 期数: {period}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"加载样本数据失败: {e}")
            return pd.DataFrame()
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """准备特征和目标变量"""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []
            
        exclude_cols = ['symbol', 'date', 'label', 'forward_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        # 移除常数特征
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            logger.info(f"移除常数特征: {constant_features}")
            X = X.drop(columns=constant_features)
            feature_cols = [c for c in feature_cols if c not in constant_features]
        
        logger.info(f"特征准备完成: {X.shape[1]} 个特征, {len(y)} 个样本, 目标列: {target_col}")
        
        if target_col == 'label':
            logger.info(f"标签分布: {y.value_counts().to_dict()}")
        else:
            logger.info(f"目标统计: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
            
        return X, y, feature_cols
    
    def create_pipeline(self, model_type: str, params: Dict = None) -> Pipeline:
        """创建训练管线"""
        if model_type not in self.model_configs:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        config = self.model_configs[model_type]
        model_class = config['model_class']
        
        if params is None:
            params = config['default_params']
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_type, model_class(**params))
        ])
        
        return pipeline
    
    def bayesian_optimization(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_type: str, n_trials: int = 50) -> Dict[str, Any]:
        """贝叶斯超参数优化"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna不可用，回退到网格搜索")
            return self.grid_search_optimization(X_train, y_train, model_type)
        
        config = self.model_configs[model_type]
        
        def objective(trial):
            # 根据模型类型建议参数
            if model_type == 'logistic':
                params = {
                    'C': trial.suggest_loguniform('C', 0.01, 100.0),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
                }
            elif model_type == 'randomforest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
                }
            
            # 创建模型并训练
            pipeline = self.create_pipeline(model_type, params)
            
            # 使用TimeSeriesSplit进行交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                pipeline.fit(X_tr, y_tr)
                y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    scores.append(auc)
                except:
                    scores.append(0.5)
            
            return np.mean(scores)
        
        # 运行优化
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"贝叶斯优化完成: 最优AUC = {study.best_value:.4f}")
        logger.info(f"最优参数: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def grid_search_optimization(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               model_type: str) -> Dict[str, Any]:
        """网格搜索优化"""
        config = self.model_configs[model_type]
        tuning_params = config['tuning_params']
        
        pipeline = self.create_pipeline(model_type)
        
        # 使用TimeSeriesSplit进行交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            pipeline, tuning_params, cv=tscv,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"网格搜索完成: 最优AUC = {grid_search.best_score_:.4f}")
        logger.info(f"最优参数: {grid_search.best_params_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'grid_search': grid_search
        }
    
    def train_single_model(self, X: pd.DataFrame, y: pd.Series, model_type: str,
                          test_size: float = 0.2, use_optimization: bool = True,
                          optimization_trials: int = 50) -> Dict[str, Any]:
        """训练单个模型"""
        try:
            if X.empty or len(y) == 0:
                raise ValueError("特征或标签数据为空")
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"训练{model_type}模型: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
            
            # 超参数优化
            if use_optimization:
                if self.use_bayesian_optimization:
                    optimization_result = self.bayesian_optimization(
                        X_train, y_train, model_type, optimization_trials
                    )
                else:
                    optimization_result = self.grid_search_optimization(X_train, y_train, model_type)
                
                best_params = optimization_result['best_params']
                best_cv_score = optimization_result['best_score']
            else:
                best_params = self.model_configs[model_type]['default_params']
                best_cv_score = None
            
            # 训练最终模型
            final_pipeline = self.create_pipeline(model_type, best_params)
            final_pipeline.fit(X_train, y_train)
            
            # 预测和评估
            y_train_pred = final_pipeline.predict(X_train)
            y_test_pred = final_pipeline.predict(X_test)
            y_train_proba = final_pipeline.predict_proba(X_train)[:, 1]
            y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
            
            # 计算详细指标
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            logger.info(f"{model_type}模型性能:")
            logger.info(f"  训练集 - AUC: {train_auc:.4f}, 准确率: {train_accuracy:.4f}, F1: {train_f1:.4f}")
            logger.info(f"  测试集 - AUC: {test_auc:.4f}, 准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}")
            
            # 特征重要性
            if hasattr(final_pipeline.named_steps[model_type], 'feature_importances_'):
                feature_importance = dict(zip(X.columns, final_pipeline.named_steps[model_type].feature_importances_))
            elif hasattr(final_pipeline.named_steps[model_type], 'coef_'):
                # 对于线性模型，使用系数绝对值
                coef_abs = np.abs(final_pipeline.named_steps[model_type].coef_[0])
                feature_importance = dict(zip(X.columns, coef_abs))
            else:
                feature_importance = {}
            
            # 构建结果
            result = {
                'model': final_pipeline,
                'model_type': model_type,
                'best_params': best_params,
                'cv_score': best_cv_score,
                'feature_names': X.columns.tolist(),
                'feature_importance': feature_importance,
                'metrics': {
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_precision': train_precision,
                    'test_precision': test_precision,
                    'train_recall': train_recall,
                    'test_recall': test_recall,
                    'train_f1': train_f1,
                    'test_f1': test_f1
                },
                'predictions': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba
                },
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"{model_type}模型训练失败: {e}")
            raise
    
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series, 
                           model_types: List[str] = None,
                           test_size: float = 0.2,
                           use_optimization: bool = True) -> Dict[str, Any]:
        """训练集成模型"""
        if model_types is None:
            model_types = ['logistic', 'randomforest']
            if XGBOOST_AVAILABLE:
                model_types.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_types.append('lightgbm')
        
        logger.info(f"训练集成模型: {model_types}")
        
        # 训练各个基模型
        base_models = []
        individual_results = {}
        
        for model_type in model_types:
            try:
                result = self.train_single_model(
                    X, y, model_type, test_size, use_optimization, optimization_trials=30
                )
                individual_results[model_type] = result
                
                # 提取模型用于集成
                base_models.append((model_type, result['model']))
                
            except Exception as e:
                logger.error(f"训练{model_type}模型失败: {e}")
                continue
        
        if len(base_models) < 2:
            logger.warning("基模型数量不足，无法构建集成模型")
            return individual_results[list(individual_results.keys())[0]]
        
        # 创建投票分类器
        voting_classifier = VotingClassifier(
            estimators=base_models,
            voting='soft',  # 使用概率投票
            n_jobs=-1
        )
        
        # 训练集成模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        voting_classifier.fit(X_train, y_train)
        
        # 评估集成模型
        y_test_pred = voting_classifier.predict(X_test)
        y_test_proba = voting_classifier.predict_proba(X_test)[:, 1]
        
        ensemble_auc = roc_auc_score(y_test, y_test_proba)
        ensemble_accuracy = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"集成模型性能: AUC = {ensemble_auc:.4f}, 准确率 = {ensemble_accuracy:.4f}")
        
        # 构建集成结果
        ensemble_result = {
            'model': voting_classifier,
            'model_type': 'ensemble',
            'base_models': individual_results,
            'base_model_types': [name for name, _ in base_models],
            'metrics': {
                'test_auc': ensemble_auc,
                'test_accuracy': ensemble_accuracy,
                'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
            },
            'predictions': {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }
        }
        
        return ensemble_result
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series, 
                      model_types: List[str] = None,
                      test_size: float = 0.2) -> Dict[str, Any]:
        """对比多个模型的性能"""
        if model_types is None:
            model_types = ['logistic', 'randomforest']
            if XGBOOST_AVAILABLE:
                model_types.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_types.append('lightgbm')
        
        logger.info(f"开始模型对比: {model_types}")
        
        results = {}
        comparison_metrics = []
        
        for model_type in model_types:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"训练 {model_type.upper()} 模型")
                logger.info(f"{'='*50}")
                
                result = self.train_single_model(X, y, model_type, test_size, use_optimization=True)
                results[model_type] = result
                
                # 收集对比指标
                metrics = result['metrics']
                comparison_metrics.append({
                    'model_type': model_type,
                    'test_auc': metrics['test_auc'],
                    'test_accuracy': metrics['test_accuracy'],
                    'test_precision': metrics['test_precision'],
                    'test_recall': metrics['test_recall'],
                    'test_f1': metrics['test_f1'],
                    'cv_score': result.get('cv_score', None)
                })
                
            except Exception as e:
                logger.error(f"训练{model_type}模型失败: {e}")
                continue
        
        # 创建对比DataFrame
        comparison_df = pd.DataFrame(comparison_metrics)
        comparison_df = comparison_df.sort_values('test_auc', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("模型性能对比结果")
        logger.info(f"{'='*60}")
        logger.info(f"\n{comparison_df.to_string(index=False, float_format='%.4f')}")
        
        # 找出最佳模型
        best_model_type = comparison_df.iloc[0]['model_type']
        best_result = results[best_model_type]
        
        return {
            'all_results': results,
            'comparison_df': comparison_df,
            'best_model_type': best_model_type,
            'best_result': best_result
        }
    
    def save_model(self, model_result: Dict[str, Any], model_name: str = None):
        """保存训练好的模型"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = model_result['model_type']
            model_name = f"enhanced_{model_type}_{timestamp}.pkl"
        
        model_path = self.model_dir / model_name
        
        # 保存模型结果
        with open(model_path, 'wb') as f:
            pickle.dump(model_result, f)
        
        logger.info(f"模型已保存: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """加载保存的模型"""
        with open(model_path, 'rb') as f:
            model_result = pickle.load(f)
        
        logger.info(f"模型已加载: {model_path}")
        return model_result


def main():
    """测试函数"""
    # 初始化增强训练器
    trainer = EnhancedMLTrainer(use_bayesian_optimization=True)
    
    # 加载样本数据
    logger.info("加载样本数据...")
    samples = trainer.load_samples_from_db(period='10d', start_date='2024-01-01', end_date='2024-06-01')
    
    if len(samples) < 100:
        logger.error("样本数据不足")
        return
    
    # 准备特征
    X, y, features = trainer.prepare_features_and_target(samples, 'label')
    logger.info(f"特征数量: {len(features)}, 样本数量: {len(X)}")
    
    # 对比多个模型
    logger.info("开始模型对比...")
    comparison_result = trainer.compare_models(X, y, test_size=0.2)
    
    # 保存最佳模型
    best_result = comparison_result['best_result']
    model_path = trainer.save_model(best_result)
    
    logger.info(f"优化完成！最佳模型: {comparison_result['best_model_type']}")
    logger.info(f"模型已保存至: {model_path}")


if __name__ == "__main__":
    main()