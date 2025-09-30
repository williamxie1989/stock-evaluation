#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型训练脚本
整合分类和回归模型训练功能
适配现有数据接入和数据库访问方式
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any, Union
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import warnings

# 添加项目根目录到路径
# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.unified_data_access import UnifiedDataAccessLayer
from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.core.unified_data_access_factory import create_unified_data_access
from src.ml.features.enhanced_features import EnhancedFeatureGenerator
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from src.ml.training.enhanced_ml_trainer import EnhancedMLTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('unified_training.log')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.WARNING)

class UnifiedModelTrainer:
    """统一模型训练器，支持分类和回归任务"""
    
    def __init__(self, enable_feature_selection: bool = True):
        """初始化统一模型训练器
        Args:
            enable_feature_selection: 是否在数据预处理阶段启用特征选择优化器
        """
        # 使用现有的统一数据访问层
        self.data_access = create_unified_data_access()
        self.db_manager = self.data_access.db_manager
        self.feature_generator = EnhancedFeatureGenerator()
        # 新增: 特征选择开关
        self.enable_feature_selection = enable_feature_selection
    
    def prepare_training_data(self, stock_list: List[str] = None, mode: str = 'both', lookback_days: int = 365, 
                            n_stocks: int = 1000, prediction_period: int = 30) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        准备训练数据 - 添加数据获取限制和快速失败机制
        
        Args:
            mode: 'classification', 'regression', 或 'both'
            lookback_days: 回溯天数
            n_stocks: 使用的股票数量
            prediction_period: 预测周期（天数）
            
        Returns:
            X: 特征数据
            y: 标签数据字典
        """
        logger.info(f"开始准备训练数据 (模式: {mode})...")
        
        # 获取股票列表 - 使用统一数据访问层
        stock_list = self.data_access.get_all_stock_list()
        if stock_list is None or stock_list.empty:
            raise ValueError("无法获取股票列表")
        
        if 'symbol' not in stock_list.columns:
            raise ValueError("股票列表缺少 'symbol' 列，请检查数据访问层是否已生成标准化代码")
        symbols = stock_list['symbol'].tolist()
        logger.info(f"获取到 {len(symbols)} 只股票，使用前 {n_stocks} 只")
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        logger.info(f"数据范围: {start_date.date()} 到 {end_date.date()}")
        
        all_features = []
        all_cls_labels = []
        all_reg_labels = []
        failed_stocks = []  # 记录失败的股票
        max_consecutive_failures = 3  # 连续失败阈值
        consecutive_failures = 0
        
        # 处理每只股票
        for i, symbol in enumerate(symbols[:n_stocks]):
            if i % 10 == 0:
                logger.info(f"处理第 {i+1}/{min(n_stocks, len(symbols))} 只股票: {symbol}")
            
            # 检查是否连续失败过多，跳过当前股票
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(f"连续失败过多 ({consecutive_failures})，跳过当前股票 {symbol}")
                consecutive_failures = 0  # 重置计数器，继续处理下一只股票
                continue
            
            try:
                # 获取股票数据 - 使用统一数据访问层，关闭自动同步以避免循环
                stock_data = self.data_access.get_stock_data(symbol, start_date, end_date, auto_sync=False)

                # 至少保证 prediction_period+15 天的数据用于生成标签，若不足则跳过
                min_required_len = prediction_period + 15  # 进一步降低要求到15天，适配宽松的特征生成器
                if stock_data is None or stock_data.empty or len(stock_data) < min_required_len:
                    failed_stocks.append(symbol)
                    consecutive_failures += 1
                    continue
                
                # 重置连续失败计数
                consecutive_failures = 0
                
                # 重置索引以确保日期列存在
                stock_data = stock_data.reset_index()
                
                # 确保数据类型正确
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in stock_data.columns:
                        # 转为 float，避免 Decimal 类型导致后续计算报错
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').astype(float)
                
                # 删除包含NaN的行，仅针对关键数值列，避免因无关列缺失导致数据量骤减
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                stock_data = stock_data.dropna(subset=required_cols)
                
                # 生成增强特征
                features_df = self.feature_generator.generate_features(stock_data)
                if features_df.empty:
                    logger.warning(f"{symbol} 生成增强特征失败，尝试基础特征生成器")
                    from src.ml.features.feature_generator import FeatureGenerator
                    basic_generator = FeatureGenerator()
                    features_df = basic_generator.generate_all_features(stock_data)
                    if features_df.empty:
                        logger.warning(f"{symbol} 基础特征生成也失败，跳过")
                        failed_stocks.append(symbol)
                        consecutive_failures += 1
                        continue
                
                # 检查特征数量，如果太少则尝试生成基础特征
                if len(features_df.columns) < 10:
                    logger.warning(f"{symbol} 特征数量较少({len(features_df.columns)}个)，尝试生成基础特征")
                    # 尝试单独生成各类特征
                    tech_features = basic_generator.generate_technical_features(stock_data)
                    if not tech_features.empty:
                        features_df = tech_features
                        logger.info(f"{symbol} 使用技术指标特征({len(features_df.columns)}个)")
                
                # 生成标签
                close_prices = stock_data['close'].values
                reg_labels = []
                cls_labels = []
                
                for j in range(len(close_prices) - prediction_period):
                    current_price = close_prices[j]
                    future_price = close_prices[j + prediction_period]
                    return_rate = (future_price - current_price) / current_price
                    
                    reg_labels.append(return_rate)
                    cls_labels.append(1 if return_rate > 0.05 else 0)  # 5%阈值
                
                # 对齐特征和标签
                aligned_features = features_df.iloc[:-prediction_period].copy()
                
                if len(aligned_features) != len(reg_labels):
                    min_len = min(len(aligned_features), len(reg_labels))
                    aligned_features = aligned_features.iloc[:min_len]
                    reg_labels = reg_labels[:min_len]
                    cls_labels = cls_labels[:min_len]
                
                # 若对齐后为空则跳过
                if aligned_features.empty:
                    failed_stocks.append(symbol)
                    continue
                # 添加股票标识
                aligned_features['symbol'] = symbol
                
                all_features.append(aligned_features)
                all_reg_labels.extend(reg_labels)
                all_cls_labels.extend(cls_labels)
                
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
                import traceback
                logger.warning(f"详细错误信息: {traceback.format_exc()}")
                failed_stocks.append(symbol)
                consecutive_failures += 1
        
        if not all_features:
            logger.error(f"特征生成失败，失败股票数: {len(failed_stocks)}，尝试放宽过滤条件或检查数据源")
            raise ValueError("没有生成任何训练数据")
        
        # 合并所有数据
        X_combined = pd.concat(all_features, ignore_index=1)
        y_reg = pd.Series(all_reg_labels, name='return_rate')
        y_cls = pd.Series(all_cls_labels, name='label_cls')
        
        logger.info(f"总共生成 {len(X_combined)} 个样本，失败股票数: {len(failed_stocks)}")
        if failed_stocks:
            logger.info(f"失败股票列表: {failed_stocks}")
        
        # 数据预处理
        X_processed, y_reg_processed, y_cls_processed = self._preprocess_data(X_combined, y_reg, y_cls)
        
        # 根据模式返回相应的标签
        y_dict = {}
        if mode in ['classification', 'both']:
            y_dict['cls'] = y_cls_processed
        if mode in ['regression', 'both']:
            y_dict['reg'] = y_reg_processed
        
        return X_processed, y_dict
    
    def _preprocess_data(self, X: pd.DataFrame, y_reg: pd.Series, y_cls: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """数据预处理 - 包含特征选择优化"""
        logger.info("开始数据预处理...")
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        
        # 处理无穷大和NaN
        X_cleaned = X_numeric.replace([np.inf, -np.inf], np.nan)
        
        # ---------------- 新增: 先按列统计缺失比例，删除缺失过多的列 ----------------
        na_ratio = X_cleaned.isna().mean()
        cols_to_drop = na_ratio[na_ratio > 0.5].index  # 若某列超过50%缺失，则直接删除
        if len(cols_to_drop) > 0:
            logger.info(f"删除缺失过多的列: {list(cols_to_drop)} (阈值50%)")
        X_reduced = X_cleaned.drop(columns=cols_to_drop)
        
        # 数据对齐
        X_imputed = X_reduced.fillna(X_reduced.median())
        
        # 移除仍包含NaN的行（极端情况下中位数为NaN 或全部缺失）
        valid_indices = X_imputed.notna().all(axis=1)
        X_final = X_imputed[valid_indices]
        
        # 使用索引对齐标签数据
        valid_mask = pd.Series(valid_indices, index=X_imputed.index)
        y_reg_final = y_reg[valid_mask]
        y_cls_final = y_cls[valid_mask]
        
        logger.info(f"清洗后样本数量: {len(X_final)}")
        
        # 移除极端异常值（收益率在±50%之外）
        reasonable_returns = (y_reg_final.abs() <= 0.5)
        X_final = X_final[reasonable_returns]
        y_reg_final = y_reg_final[reasonable_returns]
        y_cls_final = y_cls_final[reasonable_returns]
        
        logger.info(f"移除极端值后样本数量: {len(X_final)}")
        logger.info(f"收益率范围: {y_reg_final.min():.3f} 到 {y_reg_final.max():.3f}")
        logger.info(f"收益率均值: {y_reg_final.mean():.6f}, 标准差: {y_reg_final.std():.6f}")
        logger.info(f"正样本比例: {y_cls_final.mean():.3f}")
        
        # 标准化特征名称
        X_final.columns = [f'feature_{i}' for i in range(X_final.shape[1])]

        # 如果关闭特征选择，直接返回
        if not getattr(self, 'enable_feature_selection', True):
            logger.info("已关闭特征选择优化，直接返回清洗后的全部特征")
            return X_final, y_reg_final, y_cls_final

        # 特征选择优化
        logger.info("开始特征选择优化...")
        try:
            from src.ml.features.feature_selector_optimizer import FeatureSelectorOptimizer
            
            # 为分类和回归任务分别选择特征
            n_features = min(50, X_final.shape[1])  # 选择最多50个特征
            
            # 分类特征选择
            if len(np.unique(y_cls_final)) > 1:  # 确保有多于一个类别
                cls_selector = FeatureSelectorOptimizer(
                    task_type='classification', 
                    target_n_features=n_features
                )
                cls_results = cls_selector.optimize_feature_selection(
                    X_final, y_cls_final, method='auto'
                )
                cls_features = cls_results['selected_features']
                logger.info(f"分类任务选择了 {len(cls_features)} 个特征")
            else:
                cls_features = X_final.columns.tolist()
            
            # 回归特征选择
            reg_selector = FeatureSelectorOptimizer(
                task_type='regression', 
                target_n_features=n_features
            )
            reg_results = reg_selector.optimize_feature_selection(
                X_final, y_reg_final, method='auto'
            )
            reg_features = reg_results['selected_features']
            logger.info(f"回归任务选择了 {len(reg_features)} 个特征")
            
            # 合并两个任务的特征
            all_selected_features = list(set(cls_features + reg_features))
            X_selected = X_final[all_selected_features]
            
            logger.info(f"特征选择优化完成: 从 {X_final.shape[1]} 个特征中选择 {len(all_selected_features)} 个")
            logger.info(f"特征缩减比例: {(1 - len(all_selected_features)/X_final.shape[1])*100:.1f}%")
            
            return X_selected, y_reg_final, y_cls_final
            
        except Exception as e:
            logger.warning(f"特征选择优化失败: {e}，使用全部特征")
            return X_final, y_reg_final, y_cls_final
    
    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series],
                 mode: str = 'both', use_grid_search: bool = 1,
                 use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """
        训练模型（多线程优化版本）
        
        Args:
            X: 特征数据
            y: 标签数据字典
            mode: 'classification', 'regression', 或 'both'
            use_grid_search: 是否使用网格搜索
            
        Returns:
            results: 训练结果字典
        """
        logger.info(f"开始训练模型 (模式: {mode})...")
        
        results = {}
        
        if mode == 'both':
            # 并行训练分类和回归模型
            logger.info("并行训练分类和回归模型...")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交分类和回归训练任务
                future_to_task = {}
                
                if 'cls' in y:
                    future_to_task[executor.submit(self._train_classification_models, 
                                                  X, y['cls'], use_grid_search, use_optuna, optimization_trials)] = 'classification'
                
                if 'reg' in y:
                    future_to_task[executor.submit(self._train_regression_models, 
                                                  X, y['reg'], use_grid_search, use_optuna, optimization_trials)] = 'regression'
                
                # 收集结果
                for future in as_completed(future_to_task):
                    task_type = future_to_task[future]
                    try:
                        task_results = future.result()
                        results.update(task_results)
                        logger.info(f"{task_type} 模型训练完成")
                    except Exception as e:
                        logger.error(f"{task_type} 模型训练异常: {e}")
        
        else:
            # 单独训练分类或回归模型
            if mode == 'classification' and 'cls' in y:
                logger.info("训练分类模型...")
                cls_results = self._train_classification_models(X, y['cls'], use_grid_search, use_optuna, optimization_trials)
                results.update(cls_results)
            
            if mode == 'regression' and 'reg' in y:
                logger.info("训练回归模型...")
                reg_results = self._train_regression_models(X, y['reg'], use_grid_search, use_optuna, optimization_trials)
                results.update(reg_results)
        
        return results
    
    def _train_single_classification_model(self, trainer, X, y, model_type, use_grid_search, use_optuna, optimization_trials):
        """训练单个分类模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 分类模型...")
            
            # 如果启用Optuna且安装optuna，则进行贝叶斯优化
            if use_optuna:
                try:
                    import optuna
                    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    from sklearn.linear_model import LogisticRegression
                    # 根据模型类型选择分类器
                    if model_type == 'xgboost':
                        try:
                            from xgboost import XGBClassifier
                            classifier = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
                        except ImportError:
                            logger.warning("XGBoost 未安装, 回退使用 trainer 默认实现")
                            raise ImportError
                        def objective(trial):
                            trial_params = {
                                'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300),
                                'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
                                'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.3, log=True),
                                'classifier__subsample': trial.suggest_float('classifier__subsample', 0.6, 1.0),
                                'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0.6, 1.0)
                            }
                            pipe.set_params(**trial_params)
                            cv = KFold(n_splits=3, shuffle=True, random_state=42)
                            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                            return 1 - scores.mean()  # minimize
                    elif model_type == 'logistic':
                        classifier = LogisticRegression(max_iter=1000, solver='liblinear')
                        def objective(trial):
                            trial_params = {
                                'classifier__C': trial.suggest_float('classifier__C', 0.001, 10.0, log=True),
                                'classifier__penalty': trial.suggest_categorical('classifier__penalty', ['l1', 'l2'])
                            }
                            pipe.set_params(**trial_params)
                            cv = KFold(n_splits=3, shuffle=True, random_state=42)
                            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                            return 1 - scores.mean()
                    else:
                        raise ValueError("不支持的分类模型类型")

                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', classifier)
                    ])

                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=optimization_trials, show_progress_bar=False)

                    best_params = study.best_params
                    logger.info(f"Optuna完成: 最佳参数 {best_params}")

                    pipe.set_params(**best_params)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipe.fit(X_train, y_train)
                    y_train_pred = pipe.predict(X_train)
                    y_test_pred = pipe.predict(X_test)

                    result = {
                        'model': pipe,
                        'model_type': model_type,
                        'best_params': best_params,
                        'metrics': {
                            'train_accuracy': accuracy_score(y_train, y_train_pred),
                            'test_accuracy': accuracy_score(y_test, y_test_pred),
                            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
                            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
                            'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
                        }
                    }
                except ImportError:
                    # 回退至原 trainer 逻辑
                    result = trainer.train_single_model(X, y, model_type, 
                                              use_optimization=use_grid_search, 
                                              optimization_trials=optimization_trials)
            else:
                result = trainer.train_single_model(X, y, model_type, 
                                              use_optimization=use_grid_search, 
                                              optimization_trials=optimization_trials)
            
            # 保存模型（与回归模型保持一致，只保存模型对象本身）
            model_name = f"{model_type}_classification.pkl"
            model_path = os.path.join('models', model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            logger.info(f"{model_type} 分类模型已保存到: {model_path}")
            
            # 记录性能指标
            if 'metrics' in result:
                metrics = result['metrics']
                train_acc = metrics.get('train_accuracy', 0)
                test_acc = metrics.get('test_accuracy', 0)
                test_precision = metrics.get('test_precision', 0)
                test_recall = metrics.get('test_recall', 0)
                test_f1 = metrics.get('test_f1', 0)
                logger.info(f"{model_type} 分类模型性能: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, 精确率={test_precision:.4f}, 召回率={test_recall:.4f}, F1={test_f1:.4f}")

            # 记录最佳得分（如果有）
            if 'best_score' in result:
                logger.info(f"{model_type} 最佳得分: {result['best_score']:.4f}")
            
            return model_type, result
            
        except Exception as e:
            logger.error(f"训练 {model_type} 分类模型失败: {e}")
            return model_type, None

    def _train_classification_models(self, X: pd.DataFrame, y: pd.Series, 
                                   use_grid_search: bool = 1,
                                   use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """训练分类模型（多线程版本）"""
        # 初始化增强版训练器（用于训练单个分类模型）
        trainer = EnhancedMLTrainer(model_dir="models")
        results = {}
        
        # 分类模型类型（精简版：选择表现最好的两个）
        classification_models = ['xgboost', 'logistic']  # xgboost表现最佳，logistic作为简单模型备选
        
        logger.info(f"开始并行训练 {len(classification_models)} 个分类模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_classification_model, 
                               trainer, X, y, model_type, use_grid_search, use_optuna, optimization_trials): model_type
                for model_type in classification_models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    model_type, result = future.result()
                    if result is not None:
                        results[model_type] = result
                        logger.info(f"{model_type} 分类模型训练完成")
                except Exception as e:
                    logger.error(f"{model_type} 分类模型训练异常: {e}")
        
        return results
    
    def _train_single_regression_model(self, X, y, model_type, use_grid_search, use_optuna, optimization_trials):
        """训练单个回归模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 回归模型...")
            
            if model_type in ['ridge', 'lasso', 'elasticnet']:
                # 线性模型
                result = self._train_linear_model(X, y, model_type, use_grid_search)
            else:
                # 树模型
                result = self._train_tree_regression_model(X, y, model_type, use_grid_search, use_optuna, optimization_trials)
            
            # 保存模型
            model_name = f"{model_type}_regression.pkl"
            with open(os.path.join('models', model_name), 'wb') as f:
                pickle.dump(result['model'], f)
            logger.info(f"{model_type} 回归模型已保存到: models/{model_name}")
            
            # 记录性能指标
            metrics = result['metrics']
            logger.info(f"{model_type} 性能: R²={metrics.get('test_r2', 0):.4f}, MSE={metrics.get('test_mse', 0):.6f}")
            
            return model_type, result
            
        except Exception as e:
            logger.error(f"训练 {model_type} 回归模型失败: {e}")
            return model_type, None

    def _train_regression_models(self, X: pd.DataFrame, y: pd.Series, 
                               use_grid_search: bool = 1,
                               use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """训练回归模型（多线程版本）"""
        results = {}
        
        # 回归模型类型（精简版：选择表现最好的两个）
        regression_models = ['xgboost', 'lightgbm', 'catboost', 'lasso']  # 新增LightGBM和CatBoost
        
        logger.info(f"开始并行训练 {len(regression_models)} 个回归模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_regression_model, 
                               X, y, model_type, use_grid_search, use_optuna, optimization_trials): model_type
                for model_type in regression_models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    model_type, result = future.result()
                    if result is not None:
                        results[model_type] = result
                        logger.info(f"{model_type} 回归模型训练完成")
                except Exception as e:
                    logger.error(f"{model_type} 回归模型训练异常: {e}")
        
        return results
    
    def _train_linear_model(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                          use_grid_search: bool = 1) -> Dict[str, Any]:
        """训练线性回归模型"""
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"训练{model_type}回归模型: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        # 根据模型类型选择回归器
        if model_type == 'ridge':
            regressor = Ridge(random_state=42)
            param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'lasso':
            regressor = Lasso(random_state=42, max_iter=10000)
            param_grid = {'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        elif model_type == 'elasticnet':
            regressor = ElasticNet(random_state=42, max_iter=10000)
            param_grid = {
                'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            raise ValueError(f"不支持的线性模型类型: {model_type}")
        
        # 线性模型不支持Optuna优化，仅使用网格搜索
        # 移除树模型的Optuna优化逻辑，这些逻辑应该在 _train_tree_regression_model 中处理

        # 创建管线
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        if use_grid_search:
            # 网格搜索优化
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"网格搜索完成: 最佳参数 {best_params}, 最佳得分 {-best_score:.4f}")
            
            # 使用最佳参数重新训练
            pipeline.set_params(**best_params)
        
        pipeline.fit(X_train, y_train)
        
        # 预测
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # 计算评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 获取特征重要性（系数绝对值）
        feature_importance = {}
        if hasattr(pipeline.named_steps['regressor'], 'coef_'):
            coef = pipeline.named_steps['regressor'].coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # 处理多输出情况
            
            feature_names = X.columns.tolist()
            for i, importance in enumerate(np.abs(coef)):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        
        # 构建结果
        result = {
            'model': pipeline,
            'model_type': model_type,
            'best_params': best_params if use_grid_search else {},
            'cv_score': best_score if use_grid_search else None,
            'feature_names': X.columns.tolist(),
            'feature_importance': feature_importance,
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'predictions': {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        }
        
        logger.info(f"{model_type}回归模型训练完成: 训练集R²={train_r2:.4f}, 测试集R²={test_r2:.4f}")
        logger.info(f"训练集MSE={train_mse:.6f}, 测试集MSE={test_mse:.6f}")
        
        return result
    
    def _train_tree_regression_model(self, X: pd.DataFrame, y: pd.Series, model_type: str,
                                   use_grid_search: bool = 1, use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """训练树回归模型"""
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"训练{model_type}回归模型: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        # 根据模型类型选择回归器
        best_params = {}
        best_score = None
        
        if model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                # 设置 verbosity=-1 以关闭 LightGBM 冗余警告（如 "No further splits with positive gain"）
                regressor = LGBMRegressor(objective='huber', random_state=42, n_jobs=-1, verbosity=-1)
                param_grid = {
                    'regressor__n_estimators': [100, 300],
                    'regressor__learning_rate': [0.03, 0.1],
                    'regressor__max_depth': [-1, 6, 10],
                    'regressor__subsample': [0.8, 1.0],
                    'regressor__colsample_bytree': [0.8, 1.0],
                    # 始终保持静默
                    'regressor__verbosity': [-1]
                }
            except ImportError:
                logger.error("LightGBM未安装，跳过LightGBM训练")
                return self._empty_regression_result(model_type, X)
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                regressor = CatBoostRegressor(
                    loss_function='Huber:delta=1',
                    depth=6,
                    verbose=False,
                    random_state=42
                )
                param_grid = {
                    'regressor__iterations': [300, 600],
                    'regressor__learning_rate': [0.03, 0.1],
                    'regressor__depth': [4, 6, 8]
                }
            except ImportError:
                logger.error("CatBoost未安装，跳过CatBoost训练")
                return self._empty_regression_result(model_type, X)
        elif model_type == 'randomforest':
            regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            }
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                regressor = XGBRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'regressor__n_estimators': [50, 100],
                    'regressor__max_depth': [3, 6, 9],
                    'regressor__learning_rate': [0.01, 0.1],
                    'regressor__subsample': [0.8, 1.0],
                    'regressor__colsample_bytree': [0.8, 1.0]
                }
            except ImportError:
                logger.error("XGBoost未安装，跳过XGBoost训练")
                return {
                    'model': None,
                    'model_type': model_type,
                    'best_params': {},
                    'cv_score': None,
                    'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
                    'feature_importance': {},
                    'metrics': {
                        'train_mse': 0,
                        'test_mse': 0,
                        'train_mae': 0,
                        'test_mae': 0,
                        'train_r2': 0,
                        'test_r2': 0
                    }
                }
        else:
            raise ValueError(f"不支持的树回归模型类型: {model_type}")
        
        # 创建管线
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        if use_optuna:
            try:
                import optuna
                from sklearn.model_selection import KFold, cross_val_score

                def objective(trial):
                    trial_params = {}
                    # 根据模型类型定义搜索空间
                    if model_type == 'lightgbm':
                        trial_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 400),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.01, 0.3, log=True),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', -1, 10),
                            'regressor__subsample': trial.suggest_float('regressor__subsample', 0.6, 1.0),
                            'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.6, 1.0)
                        }
                    elif model_type == 'catboost':
                        trial_params = {
                            'regressor__iterations': trial.suggest_int('regressor__iterations', 200, 800),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.01, 0.3, log=True),
                            'regressor__depth': trial.suggest_int('regressor__depth', 4, 10)
                        }
                    elif model_type == 'randomforest':
                        trial_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 300),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', 5, 30),
                            'regressor__min_samples_split': trial.suggest_int('regressor__min_samples_split', 2, 10),
                            'regressor__min_samples_leaf': trial.suggest_int('regressor__min_samples_leaf', 1, 5)
                        }
                    elif model_type == 'xgboost':
                        trial_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 400),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', 3, 10),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.01, 0.3, log=True),
                            'regressor__subsample': trial.suggest_float('regressor__subsample', 0.6, 1.0),
                            'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.6, 1.0)
                        }
                    else:
                        raise ValueError("不支持的模型类型")

                    pipeline.set_params(**trial_params)
                    cv = KFold(n_splits=3, shuffle=True, random_state=42)
                    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
                    return -scores.mean()

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=optimization_trials, show_progress_bar=False)
                best_params = study.best_params
                best_score = -study.best_value
                logger.info(f"Optuna优化完成: 最佳参数 {best_params}, 最佳得分 {best_score:.4f}")

                # 使用最佳参数训练
                pipeline.set_params(**best_params)
            except ImportError:
                logger.warning("Optuna 未安装, 回退使用 GridSearchCV")
                use_grid_search = True  # fallback
        
        if use_grid_search and not use_optuna:
            # 网格搜索优化
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_

            logger.info(f"网格搜索完成: 最佳参数 {best_params}, 最佳得分 {best_score:.4f}")

            # 使用最佳参数重新训练
            pipeline.set_params(**best_params)
        
        pipeline.fit(X_train, y_train)
        
        # 预测
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # 计算评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 获取特征重要性
        feature_importance = {}
        if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
            importances = pipeline.named_steps['regressor'].feature_importances_
            feature_names = X.columns.tolist()
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        
        # 构建结果
        result = {
            'model': pipeline,
            'model_type': model_type,
            'best_params': best_params if use_grid_search else {},
            'cv_score': best_score if use_grid_search else None,
            'feature_names': X.columns.tolist(),
            'feature_importance': feature_importance,
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'predictions': {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        }
        
        logger.info(f"{model_type}回归模型训练完成: 训练集R²={train_r2:.4f}, 测试集R²={test_r2:.4f}")
        logger.info(f"训练集MSE={train_mse:.6f}, 测试集MSE={test_mse:.6f}")
        
        return result

    def _empty_regression_result(self, model_type: str, X: pd.DataFrame):
        """当缺少依赖时返回占位结果"""
        return {
            'model': None,
            'model_type': model_type,
            'best_params': {},
            'cv_score': None,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
            'feature_importance': {},
            'metrics': {
                'train_mse': 0,
                'test_mse': 0,
                'train_mae': 0,
                'test_mae': 0,
                'train_r2': 0,
                'test_r2': 0
            }
        }

    def evaluate_models(self, results: Dict[str, Any]):
        """评估模型性能"""
        logger.info("\n=== 模型评估结果 ===")
        
        # 分类模型评估
        cls_models = [k for k in results.keys() if k in ['logistic', 'randomforest', 'xgboost']]
        if cls_models:
            logger.info("分类模型:")
            best_cls_model = None
            best_cls_score = -float('inf')
            
            for model_type in cls_models:
                result = results[model_type]
                if 'best_score' in result:
                    score = result['best_score']
                    logger.info(f"  {model_type:12s}: 最佳得分 = {score:.4f}")
                    if score > best_cls_score:
                        best_cls_score = score
                        best_cls_model = model_type
            
            if best_cls_model:
                logger.info(f"  最佳分类模型: {best_cls_model} (得分={best_cls_score:.4f})")
        
        # 回归模型评估
        reg_models = [k for k in results.keys() if k in ['ridge', 'lasso', 'elasticnet', 'randomforest', 'xgboost', 'lightgbm', 'catboost']]
        if reg_models:
            logger.info("回归模型:")
            best_reg_model = None
            best_reg_r2 = -float('inf')
            
            for model_type in reg_models:
                result = results[model_type]
                metrics = result.get('metrics', {})
                test_r2 = metrics.get('test_r2', 0)
                test_mse = metrics.get('test_mse', 0)
                
                logger.info(f"  {model_type:12s}: R²={test_r2:8.4f}, MSE={test_mse:10.6f}")
                
                if test_r2 > best_reg_r2:
                    best_reg_r2 = test_r2
                    best_reg_model = model_type
            
            if best_reg_model:
                logger.info(f"  最佳回归模型: {best_reg_model} (R²={best_reg_r2:.4f})")

import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一模型训练脚本（支持小规模快速验证）")
    parser.add_argument("--lookback", type=int, default=365, help="回溯天数，默认为365")
    parser.add_argument("--n_stocks", type=int, default=50, help="使用的股票数量，默认为50")
    parser.add_argument("--prediction_period", type=int, default=30, help="预测周期（天数），默认为30")
    parser.add_argument("--mode", choices=['classification', 'regression', 'both'], default='both', help="训练模式")
    parser.add_argument("--no_grid_search", action="store_true", help="禁用网格搜索以加快速度")
    parser.add_argument("--use_optuna", action="store_true", help="使用Optuna进行超参数优化（优先级高于网格搜索）")
    parser.add_argument("--optimization_trials", type=int, default=50, help="Optuna/Bayesian优化时的迭代次数 (默认50)")
    # 新增: 关闭特征选择开关
    parser.add_argument("--disable_feature_selection", action="store_true", help="关闭特征选择优化，加快训练速度")
    args = parser.parse_args()

    logger.info("=== 开始统一模型训练流程 ===")
    logger.info(f"参数: lookback={args.lookback}, n_stocks={args.n_stocks}, prediction_period={args.prediction_period}, mode={args.mode}, grid_search={'off' if args.no_grid_search else 'on'}")

    try:
        trainer = UnifiedModelTrainer(enable_feature_selection=(not args.disable_feature_selection))
        if args.disable_feature_selection:
            logger.info("已通过命令行开关关闭特征选择优化")
            
        # 准备训练数据（同时生成分类和回归标签）
        X, y = trainer.prepare_training_data(mode=args.mode, lookback_days=args.lookback, n_stocks=args.n_stocks, prediction_period=args.prediction_period)
        
        if X is None or not y:
            logger.error("数据准备失败")
            return
        
        logger.info(f"最终训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 训练所有模型
        results = trainer.train_models(
            X,
            y,
            mode=args.mode,
            use_grid_search=(0 if args.no_grid_search else 1) if not args.use_optuna else 0,
            use_optuna=1 if args.use_optuna else 0,
            optimization_trials=args.optimization_trials
        )
        
        # 评估模型
        trainer.evaluate_models(results)
        
        logger.info("统一模型训练完成!")
        
    except Exception as e:
        logger.error(f"训练流程失败: {e}", exc_info=1)

if __name__ == "__main__":
    main()