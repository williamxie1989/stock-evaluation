#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型训练脚本
整合分类和回归模型训练功能
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import DatabaseManager
from enhanced_ml_trainer import EnhancedMLTrainer
from enhanced_features import EnhancedFeatureGenerator

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

class UnifiedModelTrainer:
    """统一模型训练器，支持分类和回归任务"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()
    
    def prepare_training_data(self, mode: str = 'both', lookback_days: int = 365, 
                            n_stocks: int = 1000, prediction_period: int = 30) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        准备训练数据
        
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
        
        # 获取股票列表
        stock_records = self.db_manager.list_symbols()
        symbols = [record['symbol'] for record in stock_records]
        logger.info(f"获取到 {len(symbols)} 只股票，使用前 {n_stocks} 只")
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        logger.info(f"数据范围: {start_date} 到 {end_date}")
        
        all_features = []
        all_cls_labels = []
        all_reg_labels = []
        
        # 处理每只股票
        for i, symbol in enumerate(symbols[:n_stocks]):
            if i % 10 == 0:
                logger.info(f"处理第 {i+1}/{min(n_stocks, len(symbols))} 只股票: {symbol}")
            
            try:
                # 获取股票数据
                with self.db_manager.get_conn() as conn:
                    query = """
                        SELECT symbol, date, open, high, low, close, volume
                        FROM prices_daily 
                        WHERE symbol = ? AND date >= ? AND date <= ?
                        ORDER BY date
                    """
                    stock_data = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
                    
                    if stock_data.empty or len(stock_data) < 60:  # 至少需要60天数据
                        continue
                    
                    # 转换数据类型
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                    
                    # 删除包含NaN的行
                    stock_data = stock_data.dropna()
                
                # 生成增强特征
                features_df = self.feature_generator.generate_features(stock_data)
                if features_df.empty:
                    continue
                
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
                
                # 添加股票标识
                aligned_features['symbol'] = symbol
                
                all_features.append(aligned_features)
                all_reg_labels.extend(reg_labels)
                all_cls_labels.extend(cls_labels)
                
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
        
        if not all_features:
            raise ValueError("没有生成任何训练数据")
        
        # 合并所有数据
        X_combined = pd.concat(all_features, ignore_index=True)
        y_reg = pd.Series(all_reg_labels, name='return_rate')
        y_cls = pd.Series(all_cls_labels, name='label_cls')
        
        logger.info(f"总共生成 {len(X_combined)} 个样本")
        
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
        """数据预处理"""
        logger.info("开始数据预处理...")
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        
        # 处理无穷大和NaN
        X_cleaned = X_numeric.replace([np.inf, -np.inf], np.nan)
        
        # 移除包含NaN的行
        valid_indices = X_cleaned.notna().all(axis=1)
        X_final = X_cleaned[valid_indices]
        y_reg_final = y_reg[valid_indices]
        y_cls_final = y_cls[valid_indices]
        
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
        
        return X_final, y_reg_final, y_cls_final
    
    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series], 
                    mode: str = 'both', use_grid_search: bool = True) -> Dict[str, Any]:
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
                                                  X, y['cls'], use_grid_search)] = 'classification'
                
                if 'reg' in y:
                    future_to_task[executor.submit(self._train_regression_models, 
                                                  X, y['reg'], use_grid_search)] = 'regression'
                
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
                cls_results = self._train_classification_models(X, y['cls'], use_grid_search)
                results.update(cls_results)
            
            if mode == 'regression' and 'reg' in y:
                logger.info("训练回归模型...")
                reg_results = self._train_regression_models(X, y['reg'], use_grid_search)
                results.update(reg_results)
        
        return results
    
    def _train_single_classification_model(self, trainer, X, y, model_type, use_grid_search):
        """训练单个分类模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 分类模型...")
            
            result = trainer.train_single_model(X, y, model_type, 
                                              use_optimization=use_grid_search, 
                                              optimization_trials=5)  # 减少试验次数以加快训练
            
            # 保存模型（与回归模型保持一致，只保存模型对象本身）
            model_name = f"{model_type}_classification.pkl"
            model_path = os.path.join('models', model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            logger.info(f"{model_type} 分类模型已保存到: {model_path}")
            
            # 记录性能指标
            if 'best_score' in result:
                logger.info(f"{model_type} 最佳得分: {result['best_score']:.4f}")
            
            return model_type, result
            
        except Exception as e:
            logger.error(f"训练 {model_type} 分类模型失败: {e}")
            return model_type, None

    def _train_classification_models(self, X: pd.DataFrame, y: pd.Series, 
                                   use_grid_search: bool = True) -> Dict[str, Any]:
        """训练分类模型（多线程版本）"""
        trainer = EnhancedMLTrainer(model_dir="models")  # 修改为保存到根目录
        results = {}
        
        # 分类模型类型（精简版：选择表现最好的两个）
        classification_models = ['xgboost', 'logistic']  # xgboost表现最佳，logistic作为简单模型备选
        
        logger.info(f"开始并行训练 {len(classification_models)} 个分类模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_classification_model, 
                               trainer, X, y, model_type, use_grid_search): model_type
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
    
    def _train_single_regression_model(self, X, y, model_type, use_grid_search):
        """训练单个回归模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 回归模型...")
            
            if model_type in ['ridge', 'lasso', 'elasticnet']:
                # 线性模型
                result = self._train_linear_model(X, y, model_type, use_grid_search)
            else:
                # 树模型
                result = self._train_tree_regression_model(X, y, model_type, use_grid_search)
            
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
                               use_grid_search: bool = True) -> Dict[str, Any]:
        """训练回归模型（多线程版本）"""
        results = {}
        
        # 回归模型类型（精简版：选择表现最好的两个）
        regression_models = ['xgboost', 'lasso']  # xgboost表现最佳，lasso作为线性模型备选
        
        logger.info(f"开始并行训练 {len(regression_models)} 个回归模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_regression_model, 
                               X, y, model_type, use_grid_search): model_type
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
                          use_grid_search: bool = True) -> Dict[str, Any]:
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
                                   use_grid_search: bool = True) -> Dict[str, Any]:
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
        if model_type == 'randomforest':
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
                    'feature_names': X.columns.tolist(),
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
        
        # 创建管线
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        if use_grid_search:
            # 网格搜索优化
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
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
        reg_models = [k for k in results.keys() if k in ['ridge', 'lasso', 'elasticnet', 'randomforest', 'xgboost']]
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

def main():
    """主函数"""
    logger.info("=== 开始统一模型训练流程 ===")
    
    try:
        trainer = UnifiedModelTrainer()
        
        # 准备训练数据（同时生成分类和回归标签）
        X, y = trainer.prepare_training_data(mode='both', lookback_days=365, n_stocks=1000, prediction_period=30)
        
        if X is None or not y:
            logger.error("数据准备失败")
            return
        
        logger.info(f"最终训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 训练所有模型
        results = trainer.train_models(X, y, mode='both', use_grid_search=True)
        
        # 评估模型
        trainer.evaluate_models(results)
        
        logger.info("统一模型训练完成!")
        
    except Exception as e:
        logger.error(f"训练流程失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()