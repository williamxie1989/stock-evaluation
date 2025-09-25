#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化回归任务训练脚本
专门针对股票收益率预测进行优化
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import pickle

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
        logging.FileHandler('regression_training.log')
    ]
)
logger = logging.getLogger(__name__)

class RegressionTrainer:
    """优化回归任务训练器"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()
        
    def prepare_regression_data(self, lookback_days: int = 365, n_stocks: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备回归任务专用数据
        
        Args:
            lookback_days: 回溯天数
            n_stocks: 使用的股票数量
            
        Returns:
            X: 特征数据
            y: 收益率标签
        """
        logger.info("开始准备回归训练数据...")
        
        # 获取股票列表
        stock_records = self.db_manager.list_symbols()
        symbols = [record['symbol'] for record in stock_records]
        logger.info(f"获取到 {len(symbols)} 只股票")
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        logger.info(f"数据范围: {start_date} 到 {end_date}")
        
        all_features = []
        all_labels = []
        
        # 处理每只股票
        for i, symbol in enumerate(symbols[:n_stocks]):
            if i % 10 == 0:
                logger.info(f"处理第 {i+1}/{min(n_stocks, len(symbols))} 只股票: {symbol}")
            
            try:
                # 获取股票数据 - 使用DatabaseManager的正确方法
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
                
                # 生成回归标签（未来5天收益率）
                close_prices = stock_data['close'].values
                labels = []
                
                for j in range(len(close_prices) - 5):
                    current_price = close_prices[j]
                    future_price = close_prices[j + 5]  # 5天后价格
                    return_rate = (future_price - current_price) / current_price
                    labels.append(return_rate)
                
                # 对齐特征和标签
                aligned_features = features_df.iloc[:-5].copy()
                aligned_labels = labels
                
                if len(aligned_features) != len(aligned_labels):
                    min_len = min(len(aligned_features), len(aligned_labels))
                    aligned_features = aligned_features.iloc[:min_len]
                    aligned_labels = aligned_labels[:min_len]
                
                # 添加股票标识
                aligned_features['symbol'] = symbol
                
                all_features.append(aligned_features)
                all_labels.extend(aligned_labels)
                
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
        
        if not all_features:
            raise ValueError("没有生成任何训练数据")
        
        # 合并所有数据
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = pd.Series(all_labels, name='return_5d')
        
        logger.info(f"总共生成 {len(X_combined)} 个样本")
        
        # 数据预处理
        X_processed, y_processed = self._preprocess_data(X_combined, y_combined)
        
        return X_processed, y_processed
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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
        y_final = y[valid_indices]
        
        logger.info(f"清洗后样本数量: {len(X_final)}")
        
        # 移除极端异常值（收益率在±50%之外）
        reasonable_returns = (y_final.abs() <= 0.5)
        X_final = X_final[reasonable_returns]
        y_final = y_final[reasonable_returns]
        
        logger.info(f"移除极端值后样本数量: {len(X_final)}")
        logger.info(f"收益率范围: {y_final.min():.3f} 到 {y_final.max():.3f}")
        logger.info(f"收益率均值: {y_final.mean():.6f}, 标准差: {y_final.std():.6f}")
        
        # 标准化特征名称
        X_final.columns = [f'feature_{i}' for i in range(X_final.shape[1])]
        
        return X_final, y_final
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练回归模型"""
        logger.info("开始训练回归模型...")
        
        trainer = EnhancedMLTrainer()
        results = {}
        
        # 定义回归模型类型
        regression_models = [
            'ridge',
            'lasso',
            'elasticnet',
            'randomforest',
            'xgboost'
        ]
        
        for model_type in regression_models:
            logger.info(f"训练 {model_type} 回归模型...")
            
            try:
                if model_type in ['ridge', 'lasso', 'elasticnet']:
                    # 线性模型需要单独处理
                    result = self._train_linear_model(X, y, model_type)
                else:
                    # 树模型使用回归版本
                    result = self._train_tree_regression_model(X, y, model_type)
                
                results[model_type] = result
                
                # 保存模型
                model_name = f"{model_type}_regression.pkl"
                model_path = trainer.save_model(result['model'], model_name)
                logger.info(f"{model_type} 模型已保存到: {model_path}")
                
                # 记录性能指标
                metrics = result['metrics']
                logger.info(f"{model_type} 性能: R²={metrics.get('test_r2', 0):.4f}, MSE={metrics.get('test_mse', 0):.6f}")
                
            except Exception as e:
                logger.error(f"训练 {model_type} 模型失败: {e}")
        
        return results
    
    def _train_linear_model(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
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
            'best_params': best_params,
            'cv_score': best_score,
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
    
    def _train_tree_regression_model(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
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
            'best_params': best_params,
            'cv_score': best_score,
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
    
    def evaluate_models(self, results: Dict[str, Any], X_test: pd.DataFrame = None, y_test: pd.Series = None):
        """评估模型性能"""
        logger.info("\n=== 模型评估结果 ===")
        
        best_model = None
        best_r2 = -float('inf')
        
        for model_type, result in results.items():
            metrics = result['metrics']
            test_r2 = metrics.get('test_r2', 0)
            test_mse = metrics.get('test_mse', 0)
            
            logger.info(f"{model_type:12s}: R²={test_r2:8.4f}, MSE={test_mse:10.6f}")
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model_type
        
        if best_model:
            logger.info(f"\n最佳模型: {best_model} (R²={best_r2:.4f})")
        
        return best_model

def main():
    """主函数"""
    logger.info("=== 开始优化回归训练流程 ===")
    
    try:
        trainer = RegressionTrainer()
        
        # 准备数据
        X, y = trainer.prepare_regression_data(lookback_days=365, n_stocks=50)
        
        if X is None or y is None or len(X) == 0:
            logger.error("数据准备失败")
            return
        
        logger.info(f"最终训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 训练模型
        results = trainer.train_regression_models(X, y)
        
        # 评估模型
        best_model = trainer.evaluate_models(results)
        
        logger.info("回归训练完成!")
        
    except Exception as e:
        logger.error(f"训练流程失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()