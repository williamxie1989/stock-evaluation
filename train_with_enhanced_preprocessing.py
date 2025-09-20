#!/usr/bin/env python3
"""
使用增强预处理pipeline的模型训练脚本
集成特征选择、异常值处理、特征工程等功能
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
import joblib
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# 导入自定义模块
from db import DatabaseManager
from enhanced_features import EnhancedFeatureGenerator
from enhanced_preprocessing import EnhancedPreprocessingPipeline, create_enhanced_preprocessing_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """增强的模型训练器"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()
        
        # 预处理配置
        self.cls_preprocessing_config = create_enhanced_preprocessing_config('classification', 'medium')
        self.reg_preprocessing_config = create_enhanced_preprocessing_config('regression', 'medium')
        
        # 预处理器
        self.cls_preprocessor = None
        self.reg_preprocessor = None
        
        # 模型配置 - 优化后的参数减少训练时间
        self.model_configs = {
            'classification': {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    }
                }
            },
            'regression': {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'ridge': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                }
            }
        }
    
    def prepare_training_data(self, period: str = '2y', min_samples: int = 1000):
        """准备训练数据"""
        logger.info(f"准备训练数据，时间范围: {period}")
        
        # 获取股票列表
        symbols = [s['symbol'] for s in self.db_manager.list_symbols(markets=['SH', 'SZ'])]
        logger.info(f"获取到 {len(symbols)} 只股票")
        
        # 获取价格数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        if period == '1y':
            start_date = (datetime.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        elif period == '2y':
            start_date = (datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        elif period == '3y':
            start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        
        logger.info(f"数据时间范围: {start_date} 到 {end_date}")
        
        # 批量获取价格数据
        all_data = []
        batch_size = 50
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            try:
                prices = self.db_manager.get_last_n_bars(batch_symbols, n=1000)
                if prices is not None and not prices.empty:
                    all_data.append(prices)
                    logger.info(f"已处理 {i+len(batch_symbols)}/{len(symbols)} 只股票")
            except Exception as e:
                logger.error(f"获取价格数据失败 (batch {i//batch_size + 1}): {e}")
                continue
        
        if not all_data:
            raise ValueError("无法获取价格数据")
        
        # 合并所有数据
        all_prices = pd.concat(all_data, ignore_index=True)
        logger.info(f"获取到 {len(all_prices)} 条价格记录")
        
        # 生成特征和标签
        features_list = []
        labels_cls_list = []
        labels_reg_list = []
        
        for symbol in symbols[:200]:  # 限制股票数量以加快训练
            try:
                symbol_data = all_prices[all_prices['symbol'] == symbol].copy()
                if len(symbol_data) < 60:  # 至少需要60天数据
                    continue
                
                symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
                
                # 生成特征
                features = self.feature_generator.generate_features(symbol_data)
                if features.empty:
                    continue
                
                # 生成标签
                cls_labels, reg_labels = self._generate_labels(symbol_data)
                
                # 确保特征和标签长度一致
                min_len = min(len(features), len(cls_labels), len(reg_labels))
                if min_len > 0:
                    features_list.append(features.iloc[:min_len])
                    labels_cls_list.extend(cls_labels[:min_len])
                    labels_reg_list.extend(reg_labels[:min_len])
                
            except Exception as e:
                logger.error(f"处理股票 {symbol} 时出错: {e}")
                continue
        
        if not features_list:
            raise ValueError("无法生成特征数据")
        
        # 合并所有特征
        X = pd.concat(features_list, ignore_index=True)
        y_cls = np.array(labels_cls_list)
        y_reg = np.array(labels_reg_list)
        
        logger.info(f"训练数据准备完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        logger.info(f"分类标签分布: {np.bincount(y_cls)}")
        logger.info(f"回归标签统计: 均值={np.mean(y_reg):.4f}, 标准差={np.std(y_reg):.4f}")
        
        return X, y_cls, y_reg
    
    def _generate_labels(self, price_data: pd.DataFrame, forward_days: int = 30):
        """生成分类和回归标签"""
        price_data = price_data.sort_values('date').reset_index(drop=True)
        
        cls_labels = []
        reg_labels = []
        
        for i in range(len(price_data) - forward_days):
            current_price = price_data.iloc[i]['close']
            future_price = price_data.iloc[i + forward_days]['close']
            
            # 计算收益率
            return_rate = (future_price - current_price) / current_price
            
            # 分类标签：收益率 > 5% 为正类
            cls_label = 1 if return_rate > 0.05 else 0
            cls_labels.append(cls_label)
            
            # 回归标签：直接使用收益率
            reg_labels.append(return_rate)
        
        return cls_labels, reg_labels
    
    def train_classification_model(self, X, y, model_type='random_forest', use_grid_search=True):
        """训练分类模型"""
        logger.info(f"开始训练分类模型: {model_type}")
        
        # 初始化预处理器
        self.cls_preprocessor = EnhancedPreprocessingPipeline('classification', self.cls_preprocessing_config)
        
        # 预处理数据
        X_processed = self.cls_preprocessor.fit_transform(X, y)
        logger.info(f"预处理后特征数: {X_processed.shape[1]}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 获取模型配置
        model_config = self.model_configs['classification'][model_type]
        base_model = model_config['model']
        
        # 网格搜索或直接训练
        if use_grid_search and len(X_train) > 1000:
            logger.info("使用网格搜索优化超参数")
            grid_search = GridSearchCV(
                base_model, 
                model_config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"最佳参数: {grid_search.best_params_}")
        else:
            logger.info("使用默认参数训练")
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        # 概率校准
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"分类模型性能: {metrics}")
        
        # 特征重要性报告
        feature_report = self.cls_preprocessor.get_feature_importance_report()
        logger.info(f"特征重要性报告:\n{feature_report}")
        
        return calibrated_model, metrics, self.cls_preprocessor
    
    def train_regression_model(self, X, y, model_type='random_forest', use_grid_search=True):
        """训练回归模型"""
        logger.info(f"开始训练回归模型: {model_type}")
        
        # 初始化预处理器
        self.reg_preprocessor = EnhancedPreprocessingPipeline('regression', self.reg_preprocessing_config)
        
        # 预处理数据
        X_processed = self.reg_preprocessor.fit_transform(X, y)
        logger.info(f"预处理后特征数: {X_processed.shape[1]}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # 获取模型配置
        model_config = self.model_configs['regression'][model_type]
        base_model = model_config['model']
        
        # 网格搜索或直接训练
        if use_grid_search and len(X_train) > 1000:
            logger.info("使用网格搜索优化超参数")
            grid_search = GridSearchCV(
                base_model, 
                model_config['params'], 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"最佳参数: {grid_search.best_params_}")
        else:
            logger.info("使用默认参数训练")
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"回归模型性能: {metrics}")
        
        # 特征重要性报告
        feature_report = self.reg_preprocessor.get_feature_importance_report()
        logger.info(f"特征重要性报告:\n{feature_report}")
        
        return best_model, metrics, self.reg_preprocessor
    
    def save_models(self, cls_model, reg_model, cls_preprocessor, reg_preprocessor, 
                   cls_metrics, reg_metrics, period='30d'):
        """保存模型和预处理器"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建模型目录
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # 保存分类模型
        cls_model_data = {
            'model': cls_model,
            'preprocessor': cls_preprocessor,
            'metrics': cls_metrics,
            'feature_names': cls_preprocessor.feature_names_,
            'period': period,
            'timestamp': timestamp,
            'model_type': 'classification'
        }
        
        cls_filename = f'enhanced_classification_model_{period}_{timestamp}.pkl'
        cls_path = os.path.join(models_dir, cls_filename)
        
        with open(cls_path, 'wb') as f:
            pickle.dump(cls_model_data, f)
        
        logger.info(f"分类模型已保存: {cls_path}")
        
        # 保存回归模型
        reg_model_data = {
            'model': reg_model,
            'preprocessor': reg_preprocessor,
            'metrics': reg_metrics,
            'feature_names': reg_preprocessor.feature_names_,
            'period': period,
            'timestamp': timestamp,
            'model_type': 'regression'
        }
        
        reg_filename = f'enhanced_regression_model_{period}_{timestamp}.pkl'
        reg_path = os.path.join(models_dir, reg_filename)
        
        with open(reg_path, 'wb') as f:
            pickle.dump(reg_model_data, f)
        
        logger.info(f"回归模型已保存: {reg_path}")
        
        return cls_path, reg_path
    
    def train_and_save_models(self, period='2y', cls_model_type='random_forest', 
                             reg_model_type='random_forest', use_grid_search=True):
        """完整的训练和保存流程"""
        logger.info("开始增强模型训练流程")
        
        try:
            # 准备数据
            X, y_cls, y_reg = self.prepare_training_data(period)
            
            # 训练分类模型
            cls_model, cls_metrics, cls_preprocessor = self.train_classification_model(
                X, y_cls, cls_model_type, use_grid_search
            )
            
            # 训练回归模型
            reg_model, reg_metrics, reg_preprocessor = self.train_regression_model(
                X, y_reg, reg_model_type, use_grid_search
            )
            
            # 保存模型
            cls_path, reg_path = self.save_models(
                cls_model, reg_model, cls_preprocessor, reg_preprocessor,
                cls_metrics, reg_metrics, period
            )
            
            logger.info("增强模型训练完成")
            
            return {
                'classification': {
                    'model_path': cls_path,
                    'metrics': cls_metrics
                },
                'regression': {
                    'model_path': reg_path,
                    'metrics': reg_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise

def main():
    """主函数"""
    trainer = EnhancedModelTrainer()
    
    # 训练配置
    configs = [
        {
            'period': '2y',
            'cls_model_type': 'random_forest',
            'reg_model_type': 'random_forest',
            'use_grid_search': True
        },
        {
            'period': '2y',
            'cls_model_type': 'logistic_regression',
            'reg_model_type': 'ridge',
            'use_grid_search': True
        }
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"训练配置: {config}")
        try:
            result = trainer.train_and_save_models(**config)
            results.append(result)
            logger.info(f"配置 {config} 训练完成")
        except Exception as e:
            logger.error(f"配置 {config} 训练失败: {e}")
            continue
    
    # 输出结果摘要
    logger.info("=" * 60)
    logger.info("训练结果摘要:")
    for i, result in enumerate(results):
        logger.info(f"配置 {i+1}:")
        logger.info(f"  分类模型: {result['classification']['metrics']}")
        logger.info(f"  回归模型: {result['regression']['metrics']}")
    
    return results

if __name__ == "__main__":
    results = main()