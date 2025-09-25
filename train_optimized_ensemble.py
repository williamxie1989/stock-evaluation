#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化集成模型训练脚本
使用EnhancedMLTrainer训练多种先进模型并进行集成
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple
import joblib

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import DatabaseManager
from enhanced_ml_trainer import EnhancedMLTrainer
from enhanced_features import EnhancedFeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedEnsembleTrainer:
    """优化的集成模型训练器"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()
        
    def prepare_training_data(self, 
                            start_date: str = "2020-01-01", 
                            end_date: str = None,
                            min_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"准备训练数据: {start_date} 到 {end_date}")
        
        # 获取所有股票代码
        symbols = [s['symbol'] for s in self.db.list_symbols(markets=['SH', 'SZ'])]
        logger.info(f"获取到 {len(symbols)} 只股票")
        
        all_features = []
        all_labels = []
        
        # 分批处理股票，避免内存问题
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {len(batch_symbols)} 只股票")
            
            for symbol in batch_symbols:
                try:
                    # 获取指定日期范围内的价格数据
                    bars = self._get_stock_data_in_range(symbol, start_date, end_date)
                    if bars.empty:
                        continue
                    
                    # 生成特征
                    features_df = self.feature_generator.generate_features(bars)
                    
                    if features_df.empty:
                        continue
                    
                    # 添加symbol和date列
                    features_df['symbol'] = symbol
                    if 'date' not in features_df.columns:
                        features_df['date'] = bars['date'].iloc[-len(features_df):].values
                    
                    all_features.append(features_df)
                        
                except Exception as e:
                    logger.warning(f"处理股票 {symbol} 失败: {e}")
                    continue
        
        if not all_features:
            raise ValueError("没有生成任何训练数据")
            
        # 合并所有特征数据
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"总共生成 {len(combined_features)} 条特征样本")
        
        # 生成标签数据
        logger.info("生成标签数据...")
        labels_df = self._generate_labels(symbols, start_date, end_date)
        
        # 合并特征和标签
        combined_df = combined_features.merge(labels_df, on=['symbol', 'date'], how='inner')
        logger.info(f"合并后样本数量: {len(combined_df)}")
        
        if len(combined_df) < min_samples:
            logger.warning(f"训练样本数量 {len(combined_df)} 少于最小要求 {min_samples}")
        
        # 分离特征和标签
        feature_cols = [col for col in combined_df.columns 
                       if col not in ['symbol', 'date', 'label_cls', 'label_reg']]
        
        X = combined_df[feature_cols]
        y_cls = combined_df['label_cls']
        y_reg = combined_df['label_reg']
        
        # 首先检查并移除所有非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < X.shape[1]:
            non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
            logger.warning(f"发现非数值列，将被移除: {non_numeric_cols}")
            X = X[numeric_cols]
            feature_cols = numeric_cols
        
        # 修复特征名称警告：为特征列添加标准名称
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 数据清洗：处理无穷大和异常值
        # 保存原始索引以便对齐y
        original_index = X.index
        X_cleaned = self._clean_data(X)
        
        # 确保y与清洗后的X对齐
        if len(X_cleaned) < len(X):
            # 获取清洗后保留的索引
            remaining_indices = X_cleaned.index
            y_cls = y_cls.loc[remaining_indices]
            y_reg = y_reg.loc[remaining_indices]
            logger.info(f"标签数据已对齐，清洗后样本数量: {len(X_cleaned)}")
        
        X = X_cleaned
        
        # 数据质量检查
        self._check_data_quality(X, y_cls, y_reg)
        
        return X, pd.DataFrame({'cls': y_cls, 'reg': y_reg})
    
    def _get_stock_data_in_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指定日期范围内的股票数据"""
        try:
            with self.db.get_conn() as conn:
                query = """
                    SELECT symbol, date, open, high, low, close, volume
                    FROM prices_daily 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
                
                if df.empty:
                    return pd.DataFrame()
                
                # 转换数据类型
                df['date'] = pd.to_datetime(df['date'])
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 删除包含NaN的行
                df = df.dropna()
                
                return df
                
        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_labels(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """生成标签数据"""
        labels = []
        
        for symbol in symbols:
            try:
                # 使用新的日期范围查询方法获取数据
                bars = self._get_stock_data_in_range(symbol, start_date, end_date)
                if bars.empty:
                    continue
                
                bars = bars.sort_values('date')
                
                for i in range(len(bars) - 30):  # 确保有30天的未来数据
                    current_date = bars.iloc[i]['date']
                    current_price = bars.iloc[i]['close']
                    
                    # 30天后的价格
                    if i + 30 < len(bars):
                        future_price = bars.iloc[i + 30]['close']
                        return_rate = (future_price - current_price) / current_price
                        
                        # 分类标签：收益率 > 5% 为正样本
                        label_cls = 1 if return_rate > 0.05 else 0
                        
                        # 回归标签：实际收益率
                        label_reg = return_rate
                        
                        labels.append({
                            'symbol': symbol,
                            'date': current_date,
                            'label_cls': label_cls,
                            'label_reg': label_reg
                        })
                        
            except Exception as e:
                logger.warning(f"生成标签失败 {symbol}: {e}")
                continue
                
        return pd.DataFrame(labels)
    
    def _check_data_quality(self, X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series):
        """检查数据质量"""
        logger.info("=== 数据质量检查 ===")
        logger.info(f"特征数量: {X.shape[1]}")
        logger.info(f"样本数量: {X.shape[0]}")
        
        # 检查缺失值
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        logger.info(f"特征缺失值比例: {missing_ratio:.3f}")
        
        # 检查标签分布
        pos_ratio = y_cls.mean()
        logger.info(f"正样本比例: {pos_ratio:.3f}")
        
        # 检查收益率分布
        logger.info(f"收益率统计: 均值={y_reg.mean():.3f}, 标准差={y_reg.std():.3f}")
        logger.info(f"收益率范围: [{y_reg.min():.3f}, {y_reg.max():.3f}]")
        
        # 检查特征方差
        low_var_features = X.columns[X.var() < 1e-6].tolist()
        if low_var_features:
            logger.warning(f"低方差特征 ({len(low_var_features)}个): {low_var_features[:5]}...")
    
    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """清洗数据：处理无穷大和异常值"""
        logger.info("=== 数据清洗 ===")
        
        # 记录原始样本数量
        original_count = len(X)
        
        # 1. 处理无穷大值
        # 替换无穷大为NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 2. 处理极端异常值（使用IQR方法）
        for col in X.columns:
            # 跳过非数值列
            if not np.issubdtype(X[col].dtype, np.number):
                continue
                
            # 计算四分位数
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # 将超出3倍IQR的异常值替换为边界值
            X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
        
        # 3. 删除包含NaN的行（包括之前替换的无穷大值）
        X_cleaned = X.dropna()
        
        # 4. 检查并处理极端值（使用Z-score方法）
        # 对于数值列，计算Z-score并移除极端异常值
        numeric_cols = X_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            z_scores = np.abs((X_cleaned[numeric_cols] - X_cleaned[numeric_cols].mean()) / X_cleaned[numeric_cols].std())
            # 移除Z-score > 8的极端异常值
            extreme_outliers = (z_scores > 8).any(axis=1)
            X_cleaned = X_cleaned[~extreme_outliers]
        
        # 记录清洗结果
        cleaned_count = len(X_cleaned)
        removed_count = original_count - cleaned_count
        removal_ratio = removed_count / original_count
        
        logger.info(f"原始样本数: {original_count}")
        logger.info(f"清洗后样本数: {cleaned_count}")
        logger.info(f"移除样本数: {removed_count} ({removal_ratio:.3%})")
        
        if removed_count > 0:
            logger.warning(f"移除了 {removed_count} 个包含异常值的样本")
        
        return X_cleaned
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.DataFrame, 
                             save_dir: str = "models/enhanced",
                             model_types: List[str] = None) -> Dict[str, Any]:
        """训练集成模型"""
        logger.info("开始训练集成模型...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_types is None:
            model_types = ['logistic', 'randomforest', 'xgboost']
        
        # 初始化增强训练器
        trainer = EnhancedMLTrainer()
        
        results = {}
        
        # 训练分类模型
        logger.info("训练分类模型...")
        for model_type in model_types:
            try:
                logger.info(f"训练 {model_type} 分类模型...")
                result = trainer.train_single_model(X, y['cls'], model_type=model_type)
                results[f'{model_type}_cls'] = result
                
                # 保存模型
                model_path = os.path.join(save_dir, f"{model_type}_cls_{timestamp}.pkl")
                joblib.dump(result['model'], model_path)
                logger.info(f"{model_type} 分类模型保存到: {model_path}")
                
            except Exception as e:
                logger.error(f"训练 {model_type} 分类模型失败: {e}")
                continue
        
        # 训练回归模型
        logger.info("训练回归模型...")
        try:
            logger.info("训练 Ridge 回归模型...")
            reg_result = trainer.train_regression_model(X, y['reg'], use_grid_search=False)
            results['ridge_reg'] = reg_result
            
            # 保存回归模型
            reg_model_path = os.path.join(save_dir, f"ridge_reg_{timestamp}.pkl")
            joblib.dump(reg_result['model'], reg_model_path)
            logger.info(f"Ridge 回归模型保存到: {reg_model_path}")
            
        except Exception as e:
            logger.error(f"训练回归模型失败: {e}")
        
        # 模型评估
        logger.info("=== 模型评估 ===")
        self._evaluate_models(results, X, y)
        
        logger.info("集成模型训练完成！")
        return results
    
    def _evaluate_models(self, results: Dict[str, Any], X: pd.DataFrame, y: pd.DataFrame):
        """评估模型性能"""
        for model_name, result in results.items():
            if 'cls' in model_name:
                # 分类模型评估
                metrics = result['metrics']
                logger.info(f"{model_name} - AUC: {metrics['test_auc']:.4f}, 准确率: {metrics['test_accuracy']:.4f}")
            elif 'reg' in model_name:
                # 回归模型评估
                metrics = result['metrics']
                logger.info(f"{model_name} - R²: {metrics['test_r2']:.4f}, MSE: {metrics['test_mse']:.6f}")

def main():
    """主函数"""
    try:
        trainer = OptimizedEnsembleTrainer()
        
        # 准备训练数据
        X, y = trainer.prepare_training_data(
            start_date="2020-01-01",
            end_date="2024-01-01",
            min_samples=1000
        )
        
        # 训练集成模型
        results = trainer.train_ensemble_models(
            X, y,
            model_types=['logistic', 'randomforest', 'xgboost']
        )
        
        logger.info("优化集成模型训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()