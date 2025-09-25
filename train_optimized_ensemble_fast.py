#!/usr/bin/env python3
"""
优化版本的集成模型训练脚本
减少试验次数，提高训练效率
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ml_trainer import EnhancedMLTrainer
from enhanced_data_provider import EnhancedDataProvider
from enhanced_preprocessing import EnhancedPreprocessingPipeline
from db import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_training_data_fast():
    """准备训练数据（快速版本）"""
    logger.info("开始准备训练数据...")
    
    # 直接从数据库获取数据（与原始脚本相同的方式）
    db = DatabaseManager()
    
    # 获取所有股票代码
    logger.info("获取股票列表...")
    with db.get_conn() as conn:
        stock_df = pd.read_sql_query("SELECT DISTINCT symbol FROM prices_daily", conn)
    symbols = stock_df['symbol'].tolist()
    
    # 设置日期范围（使用最近2年数据以减少数据量）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    logger.info(f"使用数据范围: {start_date} 到 {end_date}")
    
    # 获取特征数据（简化版本）
    logger.info("获取特征数据...")
    all_features = []
    
    for symbol in symbols[:100]:  # 只使用前100只股票以加快速度
        try:
            with db.get_conn() as conn:
                query = """
                    SELECT symbol, date, open, high, low, close, volume
                    FROM prices_daily 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
                
                if not df.empty:
                    # 生成简单特征
                    df['returns'] = df['close'].pct_change()
                    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                    df['sma_5'] = df['close'].rolling(5).mean()
                    df['sma_20'] = df['close'].rolling(20).mean()
                    df['volatility'] = df['returns'].rolling(20).std()
                    
                    # 移除NaN
                    df = df.dropna()
                    
                    if not df.empty:
                        all_features.append(df)
                        
        except Exception as e:
            logger.warning(f"获取股票数据失败 {symbol}: {e}")
    
    if not all_features:
        raise ValueError("没有生成任何训练数据")
    
    # 合并所有特征数据
    combined_features = pd.concat(all_features, ignore_index=True)
    logger.info(f"总共生成 {len(combined_features)} 条特征样本")
    
    # 生成简单标签（30天收益率）
    logger.info("生成标签数据...")
    labels = []
    
    for symbol in combined_features['symbol'].unique():
        symbol_data = combined_features[combined_features['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        for i in range(len(symbol_data) - 30):
            current_price = symbol_data.iloc[i]['close']
            future_price = symbol_data.iloc[i + 30]['close']
            return_rate = (future_price - current_price) / current_price
            
            labels.append({
                'symbol': symbol,
                'date': symbol_data.iloc[i]['date'],
                'label_cls': 1 if return_rate > 0.05 else 0,
                'label_reg': return_rate
            })
    
    labels_df = pd.DataFrame(labels)
    
    # 合并特征和标签
    combined_df = combined_features.merge(labels_df, on=['symbol', 'date'], how='inner')
    logger.info(f"合并后样本数量: {len(combined_df)}")
    
    # 分离特征和标签
    feature_cols = [col for col in combined_df.columns 
                   if col not in ['symbol', 'date', 'label_cls', 'label_reg']]
    
    X = combined_df[feature_cols]
    y_cls = combined_df['label_cls']
    y_reg = combined_df['label_reg']
    
    # 移除非数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < X.shape[1]:
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        logger.warning(f"发现非数值列，将被移除: {non_numeric_cols}")
        X = X[numeric_cols]
    
    # 修复特征名称
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    # 简单数据清洗
    logger.info("数据清洗...")
    original_indices = X.index
    
    # 处理无穷大和NaN
    X_cleaned = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 对齐标签
    if len(X_cleaned) < len(X):
        remaining_indices = X_cleaned.index
        y_cls = y_cls.loc[remaining_indices]
        y_reg = y_reg.loc[remaining_indices]
        logger.info(f"标签数据已对齐，清洗后样本数量: {len(X_cleaned)}")
    
    X = X_cleaned
    
    # 数据质量检查
    logger.info("数据质量检查...")
    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    logger.info(f"正样本比例: {y_cls.mean():.3f}")
    logger.info(f"收益率均值: {y_reg.mean():.3f}, 标准差: {y_reg.std():.3f}")
    
    y = {'cls': y_cls, 'reg': y_reg}
    
    return X, y

def train_ensemble_models_fast(X, y):
    """训练集成模型（快速版本）"""
    logger.info("开始训练集成模型...")
    
    # 初始化训练器
    trainer = EnhancedMLTrainer()
    
    results = {}
    
    # 训练分类模型（减少试验次数）
    model_types = ['logistic', 'randomforest', 'xgboost']
    
    for model_type in model_types:
        logger.info(f"训练{model_type}模型...")
        
        try:
            # 使用快速配置：减少试验次数到10
            result = trainer.train_single_model(X, y['cls'], model_type, 
                                              use_optimization=True, 
                                              optimization_trials=10)
            results[model_type] = result
            
            # 保存模型
            model_name = f"{model_type}_fast.pkl"
            model_path = trainer.save_model(result['model'], model_name)
            logger.info(f"{model_type}模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"训练{model_type}模型失败: {e}")
    
    # 训练回归模型
    if y['reg'] is not None:
        logger.info("训练Ridge回归模型...")
        try:
            reg_result = trainer.train_regression_model(X, y['reg'], use_grid_search=False)
            results['ridge'] = reg_result
            
            # 保存回归模型
            model_name = "ridge_fast.pkl"
            model_path = trainer.save_model(reg_result['model'], model_name)
            logger.info(f"Ridge回归模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"训练回归模型失败: {e}")
    
    return results

def main():
    """主函数"""
    logger.info("=== 开始快速训练流程 ===")
    
    try:
        # 准备训练数据
        X, y = prepare_training_data_fast()
        
        if X is None or y is None:
            logger.error("数据准备失败")
            return
        
        # 训练模型
        results = train_ensemble_models_fast(X, y)
        
        # 评估结果
        logger.info("\n=== 训练结果汇总 ===")
        for model_type, result in results.items():
            if 'best_score' in result:
                logger.info(f"{model_type}: 最佳得分 = {result['best_score']:.4f}")
        
        logger.info("快速训练完成!")
        
    except Exception as e:
        logger.error(f"训练流程失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()