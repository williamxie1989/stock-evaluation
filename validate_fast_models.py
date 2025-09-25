#!/usr/bin/env python3
"""
验证快速训练模型的性能
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from enhanced_ml_trainer import EnhancedMLTrainer
from db import DatabaseManager
import pickle
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_validation_data():
    """准备验证数据（使用与训练不同的时间段）"""
    logger.info("开始准备验证数据...")
    
    # 直接从数据库获取数据
    db = DatabaseManager()
    
    # 获取所有股票代码
    logger.info("获取股票列表...")
    with db.get_conn() as conn:
        stock_df = pd.read_sql_query("SELECT DISTINCT symbol FROM prices_daily", conn)
    symbols = stock_df['symbol'].tolist()
    
    # 设置验证日期范围（使用训练数据之后的时间段）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 最近1年数据
    
    logger.info(f"验证数据范围: {start_date} 到 {end_date}")
    
    # 获取特征数据
    logger.info("获取特征数据...")
    all_features = []
    
    for symbol in symbols[:100]:  # 使用前100只股票
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
        raise ValueError("没有生成任何验证数据")
    
    # 合并所有特征数据
    combined_features = pd.concat(all_features, ignore_index=True)
    logger.info(f"总共生成 {len(combined_features)} 条验证特征样本")
    
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
    logger.info(f"合并后验证样本数量: {len(combined_df)}")
    
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
    X_cleaned = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 对齐标签
    if len(X_cleaned) < len(X):
        remaining_indices = X_cleaned.index
        y_cls = y_cls.loc[remaining_indices]
        y_reg = y_reg.loc[remaining_indices]
        logger.info(f"标签数据已对齐，清洗后验证样本数量: {len(X_cleaned)}")
    
    X = X_cleaned
    
    # 数据质量检查
    logger.info("验证数据质量检查...")
    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    logger.info(f"正样本比例: {y_cls.mean():.3f}")
    logger.info(f"收益率均值: {y_reg.mean():.3f}, 标准差: {y_reg.std():.3f}")
    
    y = {'cls': y_cls, 'reg': y_reg}
    
    return X, y

def load_models():
    """加载训练好的模型"""
    logger.info("加载训练好的模型...")
    
    models = {}
    model_files = {
        'logistic': 'models/logistic_fast.pkl',
        'randomforest': 'models/randomforest_fast.pkl',
        'xgboost': 'models/xgboost_fast.pkl',
        'ridge': 'models/ridge_fast.pkl'
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
                
                # 检查对象类型并提取模型
                if hasattr(model_obj, 'named_steps') and 'logistic' in model_obj.named_steps:
                    # 这是Pipeline对象，提取实际的模型
                    actual_model = model_obj.named_steps[model_name]
                    models[model_name] = actual_model
                else:
                    # 直接是模型对象
                    models[model_name] = model_obj
                    
                logger.info(f"成功加载 {model_name} 模型")
            except Exception as e:
                logger.error(f"加载 {model_name} 模型失败: {e}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")
    
    return models

def evaluate_classification_model(model, X_test, y_test, model_name):
    """评估分类模型性能"""
    try:
        # 获取预测结果
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5
        
        # 计算正样本预测准确率
        positive_mask = y_test == 1
        if positive_mask.sum() > 0:
            positive_accuracy = accuracy_score(y_test[positive_mask], y_pred[positive_mask])
        else:
            positive_accuracy = 0.0
        
        logger.info(f"{model_name} 验证结果:")
        logger.info(f"  准确率: {accuracy:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  正样本准确率: {positive_accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'positive_accuracy': positive_accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
    except Exception as e:
        logger.error(f"评估 {model_name} 模型失败: {e}")
        return None

def evaluate_regression_model(model, X_test, y_test, model_name):
    """评估回归模型性能"""
    try:
        # 获取预测结果
        y_pred = model.predict(X_test)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算预测误差统计
        errors = y_pred - y_test
        mean_error = errors.mean()
        std_error = errors.std()
        
        logger.info(f"{model_name} 验证结果:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  平均误差: {mean_error:.4f}")
        logger.info(f"  误差标准差: {std_error:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"评估 {model_name} 模型失败: {e}")
        return None

def main():
    """主验证函数"""
    logger.info("=== 开始模型验证 ===")
    
    try:
        # 准备验证数据
        X_val, y_val = prepare_validation_data()
        
        # 加载模型
        models = load_models()
        
        if not models:
            logger.error("没有可用的模型进行验证")
            return
        
        logger.info(f"共有 {len(models)} 个模型待验证")
        
        # 评估每个模型
        validation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n验证 {model_name} 模型...")
            
            if model_name in ['logistic', 'randomforest', 'xgboost']:
                # 分类模型
                result = evaluate_classification_model(model, X_val, y_val['cls'], model_name)
                validation_results[model_name] = result
                
            elif model_name == 'ridge':
                # 回归模型
                result = evaluate_regression_model(model, X_val, y_val['reg'], model_name)
                validation_results[model_name] = result
        
        # 汇总验证结果
        logger.info("\n" + "="*60)
        logger.info("=== 模型验证结果汇总 ===")
        logger.info("="*60)
        
        for model_name, result in validation_results.items():
            if result:
                if model_name in ['logistic', 'randomforest', 'xgboost']:
                    logger.info(f"{model_name}: 准确率={result['accuracy']:.4f}, AUC={result['auc']:.4f}, 正样本准确率={result['positive_accuracy']:.4f}")
                elif model_name == 'ridge':
                    logger.info(f"{model_name}: MSE={result['mse']:.4f}, R²={result['r2']:.4f}")
        
        # 找出最佳分类模型
        cls_results = {k: v for k, v in validation_results.items() 
                       if k in ['logistic', 'randomforest', 'xgboost'] and v}
        
        if cls_results:
            best_cls_model = max(cls_results.items(), key=lambda x: x[1]['auc'])
            logger.info(f"\n最佳分类模型: {best_cls_model[0]} (AUC={best_cls_model[1]['auc']:.4f})")
        
        logger.info("\n验证完成!")
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        raise

if __name__ == "__main__":
    main()