#!/usr/bin/env python3
"""
验证统一训练脚本训练模型的性能
支持验证 train_unified_models.py 训练的所有模型
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from db import DatabaseManager
import pickle
import os
import glob

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
    
    # 使用与训练相同的特征生成器
    from enhanced_features import EnhancedFeatureGenerator
    feature_generator = EnhancedFeatureGenerator()
    
    all_features = []
    all_cls_labels = []
    all_reg_labels = []
    prediction_period = 5  # 与训练保持一致
    
    # 处理每只股票
    for i, symbol in enumerate(symbols[:100]):  # 使用前100只股票
        if i % 10 == 0:
            logger.info(f"处理第 {i+1}/{min(100, len(symbols))} 只股票: {symbol}")
        
        try:
            # 获取股票数据
            with db.get_conn() as conn:
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
            
            # 生成增强特征（与训练相同）
            features_df = feature_generator.generate_features(stock_data)
            if features_df.empty:
                continue
            
            # 生成标签（与训练相同）
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
        raise ValueError("没有生成任何验证数据")
    
    # 合并所有数据
    X_combined = pd.concat(all_features, ignore_index=True)
    y_reg = pd.Series(all_reg_labels, name='return_rate')
    y_cls = pd.Series(all_cls_labels, name='label_cls')
    
    logger.info(f"总共生成 {len(X_combined)} 个验证样本")
    
    # 数据预处理（与训练相同）
    X_processed, y_reg_processed, y_cls_processed = _preprocess_validation_data(X_combined, y_reg, y_cls)
    
    y = {'cls': y_cls_processed, 'reg': y_reg_processed}
    
    return X_processed, y

def _preprocess_validation_data(X: pd.DataFrame, y_reg: pd.Series, y_cls: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """验证数据预处理（与训练相同）"""
    logger.info("开始验证数据预处理...")
    
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
    
    logger.info(f"清洗后验证样本数量: {len(X_final)}")
    
    # 移除极端异常值（收益率在±50%之外）
    reasonable_returns = (y_reg_final.abs() <= 0.5)
    X_final = X_final[reasonable_returns]
    y_reg_final = y_reg_final[reasonable_returns]
    y_cls_final = y_cls_final[reasonable_returns]
    
    logger.info(f"移除极端值后验证样本数量: {len(X_final)}")
    logger.info(f"验证数据特征数量: {X_final.shape[1]}")
    logger.info(f"验证样本数量: {X_final.shape[0]}")
    logger.info(f"正样本比例: {y_cls_final.mean():.3f}")
    logger.info(f"收益率均值: {y_reg_final.mean():.6f}, 标准差: {y_reg_final.std():.6f}")
    
    # 标准化特征名称（与训练保持一致）
    X_final.columns = [f'feature_{i}' for i in range(X_final.shape[1])]
    
    return X_final, y_reg_final, y_cls_final

def load_unified_models():
    """加载统一训练脚本训练的所有模型"""
    logger.info("加载统一训练脚本训练的模型...")
    
    models = {}
    
    # 分类模型文件模式
    classification_patterns = [
        'logistic_classification.pkl',
        'randomforest_classification.pkl', 
        'xgboost_classification.pkl'
    ]
    
    # 回归模型文件模式
    regression_patterns = [
        'ridge_regression.pkl',
        'lasso_regression.pkl',
        'elasticnet_regression.pkl',
        'randomforest_regression.pkl',
        'xgboost_regression.pkl'
    ]
    
    # 查找所有模型文件
    model_files = {}
    
    for pattern in classification_patterns + regression_patterns:
        file_path = os.path.join('models', pattern)
        if os.path.exists(file_path):
            model_name = pattern.replace('.pkl', '')
            model_files[model_name] = file_path
            logger.info(f"找到模型文件: {file_path}")
    
    # 加载所有找到的模型
    for model_name, model_path in model_files.items():
        try:
            with open(model_path, 'rb') as f:
                model_obj = pickle.load(f)
            
            # 统一训练脚本保存的是完整的模型对象
            models[model_name] = model_obj
            logger.info(f"成功加载 {model_name} 模型")
            
        except Exception as e:
            logger.error(f"加载 {model_name} 模型失败: {e}")
    
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
        
        # 计算方向准确性（预测方向是否正确）
        direction_accuracy = np.mean((y_pred * y_test) > 0) if len(y_test) > 0 else 0.0
        
        logger.info(f"{model_name} 验证结果:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  平均误差: {mean_error:.6f}")
        logger.info(f"  误差标准差: {std_error:.6f}")
        logger.info(f"  方向准确性: {direction_accuracy:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'direction_accuracy': direction_accuracy,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"评估 {model_name} 模型失败: {e}")
        return None

def main():
    """主验证函数"""
    logger.info("=== 开始统一模型验证 ===")
    
    try:
        # 准备验证数据
        X_val, y_val = prepare_validation_data()
        
        # 加载模型
        models = load_unified_models()
        
        if not models:
            logger.error("没有可用的模型进行验证")
            return
        
        logger.info(f"共有 {len(models)} 个模型待验证")
        
        # 评估每个模型
        validation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n验证 {model_name} 模型...")
            
            if 'classification' in model_name:
                # 分类模型
                result = evaluate_classification_model(model, X_val, y_val['cls'], model_name)
                validation_results[model_name] = result
                
            elif 'regression' in model_name:
                # 回归模型
                result = evaluate_regression_model(model, X_val, y_val['reg'], model_name)
                validation_results[model_name] = result
        
        # 汇总验证结果
        logger.info("\n" + "="*80)
        logger.info("=== 统一模型验证结果汇总 ===")
        logger.info("="*80)
        
        # 分类模型结果
        cls_results = {k: v for k, v in validation_results.items() 
                       if 'classification' in k and v}
        if cls_results:
            logger.info("\n分类模型结果:")
            for model_name, result in cls_results.items():
                logger.info(f"{model_name}: 准确率={result['accuracy']:.4f}, AUC={result['auc']:.4f}, 正样本准确率={result['positive_accuracy']:.4f}")
            
            # 找出最佳分类模型
            best_cls_model = max(cls_results.items(), key=lambda x: x[1]['auc'])
            logger.info(f"\n最佳分类模型: {best_cls_model[0]} (AUC={best_cls_model[1]['auc']:.4f})")
        
        # 回归模型结果
        reg_results = {k: v for k, v in validation_results.items() 
                       if 'regression' in k and v}
        if reg_results:
            logger.info("\n回归模型结果:")
            for model_name, result in reg_results.items():
                logger.info(f"{model_name}: R²={result['r2']:.4f}, MSE={result['mse']:.6f}, 方向准确性={result['direction_accuracy']:.4f}")
            
            # 找出最佳回归模型
            best_reg_model = max(reg_results.items(), key=lambda x: x[1]['r2'])
            logger.info(f"\n最佳回归模型: {best_reg_model[0]} (R²={best_reg_model[1]['r2']:.4f})")
        
        # 总体统计
        logger.info("\n" + "="*80)
        logger.info("=== 总体统计 ===")
        logger.info(f"验证样本数量: {X_val.shape[0]}")
        logger.info(f"验证特征数量: {X_val.shape[1]}")
        logger.info(f"成功验证的模型数量: {len([v for v in validation_results.values() if v])}/{len(models)}")
        
        logger.info("\n验证完成!")
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        raise

if __name__ == "__main__":
    main()