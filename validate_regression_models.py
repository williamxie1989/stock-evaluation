#!/usr/bin/env python3
"""
验证回归模型性能
专门用于验证 train_regression_optimized.py 训练的回归模型
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from db import DatabaseManager
import pickle
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_regression_validation_data(lookback_days: int = 180):
    """准备回归验证数据"""
    logger.info("开始准备回归验证数据...")
    
    # 直接从数据库获取数据
    db = DatabaseManager()
    
    # 获取所有股票代码
    logger.info("获取股票列表...")
    with db.get_conn() as conn:
        stock_df = pd.read_sql_query("SELECT DISTINCT symbol FROM prices_daily", conn)
    symbols = stock_df['symbol'].tolist()
    
    # 设置验证日期范围（使用训练数据之后的时间段）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    logger.info(f"验证数据范围: {start_date} 到 {end_date}")
    
    all_features = []
    all_labels = []
    
    for symbol in symbols[:50]:  # 使用前50只股票
        try:
            with db.get_conn() as conn:
                query = """
                    SELECT symbol, date, open, high, low, close, volume
                    FROM prices_daily 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
                
                if not df.empty and len(df) >= 60:  # 至少需要60天数据
                    # 转换数据类型
                    df['date'] = pd.to_datetime(df['date'])
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 删除包含NaN的行
                    df = df.dropna()
                    
                    if len(df) >= 60:
                        # 生成简单特征（与训练时类似）
                        df['returns'] = df['close'].pct_change()
                        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                        df['sma_5'] = df['close'].rolling(5).mean()
                        df['sma_20'] = df['close'].rolling(20).mean()
                        df['volatility'] = df['returns'].rolling(20).std()
                        df['rsi'] = self._calculate_rsi(df['close'])
                        
                        # 移除NaN
                        df_features = df.dropna()
                        
                        # 生成回归标签（未来5天收益率）
                        close_prices = df['close'].values
                        labels = []
                        
                        for j in range(len(close_prices) - 5):
                            current_price = close_prices[j]
                            future_price = close_prices[j + 5]
                            return_rate = (future_price - current_price) / current_price
                            labels.append(return_rate)
                        
                        # 对齐特征和标签
                        aligned_features = df_features.iloc[:-5].copy()
                        aligned_labels = labels
                        
                        if len(aligned_features) != len(aligned_labels):
                            min_len = min(len(aligned_features), len(aligned_labels))
                            aligned_features = aligned_features.iloc[:min_len]
                            aligned_labels = aligned_labels[:min_len]
                        
                        all_features.append(aligned_features)
                        all_labels.extend(aligned_labels)
                        
        except Exception as e:
            logger.warning(f"处理股票 {symbol} 失败: {e}")
    
    if not all_features:
        raise ValueError("没有生成任何验证数据")
    
    # 合并所有数据
    X_combined = pd.concat(all_features, ignore_index=True)
    y_combined = pd.Series(all_labels, name='return_5d')
    
    logger.info(f"总共生成 {len(X_combined)} 个验证样本")
    
    # 数据预处理
    X_processed, y_processed = _preprocess_validation_data(X_combined, y_combined)
    
    return X_processed, y_processed

def _preprocess_validation_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """验证数据预处理"""
    logger.info("验证数据预处理...")
    
    # 移除非数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].copy()
    
    # 处理无穷大和NaN
    X_cleaned = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # 移除包含NaN的行
    valid_indices = X_cleaned.notna().all(axis=1)
    X_final = X_cleaned[valid_indices]
    y_final = y[valid_indices]
    
    logger.info(f"清洗后验证样本数量: {len(X_final)}")
    
    # 移除极端异常值（收益率在±50%之外）
    reasonable_returns = (y_final.abs() <= 0.5)
    X_final = X_final[reasonable_returns]
    y_final = y_final[reasonable_returns]
    
    logger.info(f"移除极端值后验证样本数量: {len(X_final)}")
    logger.info(f"验证收益率范围: {y_final.min():.3f} 到 {y_final.max():.3f}")
    
    # 标准化特征名称
    X_final.columns = [f'feature_{i}' for i in range(X_final.shape[1])]
    
    return X_final, y_final

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_regression_models():
    """加载回归模型"""
    logger.info("加载回归模型...")
    
    models = {}
    model_files = {
        'ridge': 'models/ridge_regression.pkl',
        'lasso': 'models/lasso_regression.pkl', 
        'elasticnet': 'models/elasticnet_regression.pkl',
        'randomforest': 'models/randomforest_regression.pkl',
        'xgboost': 'models/xgboost_regression.pkl'
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
                
                models[model_name] = model_obj
                logger.info(f"成功加载 {model_name} 回归模型")
                
            except Exception as e:
                logger.error(f"加载 {model_name} 回归模型失败: {e}")
        else:
            logger.warning(f"回归模型文件不存在: {model_path}")
    
    return models

def evaluate_regression_model(model, X_test, y_test, model_name):
    """评估回归模型性能"""
    try:
        # 获取预测结果
        y_pred = model.predict(X_test)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 计算预测误差统计
        errors = y_pred - y_test
        mean_error = errors.mean()
        std_error = errors.std()
        
        # 计算方向准确性
        direction_accuracy = np.mean((y_test * y_pred) > 0)
        
        logger.info(f"{model_name} 验证结果:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  平均误差: {mean_error:.6f}")
        logger.info(f"  误差标准差: {std_error:.6f}")
        logger.info(f"  方向准确性: {direction_accuracy:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'mean_error': mean_error,
            'std_error': std_error,
            'direction_accuracy': direction_accuracy,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"评估 {model_name} 回归模型失败: {e}")
        return None

def main():
    """主验证函数"""
    logger.info("=== 开始回归模型验证 ===")
    
    try:
        # 准备验证数据
        X_val, y_val = prepare_regression_validation_data(lookback_days=180)
        
        # 加载模型
        models = load_regression_models()
        
        if not models:
            logger.error("没有可用的回归模型进行验证")
            return
        
        logger.info(f"共有 {len(models)} 个回归模型待验证")
        
        # 评估每个模型
        validation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n验证 {model_name} 回归模型...")
            result = evaluate_regression_model(model, X_val, y_val, model_name)
            validation_results[model_name] = result
        
        # 汇总验证结果
        logger.info("\n" + "="*60)
        logger.info("=== 回归模型验证结果汇总 ===")
        logger.info("="*60)
        
        for model_name, result in validation_results.items():
            if result:
                logger.info(f"{model_name:12s}: R²={result['r2']:8.4f}, MSE={result['mse']:10.6f}, MAE={result['mae']:10.6f}, 方向准确性={result['direction_accuracy']:.4f}")
        
        # 找出最佳回归模型
        if validation_results:
            best_model = max(validation_results.items(), key=lambda x: x[1]['r2'] if x[1] else -float('inf'))
            if best_model[1]:
                logger.info(f"\n最佳回归模型: {best_model[0]} (R²={best_model[1]['r2']:.4f})")
        
        logger.info("\n回归模型验证完成!")
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        raise

if __name__ == "__main__":
    main()