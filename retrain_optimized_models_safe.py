#!/usr/bin/env python3
"""
优化的模型重训练脚本 - 内存安全版本
解决内存泄漏和worker超时问题
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrain_safe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 限制内存使用和并行度
MAX_MEMORY_GB = 4  # 最大内存使用
MAX_WORKERS = 2    # 最大工作进程数
BATCH_SIZE = 25    # 每批股票数量（减少内存压力）

# 设置环境变量限制内存
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from db import DatabaseManager
    from ml_trainer import MLTrainer
    from features import FeatureGenerator
    from concurrent_data_sync_service import ConcurrentDataSyncService
    from concurrent.futures import TimeoutError
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

class MemorySafeTrainer:
    """内存安全的训练器"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_sync = ConcurrentDataSyncService()
        
    def prepare_training_data_safe(self, start_date: str = "2022-01-01", 
                                 end_date: str = None,
                                 max_stocks_per_batch: int = BATCH_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """内存安全的数据准备"""
        logger.info(f"开始安全数据准备: {start_date} 到 {end_date or '今天'}")
        
        # 获取股票列表
        with self.db_manager.get_conn() as conn:
            symbols_df = pd.read_sql_query(
                "SELECT DISTINCT symbol FROM stock_info WHERE status = 'active' ORDER BY symbol", 
                conn
            )
        
        if symbols_df.empty:
            raise ValueError("没有找到活跃的股票")
            
        all_symbols = symbols_df['symbol'].tolist()
        logger.info(f"总共 {len(all_symbols)} 只股票，每批处理 {max_stocks_per_batch} 只")
        
        # 分批处理，避免内存溢出
        all_features = []
        all_labels = []
        
        for i in range(0, len(all_symbols), max_stocks_per_batch):
            batch_symbols = all_symbols[i:i + max_stocks_per_batch]
            logger.info(f"处理批次 {i//max_stocks_per_batch + 1}/{(len(all_symbols)-1)//max_stocks_per_batch + 1}: {len(batch_symbols)} 只股票")
            
            try:
                # 处理单批数据
                batch_features, batch_labels = self._process_batch_safe(batch_symbols, start_date, end_date)
                
                if not batch_features.empty:
                    all_features.append(batch_features)
                    all_labels.append(batch_labels)
                    
                # 强制垃圾回收
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有成功处理任何股票数据")
            
        # 合并所有数据
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        logger.info(f"数据准备完成: {len(X)} 条样本, {X.shape[1]} 个特征")
        return X, y
    
    def _process_batch_safe(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """安全处理单批股票"""
        features_list = []
        labels_list = []
        
        for symbol in tqdm(symbols, desc="处理股票"):
            try:
                # 获取股票数据
                query = """
                SELECT date, open, high, low, close, volume, turnover, amplitude, change_rate
                FROM stock_prices 
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
                """
                
                with self.db_manager.get_conn() as conn:
                    df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date or datetime.now().strftime('%Y-%m-%d')])
                
                if len(df) < 60:  # 确保有足够的历史数据
                    continue
                    
                # 生成特征
                feature_gen = FeatureGenerator()
                features_df = feature_gen.generate_features_batch(df)
                
                if features_df.empty or len(features_df) < 30:
                    continue
                
                # 生成标签（30天收益率）
                labels_df = self._generate_labels_safe(df, symbol)
                
                if labels_df.empty:
                    continue
                
                # 合并特征和标签
                merged_df = pd.merge(features_df, labels_df, on='date', how='inner')
                
                if merged_df.empty:
                    continue
                
                # 分离特征和标签
                feature_cols = [col for col in merged_df.columns if col not in ['symbol', 'date', 'label_cls', 'label_reg']]
                X_batch = merged_df[feature_cols].copy()
                y_batch = merged_df[['label_cls', 'label_reg']].copy()
                
                # 数据清洗
                X_batch = X_batch.fillna(X_batch.median())
                X_batch = X_batch.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 移除缺失值
                valid_mask = ~(X_batch.isnull().any(axis=1) | y_batch.isnull().any(axis=1))
                X_batch = X_batch[valid_mask]
                y_batch = y_batch[valid_mask]
                
                if len(X_batch) > 0:
                    features_list.append(X_batch)
                    labels_list.append(y_batch)
                    
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
                continue
        
        if not features_list:
            return pd.DataFrame(), pd.DataFrame()
            
        X_batch = pd.concat(features_list, ignore_index=True)
        y_batch = pd.concat(labels_list, ignore_index=True)
        
        return X_batch, y_batch
    
    def _generate_labels_safe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """安全生成标签"""
        labels = []
        
        for i in range(len(df) - 30):
            current_date = df.iloc[i]['date']
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + 30]['close']
            
            return_rate = (future_price - current_price) / current_price
            
            # 分类标签：收益率 > 5% 为正样本
            label_cls = 1 if return_rate > 0.05 else 0
            
            labels.append({
                'symbol': symbol,
                'date': current_date,
                'label_cls': label_cls,
                'label_reg': return_rate
            })
        
        return pd.DataFrame(labels)
    
    def train_models_safe(self, X: pd.DataFrame, y: pd.DataFrame, 
                         save_dir: str = "models") -> Dict[str, Any]:
        """内存安全的模型训练"""
        logger.info("开始内存安全的模型训练...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        try:
            # 1. 训练分类模型 - 简化参数搜索
            logger.info("训练分类模型...")
            
            # 使用简化的参数配置
            simple_params = {
                'logistic__C': 0.1,
                'logistic__penalty': 'l1',
                'logistic__solver': 'liblinear',
                'logistic__max_iter': 1000,
                'logistic__class_weight': 'balanced'
            }
            
            trainer_cls = MLTrainer(use_enhanced_features=False)
            
            # 训练模型（不使用网格搜索，直接使用优化参数）
            cls_result = trainer_cls.train_model(
                X, y['label_cls'], 
                use_grid_search=False, 
                pipeline_params=simple_params
            )
            
            cls_model = cls_result['model']
            cls_scaler = cls_result['model'].named_steps['scaler']
            
            # 保存分类模型
            cls_model_path = os.path.join(save_dir, f'safe_cls_model_{timestamp}.pkl')
            cls_scaler_path = os.path.join(save_dir, f'safe_cls_scaler_{timestamp}.pkl')
            
            import joblib
            joblib.dump(cls_model, cls_model_path)
            joblib.dump(cls_scaler, cls_scaler_path)
            
            results['classification'] = {
                'model_path': cls_model_path,
                'scaler_path': cls_scaler_path,
                'model': cls_model,
                'scaler': cls_scaler,
                'metrics': cls_result['metrics']
            }
            
            logger.info(f"分类模型训练完成: AUC={cls_result['metrics']['test_auc']:.4f}")
            
            # 强制内存回收
            import gc
            gc.collect()
            
            # 2. 训练回归模型
            logger.info("训练回归模型...")
            
            trainer_reg = MLTrainer(use_enhanced_features=False)
            
            # 使用简化的回归参数
            reg_params = {'ridge__alpha': 1.0}
            reg_result = trainer_reg.train_regression_model(
                X, y['label_reg'],
                use_grid_search=False,
                scoring='neg_mean_absolute_error'
            )
            
            reg_model = reg_result['model']
            reg_scaler = reg_result['model'].named_steps['scaler']
            
            # 保存回归模型
            reg_model_path = os.path.join(save_dir, f'safe_reg_model_{timestamp}.pkl')
            reg_scaler_path = os.path.join(save_dir, f'safe_reg_scaler_{timestamp}.pkl')
            
            joblib.dump(reg_model, reg_model_path)
            joblib.dump(reg_scaler, reg_scaler_path)
            
            results['regression'] = {
                'model_path': reg_model_path,
                'scaler_path': reg_scaler_path,
                'model': reg_model,
                'scaler': reg_scaler,
                'metrics': reg_result['metrics']
            }
            
            logger.info(f"回归模型训练完成: R2={reg_result['metrics']['test_r2']:.4f}")
            
            # 评估模型
            self._evaluate_models_safe(X, y, results)
            
            logger.info(f"模型训练完成，保存在: {save_dir}")
            return results
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _evaluate_models_safe(self, X: pd.DataFrame, y: pd.DataFrame, results: Dict[str, Any]):
        """安全评估模型"""
        logger.info("=== 模型评估 ===")
        
        try:
            # 评估分类模型
            cls_model = results['classification']['model']
            cls_scaler = results['classification']['scaler']
            
            X_scaled = cls_scaler.transform(X)
            y_pred_proba = cls_model.predict_proba(X_scaled)[:, 1]
            y_pred = cls_model.predict(X_scaled)
            
            # 计算指标
            from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
            
            auc_score = roc_auc_score(y['label_cls'], y_pred_proba)
            accuracy = (y_pred == y['label_cls']).mean()
            pos_ratio = y_pred.mean()
            
            logger.info(f"分类模型评估:")
            logger.info(f"  AUC: {auc_score:.4f}")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  预测正样本比例: {pos_ratio:.4f}")
            
            # 评估回归模型
            reg_model = results['regression']['model']
            reg_scaler = results['regression']['scaler']
            
            X_scaled_reg = reg_scaler.transform(X)
            y_reg_pred = reg_model.predict(X_scaled_reg)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y['label_reg'], y_reg_pred)
            mae = mean_absolute_error(y['label_reg'], y_reg_pred)
            r2 = r2_score(y['label_reg'], y_reg_pred)
            
            logger.info(f"回归模型评估:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  R2: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")

def main():
    """主函数"""
    logger.info("开始内存安全的优化模型训练...")
    
    try:
        # 检查内存限制
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)
        logger.info(f"可用内存: {available_memory:.2f} GB")
        
        if available_memory < MAX_MEMORY_GB:
            logger.warning(f"可用内存不足 {MAX_MEMORY_GB}GB，可能影响训练")
        
        # 初始化训练器
        trainer = MemorySafeTrainer()
        
        # 准备训练数据
        X, y = trainer.prepare_training_data_safe(
            start_date="2022-01-01",
            end_date=None,
            max_stocks_per_batch=BATCH_SIZE
        )
        
        # 训练模型
        results = trainer.train_models_safe(X, y)
        
        logger.info("内存安全的模型训练完成！")
        logger.info("建议:")
        logger.info("1. 备份原有模型文件")
        logger.info("2. 将新模型文件重命名为原有模型文件名")
        logger.info("3. 重新运行选股程序测试效果")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()