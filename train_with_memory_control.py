#!/usr/bin/env python3
"""
内存控制的模型训练脚本
专门解决大规模数据训练的内存问题
"""

import os
import sys
import gc
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import signal
import resource
from contextlib import contextmanager
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_controlled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 内存限制配置
MAX_MEMORY_GB = 3.0  # 最大内存使用限制
MEMORY_CHECK_INTERVAL = 30  # 内存检查间隔（秒）
BATCH_SIZE = 20  # 每批股票数量
CHUNK_SIZE = 5000  # 数据分块大小

# 设置环境变量优化内存使用
os.environ.update({
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 
    'OPENBLAS_NUM_THREADS': '1',
    'NUMBA_NUM_THREADS': '1',
    'PYTHONHASHSEED': '42'
})

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from db import DatabaseManager
    from ml_trainer import MLTrainer
    from features import FeatureGenerator
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, max_memory_gb: float = MAX_MEMORY_GB):
        self.max_memory_gb = max_memory_gb
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（GB）"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def check_memory(self) -> bool:
        """检查内存是否超限"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_gb:
            logger.warning(f"内存使用超限: {current_memory:.2f}GB > {self.max_memory_gb}GB")
            return False
        return True
    
    def force_garbage_collect(self):
        """强制垃圾回收"""
        gc.collect()
        current_memory = self.get_memory_usage()
        logger.info(f"垃圾回收后内存使用: {current_memory:.2f}GB")

@contextmanager
def memory_limit(max_memory_gb: float = MAX_MEMORY_GB):
    """内存限制上下文管理器"""
    monitor = MemoryMonitor(max_memory_gb)
    
    def signal_handler(signum, frame):
        logger.error("内存超限，强制退出")
        sys.exit(1)
    
    signal.signal(signal.SIGUSR1, signal_handler)
    
    try:
        yield monitor
    finally:
        monitor.force_garbage_collect()

class ChunkedDataProcessor:
    """分块数据处理器"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.monitor = MemoryMonitor()
        
    def process_in_chunks(self, data: pd.DataFrame, processor_func) -> List[Dict[str, pd.DataFrame]]:
        """分块处理数据"""
        results = []
        
        for start_idx in range(0, len(data), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(data))
            chunk = data.iloc[start_idx:end_idx].copy()
            
            logger.info(f"处理数据块: {start_idx}-{end_idx}/{len(data)}")
            
            try:
                result = processor_func(chunk)
                # 检查返回的是字典且包含有效数据
                if isinstance(result, dict) and 'features' in result and 'labels' in result:
                    if not result['features'].empty and not result['labels'].empty:
                        results.append(result)
                    
                # 内存检查
                if not self.monitor.check_memory():
                    self.monitor.force_garbage_collect()
                    
            except Exception as e:
                logger.error(f"处理数据块失败: {e}")
                continue
                
        return results

class MemoryControlledTrainer:
    """内存控制的训练器"""
    
    def __init__(self, max_memory_gb: float = MAX_MEMORY_GB):
        self.db_manager = DatabaseManager()
        self.max_memory_gb = max_memory_gb
        self.monitor = MemoryMonitor(max_memory_gb)
        self.chunk_processor = ChunkedDataProcessor()
        
    def prepare_data_memory_safe(self, start_date: str = "2022-01-01",
                                end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """内存安全的数据准备"""
        logger.info(f"开始内存安全的数据准备: {start_date} 到 {end_date or '今天'}")
        
        # 检查初始内存状态
        initial_memory = self.monitor.get_memory_usage()
        logger.info(f"初始内存使用: {initial_memory:.2f}GB")
        
        # 获取股票列表 - 从samples表中获取
        with self.db_manager.get_conn() as conn:
            symbols_df = pd.read_sql_query(
                "SELECT DISTINCT symbol FROM samples ORDER BY symbol",
                conn
            )
        
        if symbols_df.empty:
            logger.warning("samples表中没有找到数据，尝试从其他表获取股票列表")
            # 尝试从prices_daily表获取股票代码
            with self.db_manager.get_conn() as conn:
                symbols_df = pd.read_sql_query(
                    "SELECT DISTINCT symbol FROM prices_daily ORDER BY symbol",
                    conn
                )
            if symbols_df.empty:
                raise ValueError("没有找到任何股票数据")
            
        all_symbols = symbols_df['symbol'].tolist()
        logger.info(f"总共 {len(all_symbols)} 只股票")
        
        # 分批处理股票
        all_features = []
        all_labels = []
        
        for i in range(0, len(all_symbols), BATCH_SIZE):
            batch_symbols = all_symbols[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(all_symbols) - 1) // BATCH_SIZE + 1
            
            logger.info(f"处理批次 {batch_num}/{total_batches}: {len(batch_symbols)} 只股票")
            
            try:
                # 处理批次
                batch_features, batch_labels = self._process_batch_memory_safe(batch_symbols, start_date, end_date)
                
                if not batch_features.empty:
                    all_features.append(batch_features)
                    all_labels.append(batch_labels)
                    
                # 定期内存检查和回收
                if batch_num % 5 == 0:
                    self.monitor.force_garbage_collect()
                    
                # 内存超限检查
                if not self.monitor.check_memory():
                    logger.warning("内存使用接近上限，暂停处理")
                    break
                    
            except Exception as e:
                logger.error(f"批次 {batch_num} 处理失败: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有成功处理任何股票数据")
        
        # 合并所有数据
        logger.info("合并所有批次数据...")
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # 最终内存检查
        final_memory = self.monitor.get_memory_usage()
        logger.info(f"数据准备完成: {len(X)} 条样本, {X.shape[1]} 个特征, 内存使用: {final_memory:.2f}GB")
        
        return X, y
    
    def _process_batch_memory_safe(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """内存安全的批次处理"""
        features_list = []
        labels_list = []
        
        for symbol in symbols:
            try:
                # 获取股票数据 - 使用正确的表名prices_daily
                query = """
                SELECT date, open, high, low, close, volume
                FROM prices_daily 
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
                """
                
                with self.db_manager.get_conn() as conn:
                    df = pd.read_sql_query(query, conn, params=[
                        symbol, 
                        start_date, 
                        end_date or datetime.now().strftime('%Y-%m-%d')
                    ])
                
                if len(df) < 60:  # 确保有足够的历史数据
                    continue
                    
                # 分块处理数据
                chunk_results = self.chunk_processor.process_in_chunks(df, self._process_single_stock)
                
                for result in chunk_results:
                    if 'features' in result and 'labels' in result:
                        features_list.append(result['features'])
                        labels_list.append(result['labels'])
                
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
                continue
        
        if not features_list:
            return pd.DataFrame(), pd.DataFrame()
            
        X_batch = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
        y_batch = pd.concat(labels_list, ignore_index=True) if labels_list else pd.DataFrame()
        
        return X_batch, y_batch
    
    def _process_single_stock(self, df_chunk: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """处理单个股票数据块 - 简化版本"""
        try:
            if len(df_chunk) < 30:
                return {'features': pd.DataFrame(), 'labels': pd.DataFrame()}
            
            # 简化特征生成 - 使用基础技术指标
            features_df = self._generate_simple_features(df_chunk)
            
            if features_df.empty:
                return {'features': pd.DataFrame(), 'labels': pd.DataFrame()}
            
            # 生成标签
            labels_df = self._generate_labels_chunk(df_chunk)
            
            if labels_df.empty:
                return {'features': pd.DataFrame(), 'labels': pd.DataFrame()}
            
            # 合并特征和标签
            merged_df = pd.merge(features_df, labels_df, on='date', how='inner')
            
            if merged_df.empty:
                return {'features': pd.DataFrame(), 'labels': pd.DataFrame()}
            
            # 分离特征和标签
            feature_cols = [col for col in merged_df.columns if col not in ['symbol', 'date', 'label_cls', 'label_reg']]
            X_chunk = merged_df[feature_cols].copy()
            y_chunk = merged_df[['label_cls', 'label_reg']].copy()
            
            # 数据清洗 - 修复数据类型转换错误
            for col in X_chunk.columns:
                if X_chunk[col].dtype == 'object':
                    try:
                        X_chunk[col] = pd.to_numeric(X_chunk[col], errors='coerce')
                    except:
                        X_chunk[col] = 0
            
            X_chunk = X_chunk.fillna(X_chunk.median())
            X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 确保所有特征都是数值类型
            X_chunk = X_chunk.astype(float)
            
            # 移除缺失值
            valid_mask = ~(X_chunk.isnull().any(axis=1) | y_chunk.isnull().any(axis=1))
            X_chunk = X_chunk[valid_mask]
            y_chunk = y_chunk[valid_mask]
            
            return {'features': X_chunk, 'labels': y_chunk}
            
        except Exception as e:
            logger.error(f"处理股票数据块失败: {e}")
            return {'features': pd.DataFrame(), 'labels': pd.DataFrame()}
    
    def _generate_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成简化特征 - 避免复杂计算"""
        try:
            if len(df) < 20:
                return pd.DataFrame()
            
            features_list = []
            
            for i in range(10, len(df)):  # 从第10天开始
                try:
                    recent_data = df.iloc[i-10:i+1].copy()
                    current_data = df.iloc[i].copy()
                    
                    # 确保数据类型正确
                    close_price = float(current_data['close'])
                    volume = float(current_data['volume'])
                    recent_close = recent_data['close'].astype(float)
                    recent_volume = recent_data['volume'].astype(float)
                    
                    # 基础价格特征
                    features = {
                        'symbol': str(current_data.get('symbol', 'unknown')),
                        'date': str(current_data['date']),
                        'close': close_price,
                        'volume': volume,
                        'price_change': (close_price - recent_close.iloc[0]) / recent_close.iloc[0],
                        'volume_avg': recent_volume.mean(),
                        'price_volatility': recent_close.std() / recent_close.mean(),
                        'price_trend': (close_price - recent_close.mean()) / recent_close.std() if recent_close.std() > 0 else 0,
                        'volume_ratio': close_price / recent_volume.mean() if recent_volume.mean() > 0 else 1
                    }
                    
                    features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"生成单日特征失败: {e}")
                    continue
            
            if not features_list:
                return pd.DataFrame()
                
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"生成简化特征失败: {e}")
            return pd.DataFrame()
    
    def _generate_labels_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """为数据块生成标签"""
        labels = []
        
        for i in range(len(df_chunk) - 30):
            try:
                current_date = str(df_chunk.iloc[i]['date'])
                current_price = float(df_chunk.iloc[i]['close'])
                future_price = float(df_chunk.iloc[i + 30]['close'])
                
                return_rate = (future_price - current_price) / current_price
                
                # 分类标签：收益率 > 5% 为正样本
                label_cls = 1 if return_rate > 0.05 else 0
                
                labels.append({
                    'symbol': str(df_chunk.iloc[i].get('symbol', 'unknown')),
                    'date': current_date,
                    'label_cls': label_cls,
                    'label_reg': return_rate
                })
                
            except Exception as e:
                logger.warning(f"生成标签失败: {e}")
                continue
        
        return pd.DataFrame(labels) if labels else pd.DataFrame()
    
    def train_models_memory_safe(self, X: pd.DataFrame, y: pd.DataFrame, 
                               save_dir: str = "models") -> Dict[str, Any]:
        """内存安全的模型训练"""
        logger.info("开始内存安全的模型训练...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        try:
            # 检查内存状态
            train_memory = self.monitor.get_memory_usage()
            logger.info(f"训练前内存使用: {train_memory:.2f}GB")
            
            # 1. 训练分类模型 - 使用最简配置
            logger.info("训练分类模型...")
            
            # 使用预定义的优化参数，避免网格搜索
            optimal_params = {
                'C': 0.1,
                'penalty': 'l1',
                'solver': 'liblinear',
                'max_iter': 500,  # 减少迭代次数
                'class_weight': 'balanced'
            }
            
            trainer_cls = MLTrainer(use_enhanced_features=False)
            
            # 使用较小的测试集比例减少内存使用
            cls_result = trainer_cls.train_model(
                X, y['label_cls'],
                test_size=0.15,  # 减少测试集大小
                use_grid_search=False,
                pipeline_params=optimal_params,
                cv_folds=3  # 减少交叉验证折数
            )
            
            cls_model = cls_result['model']
            cls_scaler = cls_result['model'].named_steps['scaler']
            
            # 保存分类模型
            cls_model_path = os.path.join(save_dir, f'memory_safe_cls_{timestamp}.pkl')
            cls_scaler_path = os.path.join(save_dir, f'memory_safe_cls_scaler_{timestamp}.pkl')
            
            joblib.dump(cls_model, cls_model_path)
            joblib.dump(cls_scaler, cls_scaler_path)
            
            results['classification'] = {
                'model_path': cls_model_path,
                'scaler_path': cls_scaler_path,
                'metrics': cls_result['metrics']
            }
            
            logger.info(f"分类模型训练完成: AUC={cls_result['metrics']['test_auc']:.4f}")
            
            # 强制内存回收
            self.monitor.force_garbage_collect()
            
            # 2. 训练回归模型
            logger.info("训练回归模型...")
            
            # 使用简化的回归参数
            reg_optimal_params = {'alpha': 1.0}
            
            reg_result = trainer_cls.train_regression_model(
                X, y['label_reg'],
                test_size=0.15,
                use_grid_search=False,
                cv_folds=3,
                scoring='neg_mean_absolute_error'
            )
            
            reg_model = reg_result['model']
            reg_scaler = reg_result['model'].named_steps['scaler']
            
            # 保存回归模型
            reg_model_path = os.path.join(save_dir, f'memory_safe_reg_{timestamp}.pkl')
            reg_scaler_path = os.path.join(save_dir, f'memory_safe_reg_scaler_{timestamp}.pkl')
            
            joblib.dump(reg_model, reg_model_path)
            joblib.dump(reg_scaler, reg_scaler_path)
            
            results['regression'] = {
                'model_path': reg_model_path,
                'scaler_path': reg_scaler_path,
                'metrics': reg_result['metrics']
            }
            
            logger.info(f"回归模型训练完成: R2={reg_result['metrics']['test_r2']:.4f}")
            
            # 最终内存状态
            final_memory = self.monitor.get_memory_usage()
            logger.info(f"训练完成，最终内存使用: {final_memory:.2f}GB")
            
            return results
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """主函数"""
    logger.info("开始内存控制的模型训练...")
    
    try:
        # 检查系统资源
        memory_info = psutil.virtual_memory()
        logger.info(f"系统总内存: {memory_info.total / (1024**3):.2f}GB")
        logger.info(f"可用内存: {memory_info.available / (1024**3):.2f}GB")
        
        if memory_info.available / (1024**3) < MAX_MEMORY_GB:
            logger.warning(f"可用内存不足 {MAX_MEMORY_GB}GB，可能影响训练")
        
        # 使用内存限制上下文
        with memory_limit(MAX_MEMORY_GB):
            # 初始化训练器
            trainer = MemoryControlledTrainer()
            
            # 准备训练数据
            X, y = trainer.prepare_data_memory_safe(
                start_date="2022-01-01",
                end_date=None
            )
            
            # 训练模型
            results = trainer.train_models_memory_safe(X, y)
            
            logger.info("内存控制的模型训练完成！")
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