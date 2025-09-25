#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化模型重训练脚本
使用改进的参数配置重新训练模型，解决预测过于极端的问题
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
from ml_trainer import MLTrainer
from features import FeatureGenerator
from enhanced_features import EnhancedFeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedModelTrainer:
    """优化的模型训练器"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()  # 不传递参数
        
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
                    # 获取价格数据用于特征生成 - 使用正确的方法
                    bars = self.db.get_last_n_bars([symbol], n=200)  # 获取足够的历史数据
                    if bars.empty:
                        continue
                    
                    # 过滤指定日期范围的数据
                    bars['date'] = pd.to_datetime(bars['date'])
                    bars = bars[(bars['date'] >= start_date) & (bars['date'] <= end_date)]
                    
                    if bars.empty:
                        continue
                    
                    # 生成特征 - 需要为每个股票单独生成
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
        
        # 修复特征名称警告：为特征列添加标准名称
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 数据质量检查
        self._check_data_quality(X, y_cls, y_reg)
        
        return X, pd.DataFrame({'cls': y_cls, 'reg': y_reg})
    
    def _generate_labels(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """生成标签数据"""
        labels = []
        
        for symbol in symbols:
            try:
                # 获取价格数据 - 使用正确的方法
                bars = self.db.get_last_n_bars([symbol], n=200)  # 获取足够的历史数据
                if bars.empty:
                    continue
                
                # 过滤指定日期范围的数据
                bars['date'] = pd.to_datetime(bars['date'])
                bars = bars[(bars['date'] >= start_date) & (bars['date'] <= end_date)]
                
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
    
    def train_optimized_models(self, X: pd.DataFrame, y: pd.DataFrame, 
                             save_dir: str = "models") -> Dict[str, Any]:
        """训练优化的模型"""
        logger.info("开始训练优化模型...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        # 1. 训练分类模型 - 优化超参数提高区分度
        logger.info("训练分类模型...")
        trainer_cls = MLTrainer()
        
        # 训练模型
        cls_result = trainer_cls.train_model(X, y['cls'])
        cls_model = cls_result['model']
        cls_scaler = cls_result['model'].named_steps['scaler']
        
        # 保存分类模型
        cls_model_path = os.path.join(save_dir, f'optimized_cls_model_{timestamp}.pkl')
        cls_scaler_path = os.path.join(save_dir, f'optimized_cls_scaler_{timestamp}.pkl')
        
        joblib.dump(cls_model, cls_model_path)
        joblib.dump(cls_scaler, cls_scaler_path)
        
        results['classification'] = {
            'model_path': cls_model_path,
            'scaler_path': cls_scaler_path,
            'model': cls_model,
            'scaler': cls_scaler
        }
        
        # 2. 训练回归模型 - 优化参数提高预测精度
        logger.info("训练回归模型...")
        trainer_reg = MLTrainer()
        
        reg_result = trainer_reg.train_regression_model(X, y['reg'])
        reg_model = reg_result['model']
        reg_scaler = reg_result['model'].named_steps['scaler']
        
        # 保存回归模型
        reg_model_path = os.path.join(save_dir, f'optimized_reg_model_{timestamp}.pkl')
        reg_scaler_path = os.path.join(save_dir, f'optimized_reg_scaler_{timestamp}.pkl')
        
        joblib.dump(reg_model, reg_model_path)
        joblib.dump(reg_scaler, reg_scaler_path)
        
        results['regression'] = {
            'model_path': reg_model_path,
            'scaler_path': reg_scaler_path,
            'model': reg_model,
            'scaler': reg_scaler
        }
        
        # 3. 评估模型
        self._evaluate_models(X, y, results)
        
        logger.info(f"模型训练完成，保存在: {save_dir}")
        return results
    
    def _evaluate_models(self, X: pd.DataFrame, y: pd.DataFrame, results: Dict[str, Any]):
        """评估模型性能"""
        logger.info("=== 模型评估 ===")
        
        # 评估分类模型
        cls_model = results['classification']['model']
        cls_scaler = results['classification']['scaler']
        
        X_scaled = cls_scaler.transform(X)
        y_pred_proba = cls_model.predict_proba(X_scaled)[:, 1]
        y_pred = cls_model.predict(X_scaled)
        
        # 检查预测概率分布
        prob_mean = np.mean(y_pred_proba)
        prob_std = np.std(y_pred_proba)
        prob_min = np.min(y_pred_proba)
        prob_max = np.max(y_pred_proba)
        
        logger.info(f"分类模型预测概率统计:")
        logger.info(f"  均值: {prob_mean:.4f}")
        logger.info(f"  标准差: {prob_std:.4f}")
        logger.info(f"  范围: [{prob_min:.4f}, {prob_max:.4f}]")
        
        # 检查是否还会触发极端判断
        is_extreme = prob_std < 0.05 or prob_mean > 0.95 or prob_mean < 0.05
        logger.info(f"  是否极端: {'是' if is_extreme else '否'}")
        
        # 准确率
        accuracy = np.mean(y_pred == y['cls'])
        logger.info(f"  准确率: {accuracy:.4f}")
        
        # 正样本预测比例
        pos_pred_ratio = np.mean(y_pred)
        logger.info(f"  预测正样本比例: {pos_pred_ratio:.4f}")
        
        # 评估回归模型
        reg_model = results['regression']['model']
        reg_scaler = results['regression']['scaler']
        
        y_reg_pred = reg_model.predict(X_scaled)
        
        # 回归指标
        mse = np.mean((y_reg_pred - y['reg']) ** 2)
        mae = np.mean(np.abs(y_reg_pred - y['reg']))
        
        logger.info(f"回归模型统计:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  预测均值: {np.mean(y_reg_pred):.4f}")
        logger.info(f"  预测标准差: {np.std(y_reg_pred):.4f}")

def main():
    """主函数"""
    logger.info("开始优化模型训练...")
    
    try:
        # 初始化训练器
        trainer = OptimizedModelTrainer()
        
        # 准备训练数据
        X, y = trainer.prepare_training_data(
            start_date="2022-01-01",  # 使用更近期的数据
            end_date=None,
            min_samples=500
        )
        
        # 训练优化模型
        results = trainer.train_optimized_models(X, y)
        
        logger.info("优化模型训练完成！")
        logger.info("建议:")
        logger.info("1. 备份原有模型文件")
        logger.info("2. 将新模型文件重命名为原有模型文件名")
        logger.info("3. 重新运行选股程序测试效果")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()