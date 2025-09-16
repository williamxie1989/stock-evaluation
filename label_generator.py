#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签定义与样本生成模块
基于T日后的滚动收益构建二分类标签，处理去重与泄露防护
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from db import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelGenerator:
    """
    标签生成器
    基于未来收益率生成二分类标签
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def calculate_forward_returns(self, symbol: str, periods: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """
        计算前瞻收益率
        
        Args:
            symbol: 股票代码
            periods: 前瞻期数列表，如[5, 10, 20, 30]表示5日、10日、20日、30日后收益率
            
        Returns:
            包含前瞻收益率的DataFrame
        """
        try:
            # 获取价格数据
            bars = self.db_manager.get_last_n_bars([symbol], 500)
            if bars is None or len(bars) == 0:
                logger.warning(f"无法获取 {symbol} 的价格数据")
                return pd.DataFrame()
                
            # 确保数据按日期排序
            bars = bars.sort_values('date')
            bars = bars.reset_index(drop=True)
            
            # 计算各期前瞻收益率
            for period in periods:
                # 计算period日后的收益率
                bars[f'forward_return_{period}d'] = (
                    bars['close'].shift(-period) / bars['close'] - 1
                )
                
            # 移除没有前瞻收益率的行（最后几行）
            max_period = max(periods)
            bars = bars.iloc[:-max_period]
            
            logger.info(f"计算完成 {symbol} 的前瞻收益率，数据量: {len(bars)}")
            return bars
            
        except Exception as e:
            logger.error(f"计算前瞻收益率失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_labels(self, returns_df: pd.DataFrame, 
                       period: int = 10, 
                       threshold_method: str = 'quantile',
                       threshold_value: float = 0.7) -> pd.DataFrame:
        """
        生成二分类标签
        
        Args:
            returns_df: 包含前瞻收益率的DataFrame
            period: 使用的前瞻期数
            threshold_method: 阈值方法 ('quantile', 'absolute')
            threshold_value: 阈值参数
                - quantile方法: 分位数阈值，如0.7表示前30%为正样本
                - absolute方法: 绝对收益率阈值，如0.05表示5%以上为正样本
                
        Returns:
            包含标签的DataFrame
        """
        if returns_df.empty:
            return pd.DataFrame()
            
        df = returns_df.copy()
        return_col = f'forward_return_{period}d'
        
        if return_col not in df.columns:
            logger.error(f"缺少列: {return_col}")
            return pd.DataFrame()
            
        # 移除缺失值
        df = df.dropna(subset=[return_col])
        
        if len(df) == 0:
            logger.warning("移除缺失值后数据为空")
            return pd.DataFrame()
            
        # 根据方法生成标签
        if threshold_method == 'quantile':
            threshold = df[return_col].quantile(threshold_value)
            df[f'label_{period}d'] = (df[return_col] > threshold).astype(int)
        elif threshold_method == 'absolute':
            df[f'label_{period}d'] = (df[return_col] > threshold_value).astype(int)
        else:
            raise ValueError(f"不支持的阈值方法: {threshold_method}")
            
        # 添加标签统计信息
        positive_ratio = df[f'label_{period}d'].mean()
        logger.info(f"标签生成完成，正样本比例: {positive_ratio:.3f}")
        
        return df
    
    def create_samples(self, symbol: str, 
                      feature_start_date: str = None,
                      feature_end_date: str = None,
                      periods: List[int] = [5, 10, 20, 30],
                      threshold_method: str = 'quantile',
                      threshold_value: float = 0.7) -> Dict[str, pd.DataFrame]:
        """
        创建训练样本
        
        Args:
            symbol: 股票代码
            feature_start_date: 特征开始日期
            feature_end_date: 特征结束日期
            periods: 前瞻期数列表
            threshold_method: 阈值方法
            threshold_value: 阈值参数
            
        Returns:
            包含不同期数样本的字典
        """
        try:
            # 1. 获取特征数据
            features_df = self._get_features_from_db(symbol, feature_start_date, feature_end_date)
            if features_df.empty:
                logger.warning(f"无法获取 {symbol} 的特征数据")
                return {}
                
            # 2. 计算前瞻收益率
            returns_df = self.calculate_forward_returns(symbol, periods)
            if returns_df.empty:
                logger.warning(f"无法计算 {symbol} 的前瞻收益率")
                return {}
                
            # 3. 合并特征和收益率数据
            # 确保日期列格式一致
            features_df['date'] = pd.to_datetime(features_df['date'])
            returns_df['date'] = pd.to_datetime(returns_df['date'])
            
            # 按日期合并
            merged_df = pd.merge(features_df, returns_df[['date'] + [f'forward_return_{p}d' for p in periods]], 
                               on='date', how='inner')
            
            if merged_df.empty:
                logger.warning(f"特征和收益率数据合并后为空: {symbol}")
                return {}
                
            logger.info(f"合并后数据量: {len(merged_df)}")
            
            # 4. 为每个期数生成标签和样本
            samples = {}
            for period in periods:
                # 生成标签
                labeled_df = self.generate_labels(merged_df, period, threshold_method, threshold_value)
                if not labeled_df.empty:
                    # 移除泄露特征（如果有的话）
                    clean_df = self._remove_leakage_features(labeled_df, period)
                    samples[f'{period}d'] = clean_df
                    
                    logger.info(f"生成 {period}日期样本: {len(clean_df)} 条")
                    
            return samples
            
        except Exception as e:
            logger.error(f"创建样本失败 {symbol}: {e}")
            return {}
    
    def _get_features_from_db(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从数据库获取特征数据，将因子记录按日期聚合为特征向量
        """
        try:
            query = """
            SELECT * FROM factors 
            WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date, factor_name"
            
            with self.db_manager.get_conn() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if df.empty:
                return pd.DataFrame()
                
            # 将因子记录按日期聚合为特征向量
            features_df = df.pivot(index='date', columns='factor_name', values='value')
            features_df.reset_index(inplace=True)
            features_df['symbol'] = symbol
                
            logger.info(f"从数据库获取特征数据: {len(features_df)} 条日期, {len(features_df.columns)-2} 个特征")
            return features_df
            
        except Exception as e:
            logger.error(f"获取特征数据失败: {e}")
            return pd.DataFrame()
    
    def _remove_leakage_features(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        移除可能导致数据泄露的特征
        
        Args:
            df: 包含特征和标签的DataFrame
            period: 前瞻期数
            
        Returns:
            清理后的DataFrame
        """
        # 这里可以根据具体情况移除可能泄露的特征
        # 例如：移除基于未来信息计算的特征
        
        # 目前保持所有特征，后续可根据需要调整
        return df
    
    def save_samples_to_db(self, symbol: str, samples: Dict[str, pd.DataFrame]) -> bool:
        """
        保存样本到数据库
        
        Args:
            symbol: 股票代码
            samples: 样本数据字典
            
        Returns:
            是否保存成功
        """
        try:
            # 创建samples表（如果不存在）
            self._create_samples_table()
            
            total_saved = 0
            
            with self.db_manager.get_conn() as conn:
                for period, df in samples.items():
                    if df.empty:
                        continue
                        
                    # 准备数据
                    records = []
                    for _, row in df.iterrows():
                        # 提取特征列（排除非特征列）
                        exclude_cols = ['symbol', 'date', 'close', 'open', 'high', 'low', 'volume'] + \
                                     [col for col in df.columns if col.startswith(('forward_return_', 'label_'))]
                        
                        feature_cols = [col for col in df.columns if col not in exclude_cols]
                        features = {col: row[col] for col in feature_cols if pd.notna(row[col])}
                        
                        record = {
                            'symbol': symbol,
                            'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                            'period': period,
                            'label': int(row[f'label_{period}']),
                            'forward_return': float(row[f'forward_return_{period}']),
                            'features': json.dumps(features),  # JSON字符串存储
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        records.append(record)
                    
                    # 批量插入
                    if records:
                        conn.executemany("""
                            INSERT OR REPLACE INTO samples 
                            (symbol, date, period, label, forward_return, features, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [(r['symbol'], r['date'], r['period'], r['label'], 
                               r['forward_return'], r['features'], r['created_at']) for r in records])
                        
                        total_saved += len(records)
                        logger.info(f"保存 {period} 样本: {len(records)} 条")
                
                conn.commit()
                
            logger.info(f"样本保存完成: {total_saved} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"保存样本失败: {e}")
            return False
    
    def _create_samples_table(self):
        """
        创建样本表
        """
        try:
            with self.db_manager.get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        period TEXT NOT NULL,
                        label INTEGER NOT NULL,
                        forward_return REAL NOT NULL,
                        features TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(symbol, date, period)
                    )
                """)
                
                # 创建索引
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_samples_symbol_date 
                    ON samples(symbol, date)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_samples_period 
                    ON samples(period)
                """)
                
                conn.commit()
                logger.info("样本表创建完成")
                
        except Exception as e:
            logger.error(f"创建样本表失败: {e}")
            raise

def main():
    """测试标签生成功能"""
    try:
        # 初始化
        db_manager = DatabaseManager()
        label_gen = LabelGenerator(db_manager)
        
        # 测试股票
        test_symbol = '600519.SS'
        
        logger.info(f"开始为 {test_symbol} 生成标签和样本")
        
        # 创建样本
        samples = label_gen.create_samples(
            symbol=test_symbol,
            periods=[5, 10, 20, 30],
            threshold_method='quantile',
            threshold_value=0.7
        )
        
        if samples:
            logger.info(f"样本生成完成:")
            for period, df in samples.items():
                if not df.empty:
                    positive_ratio = df[f'label_{period}'].mean()
                    logger.info(f"  {period}: {len(df)} 条样本, 正样本比例: {positive_ratio:.3f}")
                    
                    # 显示前几条样本
                    print(f"\n{period} 样本预览:")
                    feature_cols = [col for col in df.columns if col.startswith(('trend_', 'momentum_', 'volatility_', 'volume_'))]
                    preview_cols = ['date', f'label_{period}', f'forward_return_{period}'] + feature_cols[:5]
                    print(df[preview_cols].head())
            
            # 保存到数据库
            success = label_gen.save_samples_to_db(test_symbol, samples)
            if success:
                logger.info("样本已保存到数据库")
            else:
                logger.error("样本保存失败")
        else:
            logger.warning("未生成任何样本")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

if __name__ == '__main__':
    main()