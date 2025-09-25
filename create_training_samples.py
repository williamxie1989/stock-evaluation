#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练样本生成器
从股票历史数据生成机器学习训练样本
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from typing import List, Dict, Tuple
from db import DatabaseManager
from enhanced_features import EnhancedFeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingSampleGenerator:
    """训练样本生成器"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.feature_generator = EnhancedFeatureGenerator()
        
    def create_samples_table(self):
        """创建样本表"""
        try:
            # 使用stocks.db数据库
            import sqlite3
            conn = sqlite3.connect('stocks.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    period TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    forward_return REAL NOT NULL,
                    features TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, period)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_symbol ON samples(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_date ON samples(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_period ON samples(period)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_symbol_date ON samples(symbol, date)")
            
            conn.commit()
            conn.close()
            logger.info("样本表创建成功")
            
        except Exception as e:
            logger.error(f"创建样本表失败: {e}")
            raise
    
    def save_samples_to_db(self, samples_df: pd.DataFrame):
        """保存样本到数据库"""
        if samples_df.empty:
            logger.warning("尝试保存空样本数据")
            return
            
        try:
            logger.info(f"开始保存 {len(samples_df)} 个样本到数据库")
            
            # 使用sqlite3直接连接，避免db_manager的问题
            import sqlite3
            conn = sqlite3.connect('stocks.db')
            cursor = conn.cursor()
            
            success_count = 0
            error_count = 0
            
            logger.debug(f"样本数据类型检查:")
            for col in samples_df.columns:
                logger.debug(f"  {col}: {samples_df[col].dtype}")
                if len(samples_df) > 0:
                    logger.debug(f"  示例值: {samples_df[col].iloc[0]} (类型: {type(samples_df[col].iloc[0])})")
            
            for i, (_, row) in enumerate(samples_df.iterrows()):
                try:
                    # 确保features字段是字符串类型
                    features_data = row['features']
                    if isinstance(features_data, bytes):
                        try:
                            features_data = features_data.decode('utf-8')
                        except UnicodeDecodeError:
                            features_data = features_data.decode('latin-1')
                    elif not isinstance(features_data, str):
                        features_data = str(features_data)
                    
                    logger.debug(f"处理第 {i+1} 行: symbol={row['symbol']}, date={row['date']}, features类型={type(features_data)}")
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO samples 
                        (symbol, date, period, label, forward_return, features)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], row['date'], row['period'], 
                        row['label'], row['forward_return'], features_data
                    ))
                    success_count += 1
                    
                    if i % 100 == 0:  # 每100条记录记录一次进度
                        logger.debug(f"已处理 {i+1} 行")
                        
                except Exception as e:
                    logger.error(f"插入第 {i+1} 行失败: {e}")
                    logger.error(f"失败数据: {row.to_dict()}")
                    error_count += 1
                    continue
            
            logger.info(f"准备提交事务: 成功 {success_count}, 失败 {error_count}")
            conn.commit()
            logger.info(f"保存完成: 成功 {success_count}, 失败 {error_count}")
            
            # 验证保存结果
            cursor.execute("SELECT COUNT(*) FROM samples")
            total_count = cursor.fetchone()[0]
            logger.info(f"数据库中样本总数: {total_count}")
            
            conn.close()
            
            if error_count > 0:
                logger.warning(f"保存过程中有 {error_count} 个样本失败")
            
        except Exception as e:
            logger.error(f"保存样本失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_stock_list(self, limit: int = None) -> List[str]:
        """获取股票列表"""
        try:
            # 使用stock_data.sqlite3数据库
            import sqlite3
            conn = sqlite3.connect('stock_data.sqlite3')
            
            query = "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol"
            if limit:
                query += f" LIMIT {limit}"
                
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            symbols = df['symbol'].tolist()
            logger.info(f"获取到 {len(symbols)} 只股票")
            return symbols
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            conn = sqlite3.connect('stock_data.sqlite3')
            query = """
                SELECT symbol, date, open, high, low, close, volume, adj_close
                FROM stock_prices 
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 转换数据类型为float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除包含NaN的行
            df = df.dropna()
            
            # 计算涨跌幅
            df['change_pct'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_forward_return(self, df: pd.DataFrame, periods: List[int]) -> Dict[int, pd.Series]:
        """计算未来收益率"""
        forward_returns = {}
        
        # 检查DataFrame中是否包含close列，如果不包含则使用原始数据中的close
        if 'close' not in df.columns:
            logger.error(f"DataFrame中缺少close列，可用列: {df.columns.tolist()}")
            return forward_returns
            
        for period in periods:
            try:
                # 计算未来period天的收益率
                future_close = df['close'].shift(-period)
                forward_return = (future_close - df['close']) / df['close'] * 100
                forward_returns[period] = forward_return
            except Exception as e:
                logger.error(f"计算{period}天前向收益失败: {e}")
                continue
            
        return forward_returns
    
    def create_label(self, forward_return: pd.Series, threshold: float = 1.0) -> pd.Series:
        """创建分类标签"""
        # 未来收益率大于阈值为1（上涨），否则为0（下跌或持平）
        return (forward_return > threshold).astype(int)
    
    def generate_samples_for_stock(self, symbol: str, start_date: str, end_date: str, 
                                   periods: List[int] = [10, 30]) -> pd.DataFrame:
        """为单只股票生成训练样本"""
        
        try:
            # 获取股票数据
            df = self.get_stock_data(symbol, start_date, end_date)
            if df.empty or len(df) < 60:  # 至少需要60天的数据
                return pd.DataFrame()
            
            # 保存原始价格列用于后续计算
            original_close = df['close'].copy()
            
            # 计算技术指标 - 使用generate_features方法
            df_with_features = self.feature_generator.generate_features(df.copy())
            if df_with_features.empty:
                logger.warning(f"特征生成失败 {symbol}: 生成的特征为空")
                return pd.DataFrame()
            
            # 如果特征生成后没有close列，添加回去
            if 'close' not in df_with_features.columns:
                df_with_features['close'] = original_close
            
            # 计算未来收益率
            forward_returns = self.calculate_forward_return(df_with_features, periods)
            if not forward_returns:  # 如果前向收益计算失败
                logger.warning(f"前向收益计算失败 {symbol}")
                return pd.DataFrame()
            
            samples = []
            
            for period in periods:
                # 创建标签
                labels = self.create_label(forward_returns[period])
                
                # 生成样本（排除最近period天，因为无法计算未来收益）
                valid_data = df_with_features.iloc[:-period]
                valid_labels = labels.iloc[:-period]
                valid_returns = forward_returns[period].iloc[:-period]
                
                for idx in valid_data.index:
                    # 获取特征数据
                    feature_data = valid_data.loc[idx]
                    
                    # 创建样本
                    sample = {
                        'symbol': symbol,
                        'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'period': f"{period}d",
                        'label': int(valid_labels.loc[idx]),
                        'forward_return': float(valid_returns.loc[idx]),
                        'features': json.dumps(feature_data.to_dict())
                    }
                    
                    samples.append(sample)
            
            if samples:
                result_df = pd.DataFrame(samples)
                logger.info(f"为 {symbol} 生成 {len(result_df)} 个样本")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"生成样本失败 {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_samples_to_db(self, samples_df: pd.DataFrame):
        """保存样本到数据库"""
        if samples_df.empty:
            logger.warning("尝试保存空样本数据")
            return
            
        try:
            logger.info(f"开始保存 {len(samples_df)} 个样本到数据库")
            
            # 使用sqlite3直接连接，避免db_manager的问题
            import sqlite3
            conn = sqlite3.connect('stocks.db')
            cursor = conn.cursor()
            
            success_count = 0
            error_count = 0
            
            logger.debug(f"样本数据类型检查:")
            for col in samples_df.columns:
                logger.debug(f"  {col}: {samples_df[col].dtype}")
                if len(samples_df) > 0:
                    logger.debug(f"  示例值: {samples_df[col].iloc[0]} (类型: {type(samples_df[col].iloc[0])})")
            
            for i, (_, row) in enumerate(samples_df.iterrows()):
                try:
                    # 确保features字段是字符串类型
                    features_data = row['features']
                    if isinstance(features_data, bytes):
                        try:
                            features_data = features_data.decode('utf-8')
                        except UnicodeDecodeError:
                            features_data = features_data.decode('latin-1')
                    elif not isinstance(features_data, str):
                        features_data = str(features_data)
                    
                    logger.debug(f"处理第 {i+1} 行: symbol={row['symbol']}, date={row['date']}, features类型={type(features_data)}")
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO samples 
                        (symbol, date, period, label, forward_return, features)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], row['date'], row['period'], 
                        row['label'], row['forward_return'], features_data
                    ))
                    success_count += 1
                    
                    if i % 100 == 0:  # 每100条记录记录一次进度
                        logger.debug(f"已处理 {i+1} 行")
                        
                except Exception as e:
                    logger.error(f"插入第 {i+1} 行失败: {e}")
                    logger.error(f"失败数据: {row.to_dict()}")
                    error_count += 1
                    continue
            
            logger.info(f"准备提交事务: 成功 {success_count}, 失败 {error_count}")
            conn.commit()
            logger.info(f"保存完成: 成功 {success_count}, 失败 {error_count}")
            
            # 验证保存结果
            cursor.execute("SELECT COUNT(*) FROM samples")
            total_count = cursor.fetchone()[0]
            logger.info(f"数据库中样本总数: {total_count}")
            
            conn.close()
            
            if error_count > 0:
                logger.warning(f"保存过程中有 {error_count} 个样本失败")
            
        except Exception as e:
            logger.error(f"保存样本失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_samples(self, symbols: List[str] = None, start_date: str = None, 
                        end_date: str = None, periods: List[int] = [10, 30], 
                        batch_size: int = 50):
        """批量生成训练样本"""
        
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
        # 获取股票列表
        if not symbols:
            symbols = self.get_stock_list(limit=100)  # 限制100只股票用于测试
            
        logger.info(f"开始生成样本: {len(symbols)} 只股票, {start_date} 到 {end_date}")
        
        # 创建样本表
        self.create_samples_table()
        
        # 分批处理
        total_samples = 0
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {len(batch_symbols)} 只股票")
            
            batch_samples = pd.DataFrame()
            
            for symbol in batch_symbols:
                stock_samples = self.generate_samples_for_stock(
                    symbol, start_date, end_date, periods
                )
                if not stock_samples.empty:
                    batch_samples = pd.concat([batch_samples, stock_samples], ignore_index=True)
                    logger.info(f"已收集 {len(batch_samples)} 个样本，准备保存...")
                
                # 每处理10只股票就保存一次
                if len(batch_samples) >= 1000:
                    logger.info(f"保存批次样本: {len(batch_samples)} 个")
                    self.save_samples_to_db(batch_samples)
                    total_samples += len(batch_samples)
                    batch_samples = pd.DataFrame()
            
            # 保存剩余样本
            if not batch_samples.empty:
                logger.info(f"保存剩余批次样本: {len(batch_samples)} 个")
                self.save_samples_to_db(batch_samples)
                total_samples += len(batch_samples)
        
        logger.info(f"样本生成完成! 总计生成 {total_samples} 个样本")
        return total_samples
    
    def get_sample_statistics(self) -> Dict:
        """获取样本统计信息"""
        try:
            # 使用stocks.db数据库
            import sqlite3
            conn = sqlite3.connect('stocks.db')
            
            # 基本统计
            stats_query = """
                SELECT 
                    COUNT(*) as total_samples,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT period) as unique_periods,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM samples
            """
            stats_df = pd.read_sql_query(stats_query, conn)
            stats = stats_df.iloc[0].to_dict()
            
            # 按周期统计
            period_stats_query = """
                SELECT 
                    period,
                    COUNT(*) as count,
                    AVG(label) as positive_rate,
                    AVG(forward_return) as avg_return
                FROM samples
                GROUP BY period
            """
            period_stats_df = pd.read_sql_query(period_stats_query, conn)
            period_stats = period_stats_df.to_dict('records')
            
            conn.close()
            
            stats['period_stats'] = period_stats
            return stats
            
        except Exception as e:
            logger.error(f"获取样本统计失败: {e}")
            return {}

if __name__ == "__main__":
    # 创建样本生成器
    generator = TrainingSampleGenerator()
    
    # 生成样本
    print("开始生成训练样本...")
    total_samples = generator.generate_samples(
        symbols=None,  # 使用默认股票列表
        start_date='2022-01-01',
        end_date='2024-06-01',
        periods=[10, 30],
        batch_size=20  # 减少批次大小用于测试
    )
    
    # 获取统计信息
    stats = generator.get_sample_statistics()
    print(f"\n样本统计信息:")
    print(f"总样本数: {stats.get('total_samples', 0)}")
    print(f"股票数量: {stats.get('unique_symbols', 0)}")
    print(f"周期数量: {stats.get('unique_periods', 0)}")
    print(f"日期范围: {stats.get('min_date', 'N/A')} 到 {stats.get('max_date', 'N/A')}")
    
    if 'period_stats' in stats:
        print("\n按周期统计:")
        for period_stat in stats['period_stats']:
            print(f"周期 {period_stat['period']}: {period_stat['count']} 样本, "
                  f"正例率: {period_stat['positive_rate']:.2%}, "
                  f"平均收益: {period_stat['avg_return']:.2f}%")