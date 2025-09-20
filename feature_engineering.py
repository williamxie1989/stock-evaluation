#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程与多因子计算模块

基于signal_generator的因子计算逻辑，扩展为面向机器学习的特征矩阵生成。
支持多种技术指标、价量因子、风险因子的计算与标准化。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from signal_generator import SignalGenerator
from db import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    特征工程类，负责从原始价格数据生成机器学习特征
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.signal_gen = SignalGenerator()
        
        # 特征配置
        self.feature_config = {
            'trend_windows': [5, 10, 20, 60, 120],
            'volatility_windows': [20, 60, 120],
            'volume_windows': [5, 20, 60, 120],
            # 技术指标配置（新增，避免KeyError）
            'rsi_windows': [6, 14],
            'bb_windows': [20],
            # (fast, slow, signal)
            'macd_params': [(12, 26, 9)]
        }
    
    def calculate_features(self, symbol: str, start_date: str = None, 
                          end_date: str = None, min_periods: int = 30) -> pd.DataFrame:
        """
        计算单个股票的所有特征
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            min_periods: 最小数据量要求
            
        Returns:
            包含所有特征的DataFrame，索引为日期
        """
        try:
            # 获取价格数据
            bars = self._get_price_data(symbol, start_date, end_date, min_periods)
            if bars is None or len(bars) < min_periods:
                logger.warning(f"数据不足: {symbol}, 需要{min_periods}条，实际{len(bars) if bars is not None else 0}条")
                return pd.DataFrame()
            
            # 计算基础因子（复用signal_generator）
            factors_dict = self.signal_gen.calculate_factors(bars)
            
            # 将因子字典转换为DataFrame
            factors_df = pd.DataFrame(factors_dict, index=bars.index)
            
            # 计算扩展特征
            features = self._calculate_extended_features(bars, factors_df)
            
            # 添加元信息
            features['symbol'] = symbol
            features['date'] = features.index
            
            logger.info(f"完成特征计算: {symbol}, 特征数: {len(features.columns)}, 样本数: {len(features)}")
            return features
            
        except Exception as e:
            logger.error(f"特征计算失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_price_data(self, symbol: str, start_date: str = None, 
                       end_date: str = None, min_periods: int = 30) -> Optional[pd.DataFrame]:
        """
        获取价格数据并预处理
        """
        try:
            # 计算实际需要的数据量（考虑指标计算窗口）
            max_window = max(
                max(self.feature_config['trend_windows']),
                max(self.feature_config['volatility_windows']),
                max(self.feature_config['volume_windows']),
                60  # 额外缓冲
            )
            
            # 获取足够的历史数据
            required_bars = min_periods + max_window * 2
            
            # 从数据库获取数据（使用现有的get_last_n_bars方法）
            bars = self.db.get_last_n_bars([symbol], n=required_bars)
            
            if bars is None or len(bars) == 0:
                return None
            
            # 过滤指定symbol的数据
            bars = bars[bars['symbol'] == symbol].copy()
            
            if len(bars) == 0:
                return None
            
            # 重命名列以匹配signal_generator期望的大写格式
            bars = bars.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # 设置日期索引
            bars['date'] = pd.to_datetime(bars['date'])
            bars = bars.set_index('date').sort_index()
            
            # 过滤到指定日期范围
            if start_date:
                bars = bars[bars.index >= start_date]
            if end_date:
                bars = bars[bars.index <= end_date]
            
            # 确保有足够的数据
            if len(bars) < min_periods:
                logger.warning(f"数据不足: {symbol}, 需要{min_periods}条，实际{len(bars)}条")
                return None
            
            return bars
            
        except Exception as e:
            logger.error(f"获取价格数据失败 {symbol}: {e}")
            return None
    
    def _calculate_extended_features(self, bars: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算扩展特征
        """
        features = factors.copy()
        
        # 价格相关特征
        features.update(self._calculate_price_features(bars))
        
        # 成交量特征
        features.update(self._calculate_volume_features(bars))
        
        # 波动率特征
        features.update(self._calculate_volatility_features(bars))
        
        # 技术指标特征
        features.update(self._calculate_technical_features(bars))
        
        # 相对强弱特征
        features.update(self._calculate_relative_features(bars))
        
        # 时间特征
        features.update(self._calculate_time_features(bars))
        
        return features
    
    def _calculate_price_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算价格相关特征
        """
        features = {}
        
        # 收益率特征
        for window in self.feature_config['trend_windows']:
            features[f'return_{window}d'] = bars['Close'].pct_change(window)
            features[f'log_return_{window}d'] = np.log(bars['Close'] / bars['Close'].shift(window))
        
        # 价格位置特征
        for window in [20, 60, 120]:
            high_roll = bars['High'].rolling(window).max()
            low_roll = bars['Low'].rolling(window).min()
            features[f'price_position_{window}d'] = (bars['Close'] - low_roll) / (high_roll - low_roll)
        
        # 价格差值特征
        features['hl_ratio'] = (bars['High'] - bars['Low']) / bars['Close']
        features['oc_ratio'] = (bars['Close'] - bars['Open']) / bars['Open']
        features['gap'] = (bars['Open'] - bars['Close'].shift(1)) / bars['Close'].shift(1)
        
        return features
    
    def _calculate_volume_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算成交量特征
        """
        features = {}
        
        # 成交量变化率
        for window in self.feature_config['volume_windows']:
            features[f'volume_change_{window}d'] = bars['Volume'].pct_change(window)
            features[f'volume_ma_ratio_{window}d'] = bars['Volume'] / bars['Volume'].rolling(window).mean()
        
        # 价量关系
        features['price_volume_corr_20d'] = bars['Close'].rolling(20).corr(bars['Volume'])
        features['vwap_ratio'] = bars['Close'] / ((bars['Close'] * bars['Volume']).rolling(20).sum() / bars['Volume'].rolling(20).sum())
        
        # 成交量强度
        features['volume_intensity'] = bars['Volume'] / bars['Volume'].rolling(60).mean()
        
        return features
    
    def _calculate_volatility_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算波动率特征
        """
        features = {}
        
        # 历史波动率
        returns = bars['Close'].pct_change()
        for window in self.feature_config['volatility_windows']:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            features[f'volatility_rank_{window}d'] = returns.rolling(window).std().rolling(252).rank(pct=True)
        
        # 真实波动率
        tr1 = bars['High'] - bars['Low']
        tr2 = abs(bars['High'] - bars['Close'].shift(1))
        tr3 = abs(bars['Low'] - bars['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_20d'] = tr.rolling(20).mean() / bars['Close']
        
        # 波动率偏度和峰度
        features['return_skew_20d'] = returns.rolling(20).skew()
        features['return_kurt_20d'] = returns.rolling(20).kurt()
        
        return features
    
    def _calculate_technical_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算技术指标特征
        """
        features = {}
        
        # RSI
        for window in self.feature_config['rsi_windows']:
            delta = bars['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            features[f'rsi_{window}d'] = 100 - (100 / (1 + rs))
        
        # 布林带
        for window in self.feature_config['bb_windows']:
            ma = bars['Close'].rolling(window).mean()
            std = bars['Close'].rolling(window).std()
            features[f'bb_upper_{window}d'] = (ma + 2 * std - bars['Close']) / bars['Close']
            features[f'bb_lower_{window}d'] = (bars['Close'] - (ma - 2 * std)) / bars['Close']
            features[f'bb_width_{window}d'] = (4 * std) / ma
        
        # MACD
        for fast, slow, signal in self.feature_config['macd_params']:
            ema_fast = bars['Close'].ewm(span=fast).mean()
            ema_slow = bars['Close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            features[f'macd_{fast}_{slow}'] = macd_line / bars['Close']
            features[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line / bars['Close']
            features[f'macd_hist_{fast}_{slow}_{signal}'] = (macd_line - signal_line) / bars['Close']
        
        return features
    
    def _calculate_relative_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算相对强弱特征
        """
        features = {}
        
        # 相对于自身历史的强弱
        for window in [60, 120, 252]:
            features[f'price_rank_{window}d'] = bars['Close'].rolling(window).rank(pct=True)
            features[f'volume_rank_{window}d'] = bars['Volume'].rolling(window).rank(pct=True)
        
        return features
    
    def _calculate_time_features(self, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算时间特征
        """
        features = {}
        
        # 周期性特征
        dates = pd.to_datetime(bars.index)
        features['day_of_week'] = dates.dayofweek
        features['day_of_month'] = dates.day
        features['month'] = dates.month
        features['quarter'] = dates.quarter
        
        # 距离重要时点的天数
        features['days_to_month_end'] = dates.to_series().apply(lambda x: (x + pd.offsets.MonthEnd(0) - x).days)
        features['days_to_quarter_end'] = dates.to_series().apply(lambda x: (x + pd.offsets.QuarterEnd(0) - x).days)
        
        return features
    
    def calculate_batch_features(self, symbols: List[str], start_date: str = None, 
                                end_date: str = None, min_periods: int = 30) -> pd.DataFrame:
        """
        批量计算多个股票的特征
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            min_periods: 最小数据量要求
            
        Returns:
            包含所有股票特征的DataFrame，包含symbol和date列
        """
        all_features = []
        
        for symbol in symbols:
            logger.info(f"计算特征: {symbol}")
            features = self.calculate_features(symbol, start_date, end_date, min_periods)
            
            if not features.empty:
                all_features.append(features)
            else:
                logger.warning(f"跳过 {symbol}: 特征计算失败或数据不足")
        
        if not all_features:
            logger.warning("没有成功计算任何特征")
            return pd.DataFrame()
        
        # 合并所有特征
        result = pd.concat(all_features, ignore_index=True)
        
        # 重新排序列
        cols = ['symbol', 'date'] + [col for col in result.columns if col not in ['symbol', 'date']]
        result = result[cols]
        
        logger.info(f"批量特征计算完成: {len(symbols)}个股票, {len(result)}条记录, {len(result.columns)}个特征")
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名称
        """
        # 这里返回一个示例特征列表，实际使用时应该基于配置动态生成
        feature_names = []
        
        # 基础因子（来自signal_generator）
        base_factors = [
            'ma_5', 'ma_10', 'ma_20', 'ma_60',
            'ema_12', 'ema_26',
            'momentum_5', 'momentum_14', 'momentum_20',
            'volatility_10', 'volatility_20', 'volatility_60',
            'volume_ma_5', 'volume_ma_20', 'volume_ma_60',
            'rsi_14', 'bb_upper', 'bb_lower', 'bb_width',
            'macd', 'macd_signal', 'macd_hist'
        ]
        feature_names.extend(base_factors)
        
        # 扩展特征
        for window in self.feature_config['trend_windows']:
            feature_names.extend([f'return_{window}d', f'log_return_{window}d'])
        
        for window in [20, 60, 120]:
            feature_names.append(f'price_position_{window}d')
        
        feature_names.extend([
            'hl_ratio', 'oc_ratio', 'gap',
            'price_volume_corr_20d', 'vwap_ratio', 'volume_intensity',
            'atr_20d', 'return_skew_20d', 'return_kurt_20d'
        ])
        
        return feature_names
    
    def save_features_to_db(self, features: pd.DataFrame) -> bool:
        """
        将特征保存到数据库的factors表
        
        Args:
            features: 特征DataFrame
            
        Returns:
            是否保存成功
        """
        try:
            if features.empty:
                logger.warning("特征数据为空，跳过保存")
                return False
            
            # 准备数据格式
            feature_records = []
            
            for _, row in features.iterrows():
                symbol = row['symbol']
                date = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                
                # 遍历所有特征列（排除symbol和date）
                for col in features.columns:
                    if col not in ['symbol', 'date'] and pd.notna(row[col]):
                        feature_records.append({
                            'symbol': symbol,
                            'date': date,
                            'factor_name': col,
                            'value': float(row[col])
                        })
            
            if not feature_records:
                logger.warning("没有有效的特征记录可保存")
                return False
            
            # 批量保存到数据库（使用现有的factors表结构）
            with self.db.get_conn() as conn:
                cur = conn.cursor()
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO factors(symbol, date, factor_name, value)
                    VALUES(?, ?, ?, ?)
                    """,
                    [(r['symbol'], r['date'], r['factor_name'], r['value']) for r in feature_records]
                )
                conn.commit()
            
            logger.info(f"特征保存完成: {len(feature_records)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"保存特征到数据库失败: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    from db import DatabaseManager
    
    db = DatabaseManager()
    fe = FeatureEngineer(db)
    
    # 测试单个股票特征计算
    symbol = "600519.SS"
    features = fe.calculate_features(symbol, min_periods=60)
    
    if not features.empty:
        print(f"\n{symbol} 特征计算结果:")
        print(f"特征数量: {len(features.columns)}")
        print(f"样本数量: {len(features)}")
        print(f"特征名称: {list(features.columns)[:10]}...")  # 显示前10个特征名
        print(f"\n最新5条记录:")
        print(features.tail().round(4))
        
        # 测试保存到数据库
        if fe.save_features_to_db(features):
            print("\n特征已保存到数据库")
    else:
        print(f"\n{symbol} 特征计算失败")