#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块
计算技术指标和价量特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    特征生成器
    """
    
    def __init__(self):
        pass
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        if df.empty or len(df) < 20:
            return pd.DataFrame()
            
        try:
            # 确保数据格式正确
            df = df.copy()
            df = df.sort_values('date' if 'date' in df.columns else df.index)
            
            # 提取OHLCV数据 - 支持大小写列名
            high = df['High'].astype(float) if 'High' in df.columns else df['high'].astype(float)
            low = df['Low'].astype(float) if 'Low' in df.columns else df['low'].astype(float)
            close = df['Close'].astype(float) if 'Close' in df.columns else df['close'].astype(float)
            volume = df['Volume'].astype(float) if 'Volume' in df.columns else df['volume'].astype(float)
            open_price = df['Open'].astype(float) if 'Open' in df.columns else df['open'].astype(float)
            
            features = pd.DataFrame(index=df.index)
            
            # 价格特征
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_change'] = close - close.shift(1)
            features['price_change_pct'] = (close - close.shift(1)) / close.shift(1)
            
            # 移动平均线
            features['sma_5'] = close.rolling(window=5).mean()
            features['sma_10'] = close.rolling(window=10).mean()
            features['sma_20'] = close.rolling(window=20).mean()
            # 新增中长期均线
            features['sma_60'] = close.rolling(window=60).mean()
            features['sma_120'] = close.rolling(window=120).mean()
            
            features['ema_12'] = close.ewm(span=12).mean()
            features['ema_26'] = close.ewm(span=26).mean()
            
            # 移动平均线相对位置
            features['price_vs_sma5'] = close / features['sma_5'] - 1
            features['price_vs_sma10'] = close / features['sma_10'] - 1
            features['price_vs_sma20'] = close / features['sma_20'] - 1
            # 新增中长期相对位置
            features['price_vs_sma60'] = close / features['sma_60'] - 1
            features['price_vs_sma120'] = close / features['sma_120'] - 1
            # 新增价格在区间内的位置（与feature_engineering一致）
            high_roll_60 = high.rolling(60).max()
            low_roll_60 = low.rolling(60).min()
            features['price_position_60'] = (close - low_roll_60) / (high_roll_60 - low_roll_60)
            high_roll_120 = high.rolling(120).max()
            low_roll_120 = low.rolling(120).min()
            features['price_position_120'] = (close - low_roll_120) / (high_roll_120 - low_roll_120)
            
            # 波动率特征
            features['volatility_5'] = close.rolling(window=5).std()
            features['volatility_10'] = close.rolling(window=10).std()
            features['volatility_20'] = close.rolling(window=20).std()
            # 新增中长期波动率
            features['volatility_60'] = close.rolling(window=60).std()
            features['volatility_120'] = close.rolling(window=120).std()
            
            # RSI
            try:
                features['rsi_14'] = self._calculate_rsi(close, 14)
            except:
                features['rsi_14'] = 50
                
            # MACD
            try:
                macd_line = features['ema_12'] - features['ema_26']
                features['macd'] = macd_line
                features['macd_signal'] = macd_line.ewm(span=9).mean()
                features['macd_histogram'] = macd_line - features['macd_signal']
            except:
                features['macd'] = 0
                features['macd_signal'] = 0
                features['macd_histogram'] = 0
            
            # 布林带
            try:
                bb_middle = close.rolling(window=20).mean()
                bb_std = close.rolling(window=20).std()
                features['bb_upper'] = bb_middle + (bb_std * 2)
                features['bb_lower'] = bb_middle - (bb_std * 2)
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
                features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            except:
                features['bb_upper'] = close
                features['bb_lower'] = close
                features['bb_width'] = 0
                features['bb_position'] = 0.5
            
            # 成交量特征
            features['volume_sma_5'] = volume.rolling(window=5).mean()
            features['volume_sma_10'] = volume.rolling(window=10).mean()
            # 新增中期成交量均线
            features['volume_sma_20'] = volume.rolling(window=20).mean()
            features['volume_ratio'] = volume / features['volume_sma_10']
            # 新增60日与6个月（120日）成交量窗口特征
            features['volume_sma_60'] = volume.rolling(window=60).mean()
            features['volume_ratio_60'] = volume / features['volume_sma_60']
            features['volume_sma_120'] = volume.rolling(window=120).mean()
            features['volume_ratio_120'] = volume / features['volume_sma_120']
            
            # 价量关系
            features['price_volume_trend'] = ((close - close.shift(1)) / close.shift(1)) * volume
            
            # 高低点特征
            features['high_low_ratio'] = high / low
            features['close_high_ratio'] = close / high
            features['close_low_ratio'] = close / low
            
            # 缺口特征
            features['gap'] = (open_price - close.shift(1)) / close.shift(1)
            features['gap_filled'] = ((low <= close.shift(1)) & (features['gap'] > 0)) | ((high >= close.shift(1)) & (features['gap'] < 0))
            
            # 动量指标
            features['momentum_5'] = close / close.shift(5) - 1
            features['momentum_10'] = close / close.shift(10) - 1
            features['momentum_20'] = close / close.shift(20) - 1
            # 新增中长期动量
            features['momentum_60'] = close / close.shift(60) - 1
            features['momentum_120'] = close / close.shift(120) - 1
            
            # 威廉指标
            try:
                highest_high = high.rolling(window=14).max()
                lowest_low = low.rolling(window=14).min()
                features['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low)
            except:
                features['williams_r'] = -50
            
            # 随机指标
            try:
                lowest_low_9 = low.rolling(window=9).min()
                highest_high_9 = high.rolling(window=9).max()
                k_percent = 100 * (close - lowest_low_9) / (highest_high_9 - lowest_low_9)
                features['stoch_k'] = k_percent.rolling(window=3).mean()
                features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
            except:
                features['stoch_k'] = 50
                features['stoch_d'] = 50
            
            # 填充缺失值
            features = features.ffill().fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"特征生成失败: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI指标
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_factor_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算单个时点的因子特征
        
        Args:
            df: 历史价格数据
            
        Returns:
            因子特征字典
        """
        if df.empty or len(df) < 20:
            return {}
            
        try:
            features = self.generate_features(df)
            if features.empty:
                return {}
                
            # 取最新一行的特征值
            latest_features = features.iloc[-1]
            
            # 转换为字典格式
            factor_dict = {}
            for col in features.columns:
                value = latest_features[col]
                if pd.isna(value) or np.isinf(value):
                    factor_dict[col] = 0.0
                else:
                    factor_dict[col] = float(value)
                    
            return factor_dict
            
        except Exception as e:
            logger.error(f"因子特征计算失败: {e}")
            return {}


if __name__ == "__main__":
    # 测试特征生成
    from db import DatabaseManager
    
    db = DatabaseManager()
    symbols = db.list_symbols()
    
    if symbols:
        symbol = symbols[0]['symbol']
        prices = db.get_daily_prices(symbol, '2024-01-01', '2024-12-31')
        
        if not prices.empty:
            fg = FeatureGenerator()
            features = fg.generate_features(prices)
            print(f"生成特征数量: {len(features.columns)}")
            print(f"特征列名: {list(features.columns)}")
            print(f"最新特征值:")
            print(features.iloc[-1].to_dict())
        else:
            print("没有价格数据")
    else:
        print("没有股票数据")