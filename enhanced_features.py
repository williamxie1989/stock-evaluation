#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强特征工程模块
在原有特征基础上添加更多有效的技术指标和市场微观结构特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from features import FeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureGenerator(FeatureGenerator):
    """
    增强特征生成器
    继承原有特征生成器，添加更多高级特征
    """
    
    def __init__(self):
        super().__init__()
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成增强技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含增强特征的DataFrame
        """
        if df.empty or len(df) < 30:
            return pd.DataFrame()
            
        try:
            # 确保数据格式正确
            df = df.copy()
            
            # 检查索引类型，如果不是日期索引则使用默认索引
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                else:
                    # 创建默认索引
                    df.index = pd.RangeIndex(len(df))
            
            # 提取OHLCV数据
            high = df['High'].astype(float) if 'High' in df.columns else df['high'].astype(float)
            low = df['Low'].astype(float) if 'Low' in df.columns else df['low'].astype(float)
            close = df['Close'].astype(float) if 'Close' in df.columns else df['close'].astype(float)
            volume = df['Volume'].astype(float) if 'Volume' in df.columns else df['volume'].astype(float)
            open_price = df['Open'].astype(float) if 'Open' in df.columns else df['open'].astype(float)
            
            # 先生成基础特征
            features = super().generate_features(df)
            if features.empty:
                return pd.DataFrame()
            
            # === 新增高级技术指标 ===
            
            # 1. 多时间框架趋势强度
            features['trend_strength_5'] = self._calculate_trend_strength(close, 5)
            features['trend_strength_10'] = self._calculate_trend_strength(close, 10)
            features['trend_strength_20'] = self._calculate_trend_strength(close, 20)
            
            # 2. 价格加速度（二阶导数）
            features['price_acceleration'] = close.diff().diff()
            features['price_acceleration_norm'] = features['price_acceleration'] / close
            
            # 3. 波动率比率（短期vs长期）
            vol_5 = close.rolling(5).std()
            vol_20 = close.rolling(20).std()
            vol_60 = close.rolling(60).std()
            features['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-8)
            features['vol_ratio_20_60'] = vol_20 / (vol_60 + 1e-8)
            
            # 4. 相对强弱指数变化率
            rsi_14 = features['rsi_14']
            features['rsi_change'] = rsi_14.diff()
            features['rsi_momentum'] = rsi_14 - rsi_14.shift(5)
            
            # 5. 布林带挤压指标
            bb_width = features['bb_width']
            features['bb_squeeze'] = bb_width < bb_width.rolling(20).quantile(0.2)
            features['bb_expansion'] = bb_width > bb_width.rolling(20).quantile(0.8)
            
            # 6. 成交量价格趋势（VPT）
            features['vpt'] = (volume * close.pct_change()).cumsum()
            features['vpt_sma'] = features['vpt'].rolling(10).mean()
            features['vpt_signal'] = features['vpt'] - features['vpt_sma']
            
            # 7. 资金流量指标（MFI）
            features['mfi'] = self._calculate_mfi(high, low, close, volume, 14)
            
            # 8. 商品通道指数（CCI）
            features['cci'] = self._calculate_cci(high, low, close, 20)
            
            # 9. 平均真实波幅（ATR）
            features['atr'] = self._calculate_atr(high, low, close, 14)
            features['atr_ratio'] = features['atr'] / close
            
            # 10. 抛物线SAR
            features['sar'] = self._calculate_sar(high, low, close)
            features['sar_signal'] = (close > features['sar']).astype(int)
            
            # 11. 一目均衡表
            features['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
            features['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
            features['tenkan_kijun_diff'] = features['tenkan_sen'] - features['kijun_sen']
            
            # 12. 威廉指标平滑版
            features['williams_r_smooth'] = features['williams_r'].rolling(3).mean()
            features['williams_r_signal'] = (features['williams_r'] > -20).astype(int)
            
            # 13. 动量震荡器
            features['momentum_oscillator'] = close / close.shift(10) - 1
            features['momentum_sma'] = features['momentum_oscillator'].rolling(5).mean()
            
            # 14. 通道位置指标
            high_20 = high.rolling(20).max()
            low_20 = low.rolling(20).min()
            features['channel_position_20'] = (close - low_20) / (high_20 - low_20)
            
            high_60 = high.rolling(60).max()
            low_60 = low.rolling(60).min()
            features['channel_position_60'] = (close - low_60) / (high_60 - low_60)
            
            # 15. 成交量RSI
            features['volume_rsi'] = self._calculate_rsi(volume, 14)
            
            # 16. 支撑阻力强度
            features['support_strength'] = self._calculate_support_resistance(close, low.rolling(20).min(), 20, 'support')
            features['resistance_strength'] = self._calculate_support_resistance(close, high.rolling(20).max(), 20, 'resistance')
            
            # 17. 日内波动特征
            features['intraday_range'] = (high - low) / close
            features['close_position'] = (close - low) / (high - low)
            features['upper_shadow'] = (high - np.maximum(open_price, close)) / close
            features['lower_shadow'] = (np.minimum(open_price, close) - low) / close
            
            # 18. 趋势一致性
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            features['trend_alignment'] = ((ma5 > ma10) & (ma10 > ma20)).astype(int)
            features['trend_consistency'] = (close > ma5).rolling(5).sum() / 5
            
            # 19. 成交量突破
            vol_ma = volume.rolling(20).mean()
            features['vol_breakout'] = (volume > vol_ma * 1.5).astype(int)
            features['volume_breakout'] = volume / vol_ma
            
            # 20. 动量背离
            price_momentum = close / close.shift(5) - 1
            rsi_momentum = features['rsi_14'] - features['rsi_14'].shift(5)
            features['momentum_divergence'] = np.where(
                (price_momentum > 0) & (rsi_momentum < 0), -1,
                np.where((price_momentum < 0) & (rsi_momentum > 0), 1, 0)
            )
            
            # 21. 斐波那契回撤位
            period_high = high.rolling(60).max()
            period_low = low.rolling(60).min()
            fib_range = period_high - period_low
            features['fib_23.6'] = period_high - fib_range * 0.236
            features['fib_38.2'] = period_high - fib_range * 0.382
            features['fib_61.8'] = period_high - fib_range * 0.618
            
            features['price_vs_fib_23.6'] = (close - features['fib_23.6']) / close
            features['price_vs_fib_38.2'] = (close - features['fib_38.2']) / close
            features['price_vs_fib_61.8'] = (close - features['fib_61.8']) / close
            
            # === 添加模型期望的特征名称映射 ===
            # 为了与训练好的模型兼容，添加特定的特征名称
            
            # 使用pd.concat一次性添加所有特征，避免DataFrame碎片化
            additional_features = pd.DataFrame({
                # MACD相关特征
                'momentum_macd': features['macd'],
                'momentum_macd_signal': features['macd_signal'],
                
                # 形态识别特征（简化版）
                'pattern_engulfing': self._detect_engulfing_pattern(open_price, high, low, close),
                'pattern_hammer': self._detect_hammer_pattern(open_price, high, low, close),
                
                # 成交量指标
                'volume_obv': self._calculate_obv(close, volume),
                'volume_vwap': self._calculate_vwap(high, low, close, volume),
                
                # 趋势指标
                'trend_ma5': features['sma_5'],
                'trend_ma20': features['sma_20'],
                'trend_ma60': features['sma_60'],
                
                # RSI和ATR
                'momentum_rsi': features['rsi_14'],
                'volatility_atr': features['atr']
            }, index=features.index)
            
            # 使用pd.concat一次性合并所有特征
            features = pd.concat([features, additional_features], axis=1)
            
            # 填充NaN值 - 使用新的API替代弃用的method参数
            features = features.ffill().fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"特征生成失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _calculate_trend_strength(self, prices: pd.Series, period: int) -> pd.Series:
        """计算趋势强度"""
        sma = prices.rolling(period).mean()
        trend_up = (sma > sma.shift(1)).astype(int)
        trend_strength = trend_up.rolling(period).sum() / period
        return trend_strength
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series, period: int = 14) -> pd.Series:
        """计算资金流量指数"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        return mfi
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 20) -> pd.Series:
        """计算商品通道指数"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_sar(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      af_start: float = 0.02, af_increment: float = 0.02, 
                      af_max: float = 0.2) -> pd.Series:
        """计算抛物线SAR（简化版）"""
        sar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        af = pd.Series(index=close.index, dtype=float)
        ep = pd.Series(index=close.index, dtype=float)
        
        # 初始化
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1为上升趋势，-1为下降趋势
        af.iloc[0] = af_start
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(close)):
            if trend.iloc[i-1] == 1:  # 上升趋势
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af_start
                    ep.iloc[i] = low.iloc[i]
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # 下降趋势
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af_start
                    ep.iloc[i] = high.iloc[i]
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar
    
    def _calculate_support_resistance(self, close: pd.Series, extreme: pd.Series, 
                                    period: int, type_: str) -> pd.Series:
        """计算支撑阻力强度"""
        strength = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            window_close = close.iloc[i-period:i]
            window_extreme = extreme.iloc[i-period:i]
            
            current_price = close.iloc[i]
            
            if type_ == 'support':
                # 计算支撑强度：价格接近历史低点的次数
                support_levels = window_extreme[window_extreme <= current_price * 1.02]
                strength.iloc[i] = len(support_levels) / period
            else:  # resistance
                # 计算阻力强度：价格接近历史高点的次数
                resistance_levels = window_extreme[window_extreme >= current_price * 0.98]
                strength.iloc[i] = len(resistance_levels) / period
        
        return strength.fillna(0)
    
    def _detect_engulfing_pattern(self, open_price: pd.Series, high: pd.Series, 
                                 low: pd.Series, close: pd.Series) -> pd.Series:
        """检测吞没形态"""
        pattern = pd.Series(index=close.index, dtype=float)
        
        for i in range(1, len(close)):
            prev_open = open_price.iloc[i-1]
            prev_close = close.iloc[i-1]
            curr_open = open_price.iloc[i]
            curr_close = close.iloc[i]
            
            # 看涨吞没
            if (prev_close < prev_open and  # 前一根是阴线
                curr_close > curr_open and  # 当前是阳线
                curr_open < prev_close and  # 当前开盘价低于前一根收盘价
                curr_close > prev_open):    # 当前收盘价高于前一根开盘价
                pattern.iloc[i] = 1
            # 看跌吞没
            elif (prev_close > prev_open and  # 前一根是阳线
                  curr_close < curr_open and  # 当前是阴线
                  curr_open > prev_close and  # 当前开盘价高于前一根收盘价
                  curr_close < prev_open):    # 当前收盘价低于前一根开盘价
                pattern.iloc[i] = -1
            else:
                pattern.iloc[i] = 0
                
        return pattern.fillna(0)
    
    def _detect_hammer_pattern(self, open_price: pd.Series, high: pd.Series, 
                              low: pd.Series, close: pd.Series) -> pd.Series:
        """检测锤子形态"""
        pattern = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            body = abs(close.iloc[i] - open_price.iloc[i])
            upper_shadow = high.iloc[i] - max(close.iloc[i], open_price.iloc[i])
            lower_shadow = min(close.iloc[i], open_price.iloc[i]) - low.iloc[i]
            total_range = high.iloc[i] - low.iloc[i]
            
            if total_range > 0:
                # 锤子形态：下影线长，上影线短，实体小
                if (lower_shadow > body * 2 and  # 下影线至少是实体的2倍
                    upper_shadow < body * 0.5 and  # 上影线小于实体的一半
                    body > 0):  # 有实体
                    pattern.iloc[i] = 1
                # 倒锤子形态：上影线长，下影线短，实体小
                elif (upper_shadow > body * 2 and  # 上影线至少是实体的2倍
                      lower_shadow < body * 0.5 and  # 下影线小于实体的一半
                      body > 0):  # 有实体
                    pattern.iloc[i] = -1
                else:
                    pattern.iloc[i] = 0
            else:
                pattern.iloc[i] = 0
                
        return pattern.fillna(0)
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算能量潮指标（On-Balance Volume）"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def _calculate_vwap(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算成交量加权平均价格（VWAP）"""
        typical_price = (high + low + close) / 3
        vwap = pd.Series(index=close.index, dtype=float)
        
        cumulative_volume = volume.cumsum()
        cumulative_pv = (typical_price * volume).cumsum()
        
        vwap = cumulative_pv / cumulative_volume
        
        return vwap.fillna(close)


if __name__ == "__main__":
    # 测试增强特征生成器
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.02)
    volumes = np.random.randint(1000000, 10000000, 200)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(200) * 0.5,
        'high': prices + np.abs(np.random.randn(200) * 1.0),
        'low': prices - np.abs(np.random.randn(200) * 1.0),
        'close': prices,
        'volume': volumes
    })
    
    efg = EnhancedFeatureGenerator()
    features = efg.generate_features(df)
    
    print(f'增强特征数量: {len(features.columns)}')
    print(f'相比基础特征增加: {len(features.columns) - 53}个')
    print(f'新增特征示例:')
    
    # 显示新增的特征
    base_features = ['returns', 'log_returns', 'price_change', 'price_change_pct', 'sma_5']
    new_features = [col for col in features.columns if col not in base_features]
    
    for i, feature in enumerate(new_features[:15]):
        value = features[feature].iloc[-1]
        print(f'  {feature}: {value:.4f}')
    
    print(f'... 还有 {len(new_features) - 15} 个新特征')