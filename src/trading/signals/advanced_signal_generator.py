"""
高级信号生成器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 到 1.0
    timestamp: datetime
    price: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

class AdvancedSignalGenerator:
    """高级信号生成器 - 生成高质量交易信号"""
    
    def __init__(self):
        self.signals_cache = {}
        self.signal_history = []
        self.performance_metrics = {}
        
        logger.info("AdvancedSignalGenerator initialized")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str = 'Unknown') -> List[Signal]:
        """生成交易信号"""
        try:
            if data.empty or len(data) < 20:
                logger.warning(f"数据不足，无法生成信号: {len(data)}")
                return []
            
            signals = []
            
            # 1. 移动平均线信号
            ma_signals = self._generate_ma_signals(data, symbol)
            signals.extend(ma_signals)
            
            # 2. RSI信号
            rsi_signals = self._generate_rsi_signals(data, symbol)
            signals.extend(rsi_signals)
            
            # 3. MACD信号
            macd_signals = self._generate_macd_signals(data, symbol)
            signals.extend(macd_signals)
            
            # 4. 布林带信号
            bb_signals = self._generate_bollinger_signals(data, symbol)
            signals.extend(bb_signals)
            
            # 5. 成交量信号
            volume_signals = self._generate_volume_signals(data, symbol)
            signals.extend(volume_signals)
            
            # 6. 价格动量信号
            momentum_signals = self._generate_momentum_signals(data, symbol)
            signals.extend(momentum_signals)
            
            # 过滤和排序信号
            filtered_signals = self._filter_signals(signals, data, symbol)
            
            # 缓存信号
            self.signals_cache[symbol] = filtered_signals
            self.signal_history.extend(filtered_signals)
            
            logger.info(f"生成了 {len(filtered_signals)} 个信号 for {symbol}")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return []
    
    def _generate_ma_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成移动平均线信号"""
        try:
            signals = []
            if len(data) < 50:
                return signals
            
            # 计算移动平均线
            ma_short = data['close'].rolling(window=5).mean()
            ma_long = data['close'].rolling(window=20).mean()
            
            # 生成信号
            for i in range(1, len(data)):
                if pd.isna(ma_short.iloc[i]) or pd.isna(ma_long.iloc[i]):
                    continue
                
                # 金叉（买入信号）
                if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.7,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.8,
                        metadata={'type': 'MA_GOLDEN_CROSS', 'ma_short': ma_short.iloc[i], 'ma_long': ma_long.iloc[i]}
                    )
                    signals.append(signal)
                
                # 死叉（卖出信号）
                elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=0.7,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.8,
                        metadata={'type': 'MA_DEATH_CROSS', 'ma_short': ma_short.iloc[i], 'ma_long': ma_long.iloc[i]}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成MA信号失败: {e}")
            return []
    
    def _generate_rsi_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成RSI信号"""
        try:
            signals = []
            if len(data) < 14:
                return signals
            
            # 计算RSI
            rsi = self._calculate_rsi(data['close'], period=14)
            
            # 生成信号
            for i in range(len(data)):
                if pd.isna(rsi.iloc[i]):
                    continue
                
                current_rsi = rsi.iloc[i]
                
                # 超卖（买入信号）
                if current_rsi < 30:
                    strength = min(0.9, (30 - current_rsi) / 30)
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=strength,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.7,
                        metadata={'type': 'RSI_OVERSOLD', 'rsi': current_rsi}
                    )
                    signals.append(signal)
                
                # 超买（卖出信号）
                elif current_rsi > 70:
                    strength = min(0.9, (current_rsi - 70) / 30)
                    signal = Signal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=strength,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.7,
                        metadata={'type': 'RSI_OVERBOUGHT', 'rsi': current_rsi}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成RSI信号失败: {e}")
            return []
    
    def _generate_macd_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成MACD信号"""
        try:
            signals = []
            if len(data) < 35:
                return signals
            
            # 计算MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # 生成信号
            for i in range(1, len(data)):
                if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
                    continue
                
                # MACD金叉（买入信号）
                if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.8,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.8,
                        metadata={'type': 'MACD_GOLDEN_CROSS', 'macd': macd_line.iloc[i], 'signal': signal_line.iloc[i]}
                    )
                    signals.append(signal)
                
                # MACD死叉（卖出信号）
                elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=0.8,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.8,
                        metadata={'type': 'MACD_DEATH_CROSS', 'macd': macd_line.iloc[i], 'signal': signal_line.iloc[i]}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成MACD信号失败: {e}")
            return []
    
    def _generate_bollinger_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成布林带信号"""
        try:
            signals = []
            if len(data) < 20:
                return signals
            
            # 计算布林带
            ma_20 = data['close'].rolling(window=20).mean()
            std_20 = data['close'].rolling(window=20).std()
            upper_band = ma_20 + 2 * std_20
            lower_band = ma_20 - 2 * std_20
            
            # 生成信号
            for i in range(len(data)):
                if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                    continue
                
                current_price = data['close'].iloc[i]
                
                # 价格触及下轨（买入信号）
                if current_price <= lower_band.iloc[i]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.6,
                        timestamp=data.index[i],
                        price=current_price,
                        confidence=0.6,
                        metadata={'type': 'BB_LOWER_TOUCH', 'price': current_price, 'lower_band': lower_band.iloc[i]}
                    )
                    signals.append(signal)
                
                # 价格触及上轨（卖出信号）
                elif current_price >= upper_band.iloc[i]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=0.6,
                        timestamp=data.index[i],
                        price=current_price,
                        confidence=0.6,
                        metadata={'type': 'BB_UPPER_TOUCH', 'price': current_price, 'upper_band': upper_band.iloc[i]}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成布林带信号失败: {e}")
            return []
    
    def _generate_volume_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成成交量信号"""
        try:
            signals = []
            if len(data) < 20:
                return signals
            
            # 计算成交量移动平均
            volume_ma = data['volume'].rolling(window=20).mean()
            
            # 生成信号
            for i in range(len(data)):
                if pd.isna(volume_ma.iloc[i]) or pd.isna(data['volume'].iloc[i]):
                    continue
                
                current_volume = data['volume'].iloc[i]
                avg_volume = volume_ma.iloc[i]
                
                # 成交量放大（买入信号）
                if current_volume > avg_volume * 1.5:
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.5,
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.5,
                        metadata={'type': 'VOLUME_SPIKE', 'volume': current_volume, 'avg_volume': avg_volume}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成成交量信号失败: {e}")
            return []
    
    def _generate_momentum_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """生成动量信号"""
        try:
            signals = []
            if len(data) < 10:
                return signals
            
            # 计算价格动量
            momentum = data['close'].pct_change(periods=5)
            
            # 生成信号
            for i in range(len(data)):
                if pd.isna(momentum.iloc[i]):
                    continue
                
                current_momentum = momentum.iloc[i]
                
                # 正动量（买入信号）
                if current_momentum > 0.02:  # 2%涨幅
                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=min(0.8, current_momentum * 10),
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.6,
                        metadata={'type': 'POSITIVE_MOMENTUM', 'momentum': current_momentum}
                    )
                    signals.append(signal)
                
                # 负动量（卖出信号）
                elif current_momentum < -0.02:  # 2%跌幅
                    signal = Signal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=min(0.8, abs(current_momentum) * 10),
                        timestamp=data.index[i],
                        price=data['close'].iloc[i],
                        confidence=0.6,
                        metadata={'type': 'NEGATIVE_MOMENTUM', 'momentum': current_momentum}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"生成动量信号失败: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def _filter_signals(self, signals: List[Signal], data: pd.DataFrame, symbol: str) -> List[Signal]:
        """过滤信号"""
        try:
            if not signals:
                return []
            
            # 按时间排序
            signals.sort(key=lambda x: x.timestamp)
            
            # 移除重复信号（同一类型在3天内只保留最强的一个）
            filtered = []
            last_signals = {}
            
            for signal in signals:
                signal_key = f"{signal.signal_type}_{signal.symbol}"
                
                # 检查是否已有相同类型的信号
                if signal_key in last_signals:
                    last_signal = last_signals[signal_key]
                    time_diff = (signal.timestamp - last_signal.timestamp).days
                    
                    # 如果间隔小于3天，保留更强的信号
                    if time_diff < 3:
                        if signal.strength > last_signal.strength:
                            # 移除之前的信号，添加新信号
                            if last_signal in filtered:
                                filtered.remove(last_signal)
                            filtered.append(signal)
                            last_signals[signal_key] = signal
                    else:
                        # 间隔足够大，添加新信号
                        filtered.append(signal)
                        last_signals[signal_key] = signal
                else:
                    # 第一个此类信号
                    filtered.append(signal)
                    last_signals[signal_key] = signal
            
            # 按强度排序
            filtered.sort(key=lambda x: x.strength, reverse=True)
            
            # 限制信号数量（每个股票最多10个信号）
            max_signals = 10
            if len(filtered) > max_signals:
                filtered = filtered[:max_signals]
            
            logger.info(f"过滤后剩余 {len(filtered)} 个信号 for {symbol}")
            return filtered
            
        except Exception as e:
            logger.error(f"过滤信号失败: {e}")
            return signals
    
    def get_latest_signals(self, symbol: str = None, limit: int = 10) -> List[Signal]:
        """获取最新信号"""
        try:
            if symbol and symbol in self.signals_cache:
                signals = self.signals_cache[symbol]
            else:
                # 获取所有信号
                signals = []
                for symbol_signals in self.signals_cache.values():
                    signals.extend(symbol_signals)
            
            # 按时间排序
            signals.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 限制数量
            if limit and len(signals) > limit:
                signals = signals[:limit]
            
            return signals
            
        except Exception as e:
            logger.error(f"获取最新信号失败: {e}")
            return []
    
    def get_signal_summary(self, symbol: str = None) -> Dict[str, Any]:
        """获取信号摘要"""
        try:
            if symbol and symbol in self.signals_cache:
                signals = self.signals_cache[symbol]
            else:
                signals = []
                for symbol_signals in self.signals_cache.values():
                    signals.extend(symbol_signals)
            
            if not signals:
                return {'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0, 'avg_strength': 0.0}
            
            summary = {
                'total_signals': len(signals),
                'buy_signals': len([s for s in signals if s.signal_type == 'BUY']),
                'sell_signals': len([s for s in signals if s.signal_type == 'SELL']),
                'avg_strength': np.mean([s.strength for s in signals]),
                'avg_confidence': np.mean([s.confidence for s in signals]),
                'latest_signal': max(signals, key=lambda x: x.timestamp) if signals else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取信号摘要失败: {e}")
            return {'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0, 'avg_strength': 0.0}
    
    def reset(self):
        """重置信号生成器"""
        self.signals_cache.clear()
        self.signal_history.clear()
        self.performance_metrics.clear()
        logger.info("高级信号生成器已重置")