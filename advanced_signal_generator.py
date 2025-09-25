import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """交易信号数据类"""
    date: datetime
    type: str  # BUY/SELL
    price: float
    reason: str
    confidence: float
    factor: str
    stop_loss: float = None
    take_profit: float = None
    timeframe: str = "daily"
    volume_confirmation: bool = False
    multi_timeframe_confirmed: bool = False

class AdvancedSignalGenerator:
    """高级信号生成器 - 实现多指标确认和风险控制"""
    
    def __init__(self):
        self.signal_weights = {
            'trend': 0.25,      # 趋势因子权重
            'momentum': 0.20,   # 动量因子权重
            'volatility': 0.15, # 波动率因子权重
            'volume': 0.15,     # 成交量因子权重
            'pattern': 0.10,    # 形态因子权重
            'multi_timeframe': 0.15  # 多时间框架确认权重
        }
        
        # 信号过滤参数
        self.min_confidence = 0.6
        self.volume_threshold = 1.2  # 成交量确认阈值
        self.atr_multiplier = 2.0    # ATR止损倍数
        
    def generate_advanced_signals(self, 
                                 daily_data: pd.DataFrame,
                                 hourly_data: pd.DataFrame = None,
                                 weekly_data: pd.DataFrame = None) -> List[Signal]:
        """
        生成高级交易信号（多时间框架确认）
        
        Args:
            daily_data: 日线数据
            hourly_data: 小时线数据（可选）
            weekly_data: 周线数据（可选）
            
        Returns:
            List[Signal]: 过滤后的交易信号列表
        """
        
        # 1. 生成基础信号
        base_signals = self._generate_base_signals(daily_data)
        
        # 2. 多时间框架确认
        if hourly_data is not None:
            base_signals = self._confirm_with_hourly_data(base_signals, hourly_data)
        
        if weekly_data is not None:
            base_signals = self._confirm_with_weekly_data(base_signals, weekly_data)
        
        # 3. 成交量确认
        base_signals = self._confirm_with_volume(base_signals, daily_data)
        
        # 4. 计算止损止盈
        base_signals = self._calculate_stop_loss_take_profit(base_signals, daily_data)
        
        # 5. 信号过滤
        filtered_signals = self._filter_signals(base_signals)
        
        return filtered_signals
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        兼容性方法 - 生成交易信号（单时间框架）
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            List[Signal]: 交易信号列表
        """
        return self.generate_advanced_signals(df)
    
    def _generate_base_signals(self, df: pd.DataFrame) -> List[Signal]:
        """生成基础技术指标信号"""
        signals = []
        
        # 计算技术指标
        factors = self._calculate_technical_factors(df)
        
        # 趋势信号
        signals.extend(self._generate_trend_signals(df, factors))
        
        # 动量信号
        signals.extend(self._generate_momentum_signals(df, factors))
        
        # 波动率信号
        signals.extend(self._generate_volatility_signals(df, factors))
        
        # 形态信号
        signals.extend(self._generate_pattern_signals(df, factors))
        
        return signals
    
    def _calculate_technical_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术指标"""
        factors = {}
        
        # 移动平均线
        factors['ma5'] = df['Close'].rolling(window=5).mean()
        factors['ma20'] = df['Close'].rolling(window=20).mean()
        factors['ma60'] = df['Close'].rolling(window=60).mean()
        
        # RSI
        factors['rsi'] = self._calculate_rsi(df['Close'])
        
        # MACD
        factors['macd'], factors['macd_signal'] = self._calculate_macd(df['Close'])
        
        # ATR
        factors['atr'] = self._calculate_atr(df)
        
        # 布林带
        bb = self._calculate_bollinger_bands(df['Close'])
        factors.update(bb)
        
        # OBV
        factors['obv'] = self._calculate_obv(df)
        
        return factors
    
    def _generate_trend_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Signal]:
        """生成趋势信号"""
        signals = []
        
        for i in range(2, len(df)):
            # 多重均线确认
            ma_condition = (
                factors['ma5'].iloc[i] > factors['ma20'].iloc[i] > factors['ma60'].iloc[i] and
                factors['ma5'].iloc[i-1] <= factors['ma20'].iloc[i-1]
            )
            
            if ma_condition:
                signals.append(Signal(
                    date=df.index[i],
                    type='BUY',
                    price=df['Close'].iloc[i],
                    reason='多重均线金叉确认（5>20>60）',
                    confidence=0.75,
                    factor='trend'
                ))
            
            # 趋势反转信号
            trend_reversal = (
                factors['ma5'].iloc[i] < factors['ma20'].iloc[i] < factors['ma60'].iloc[i] and
                factors['ma5'].iloc[i-1] >= factors['ma20'].iloc[i-1]
            )
            
            if trend_reversal:
                signals.append(Signal(
                    date=df.index[i],
                    type='SELL',
                    price=df['Close'].iloc[i],
                    reason='多重均线死叉确认（5<20<60）',
                    confidence=0.75,
                    factor='trend'
                ))
        
        return signals
    
    def _generate_momentum_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Signal]:
        """生成动量信号"""
        signals = []
        
        for i in range(2, len(df)):
            # RSI超买超卖 + MACD确认
            rsi_oversold = factors['rsi'].iloc[i] < 30 and factors['rsi'].iloc[i-1] >= 30
            macd_bullish = factors['macd'].iloc[i] > factors['macd_signal'].iloc[i]
            
            if rsi_oversold and macd_bullish:
                signals.append(Signal(
                    date=df.index[i],
                    type='BUY',
                    price=df['Close'].iloc[i],
                    reason='RSI超卖区回升 + MACD金叉确认',
                    confidence=0.70,
                    factor='momentum'
                ))
            
            # RSI超买 + MACD死叉
            rsi_overbought = factors['rsi'].iloc[i] > 70 and factors['rsi'].iloc[i-1] <= 70
            macd_bearish = factors['macd'].iloc[i] < factors['macd_signal'].iloc[i]
            
            if rsi_overbought and macd_bearish:
                signals.append(Signal(
                    date=df.index[i],
                    type='SELL',
                    price=df['Close'].iloc[i],
                    reason='RSI超买区回落 + MACD死叉确认',
                    confidence=0.70,
                    factor='momentum'
                ))
        
        return signals
    
    def _generate_volatility_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Signal]:
        """生成波动率信号"""
        signals = []
        
        for i in range(2, len(df)):
            # 布林带突破 + 低波动率
            bb_breakout = df['Close'].iloc[i] > factors['bb_upper'].iloc[i]
            low_volatility = factors['atr'].iloc[i] < factors['atr'].rolling(20).mean().iloc[i]
            
            if bb_breakout and low_volatility:
                signals.append(Signal(
                    date=df.index[i],
                    type='BUY',
                    price=df['Close'].iloc[i],
                    reason='布林带突破 + 低波动率环境',
                    confidence=0.65,
                    factor='volatility'
                ))
            
            # 布林带下轨支撑 + 高波动率
            bb_support = df['Close'].iloc[i] < factors['bb_lower'].iloc[i]
            high_volatility = factors['atr'].iloc[i] > factors['atr'].rolling(20).mean().iloc[i]
            
            if bb_support and high_volatility:
                signals.append(Signal(
                    date=df.index[i],
                    type='SELL',
                    price=df['Close'].iloc[i],
                    reason='布林带下轨支撑 + 高波动率环境',
                    confidence=0.65,
                    factor='volatility'
                ))
        
        return signals
    
    def _generate_pattern_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Signal]:
        """生成形态信号"""
        signals = []
        
        for i in range(2, len(df)):
            # 吞没形态确认
            engulfing = self._detect_engulfing_pattern(df.iloc[i-1:i+1])
            if engulfing == 1:  # 看涨吞没
                signals.append(Signal(
                    date=df.index[i],
                    type='BUY',
                    price=df['Close'].iloc[i],
                    reason='看涨吞没形态确认',
                    confidence=0.60,
                    factor='pattern'
                ))
            elif engulfing == -1:  # 看跌吞没
                signals.append(Signal(
                    date=df.index[i],
                    type='SELL',
                    price=df['Close'].iloc[i],
                    reason='看跌吞没形态确认',
                    confidence=0.60,
                    factor='pattern'
                ))
        
        return signals
    
    def _confirm_with_hourly_data(self, signals: List[Signal], hourly_data: pd.DataFrame) -> List[Signal]:
        """使用小时线数据确认信号"""
        confirmed_signals = []
        
        for signal in signals:
            # 找到对应的小时线数据
            signal_time = signal.date
            hourly_subset = hourly_data[hourly_data.index >= signal_time - timedelta(hours=24)]
            
            if len(hourly_subset) > 0:
                # 检查小时线趋势
                hourly_trend = self._analyze_hourly_trend(hourly_subset)
                
                if (signal.type == 'BUY' and hourly_trend == 'bullish') or \
                   (signal.type == 'SELL' and hourly_trend == 'bearish'):
                    signal.multi_timeframe_confirmed = True
                    signal.confidence *= 1.2  # 提高置信度
                
                confirmed_signals.append(signal)
        
        return confirmed_signals
    
    def _confirm_with_weekly_data(self, signals: List[Signal], weekly_data: pd.DataFrame) -> List[Signal]:
        """使用周线数据确认信号"""
        confirmed_signals = []
        
        for signal in signals:
            # 找到对应的周线数据
            signal_time = signal.date
            weekly_subset = weekly_data[weekly_data.index >= signal_time - timedelta(days=7)]
            
            if len(weekly_subset) > 0:
                # 检查周线趋势
                weekly_trend = self._analyze_weekly_trend(weekly_subset)
                
                if (signal.type == 'BUY' and weekly_trend == 'bullish') or \
                   (signal.type == 'SELL' and weekly_trend == 'bearish'):
                    signal.multi_timeframe_confirmed = True
                    signal.confidence *= 1.1  # 提高置信度
                
                confirmed_signals.append(signal)
        
        return confirmed_signals
    
    def _confirm_with_volume(self, signals: List[Signal], daily_data: pd.DataFrame) -> List[Signal]:
        """使用成交量确认信号"""
        confirmed_signals = []
        
        for signal in signals:
            signal_date = signal.date
            
            # 找到对应的成交量数据
            volume_data = daily_data[daily_data.index == signal_date]
            
            if len(volume_data) > 0:
                current_volume = volume_data['Volume'].iloc[0]
                avg_volume = daily_data['Volume'].rolling(20).mean().iloc[-1]
                
                if current_volume > avg_volume * self.volume_threshold:
                    signal.volume_confirmation = True
                    signal.confidence *= 1.15  # 提高置信度
                
                confirmed_signals.append(signal)
        
        return confirmed_signals
    
    def _calculate_stop_loss_take_profit(self, signals: List[Signal], daily_data: pd.DataFrame) -> List[Signal]:
        """计算止损止盈价格"""
        for signal in signals:
            atr = self._calculate_atr(daily_data).iloc[-1]
            
            if signal.type == 'BUY':
                signal.stop_loss = signal.price - (atr * self.atr_multiplier)
                signal.take_profit = signal.price + (atr * 3)  # 3倍ATR止盈
            else:  # SELL
                signal.stop_loss = signal.price + (atr * self.atr_multiplier)
                signal.take_profit = signal.price - (atr * 3)
        
        return signals
    
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """过滤信号"""
        filtered_signals = []
        
        for signal in signals:
            # 置信度过滤
            if signal.confidence < self.min_confidence:
                continue
            
            # 多时间框架确认的信号优先
            if signal.multi_timeframe_confirmed:
                filtered_signals.append(signal)
            # 成交量确认的信号次优先
            elif signal.volume_confirmation:
                filtered_signals.append(signal)
            # 其他高质量信号
            elif signal.confidence > 0.75:
                filtered_signals.append(signal)
        
        # 按置信度排序
        filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_signals
    
    def _analyze_hourly_trend(self, hourly_data: pd.DataFrame) -> str:
        """分析小时线趋势"""
        if len(hourly_data) < 5:
            return 'neutral'
        
        ma5 = hourly_data['Close'].rolling(5).mean()
        ma20 = hourly_data['Close'].rolling(20).mean()
        
        if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-1] > ma5.iloc[-2]:
            return 'bullish'
        elif ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-1] < ma5.iloc[-2]:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_weekly_trend(self, weekly_data: pd.DataFrame) -> str:
        """分析周线趋势"""
        if len(weekly_data) < 3:
            return 'neutral'
        
        # 简单的周线趋势判断
        price_change = (weekly_data['Close'].iloc[-1] - weekly_data['Close'].iloc[0]) / weekly_data['Close'].iloc[0]
        
        if price_change > 0.02:  # 上涨2%
            return 'bullish'
        elif price_change < -0.02:  # 下跌2%
            return 'bearish'
        else:
            return 'neutral'
    
    # 技术指标计算方法（与原始SignalGenerator相同）
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """计算MACD指标"""
        exp_fast = prices.ewm(span=fast, adjust=False).mean()
        exp_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'bb_upper': upper, 'bb_middle': middle, 'bb_lower': lower}
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """计算OBV指标"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    
    def _detect_engulfing_pattern(self, df: pd.DataFrame) -> int:
        """检测吞没形态"""
        if len(df) < 2:
            return 0
        
        # 看涨吞没
        bullish_engulfing = (
            (df['Close'].iloc[-1] > df['Open'].iloc[-1]) and  # 当前阳线
            (df['Close'].iloc[-2] < df['Open'].iloc[-2]) and  # 前一根阴线
            (df['Close'].iloc[-1] > df['Open'].iloc[-2]) and  # 当前收盘价高于前一根开盘价
            (df['Open'].iloc[-1] < df['Close'].iloc[-2])   # 当前开盘价低于前一根收盘价
        )
        
        # 看跌吞没
        bearish_engulfing = (
            (df['Close'].iloc[-1] < df['Open'].iloc[-1]) and  # 当前阴线
            (df['Close'].iloc[-2] > df['Open'].iloc[-2]) and  # 前一根阳线
            (df['Close'].iloc[-1] < df['Open'].iloc[-2]) and  # 当前收盘价低于前一根开盘价
            (df['Open'].iloc[-1] > df['Close'].iloc[-2])   # 当前开盘价高于前一根收盘价
        )
        
        return 1 if bullish_engulfing else (-1 if bearish_engulfing else 0)

# 使用示例
if __name__ == "__main__":
    # 创建高级信号生成器
    advanced_generator = AdvancedSignalGenerator()
    
    # 示例数据（实际使用时需要真实数据）
    # daily_data = pd.read_csv('daily_data.csv', index_col=0, parse_dates=True)
    # hourly_data = pd.read_csv('hourly_data.csv', index_col=0, parse_dates=True)
    
    # 生成信号
    # signals = advanced_generator.generate_advanced_signals(daily_data, hourly_data)
    
    print("高级信号生成器已创建，支持多时间框架确认和风险控制")