import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """多因子信号生成器"""
    
    def __init__(self):
        self.factors = {}
        self.signal_weights = {
            'trend': 0.3,      # 趋势因子权重
            'momentum': 0.25,  # 动量因子权重
            'volatility': 0.2, # 波动率因子权重
            'volume': 0.15,    # 成交量因子权重
            'pattern': 0.1     # 形态因子权重
        }
    
    def calculate_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算多因子指标"""
        factors = {}
        
        # 趋势因子
        factors['trend_ma5'] = df['Close'].rolling(window=5).mean()
        factors['trend_ma20'] = df['Close'].rolling(window=20).mean()
        factors['trend_ma60'] = df['Close'].rolling(window=60).mean()
        
        # 动量因子
        factors['momentum_rsi'] = self._calculate_rsi(df['Close'])
        factors['momentum_macd'], factors['momentum_macd_signal'] = self._calculate_macd(df['Close'])
        
        # 波动率因子
        factors['volatility_atr'] = self._calculate_atr(df)
        factors['volatility_bb'] = self._calculate_bollinger_bands(df['Close'])
        
        # 成交量因子
        factors['volume_obv'] = self._calculate_obv(df)
        factors['volume_vwap'] = self._calculate_vwap(df)
        
        # 形态因子
        factors['pattern_engulfing'] = self._detect_engulfing_pattern(df)
        factors['pattern_hammer'] = self._detect_hammer_pattern(df)
        
        return factors
    
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
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """计算OBV指标"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """计算VWAP指标"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def _detect_engulfing_pattern(self, df: pd.DataFrame) -> pd.Series:
        """检测吞没形态"""
        # 看涨吞没
        bullish_engulfing = (
            (df['Close'] > df['Open']) &  # 当前阳线
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # 前一根阴线
            (df['Close'] > df['Open'].shift(1)) &  # 当前收盘价高于前一根开盘价
            (df['Open'] < df['Close'].shift(1))   # 当前开盘价低于前一根收盘价
        )
        
        # 看跌吞没
        bearish_engulfing = (
            (df['Close'] < df['Open']) &  # 当前阴线
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # 前一根阳线
            (df['Close'] < df['Open'].shift(1)) &  # 当前收盘价低于前一根开盘价
            (df['Open'] > df['Close'].shift(1))   # 当前开盘价高于前一根收盘价
        )
        
        return pd.Series(np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0)), index=df.index)
    
    def _detect_hammer_pattern(self, df: pd.DataFrame) -> pd.Series:
        """检测锤子形态"""
        # 锤子线特征：小实体，长下影线
        body_size = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        lower_shadow = df['Close'] - df['Low']  # 对于阳线
        lower_shadow = np.where(df['Close'] < df['Open'], df['Open'] - df['Low'], lower_shadow)  # 对于阴线
        
        hammer = (
            (body_size / total_range < 0.3) &  # 小实体
            (lower_shadow / total_range > 0.6) &  # 长下影线
            ((df['High'] - df['Close']) / total_range < 0.1)  # 短上影线
        )
        
        return pd.Series(np.where(hammer, 1, 0), index=df.index)
    
    def generate_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []
        
        # 趋势信号
        trend_signals = self._generate_trend_signals(df, factors)
        signals.extend(trend_signals)
        
        # 动量信号
        momentum_signals = self._generate_momentum_signals(df, factors)
        signals.extend(momentum_signals)
        
        # 波动率信号
        volatility_signals = self._generate_volatility_signals(df, factors)
        signals.extend(volatility_signals)
        
        # 成交量信号
        volume_signals = self._generate_volume_signals(df, factors)
        signals.extend(volume_signals)
        
        # 形态信号
        pattern_signals = self._generate_pattern_signals(df, factors)
        signals.extend(pattern_signals)
        
        # 信号过滤和去重
        filtered_signals = self._filter_signals(signals)
        
        return filtered_signals
    
    def _generate_trend_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成趋势信号"""
        signals = []
        
        # 均线金叉死叉
        for i in range(1, len(df)):
            # MA5上穿MA20
            if (factors['trend_ma5'].iloc[i] > factors['trend_ma20'].iloc[i] and 
                factors['trend_ma5'].iloc[i-1] <= factors['trend_ma20'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '5日均线上穿20日均线（金叉）',
                    'confidence': 0.7,
                    'factor': 'trend'
                })
            
            # MA5下穿MA20
            elif (factors['trend_ma5'].iloc[i] < factors['trend_ma20'].iloc[i] and 
                  factors['trend_ma5'].iloc[i-1] >= factors['trend_ma20'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '5日均线下穿20日均线（死叉）',
                    'confidence': 0.7,
                    'factor': 'trend'
                })
        
        return signals
    
    def _generate_momentum_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成动量信号"""
        signals = []
        
        # RSI超买超卖
        for i in range(1, len(df)):
            # RSI从超卖区回升
            if (factors['momentum_rsi'].iloc[i] < 30 and 
                factors['momentum_rsi'].iloc[i-1] >= 30):
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': 'RSI从超卖区回升',
                    'confidence': 0.6,
                    'factor': 'momentum'
                })
            
            # RSI进入超买区
            elif (factors['momentum_rsi'].iloc[i] > 70 and 
                  factors['momentum_rsi'].iloc[i-1] <= 70):
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': 'RSI进入超买区',
                    'confidence': 0.6,
                    'factor': 'momentum'
                })
        
        # MACD金叉死叉
        for i in range(1, len(df)):
            # MACD金叉
            if (factors['momentum_macd'].iloc[i] > factors['momentum_macd_signal'].iloc[i] and 
                factors['momentum_macd'].iloc[i-1] <= factors['momentum_macd_signal'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': 'MACD金叉',
                    'confidence': 0.65,
                    'factor': 'momentum'
                })
            
            # MACD死叉
            elif (factors['momentum_macd'].iloc[i] < factors['momentum_macd_signal'].iloc[i] and 
                  factors['momentum_macd'].iloc[i-1] >= factors['momentum_macd_signal'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': 'MACD死叉',
                    'confidence': 0.65,
                    'factor': 'momentum'
                })
        
        return signals
    
    def _generate_volatility_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成波动率信号"""
        signals = []
        
        # 布林带信号
        bb = factors['volatility_bb']
        for i in range(1, len(df)):
            # 价格触及布林带下轨
            if (df['Close'].iloc[i] <= bb['lower'].iloc[i] and 
                df['Close'].iloc[i-1] > bb['lower'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '价格触及布林带下轨',
                    'confidence': 0.55,
                    'factor': 'volatility'
                })
            
            # 价格触及布林带上轨
            elif (df['Close'].iloc[i] >= bb['upper'].iloc[i] and 
                  df['Close'].iloc[i-1] < bb['upper'].iloc[i-1]):
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '价格触及布林带上轨',
                    'confidence': 0.55,
                    'factor': 'volatility'
                })
        
        return signals
    
    def _generate_volume_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成成交量信号"""
        signals = []
        
        # OBV突破信号
        for i in range(2, len(df)):
            # OBV突破20日高点
            if (factors['volume_obv'].iloc[i] > factors['volume_obv'].rolling(window=20).max().iloc[i-1] and 
                factors['volume_obv'].iloc[i-1] <= factors['volume_obv'].rolling(window=20).max().iloc[i-2]):
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': 'OBV突破20日高点',
                    'confidence': 0.5,
                    'factor': 'volume'
                })
        
        return signals
    
    def _generate_pattern_signals(self, df: pd.DataFrame, factors: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """生成形态信号"""
        signals = []
        
        # 吞没形态
        for i in range(len(df)):
            if factors['pattern_engulfing'].iloc[i] == 1:  # 看涨吞没
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '看涨吞没形态',
                    'confidence': 0.6,
                    'factor': 'pattern'
                })
            elif factors['pattern_engulfing'].iloc[i] == -1:  # 看跌吞没
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '看跌吞没形态',
                    'confidence': 0.6,
                    'factor': 'pattern'
                })
        
        # 锤子形态
        for i in range(len(df)):
            if factors['pattern_hammer'].iloc[i] == 1:  # 锤子线
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '锤子形态',
                    'confidence': 0.55,
                    'factor': 'pattern'
                })
        
        return signals
    
    def _filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤和去重信号"""
        if not signals:
            return []
        
        # 按日期排序
        signals.sort(key=lambda x: x['date'])
        
        filtered_signals = []
        last_signal_date = None
        last_signal_type = None
        
        for signal in signals:
            # 跳过置信度过低的信号
            if signal.get('confidence', 0) < 0.5:
                continue
            
            # 检查时间间隔（至少间隔3天）
            if last_signal_date and (signal['date'] - last_signal_date).days < 3:
                # 如果同类型信号，跳过；如果不同类型，考虑反转
                if signal['type'] == last_signal_type:
                    continue
                else:
                    # 反转信号，保留但降低置信度
                    signal['confidence'] *= 0.8
            
            filtered_signals.append(signal)
            last_signal_date = signal['date']
            last_signal_type = signal['type']
        
        return filtered_signals

# 使用示例
if __name__ == "__main__":
    # 示例数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    data = {
        'Open': np.random.normal(100, 5, 100).cumsum(),
        'High': np.random.normal(105, 5, 100).cumsum(),
        'Low': np.random.normal(95, 5, 100).cumsum(),
        'Close': np.random.normal(100, 5, 100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # 生成信号
    generator = SignalGenerator()
    factors = generator.calculate_factors(df)
    signals = generator.generate_signals(df, factors)
    
    print(f"生成 {len(signals)} 个交易信号:")
    for signal in signals:
        print(f"{signal['date'].strftime('%Y-%m-%d')} - {signal['type']} @ {signal['price']:.2f} "
              f"({signal['reason']}, 置信度: {signal['confidence']})")