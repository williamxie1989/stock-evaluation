# -*- coding: utf-8 -*-
"""
价量特征工程模块
实现基于价格和成交量的内生代理特征，完全不依赖外部数据源
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class PriceVolumeFeatureGenerator:
    """
    价量特征生成器
    所有特征均基于 OHLCV (Open, High, Low, Close, Volume) 数据计算
    """
    
    def __init__(self, lookback_days: int = 180):
        """
        初始化价量特征生成器
        
        Args:
            lookback_days: 回溯天数，用于计算滚动特征
        """
        self.lookback_days = lookback_days
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有价量特征
        
        Args:
            df: 包含 date, open, high, low, close, volume 列的 DataFrame
                要求 date 列为 datetime 类型，并已设置为索引
        
        Returns:
            添加了所有价量特征的 DataFrame
        """
        # 🔧 确保数值列为float类型（避免Decimal类型问题）
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 基础收益率（避免FutureWarning）
        df['ret'] = df['close'].pct_change(fill_method=None)
        
        # 1. 成交额规模代理特征
        df = self._add_turnover_features(df)
        
        # 2. 流动性特征 (Amihud)
        df = self._add_liquidity_features(df)
        
        # 3. 成交活跃度特征
        df = self._add_volume_activity_features(df)
        
        # 4. 波动率/风险特征
        df = self._add_volatility_features(df)
        
        # 5. 动量/反转特征
        df = self._add_momentum_features(df)
        
        # 6. 趋势质量特征
        df = self._add_trend_features(df)
        
        # 7. 日内结构特征
        df = self._add_intraday_features(df)
        
        # 8. 资金流特征
        df = self._add_money_flow_features(df)
        
        # 9. VWAP特征 (成交量加权平均价格)
        df = self._add_vwap_features(df)
        
        # 10. 高级波动率特征
        df = self._add_advanced_volatility(df)
        
        # 11. 高级动量特征
        df = self._add_advanced_momentum(df)
        
        # 12. 高级流动性特征
        df = self._add_advanced_liquidity(df)
        
        return df
    
    def _add_turnover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交额规模代理特征"""
        # 成交额 = 收盘价 * 成交量
        df['turnover'] = df['close'] * df['volume']
        
        # ADV_20: 20日平均成交额 (使用EMA近似)
        df['ADV_20'] = df['turnover'].ewm(span=20, adjust=False).mean()
        
        # log_ADV_20: 对数化成交额
        df['log_ADV_20'] = np.log(df['ADV_20'] + 1)
        
        return df
    
    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """流动性特征 (Amihud 非流动性指标)"""
        # illiq_20: 20日平均 |收益率| / 成交额
        # 值越大表示非流动性越高
        df['abs_ret'] = df['ret'].abs()
        df['illiq_20'] = (
            df['abs_ret'] / (df['turnover'] + 1e-9)
        ).rolling(window=20, min_periods=10).mean()
        
        # 清理临时列
        df.drop('abs_ret', axis=1, inplace=True)
        
        return df
    
    def _add_volume_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交活跃度特征"""
        # vol_norm_20: 当日成交量 / 20日平均成交量
        df['vol_mean_20'] = df['volume'].rolling(window=20, min_periods=10).mean()
        df['vol_norm_20'] = df['volume'] / (df['vol_mean_20'] + 1e-9)
        
        # z_vol_20: 成交量的标准化分数
        df['vol_std_20'] = df['volume'].rolling(window=20, min_periods=10).std()
        df['z_vol_20'] = (df['volume'] - df['vol_mean_20']) / (df['vol_std_20'] + 1e-9)
        
        # 清理临时列
        df.drop(['vol_mean_20', 'vol_std_20'], axis=1, inplace=True)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率/风险特征"""
        # vol_20: 20日收益率标准差 (年化可选)
        df['vol_20'] = df['ret'].rolling(window=20, min_periods=10).std()
        
        # vol_60: 60日收益率标准差
        df['vol_60'] = df['ret'].rolling(window=60, min_periods=30).std()
        
        # downside_vol_20: 下行波动率 (仅考虑负收益)
        negative_ret = df['ret'].copy()
        negative_ret[negative_ret > 0] = 0
        df['downside_vol_20'] = negative_ret.rolling(window=20, min_periods=10).std()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量/反转特征"""
        # ret_5, ret_20, ret_60, ret_120: 不同周期的累计收益
        df['ret_5'] = df['close'].pct_change(periods=5, fill_method=None)
        df['ret_20'] = df['close'].pct_change(periods=20, fill_method=None)
        df['ret_60'] = df['close'].pct_change(periods=60, fill_method=None)
        df['ret_120'] = df['close'].pct_change(periods=120, fill_method=None)
        
        # risk_adj_mom_60: 风险调整动量 = 60日收益 / 60日波动率
        df['risk_adj_mom_60'] = df['ret_60'] / (df['vol_60'] + 1e-9)
        
        # short_rev: 短期反转 = -5日收益
        df['short_rev'] = -df['ret_5']
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """趋势质量特征 (基于线性回归)"""
        # 对 log(close) 做60日线性回归
        log_close = np.log(df['close'])
        
        # 使用滚动窗口计算回归斜率和R²
        window = 60
        slopes = []
        r_squares = []
        
        for i in range(len(df)):
            if i < window - 1:
                slopes.append(np.nan)
                r_squares.append(np.nan)
            else:
                y = log_close.iloc[i - window + 1:i + 1].values
                x = np.arange(window)
                
                # 计算线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                slopes.append(slope)
                r_squares.append(r_value ** 2)
        
        df['trend_slope_60'] = slopes
        df['trend_R2_60'] = r_squares
        
        return df
    
    def _add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """日内结构特征"""
        # range_ratio: (最高价 - 最低价) / 收盘价
        df['range_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
        
        # gap_ratio: |开盘价 - 前收盘价| / 前收盘价
        df['prev_close'] = df['close'].shift(1)
        df['gap_ratio'] = (df['open'] - df['prev_close']).abs() / (df['prev_close'] + 1e-9)
        
        # 清理临时列
        df.drop('prev_close', axis=1, inplace=True)
        
        return df
    
    def _add_money_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """资金流特征 (只依赖价量数据)"""
        # OBV (On-Balance Volume)
        df['price_change'] = df['close'].diff()
        df['obv_delta'] = df['volume'].where(df['price_change'] > 0, 
                                              -df['volume'].where(df['price_change'] < 0, 0))
        df['OBV'] = df['obv_delta'].cumsum()
        
        # CMF (Chaikin Money Flow) - 20日
        df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
        df['mf_volume'] = df['mf_multiplier'] * df['volume']
        df['CMF'] = (
            df['mf_volume'].rolling(window=20, min_periods=10).sum() / 
            (df['volume'].rolling(window=20, min_periods=10).sum() + 1e-9)
        )
        
        # MFI (Money Flow Index) - 14日
        df = self._add_mfi(df)
        
        # 清理临时列
        temp_cols = ['price_change', 'obv_delta', 'mf_multiplier', 'mf_volume']
        df.drop(temp_cols, axis=1, inplace=True, errors='ignore')
        
        return df


    def _add_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP (成交量加权平均价格) 特征"""
        # VWAP = sum(price * volume) / sum(volume)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
        
        # 价格偏离VWAP (溢价/折价)
        df['price_to_vwap'] = df['close'] / (df['vwap'] + 1e-9) - 1.0
        
        # VWAP动量
        df['vwap_mom_5'] = df['vwap'].pct_change(5, fill_method=None)
        df['vwap_mom_20'] = df['vwap'].pct_change(20, fill_method=None)
        
        # VWAP趋势强度
        df['vwap_trend'] = (df['vwap'] - df['vwap'].shift(20)) / (df['vwap'].shift(20) + 1e-9)
        
        return df


    def _add_advanced_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级波动率特征 (更精确的波动率估计)"""
        # Parkinson波动率 (基于高低价，比收盘价波动率更有效)
        # 理论上比简单波动率效率提升5倍
        df['parkinson_vol_20'] = np.sqrt(
            ((np.log(df['high'] / (df['low'] + 1e-9)) ** 2) / (4 * np.log(2))).rolling(20, min_periods=10).mean()
        )
        
        # Garman-Klass波动率 (结合OHLC，效率最高)
        df['gk_vol_20'] = np.sqrt(
            (0.5 * (np.log(df['high'] / (df['low'] + 1e-9)) ** 2) - 
             (2 * np.log(2) - 1) * (np.log(df['close'] / (df['open'] + 1e-9)) ** 2)).rolling(20, min_periods=10).mean()
        )
        
        # 波动率的波动率 (vol of vol, 识别波动率regime变化)
        df['vol_of_vol_60'] = df['vol_20'].rolling(60, min_periods=30).std()
        
        # 波动率偏度 (skewness, 识别尾部风险)
        df['vol_skew_60'] = df['ret'].rolling(60, min_periods=30).skew()
        
        # 波动率峰度 (kurtosis, 识别黑天鹅风险)
        df['vol_kurt_60'] = df['ret'].rolling(60, min_periods=30).kurt()
        
        # 波动率比率 (短期/长期)
        df['vol_ratio'] = df['vol_20'] / (df['vol_60'] + 1e-9)
        
        return df


    def _add_advanced_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级动量特征 (经典技术指标)"""
        # 动量加速度 (momentum of momentum)
        df['mom_accel_20'] = df['ret_20'] - df['ret_20'].shift(20)
        
        # RSI (相对强弱指标, 14日标准)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        # RSI超买超卖信号
        df['RSI_oversold'] = (df['RSI'] < 30).astype(float)  # RSI<30为超卖
        df['RSI_overbought'] = (df['RSI'] > 70).astype(float)  # RSI>70为超买
        
        # 价格新高/新低 (突破信号)
        df['new_high_20'] = (df['close'] == df['close'].rolling(20, min_periods=10).max()).astype(float)
        df['new_low_20'] = (df['close'] == df['close'].rolling(20, min_periods=10).min()).astype(float)
        df['new_high_60'] = (df['close'] == df['close'].rolling(60, min_periods=30).max()).astype(float)
        
        # 动量持续性 (连续上涨/下跌天数)
        df['up_days'] = (df['ret'] > 0).astype(int).rolling(20, min_periods=10).sum()
        df['down_days'] = (df['ret'] < 0).astype(int).rolling(20, min_periods=10).sum()
        
        return df


    def _add_advanced_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级流动性特征"""
        # Roll隐含价差 (基于序列相关性估计买卖价差)
        def roll_spread(returns):
            if len(returns) < 2:
                return 0
            cov = np.cov(returns[:-1], returns[1:])[0, 1]
            return 2 * np.sqrt(max(-cov, 0))
        
        df['roll_spread'] = df['ret'].rolling(20, min_periods=10).apply(roll_spread, raw=True)
        
        # 成交量持续性 (autocorrelation)
        def autocorr(x):
            if len(x) < 2:
                return 0
            return np.corrcoef(x[:-1], x[1:])[0, 1] if len(set(x)) > 1 else 0
        
        df['volume_persistence'] = df['volume'].rolling(20, min_periods=10).apply(autocorr, raw=True)
        
        # 价格冲击 (price impact, 单位成交量引起的价格变化)
        df['price_impact'] = df['ret'].abs() / (df['vol_norm_20'] + 1e-9)
        
        # Amihud非流动性指标 (已有illiq_20, 这里添加60日版本)
        df['illiq_60'] = (
            df['ret'].abs() / (df['turnover'] + 1e-9)
        ).rolling(window=60, min_periods=30).mean()
        
        # 买卖压力 (基于成交量方向性)
        # 上涨日成交量 / 总成交量
        up_volume = (df['ret'] > 0).astype(float) * df['volume']
        df['buy_pressure'] = up_volume.rolling(20, min_periods=10).sum() / (df['volume'].rolling(20, min_periods=10).sum() + 1e-9)
        
        return df

    
    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算资金流量指标 (Money Flow Index)"""
        # 典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 资金流
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # 正负资金流
        price_diff = df['typical_price'].diff()
        positive_flow = df['money_flow'].where(price_diff > 0, 0)
        negative_flow = df['money_flow'].where(price_diff < 0, 0)
        
        # 计算资金流比率
        positive_mf = positive_flow.rolling(window=period, min_periods=period//2).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=period//2).sum()
        
        mf_ratio = positive_mf / (negative_mf + 1e-9)
        df['MFI'] = 100 - (100 / (1 + mf_ratio))
        
        # 清理临时列
        df.drop(['typical_price', 'money_flow'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有价量特征的名称列表
        
        Returns:
            特征名称列表
        """
        return [
            # 成交额规模
            'turnover', 'ADV_20', 'log_ADV_20',
            # 流动性
            'illiq_20',
            # 成交活跃度
            'vol_norm_20', 'z_vol_20',
            # 波动率
            'vol_20', 'vol_60', 'downside_vol_20',
            # 动量/反转
            'ret', 'ret_5', 'ret_20', 'ret_60', 'ret_120',
            'risk_adj_mom_60', 'short_rev',
            # 趋势
            'trend_slope_60', 'trend_R2_60',
            # 日内
            'range_ratio', 'gap_ratio',
            # 资金流
            'OBV', 'CMF', 'MFI'
        ]


def build_price_volume_features(symbols: List[str], 
                                  data_access,
                                  lookback: int = 180,
                                  as_of_date: Optional[str] = None) -> pd.DataFrame:
    """
    为多个股票批量构建价量特征
    
    Args:
        symbols: 股票代码列表
        data_access: 数据访问层对象
        lookback: 回溯天数
        as_of_date: 截止日期 (格式: 'YYYY-MM-DD')，None表示最新数据
    
    Returns:
        包含所有股票特征的 DataFrame，MultiIndex (symbol, date)
    """
    generator = PriceVolumeFeatureGenerator(lookback_days=lookback)
    
    all_features = []
    
    for symbol in symbols:
        try:
            # 获取历史数据
            df = data_access.get_stock_data(
                symbol=symbol,
                start_date=(pd.Timestamp(as_of_date or pd.Timestamp.now()) - pd.Timedelta(days=lookback + 30)).strftime('%Y-%m-%d'),
                end_date=as_of_date
            )
            
            if df is None or len(df) < 45:  # 最小历史数据要求
                logger.warning(f"股票 {symbol} 数据不足，跳过")
                continue
            
            # 确保列名标准化
            df.columns = df.columns.str.lower()
            
            # 设置日期索引
            if 'date' in df.columns and df.index.name != 'date':
                df.set_index('date', inplace=True)
            
            # 生成特征
            df_features = generator.generate_features(df)
            
            # 只保留最后一行 (as_of_date 的特征)
            df_features = df_features.tail(1).copy()
            df_features['symbol'] = symbol
            
            all_features.append(df_features)
            
        except Exception as e:
            logger.error(f"处理股票 {symbol} 时出错: {e}")
            continue
    
    if not all_features:
        return pd.DataFrame()
    
    # 合并所有特征
    result = pd.concat(all_features, ignore_index=False)
    result.reset_index(inplace=True)
    
    return result
