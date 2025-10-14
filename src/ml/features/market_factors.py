# -*- coding: utf-8 -*-
"""
市场因子特征工程模块
实现市场因子、Beta、特质波动率等特征，完全内生构建
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MarketFactorGenerator:
    """
    市场因子生成器
    构建市场收益因子、Beta、特质波动率等
    """
    
    def __init__(self, lookback_days: int = 180):
        """
        初始化市场因子生成器
        
        Args:
            lookback_days: 回溯天数
        """
        self.lookback_days = lookback_days
        self.market_returns = None  # 缓存市场收益率数据
    
    def build_market_returns(self, 
                             all_stocks_data: Dict[str, pd.DataFrame],
                             min_stocks: int = 50) -> pd.DataFrame:
        """
        构建市场收益因子 (等权平均)
        
        Args:
            all_stocks_data: 字典 {symbol: DataFrame with 'date', 'close', 'ret'}
            min_stocks: 每日最少股票数量要求
        
        Returns:
            DataFrame with columns: date, MKT (市场收益率), count (参与计算的股票数)
        """
        logger.info(f"开始构建市场因子: 输入 {len(all_stocks_data)} 只股票, 阈值 min_stocks={min_stocks}")
        
        # 提取所有股票的收益率
        all_rets = []
        
        for symbol, df in all_stocks_data.items():
            if df is None or len(df) == 0:
                continue
            
            # 确保有收益率列
            if 'ret' not in df.columns:
                if 'close' in df.columns:
                    df = df.copy()  # 避免修改原始数据
                    # 确保 close 列为 float 类型（避免 Decimal 类型问题）
                    df['close'] = df['close'].astype(float)
                    df['ret'] = df['close'].pct_change(fill_method=None)
                else:
                    continue
            
            # 🔧 确保索引是 DatetimeIndex（用于后续 date 列生成）
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('date')
            
            ret_df = df[['ret']].copy()
            # 确保收益率列为 float 类型
            ret_df = ret_df.astype(float)
            ret_df.columns = [symbol]
            all_rets.append(ret_df)
        
        if not all_rets:
            logger.error("❌ 没有有效的股票数据用于构建市场因子")
            return pd.DataFrame()
        
        logger.info(f"  成功提取 {len(all_rets)} 只股票的收益率")
        
        # 合并所有收益率
        combined = pd.concat(all_rets, axis=1)
        logger.info(f"  合并后时间序列长度: {len(combined)} 天")
        
        # 🔧 修复: 确保数据类型为 float，避免 Decimal 类型导致计算失败
        combined = combined.astype(float)
        
        # 过滤异常值 (收益率超过 ±50% 视为异常)
        combined = combined.clip(-0.5, 0.5)
        
        # 计算等权平均市场收益
        market_df = pd.DataFrame()
        market_df['MKT'] = combined.mean(axis=1, skipna=True)
        market_df['count'] = combined.count(axis=1)
        
        initial_days = len(market_df)
        logger.info(f"  初始市场因子: {initial_days} 天, 每日平均股票数: {market_df['count'].mean():.1f}")
        logger.info(f"  每日股票数统计: min={market_df['count'].min():.0f}, max={market_df['count'].max():.0f}, median={market_df['count'].median():.0f}")
        
        # 过滤样本量不足的日期
        before_filter = len(market_df)
        market_df = market_df[market_df['count'] >= min_stocks].copy()
        after_filter = len(market_df)
        
        if after_filter < before_filter:
            logger.warning(f"  过滤低于 {min_stocks} 只股票的日期: {before_filter - after_filter} 天被移除")
        
        if len(market_df) == 0:
            logger.error(f"❌ 所有日期的股票数均 < {min_stocks}，市场因子为空")
            return pd.DataFrame()
        
        # 🔧 关键修复: 确保返回 DataFrame 有 date 列（用于标签模块匹配）
        if market_df.index.name != 'date':
            market_df.index.name = 'date'
        market_df = market_df.reset_index()  # 将 date 从索引转为列
        
        # 缓存市场收益数据
        self.market_returns = market_df
        
        logger.info(f"✅ 市场因子构建完成: {len(market_df)} 天, 平均 {market_df['count'].mean():.0f} 只股票/天")
        
        return market_df
    
    def add_market_features(self, 
                            df: pd.DataFrame, 
                            symbol: str,
                            market_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        为单个股票添加市场相关特征
        
        Args:
            df: 股票数据，需包含 'ret' 列
            symbol: 股票代码
            market_returns: 市场收益数据 (可选，若无则使用缓存)
        
        Returns:
            添加了市场特征的 DataFrame
        """
        if market_returns is None:
            market_returns = self.market_returns
        
        if market_returns is None or len(market_returns) == 0:
            logger.warning(f"市场收益数据不可用，无法为 {symbol} 添加市场特征")
            return df
        
        # 确保日期索引对齐
        if df.index.name != 'date' and 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # 🔧 确保数值列为 float 类型（避免 Decimal 问题）
        numeric_cols = ['ret', 'close', 'volume', 'open', 'high', 'low', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 🔧 关键修复：market_returns 现在有 date 列，需要转为索引才能 join
        market_for_join = market_returns.copy()
        if 'date' in market_for_join.columns and not isinstance(market_for_join.index, pd.DatetimeIndex):
            market_for_join['date'] = pd.to_datetime(market_for_join['date'])
            market_for_join.set_index('date', inplace=True)
        
        # 合并市场收益
        df = df.join(market_for_join[['MKT']], how='left')
        
        # 填充缺失的市场收益 (用0填充)
        df['MKT'].fillna(0, inplace=True)
        
        # 计算 Beta 和特质波动率 (多窗口滚动回归)
        df = self._add_beta_features(df, windows=[60, 120])

        # 规模与流动性代理指标
        df = self._add_size_liquidity_features(df)

        # 风格/相对表现特征
        df = self._add_style_factors(df)
        
        # 计算相对强弱指标
        df = self._add_relative_strength(df)
        
        return df
    
    def _add_beta_features(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """添加多窗口 Beta、Alpha、特质波动率与解释力度"""
        if windows is None or len(windows) == 0:
            windows = [60]

        for window in sorted(set(windows)):
            betas: List[float] = []
            idio_vols: List[float] = []
            alphas: List[float] = []
            r_squareds: List[float] = []

            down_betas: List[float] = []
            up_betas: List[float] = []

            for i in range(len(df)):
                if i < window - 1:
                    betas.append(np.nan)
                    idio_vols.append(np.nan)
                    alphas.append(np.nan)
                    r_squareds.append(np.nan)
                    down_betas.append(np.nan)
                    up_betas.append(np.nan)
                    continue

                y = df['ret'].iloc[i - window + 1:i + 1].values
                x = df['MKT'].iloc[i - window + 1:i + 1].values

                valid_mask = ~(np.isnan(y) | np.isnan(x))

                if valid_mask.sum() < max(window // 2, 10):
                    betas.append(np.nan)
                    idio_vols.append(np.nan)
                    alphas.append(np.nan)
                    r_squareds.append(np.nan)
                    down_betas.append(np.nan)
                    up_betas.append(np.nan)
                    continue

                y_valid = y[valid_mask]
                x_valid = x[valid_mask]

                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                    y_pred = intercept + slope * x_valid
                    residuals = y_valid - y_pred
                    idio_vol = np.std(residuals, ddof=1)

                    betas.append(slope)
                    alphas.append(intercept)
                    idio_vols.append(idio_vol)
                    r_squareds.append(r_value ** 2)
                except Exception:
                    betas.append(np.nan)
                    alphas.append(np.nan)
                    idio_vols.append(np.nan)
                    r_squareds.append(np.nan)

                # 下行 / 上行 Beta
                down_mask = x_valid < 0
                up_mask = x_valid > 0
                down_betas.append(self._compute_partial_beta(x_valid, y_valid, down_mask))
                up_betas.append(self._compute_partial_beta(x_valid, y_valid, up_mask))

            df[f'beta_{window}'] = betas
            df[f'alpha_{window}'] = alphas
            df[f'idio_vol_{window}'] = idio_vols
            df[f'market_R2_{window}'] = r_squareds
            df[f'down_beta_{window}'] = down_betas
            df[f'up_beta_{window}'] = up_betas

        return df

    def _compute_partial_beta(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        """计算条件 Beta（上行或下行），不足数据时返回 NaN"""
        if mask.sum() < 5:
            return np.nan
        x_sel = x[mask]
        y_sel = y[mask]
        try:
            slope, _, _, _, _ = stats.linregress(x_sel, y_sel)
            return slope
        except Exception:
            return np.nan

    def _add_size_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于成交额的规模与流动性代理指标"""
        if 'turnover' not in df.columns:
            if 'close' in df.columns and 'volume' in df.columns:
                df['turnover'] = pd.to_numeric(df['close'], errors='coerce') * pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['turnover'] = np.nan

        turnover = df['turnover'].replace([np.inf, -np.inf], np.nan)

        df['turnover_ema_20'] = turnover.ewm(span=20, adjust=False, min_periods=5).mean()
        df['log_turnover_20'] = np.log(df['turnover_ema_20'] + 1e-9)

        df['turnover_vol_20'] = turnover.rolling(window=20, min_periods=10).std()

        mean_20 = turnover.rolling(window=20, min_periods=10).mean()
        std_20 = turnover.rolling(window=20, min_periods=10).std()
        df['turnover_z_20'] = (turnover - mean_20) / (std_20 + 1e-9)

        df['turnover_autocorr_5'] = turnover.rolling(window=10, min_periods=6).apply(
            lambda arr: pd.Series(arr).autocorr(lag=1), raw=False
        )

        return df

    def _add_style_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """风格与收益分布特征"""
        df['ret_skew_60'] = df['ret'].rolling(window=60, min_periods=30).skew()
        df['ret_kurt_60'] = df['ret'].rolling(window=60, min_periods=30).kurt()

        df['momentum_diff_60'] = (
            df['ret'].rolling(window=60, min_periods=30).mean() -
            df['MKT'].rolling(window=60, min_periods=30).mean()
        )

        stock_vol_20 = df['ret'].rolling(window=20, min_periods=10).std()
        market_vol_20 = df['MKT'].rolling(window=20, min_periods=10).std()
        df['vol_diff_20'] = stock_vol_20 - market_vol_20

        return df
    
    def _add_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加相对强弱指标
        
        个股收益 - 市场收益
        """
        # 计算不同周期的累计收益
        for period in [20, 60]:
            # 个股累计收益
            if 'close' in df.columns:
                stock_ret = df['close'].pct_change(periods=period, fill_method=None)
            else:
                stock_ret = df['ret'].rolling(window=period).sum()
            
            # 市场累计收益
            market_ret = df['MKT'].rolling(window=period).sum()
            
            # 相对强弱
            df[f'rel_strength_{period}'] = stock_ret - market_ret
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有市场因子特征的名称列表
        
        Returns:
            特征名称列表
        """
        return [
            'MKT',                    # 市场收益
            'beta_60', 'beta_120',    # 不同窗口Beta
            'alpha_60', 'alpha_120',  # 不同窗口Alpha
            'idio_vol_60', 'idio_vol_120',  # 特质波动率
            'market_R2_60', 'market_R2_120',  # 市场解释力度
            'down_beta_60', 'down_beta_120',  # 下行Beta
            'up_beta_60', 'up_beta_120',      # 上行Beta
            'turnover_ema_20', 'log_turnover_20',
            'turnover_vol_20', 'turnover_z_20', 'turnover_autocorr_5',
            'ret_skew_60', 'ret_kurt_60',
            'momentum_diff_60', 'vol_diff_20',
            'rel_strength_20',        # 20日相对强弱
            'rel_strength_60',        # 60日相对强弱
        ]


def build_market_factors_for_universe(
    symbols: List[str],
    data_access,
    lookback: int = 180,
    as_of_date: Optional[str] = None
) -> pd.DataFrame:
    """
    为股票池批量构建市场因子特征
    
    Args:
        symbols: 股票代码列表
        data_access: 数据访问层对象
        lookback: 回溯天数
        as_of_date: 截止日期 (格式: 'YYYY-MM-DD')
    
    Returns:
        包含市场特征的 DataFrame
    """
    generator = MarketFactorGenerator(lookback_days=lookback)
    
    # Step 1: 获取所有股票数据并构建市场收益
    logger.info(f"正在获取 {len(symbols)} 只股票的数据...")
    
    all_stocks_data = {}
    for symbol in symbols:
        try:
            df = data_access.get_stock_data(
                symbol=symbol,
                start_date=(pd.Timestamp(as_of_date or pd.Timestamp.now()) - pd.Timedelta(days=lookback + 30)).strftime('%Y-%m-%d'),
                end_date=as_of_date
            )
            
            if df is not None and len(df) >= 45:
                df.columns = df.columns.str.lower()
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                if 'ret' not in df.columns and 'close' in df.columns:
                    df['ret'] = df['close'].pct_change()
                
                all_stocks_data[symbol] = df
        except Exception as e:
            logger.debug(f"获取 {symbol} 数据失败: {e}")
            continue
    
    logger.info(f"成功获取 {len(all_stocks_data)} 只股票的数据")
    
    # Step 2: 构建市场收益因子
    market_returns = generator.build_market_returns(all_stocks_data)
    
    if len(market_returns) == 0:
        logger.error("市场因子构建失败")
        return pd.DataFrame()
    
    # Step 3: 为每只股票添加市场特征
    all_features = []
    
    for symbol in symbols:
        if symbol not in all_stocks_data:
            continue
        
        try:
            df = all_stocks_data[symbol].copy()
            df = generator.add_market_features(df, symbol, market_returns)
            
            # 只保留最后一行
            df_last = df.tail(1).copy()
            df_last['symbol'] = symbol
            
            all_features.append(df_last)
        except Exception as e:
            logger.error(f"处理 {symbol} 市场特征时出错: {e}")
            continue
    
    if not all_features:
        return pd.DataFrame()
    
    # 合并所有特征
    result = pd.concat(all_features, ignore_index=False)
    result.reset_index(inplace=True)
    
    return result
