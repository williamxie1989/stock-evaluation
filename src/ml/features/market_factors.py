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
        # 提取所有股票的收益率
        all_rets = []
        
        for symbol, df in all_stocks_data.items():
            if df is None or len(df) == 0:
                continue
            
            # 确保有收益率列
            if 'ret' not in df.columns:
                if 'close' in df.columns:
                    df['ret'] = df['close'].pct_change()
                else:
                    continue
            
            ret_df = df[['ret']].copy()
            ret_df.columns = [symbol]
            all_rets.append(ret_df)
        
        if not all_rets:
            logger.error("没有有效的股票数据用于构建市场因子")
            return pd.DataFrame()
        
        # 合并所有收益率
        combined = pd.concat(all_rets, axis=1)
        
        # 过滤异常值 (收益率超过 ±50% 视为异常)
        combined = combined.clip(-0.5, 0.5)
        
        # 计算等权平均市场收益
        market_df = pd.DataFrame()
        market_df['MKT'] = combined.mean(axis=1, skipna=True)
        market_df['count'] = combined.count(axis=1)
        
        # 过滤样本量不足的日期
        market_df = market_df[market_df['count'] >= min_stocks].copy()
        
        # 缓存市场收益数据
        self.market_returns = market_df
        
        logger.info(f"市场因子构建完成: {len(market_df)} 天, 平均 {market_df['count'].mean():.0f} 只股票")
        
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
        
        # 合并市场收益
        df = df.join(market_returns[['MKT']], how='left')
        
        # 填充缺失的市场收益 (用0填充)
        df['MKT'].fillna(0, inplace=True)
        
        # 计算 Beta 和特质波动率 (60日滚动回归)
        df = self._add_beta_features(df, window=60)
        
        # 计算相对强弱指标
        df = self._add_relative_strength(df)
        
        return df
    
    def _add_beta_features(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        添加 Beta 和特质波动率特征
        
        使用滚动窗口回归: ret_i = alpha + beta * MKT + epsilon
        """
        betas = []
        idio_vols = []
        alphas = []
        r_squareds = []
        
        for i in range(len(df)):
            if i < window - 1:
                betas.append(np.nan)
                idio_vols.append(np.nan)
                alphas.append(np.nan)
                r_squareds.append(np.nan)
            else:
                # 提取窗口数据
                y = df['ret'].iloc[i - window + 1:i + 1].values
                x = df['MKT'].iloc[i - window + 1:i + 1].values
                
                # 过滤 NaN
                valid_mask = ~(np.isnan(y) | np.isnan(x))
                
                if valid_mask.sum() < window // 2:  # 至少需要一半的数据
                    betas.append(np.nan)
                    idio_vols.append(np.nan)
                    alphas.append(np.nan)
                    r_squareds.append(np.nan)
                    continue
                
                y_valid = y[valid_mask]
                x_valid = x[valid_mask]
                
                # 线性回归
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                    
                    # 计算残差的标准差 (特质波动率)
                    y_pred = intercept + slope * x_valid
                    residuals = y_valid - y_pred
                    idio_vol = np.std(residuals, ddof=1)
                    
                    betas.append(slope)
                    alphas.append(intercept)
                    idio_vols.append(idio_vol)
                    r_squareds.append(r_value ** 2)
                except:
                    betas.append(np.nan)
                    idio_vols.append(np.nan)
                    alphas.append(np.nan)
                    r_squareds.append(np.nan)
        
        df['beta_60'] = betas
        df['alpha_60'] = alphas
        df['idio_vol_60'] = idio_vols
        df['market_R2_60'] = r_squareds
        
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
                stock_ret = df['close'].pct_change(periods=period)
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
            'beta_60',                # 60日Beta
            'alpha_60',               # 60日Alpha
            'idio_vol_60',            # 特质波动率
            'market_R2_60',           # 市场解释力度
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
