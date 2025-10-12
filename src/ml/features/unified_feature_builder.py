# -*- coding: utf-8 -*-
"""
统一特征构建器
整合价量、市场、行业、板块等所有特征的构建逻辑
确保训练和预测使用完全一致的特征
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta

# 导入配置
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import (
    LOOKBACK_DAYS,
    MIN_HISTORY_DAYS,
    ENABLE_PRICE_VOLUME_FEATURES,
    ENABLE_MARKET_FACTOR,
    ENABLE_INDUSTRY_FEATURES,
    ENABLE_BOARD_ONEHOT,
    MIN_STOCKS_FOR_MARKET,
    INDUSTRY_MIN_FREQUENCY
)

# 导入特征生成器
from .price_volume import PriceVolumeFeatureGenerator, build_price_volume_features
from .market_factors import MarketFactorGenerator, build_market_factors_for_universe
from .industry import IndustryFeatureGenerator, add_industry_features
from .board import BoardFeatureGenerator, add_board_features

logger = logging.getLogger(__name__)


class UnifiedFeatureBuilder:
    """
    统一特征构建器
    
    职责：
    1. 整合所有特征模块（价量、市场、行业、板块）
    2. 确保训练和预测使用相同的特征构建逻辑
    3. 提供特征列表和元数据
    """
    
    def __init__(self, 
                 data_access,
                 db_manager,
                 lookback_days: int = LOOKBACK_DAYS,
                 enable_price_volume: bool = ENABLE_PRICE_VOLUME_FEATURES,
                 enable_market: bool = ENABLE_MARKET_FACTOR,
                 enable_industry: bool = ENABLE_INDUSTRY_FEATURES,
                 enable_board: bool = ENABLE_BOARD_ONEHOT):
        """
        初始化统一特征构建器
        
        Args:
            data_access: 数据访问层对象
            db_manager: 数据库管理器对象
            lookback_days: 回溯天数
            enable_price_volume: 启用价量特征
            enable_market: 启用市场因子
            enable_industry: 启用行业特征
            enable_board: 启用板块特征
        """
        self.data_access = data_access
        self.db_manager = db_manager
        self.lookback_days = lookback_days
        
        # 特征开关
        self.enable_price_volume = enable_price_volume
        self.enable_market = enable_market
        self.enable_industry = enable_industry
        self.enable_board = enable_board
        
        # 初始化各特征生成器
        self.pv_generator = PriceVolumeFeatureGenerator(lookback_days) if enable_price_volume else None
        self.market_generator = MarketFactorGenerator(lookback_days) if enable_market else None
        self.industry_generator = IndustryFeatureGenerator(db_manager, min_frequency=INDUSTRY_MIN_FREQUENCY) if enable_industry else None
        self.board_generator = BoardFeatureGenerator() if enable_board else None
        
        logger.info(f"UnifiedFeatureBuilder 初始化完成")
        logger.info(f"  - 价量特征: {enable_price_volume}")
        logger.info(f"  - 市场因子: {enable_market}")
        logger.info(f"  - 行业特征: {enable_industry}")
        logger.info(f"  - 板块特征: {enable_board}")
    
    def build_features(self, 
                       symbols: List[str],
                       as_of_date: Optional[str] = None,
                       return_labels: bool = False,
                       label_period: int = 30) -> pd.DataFrame:
        """
        为股票列表构建所有特征
        
        Args:
            symbols: 股票代码列表
            as_of_date: 截止日期 (格式: 'YYYY-MM-DD')，None表示最新
            return_labels: 是否返回标签（用于训练）
            label_period: 标签预测周期（天数）
        
        Returns:
            包含所有特征的 DataFrame，index为symbol
        """
        logger.info(f"开始为 {len(symbols)} 只股票构建特征...")
        
        # Step 1: 构建价量特征
        if self.enable_price_volume:
            logger.info("构建价量特征...")
            df_pv = self._build_price_volume_features(symbols, as_of_date)
        else:
            df_pv = pd.DataFrame({'symbol': symbols})
        
        # Step 2: 构建市场因子特征
        if self.enable_market:
            logger.info("构建市场因子...")
            df_market = self._build_market_features(symbols, as_of_date)
            # 合并
            if len(df_market) > 0:
                df_pv = df_pv.merge(df_market, on='symbol', how='left', suffixes=('', '_market'))
        
        # Step 3: 添加行业特征
        if self.enable_industry:
            logger.info("添加行业特征...")
            df_pv = add_industry_features(df_pv, self.db_manager, merge_low_freq=True)
        
        # Step 4: 添加板块特征
        if self.enable_board:
            logger.info("添加板块特征...")
            df_pv = add_board_features(df_pv, symbol_col='symbol')
        
        # Step 5: 构建标签（如果需要）
        if return_labels:
            logger.info(f"构建 {label_period} 天预测标签...")
            df_pv = self._add_labels(df_pv, label_period, as_of_date)
        
        # 清理和验证
        df_pv = self._clean_features(df_pv)
        
        logger.info(f"特征构建完成: {len(df_pv)} 行 x {len(df_pv.columns)} 列")
        
        return df_pv
    
    def _build_price_volume_features(self, symbols: List[str], as_of_date: Optional[str]) -> pd.DataFrame:
        """构建价量特征"""
        all_features = []
        
        for symbol in symbols:
            try:
                # 获取历史数据
                start_date = (pd.Timestamp(as_of_date or datetime.now()) - timedelta(days=self.lookback_days + 30)).strftime('%Y-%m-%d')
                df = self.data_access.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=as_of_date
                )
                
                if df is None or len(df) < MIN_HISTORY_DAYS:
                    logger.warning(f"{symbol}: 数据不足 (< {MIN_HISTORY_DAYS} 天)")
                    continue
                
                # 标准化列名
                df.columns = df.columns.str.lower()
                
                # 设置日期索引
                if 'date' in df.columns and df.index.name != 'date':
                    df.set_index('date', inplace=True)
                
                # 生成价量特征
                df_features = self.pv_generator.generate_features(df)
                
                # 只保留最后一行
                df_last = df_features.tail(1).copy()
                df_last['symbol'] = symbol
                
                all_features.append(df_last)
                
            except Exception as e:
                logger.error(f"{symbol}: 价量特征构建失败 - {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        result = pd.concat(all_features, ignore_index=True)
        
        # 重置索引，移除date列（保留为普通列）
        if 'date' in result.index.names:
            result.reset_index(inplace=True)
        
        return result
    
    def _build_market_features(self, symbols: List[str], as_of_date: Optional[str]) -> pd.DataFrame:
        """构建市场因子特征"""
        try:
            # 获取所有股票数据用于构建市场收益
            logger.info("获取股票池数据用于构建市场因子...")
            all_stocks_data = {}
            
            start_date = (pd.Timestamp(as_of_date or datetime.now()) - timedelta(days=self.lookback_days + 30)).strftime('%Y-%m-%d')
            
            for symbol in symbols:
                try:
                    df = self.data_access.get_stock_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=as_of_date
                    )
                    
                    if df is not None and len(df) >= MIN_HISTORY_DAYS:
                        df.columns = df.columns.str.lower()
                        if 'date' in df.columns:
                            df.set_index('date', inplace=True)
                        if 'ret' not in df.columns and 'close' in df.columns:
                            df['ret'] = df['close'].pct_change()
                        
                        all_stocks_data[symbol] = df
                except:
                    continue
            
            if len(all_stocks_data) < MIN_STOCKS_FOR_MARKET:
                logger.warning(f"股票数量不足以构建市场因子 ({len(all_stocks_data)} < {MIN_STOCKS_FOR_MARKET})")
                return pd.DataFrame()
            
            # 构建市场收益
            market_returns = self.market_generator.build_market_returns(all_stocks_data, min_stocks=MIN_STOCKS_FOR_MARKET)
            
            if len(market_returns) == 0:
                return pd.DataFrame()
            
            # 为每只股票添加市场特征
            all_features = []
            
            for symbol in symbols:
                if symbol not in all_stocks_data:
                    continue
                
                try:
                    df = all_stocks_data[symbol].copy()
                    df = self.market_generator.add_market_features(df, symbol, market_returns)
                    
                    # 只保留最后一行
                    df_last = df.tail(1).copy()
                    df_last['symbol'] = symbol
                    
                    all_features.append(df_last)
                except Exception as e:
                    logger.error(f"{symbol}: 市场特征添加失败 - {e}")
                    continue
            
            if not all_features:
                return pd.DataFrame()
            
            result = pd.concat(all_features, ignore_index=True)
            
            # 重置索引
            if 'date' in result.index.names:
                result.reset_index(inplace=True)
            
            # 移除date列（避免重复）
            if 'date' in result.columns:
                result.drop('date', axis=1, inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"市场因子构建失败: {e}")
            return pd.DataFrame()
    
    def _add_labels(self, df: pd.DataFrame, period: int, as_of_date: Optional[str]) -> pd.DataFrame:
        """添加预测标签"""
        # 计算未来收益
        future_returns = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            try:
                # 获取未来数据
                start_date = as_of_date or datetime.now().strftime('%Y-%m-%d')
                end_date = (pd.Timestamp(start_date) + timedelta(days=period + 10)).strftime('%Y-%m-%d')
                
                df_future = self.data_access.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df_future is None or len(df_future) < period // 2:
                    future_returns.append(np.nan)
                    continue
                
                # 确保列名
                df_future.columns = df_future.columns.str.lower()
                
                # 计算收益率
                if len(df_future) >= 2:
                    ret = (df_future['close'].iloc[-1] - df_future['close'].iloc[0]) / df_future['close'].iloc[0]
                    future_returns.append(ret)
                else:
                    future_returns.append(np.nan)
                    
            except Exception as e:
                logger.debug(f"{symbol}: 获取未来数据失败 - {e}")
                future_returns.append(np.nan)
        
        df[f'ret_{period}d'] = future_returns
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征"""
        # 移除完全重复的列
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 移除日期相关列（除了必要的）
        date_cols = [col for col in df.columns if 'date' in col.lower() and col != 'symbol']
        if date_cols:
            df.drop(date_cols, axis=1, inplace=True, errors='ignore')
        
        # 替换inf为nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df
    
    def get_numerical_features(self) -> List[str]:
        """获取数值特征列表"""
        features = []
        
        if self.enable_price_volume and self.pv_generator:
            features.extend(self.pv_generator.get_feature_names())
        
        if self.enable_market and self.market_generator:
            features.extend(self.market_generator.get_feature_names())
        
        return features
    
    def get_categorical_features(self) -> List[str]:
        """获取类别特征列表"""
        features = []
        
        if self.enable_industry:
            features.append('industry')
        
        if self.enable_board:
            features.append('board')
        
        return features
    
    def get_all_features(self) -> Dict[str, List[str]]:
        """获取所有特征的分类信息"""
        return {
            'numerical': self.get_numerical_features(),
            'categorical': self.get_categorical_features()
        }
