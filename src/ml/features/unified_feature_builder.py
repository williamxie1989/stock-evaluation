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
from src.ml.features.price_volume import PriceVolumeFeatureGenerator
from src.ml.features.market_factors import MarketFactorGenerator
from src.ml.features.industry import IndustryFeatureGenerator, add_industry_features
from src.ml.features.board import BoardFeatureGenerator, add_board_features
from src.ml.features.feature_cache_manager import FeatureCacheManager

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
                 enable_board: bool = ENABLE_BOARD_ONEHOT,
                 enable_cache: bool = True):
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
            enable_cache: 启用特征缓存（默认True）
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
        
        # 🚀 初始化特征缓存管理器（整合L0-L2三层缓存）
        self.cache_manager = FeatureCacheManager(enable_cache=enable_cache)
        
        logger.info(f"UnifiedFeatureBuilder 初始化完成")
        logger.info(f"  - 价量特征: {enable_price_volume}")
        logger.info(f"  - 市场因子: {enable_market}")
        logger.info(f"  - 行业特征: {enable_industry}")
        logger.info(f"  - 板块特征: {enable_board}")
        logger.info(f"  - 特征缓存: {enable_cache}")
    
    def build_features(self, 
                       symbols: List[str],
                       as_of_date: Optional[str] = None,
                       return_labels: bool = False,
                       label_period: int = 30,
                       force_refresh: bool = False,
                       universe_symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        为股票列表构建所有特征（带缓存）
        
        Args:
            symbols: 股票代码列表
            as_of_date: 截止日期 (格式: 'YYYY-MM-DD')，None表示最新
            return_labels: 是否返回标签（用于训练）
            label_period: 标签预测周期（天数）
            force_refresh: 强制刷新，跳过缓存
        
        Returns:
            包含所有特征的 DataFrame，index为symbol
        """
        logger.info(f"开始为 {len(symbols)} 只股票构建特征...")
        
        # 🚀 步骤0: 检查缓存
        if not force_refresh and not return_labels:  # 标签数据不缓存
            feature_config = {
                'enable_price_volume': self.enable_price_volume,
                'enable_market': self.enable_market,
                'enable_industry': self.enable_industry,
                'enable_board': self.enable_board,
                'lookback_days': self.lookback_days
            }
            
            cached_features = self.cache_manager.get(symbols, as_of_date, feature_config)
            if cached_features is not None:
                logger.info(f"✅ 缓存命中！跳过特征计算 ({len(cached_features)}行 x {len(cached_features.columns)}列)")
                return cached_features
        
        # 🔄 统一加载股票历史数据（避免重复 IO）
        history_symbols = set(symbols)
        if universe_symbols:
            history_symbols.update(universe_symbols)
        history_data = self._load_stock_history(sorted(history_symbols), as_of_date)

        # Step 1: 构建价量特征
        if self.enable_price_volume:
            logger.info("构建价量特征...")
            df_pv = self._build_price_volume_features(symbols, as_of_date, history_data)
        else:
            df_pv = pd.DataFrame({'symbol': symbols})
        
        # Step 2: 构建市场因子特征
        if self.enable_market:
            logger.info("构建市场因子...")
            market_universe = universe_symbols or symbols
            df_market = self._build_market_features(symbols, as_of_date, history_data, market_universe)
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
        
        logger.info(f"✅ 特征构建完成: {len(df_pv)} 行 x {len(df_pv.columns)} 列")
        
        # 🚀 保存到缓存（不缓存标签数据）
        if not return_labels:
            feature_config = {
                'enable_price_volume': self.enable_price_volume,
                'enable_market': self.enable_market,
                'enable_industry': self.enable_industry,
                'enable_board': self.enable_board,
                'lookback_days': self.lookback_days
            }
            self.cache_manager.set(symbols, as_of_date, feature_config, df_pv)
        
        return df_pv
    
    def build_features_from_dataframe(self,
                                     stock_data: pd.DataFrame,
                                     symbol: str) -> pd.DataFrame:
        """
        从已有DataFrame构建特征（用于training场景）
        
        这个方法支持training script的用法，允许调用者：
        1. 自己控制数据获取过程（可以选择adjust_mode）
        2. 在特征构建前进行数据质量过滤
        3. 分别处理特征数据和标签数据
        
        Args:
            stock_data: 包含OHLCV数据的DataFrame（date可能在索引或列中）
            symbol: 股票代码
        
        Returns:
            包含价量特征的DataFrame（date作为列，用于后续标签合并）
        """
        try:
            # 复制数据避免修改原始数据
            df = stock_data.copy()
            
            # 🔧 关键修复1：首先保存原始日期数据
            original_dates = None
            if df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                # date在索引中
                original_dates = df.index.copy()
                logger.debug(f"{symbol}: date在索引中，已保存 (name={df.index.name}, len={len(original_dates)})")
            elif 'date' in df.columns:
                # date在列中
                original_dates = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logger.debug(f"{symbol}: date在列中，已转为索引")
            elif 'Date' in df.columns:
                # Date在列中（大写）
                original_dates = pd.to_datetime(df['Date'])
                df.drop('Date', axis=1, inplace=True)
                df.index = original_dates
                logger.debug(f"{symbol}: Date在列中，已转为索引")
            else:
                logger.warning(f"{symbol}: 无法找到日期数据，index.name={df.index.name}, is_DatetimeIndex={isinstance(df.index, pd.DatetimeIndex)}, columns={list(df.columns[:5])}")
            
            # 标准化列名（不影响索引）
            df.columns = df.columns.str.lower()
            
            # 🔧 确保数值列为正确类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 确保索引是DatetimeIndex
            if original_dates is not None and not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(original_dates)
                df.index.name = 'date'
            
            # 生成价量特征
            if not self.enable_price_volume:
                df_features = pd.DataFrame(index=df.index)  # 空DataFrame，保留索引
            else:
                df_features = self.pv_generator.generate_features(df)
            
            # 🔧 关键修复2：将date从索引转为列（用于标签合并）
            if df_features.index.name == 'date' or isinstance(df_features.index, pd.DatetimeIndex):
                df_features.reset_index(inplace=True)
                # reset_index后，如果原索引名是'date'，会自动创建名为'date'的列
                # 如果没有名字，会创建名为'index'的列
                if 'index' in df_features.columns and 'date' not in df_features.columns:
                    df_features.rename(columns={'index': 'date'}, inplace=True)
                logger.debug(f"{symbol}: date已从索引转为列，列数={len(df_features.columns)}")
            elif original_dates is not None:
                # 如果特征生成后丢失了索引，手动添加date列
                df_features['date'] = original_dates[:len(df_features)]
                logger.debug(f"{symbol}: 手动添加date列")
            
            # 添加行业特征（如果启用）
            if self.enable_industry:
                # 创建临时DataFrame用于添加行业特征
                temp_df = df_features.tail(1).copy()
                temp_df['symbol'] = symbol
                temp_df = add_industry_features(temp_df, self.db_manager, merge_low_freq=True)
                # 去掉symbol列后合并回来
                industry_cols = [col for col in temp_df.columns if col not in df_features.columns and col != 'symbol']
                for col in industry_cols:
                    df_features[col] = temp_df[col].iloc[0]
            
            # 注意：市场因子和板块特征需要多只股票，在单股票场景下跳过
            # 这些特征应该在prepare_training_data的批处理阶段添加
            
            # 🔧 最终验证：确保date列存在
            if 'date' not in df_features.columns:
                logger.error(f"{symbol}: date列在最终结果中丢失！列={list(df_features.columns[:10])}")
                # 如果original_dates存在，强制添加
                if original_dates is not None:
                    logger.warning(f"{symbol}: 强制添加date列")
                    if len(original_dates) == len(df_features):
                        df_features.insert(0, 'date', original_dates)
                    else:
                        df_features.insert(0, 'date', original_dates[:len(df_features)])
            else:
                logger.debug(f"{symbol}: ✅ date列验证通过")
            
            return df_features
            
        except Exception as e:
            logger.error(f"{symbol}: build_features_from_dataframe失败 - {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _load_stock_history(self,
                             symbols: List[str],
                             as_of_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """批量加载股票历史数据并标准化"""
        if not symbols:
            return {}

        end_dt = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
        start_dt = end_dt - timedelta(days=self.lookback_days + 60)

        history: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                df = self.data_access.get_stock_data(
                    symbol=symbol,
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d')
                )

                if df is None or len(df) < MIN_HISTORY_DAYS:
                    logger.debug(f"{symbol}: 历史数据不足，len={len(df) if df is not None else 0}")
                    continue

                df = df.copy()
                df.columns = df.columns.str.lower()

                # 设置日期索引
                if 'date' in df.columns and df.index.name != 'date':
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')

                df.sort_index(inplace=True)

                # 仅保留 lookback_days + 缓冲
                df = df[df.index >= start_dt]

                # 数值列转型
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if 'close' in df.columns:
                    df['ret'] = df['close'].pct_change(fill_method=None)

                history[symbol] = df

            except Exception as exc:
                logger.warning(f"{symbol}: 加载历史数据失败 - {exc}")
                continue

        return history

    def _build_price_volume_features(self,
                                      symbols: List[str],
                                      as_of_date: Optional[str],
                                      history_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建价量特征"""
        all_features = []

        for symbol in symbols:
            df = history_data.get(symbol)
            if df is None or len(df) < MIN_HISTORY_DAYS:
                logger.warning(f"{symbol}: 缺失价量特征所需历史数据，跳过")
                continue

            try:
                df_local = df.copy()

                if as_of_date:
                    df_local = df_local[df_local.index <= pd.Timestamp(as_of_date)]

                df_local = df_local.tail(self.lookback_days + 5)

                df_features = self.pv_generator.generate_features(df_local)

                df_last = df_features.tail(1).copy()
                df_last['symbol'] = symbol

                all_features.append(df_last)
            except Exception as exc:
                logger.error(f"{symbol}: 价量特征构建失败 - {exc}")
                continue

        if not all_features:
            return pd.DataFrame()

        result = pd.concat(all_features, ignore_index=True)

        if 'date' in result.index.names:
            result.reset_index(inplace=True)

        return result

    def _build_market_features(self,
                               symbols: List[str],
                               as_of_date: Optional[str],
                               history_data: Dict[str, pd.DataFrame],
                               universe_symbols: List[str]) -> pd.DataFrame:
        """构建市场因子特征"""
        try:
            # 🔧 修复：确保end_date不为None
            end_date = as_of_date if as_of_date else datetime.now().strftime('%Y-%m-%d')
            
            # 获取所有股票数据用于构建市场收益
            logger.info("获取股票池数据用于构建市场因子...")
            all_stocks_data: Dict[str, pd.DataFrame] = {}

            for symbol in universe_symbols:
                df = history_data.get(symbol)
                if df is None or len(df) < MIN_HISTORY_DAYS:
                    continue

                df_local = df.copy()
                df_local = df_local[df_local.index <= pd.Timestamp(end_date)]

                if 'ret' not in df_local.columns and 'close' in df_local.columns:
                    df_local['ret'] = df_local['close'].pct_change(fill_method=None)

                all_stocks_data[symbol] = df_local
            
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
        
        # 移除原始OHLCV数据列（这些不是特征，只是用于计算的原始数据）
        raw_data_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                        'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                        'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
                        'source']
        cols_to_drop = [col for col in raw_data_cols if col in df.columns]
        if cols_to_drop:
            logger.info(f"移除原始数据列: {cols_to_drop}")
            df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
        
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
