"""
统一数据访问层 - 整合数据库操作和统一数据提供者
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import threading

# 使用绝对导入
from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.data.providers.unified_data_provider import UnifiedDataProvider
from src.data.providers.data_source_validator import DataSourceValidator
from src.data.db.symbol_standardizer import standardize_symbol, get_symbol_standardizer

logger = logging.getLogger(__name__)


@dataclass
class DataAccessConfig:
    """数据访问配置"""
    use_cache: bool = True
    cache_ttl: int = 300  # 5分钟
    quality_threshold: float = 0.7
    auto_sync: bool = True
    sync_batch_size: int = 50
    sync_delay: float = 0.5
    max_sync_retries: int = 3
    enable_data_validation: bool = True
    preferred_data_sources: List[str] = None
    
    def __post_init__(self):
        if self.preferred_data_sources is None:
            self.preferred_data_sources = ['eastmoney', 'tencent', 'optimized_enhanced', 'enhanced_realtime', 'akshare', 'netease']


class UnifiedDataAccessLayer:
    """统一数据访问层 - 整合数据库和外部数据源"""

    def clear_all_caches(self) -> None:
        """清空内部缓存，用于测试或重置"""
        try:
            # 清理统一 provider 缓存
            if hasattr(self, "unified_provider") and hasattr(self.unified_provider, "cache"):
                self.unified_provider.cache.clear()
            # 清理数据库管理器缓存（如果有）
            if hasattr(self.db_manager, "cache"):
                self.db_manager.cache.clear()
            # 重置内部统计
            self._cache_stats = {'hits': 0, 'misses': 0}
        except Exception:
            pass
    
    def __init__(self, db_manager: Optional[UnifiedDatabaseManager] = None, 
                 config: Optional[DataAccessConfig] = None):
        """
        初始化统一数据访问层
        
        Args:
            db_manager: 数据库管理器
            config: 数据访问配置
        """
        self.db_manager = db_manager or UnifiedDatabaseManager()
        self.config = config or DataAccessConfig()
        self.unified_provider = UnifiedDataProvider(
            cache_ttl=self.config.cache_ttl
        )
        self.data_validator = DataSourceValidator(self.unified_provider)
        self._sync_lock = threading.Lock()
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info("UnifiedDataAccessLayer initialized")
    
    def initialize_data_sources(self, validate_sources: bool = True) -> Dict[str, Any]:
        """
        初始化数据源
        
        Args:
            validate_sources: 是否验证数据源
            
        Returns:
            初始化结果
        """
        logger.info("Initializing data sources...")
        
        try:
            # 导入并初始化各种数据提供者
            from src.data.providers.akshare_provider import AkshareDataProvider
            from src.data.providers.enhanced_realtime_provider import EnhancedRealtimeProvider
            from src.data.providers.optimized_enhanced_data_provider import OptimizedEnhancedDataProvider
            from src.data.providers.domestic.eastmoney_provider import EastmoneyDataProvider
            from src.data.providers.domestic.tencent_provider import TencentDataProvider
            from src.data.providers.domestic.netease_provider import NeteaseDataProvider
            
            providers = {
                'akshare': AkshareDataProvider(),
                'enhanced_realtime': EnhancedRealtimeProvider(),
                'optimized_enhanced': OptimizedEnhancedDataProvider(),
                'eastmoney': EastmoneyDataProvider(),
                'tencent': TencentDataProvider(),
                'netease': NeteaseDataProvider()
            }
            
            initialization_results = {}
            
            # 根据配置的首选顺序添加数据提供者
            for source_name in self.config.preferred_data_sources:
                if source_name in providers:
                    provider = providers[source_name]
                    
                    if validate_sources and self.config.enable_data_validation:
                        # 验证数据源
                        validation_result = self.unified_provider.validate_data_source(provider)
                        initialization_results[source_name] = validation_result
                        
                        # 根据验证结果决定是否添加
                        if validation_result['overall_score'] >= self.config.quality_threshold:
                            self.unified_provider.add_primary_provider(provider)
                            logger.info(f"Added {source_name} as primary provider (score: {validation_result['overall_score']:.2f})")
                        else:
                            self.unified_provider.add_fallback_provider(provider)
                            logger.warning(f"Added {source_name} as fallback provider (score: {validation_result['overall_score']:.2f})")
                    else:
                        # 不验证直接添加为主要提供者
                        self.unified_provider.add_primary_provider(provider)
                        initialization_results[source_name] = {
                            'provider': source_name,
                            'overall_score': 1.0,
                            'recommendation': 'added_without_validation'
                        }
                        logger.info(f"Added {source_name} as primary provider (validation skipped)")
            
            logger.info("Data sources initialization completed")
            return initialization_results
            
        except Exception as e:
            logger.error(f"Failed to initialize data sources: {e}")
            raise
    
    async def get_historical_data(self, symbol: str, start_date: Union[str, datetime], 
                                 end_date: Union[str, datetime],
                                 force_refresh: bool = False, auto_sync: bool = None) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据 - 添加防循环和快速失败机制
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 强制刷新，不从数据库读取
            auto_sync: 是否自动同步缺失数据
            
        Returns:
            股票历史数据
        """
        try:
            # 标准化股票代码
            standardized_symbol = standardize_symbol(symbol)
            logger.info(f"Standardized symbol: {symbol} -> {standardized_symbol}")
            
            # 检查是否已经在处理中，防止循环调用
            processing_key = f"processing_{standardized_symbol}_{start_date}_{end_date}"
            if hasattr(self, '_processing_symbols'):
                if self._processing_symbols.get(processing_key, False):
                    logger.warning(f"Historical data request already in progress for {standardized_symbol} {start_date} to {end_date}, skipping to prevent loop")
                    return None
            else:
                self._processing_symbols = {}
            
            # 标记处理进行中
            self._processing_symbols[processing_key] = True
            
            try:
                # 设置自动同步标志
                if auto_sync is None:
                    auto_sync = self.config.auto_sync
                
                # 转换日期格式
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, '%Y-%m-%d')
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, '%Y-%m-%d')
                
                # 首先尝试从数据库获取
                if not force_refresh:
                    db_data = await self.db_manager.get_stock_data(standardized_symbol, start_date, end_date)
                    if db_data is not None and not db_data.empty:
                        # 检查数据完整性
                        if self._is_data_complete(db_data, start_date, end_date):
                            logger.info(f"Retrieved complete data for {standardized_symbol} from database")
                            return db_data
                        else:
                            # 检查同步次数，避免无限同步
                            sync_key = f"sync_count_{standardized_symbol}"
                            if not hasattr(self, '_sync_counts'):
                                self._sync_counts = {}
                            
                            sync_count = self._sync_counts.get(sync_key, 0)
                            if sync_count >= 2:  # 最多同步2次
                                logger.warning(f"Maximum sync attempts ({sync_count}) reached for {standardized_symbol}, returning existing data")
                                return db_data
                            
                            self._sync_counts[sync_key] = sync_count + 1
                            logger.info(f"Database data incomplete for {standardized_symbol}, syncing missing data... (attempt {sync_count + 1}/2)")
                            if auto_sync:
                                return await self._sync_and_return_data(standardized_symbol, start_date, end_date, db_data)
                
                # 从外部数据源获取（注意：这是同步方法，不需要await）
                logger.info(f"Fetching data for {standardized_symbol} from external sources")
                
                # 转换日期格式为字符串，以适配数据提供者
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                external_data = self.unified_provider.get_historical_data(
                    standardized_symbol, start_date_str, end_date_str, 
                    quality_threshold=self.config.quality_threshold
                )
                
                if external_data is not None and not external_data.empty:
                    # 保存到数据库
                    await self.db_manager.save_stock_data(standardized_symbol, external_data)
                    logger.info(f"Saved external data for {standardized_symbol} to database")
                    return external_data
                
                logger.warning(f"Failed to get data for {standardized_symbol} from all sources")
                return None
                
            finally:
                # 清除处理标记
                self._processing_symbols[processing_key] = False
                
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    async def get_realtime_data(self, symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        获取实时数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时数据字典
        """
        try:
            if not symbols:
                return {}
            
            # 标准化股票代码列表
            standardized_symbols = [standardize_symbol(symbol) for symbol in symbols]
            logger.info(f"Standardized symbols: {symbols} -> {standardized_symbols}")
            
            # 从统一数据提供者获取实时数据（注意：这是同步方法，不需要await）
            realtime_data = self.unified_provider.get_realtime_data(standardized_symbols)
            
            if realtime_data:
                # 保存实时数据到数据库
                await self.db_manager.save_realtime_data(realtime_data)
                logger.info(f"Retrieved and saved realtime data for {len(realtime_data)} symbols")
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Error getting realtime data: {e}")
            return None
    
    def _convert_stock_list_to_dataframe(self, stock_list: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        将股票列表转换为DataFrame格式
        
        Args:
            stock_list: 股票列表（List[Dict]）
            
        Returns:
            股票列表DataFrame，格式与akshare_provider一致
        """
        if not stock_list:
            logger.warning("No stocks to convert to DataFrame")
            return None
        
        try:
            # 转换为DataFrame格式，与akshare_provider保持一致
            df_data = []
            for stock in stock_list:
                # 根据symbol提取市场信息
                symbol = stock.get('symbol', '')
                if '.SH' in symbol or '.SS' in symbol:
                    market_code = 'SH'
                elif '.SZ' in symbol:
                    market_code = 'SZ'
                else:
                    market_code = stock.get('market', 'UNKNOWN')
                
                # 提取股票代码（去掉后缀）
                code = symbol.replace('.SH', '').replace('.SS', '').replace('.SZ', '') if symbol else stock.get('code', '')
                
                # 生成带后缀的标准化 symbol
                try:
                    if symbol:
                        std_symbol = standardize_symbol(symbol)
                    else:
                        # 如果原始数据只有 code，没有 symbol 字段，根据市场信息补全后缀
                        tentative_symbol = f"{code}.{market_code}" if market_code in ['SH', 'SZ'] else code
                        std_symbol = standardize_symbol(tentative_symbol)
                except Exception:
                    # 如果标准化失败，则回退使用原始 symbol 或 code
                    std_symbol = symbol if symbol else code

                df_data.append({
                    'code': code,
                    'symbol': std_symbol,
                    'name': stock.get('name', ''),
                    'market': market_code,
                    'board_type': stock.get('board_type', '')
                })
            
            result_df = pd.DataFrame(df_data)
            logger.info(f"Converted {len(result_df)} stocks to DataFrame format")
            return result_df
            
        except Exception as e:
            logger.error(f"Error converting stock list to DataFrame: {e}")
            return None

    def get_all_stock_list(self, market: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取全市场股票列表（同步方法）
        
        Args:
            market: 市场代码 (可选)
            
        Returns:
            股票列表DataFrame，格式与akshare_provider一致
        """
        try:
            # 使用异步的get_stock_list方法，但同步调用
            import asyncio
            import threading
            
            # 检查当前是否已经有事件循环在运行
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环已经在运行，我们不能使用run_until_complete
                    # 使用线程来运行异步方法
                    result = None
                    
                    def run_async_in_thread():
                        nonlocal result
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            stock_list = new_loop.run_until_complete(self.get_stock_list(market=market))
                            # 确保转换为DataFrame格式
                            result = self._convert_stock_list_to_dataframe(stock_list)
                        finally:
                            new_loop.close()
                    
                    thread = threading.Thread(target=run_async_in_thread)
                    thread.start()
                    thread.join()
                    return result
            except RuntimeError:
                pass
            
            # 如果没有事件循环或事件循环未运行，直接使用当前循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 运行异步方法
            stock_list = loop.run_until_complete(self.get_stock_list(market=market))
            return self._convert_stock_list_to_dataframe(stock_list)
                
        except Exception as e:
            logger.error(f"Error getting all stock list: {e}")
            return None
    
    def get_stock_data(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime], force_refresh: bool = False, auto_sync: bool = None) -> Optional[pd.DataFrame]:
        """同步获取股票历史数据，包装异步 get_historical_data 以兼容旧接口"""
        try:
            # 标准化股票代码
            standardized_symbol = standardize_symbol(symbol)
            logger.info(f"Standardized symbol: {symbol} -> {standardized_symbol}")
            
            import asyncio
            import threading
            result: Optional[pd.DataFrame] = None

            def run_async_in_thread():
                nonlocal result
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    coro = self.get_historical_data(standardized_symbol, start_date, end_date, force_refresh=force_refresh, auto_sync=auto_sync)
                    result = new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            # 如果当前已有事件循环在运行，则在新线程中执行
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    thread = threading.Thread(target=run_async_in_thread)
                    thread.start()
                    thread.join()
                    return result
            except RuntimeError:
                # 当前线程没有事件循环
                pass

            # 无事件循环，直接新建并运行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            coro = self.get_historical_data(standardized_symbol, start_date, end_date, force_refresh=force_refresh, auto_sync=auto_sync)
            result = loop.run_until_complete(coro)
            return result

        except Exception as e:
            logger.error(f"Error getting stock data synchronously for {symbol}: {e}")
            return None

    async def get_stock_list(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取股票列表（异步方法）
        
        Args:
            market: 市场代码 (可选)
            
        Returns:
            股票列表
        """
        try:
            # 首先尝试从数据库获取
            db_stocks = await self.db_manager.get_stock_list(market=market)
            
            if db_stocks:
                logger.info(f"Retrieved {len(db_stocks)} stocks from database")
                return db_stocks
            
            # 从外部数据源获取
            external_stocks = await self.unified_provider.get_all_stock_list(market=market)
            
            if external_stocks:
                # 保存到数据库
                await self.db_manager.upsert_stocks(external_stocks)
                logger.info(f"Saved {len(external_stocks)} stocks to database")
                return external_stocks
            
            logger.warning("Failed to get stock list from all sources")
            return []
            
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return []
    
    async def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        保存股票数据到数据库
        
        Args:
            symbol: 股票代码
            data: 股票数据DataFrame
            
        Returns:
            是否成功
        """
        try:
            return await self.db_manager.save_stock_data(symbol, data)
        except Exception as e:
            logger.error(f"Error saving stock data for {symbol}: {e}")
            return False
    
    async def upsert_stocks(self, stocks: List[Dict[str, Any]]) -> bool:
        """
        更新或插入股票列表
        
        Args:
            stocks: 股票列表
            
        Returns:
            是否成功
        """
        try:
            return await self.db_manager.upsert_stocks(stocks)
        except Exception as e:
            logger.error(f"Error upserting stocks: {e}")
            return False
    
    async def sync_market_data(self, symbols: Optional[List[str]] = None,
                              batch_size: Optional[int] = None,
                              delay: Optional[float] = None) -> Dict[str, Any]:
        """
        同步市场数据
        
        Args:
            symbols: 股票代码列表 (可选，None表示全市场)
            batch_size: 批次大小
            delay: 延迟时间
            
        Returns:
            同步结果
        """
        try:
            # 使用配置参数
            batch_size = batch_size or self.config.sync_batch_size
            delay = delay or self.config.sync_delay
            
            with self._sync_lock:
                logger.info(f"Starting market data sync for {len(symbols) if symbols else 'all'} symbols")
                
                # 获取股票列表
                if symbols is None:
                    symbols = [stock['symbol'] for stock in await self.get_stock_list()]
                
                if not symbols:
                    return {'status': 'no_symbols', 'synced': 0, 'errors': 0}
                
                # 分批同步
                total_synced = 0
                total_errors = 0
                
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    
                    # 同步批次
                    batch_result = await self._sync_batch(batch, delay)
                    total_synced += batch_result['synced']
                    total_errors += batch_result['errors']
                    
                    logger.info(f"Batch {i//batch_size + 1}: synced {batch_result['synced']}, errors {batch_result['errors']}")
                
                result = {
                    'status': 'completed',
                    'total_symbols': len(symbols),
                    'synced': total_synced,
                    'errors': total_errors,
                    'success_rate': total_synced / len(symbols) if symbols else 0
                }
                
                logger.info(f"Market data sync completed: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Error during market data sync: {e}")
            return {'status': 'error', 'error': str(e), 'synced': 0, 'errors': 0}
    
    async def _sync_batch(self, symbols: List[str], delay: float) -> Dict[str, int]:
        """同步一批股票"""
        synced = 0
        errors = 0
        
        for symbol in symbols:
            try:
                # 获取最近30天的数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                # 同步数据
                data = await self.get_historical_data(symbol, start_date, end_date, force_refresh=False)
                
                if data is not None and not data.empty:
                    synced += 1
                else:
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Error syncing {symbol}: {e}")
                errors += 1
            
            # 延迟
            await asyncio.sleep(delay)
        
        return {'synced': synced, 'errors': errors}
    
    async def _sync_and_return_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                                   existing_data: pd.DataFrame) -> pd.DataFrame:
        """同步缺失数据并返回完整数据 - 添加防循环机制"""
        try:
            # 检查是否已经在同步中，防止循环调用
            sync_key = f"syncing_{symbol}"
            if hasattr(self, '_sync_in_progress'):
                if self._sync_in_progress.get(sync_key, False):
                    logger.warning(f"Sync already in progress for {symbol}, skipping to prevent loop")
                    return existing_data
            else:
                self._sync_in_progress = {}
            
            # 标记同步进行中
            self._sync_in_progress[sync_key] = True
            
            try:
                # 找出缺失的日期范围
                missing_ranges = self._find_missing_date_ranges(existing_data, start_date, end_date)
                
                if not missing_ranges:
                    return existing_data
                
                logger.info(f"Found {len(missing_ranges)} missing date ranges for {symbol}")
                
                # 同步缺失的数据 - 限制重试次数和范围
                all_data = [existing_data]
                max_sync_ranges = 3  # 最多同步3个缺失范围
                sync_count = 0
                
                for missing_start, missing_end in missing_ranges:
                    if sync_count >= max_sync_ranges:
                        logger.warning(f"Reached maximum sync ranges ({max_sync_ranges}) for {symbol}")
                        break
                    
                    # 限制单次同步的时间范围（最多30天）
                    if (missing_end - missing_start).days > 30:
                        logger.warning(f"Missing range too large ({(missing_end - missing_start).days} days), limiting to 30 days")
                        missing_end = missing_start + timedelta(days=30)
                    
                    logger.info(f"Syncing missing data for {symbol}: {missing_start.date()} to {missing_end.date()}")
                    
                    # unified_provider.get_historical_data 是同步方法，不应使用 await
                    missing_data = self.unified_provider.get_historical_data(
                        symbol, missing_start.strftime('%Y-%m-%d'), missing_end.strftime('%Y-%m-%d'),
                        quality_threshold=self.config.quality_threshold * 0.7  # 降低质量要求
                    )
                    
                    if missing_data is not None and not missing_data.empty:
                        all_data.append(missing_data)
                        await self.db_manager.save_stock_data(symbol, missing_data)
                        logger.info(f"Successfully synced {len(missing_data)} records for {symbol}")
                        sync_count += 1
                    else:
                        logger.warning(f"Failed to sync missing data for {symbol} in range {missing_start.date()} to {missing_end.date()}")
                        # 如果某个范围同步失败，继续尝试下一个范围，但不中断整个流程
                        sync_count += 1  # 仍然计数，避免无限尝试
                
                # 合并所有数据
                if len(all_data) > 1:
                    combined_data = pd.concat(all_data).sort_index().drop_duplicates()
                    logger.info(f"Combined data for {symbol}: {len(combined_data)} total records")
                    return combined_data
                else:
                    logger.info(f"No additional data synced for {symbol}")
                    return existing_data
                    
            finally:
                # 清除同步标记
                self._sync_in_progress[sync_key] = False
                
        except Exception as e:
            logger.error(f"Error syncing missing data for {symbol}: {e}")
            return existing_data
    
    def _find_missing_date_ranges(self, existing_data: pd.DataFrame, 
                                  start_date: datetime, end_date: datetime) -> List[tuple]:
        """找出缺失的日期范围"""
        if existing_data.empty:
            return [(start_date, end_date)]
        
        # 获取现有数据的日期范围
        existing_dates = set(existing_data.index.date)
        expected_dates = set(pd.date_range(start_date, end_date, freq='D').date)
        
        # 找出缺失的日期
        missing_dates = expected_dates - existing_dates
        if not missing_dates:
            return []
        
        # 将缺失日期转换为范围
        missing_ranges = []
        sorted_missing = sorted(missing_dates)
        
        if not sorted_missing:
            return []
        
        current_start = sorted_missing[0]
        current_end = sorted_missing[0]
        for date in sorted_missing[1:]:
            if date == current_end + timedelta(days=1):
                current_end = date
            else:
                missing_ranges.append((
                    datetime.combine(current_start, datetime.min.time()),
                    datetime.combine(current_end, datetime.min.time())
                ))
                current_start = date
                current_end = date
        
        # 添加最后一个范围
        missing_ranges.append((
            datetime.combine(current_start, datetime.min.time()),
            datetime.combine(current_end, datetime.min.time())
        ))
        
        return missing_ranges
    
    def _is_data_complete(self, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> bool:
        """改进后的完整性检测：
        1. 先确保时间范围覆盖（首尾日期基本落在目标范围内，可容忍1天误差）
        2. 再计算工作日数量的完整度，允许10% 以内缺失（考虑节假日造成的误判）
        """
        if data.empty:
            return False
    
        # 确保 index 为 datetime 类型
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data = data.set_index(pd.to_datetime(data.index))
            except Exception:
                return False
    
        min_date = data.index.min().date()
        max_date = data.index.max().date()
    
        # 允许首尾各 3 天误差（考虑节假日或停牌）
        head_gap = (min_date - start_date.date()).days
        if head_gap > 3:
            return False
        # 如果尾部缺口<=3天则认为可接受
        tail_gap = (end_date.date() - max_date).days
        if tail_gap > 3:
            return False
    
        # 计算期望交易日（工作日）数量
        expected_days = len(pd.bdate_range(start_date, end_date))
        actual_days = len(data)
    
        if expected_days == 0:
            return False
    
        completeness_ratio = actual_days / expected_days
        # 允许 10% 缺失
        return completeness_ratio >= 0.8
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        获取数据质量报告
        
        Returns:
            数据质量报告
        """
        try:
            # 获取数据源状态
            data_sources = {}
            
            # 获取所有提供者
            all_providers = (self.unified_provider.primary_providers + 
                          self.unified_provider.fallback_providers)
            
            for provider in all_providers:
                provider_name = provider.__class__.__name__
                try:
                    # 简单检查提供者是否可用
                    data_sources[provider_name] = {
                        'available': True,
                        'type': 'primary' if provider in self.unified_provider.primary_providers else 'fallback',
                        'last_check': datetime.now().isoformat()
                    }
                except Exception as e:
                    data_sources[provider_name] = {
                        'available': False,
                        'error': str(e),
                        'last_check': datetime.now().isoformat()
                    }
            
            # 获取缓存统计
            cache_stats = {
                'hits': self._cache_stats['hits'],
                'misses': self._cache_stats['misses'],
                'hit_rate': (self._cache_stats['hits'] / 
                           (self._cache_stats['hits'] + self._cache_stats['misses']))
                           if (self._cache_stats['hits'] + self._cache_stats['misses']) > 0 else 0
            }
            
            # 数据库状态
            db_status = {
                'connected': self.db_manager.test_connection(),
                'tables': ['stocks', 'stock_data', 'realtime_data'],
                'last_check': datetime.now().isoformat()
            }
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'primary_source': self.config.preferred_data_sources[0] if self.config.preferred_data_sources else 'unknown',
                'data_sources': data_sources,
                'cache_stats': cache_stats,
                'database': db_status,
                'overall_health': 'healthy' if data_sources and any(ds.get('available') for ds in data_sources.values()) else 'degraded'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_health': 'error'
            }
    
    async def validate_data_sources(self) -> Dict[str, Any]:
        """
        验证数据源
        
        Returns:
            验证结果
        """
        try:
            # 初始化数据源
            return self.initialize_data_sources(validate_sources=True)
        except Exception as e:
            logger.error(f"Error validating data sources: {e}")
            return {'error': str(e)}
    
    def set_primary_data_source(self, source_name: str) -> bool:
        """
        设置主要数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            是否成功
        """
        try:
            # 重新排序首选数据源
            if source_name in self.config.preferred_data_sources:
                # 将指定数据源移到第一位
                self.config.preferred_data_sources.remove(source_name)
                self.config.preferred_data_sources.insert(0, source_name)
                
                # 重新初始化数据源
                self.initialize_data_sources(validate_sources=False)
                
                logger.info(f"Set {source_name} as primary data source")
                return True
            else:
                logger.warning(f"Data source {source_name} not found in preferred sources")
                return False
                
        except Exception as e:
            logger.error(f"Error setting primary data source: {e}")
            return False
    
    async def initialize(self) -> bool:
        """
        初始化统一数据访问层
        
        Returns:
            是否成功
        """
        try:
            # 初始化数据库
            db_initialized = await self.db_manager.initialize()
            if not db_initialized:
                logger.error("Failed to initialize database")
                return False
            
            # 初始化数据源
            self.initialize_data_sources()
            
            logger.info("UnifiedDataAccessLayer initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedDataAccessLayer: {e}")
            return False