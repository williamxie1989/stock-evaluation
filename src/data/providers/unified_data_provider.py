"""
统一数据提供者 - 整合多个数据源，提供统一的数据访问接口
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

from ...core.interfaces import DataProviderInterface, UnifiedDataProviderInterface

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 0.0      # 完整性
    accuracy: float = 0.0          # 准确性
    timeliness: float = 0.0        # 及时性
    consistency: float = 0.0       # 一致性
    reliability: float = 0.0         # 可靠性
    overall_score: float = 0.0     # 综合评分
    
    def __post_init__(self):
        """计算综合评分"""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'timeliness': 0.20,
            'consistency': 0.15,
            'reliability': 0.15
        }
        
        self.overall_score = (
            self.completeness * weights['completeness'] +
            self.accuracy * weights['accuracy'] +
            self.timeliness * weights['timeliness'] +
            self.consistency * weights['consistency'] +
            self.reliability * weights['reliability']
        )


class UnifiedDataProvider(UnifiedDataProviderInterface):
    """统一数据提供者 - 整合多个数据源"""
    
    def __init__(self, cache_ttl: int = 300, max_workers: int = 8):
        """
        初始化统一数据提供者
        
        Args:
            cache_ttl: 缓存过期时间（秒）
            max_workers: 最大工作线程数
        """
        self.primary_providers: List[DataProviderInterface] = []
        self.fallback_providers: List[DataProviderInterface] = []
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.data_quality_scores: Dict[str, float] = {}
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        self._cache_lock = threading.Lock()
        
        logger.info(f"UnifiedDataProvider initialized with cache_ttl={cache_ttl}s, max_workers={max_workers}")
    
    def add_primary_provider(self, provider: DataProviderInterface) -> None:
        """添加主要数据提供者"""
        self.primary_providers.append(provider)
        logger.info(f"Added primary provider: {provider.__class__.__name__}")
    
    def add_fallback_provider(self, provider: DataProviderInterface) -> None:
        """添加备用数据提供者"""
        self.fallback_providers.append(provider)
        logger.info(f"Added fallback provider: {provider.__class__.__name__}")
    
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """生成缓存键"""
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
        return f"{data_type}_{params_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        with self._cache_lock:
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if datetime.now() < cached_data['expires_at']:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_data['data']
                else:
                    # 缓存过期，删除
                    del self.cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """设置缓存"""
        with self._cache_lock:
            self.cache[cache_key] = {
                'data': data,
                'expires_at': datetime.now() + timedelta(seconds=self.cache_ttl)
            }
            logger.debug(f"Cache set for key: {cache_key}")
    
    def _assess_data_quality(self, df: pd.DataFrame, symbol: str, data_type: str = "historical") -> DataQualityMetrics:
        """评估数据质量"""
        if df is None or df.empty:
            return DataQualityMetrics()
        
        metrics = DataQualityMetrics()
        
        try:
            # 完整性评估 - 支持中英文列名
            expected_columns_mapping = {
                'open': ['open', '开盘', 'Open'],
                'high': ['high', '最高', 'High'], 
                'low': ['low', '最低', 'Low'],
                'close': ['close', '收盘', 'Close'],
                'volume': ['volume', '成交量', 'Volume']
            }
            
            available_columns = 0
            for english_name, possible_names in expected_columns_mapping.items():
                if any(col in df.columns for col in possible_names):
                    available_columns += 1
            
            metrics.completeness = available_columns / len(expected_columns_mapping)
            
            # 准确性评估
            if data_type == "historical":
                # 获取实际列名映射
                column_mapping = {}
                for english_name, possible_names in expected_columns_mapping.items():
                    for col_name in possible_names:
                        if col_name in df.columns:
                            column_mapping[english_name] = col_name
                            break
                
                # 检查价格合理性
                price_valid = True
                for english_col in ['open', 'high', 'low', 'close']:
                    if english_col in column_mapping:
                        actual_col = column_mapping[english_col]
                        if (df[actual_col] <= 0).any() or (df[actual_col] > 10000).any():
                            price_valid = False
                            break
                
                # 检查价格逻辑
                if price_valid and 'high' in column_mapping and 'low' in column_mapping:
                    high_col = column_mapping['high']
                    low_col = column_mapping['low']
                    if (df[high_col] < df[low_col]).any():
                        price_valid = False
                
                metrics.accuracy = 1.0 if price_valid else 0.0
            else:
                # 实时数据准确性检查
                price = df.get('price', 0)
                change = df.get('change', 0)
                volume = df.get('volume', 0)
                
                if (price > 0 and price <= 10000 and 
                    -21 <= change <= 21 and 
                    volume >= 0):
                    metrics.accuracy = 1.0
                else:
                    metrics.accuracy = 0.0
            
            # 及时性评估
            date_columns = ['date', '日期', 'Date', 'time', 'Time']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                try:
                    latest_date = pd.to_datetime(df[date_col]).max()
                    days_diff = (datetime.now() - latest_date).days
                    if days_diff <= 1:
                        metrics.timeliness = 1.0
                    elif days_diff <= 3:
                        metrics.timeliness = 0.8
                    elif days_diff <= 7:
                        metrics.timeliness = 0.6
                    else:
                        metrics.timeliness = 0.3
                except:
                    metrics.timeliness = 0.5
            else:
                metrics.timeliness = 0.5
            
            # 一致性评估
            if len(df) > 1:
                # 检查数据连续性（没有大的间隔）
                if date_col:
                    # 使用日期列排序
                    df_sorted = df.sort_values(date_col)
                    try:
                        date_diffs = pd.to_datetime(df_sorted[date_col]).diff().dt.days
                        large_gaps = (date_diffs > 10).sum()
                        metrics.consistency = max(0.0, 1.0 - (large_gaps / len(df)))
                    except Exception:
                        metrics.consistency = 0.8
                else:
                    # 若缺少日期列，则按索引排序并检查索引是否为 DatetimeIndex
                    if isinstance(df.index, pd.DatetimeIndex):
                        df_sorted = df.sort_index()
                        try:
                            date_diffs = df_sorted.index.to_series().diff().dt.days
                            large_gaps = (date_diffs > 10).sum()
                            metrics.consistency = max(0.0, 1.0 - (large_gaps / len(df)))
                        except Exception:
                            metrics.consistency = 0.8
                    else:
                        # 无法评估一致性，使用默认值
                        metrics.consistency = 0.5
            else:
                metrics.consistency = 0.5
            
            # 可靠性评估（基于历史表现）
            provider_name = symbol.split('_')[0] if '_' in symbol else 'unknown'
            historical_score = self.data_quality_scores.get(provider_name, 0.8)
            metrics.reliability = historical_score
            
            # 重新计算总体评分（因为字段赋值后__post_init__不会重新计算）
            weights = {
                'completeness': 0.25,
                'accuracy': 0.25,
                'timeliness': 0.20,
                'consistency': 0.15,
                'reliability': 0.15
            }
            
            metrics.overall_score = (
                metrics.completeness * weights['completeness'] +
                metrics.accuracy * weights['accuracy'] +
                metrics.timeliness * weights['timeliness'] +
                metrics.consistency * weights['consistency'] +
                metrics.reliability * weights['reliability']
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality for {symbol}: {e}")
            metrics = DataQualityMetrics()
        
        return metrics
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          quality_threshold: float = 0.8) -> Optional[pd.DataFrame]:
        """获取历史数据，支持质量评估和自动切换 - 添加快速失败机制"""
        cache_key = self._get_cache_key("historical", symbol=symbol, start_date=start_date, end_date=end_date)
        
        # 检查缓存
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # 检查是否已经在处理中，防止循环调用
        processing_key = f"historical_{symbol}_{start_date}_{end_date}"
        if not hasattr(self, '_processing_requests'):
            self._processing_requests = {}
        if self._processing_requests.get(processing_key, False):
            logger.warning(f"Historical data request already in progress for {symbol} {start_date} to {end_date}, skipping to prevent loop")
            return None
        
        # 标记处理进行中
        self._processing_requests[processing_key] = True
        
        try:
            # 尝试主要数据提供者
            for provider in self.primary_providers:
                try:
                    logger.info(f"Trying primary provider {provider.__class__.__name__} for {symbol}")
                    data = None
                    
                    # 优先尝试 get_historical_data 方法
                    if hasattr(provider, 'get_historical_data'):
                        data = provider.get_historical_data(symbol, start_date, end_date)
                    # 其次尝试 get_stock_data 方法
                    elif hasattr(provider, 'get_stock_data'):
                        data = provider.get_stock_data(symbol, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # 评估数据质量
                        quality_metrics = self._assess_data_quality(data, symbol, "historical")
                        
                        if quality_metrics.overall_score >= quality_threshold:
                            logger.info(f"Primary provider {provider.__class__.__name__} succeeded for {symbol}, quality: {quality_metrics.overall_score:.2f}")
                            self._set_cache(cache_key, data)
                            return data
                        else:
                            logger.warning(f"Primary provider data quality too low for {symbol}: {quality_metrics.overall_score:.2f}")
                    
                except Exception as e:
                    logger.error(f"Primary provider {provider.__class__.__name__} failed for {symbol}: {e}")
                    continue
            
            # 尝试备用数据提供者，限制尝试次数避免无限循环
            max_fallback_attempts = 3
            fallback_attempts = 0
            
            for provider in self.fallback_providers:
                if fallback_attempts >= max_fallback_attempts:
                    logger.warning(f"Reached maximum fallback attempts ({max_fallback_attempts}) for {symbol}")
                    break
                
                try:
                    logger.info(f"Trying fallback provider {provider.__class__.__name__} for {symbol} (attempt {fallback_attempts + 1}/{max_fallback_attempts})")
                    data = None
                    
                    # 优先尝试 get_historical_data 方法
                    if hasattr(provider, 'get_historical_data'):
                        data = provider.get_historical_data(symbol, start_date, end_date)
                    # 其次尝试 get_stock_data 方法
                    elif hasattr(provider, 'get_stock_data'):
                        data = provider.get_stock_data(symbol, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # 评估数据质量（备用源可以降低标准）
                        quality_metrics = self._assess_data_quality(data, symbol, "historical")
                        
                        if quality_metrics.overall_score >= quality_threshold * 0.7:  # 备用源标准降低
                            logger.info(f"Fallback provider {provider.__class__.__name__} succeeded for {symbol}, quality: {quality_metrics.overall_score:.2f}")
                            self._set_cache(cache_key, data)
                            return data
                        else:
                            logger.warning(f"Fallback provider data quality too low for {symbol}: {quality_metrics.overall_score:.2f}")
                    
                    fallback_attempts += 1
                    
                except Exception as e:
                    logger.error(f"Fallback provider {provider.__class__.__name__} failed for {symbol}: {e}")
                    fallback_attempts += 1
                    continue
            
            logger.error(f"All providers failed to get historical data for {symbol}")
            return None
            
        finally:
            # 清除处理标记
            self._processing_requests[processing_key] = False
    
    def get_realtime_data(self, symbols: List[str], max_retries: int = 3) -> Optional[Dict[str, Dict[str, Any]]]:
        """获取实时数据，支持失败重试"""
        if not symbols:
            return {}
        
        cache_key = self._get_cache_key("realtime", symbols=",".join(sorted(symbols)))
        
        # 检查缓存
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        result = {}
        failed_symbols = []
        
        # 使用线程池并行获取数据
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 为每个股票提交任务
            future_to_symbol = {}
            
            for symbol in symbols:
                # 尝试主要提供者
                for provider in self.primary_providers:
                    if hasattr(provider, 'get_realtime_data'):
                        future = executor.submit(self._get_realtime_data_single, provider, symbol)
                        future_to_symbol[future] = (symbol, provider.__class__.__name__)
                        break
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol, provider_name = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[symbol] = data
                        logger.info(f"Successfully got realtime data for {symbol} from {provider_name}")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"Failed to get realtime data for {symbol} from {provider_name}")
                except Exception as e:
                    logger.error(f"Error getting realtime data for {symbol} from {provider_name}: {e}")
                    failed_symbols.append(symbol)
        
        # 处理失败的符号
        if failed_symbols and max_retries > 0:
            logger.info(f"Retrying failed symbols: {failed_symbols}")
            retry_result = self.get_realtime_data(failed_symbols, max_retries - 1)
            if retry_result:
                result.update(retry_result)
        
        if result:
            self._set_cache(cache_key, result)
        
        return result if result else None
    
    def _get_realtime_data_single(self, provider: DataProviderInterface, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个股票的实时数据"""
        try:
            if hasattr(provider, 'get_realtime_data'):
                # 尝试批量获取接口
                batch_data = provider.get_realtime_data([symbol])
                if batch_data and symbol in batch_data:
                    return batch_data[symbol]
            
            # 尝试单个获取接口
            if hasattr(provider, 'get_realtime_data_single'):
                return provider.get_realtime_data_single(symbol)
            
            # 尝试使用get_realtime_data方法（单个股票）
            if hasattr(provider, 'get_realtime_data'):
                # 某些提供者可能支持单个股票调用
                result = provider.get_realtime_data(symbol)
                if isinstance(result, dict) and 'price' in result:
                    return result
                
        except Exception as e:
            logger.debug(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
        
        return None
    
    def validate_data_source(self, provider: DataProviderInterface, 
                           test_symbols: List[str] = None) -> Dict[str, Any]:
        """验证数据源可靠性"""
        if test_symbols is None:
            test_symbols = ['600519.SH', '000858.SZ', '300750.SZ']  # 默认测试股票
        
        validation_result = {
            'provider': provider.__class__.__name__,
            'test_symbols': test_symbols,
            'historical_tests': {},
            'realtime_tests': {},
            'overall_score': 0.0,
            'recommendation': 'not_recommended'
        }
        
        total_score = 0.0
        test_count = 0
        
        # 测试历史数据获取
        for symbol in test_symbols:
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                data = provider.get_stock_data(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    quality_metrics = self._assess_data_quality(data, symbol, "historical")
                    score = quality_metrics.overall_score
                    
                    validation_result['historical_tests'][symbol] = {
                        'success': True,
                        'quality_score': score,
                        'data_points': len(data),
                        'quality_details': {
                            'completeness': quality_metrics.completeness,
                            'accuracy': quality_metrics.accuracy,
                            'timeliness': quality_metrics.timeliness,
                            'consistency': quality_metrics.consistency,
                            'reliability': quality_metrics.reliability
                        }
                    }
                    
                    total_score += score
                    test_count += 1
                    
                else:
                    validation_result['historical_tests'][symbol] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    
            except Exception as e:
                validation_result['historical_tests'][symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 测试实时数据获取
        for symbol in test_symbols:
            try:
                if hasattr(provider, 'get_realtime_data'):
                    data = provider.get_realtime_data([symbol])
                    
                    if data and symbol in data:
                        quality_metrics = self._assess_data_quality(data[symbol], symbol, "realtime")
                        score = quality_metrics.overall_score
                        
                        validation_result['realtime_tests'][symbol] = {
                            'success': True,
                            'quality_score': score,
                            'quality_details': {
                                'completeness': quality_metrics.completeness,
                                'accuracy': quality_metrics.accuracy,
                                'timeliness': quality_metrics.timeliness,
                                'consistency': quality_metrics.consistency,
                                'reliability': quality_metrics.reliability
                            }
                        }
                        
                        total_score += score
                        test_count += 1
                        
                    else:
                        validation_result['realtime_tests'][symbol] = {
                            'success': False,
                            'error': 'No data returned'
                        }
                        
            except Exception as e:
                validation_result['realtime_tests'][symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 计算总体评分
        if test_count > 0:
            validation_result['overall_score'] = total_score / test_count
        else:
            validation_result['overall_score'] = 0.0
        
        # 生成推荐建议
        if validation_result['overall_score'] >= 0.8:
            validation_result['recommendation'] = 'highly_recommended'
        elif validation_result['overall_score'] >= 0.6:
            validation_result['recommendation'] = 'recommended'
        elif validation_result['overall_score'] >= 0.4:
            validation_result['recommendation'] = 'acceptable'
        else:
            validation_result['recommendation'] = 'not_recommended'
        
        logger.info(f"Validation completed for {provider.__class__.__name__}: score={validation_result['overall_score']:.2f}, recommendation={validation_result['recommendation']}")
        
        return validation_result
    
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """获取股票列表 - 尝试所有提供者"""
        # 尝试主要提供者
        for provider in self.primary_providers:
            try:
                if hasattr(provider, 'get_stock_list'):
                    data = provider.get_stock_list()
                    if data is not None and not data.empty:
                        return data
                elif hasattr(provider, 'get_all_stock_list'):
                    data = provider.get_all_stock_list()
                    if data is not None and not data.empty:
                        return data
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed to get stock list: {e}")
                continue
        
        # 尝试备用提供者
        for provider in self.fallback_providers:
            try:
                if hasattr(provider, 'get_stock_list'):
                    data = provider.get_stock_list()
                    if data is not None and not data.empty:
                        return data
                elif hasattr(provider, 'get_all_stock_list'):
                    data = provider.get_all_stock_list()
                    if data is not None and not data.empty:
                        return data
            except Exception as e:
                logger.warning(f"Fallback provider {provider.__class__.__name__} failed to get stock list: {e}")
                continue
        
        logger.error("All providers failed to get stock list")
        return None
    
    def get_all_stock_list(self) -> Optional[pd.DataFrame]:
        """获取所有股票列表 - 尝试所有提供者"""
        # 尝试主要提供者
        for provider in self.primary_providers:
            try:
                if hasattr(provider, 'get_all_stock_list'):
                    data = provider.get_all_stock_list()
                    if data is not None and not data.empty:
                        return data
                elif hasattr(provider, 'get_stock_list'):
                    data = provider.get_stock_list()
                    if data is not None and not data.empty:
                        return data
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed to get all stock list: {e}")
                continue
        
        # 尝试备用提供者
        for provider in self.fallback_providers:
            try:
                if hasattr(provider, 'get_all_stock_list'):
                    data = provider.get_all_stock_list()
                    if data is not None and not data.empty:
                        return data
                elif hasattr(provider, 'get_stock_list'):
                    data = provider.get_stock_list()
                    if data is not None and not data.empty:
                        return data
            except Exception as e:
                logger.warning(f"Fallback provider {provider.__class__.__name__} failed to get all stock list: {e}")
                continue
        
        logger.error("All providers failed to get all stock list")
        return None
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._cache_lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values() 
                                if datetime.now() >= entry['expires_at'])
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'cache_ttl': self.cache_ttl
            }

    # 兼容DataProviderInterface常用方法名，转到get_historical_data
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, quality_threshold: float = 0.8):
        """向后兼容统一命名"""
        return self.get_historical_data(symbol, start_date, end_date, quality_threshold)

    def get_market_data_by_date_range(self, start_date: str, end_date: str,
                                       symbols: Optional[List[str]] = None,
                                       quality_threshold: float = 0.7,
                                       max_retries: int = 3) -> Optional[pd.DataFrame]:
        """按日期范围获取市场数据（全市场或部分股票），带质量评估与失败回退。

        该方法会尝试所有主要数据提供者的 ``get_market_data_by_date_range`` 或 ``get_stock_data_by_date`` 方法；若主要提供者均失败，则尝试备用提供者，
        并在必要时降低质量阈值。成功返回后会写入缓存。

        Args:
            start_date: 开始日期 ``YYYY-MM-DD``
            end_date: 结束日期 ``YYYY-MM-DD``
            symbols: 股票列表，None 表示全市场
            quality_threshold: 数据质量阈值
            max_retries: 备用源最大重试次数

        Returns:
            合并后的行情 ``DataFrame``，或 ``None``（全部失败）。
        """
        cache_key = self._get_cache_key(
            "market_range", start_date=start_date, end_date=end_date,
            symbols="all" if symbols is None else ",".join(sorted(symbols))
        )
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        def _try_provider(provider):
            data = None
            if hasattr(provider, "get_market_data_by_date_range"):
                data = provider.get_market_data_by_date_range(start_date, end_date, symbols=symbols)
            elif hasattr(provider, "get_stock_data_by_date"):
                # 退化到逐日循环调用（效率可能低，但保证兼容）
                from pandas import concat
                cur_date = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                frames = []
                while cur_date <= end_dt:
                    try:
                        df_day = provider.get_stock_data_by_date(cur_date.strftime("%Y-%m-%d"), symbols=symbols)
                        if df_day is not None and not df_day.empty:
                            frames.append(df_day)
                    except Exception:
                        pass
                    cur_date += timedelta(days=1)
                if frames:
                    data = concat(frames, ignore_index=True)
            return data

        # 尝试主要提供者
        for provider in self.primary_providers:
            try:
                df = _try_provider(provider)
                if df is not None and not df.empty:
                    metrics = self._assess_data_quality(df, "market_range", "historical")
                    if metrics.overall_score >= quality_threshold:
                        self._set_cache(cache_key, df)
                        return df
            except Exception as e:
                logger.error("Primary provider %s failed in get_market_data_by_date_range: %s", provider.__class__.__name__, e)

        # 尝试备用提供者
        attempts = 0
        for provider in self.fallback_providers:
            if attempts >= max_retries:
                break
            try:
                df = _try_provider(provider)
                if df is not None and not df.empty:
                    metrics = self._assess_data_quality(df, "market_range", "historical")
                    if metrics.overall_score >= quality_threshold * 0.7:
                        self._set_cache(cache_key, df)
                        return df
            except Exception as e:
                logger.error("Fallback provider %s failed in get_market_data_by_date_range: %s", provider.__class__.__name__, e)
            attempts += 1

        logger.error("All providers failed to get market data by date range %s~%s", start_date, end_date)
        return None

    # ... existing code ...