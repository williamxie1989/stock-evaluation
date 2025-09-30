"""
并发增强数据提供器模块 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """数据请求对象"""
    symbol: str
    start_date: str
    end_date: str
    data_type: str = "stock"
    frequency: str = "daily"


class ConcurrentEnhancedDataProvider:
    """并发增强数据提供器"""
    
    def __init__(self, max_workers: int = 4, config: Optional[Dict[str, Any]] = None):
        self.max_workers = max_workers
        self.config = config or {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.request_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info(f"ConcurrentEnhancedDataProvider 初始化完成，工作线程数: {max_workers}")
    
    async def get_data(self, symbol: str, start_date: str, end_date: str, 
                      data_type: str = "stock", frequency: str = "daily") -> pd.DataFrame:
        """获取数据"""
        request = DataRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type,
            frequency=frequency
        )
        
        # 检查缓存
        cache_key = f"{symbol}_{start_date}_{end_date}_{data_type}_{frequency}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.request_stats['cache_hits'] += 1
            logger.info(f"缓存命中: {cache_key}")
            return cached_data
        
        self.request_stats['cache_misses'] += 1
        
        # 异步获取数据
        try:
            data = await self._fetch_data_async(request)
            self._save_to_cache(cache_key, data)
            self.request_stats['total_requests'] += 1
            return data
        except Exception as e:
            self.request_stats['errors'] += 1
            logger.error(f"获取数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    async def get_batch_data(self, symbols: List[str], start_date: str, end_date: str,
                           data_type: str = "stock", frequency: str = "daily") -> Dict[str, pd.DataFrame]:
        """批量获取数据"""
        logger.info(f"批量获取 {len(symbols)} 只股票的数据")
        
        # 创建任务
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self.get_data(symbol, start_date, end_date, data_type, frequency)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建结果字典
        batch_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"获取 {symbol} 数据失败: {result}")
                batch_data[symbol] = pd.DataFrame()
            else:
                batch_data[symbol] = result
        
        return batch_data
    
    async def _fetch_data_async(self, request: DataRequest) -> pd.DataFrame:
        """异步获取数据"""
        # 在线程池中执行数据获取
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._fetch_data_sync, request)
    
    def _fetch_data_sync(self, request: DataRequest) -> pd.DataFrame:
        """同步获取数据（模拟）"""
        try:
            # 模拟数据获取延迟
            import time
            time.sleep(0.1)
            
            # 生成模拟数据
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 生成模拟股票数据
            if request.data_type == "stock":
                base_price = 100.0
                data = []
                current_price = base_price
                
                for date in date_range:
                    # 模拟价格波动
                    change = np.random.normal(0, 0.02)  # 2%的日波动率
                    current_price = current_price * (1 + change)
                    
                    open_price = current_price * (1 + np.random.normal(0, 0.005))
                    high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
                    low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
                    close_price = current_price
                    
                    volume = np.random.randint(1000000, 10000000)
                    
                    data.append({
                        'date': date,
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'volume': volume,
                        'symbol': request.symbol
                    })
                
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                return df
            
            else:
                # 其他数据类型的模拟
                return pd.DataFrame({
                    'date': date_range,
                    'value': np.random.randn(len(date_range)),
                    'symbol': request.symbol
                }).set_index('date')
                
        except Exception as e:
            logger.error(f"同步获取数据失败: {e}")
            return pd.DataFrame()
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        with self.cache_lock:
            if key in self.cache:
                # 检查缓存是否过期（1小时）
                timestamp, data = self.cache[key]
                if datetime.now() - timestamp < timedelta(hours=1):
                    return data.copy()
                else:
                    # 过期缓存删除
                    del self.cache[key]
        return None
    
    def _save_to_cache(self, key: str, data: pd.DataFrame) -> None:
        """保存到缓存"""
        with self.cache_lock:
            self.cache[key] = (datetime.now(), data.copy())
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.cache_lock:
            cache_size = len(self.cache)
        
        total_requests = self.request_stats['total_requests']
        cache_hits = self.request_stats['cache_hits']
        cache_misses = self.request_stats['cache_misses']
        
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        return {
            'cache_size': cache_size,
            'total_requests': total_requests,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': hit_rate,
            'errors': self.request_stats['errors']
        }
    
    def shutdown(self) -> None:
        """关闭服务"""
        self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("ConcurrentEnhancedDataProvider 已关闭")


# 全局服务实例
_enhanced_data_provider = None

def get_enhanced_data_provider() -> ConcurrentEnhancedDataProvider:
    """获取增强数据提供器实例"""
    global _enhanced_data_provider
    if _enhanced_data_provider is None:
        _enhanced_data_provider = ConcurrentEnhancedDataProvider()
    return _enhanced_data_provider