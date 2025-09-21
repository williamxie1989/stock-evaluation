import logging
import pandas as pd
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
import queue
from dataclasses import dataclass

from enhanced_data_provider import EnhancedDataProvider


@dataclass
class DataFetchTask:
    """数据获取任务"""
    symbol: str
    period: str = "1y"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    preferred_sources: Optional[List[str]] = None


@dataclass
class DataFetchResult:
    """数据获取结果"""
    symbol: str
    success: bool
    data: Optional[pd.DataFrame] = None
    source: Optional[str] = None
    error: Optional[str] = None
    attempts: List[Dict[str, Any]] = None
    processing_time: float = 0.0


class ConcurrentEnhancedDataProvider:
    """
    支持多线程并发的增强数据提供者
    - 并发获取多只股票数据
    - 智能负载均衡
    - 数据源故障转移
    - 批量数据处理
    """
    
    def __init__(self, max_workers: int = 8, rate_limit_delay: float = 0.1):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)
        
        # 线程安全锁
        self.stats_lock = Lock()
        self.rate_limit_lock = Lock()
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.source_usage = {}
        
        self.logger.info(f"ConcurrentEnhancedDataProvider initialized: max_workers={max_workers}")
    
    def get_multiple_stocks_data(self, 
                               symbols: List[str],
                               period: str = "1y",
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               preferred_sources: Optional[List[str]] = None,
                               max_workers: Optional[int] = None) -> Dict[str, DataFetchResult]:
        """
        并发获取多只股票的历史数据
        
        Args:
            symbols: 股票代码列表
            period: 数据周期
            start_date: 开始日期
            end_date: 结束日期
            preferred_sources: 优先数据源
            max_workers: 并发线程数
            
        Returns:
            Dict[symbol, DataFetchResult]: 获取结果字典
        """
        if not symbols:
            return {}
        
        workers = max_workers or self.max_workers
        self.logger.info(f"开始并发获取 {len(symbols)} 只股票数据，使用 {workers} 个线程")
        
        # 准备任务
        tasks = [
            DataFetchTask(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                preferred_sources=preferred_sources
            )
            for symbol in symbols
        ]
        
        # 执行并发获取
        return self._execute_concurrent_fetch(tasks, workers)
    
    def get_batch_historical_data(self,
                                symbols: List[str],
                                batch_size: int = 50,
                                period: str = "1y",
                                preferred_sources: Optional[List[str]] = None,
                                progress_callback: Optional[callable] = None) -> Dict[str, DataFetchResult]:
        """
        分批并发获取历史数据
        
        Args:
            symbols: 股票代码列表
            batch_size: 批次大小
            period: 数据周期
            preferred_sources: 优先数据源
            progress_callback: 进度回调函数
            
        Returns:
            Dict[symbol, DataFetchResult]: 获取结果字典
        """
        if not symbols:
            return {}
        
        total_symbols = len(symbols)
        all_results = {}
        processed_count = 0
        
        self.logger.info(f"开始分批获取 {total_symbols} 只股票数据，批次大小: {batch_size}")
        
        # 分批处理
        for i in range(0, total_symbols, batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_symbols + batch_size - 1) // batch_size
            
            self.logger.info(f"处理第 {batch_num}/{total_batches} 批，包含 {len(batch_symbols)} 只股票")
            
            # 获取当前批次数据
            batch_results = self.get_multiple_stocks_data(
                symbols=batch_symbols,
                period=period,
                preferred_sources=preferred_sources
            )
            
            # 合并结果
            all_results.update(batch_results)
            processed_count += len(batch_symbols)
            
            # 调用进度回调
            if progress_callback:
                try:
                    progress_callback(processed_count, total_symbols, batch_results)
                except Exception as e:
                    self.logger.warning(f"进度回调执行失败: {e}")
            
            # 批次间延迟
            if i + batch_size < total_symbols:
                time.sleep(random.uniform(0.5, 1.0))
        
        return all_results
    
    def _execute_concurrent_fetch(self, tasks: List[DataFetchTask], max_workers: int) -> Dict[str, DataFetchResult]:
        """执行并发数据获取"""
        results = {}
        completed_count = 0
        total_count = len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._fetch_single_stock, task): task
                for task in tasks
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.symbol] = result
                    
                    with self.stats_lock:
                        completed_count += 1
                        self.total_requests += 1
                        
                        if result.success:
                            self.successful_requests += 1
                            if result.source:
                                self.source_usage[result.source] = self.source_usage.get(result.source, 0) + 1
                        else:
                            self.failed_requests += 1
                    
                    # 定期报告进度
                    if completed_count % 20 == 0 or completed_count == total_count:
                        success_rate = self.successful_requests / max(self.total_requests, 1) * 100
                        self.logger.info(f"获取进度: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%), 成功率: {success_rate:.1f}%")
                        
                except Exception as e:
                    with self.stats_lock:
                        completed_count += 1
                        self.total_requests += 1
                        self.failed_requests += 1
                    
                    results[task.symbol] = DataFetchResult(
                        symbol=task.symbol,
                        success=False,
                        error=f"任务执行异常: {str(e)}"
                    )
                    self.logger.error(f"获取 {task.symbol} 数据时发生异常: {e}")
        
        return results
    
    def _fetch_single_stock(self, task: DataFetchTask) -> DataFetchResult:
        """获取单只股票数据"""
        start_time = time.time()
        
        try:
            # 应用速率限制
            with self.rate_limit_lock:
                time.sleep(self.rate_limit_delay)
            
            # 创建线程本地的数据提供者实例
            provider = EnhancedDataProvider()
            
            # 设置优先数据源
            if task.preferred_sources:
                provider.set_preferred_sources(task.preferred_sources)
            
            # 获取数据
            if task.start_date and task.end_date:
                df = provider.get_stock_historical_data_range(
                    symbol=task.symbol,
                    start_date=task.start_date,
                    end_date=task.end_date
                )
            else:
                df = provider.get_stock_historical_data(
                    symbol=task.symbol,
                    period=task.period
                )
            
            # 获取尝试信息
            attempts = getattr(provider, 'last_attempts', []) or []
            last_source = getattr(provider, 'last_used_source', None)
            
            processing_time = time.time() - start_time
            
            if df is not None and not df.empty:
                return DataFetchResult(
                    symbol=task.symbol,
                    success=True,
                    data=df,
                    source=last_source,
                    attempts=attempts,
                    processing_time=processing_time
                )
            else:
                return DataFetchResult(
                    symbol=task.symbol,
                    success=False,
                    error='no_data',
                    attempts=attempts,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return DataFetchResult(
                symbol=task.symbol,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            return {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'source_usage': self.source_usage.copy()
            }
    
    def reset_statistics(self):
        """重置统计信息"""
        with self.stats_lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.source_usage.clear()


class ConcurrentStockListProvider:
    """
    并发股票列表提供者
    支持并发获取多个市场的股票列表
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def get_multiple_markets_stocks(self, markets: List[str]) -> Dict[str, pd.DataFrame]:
        """
        并发获取多个市场的股票列表
        
        Args:
            markets: 市场代码列表 (如 ['SH', 'SZ'])
            
        Returns:
            Dict[market, DataFrame]: 各市场股票列表
        """
        if not markets:
            return {}
        
        self.logger.info(f"开始并发获取 {len(markets)} 个市场的股票列表")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(markets))) as executor:
            # 提交任务
            future_to_market = {
                executor.submit(self._get_single_market_stocks, market): market
                for market in markets
            }
            
            # 处理结果
            for future in as_completed(future_to_market):
                market = future_to_market[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[market] = df
                        self.logger.info(f"成功获取 {market} 市场股票列表: {len(df)} 只股票")
                    else:
                        self.logger.warning(f"获取 {market} 市场股票列表失败: 无数据")
                except Exception as e:
                    self.logger.error(f"获取 {market} 市场股票列表异常: {e}")
        
        return results
    
    def _get_single_market_stocks(self, market: str) -> Optional[pd.DataFrame]:
        """获取单个市场的股票列表"""
        try:
            from akshare_data_provider import AkshareDataProvider
            provider = AkshareDataProvider()
            return provider.get_stock_list(market)
        except Exception as e:
            self.logger.error(f"获取 {market} 市场股票列表失败: {e}")
            return None


if __name__ == "__main__":
    # 测试并发数据提供者
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试并发获取股票数据
    provider = ConcurrentEnhancedDataProvider(max_workers=4)
    
    test_symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
    
    print("测试并发获取股票数据...")
    results = provider.get_multiple_stocks_data(
        symbols=test_symbols,
        period="90d"
    )
    
    print(f"获取结果:")
    for symbol, result in results.items():
        if result.success:
            print(f"  {symbol}: 成功, {len(result.data)} 行数据, 来源: {result.source}")
        else:
            print(f"  {symbol}: 失败, 错误: {result.error}")
    
    print(f"统计信息: {provider.get_statistics()}")
    
    # 测试并发获取股票列表
    print("\n测试并发获取股票列表...")
    list_provider = ConcurrentStockListProvider()
    market_results = list_provider.get_multiple_markets_stocks(['SH', 'SZ'])
    
    for market, df in market_results.items():
        print(f"  {market}: {len(df)} 只股票")