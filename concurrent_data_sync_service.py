import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
import time
import random
import queue
from dataclasses import dataclass

from akshare_data_provider import AkshareDataProvider
from enhanced_data_provider import EnhancedDataProvider
from db import DatabaseManager
from stock_list_manager import StockListManager
from stock_status_filter import StockStatusFilter


@dataclass
class SyncTask:
    """数据同步任务"""
    symbol: str
    stock_info: Dict[str, Any]
    sync_type: str
    period: str
    start_date: Optional[str] = None


@dataclass
class SyncResult:
    """数据同步结果"""
    symbol: str
    success: bool
    rows_inserted: int = 0
    source: Optional[str] = None
    error: Optional[str] = None
    attempts: List[Dict[str, Any]] = None
    processing_time: float = 0.0


class ConcurrentDataSyncService:
    """
    支持多线程并发的数据同步服务
    - 并发获取股票数据
    - 批量数据库操作
    - 智能错误重试
    - 动态线程池管理
    """
    
    def __init__(self, max_workers: int = 8, db_batch_size: int = 50):
        # 基础组件
        self.list_provider = AkshareDataProvider()
        self.db_manager = DatabaseManager()
        self.stock_manager = StockListManager()
        self.stock_filter = StockStatusFilter()
        self.logger = logging.getLogger(__name__)
        
        # 并发控制参数
        self.max_workers = max_workers
        self.db_batch_size = db_batch_size
        
        # 线程安全的锁
        self.db_lock = RLock()  # 数据库操作锁
        self.stats_lock = Lock()  # 统计信息锁
        
        # 批量数据缓存
        self.batch_data_queue = queue.Queue()
        self.batch_results = []
        
        self.logger.info(f"ConcurrentDataSyncService initialized: max_workers={max_workers}, db_batch_size={db_batch_size}")
    
    def sync_market_data(self, 
                        sync_type: str = "incremental",
                        markets: List[str] = None,
                        max_symbols: int = 0,
                        max_workers: int = None,
                        db_batch_size: int = None,
                        preferred_sources: Optional[List[str]] = None,
                        top_n_by: Optional[str] = None,
                        top_n: int = 0,
                        amount_window_days: int = 5) -> Dict[str, Any]:
        """
        并发同步市场数据
        
        Args:
            sync_type: 同步类型 ("full" | "incremental")
            markets: 要同步的市场列表
            max_symbols: 最大处理股票数量
            max_workers: 并发线程数（None使用默认值）
            db_batch_size: 数据库批量操作大小（None使用默认值）
            preferred_sources: 优先数据源列表
            top_n_by: 筛选策略
            top_n: 筛选数量
            amount_window_days: 成交量窗口天数
        """
        start_time = time.time()
        
        # 动态调整参数
        if max_workers is not None:
            self.max_workers = max_workers
        if db_batch_size is not None:
            self.db_batch_size = db_batch_size
            
        self.logger.info(f"开始并发数据同步: sync_type={sync_type}, max_workers={self.max_workers}")
        
        try:
            # 1. 更新股票列表
            if sync_type == "full":
                list_result = self._update_stock_list(markets)
                if not list_result.get('success', False):
                    return {'success': False, 'error': '股票列表更新失败', 'details': list_result}
            
            # 2. 获取同步目标股票
            target_stocks = self._get_sync_target_stocks(
                markets=markets, 
                max_symbols=max_symbols,
                top_n_by=top_n_by, 
                top_n=top_n, 
                amount_window_days=amount_window_days
            )
            
            if not target_stocks:
                return {'success': False, 'error': '没有找到需要同步的股票'}
            
            # 3. 准备同步任务
            sync_tasks = self._prepare_sync_tasks(target_stocks, sync_type)
            
            # 4. 执行并发数据同步
            sync_result = self._execute_concurrent_sync(sync_tasks, preferred_sources)
            
            # 5. 记录同步状态
            self._record_sync_status(sync_type, sync_result)
            
            total_time = time.time() - start_time
            sync_result.update({
                'success': True,
                'sync_type': sync_type,
                'markets': markets or ['ALL'],
                'total_time_seconds': round(total_time, 2),
                'throughput_stocks_per_second': round(sync_result.get('total_stocks', 0) / max(total_time, 0.1), 2)
            })
            
            return sync_result
            
        except Exception as e:
            self.logger.error(f"并发数据同步失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time_seconds': time.time() - start_time
            }
    
    def _update_stock_list(self, markets: List[str] = None) -> Dict[str, Any]:
        """更新股票列表（复用原有逻辑）"""
        try:
            return self.stock_manager.update_stock_list(markets or ["SH", "SZ"])  # BJ股票已移除
        except Exception as e:
            self.logger.error(f"更新股票列表失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_sync_target_stocks(self, markets: List[str] = None, max_symbols: int = 0,
                               top_n_by: Optional[str] = None, top_n: int = 0,
                               amount_window_days: int = 5) -> List[Dict[str, Any]]:
        """获取同步目标股票（复用原有逻辑）"""
        try:
            # 这里可以复用原有的 DataSyncService._get_sync_target_stocks 逻辑
            # 为简化，先返回基本的股票列表
            with self.db_manager.get_conn() as conn:
                query = """
                SELECT symbol, name, market, list_date 
                FROM stocks 
                WHERE market IN ({})
                """.format(','.join(['?' for _ in (markets or ['SH', 'SZ'])]))  # BJ股票已移除
                
                df = pd.read_sql_query(query, conn, params=markets or ['SH', 'SZ'])
                
                if max_symbols > 0:
                    df = df.head(max_symbols)
                
                return df.to_dict('records')
                
        except Exception as e:
            self.logger.error(f"获取同步目标股票失败: {e}")
            return []
    
    def _prepare_sync_tasks(self, target_stocks: List[Dict[str, Any]], sync_type: str) -> List[SyncTask]:
        """准备同步任务"""
        tasks = []
        
        # 获取增量同步的起始日期映射
        latest_map = {} if sync_type == "full" else self.db_manager.get_latest_dates_by_symbol()
        
        for stock in target_stocks:
            symbol = stock['symbol']
            
            # 计算增量起始日期
            start_date = None
            if sync_type != "full":
                last_date = latest_map.get(symbol)
                if last_date:
                    try:
                        start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    except Exception:
                        start_date = None
            
            # 选择历史抓取周期
            period = "5y" if sync_type == "full" else "180d"
            
            task = SyncTask(
                symbol=symbol,
                stock_info=stock,
                sync_type=sync_type,
                period=period,
                start_date=start_date
            )
            tasks.append(task)
        
        return tasks
    
    def _execute_concurrent_sync(self, tasks: List[SyncTask], 
                               preferred_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行并发数据同步"""
        total_tasks = len(tasks)
        completed_tasks = 0
        successful_tasks = 0
        failed_tasks = []
        total_rows_inserted = 0
        source_usage_counts = {}
        attempt_summary = {}
        
        self.logger.info(f"开始并发处理 {total_tasks} 个同步任务，使用 {self.max_workers} 个线程")
        
        # 创建线程池执行任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._process_single_stock, task, preferred_sources): task 
                for task in tasks
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    
                    with self.stats_lock:
                        completed_tasks += 1
                        
                        if result.success:
                            successful_tasks += 1
                            total_rows_inserted += result.rows_inserted
                            
                            if result.source:
                                source_usage_counts[result.source] = source_usage_counts.get(result.source, 0) + 1
                        else:
                            failed_tasks.append({
                                'symbol': result.symbol,
                                'error': result.error,
                                'attempts': result.attempts or []
                            })
                        
                        # 汇总尝试统计
                        if result.attempts:
                            for att in result.attempts:
                                src = att.get('source', 'unknown')
                                status = att.get('status', 'unknown')
                                if src not in attempt_summary:
                                    attempt_summary[src] = {'success': 0, 'empty': 0, 'exception': 0, 'unknown': 0}
                                if status not in attempt_summary[src]:
                                    attempt_summary[src][status] = 0
                                attempt_summary[src][status] += 1
                    
                    # 定期报告进度
                    if completed_tasks % 50 == 0 or completed_tasks == total_tasks:
                        self.logger.info(f"进度: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.1f}%)")
                        
                except Exception as e:
                    with self.stats_lock:
                        completed_tasks += 1
                        failed_tasks.append({
                            'symbol': task.symbol,
                            'error': f"任务执行异常: {str(e)}",
                            'attempts': []
                        })
                    self.logger.error(f"处理任务 {task.symbol} 时发生异常: {e}")
        
        return {
            'total_stocks': total_tasks,
            'processed_stocks': completed_tasks,
            'successful_stocks': successful_tasks,
            'inserted_daily_rows': total_rows_inserted,
            'failed_stocks': failed_tasks,
            'source_usage_counts': source_usage_counts,
            'attempt_summary': attempt_summary,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0
        }
    
    def _process_single_stock(self, task: SyncTask, 
                            preferred_sources: Optional[List[str]] = None) -> SyncResult:
        """处理单只股票的数据同步"""
        start_time = time.time()
        
        try:
            # 创建线程本地的数据提供者实例
            price_provider = EnhancedDataProvider()
            
            # 设置优先数据源
            if preferred_sources:
                price_provider.set_preferred_sources(preferred_sources)
            
            # 获取历史数据
            df_hist = price_provider.get_stock_historical_data(
                symbol=task.symbol, 
                period=task.period
            )
            
            # 获取尝试信息
            attempts = getattr(price_provider, 'last_attempts', []) or []
            last_source = getattr(price_provider, 'last_used_source', None)
            
            if df_hist is not None and not df_hist.empty:
                # 增量过滤
                if task.start_date is not None and 'date' in df_hist.columns:
                    try:
                        df_hist['date'] = pd.to_datetime(df_hist['date'])
                        df_hist = df_hist[df_hist['date'] >= pd.to_datetime(task.start_date)]
                    except Exception as e:
                        self.logger.warning(f"日期过滤失败 {task.symbol}: {e}")
                
                if not df_hist.empty:
                    df_hist = df_hist.copy()
                    df_hist['symbol'] = task.symbol
                    
                    # 线程安全的数据库操作
                    with self.db_lock:
                        rows_inserted = self.db_manager.upsert_prices_daily(
                            df_hist,
                            symbol_col="symbol",
                            date_col="date",
                            source=last_source
                        )
                    
                    processing_time = time.time() - start_time
                    return SyncResult(
                        symbol=task.symbol,
                        success=True,
                        rows_inserted=rows_inserted,
                        source=last_source,
                        attempts=attempts,
                        processing_time=processing_time
                    )
                else:
                    # 有数据但被增量过滤后为空
                    processing_time = time.time() - start_time
                    return SyncResult(
                        symbol=task.symbol,
                        success=True,
                        rows_inserted=0,
                        source=last_source,
                        attempts=attempts,
                        processing_time=processing_time
                    )
            else:
                # 获取失败或空数据
                processing_time = time.time() - start_time
                return SyncResult(
                    symbol=task.symbol,
                    success=False,
                    error='no_data',
                    attempts=attempts,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return SyncResult(
                symbol=task.symbol,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _record_sync_status(self, sync_type: str, sync_result: Dict[str, Any]):
        """记录同步状态"""
        try:
            with self.db_manager.get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sync_status 
                    (sync_type, sync_time, total_stocks, successful_stocks, failed_stocks, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    sync_type,
                    datetime.now().isoformat(),
                    sync_result.get('total_stocks', 0),
                    sync_result.get('successful_stocks', 0),
                    len(sync_result.get('failed_stocks', [])),
                    sync_result.get('success_rate', 0)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"记录同步状态失败: {e}")


if __name__ == "__main__":
    # 测试并发数据同步服务
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    sync_service = ConcurrentDataSyncService(max_workers=8, db_batch_size=50)
    
    # 测试增量同步
    result = sync_service.sync_market_data(
        sync_type="incremental",
        markets=["SH", "SZ"],
        max_symbols=100,  # 测试100只股票
        max_workers=8
    )
    
    print(f"并发同步结果: {result}")