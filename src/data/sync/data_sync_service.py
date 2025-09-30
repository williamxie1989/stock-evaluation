"""
数据同步服务 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import queue
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SyncTask:
    """同步任务数据结构"""
    symbol: str
    stock_info: Dict[str, Any]
    sync_type: str
    period: str = "5y"
    start_date: Optional[str] = None

class DataSyncService:
    """数据同步服务"""
    
    def __init__(self, data_provider=None, max_workers: int = 4, db_batch_size: int = 100):
        self.data_provider = data_provider
        self.max_workers = max_workers
        self.db_batch_size = db_batch_size
        self.sync_status = {}
        self.sync_lock = threading.Lock()
        self.sync_history = []
        self.is_syncing = False
        
        logger.info(f"DataSyncService initialized: max_workers={max_workers}, db_batch_size={db_batch_size}")
    
    def sync_stock_data(self, stock_codes: List[str], 
                       start_date: str, end_date: str,
                       sync_type: str = 'incremental') -> Dict[str, Any]:
        """同步股票数据"""
        try:
            if not self.data_provider:
                return {'error': '数据提供器未初始化'}
            
            logger.info(f"开始同步股票数据: {len(stock_codes)} 支股票, {sync_type} 模式")
            
            sync_result = {
                'success': [],
                'failed': [],
                'skipped': [],
                'total': len(stock_codes),
                'start_time': datetime.now(),
                'end_time': None
            }
            
            # 同步股票数据
            for stock_code in stock_codes:
                try:
                    # 获取股票数据
                    stock_data = self.data_provider.get_stock_data(
                        stock_code, start_date, end_date
                    )
                    
                    if stock_data is not None and not stock_data.empty:
                        sync_result['success'].append(stock_code)
                        logger.debug(f"股票数据同步成功: {stock_code}")
                    else:
                        sync_result['failed'].append(stock_code)
                        logger.warning(f"股票数据同步失败: {stock_code}")
                        
                except Exception as e:
                    sync_result['failed'].append(stock_code)
                    logger.error(f"同步股票 {stock_code} 失败: {e}")
            
            sync_result['end_time'] = datetime.now()
            sync_result['duration'] = (sync_result['end_time'] - sync_result['start_time']).total_seconds()
            
            # 记录同步历史
            self._record_sync_history('stock_data', sync_result)
            
            logger.info(f"股票数据同步完成: 成功 {len(sync_result['success'])}, 失败 {len(sync_result['failed'])}")
            return sync_result
            
        except Exception as e:
            logger.error(f"股票数据同步失败: {e}")
            return {'error': str(e)}
    
    def sync_realtime_data(self, stock_codes: List[str]) -> Dict[str, Any]:
        """同步实时数据"""
        try:
            if not self.data_provider:
                return {'error': '数据提供器未初始化'}
            
            logger.info(f"开始同步实时数据: {len(stock_codes)} 支股票")
            
            sync_result = {
                'success': [],
                'failed': [],
                'total': len(stock_codes),
                'start_time': datetime.now(),
                'end_time': None
            }
            
            # 批量获取实时数据
            realtime_data = self.data_provider.get_batch_realtime_data(stock_codes)
            
            if realtime_data is not None and not realtime_data.empty:
                # 处理实时数据
                for stock_code in stock_codes:
                    if stock_code in realtime_data.index:
                        sync_result['success'].append(stock_code)
                    else:
                        sync_result['failed'].append(stock_code)
            else:
                sync_result['failed'] = stock_codes
                logger.warning("实时数据获取失败")
            
            sync_result['end_time'] = datetime.now()
            sync_result['duration'] = (sync_result['end_time'] - sync_result['start_time']).total_seconds()
            
            # 记录同步历史
            self._record_sync_history('realtime_data', sync_result)
            
            logger.info(f"实时数据同步完成: 成功 {len(sync_result['success'])}, 失败 {len(sync_result['failed'])}")
            return sync_result
            
        except Exception as e:
            logger.error(f"实时数据同步失败: {e}")
            return {'error': str(e)}
    
    def sync_financial_data(self, stock_codes: List[str]) -> Dict[str, Any]:
        """同步财务数据"""
        try:
            if not self.data_provider:
                return {'error': '数据提供器未初始化'}
            
            logger.info(f"开始同步财务数据: {len(stock_codes)} 支股票")
            
            sync_result = {
                'success': [],
                'failed': [],
                'total': len(stock_codes),
                'start_time': datetime.now(),
                'end_time': None
            }
            
            # 同步财务数据
            for stock_code in stock_codes:
                try:
                    # 获取财务数据
                    financial_data = self.data_provider.get_financial_data(stock_code)
                    
                    if financial_data is not None and not financial_data.empty:
                        sync_result['success'].append(stock_code)
                        logger.debug(f"财务数据同步成功: {stock_code}")
                    else:
                        sync_result['failed'].append(stock_code)
                        logger.warning(f"财务数据同步失败: {stock_code}")
                        
                except Exception as e:
                    sync_result['failed'].append(stock_code)
                    logger.error(f"同步财务数据 {stock_code} 失败: {e}")
            
            sync_result['end_time'] = datetime.now()
            sync_result['duration'] = (sync_result['end_time'] - sync_result['start_time']).total_seconds()
            
            # 记录同步历史
            self._record_sync_history('financial_data', sync_result)
            
            logger.info(f"财务数据同步完成: 成功 {len(sync_result['success'])}, 失败 {len(sync_result['failed'])}")
            return sync_result
            
        except Exception as e:
            logger.error(f"财务数据同步失败: {e}")
            return {'error': str(e)}
    
    def sync_market_overview(self) -> Dict[str, Any]:
        """同步市场概览"""
        try:
            if not self.data_provider:
                return {'error': '数据提供器未初始化'}
            
            logger.info("开始同步市场概览")
            
            sync_result = {
                'success': [],
                'failed': [],
                'start_time': datetime.now(),
                'end_time': None
            }
            
            try:
                # 获取市场概览
                market_overview = self.data_provider.get_market_overview()
                
                if market_overview is not None and not market_overview.empty:
                    sync_result['success'].append('market_overview')
                    logger.info("市场概览同步成功")
                else:
                    sync_result['failed'].append('market_overview')
                    logger.warning("市场概览同步失败")
                    
            except Exception as e:
                sync_result['failed'].append('market_overview')
                logger.error(f"同步市场概览失败: {e}")
            
            sync_result['end_time'] = datetime.now()
            sync_result['duration'] = (sync_result['end_time'] - sync_result['start_time']).total_seconds()
            
            # 记录同步历史
            self._record_sync_history('market_overview', sync_result)
            
            logger.info("市场概览同步完成")
            return sync_result
            
        except Exception as e:
            logger.error(f"市场概览同步失败: {e}")
            return {'error': str(e)}
    
    def sync_all_data(self, stock_codes: List[str], 
                     start_date: str, end_date: str) -> Dict[str, Any]:
        """同步所有数据"""
        try:
            logger.info("开始全量数据同步")
            
            all_sync_result = {
                'stock_data': None,
                'realtime_data': None,
                'financial_data': None,
                'market_overview': None,
                'start_time': datetime.now(),
                'end_time': None,
                'total_duration': None
            }
            
            # 设置同步状态
            with self.sync_lock:
                self.is_syncing = True
            
            try:
                # 同步股票历史数据
                all_sync_result['stock_data'] = self.sync_stock_data(
                    stock_codes, start_date, end_date
                )
                
                # 同步实时数据
                all_sync_result['realtime_data'] = self.sync_realtime_data(stock_codes)
                
                # 同步财务数据
                all_sync_result['financial_data'] = self.sync_financial_data(stock_codes)
                
                # 同步市场概览
                all_sync_result['market_overview'] = self.sync_market_overview()
                
            finally:
                # 重置同步状态
                with self.sync_lock:
                    self.is_syncing = False
            
            all_sync_result['end_time'] = datetime.now()
            all_sync_result['total_duration'] = (
                all_sync_result['end_time'] - all_sync_result['start_time']
            ).total_seconds()
            
            # 统计总体结果
            total_success = 0
            total_failed = 0
            
            for sync_type, result in all_sync_result.items():
                if isinstance(result, dict) and 'success' in result:
                    total_success += len(result.get('success', []))
                    total_failed += len(result.get('failed', []))
            
            all_sync_result['summary'] = {
                'total_success': total_success,
                'total_failed': total_failed,
                'success_rate': total_success / (total_success + total_failed) if (total_success + total_failed) > 0 else 0
            }
            
            logger.info(f"全量数据同步完成: 总耗时 {all_sync_result['total_duration']:.2f} 秒")
            return all_sync_result
            
        except Exception as e:
            logger.error(f"全量数据同步失败: {e}")
            return {'error': str(e)}
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        try:
            with self.sync_lock:
                is_syncing = self.is_syncing
            
            status = {
                'is_syncing': is_syncing,
                'sync_history': self.sync_history[-10:],  # 最近10次同步记录
                'total_syncs': len(self.sync_history),
                'last_sync': self.sync_history[-1] if self.sync_history else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取同步状态失败: {e}")
            return {'error': str(e)}
    
    def _record_sync_history(self, sync_type: str, result: Dict[str, Any]):
        """记录同步历史"""
        try:
            history_entry = {
                'sync_type': sync_type,
                'timestamp': datetime.now(),
                'result': result,
                'success_count': len(result.get('success', [])),
                'failed_count': len(result.get('failed', [])),
                'duration': result.get('duration', 0)
            }
            
            self.sync_history.append(history_entry)
            
            # 保持历史记录在合理范围内
            if len(self.sync_history) > 1000:
                self.sync_history = self.sync_history[-500:]
            
        except Exception as e:
            logger.error(f"记录同步历史失败: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """清理旧数据"""
        try:
            logger.info(f"开始清理旧数据，保留最近 {days_to_keep} 天")
            
            cleanup_result = {
                'sync_history_cleaned': 0,
                'status_cleaned': 0,
                'start_time': datetime.now(),
                'end_time': None
            }
            
            # 清理同步历史
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            original_count = len(self.sync_history)
            self.sync_history = [
                entry for entry in self.sync_history
                if entry.get('timestamp', datetime.now()) > cutoff_date
            ]
            cleanup_result['sync_history_cleaned'] = original_count - len(self.sync_history)
            
            # 清理状态缓存
            original_status_count = len(self.sync_status)
            self.sync_status = {
                key: value for key, value in self.sync_status.items()
                if isinstance(value, dict) and value.get('timestamp', datetime.now()) > cutoff_date
            }
            cleanup_result['status_cleaned'] = original_status_count - len(self.sync_status)
            
            cleanup_result['end_time'] = datetime.now()
            cleanup_result['duration'] = (
                cleanup_result['end_time'] - cleanup_result['start_time']
            ).total_seconds()
            
            logger.info(f"旧数据清理完成: 同步历史清理 {cleanup_result['sync_history_cleaned']}, "
                       f"状态清理 {cleanup_result['status_cleaned']}")
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            return {'error': str(e)}
    
    def sync_market_data(self, 
                        sync_type: str = "incremental",
                        markets: List[str] = None,
                        max_symbols: int = 0,
                        preferred_sources: Optional[List[str]] = None,
                        top_n_by: Optional[str] = None,
                        top_n: int = 0,
                        amount_window_days: int = 5) -> Dict[str, Any]:
        """
        同步市场数据 - 新增方法以匹配API调用
        
        Args:
            sync_type: 同步类型 ("full" || "incremental")
            markets: 要同步的市场列表
            max_symbols: 最大处理股票数量
            preferred_sources: 优先数据源列表
            top_n_by: 筛选策略
            top_n: 筛选数量
            amount_window_days: 成交量窗口天数
        """
        start_time = time.time()
        
        logger.info(f"开始市场数据同步: sync_type={sync_type}, markets={markets}")
        
        try:
            # 获取股票列表
            stock_codes = self._get_stock_codes_for_sync(markets, max_symbols)
            
            if not stock_codes:
                return {
                    'success': 0,
                    'error': '没有找到需要同步的股票',
                    'total_time_seconds': time.time() - start_time
                }
            
            # 执行同步
            sync_result = self.sync_stock_data(
                stock_codes=stock_codes,
                start_date=self._get_start_date(sync_type),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                sync_type=sync_type
            )
            
            total_time = time.time() - start_time
            
            return {
                'success': 1,
                'sync_type': sync_type,
                'markets': markets or ['ALL'],
                'total_stocks': len(stock_codes),
                'successful_stocks': len(sync_result.get('success', [])),
                'failed_stocks': len(sync_result.get('failed', [])),
                'total_time_seconds': round(total_time, 2),
                'throughput_stocks_per_second': round(len(stock_codes) / max(total_time, 0.1), 2),
                'sync_details': sync_result
            }
            
        except Exception as e:
            logger.error(f"市场数据同步失败: {e}")
            return {
                'success': 0,
                'error': str(e),
                'total_time_seconds': time.time() - start_time
            }
    
    def _get_stock_codes_for_sync(self, markets: List[str] = None, max_symbols: int = 0) -> List[str]:
        """获取需要同步的股票代码列表"""
        try:
            # 优先使用get_all_stock_list方法，回退到get_stock_list
            if hasattr(self.data_provider, 'get_all_stock_list'):
                logger.info("使用get_all_stock_list方法获取股票列表")
                stock_list = self.data_provider.get_all_stock_list()
                logger.info(f"get_all_stock_list返回类型: {type(stock_list)}")
                if stock_list is not None:
                    logger.info(f"get_all_stock_list返回数据量: {len(stock_list) if hasattr(stock_list, '__len__') else 'unknown'}")
                    # get_all_stock_list返回的是DataFrame，需要转换为字典列表
                    if hasattr(stock_list, 'to_dict'):
                        stock_list = stock_list.to_dict('records')
                        logger.info(f"转换为字典列表后数据量: {len(stock_list)}")
            elif hasattr(self.data_provider, 'get_stock_list'):
                logger.info("使用get_stock_list方法获取股票列表")
                stock_list = self.data_provider.get_stock_list()
            else:
                # 如果没有get_stock_list方法，返回空列表
                logger.warning("数据提供器没有get_stock_list或get_all_stock_list方法")
                return []
            

            
            # 过滤市场（不区分大小写）
            if markets:
                markets_upper = [m.upper() for m in markets]
                stock_list = [stock for stock in stock_list if stock.get('market', '').upper() in markets_upper]
            
            # 限制数量
            if max_symbols > 0:
                stock_list = stock_list[:max_symbols]
            
            # 提取股票代码 - 优先使用code字段，回退到symbol字段
            stock_codes = []
            for stock in stock_list:
                code = stock.get('code') or stock.get('symbol')
                if code:
                    stock_codes.append(code)
            
            logger.info(f"获取到 {len(stock_codes)} 只股票代码")
            if stock_codes:
                logger.info(f"前5个股票代码: {stock_codes[:5]}")
            return stock_codes
            
        except Exception as e:
            logger.error(f"获取股票代码列表失败: {e}")
            return []
    
    def _get_start_date(self, sync_type: str) -> str:
        """根据同步类型获取开始日期"""
        if sync_type == "full":
            # 全量同步：获取5年数据
            return (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        else:
            # 增量同步：获取最近180天数据
            return (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    def reset(self):
        """重置数据同步服务"""
        try:
            with self.sync_lock:
                self.sync_status.clear()
                self.sync_history.clear()
                self.is_syncing = False
            
            logger.info("数据同步服务已重置")
            
        except Exception as e:
            logger.error(f"重置数据同步服务失败: {e}")