"""
并发数据同步服务模块 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class ConcurrentDataSyncService:
    """并发数据同步服务"""

    def __init__(self, data_provider=None, max_workers: int = 4, config: Optional[Dict[str, Any]] = None, **kwargs):
        """创建并发数据同步服务

        Args:
            data_provider: 统一数据访问层或兼容的数据提供器实例，外部注入避免类内部重复创建
            max_workers: 线程池大小
            config: 其他配置
            **kwargs: 额外参数，如 db_batch_size
        """
        self.data_provider = data_provider
        self.max_workers = max_workers
        self.config = config or {}
        self.db_batch_size = kwargs.get('db_batch_size', 100)  # 支持db_batch_size参数
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.sync_status: Dict[str, Any] = {}
        self.lock = threading.Lock()

        logger.info(
            f"ConcurrentDataSyncService 初始化完成，工作线程数: {max_workers}, 批次大小: {self.db_batch_size}"
        )

    async def sync_stock_data(self, stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """同步股票数据"""
        logger.info(f"开始同步 {len(stock_codes)} 只股票的数据")
        
        # 将股票代码分组以支持并发处理
        batch_size = min(10, max(1, len(stock_codes) // self.max_workers))
        batches = [stock_codes[i:i + batch_size] for i in range(0, len(stock_codes), batch_size)]
        
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._sync_batch(batch, start_date, end_date))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        sync_summary = {
            'total_stocks': len(stock_codes),
            'successful': 0,
            'failed': 0,
            'errors': [],
            'sync_time': datetime.now().isoformat()
        }
        
        for result in results:
            if isinstance(result, Exception):
                sync_summary['errors'].append(str(result))
                sync_summary['failed'] += 1
            else:
                if result.get('success', False):
                    sync_summary['successful'] += result.get('count', 0)
                else:
                    sync_summary['failed'] += 1
                    if 'error' in result:
                        sync_summary['errors'].append(result['error'])
        
        logger.info(f"数据同步完成: 成功 {sync_summary['successful']}, 失败 {sync_summary['failed']}")
        return sync_summary
    
    async def _sync_batch(self, stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """同步一批股票数据"""
        try:
            # 模拟数据同步过程
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            with self.lock:
                for code in stock_codes:
                    self.sync_status[code] = {
                        'status': 'syncing',
                        'last_sync': datetime.now().isoformat(),
                        'progress': 0
                    }
            
            # 模拟数据处理
            await asyncio.sleep(0.2)
            
            # 更新状态
            with self.lock:
                for code in stock_codes:
                    self.sync_status[code] = {
                        'status': 'completed',
                        'last_sync': datetime.now().isoformat(),
                        'progress': 100
                    }
            
            return {
                'success': True,
                'count': len(stock_codes),
                'codes': stock_codes
            }
            
        except Exception as e:
            logger.error(f"同步批次失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'count': 0
            }
    
    def get_sync_status(self, stock_code: Optional[str] = None) -> Dict[str, Any]:
        """获取同步状态"""
        with self.lock:
            if stock_code:
                return self.sync_status.get(stock_code, {
                    'status': 'unknown',
                    'last_sync': None,
                    'progress': 0
                })
            else:
                return self.sync_status.copy()
    
    def reset_sync_status(self, stock_codes: Optional[List[str]] = None) -> None:
        """重置同步状态"""
        with self.lock:
            if stock_codes:
                for code in stock_codes:
                    if code in self.sync_status:
                        del self.sync_status[code]
            else:
                self.sync_status.clear()
        
        logger.info(f"同步状态已重置")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        with self.lock:
            total_syncs = len(self.sync_status)
            completed_syncs = sum(1 for status in self.sync_status.values() 
                                if status.get('status') == 'completed')
            syncing_syncs = sum(1 for status in self.sync_status.values() 
                                if status.get('status') == 'syncing')
        
        return {
            'total_syncs': total_syncs,
            'completed_syncs': completed_syncs,
            'syncing_syncs': syncing_syncs,
            'success_rate': completed_syncs / total_syncs if total_syncs > 0 else 0,
            'max_workers': self.max_workers,
            'service_status': 'running'
        }
    
    def shutdown(self) -> None:
        """关闭服务"""
        self.executor.shutdown(wait=True)
        logger.info("ConcurrentDataSyncService 已关闭")


# 全局服务实例
_data_sync_service = None

def get_data_sync_service() -> ConcurrentDataSyncService:
    """获取数据同步服务实例"""
    global _data_sync_service
    if _data_sync_service is None:
        _data_sync_service = ConcurrentDataSyncService()
    return _data_sync_service