"""
并发数据同步服务模块 - 接入统一数据访问层与 Akshare 限频实现
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.data.providers.akshare_provider import AkshareDataProvider
from src.data.unified_data_access import DataAccessConfig, UnifiedDataAccessLayer

logger = logging.getLogger(__name__)


class ConcurrentDataSyncService:
    """并发数据同步服务"""

    def __init__(
        self,
        data_access: Optional[UnifiedDataAccessLayer] = None,
        provider: Optional[AkshareDataProvider] = None,
        *,
        max_workers: int = 4,
        config: Optional[Dict[str, Any]] = None,
        sync_delay: float = 0.6,
        step_days: int = 90,
        max_retries: int = 3,
        rollback_on_failure: bool = True,
    ):
        """
        创建并发数据同步服务

        Args:
            data_access: 统一数据访问层实例，默认自动创建
            provider: Akshare 数据提供器（共享限速器），默认自动创建
            max_workers: 最大并发协程数量
            config: 自定义数据访问配置
            sync_delay: 同步间隔，建议与 provider 限速保持一致
            step_days: 每次同步的最大区间天数
            max_retries: 单区间最大重试次数
            rollback_on_failure: 是否在失败时回滚该区间的数据
        """
        self.max_workers = max_workers
        self.sync_delay = sync_delay
        self.sync_step_days = step_days
        self.sync_max_retries = max_retries
        self.rollback_on_failure = rollback_on_failure
        self.sync_status: Dict[str, Any] = {}
        self.lock = threading.Lock()

        self.shared_provider = provider or AkshareDataProvider()
        if config is None:
            config_obj = DataAccessConfig(default_adjust_mode="origin")
        elif isinstance(config, DataAccessConfig):
            config_obj = config
        else:
            config_obj = DataAccessConfig(**config)

        self.config = config_obj
        self.data_access = data_access or UnifiedDataAccessLayer(config=self.config)

        # 注册 Akshare Provider，避免重复注册
        registered = any(
            getattr(p, "provider", None) is self.shared_provider
            for p in getattr(self.data_access.unified_provider, "primary_providers", [])
        )
        if not registered:
            self.data_access.unified_provider.add_akshare_provider_with_adjust(
                as_primary=True,
                provider=self.shared_provider,
            )

        logger.info(
            "ConcurrentDataSyncService 初始化完成: workers=%s, step_days=%s, max_retries=%s, delay=%.2fs",
            max_workers,
            step_days,
            max_retries,
            sync_delay,
        )

    async def sync_stock_data(self, stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """同步指定股票在给定日期区间内的数据"""
        if not stock_codes:
            return {
                "total_stocks": 0,
                "successful": 0,
                "failed": 0,
                "synced_rows": 0,
                "errors": [],
                "sync_time": datetime.now().isoformat(),
            }

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info("开始同步 %s 支股票: %s -> %s", len(stock_codes), start_date, end_date)

        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [
            asyncio.create_task(self._sync_single_symbol(semaphore, symbol, start_dt, end_dt))
            for symbol in stock_codes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summary = {
            "total_stocks": len(stock_codes),
            "successful": 0,
            "failed": 0,
            "synced_rows": 0,
            "errors": [],
            "sync_time": datetime.now().isoformat(),
        }

        for result in results:
            if isinstance(result, Exception):
                summary["failed"] += 1
                summary["errors"].append(str(result))
                continue

            if result.get("success"):
                summary["successful"] += 1
                summary["synced_rows"] += result.get("synced_rows", 0)
            else:
                summary["failed"] += 1
                if "error" in result:
                    summary["errors"].append(result["error"])

        logger.info(
            "同步完成: 成功 %s, 失败 %s, 新增记录 %s",
            summary["successful"],
            summary["failed"],
            summary["synced_rows"],
        )
        return summary

    async def _sync_single_symbol(
        self,
        semaphore: asyncio.Semaphore,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Dict[str, Any]:
        async with semaphore:
            with self.lock:
                self.sync_status[symbol] = {
                    "status": "syncing",
                    "last_sync": datetime.now().isoformat(),
                    "progress": 0,
                }

            try:
                stats = await self.data_access.sync_market_data_by_date(
                    symbol=symbol,
                    start_date=start_dt,
                    end_date=end_dt,
                    step_days=self.sync_step_days,
                    delay=self.sync_delay,
                    max_retries=self.sync_max_retries,
                    rollback_on_failure=self.rollback_on_failure,
                )

                with self.lock:
                    self.sync_status[symbol] = {
                        "status": "completed",
                        "last_sync": datetime.now().isoformat(),
                        "progress": 100,
                        "synced_rows": stats.get("synced", 0),
                    }

                return {
                    "success": True,
                    "symbol": symbol,
                    "synced_rows": stats.get("synced", 0),
                }

            except Exception as exc:
                logger.error("同步 %s 失败: %s", symbol, exc)
                with self.lock:
                    self.sync_status[symbol] = {
                        "status": "failed",
                        "last_sync": datetime.now().isoformat(),
                        "progress": 0,
                        "error": str(exc),
                    }
                return {
                    "success": False,
                    "symbol": symbol,
                    "synced_rows": 0,
                    "error": str(exc),
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
        return {
            "total_syncs": total_syncs,
            "completed_syncs": completed_syncs,
            "syncing_syncs": syncing_syncs,
            "success_rate": completed_syncs / total_syncs if total_syncs > 0 else 0,
            "max_workers": self.max_workers,
            "service_status": "running",
        }

    def shutdown(self) -> None:
        """关闭服务"""
        logger.info("ConcurrentDataSyncService 已关闭")


# 全局服务实例
_data_sync_service = None

def get_data_sync_service() -> ConcurrentDataSyncService:
    """获取数据同步服务实例"""
    global _data_sync_service
    if _data_sync_service is None:
        _data_sync_service = ConcurrentDataSyncService()
    return _data_sync_service
