"""Cache Prefetch Scheduler
使用 APScheduler 预热 Redis L1 缓存，避免首次访问冷启动延迟。

该模块设计思路：
1. 使用 AsyncIOScheduler，兼容项目中大量 async 数据访问接口
2. 支持从配置或环境变量读取预取符号列表、时间范围、运行间隔
3. 提供 start/stop 接口，方便在应用启动与关闭时控制
4. 预取任务内部调用 UnifiedDataAccessLayer.get_historical_data，以保证 L0->L1 全链路缓存均被写入
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AsyncIOScheduler = None
    IntervalTrigger = None

# 避免循环导入，仅在类型检查时引入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.data.unified_data_access import UnifiedDataAccessLayer

logger = logging.getLogger(__name__)


def _parse_symbol_list(env_val: str) -> List[str]:
    """解析环境变量中的符号列表，逗号分隔"""
    if not env_val:
        return []
    return [s.strip() for s in env_val.split(',') if s.strip()]


class CachePrefetchScheduler:
    """使用 APScheduler 为 UnifiedDataAccessLayer 预热缓存"""

    def __init__(
        self,
        uda: "UnifiedDataAccessLayer",
        symbols: Optional[List[str]] = None,
        lookback_days: int = 365,
        interval_minutes: int = 60,
    ) -> None:
        """构造函数

        Args:
            uda: 已初始化的 UnifiedDataAccessLayer 实例
            symbols: 需要预取的股票代码列表；若为 None，则尝试从环境变量 PRELOAD_SYMBOLS 获取；仍为空则跳过预取。
            lookback_days: 向前预取的历史天数
            interval_minutes: 任务运行间隔分钟数
        """
        self.uda = uda
        if symbols is None:
            symbols = _parse_symbol_list(os.getenv("PRELOAD_SYMBOLS", ""))
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.interval_minutes = interval_minutes
        self.scheduler = AsyncIOScheduler()
        self._job = None

        if AsyncIOScheduler is None or IntervalTrigger is None:
            logger.warning("APScheduler 未安装，缓存预取功能被禁用。请运行 `pip install apscheduler` 启用该特性。")
            self.scheduler = None
        elif not self.symbols:
            logger.warning("CachePrefetchScheduler 初始化时未检测到待预取的股票代码列表，预取任务将不会运行。")

    async def _prefetch_job(self):
        """实际执行的预取协程"""
        if not self.symbols:
            return

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.lookback_days)
        logger.info(
            f"[CachePrefetchScheduler] 开始预取 {len(self.symbols)} 支股票数据，日期范围: {start_date} - {end_date}"
        )
        successes, failures = 0, 0
        for symbol in self.symbols:
            try:
                await self.uda.get_historical_data(symbol, start_date, end_date)
                successes += 1
            except Exception as e:
                logger.error(f"预取 {symbol} 失败: {e}")
                failures += 1
        logger.info(
            f"[CachePrefetchScheduler] 预取任务完成，成功 {successes} 支，失败 {failures} 支。"
        )

    def start(self):
        """启动定时预取任务"""
        if self.scheduler is None:
            logger.info("APScheduler 不可用，跳过预取任务启动。")
            return
        if not self.symbols:
            logger.info("无预取任务，CachePrefetchScheduler 未启动。")
            return

        trigger = IntervalTrigger(minutes=self.interval_minutes)
        self._job = self.scheduler.add_job(self._prefetch_job, trigger, next_run_time=datetime.now())
        self.scheduler.start()
        logger.info(
            "CachePrefetchScheduler 已启动，间隔 %d 分钟，符号数量 %d",
            self.interval_minutes,
            len(self.symbols),
        )

    def shutdown(self, wait: bool = True):
        """停止预取任务"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("CachePrefetchScheduler 已关闭")
