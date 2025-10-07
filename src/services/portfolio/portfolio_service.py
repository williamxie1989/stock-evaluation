"""Portfolio service：生成当前模拟组合持仓列表

基于 trading.portfolio.portfolio_pipeline.PortfolioPipeline ，
在没有真实持仓数据时，按照 pipeline 的最新选股结果进行等权建仓，
返回前端所需的持仓字段。

若后续接入真实交易系统，可替换此实现。
"""
from __future__ import annotations

import logging
from time import time
from functools import lru_cache
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

# 业务依赖
from src.trading.portfolio.portfolio_pipeline import PortfolioPipeline, Holding

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_pipeline(initial_capital: float, top_n: int) -> PortfolioPipeline:  # noqa: D401
    """缓存化获取 PortfolioPipeline 实例，避免多次初始化开销"""
    return PortfolioPipeline(initial_capital=initial_capital, top_n=top_n)


# 简易缓存：避免高频重复计算
_CACHE: dict[str, Any] = {}
_CACHE_TTL_SECONDS = 3600  # 1小时


def _get_cache_key(as_of: pd.Timestamp, top_n: int, initial_capital: float) -> str:
    return f"{as_of.strftime('%Y-%m-%d')}_{top_n}_{int(initial_capital)}"


def _holding_to_dict(h: Holding, price: float) -> Dict[str, Any]:
    return {
        "symbol": h.symbol,
        "weight": round(h.weight, 6),
        "shares": round(h.shares, 2),
        "cost_price": round(price, 3),  # 此处成本价即当前价格（模拟持仓）
        "current_price": round(price, 3),
        "pnl_pct": 0.0,  # 模拟持仓成本=现价，因此盈亏0
    }


def generate_portfolio_holdings(
    as_of_date: Optional[str | datetime] = None,
    *,
    top_n: int = 20,
    initial_capital: float = 100_000.0,
) -> Dict[str, Any]:
    """生成模拟组合持仓列表

    Args:
        as_of_date: 参考日期(YYYY-MM-DD)，默认为今天
        top_n: 选股数量
        initial_capital: 用于等权建仓的初始资金

    Returns:
        dict: {"success":1, "data":{"holdings": [...], "generated_at": "..."}}
    """
    try:
        if as_of_date is None:
            as_of_date_dt = pd.Timestamp(datetime.now().date())
        else:
            as_of_date_dt = pd.Timestamp(as_of_date)
    except Exception as e:
        logger.warning(f"无法解析 as_of_date {as_of_date}: {e}，使用今日")
        as_of_date_dt = pd.Timestamp(datetime.now().date())

    pipeline = _get_pipeline(initial_capital=initial_capital, top_n=top_n)

    # 缓存检查
    cache_key = _get_cache_key(as_of_date_dt, top_n, initial_capital)
    cached = _CACHE.get(cache_key)
    # 仅当缓存未过期且对应的 pipeline 实例未变化时才使用缓存
    if (
        cached
        and (time() - cached["ts"]) < _CACHE_TTL_SECONDS
        and cached.get("pipeline_id") == id(pipeline)
    ):
        return cached["resp"]
    try:
        # 选股
        picks = pipeline.pick_stocks(as_of_date=as_of_date_dt, top_n=top_n)
        if not picks:
            return {"success": 1, "data": {"holdings": [], "generated_at": as_of_date_dt.strftime("%Y-%m-%d")}}
        # 建仓
        holdings_objs = pipeline._equal_weight_holdings(picks, as_of_date=as_of_date_dt, capital=initial_capital)
        holdings: List[Dict[str, Any]] = []
        for h in holdings_objs:
            # 获取当前价格
            df_price = pipeline._fetch_history(h.symbol, end_date=as_of_date_dt, days=5)
            if df_price is None or df_price.empty:
                price = 0.0
            else:
                price = float(df_price["close"].iloc[-1])
            holdings.append(_holding_to_dict(h, price))

        resp = {
            "success": 1,
            "data": {
                "holdings": holdings,
                "generated_at": as_of_date_dt.strftime("%Y-%m-%d"),
            },
        }
        # 缓存响应，并记录 pipeline 标识以确保测试中 patch 后不会误用旧缓存
        _CACHE[cache_key] = {"resp": resp, "ts": time(), "pipeline_id": id(pipeline)}
        return resp
    except Exception as e:
        logger.error(f"生成持仓失败: {e}")
        return {"success": 0, "error": str(e)}