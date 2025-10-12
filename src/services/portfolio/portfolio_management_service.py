"""Portfolio management service: database-backed CRUD with valuation refresh."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import os
import threading
import pandas as pd
from zoneinfo import ZoneInfo

from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.services.stock.stock_list_manager import StockListManager
from src.trading.portfolio.portfolio_pipeline import (
    Holding,
    PickResult,
    PortfolioPipeline,
    resolve_candidate_limit,
)
from src.apps.scripts.selector_service import IntelligentStockSelector
from src.services.portfolio.portfolio_utils import is_trading_day, next_trading_day

logger = logging.getLogger(__name__)

_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "Asia/Shanghai")
try:
    _TIMEZONE = ZoneInfo(_TIMEZONE_NAME)
except Exception:
    logger.warning("Invalid APP_TIMEZONE '%s', fallback到UTC", _TIMEZONE_NAME)
    _TIMEZONE = ZoneInfo("UTC")


# =========================================================================
# Dataclasses
# =========================================================================


@dataclass
class PriceQuote:
    latest_price: float
    previous_price: Optional[float]
    last_trade_at: Optional[datetime]


@dataclass
class PortfolioValuation:
    nav_total: float
    nav_value: float
    daily_return_pct: float
    total_return_pct: float
    last_valued_at: Optional[datetime]


@dataclass
class PortfolioHoldingSnapshot:
    symbol: str
    code: str
    name: str
    weight: float
    shares: float
    cost_price: float
    opened_at: datetime

    def to_dict(self, quote: Optional[PriceQuote], as_of: datetime) -> Dict[str, Any]:
        latest_price = quote.latest_price if quote else self.cost_price
        previous_price = quote.previous_price if quote else None
        cost = self.cost_price if self.cost_price > 0 else latest_price
        pnl_pct = ((latest_price - cost) / cost) if cost else 0.0
        daily_return_pct = (
            ((latest_price - previous_price) / previous_price)
            if previous_price and previous_price > 0
            else 0.0
        )
        holding_days = max((as_of - self.opened_at).days, 0)
        latest_value = latest_price * self.shares
        data = {
            "symbol": self.symbol,
            "code": self.code,
            "name": self.name,
            "weight": round(self.weight, 6),
            "weight_pct": round(self.weight * 100, 2),
            "shares": round(self.shares, 4),
            "cost_price": round(cost, 4),
            "latest_price": round(latest_price, 4),
            "latest_value": round(latest_value, 2),
            "pnl_pct": round(pnl_pct, 6),
            "daily_return_pct": round(daily_return_pct, 6),
            "holding_days": holding_days,
            "opened_at": self.opened_at.isoformat(),
        }
        if quote and quote.last_trade_at:
            data["last_trade_at"] = quote.last_trade_at.isoformat()
        return data


@dataclass
class PortfolioInfo:
    id: int
    name: str
    created_at: datetime
    top_n: int
    initial_capital: float
    holdings_count: int
    benchmark: str
    risk_level: str
    strategy_tags: List[str]

    def summary(self, valuation: PortfolioValuation) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": _safe_isoformat(self.created_at),
            "top_n": self.top_n,
            "initial_capital": round(self.initial_capital, 2),
            "holdings_count": self.holdings_count,
            "benchmark": self.benchmark,
            "risk_level": self.risk_level,
            "strategy_tags": self.strategy_tags,
            "nav_value": round(valuation.nav_value, 6),
            "total_value": round(valuation.nav_total, 2),
            "daily_return_pct": round(valuation.daily_return_pct, 6),
            "total_return_pct": round(valuation.total_return_pct, 6),
            "last_valued_at": _safe_isoformat(valuation.last_valued_at),
        }


@dataclass
class PortfolioDetail(PortfolioInfo):
    holdings: List[PortfolioHoldingSnapshot]
    notes: Optional[str] = None
    rebalance_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(
        self,
        valuation: PortfolioValuation,
        quotes: Dict[str, PriceQuote],
        nav_history: List[Dict[str, Any]],
        as_of: datetime,
    ) -> Dict[str, Any]:
        base = self.summary(valuation)
        base["metrics"] = {
            "nav_value": base["nav_value"],
            "total_value": base["total_value"],
            "daily_return_pct": base["daily_return_pct"],
            "total_return_pct": base["total_return_pct"],
            "initial_capital": round(self.initial_capital, 2),
            "last_valued_at": base["last_valued_at"],
        }
        base["holdings"] = [
            holding.to_dict(quotes.get(holding.symbol), as_of) for holding in self.holdings
        ]
        base["rebalance_history"] = self.rebalance_history
        base["notes"] = self.notes
        base["nav_history"] = nav_history
        return base


@dataclass
class PortfolioTrade:
    """交易流水数据模型"""
    id: int
    portfolio_id: int
    symbol: str
    side: str  # 'BUY' 或 'SELL'
    qty: int
    price: float
    fee: float
    trade_ts: datetime
    note: Optional[str] = None
    related_event_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": round(self.price, 6),
            "fee": round(self.fee, 6),
            "trade_ts": _safe_isoformat(self.trade_ts),
            "note": self.note,
            "related_event_id": self.related_event_id
        }


# =========================================================================
# Globals & helpers
# =========================================================================

_db_manager = UnifiedDatabaseManager()
_stock_list_manager = StockListManager()

_AUTO_SELECTOR: Optional[IntelligentStockSelector] = None
_SELECTOR_LOCK = threading.Lock()
_AUTO_PICK_CACHE: Dict[str, Any] = {}
_AUTO_PICK_CACHE_TTL = int(os.getenv("PORTFOLIO_AUTO_CACHE_SECONDS", "600"))
_AUTO_SYMBOL_LIMIT_DEFAULT = int(os.getenv("PORTFOLIO_AUTO_SYMBOL_LIMIT", "600"))
_SELECTOR_READY = False

_REFRESH_THRESHOLD_MINUTES = int(os.getenv("PORTFOLIO_REFRESH_MINUTES", "60"))

_FALLBACK_LOOKBACK_DAYS = int(os.getenv("PORTFOLIO_FALLBACK_LOOKBACK_DAYS", "120"))
_FALLBACK_MIN_ROWS = int(os.getenv("PORTFOLIO_FALLBACK_MIN_ROWS", "15"))
def _get_rebalance_interval_days() -> int:
    """动态获取调仓间隔天数，支持测试环境动态设置"""
    return int(os.getenv("PORTFOLIO_REBALANCE_INTERVAL_DAYS", "30"))


def _now() -> datetime:
    now_local = datetime.now(_TIMEZONE)
    return now_local.replace(tzinfo=None)


def _json_dumps(data: Any) -> Optional[str]:
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _json_loads(text: Any, default: Any = None) -> Any:
    if text in (None, "", b""):
        return default
    if isinstance(text, (list, dict)):
        return text
    try:
        return json.loads(text)
    except Exception:
        return default


def _safe_isoformat(dt: Optional[datetime]) -> Optional[str]:
    """Return ISO string for dt but treat obviously-bad dates (epoch/1970) as None."""
    if dt is None:
        return None
    try:
        # reject sentinel/epoch-like dates
        if getattr(dt, "year", 0) and int(getattr(dt, "year", 0)) < 2000:
            return None
        return dt.isoformat()
    except Exception:
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    try:
        return datetime.fromisoformat(value)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except Exception:
                continue
    return None


def _placeholder() -> str:
    return "%s" if _db_manager.db_type == "mysql" else "?"


def _placeholders(count: int) -> str:
    return ", ".join([_placeholder()] * count)


def _infer_risk_level(top_n: int) -> str:
    if top_n <= 10:
        return "高"
    if top_n <= 25:
        return "中"
    return "低"


def _strip_suffix(symbol: str) -> str:
    return symbol.split(".")[0] if symbol and "." in symbol else symbol


@lru_cache(maxsize=2048)
def _resolve_stock_name(symbol: str, code: str) -> str:
    """优先从 stocks 表获取股票简称，失败时回退到 StockListManager。"""
    symbol = (symbol or "").strip()
    code = (code or _strip_suffix(symbol or "")).strip()
    placeholder = _placeholder()
    candidates: List[str] = []
    if symbol:
        candidates.append(symbol)
    if code and code not in candidates:
        candidates.append(code)
    # 附加常见交易所后缀，避免代码未携带市场信息时匹配失败
    suffixes = (".SH", ".SZ", ".BJ")
    for suffix in suffixes:
        candidate = f"{code}{suffix}"
        if code and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            rows = _db_manager.execute_query(
                f"SELECT name FROM stocks WHERE symbol = {placeholder} LIMIT 1",
                (candidate,),
            )
            if rows:
                name = rows[0].get("name")
                if name:
                    return name
        except Exception:
            logger.debug("查询股票名称失败: %s", candidate, exc_info=True)
            break

    if code:
        try:
            pattern = f"%{code}"
            rows = _db_manager.execute_query(
                f"SELECT name FROM stocks WHERE symbol LIKE {placeholder} ORDER BY symbol LIMIT 1",
                (pattern,),
            )
            if rows:
                name = rows[0].get("name")
                if name:
                    return name
        except Exception:
            logger.debug("模糊查询股票名称失败: %s", code, exc_info=True)

    info = _stock_list_manager.get_stock_info(code)
    if info and info.get("name"):
        return info["name"]
    fallback = code or symbol or "未知股票"
    return f"股票{fallback}"


@lru_cache(maxsize=8)
def _get_pipeline(initial_capital: float, top_n: int) -> PortfolioPipeline:
    return PortfolioPipeline(initial_capital=initial_capital, top_n=top_n)


def _generate_rebalance_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval_days: int,
) -> pd.DatetimeIndex:
    """基于起始日生成等间隔的调仓日期序列。"""
    interval = max(int(interval_days), 1)
    dates: List[pd.Timestamp] = []
    current = start_date.normalize()
    end_norm = end_date.normalize()
    dates.append(current)
    while True:
        current = current + pd.Timedelta(days=interval)
        if current > end_norm:
            break
        dates.append(current)
    return pd.DatetimeIndex(dates)


def _format_rebalance_holdings(
    holdings: List[Dict[str, Any]],
    initial_capital: float = 0.0,
    commission_rate: float = 0.0,
    adjust_mode: str = None
) -> List[Dict[str, Any]]:
    """标准化调仓事件中的持仓数据，补充名称、收益率与市值。"""
    formatted: List[Dict[str, Any]] = []

    # 批量从本地 prices_daily 回补缺失的 price/entry_price（高优先级）
    try:
        missing_symbols = [
            (raw.get("symbol") or "").strip()
            for raw in (holdings or [])
            if (not raw.get("price") or raw.get("price") in (None, ""))
        ]
        missing_symbols = [s for s in missing_symbols if s]
        if missing_symbols:
            try:
                # 增加获取的数据量，从5天增加到30天，以确保有足够的历史数据
                df_latest = _db_manager.get_last_n_bars(symbols=missing_symbols, n=30)
                # 如果第一次获取失败，尝试获取更多数据
                if df_latest is None or df_latest.empty:
                    df_latest = _db_manager.get_last_n_bars(symbols=missing_symbols, n=90)
            except Exception:
                df_latest = None
            
            if df_latest is not None and not df_latest.empty:
                grouped = df_latest.groupby("symbol")
                price_map: Dict[str, Tuple[float, Optional[float]]] = {}
                for sym, g in grouped:
                    g = g.copy()
                    g["date"] = pd.to_datetime(g["date"])
                    g = g.sort_values("date")
                    last = g.iloc[-1]
                    prev = g.iloc[-2] if len(g) >= 2 else None
                    last_price = float(last.get("close") or last.get("Close") or 0.0)
                    prev_price = None
                    if prev is not None:
                        prev_price = float(prev.get("close") or prev.get("Close") or 0.0)
                    price_map[sym] = (last_price, prev_price)
                
                for raw in holdings or []:
                    sym = (raw.get("symbol") or "").strip()
                    if not sym or sym not in price_map:
                        continue
                    p, prevp = price_map[sym]
                    if (raw.get("price") in (None, "") or float(raw.get("price") or 0.0) <= 0.0) and p > 0:
                        raw["price"] = p
                    if (raw.get("entry_price") in (None, "") or float(raw.get("entry_price") or 0.0) <= 0.0):
                        # 优先使用prevp，如果没有则回退到p
                        if prevp and prevp > 0:
                            raw["entry_price"] = prevp
                        elif p > 0:
                            raw["entry_price"] = p
    except Exception as e:
        logger.warning(f"价格回补失败: {str(e)}")
        # 回填为辅助措施，失败则忽略
        pass

    for raw in holdings or []:
        symbol = (raw.get("symbol") or "").strip()
        code = (raw.get("code") or _strip_suffix(symbol))
        name = raw.get("name") or _resolve_stock_name(symbol, code)
        weight = float(raw.get("weight") or 0.0)
        shares = float(raw.get("shares") or 0.0)

        price_raw = raw.get("price")
        price = float(price_raw) if price_raw not in (None, "") else 0.0
        entry_raw = raw.get("entry_price")
        entry_price = float(entry_raw) if entry_raw not in (None, "") else 0.0
        # 如果 entry_price 未提供但有 price，则把 entry_price 设为 price
        if entry_price <= 0 and price > 0:
            entry_price = price
        # 如果 price 缺失但 entry_price 可用，使用 entry_price 作为价格回退
        if price <= 0 and entry_price > 0:
            price = entry_price
        
        # 使用统一的成本价格计算逻辑（仅在未提供成本价时回退计算）
        if (entry_price is None or entry_price <= 0) and initial_capital > 0 and weight > 0 and shares > 0:
            calculated_cost_price = _calculate_cost_price(
                weight,
                shares,
                initial_capital,
                commission_rate,
                None,
            )
            if calculated_cost_price > 0:
                entry_price = calculated_cost_price

        value = shares * price if price > 0 else 0.0
        # 跳过零股/零价值持仓，避免前端出现0.00%展示干扰
        if shares <= 0 or price <= 0 or value <= 0:
            continue

        return_raw = raw.get("return_pct")
        if return_raw not in (None, ""):
            try:
                return_pct = float(return_raw)  # type: ignore[arg-type]
            except Exception:
                return_pct = 0.0
        else:
            return_pct = (
                (price - entry_price) / entry_price if entry_price and entry_price > 0 and price > 0 else 0.0
            )

        formatted_entry: Dict[str, Any] = {
            "symbol": symbol,
            "code": code,
            "name": name,
            "weight": round(weight, 6),
            "weight_pct": round(weight * 100, 2),
            "shares": round(shares, 4),
            "price": round(price, 4) if price > 0 else None,
            "entry_price": round(entry_price, 4) if entry_price > 0 else None,
            "return_pct": round(return_pct, 6) if price > 0 and entry_price > 0 else 0.0,
            "value": round(value, 2) if value > 0 else 0.0,
        }
        extra_keys = {"symbol", "code", "name", "weight", "shares", "price", "entry_price", "return_pct", "value"}
        extras = {k: v for k, v in raw.items() if k not in extra_keys}
        if extras:
            formatted_entry.update(extras)
        formatted.append(formatted_entry)

    # 如果传入的每条持仓都没有 weight，但都包含 value，则用 value 重新计算权重
    try:
        local_total = sum([entry.get("value", 0.0) for entry in formatted])
        if local_total > 0:
            need_fix = any((entry.get("weight", 0.0) == 0.0 and entry.get("value", 0.0) > 0.0) for entry in formatted)
            if need_fix:
                for entry in formatted:
                    v = entry.get("value", 0.0)
                    entry["weight"] = round((v / local_total) if local_total > 0 else 0.0, 6)
                    entry["weight_pct"] = round(entry["weight"] * 100, 2)  # 同步更新权重百分比
    except Exception:
        # 保持原样，避免格式化失败影响主流程
        pass

    return formatted


def _build_rebalance_event_record(
    *,
    event_time: datetime,
    event_type: str,
    description: str,
    holdings: List[Dict[str, Any]],
    total_value: float,
    initial_capital: float,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """生成包含净值信息的调仓事件记录。"""
    nav_value: Optional[float] = None
    if initial_capital and initial_capital > 0:
        nav_value = total_value / initial_capital
    # 格式化持仓并重新计算合计市值以保证持仓明细与事件总值一致
    try:
        formatted = _format_rebalance_holdings(holdings, initial_capital=initial_capital, adjust_mode="qfq") if holdings is not None else []
        computed_total = sum([float(h.get("value") or 0.0) for h in formatted])
    except Exception:
        formatted = _format_rebalance_holdings(holdings, initial_capital=initial_capital, adjust_mode="qfq")
        computed_total = float(total_value or 0.0)

    # 如果传入的 total_value 与由持仓计算的合计相差较大，则以计算值为准并记录警告
    try:
        # 对于回溯测试，使用更严格的容差以确保数据准确性
        base_tol = max(1.0, 0.002 * computed_total)  # 最小 1 元，或 0.2% 的相对误差（比之前的0.5%更严格）
        tol = base_tol
        
        # 对于创建组合事件，保持初始净值为1.0，不进行total_value替换
        if event_type == 'create':
            # 即使computed_total与total_value有差异，也保持total_value为initial_capital
            if total_value != initial_capital:
                logger.info(
                    "组合创建事件: 强制设置total_value=initial_capital (%.2f)",
                    initial_capital
                )
            total_value = initial_capital
            nav_value = 1.0  # 强制设置初始净值为1.0
            logger.debug(
                "组合创建事件: 初始净值设置为1.0, event_time=%s",
                getattr(event_time, 'isoformat', lambda: str(event_time))()
            )
        elif total_value is None or abs(float(total_value) - computed_total) > tol:
            # 计算差异百分比
            diff_pct = abs(float(total_value or 0.0) - computed_total) / max(1.0, computed_total) * 100
            
            # 根据差异大小调整日志级别
            log_msg = "Rebalance total_value mismatch: provided=%.2f computed_from_holdings=%.2f diff=%.2f (%.2f%%) event_time=%s; using computed value"
            log_args = [
                float(total_value or 0.0),
                computed_total,
                abs(float(total_value or 0.0) - computed_total),
                diff_pct,
                getattr(event_time, 'isoformat', lambda: str(event_time))(),
            ]
            
            if diff_pct > 5:  # 差异大于5%，使用error级别
                logger.error(log_msg, *log_args)
            elif diff_pct > 1:  # 差异大于1%，使用warning级别
                logger.warning(log_msg, *log_args)
            else:  # 差异小于1%，使用info级别
                logger.info(log_msg, *log_args)
            
            # 总是使用计算值以确保数据一致性
            total_value = computed_total
            # 当 total_value 被调整时，同步更新 nav_value
            if initial_capital and initial_capital > 0:
                nav_value = total_value / initial_capital
                logger.debug(
                    "调仓事件: 调整nav_value=%.6f (total_value=%.2f, initial_capital=%.2f)",
                    nav_value, total_value, initial_capital
                )
    except Exception:
        # 在任何错误情况下，使用原始 total_value
        total_value = float(total_value or 0.0)
        logger.error(
            "计算调仓事件净值时发生错误: event_time=%s, event_type=%s, error=%s",
            getattr(event_time, 'isoformat', lambda: str(event_time))(),
            event_type,
            str(sys.exc_info()[1])
        )

    record: Dict[str, Any] = {
        "timestamp": event_time.isoformat(),
        "type": event_type,
        "description": description,
        "holdings": formatted,
        "total_value": round(float(total_value or 0.0), 2),
    }
    if nav_value is not None:
        record["nav_value"] = round(nav_value, 6)
        logger.debug(
            "调仓事件记录: event_type=%s, total_value=%.2f, nav_value=%.6f",
            event_type, record["total_value"], record["nav_value"]
        )
    if trades:
        serialized_trades: List[Dict[str, Any]] = []
        for trade in trades:
            serialized = dict(trade)
            ts = serialized.get("trade_ts")
            if isinstance(ts, datetime):
                serialized["trade_ts"] = ts.isoformat()
            serialized_trades.append(serialized)
        record["trades"] = serialized_trades
    return record


# =========================================================================
# Database fetch & persist helpers
# =========================================================================

def _fetch_portfolio_rows() -> List[Dict[str, Any]]:
    query = "SELECT * FROM portfolios ORDER BY created_at DESC"
    return _db_manager.execute_query(query)


def _fetch_portfolio_bundle(pid: int) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    row_query = f"SELECT * FROM portfolios WHERE id = {_placeholder()}"
    rows = _db_manager.execute_query(row_query, (pid,))
    if not rows:
        return None
    holdings_query = (
        f"SELECT * FROM portfolio_holdings WHERE portfolio_id = {_placeholder()} ORDER BY weight DESC, id ASC"
    )
    nav_query = (
        f"SELECT nav_date, nav_value, total_value FROM portfolio_nav_history "
        f"WHERE portfolio_id = {_placeholder()} ORDER BY nav_date ASC"
    )
    rebalance_query = (
        f"SELECT event_time, event_type, description, details FROM portfolio_rebalances "
        f"WHERE portfolio_id = {_placeholder()} ORDER BY event_time ASC"
    )
    holdings_rows = _db_manager.execute_query(holdings_query, (pid,))
    nav_rows = _db_manager.execute_query(nav_query, (pid,))
    rebalance_rows = _db_manager.execute_query(rebalance_query, (pid,))
    return rows[0], holdings_rows, nav_rows, rebalance_rows


def _valuation_from_row(row: Dict[str, Any]) -> PortfolioValuation:
    return PortfolioValuation(
        nav_total=float(row.get("total_value") or 0.0),
        nav_value=float(row.get("nav_value") or 0.0),
        daily_return_pct=float(row.get("daily_return_pct") or 0.0),
        total_return_pct=float(row.get("total_return_pct") or 0.0),
        last_valued_at=_parse_datetime(row.get("last_valued_at")),
    )


def _hydrate_summary(row: Dict[str, Any]) -> PortfolioInfo:
    created_at = _parse_datetime(row.get("created_at")) or _now()
    strategy_tags = _json_loads(row.get("strategy_tags"), []) or []
    return PortfolioInfo(
        id=int(row["id"]),
        name=row.get("name") or "",
        created_at=created_at,
        top_n=int(row.get("top_n") or 0),
        initial_capital=float(row.get("initial_capital") or 0.0),
        holdings_count=int(row.get("holdings_count") or 0),
        benchmark=row.get("benchmark") or "",
        risk_level=row.get("risk_level") or "",
        strategy_tags=strategy_tags,
    )


def _merge_nav_history(
    nav_history: List[Dict[str, Any]] | None,
    rebalance_history: List[Dict[str, Any]] | None,
    *,
    initial_capital: float,
) -> List[Dict[str, Any]]:
    """融合数据库净值历史与调仓事件，确保关键节点与记录一致。"""

    merged: Dict[str, Dict[str, Any]] = {}

    initial_capital = initial_capital if initial_capital and initial_capital > 0 else 1.0

    original_values: List[float] = []
    for row in nav_history or []:
        date_str = row.get("date")
        if not date_str:
            continue
        nav_val = float(row.get("nav_value") or 0.0)
        total_val = float(row.get("total_value") or (nav_val * initial_capital))
        merged[date_str] = {
            "date": date_str,
            "nav_value": nav_val,
            "total_value": total_val,
        }
        original_values.append(nav_val)

    anchor_values: Dict[str, float] = {}
    for event in rebalance_history or []:
        nav_val = event.get("nav_value")
        total_val = event.get("total_value")
        if nav_val is None and total_val is None:
            continue
        timestamp = event.get("timestamp")
        date_str = event.get("date")
        if not date_str and timestamp:
            try:
                parsed = _parse_datetime(timestamp)
                if parsed:
                    date_str = parsed.date().isoformat()
            except Exception:  # pragma: no cover - 防御性
                date_str = None
        if not date_str:
            continue
        if nav_val is None and total_val is not None:
            nav_val = total_val / initial_capital if initial_capital else 0.0
        if nav_val is None:
            continue
        if total_val is None:
            total_val = nav_val * initial_capital
        merged[date_str] = {
            "date": date_str,
            "nav_value": float(nav_val),
            "total_value": float(total_val),
        }
        anchor_values[date_str] = float(nav_val)

    # 确保创建日存在基准点（净值 = 1）
    if rebalance_history:
        first_event = next((e for e in rebalance_history if e.get("nav_value") is not None), None)
        if first_event:
            base_date = first_event.get("date")
            if not base_date and first_event.get("timestamp"):
                parsed = _parse_datetime(first_event["timestamp"])
                base_date = parsed.date().isoformat() if parsed else None
            if base_date and base_date not in merged:
                nav_val = float(first_event.get("nav_value") or 1.0)
                merged[base_date] = {
                    "date": base_date,
                    "nav_value": nav_val,
                    "total_value": float(first_event.get("total_value") or nav_val * initial_capital),
                }
                anchor_values[base_date] = nav_val

    # 若原始列表为空，直接返回锚点数据
    if not merged:
        return []

    sorted_dates = sorted(merged.keys())
    entries = [merged[date] for date in sorted_dates]

    # 构造与 entries 对应的原始值（若缺失则使用当前值）
    original_series: Dict[str, float] = {}
    if nav_history:
        for row in nav_history:
            date_str = row.get("date")
            if date_str:
                original_series[date_str] = float(row.get("nav_value") or 0.0)
    for entry in entries:
        if entry["date"] not in original_series:
            original_series[entry["date"]] = float(entry.get("nav_value") or 0.0)

    # 将锚点值写回 entries
    anchor_indices: List[int] = []
    for idx, entry in enumerate(entries):
        date_str = entry["date"]
        if date_str in anchor_values:
            entry["nav_value"] = anchor_values[date_str]
            entry["total_value"] = float(entry.get("total_value") or anchor_values[date_str] * initial_capital)
            anchor_indices.append(idx)

    def _adjust_segment(start_idx: int, end_idx: int) -> None:
        if start_idx >= end_idx:
            return
        start_date = entries[start_idx]["date"]
        end_date = entries[end_idx]["date"]
        start_orig = original_series.get(start_date, entries[start_idx]["nav_value"])
        end_orig = original_series.get(end_date, entries[end_idx]["nav_value"])
        start_target = entries[start_idx]["nav_value"]
        end_target = entries[end_idx]["nav_value"]
        orig_range = end_orig - start_orig
        span = end_idx - start_idx
        for offset, idx in enumerate(range(start_idx, end_idx + 1)):
            if idx in (start_idx, end_idx):
                entries[idx]["total_value"] = entries[idx]["nav_value"] * initial_capital
                continue
            if orig_range == 0:
                ratio = offset / span if span else 0.0
            else:
                current_orig = original_series.get(entries[idx]["date"], entries[idx]["nav_value"])
                ratio = (current_orig - start_orig) / orig_range
            ratio = max(0.0, min(1.0, ratio))
            new_val = start_target + ratio * (end_target - start_target)
            entries[idx]["nav_value"] = new_val
            entries[idx]["total_value"] = new_val * initial_capital

    if anchor_indices:
        for idx in range(len(anchor_indices) - 1):
            _adjust_segment(anchor_indices[idx], anchor_indices[idx + 1])
        # 对最后一个锚点之后的区间，维持相对变化
        last_idx = anchor_indices[-1]
        last_target = entries[last_idx]["nav_value"]
        last_orig = original_series.get(entries[last_idx]["date"], last_target)
        for idx in range(last_idx + 1, len(entries)):
            current_orig = original_series.get(entries[idx]["date"], entries[idx]["nav_value"])
            if last_orig == 0:
                entries[idx]["nav_value"] = last_target
            else:
                ratio = current_orig / last_orig
                entries[idx]["nav_value"] = last_target * ratio
            entries[idx]["total_value"] = entries[idx]["nav_value"] * initial_capital
    else:
        # 没有锚点时，保证总资产字段与净值一致
        for entry in entries:
            entry["total_value"] = entry["nav_value"] * initial_capital

    return entries


def _hydrate_detail(
    portfolio_row: Dict[str, Any],
    holdings_rows: List[Dict[str, Any]],
    nav_rows: List[Dict[str, Any]],
    rebalance_rows: List[Dict[str, Any]],
    as_of: datetime,
) -> Tuple[PortfolioDetail, PortfolioValuation, Dict[str, PriceQuote], List[Dict[str, Any]]]:
    info = _hydrate_summary(portfolio_row)
    valuation = _valuation_from_row(portfolio_row)

    holdings: List[PortfolioHoldingSnapshot] = []
    quotes: Dict[str, PriceQuote] = {}
    for row in holdings_rows:
        symbol = row.get("symbol") or ""
        code = row.get("code") or _strip_suffix(symbol)
        opened_at = _parse_datetime(row.get("opened_at")) or info.created_at
        holding = PortfolioHoldingSnapshot(
            symbol=symbol,
            code=code,
            name=row.get("name") or _resolve_stock_name(symbol, code),
            weight=float(row.get("weight") or 0.0),
            shares=float(row.get("shares") or 0.0),
            cost_price=float(row.get("cost_price") or 0.0),
            opened_at=opened_at,
        )
        holdings.append(holding)
        quotes[symbol] = PriceQuote(
            latest_price=float(row.get("latest_price") or 0.0),
            previous_price=float(row["previous_price"]) if row.get("previous_price") is not None else None,
            last_trade_at=_parse_datetime(row.get("last_trade_at")),
        )

    # 过滤掉组合创建日期之前的净值数据（修复问题：组合建立前不应有净值数据）
    created_date = info.created_at.date() if info.created_at else None
    nav_history = []
    for row in nav_rows:
        nav_date_str = (row.get("nav_date") or "") if isinstance(row.get("nav_date"), str) else (_parse_datetime(row.get("nav_date")).strftime("%Y-%m-%d") if _parse_datetime(row.get("nav_date")) else "")
        # 如果有创建日期，过滤掉创建日期之前的数据
        if created_date and nav_date_str:
            nav_dt = _parse_datetime(row.get("nav_date"))
            if nav_dt and nav_dt.date() < created_date:
                continue
        nav_history.append({
            "date": nav_date_str,
            "nav_value": float(row.get("nav_value") or 0.0),
            "total_value": float(row.get("total_value") or 0.0),
        })

    rebalance_history = []
    for row in rebalance_rows:
        details = _json_loads(row.get("details"), None)
        holdings_payload = details.get("holdings") if isinstance(details, dict) else details
        nav_value = details.get("nav_value") if isinstance(details, dict) else None
        total_value = details.get("total_value") if isinstance(details, dict) else None
        formatted_holdings = _format_rebalance_holdings(holdings_payload, initial_capital=info.initial_capital, adjust_mode="qfq") if holdings_payload else []
        event_ts = _parse_datetime(row.get("event_time")) or info.created_at
        rebalance_history.append(
            {
                "timestamp": _safe_isoformat(event_ts),
                "date": event_ts.date().isoformat() if hasattr(event_ts, "date") else None,
                "type": row.get("event_type"),
                "description": row.get("description"),
                "holdings": formatted_holdings,
                "nav_value": nav_value,
                "total_value": total_value,
            }
        )

    detail = PortfolioDetail(
        id=info.id,
        name=info.name,
        created_at=info.created_at,
        top_n=info.top_n,
        initial_capital=info.initial_capital,
        holdings_count=int(portfolio_row.get("holdings_count") or len(holdings)),
        benchmark=info.benchmark,
        risk_level=info.risk_level,
        strategy_tags=info.strategy_tags,
        holdings=holdings,
        notes=portfolio_row.get("notes"),
        rebalance_history=rebalance_history,
    )

    # 直接返回数据库中的净值历史（已经包含每日数据）
    # 不再使用 _merge_nav_history 进行融合，因为现在数据库中已经存储了完整的每日净值
    # merged_nav = _merge_nav_history(nav_history, rebalance_history, initial_capital=info.initial_capital)

    return detail, valuation, quotes, nav_history


def _get_auto_selector() -> IntelligentStockSelector:
    global _AUTO_SELECTOR
    if _AUTO_SELECTOR is None:
        with _SELECTOR_LOCK:
            if _AUTO_SELECTOR is None:
                _AUTO_SELECTOR = IntelligentStockSelector(db_manager=_db_manager)
    return _AUTO_SELECTOR


def _ensure_selector_ready(selector: IntelligentStockSelector) -> bool:
    global _SELECTOR_READY
    if _SELECTOR_READY:
        return True
    loaded = (
        selector.load_models(period="30d")
        or selector.load_models(period="10d")
        or selector.load_model()
    )
    _SELECTOR_READY = bool(loaded)
    return _SELECTOR_READY


def _auto_pick_cache_key(top_n: int, symbol_limit: int) -> str:
    return f"{top_n}:{symbol_limit}"


def _get_latest_price(symbol: str) -> float:
    try:
        bars = _db_manager.get_last_n_bars([symbol], n=1)
        if bars is not None and not bars.empty:
            row = bars.iloc[-1]
            price = row.get("close") or row.get("Close")
            if price is not None:
                return float(price)
    except Exception as exc:
        logger.debug("获取最新价格失败 %s: %s", symbol, exc)
    return 0.0


def _symbols_with_recent_data(
    limit: int,
    *,
    lookback_days: int,
    min_rows: int,
    as_of: Optional[datetime] = None,
) -> List[str]:
    """在本地数据库中筛选近期有行情的股票。"""
    if limit <= 0:
        return []
    as_of = as_of or _now()
    start_date = (as_of.date() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    placeholder = _placeholder()
    query = (
        "SELECT symbol, MAX(date) AS latest_date, COUNT(*) AS rows_count "
        "FROM prices_daily "
        f"WHERE date >= {placeholder} "
        "GROUP BY symbol "
        f"HAVING rows_count >= {placeholder} "
        "ORDER BY latest_date DESC "
        f"LIMIT {placeholder}"
    )
    try:
        rows = _db_manager.execute_query(query, (start_date, int(min_rows), int(limit)))
    except Exception:
        logger.debug("查询近端行情股票失败", exc_info=True)
        return []
    symbols: List[str] = []
    for row in rows or []:
        symbol = row.get("symbol") if isinstance(row, dict) else row[0]
        if symbol and str(symbol).endswith((".SH", ".SZ")):
            symbols.append(symbol)
    return symbols


def _fetch_quick_picks(
    top_n: int,
    symbol_limit: int,
    *,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    if top_n <= 0:
        return []
    symbol_limit = max(symbol_limit, top_n * 3)
    cache_key = _auto_pick_cache_key(top_n, symbol_limit)
    now = _now()
    if not force_refresh and _AUTO_PICK_CACHE_TTL > 0:
        cached = _AUTO_PICK_CACHE.get(cache_key)
        if cached and cached.get("expires_at") and cached["expires_at"] > now:
            picks_cached = cached.get("picks") or []
            return [dict(item) for item in picks_cached[:top_n]]

    selector = _get_auto_selector()
    picks: List[Dict[str, Any]] = []

    with _SELECTOR_LOCK:
        ready = _ensure_selector_ready(selector)
        try:
            candidates_raw = _db_manager.list_symbols(
                markets=["SH", "SZ"],
                limit=max(symbol_limit, top_n * 6),
            )
        except Exception as exc:
            logger.warning("获取候选股票失败: %s", exc)
            candidates_raw = []
        candidate_symbols = [
            row.get("symbol")
            for row in candidates_raw
            if row.get("symbol") and str(row.get("symbol")).endswith((".SH", ".SZ"))
        ]
        recent_symbols: List[str] = []
        if candidate_symbols:
            recent_symbols = _symbols_with_recent_data(
                max(symbol_limit, top_n * 6),
                lookback_days=_FALLBACK_LOOKBACK_DAYS,
                min_rows=max(5, _FALLBACK_MIN_ROWS),
            )
            if recent_symbols:
                recent_set = set(recent_symbols)
                candidate_symbols = [sym for sym in candidate_symbols if sym in recent_set]
            candidate_symbols = candidate_symbols[: max(symbol_limit, top_n * 6)]
        if ready and candidate_symbols:
            try:
                picks = selector.predict_top_n(candidate_symbols, max(top_n * 3, top_n))
            except Exception as exc:
                logger.warning("智能选股预测失败，将使用备用逻辑: %s", exc)
                picks = []
        if not picks:
            fallback = selector._fallback_stock_picks(max(top_n, 30))
            if isinstance(fallback, dict):
                picks = fallback.get("data", {}).get("picks", []) or []

    if picks and _AUTO_PICK_CACHE_TTL > 0:
        _AUTO_PICK_CACHE[cache_key] = {
            "picks": [dict(item) for item in picks[:top_n]],
            "expires_at": now + timedelta(seconds=_AUTO_PICK_CACHE_TTL),
        }
    return [dict(item) for item in picks[:top_n]]


def _build_holdings_from_picks(
    picks: List[Dict[str, Any]],
    *,
    initial_capital: float,
    pipeline: PortfolioPipeline,
) -> Tuple[List[Holding], List[Dict[str, Any]]]:
    if not picks:
        return [], []
    filtered: List[Tuple[str, float, Dict[str, Any]]] = []
    for pick in picks:
        symbol = pick.get("symbol")
        if not symbol:
            continue
        price = pick.get("last_close") or pick.get("close") or pick.get("price")
        try:
            price = float(price or 0.0)
        except Exception:
            price = 0.0
        if price <= 0:
            price = _get_latest_price(symbol)
        if price <= 0:
            continue
        filtered.append((symbol, price, pick))
    if not filtered:
        return [], []

    max_count = min(len(filtered), len(picks))
    target_weight = 1.0 / max_count
    commission = getattr(pipeline, "commission_rate", 0.0)
    holdings: List[Holding] = []
    holdings_meta: List[Dict[str, Any]] = []

    for symbol, price, pick in filtered[:max_count]:
        alloc = initial_capital * target_weight
        shares = (alloc * (1 - commission)) / price if price > 0 else 0.0
        holdings.append(Holding(symbol=symbol, weight=target_weight, shares=shares))
        meta = {
            "symbol": symbol,
            "weight": target_weight,
            "shares": shares,
            "price": price,
            "probability": pick.get("prob_up_30d") or pick.get("probability"),
            "expected_return": pick.get("expected_return_30d"),
            "score": pick.get("score"),
            "sentiment": pick.get("sentiment"),
        }
        holdings_meta.append(meta)

    return holdings, holdings_meta


def _build_holdings_from_recent_prices(
    symbols: List[str],
    *,
    initial_capital: float,
    pipeline: PortfolioPipeline,
) -> Tuple[List[Holding], List[Dict[str, Any]]]:
    """基于本地最新价格构建等权持仓，作为模型兜底方案。"""
    if not symbols:
        return [], []
    price_df = _db_manager.get_last_n_bars(symbols=symbols, n=1)
    if price_df is None or price_df.empty:
        return [], []
    valid_quotes: List[Tuple[str, float]] = []
    for _, row in price_df.iterrows():
        symbol = row.get("symbol")
        if not symbol:
            continue
        price = row.get("close") or row.get("Close")
        try:
            price = float(price)
        except Exception:
            price = 0.0
        if price and price > 0:
            valid_quotes.append((symbol, price))
    if not valid_quotes:
        return [], []
    max_count = min(len(valid_quotes), len(symbols))
    target_weight = 1.0 / max_count if max_count else 0.0
    commission = getattr(pipeline, "commission_rate", 0.0)
    holdings: List[Holding] = []
    metas: List[Dict[str, Any]] = []
    for symbol, price in valid_quotes[:max_count]:
        alloc = initial_capital * target_weight
        shares = (alloc * (1 - commission)) / price if price > 0 else 0.0
        holdings.append(Holding(symbol=symbol, weight=target_weight, shares=shares))
        metas.append(
            {
                "symbol": symbol,
                "weight": target_weight,
                "shares": shares,
                "price": price,
                "probability": None,
                "expected_return": None,
                "score": None,
                "sentiment": None,
            }
        )
    return holdings, metas


def _parse_optional_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _default_auto_trading() -> bool:
    flag = _parse_optional_bool(os.getenv("PORTFOLIO_AUTO_TRADING_DEFAULT"), True)
    return True if flag is None else flag


def _simulate_backtrack_portfolio(
    *, 
    pipeline: PortfolioPipeline, 
    start_ts: datetime, 
    end_ts: datetime, 
    top_n: int, 
    initial_capital: float, 
    auto_trading: bool, 
    weight_overrides: Optional[Dict[str, float]] = None, 
    candidate_limit: Optional[int] = None,
) -> Tuple[List[PortfolioHoldingSnapshot], Dict[str, PriceQuote], List[Dict[str, Any]], List[Dict[str, Any]], PortfolioValuation, datetime]:
    resolved_limit = resolve_candidate_limit(candidate_limit if candidate_limit is not None else pipeline.candidate_limit)
    logger.info(f"[回溯入口] auto_trading={auto_trading}, start_ts={start_ts}, end_ts={end_ts}, top_n={top_n}, initial_capital={initial_capital}, candidate_limit={candidate_limit}, resolved_limit={resolved_limit}")
    pipeline.reset_weights()
    if weight_overrides:
        pipeline.apply_weight_overrides(weight_overrides)
    try:
        if auto_trading:
            logger.info("[回溯分支] 进入_simulate_backtrack_active")
            return _simulate_backtrack_active(
                pipeline=pipeline, 
                start_ts=start_ts, 
                end_ts=end_ts, 
                top_n=top_n, 
                initial_capital=initial_capital, 
                candidate_limit=resolved_limit,
            )
        logger.info("[回溯分支] 进入_simulate_backtrack_passive")
        return _simulate_backtrack_passive(
            pipeline=pipeline, 
            start_ts=start_ts, 
            end_ts=end_ts, 
            top_n=top_n, 
            initial_capital=initial_capital, 
            candidate_limit=resolved_limit,
        )
    except Exception as e:
        import traceback
        logger.error(f"组合回溯模拟失败: {str(e)}\n{traceback.format_exc()}")
        # 发生异常时返回空的结果集，确保调用方能够正常解包
        empty_snapshots: List[PortfolioHoldingSnapshot] = []
        empty_quotes: Dict[str, PriceQuote] = {}
        empty_events: List[Dict[str, Any]] = []
        empty_history: List[Dict[str, Any]] = []
        empty_valuation = PortfolioValuation(
            nav_total=initial_capital, 
            nav_value=1.0, 
            daily_return_pct=0.0, 
            total_return_pct=0.0, 
            last_valued_at=start_ts
        )
        return empty_snapshots, empty_quotes, empty_events, empty_history, empty_valuation, start_ts
    finally:
        pipeline.reset_weights()


def _simulate_backtrack_active(
    *,
    pipeline: PortfolioPipeline,
    start_ts: datetime,
    end_ts: datetime,
    top_n: int,
    initial_capital: float,
    candidate_limit: Optional[int],
) -> Tuple[List[PortfolioHoldingSnapshot], Dict[str, PriceQuote], List[Dict[str, Any]], List[Dict[str, Any]], PortfolioValuation, datetime]:
    """基于 PortfolioPipeline + AdaptiveTradingSystem 运行回溯建仓与调仓仿真。"""

    logger.info(f"[ACTIVE入口] _simulate_backtrack_active: start_ts={start_ts}, end_ts={end_ts}, top_n={top_n}, initial_capital={initial_capital}, candidate_limit={candidate_limit}")
    from src.trading.systems.adaptive_trading_system import AdaptiveTradingSystem

    ats = AdaptiveTradingSystem(initial_capital=initial_capital)
    price_cache: Dict[str, pd.DataFrame] = {}
    last_price_map: Dict[str, float] = {}
    last_price_time: Dict[str, datetime] = {}
    nav_history: List[Dict[str, Any]] = []
    rebalance_events: List[Dict[str, Any]] = []
    # 新增：显式维护每只股票的累计买入股数与总成本
    cost_calc_map: Dict[str, Dict[str, float]] = {}  # {symbol: {"shares": float, "cost": float}}
    trade_ledger: List[Dict[str, Any]] = []
    commission_rate = float(getattr(pipeline, "commission_rate", 0.0) or 0.0)
    trade_sequence = 0

    start_date = pd.Timestamp(start_ts.date())
    end_date = pd.Timestamp(end_ts.date())
    data_access = getattr(pipeline, "data_access", None)
    if data_access and not is_trading_day(data_access, start_date.date()):
        try:
            next_day = next_trading_day(data_access, start_date.date())
            logger.info("回溯内部起始日 %s 非交易日，顺延至 %s", start_date.date(), next_day)
            start_date = pd.Timestamp(next_day)
            start_ts = datetime.combine(next_day, datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        except Exception:
            logger.warning("回溯内部起始日 %s 顺延失败，继续使用原日期", start_date, exc_info=True)

    if data_access:
        trading_list: List[pd.Timestamp] = []
        cursor = start_date.date()
        limit = 0
        while pd.Timestamp(cursor) <= end_date and limit < 6000:
            if is_trading_day(data_access, cursor):
                trading_list.append(pd.Timestamp(cursor))
            cursor += timedelta(days=1)
            limit += 1
        if not trading_list:
            business_days = pd.date_range(start=start_date, end=end_date, freq="B")
        else:
            business_days = pd.DatetimeIndex(trading_list)
    else:
        business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    # 至少包含起始日
    if len(business_days) == 0 or business_days[0].normalize() != start_date.normalize():
        business_days = business_days.insert(0, start_date)

    rebal_dates = _generate_rebalance_dates(start_date, end_date, _get_rebalance_interval_days())
    if data_access:
        adjusted: List[pd.Timestamp] = []
        seen: Set[pd.Timestamp] = set()
        for dt in rebal_dates:
            day = dt.date()
            if not is_trading_day(data_access, day):
                try:
                    day = next_trading_day(data_access, day)
                except Exception:
                    continue
            ts_day = pd.Timestamp(day)
            if ts_day < start_date or ts_day > end_date:
                continue
            if ts_day in seen:
                continue
            adjusted.append(ts_day)
            seen.add(ts_day)
        if adjusted:
            rebal_dates = pd.DatetimeIndex(adjusted)
    rebal_set = {dt.normalize() for dt in rebal_dates}
    candidate_symbols_all = pipeline._get_stock_pool(limit=candidate_limit)
    picks_initial = pipeline.pick_stocks(as_of_date=start_date, candidates=candidate_symbols_all, top_n=top_n * 2)

    def _load_price_series(symbol: str) -> pd.DataFrame:
        cached = price_cache.get(symbol)
        if cached is not None:
            return cached
        total_days = max(5, int((end_date - start_date).days) + pipeline.lookback_days + 5)
        df = pipeline._fetch_history(symbol, end_date=end_date, days=total_days)
        if df is None or df.empty:
            empty_df = pd.DataFrame()
            price_cache[symbol] = empty_df
            return empty_df
        price_df = df.copy()
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        numeric_cols = [col for col in ["open", "close", "high", "low", "volume", "Open", "Close"] if col in price_df.columns]
        for col in numeric_cols:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        for lower, upper in [("open", "Open"), ("close", "Close")]:
            if lower not in price_df.columns and upper in price_df.columns:
                price_df[lower] = price_df[upper]
            if upper not in price_df.columns and lower in price_df.columns:
                price_df[upper] = price_df[lower]
        price_cache[symbol] = price_df
        return price_df

    def _price_on(symbol: str, ts: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
        df_price = _load_price_series(symbol)
        if df_price.empty or "close" not in df_price.columns:
            return None, None, None
        ser = df_price["close"].dropna()
        if ser.empty:
            return None, None, None
        try:
            price = ser.asof(ts)
        except Exception:
            price = None
        if price is None or pd.isna(price):
            return None, None, None
        hist = ser[ser.index <= ts]
        prev = hist.iloc[-2] if len(hist) >= 2 else None
        price_time = hist.index[-1] if len(hist) >= 1 else None
        return float(price), price_time, (float(prev) if prev is not None else None)

    def _portfolio_value() -> float:
        # 现金+持仓市值，CASH伪持仓不计入
        total = float(ats.current_capital)
        for sym, pos in ats.positions.items():
            if sym == "CASH":
                continue
            price = last_price_map.get(sym)
            if price is None:
                last_time = last_price_time.get(sym)
                ts = pd.Timestamp(last_time) if last_time is not None else pd.Timestamp(end_date)
                price, _, _ = _price_on(sym, ts)
            if price is None:
                continue
            total += float(pos.get("volume", 0)) * float(price)
        return float(total)

    def _trade_dt_for(day: pd.Timestamp) -> datetime:
        ts_day = pd.Timestamp(day)
        trade_time = datetime.combine(ts_day.date(), time(hour=9, minute=35), tzinfo=start_ts.tzinfo)
        return trade_time

    def _update_cost(symbol: str, side: str, qty: int, price: float, fee: float) -> None:
        book = cost_calc_map.setdefault(symbol, {"shares": 0.0, "cost": 0.0})
        if side.upper() == "BUY":
            book["shares"] += float(qty)
            book["cost"] += float(qty) * float(price) + float(fee or 0.0)
        else:
            shares_before = book.get("shares", 0.0)
            cost_before = book.get("cost", 0.0)
            if shares_before <= 0:
                book["shares"] = 0.0
                book["cost"] = 0.0
                return
            avg_cost = cost_before / shares_before if shares_before > 0 else 0.0
            book["shares"] = max(0.0, shares_before - float(qty))
            book["cost"] = max(0.0, cost_before - avg_cost * float(qty))
            if book["shares"] == 0:
                book["cost"] = 0.0

    def _record_trade(
        symbol: str,
        side: str,
        qty: int,
        price: float,
        trade_day: pd.Timestamp,
        *,
        reason: str = "",
        event_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        nonlocal trade_sequence
        if qty <= 0 or price is None or price <= 0:
            return
        gross = float(qty) * float(price)
        fee = gross * commission_rate if commission_rate > 0 else 0.0
        fee = round(float(fee), 4)
        trade_ts = _trade_dt_for(trade_day)
        side_up = side.upper()
        if fee > 0:
            # execute_trade/adjust_position/close_position 已处理净成交额，额外扣除费用
            ats.current_capital -= fee
        net_cash = -(gross + fee) if side_up == "BUY" else gross - fee
        record = {
            "id": trade_sequence,
            "symbol": symbol,
            "side": side_up,
            "qty": int(qty),
            "price": float(price),
            "gross": round(gross, 4),
            "fee": fee,
            "net_cash": round(net_cash, 4),
            "trade_ts": trade_ts,
            "note": reason or "",
        }
        trade_sequence += 1
        trade_ledger.append(record)
        if event_trades is not None:
            event_trades.append(record)
        _update_cost(symbol, side_up, qty, price, fee)

    def _record_event(
        event_time: datetime,
        event_type: str,
        description: str,
        holdings_snapshot: List[Dict[str, Any]],
        total_value: float,
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # 记录调仓事件的核心信息到日志，便于排查权重/净值异常
        try:
            holdings_brief = [f"{h.get('symbol')}:{round(float(h.get('shares',0))*float(h.get('price') or 0),2)}" for h in holdings_snapshot]
        except Exception:
            holdings_brief = []
        logger.debug("Rebalance event: time=%s type=%s total_value=%.2f holdings=%s", event_time, event_type, float(total_value or 0.0), holdings_brief)

        record = _build_rebalance_event_record(
            event_time=event_time,
            event_type=event_type,
            description=description,
            holdings=holdings_snapshot,
            total_value=total_value,
            initial_capital=initial_capital,
            trades=trades,
        )
        rebalance_events.append(record)

    def _snapshot_holdings(current_prices: Dict[str, float], total_value: float) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for sym, pos in ats.positions.items():
            if sym == "CASH":
                continue
            volume = float(pos.get("volume", 0))
            price = current_prices.get(sym) or last_price_map.get(sym) or 0.0
            weight = (volume * price / total_value) if total_value > 0 else 0.0
            snapshot.append(
                {
                    "symbol": sym,
                    "weight": weight,
                    "shares": volume,
                    "price": price,
                    "entry_price": float(pos.get("entry_price", 0.0)),
                }
            )
        # 加入现金为伪持仓，仅用于快照展示
        cash_amount = float(ats.current_capital)
        if cash_amount > 0:
            cash_weight = (cash_amount / total_value) if total_value > 0 else 0.0
            snapshot.append(
                {
                    "symbol": "CASH",
                    "weight": cash_weight,
                    "shares": cash_amount,  # 以1元面值表示的"份额"
                    "price": 1.0,
                    "entry_price": 1.0,
                }
            )
        return snapshot

    # 首日初始化建仓
    initial_ts = start_date
    # 动态补充候选池，确保建仓股票数充足且价格有效
    logger.info(f"回溯建仓: 初始候选数={len(candidate_symbols_all)}, pick_stocks返回={len(picks_initial)}")
    valid_picks: List[PickResult] = []
    for pick in picks_initial:
        price, price_time, _ = _price_on(pick.symbol, initial_ts)
        if price is None or price <= 0:
            logger.debug(f"初选无效价格: {pick.symbol} @ {initial_ts}, price={price}")
            continue
        last_price_map[pick.symbol] = price
        if price_time is not None:
            last_price_time[pick.symbol] = price_time.to_pydatetime()
        valid_picks.append(pick)
    logger.info(f"初选有效股票数={len(valid_picks)}，目标top_n={top_n}")
    # 若有效股票不足top_n，自动补充有有效价格的股票
    if len(valid_picks) < top_n:
        all_symbols = pipeline._get_stock_pool(limit=top_n*10)
        logger.info(f"补充阶段: 全市场股票数={len(all_symbols)}")
        for sym in all_symbols:
            if any(p.symbol == sym for p in valid_picks):
                continue
            price, price_time, _ = _price_on(sym, initial_ts)
            if price is None or price <= 0:
                logger.debug(f"补充无效价格: {sym} @ {initial_ts}, price={price}")
                continue
            valid_picks.append(PickResult(symbol=sym, score=0.0, reason="补充", signal=None, risk_score=0.5))
            last_price_map[sym] = price
            if price_time is not None:
                last_price_time[sym] = price_time.to_pydatetime()
            if len(valid_picks) >= top_n:
                break
        logger.info(f"补充后有效股票数={len(valid_picks)}")
    if not valid_picks or len(valid_picks) < top_n:
        logger.error(f"回溯建仓失败：指定日期有效股票数不足，实际{len(valid_picks)}，期望{top_n}")
        logger.error(f"有效股票列表: {[p.symbol for p in valid_picks]}")
        raise RuntimeError(f"回溯建仓失败：指定日期有效股票数不足，实际{len(valid_picks)}，期望{top_n}")
    # 只取前top_n
    valid_picks = valid_picks[:top_n]

    total_value = initial_capital
    per_value = total_value / len(valid_picks)
    base_history = _load_price_series(valid_picks[0].symbol)
    market_frame = base_history[[c for c in ["close", "volume"] if c in base_history.columns]].copy() if not base_history.empty else pd.DataFrame(columns=["close"])
    if "close" not in market_frame.columns:
        market_frame["close"] = pd.Series(dtype=float)
    ats.analyze_market_state(market_frame)
    ats.assess_risk_level(market_frame)
    ats.adapt_trading_params(ats.market_state, ats.risk_level)

    build_holdings_meta: List[Dict[str, Any]] = []
    create_event_trades: List[Dict[str, Any]] = []
    for pick in valid_picks:
        price = last_price_map.get(pick.symbol)
        if price is None or price <= 0:
            continue
        target_volume = int(per_value // price)
        if target_volume <= 0:
            continue
        logger.info("[BUILD] symbol=%s price=%s per_value=%s target_volume=%s current_capital_before=%s", pick.symbol, price, per_value, target_volume, ats.current_capital)
        trade = ats.execute_trade(symbol=pick.symbol, signal="BUY", price=price, volume=target_volume)
        logger.info("[BUILD] trade result=%s current_capital_after=%s positions=%s", trade, ats.current_capital, len(ats.positions))
        if not trade.get("success"):
            logger.debug("建仓执行失败 %s: %s", pick.symbol, trade.get("error"))
            continue
        _record_trade(
            pick.symbol,
            "BUY",
            target_volume,
            price,
            pd.Timestamp(initial_ts),
            reason="initial_build",
            event_trades=create_event_trades,
        )
        build_holdings_meta.append(
            {
                "symbol": pick.symbol,
                "weight": per_value / total_value if total_value > 0 else 0.0,
                "shares": target_volume,
                "price": price,
                "entry_price": price,
            }
        )

    total_value = _portfolio_value()
    if build_holdings_meta and total_value > 0:
        for meta in build_holdings_meta:
            meta["weight"] = (meta["shares"] * meta["price"]) / total_value if total_value > 0 else 0.0
    # Use actual positions snapshot for recorded holdings to avoid mismatch between intended
    # target volumes and executed positions (which may be adjusted by ATS). This ensures
    # recorded shares/values reflect real positions used for valuation.
    create_snapshot = _snapshot_holdings(last_price_map, total_value)
    _record_event(
        datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15),
        "create",
        f"回溯建仓 Top{len(create_snapshot)} 持仓",
        create_snapshot,
        total_value,
        trades=create_event_trades,
    )

    last_nav_at = datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
    nav_history.append(
        {
            "date": last_nav_at.strftime("%Y-%m-%d"),
            "nav_date": last_nav_at.isoformat(),
            "nav_value": total_value / initial_capital if initial_capital > 0 else 1.0,
            "total_value": total_value,
        }
    )
    nav_index: Dict[str, int] = {nav_history[-1]["date"]: len(nav_history) - 1}

    for idx, current in enumerate(business_days):
        if current < start_date:
            continue
        # 计算T+1交易日（下一个有效交易日）
        next_trade_day = None
        if idx + 1 < len(business_days):
            next_trade_day = business_days[idx + 1]
        else:
            next_trade_day = current

        # 以T+1开盘价为成交价，获取T+1日的开盘价
        def get_open_price(symbol, ts):
            df_price = _load_price_series(symbol)
            if df_price.empty:
                return None
            if "open" in df_price.columns:
                series = df_price["open"].dropna()
            elif "Open" in df_price.columns:
                series = df_price["Open"].dropna()
            else:
                series = df_price.get("close", pd.Series(dtype=float)).dropna()
            if series.empty:
                return None
            try:
                price = series.asof(ts)
            except Exception:
                price = None
            if price is None or pd.isna(price):
                return None
            return float(price)

        # 更新当前持仓价格（用于止损止盈等）
        current_prices: Dict[str, float] = {}
        for sym in list(ats.positions.keys()):
            price = get_open_price(sym, current)
            if price is None:
                continue
            current_prices[sym] = price
            last_price_map[sym] = price
            last_price_time[sym] = current.to_pydatetime()

        if current_prices:
            alerts = ats.evaluate_positions(current_prices)
            for alert in alerts:
                if alert.get("action") not in {"STOP_LOSS", "TAKE_PROFIT"}:
                    continue
                sym = alert["symbol"]
                trig_price = current_prices.get(sym)
                if trig_price is None:
                    continue
                position_before = ats.positions.get(sym, {})
                volume_before = int(position_before.get("volume", 0))
                result = ats.close_position(sym, trig_price)
                if result.get("success"):
                    event_trades: List[Dict[str, Any]] = []
                    if volume_before > 0:
                        _record_trade(
                            sym,
                            "SELL",
                            volume_before,
                            trig_price,
                            pd.Timestamp(current),
                            reason=alert["action"],
                            event_trades=event_trades,
                        )
                    last_price_map[sym] = trig_price
                    event_type = "stop_loss" if alert["action"] == "STOP_LOSS" else "take_profit"
                    event_time = datetime.combine(current.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
                    event_total = _portfolio_value()
                    _record_event(
                        event_time,
                        event_type,
                        f"{sym} 触发 {alert['action']}",
                        _snapshot_holdings(current_prices, event_total),
                        event_total,
                        trades=event_trades,
                    )

        if current.normalize() in rebal_set and current > start_date:
            # 选股在current日（收盘后），实际调仓按T+1开盘价成交
            price_ts = pd.Timestamp(next_trade_day) if next_trade_day is not None else current
            picks_now = pipeline.pick_stocks(as_of_date=current, candidates=candidate_symbols_all, top_n=top_n*2)
            # 动态补充，确保调仓股票数充足且价格有效
            valid_now: List[PickResult] = []
            for pick in picks_now:
                price = get_open_price(pick.symbol, price_ts)
                if price is None or price <= 0:
                    continue
                valid_now.append(pick)
            if len(valid_now) < top_n:
                all_symbols = pipeline._get_stock_pool(limit=top_n*10)
                for sym in all_symbols:
                    if any(p.symbol == sym for p in valid_now):
                        continue
                    price = get_open_price(sym, price_ts)
                    if price is None or price <= 0:
                        continue
                    valid_now.append(PickResult(symbol=sym, score=0.0, reason="补充", signal=None, risk_score=0.5))
                    if len(valid_now) >= top_n:
                        break
            valid_now = valid_now[:top_n]
            target_symbols: List[str] = []
            pick_prices: Dict[str, float] = {}
            for pick in valid_now:
                price = get_open_price(pick.symbol, price_ts)
                if price is None or price <= 0:
                    continue
                pick_prices[pick.symbol] = price
                last_price_map[pick.symbol] = price
                last_price_time[pick.symbol] = price_ts.to_pydatetime()
                target_symbols.append(pick.symbol)
            # 对于仍缺失价格的 symbol，批量从本地 prices_daily 回填
            missing_for_db = [s for s in target_symbols if s not in pick_prices or pick_prices.get(s) in (None, 0.0)]
            if missing_for_db:
                try:
                    df_latest = _db_manager.get_last_n_bars(symbols=missing_for_db, n=3)
                except Exception:
                    df_latest = None
                if df_latest is not None and not df_latest.empty:
                    for _, row in df_latest.iterrows():
                        sym = row.get('symbol')
                        if not sym or sym not in missing_for_db:
                            continue
                        try:
                            price_val = float(row.get('open') or row.get('Open') or 0.0)
                        except Exception:
                            price_val = 0.0
                        if price_val and price_val > 0:
                            pick_prices[sym] = price_val
                            last_price_map[sym] = price_val
                            last_price_time[sym] = price_ts.to_pydatetime()
            if target_symbols:
                rebalance_trades: List[Dict[str, Any]] = []
                total_value_before = _portfolio_value()
                per_value = total_value_before / len(target_symbols) if target_symbols else 0.0
                trade_day = pd.Timestamp(price_ts)

                existing_symbols = list(ats.positions.keys())
                for sym in existing_symbols:
                    if sym not in target_symbols:
                        price = pick_prices.get(sym)
                        if price is None or price <= 0:
                            price = get_open_price(sym, price_ts) or last_price_map.get(sym)
                        if price is None or price <= 0:
                            continue
                        position_before = ats.positions.get(sym, {})
                        volume_before = int(position_before.get("volume", 0))
                        result = ats.close_position(sym, price)
                        if result.get("success") and volume_before > 0:
                            _record_trade(
                                sym,
                                "SELL",
                                volume_before,
                                price,
                                trade_day,
                                reason="rebalance_exit",
                                event_trades=rebalance_trades,
                            )
                            last_price_map[sym] = price
                            last_price_time[sym] = trade_day.to_pydatetime()

                total_value_after_close = _portfolio_value()
                event_total = total_value_after_close
                if target_symbols:
                    per_value = event_total / len(target_symbols) if event_total > 0 else 0.0

                for sym in target_symbols:
                    price = pick_prices.get(sym)
                    if price is None or price <= 0:
                        price = get_open_price(sym, price_ts) or last_price_map.get(sym)
                    if price is None or price <= 0:
                        continue
                    target_volume = int(per_value // price)
                    if target_volume <= 0:
                        continue
                    prev_volume = int(ats.positions.get(sym, {}).get("volume", 0))
                    delta_volume = target_volume - prev_volume
                    if delta_volume == 0:
                        continue
                    adjust = ats.adjust_position(sym, price=price, target_volume=target_volume)
                    if not adjust.get("success"):
                        logger.debug("调仓失败 %s: %s", sym, adjust.get("error"))
                        continue
                    side = "BUY" if delta_volume > 0 else "SELL"
                    _record_trade(
                        sym,
                        side,
                        abs(delta_volume),
                        price,
                        trade_day,
                        reason="rebalance_adjust",
                        event_trades=rebalance_trades,
                    )
                    last_price_map[sym] = price
                    last_price_time[sym] = trade_day.to_pydatetime()

                event_total = _portfolio_value()
                holdings_snapshot = _snapshot_holdings(pick_prices, event_total)
                event_time = datetime.combine(current.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
                _record_event(
                    event_time,
                    "rebalance",
                    f"{event_time.date()} 调仓",
                    holdings_snapshot,
                    event_total,
                    trades=rebalance_trades,
                )

        total_value = _portfolio_value()
        current_nav_time = datetime.combine(current.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        nav_record = {
            "date": current_nav_time.strftime("%Y-%m-%d"),
            "nav_date": current_nav_time.isoformat(),
            "nav_value": total_value / initial_capital if initial_capital > 0 else 1.0,
            "total_value": total_value,
        }
        date_key = nav_record["date"]
        if date_key in nav_index:
            nav_history[nav_index[date_key]] = nav_record
        else:
            nav_index[date_key] = len(nav_history)
            nav_history.append(nav_record)
        last_nav_at = current_nav_time

    # 生成最终持仓快照
    final_total_value = _portfolio_value()
    holdings_snapshots: List[PortfolioHoldingSnapshot] = []
    quotes: Dict[str, PriceQuote] = {}
    for sym, pos in ats.positions.items():
        volume = float(pos.get("volume", 0))
        price, price_time, prev_price = _price_on(sym, end_date)
        if price is None:
            price = last_price_map.get(sym, 0.0)
        if price_time is not None:
            last_price_time[sym] = price_time.to_pydatetime()
        weight = (volume * price / final_total_value) if final_total_value > 0 else 0.0
        opened_at = pos.get("entry_time")
        if opened_at is None:
            opened_at = datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        elif not isinstance(opened_at, datetime):
            opened_at = datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        # 优先用累计加权平均成本价
        if sym in cost_calc_map and cost_calc_map[sym]["shares"] > 0:
            cost_price = cost_calc_map[sym]["cost"] / cost_calc_map[sym]["shares"]
        else:
            cost_price = float(pos.get("entry_price", price or 0.0))
        snapshot = PortfolioHoldingSnapshot(
            symbol=sym,
            code=_strip_suffix(sym),
            name=_resolve_stock_name(sym, _strip_suffix(sym)),
            weight=weight,
            shares=volume,
            cost_price=cost_price,
            opened_at=opened_at,
        )
        holdings_snapshots.append(snapshot)
        quotes[sym] = PriceQuote(
            latest_price=float(price or 0.0),
            previous_price=float(prev_price) if prev_price else None,
            last_trade_at=last_price_time.get(sym),
        )

    if not nav_history:
        raise RuntimeError("回溯建仓未生成净值数据")

    nav_series = pd.Series(
        data=[item["total_value"] for item in nav_history],
        index=[pd.Timestamp(item["nav_date"]) for item in nav_history],
    )
    initial_value = initial_capital
    final_value = nav_series.iloc[-1]
    total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0
    if len(nav_series) >= 2:
        prev_total = nav_series.iloc[-2]
        daily_return_pct = (final_value - prev_total) / prev_total if prev_total > 0 else 0.0
    else:
        daily_return_pct = 0.0
    valuation = PortfolioValuation(
        nav_total=float(final_value),
        nav_value=float(final_value / initial_capital) if initial_capital > 0 else 1.0,
        daily_return_pct=float(daily_return_pct),
        total_return_pct=float(total_return),
        last_valued_at=last_nav_at,
    )

    return holdings_snapshots, quotes, rebalance_events, nav_history, valuation, last_nav_at


def _build_event_meta(
    holdings: List[Holding],
    event_ts: pd.Timestamp,
    prev_ts: Optional[pd.Timestamp],
    initial_capital: float = 0.0,
    commission_rate: float = 0.0,
    price_provider: Optional[Callable[[str, pd.Timestamp], Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]]] = None,
    prev_price_provider: Optional[Callable[[str, pd.Timestamp], Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    构建事件元数据
    
    Returns:
        Tuple[List[Dict[str, Any]], float]: (持仓元数据列表, 实际持仓总市值)
    """
    price_cache: Dict[str, pd.DataFrame] = {}

    def _load_price_frame(symbol: str, days: int) -> pd.DataFrame:
        cached = price_cache.get(symbol)
        if cached is not None:
            return cached
        from datetime import timedelta

        end_date = event_ts.date()
        start_date = end_date - timedelta(days=days)
        try:
            from src.core.unified_data_access_factory import get_unified_data_access

            uda = get_unified_data_access()
            df = uda.get_stock_daily(symbol, start_date=start_date, end_date=end_date)
        except Exception:
            df = None

        if df is None or df.empty:
            frame = pd.DataFrame()
        else:
            frame = df.copy()
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index().dropna(how="all")
        price_cache[symbol] = frame
        return frame

    def _default_price_on(
        symbol: str,
        ts: pd.Timestamp,
        *,
        days: int = 90,
        preferred_cols: Tuple[str, ...] = ("close", "Close", "last", "Last", "price", "Price"),
    ) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
        frame = _load_price_frame(symbol, days)
        if frame.empty:
            frame = _load_price_frame(symbol, 180)
            if frame.empty:
                return None, None, None

        for col in preferred_cols:
            if col not in frame.columns:
                continue
            series = frame[col].dropna()
            if series.empty:
                continue
            hist = series.loc[:ts]
            price = hist.iloc[-1] if not hist.empty else None
            price_time = hist.index[-1] if not hist.empty else None
            if price is None or pd.isna(price):
                future = series.loc[ts:]
                if not future.empty:
                    price = future.iloc[0]
                    price_time = future.index[0]
                    hist = series.loc[:price_time]
            if price is None or pd.isna(price):
                continue
            prev_candidates = hist.iloc[:-1]
            prev = prev_candidates.iloc[-1] if not prev_candidates.empty else None
            price_time_ts = pd.Timestamp(price_time) if price_time is not None else None
            return float(price), price_time_ts, (float(prev) if prev is not None else None)
        return None, None, None

    def _wrap_provider(
        provider: Callable[[str, pd.Timestamp], Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]]
    ) -> Callable[[str, pd.Timestamp], Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]]:
        def _wrapped(symbol: str, ts: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
            try:
                return provider(symbol, ts)
            except TypeError:
                # 兼容旧式带多余参数的提供者
                return provider(symbol, ts)  # type: ignore[misc]

        return _wrapped

    get_current = _wrap_provider(price_provider) if price_provider else lambda s, t: _default_price_on(s, t)
    get_prev = _wrap_provider(prev_price_provider) if prev_price_provider else get_current

    price_map: Dict[str, float] = {}
    total_value = 0.0
    for h in holdings:
        price_tuple = get_current(h.symbol, event_ts)
        price = float(price_tuple[0]) if price_tuple and price_tuple[0] else 0.0
        price_map[h.symbol] = price
        total_value += price * h.shares
    meta: List[Dict[str, Any]] = []
    for h in holdings:
        price = price_map.get(h.symbol, 0.0)
        prev_price = None
        if prev_ts is not None:
            prev_tuple = get_prev(h.symbol, prev_ts)
            prev_price = float(prev_tuple[0]) if prev_tuple and prev_tuple[0] else None

        entry_price = float(price) if price and price > 0 else None
        prev_price_val = float(prev_price) if prev_price else None
        return_pct = 0.0
        if prev_price_val and prev_price_val > 0 and price > 0:
            return_pct = (price - prev_price_val) / prev_price_val
        weight = (price * h.shares / total_value) if total_value > 0 else h.weight
        meta.append(
            {
                "symbol": h.symbol,
                "weight": weight,
                "shares": h.shares,
                "price": price if price > 0 else None,
                "entry_price": entry_price,
                "return_pct": return_pct,
            }
        )
    # 返回 (meta, total_value) 元组，以便调用方获取实际持仓市值
    return meta, total_value

def _simulate_backtrack_passive(
    *,  # 使用关键字参数
    pipeline: PortfolioPipeline,  # 修复参数类型
    start_ts: datetime,
    end_ts: datetime,
    top_n: int,
    initial_capital: float,
    candidate_limit: Optional[int],
) -> Tuple[List[PortfolioHoldingSnapshot], Dict[str, PriceQuote], List[Dict[str, Any]], List[Dict[str, Any]], PortfolioValuation, datetime]:
    """回溯模拟被动调仓组合表现"""
    
    # ============================================================================
    # P0 性能优化：批量预加载历史数据
    # ============================================================================
    start_date = pd.Timestamp(start_ts.date())
    end_date = pd.Timestamp(end_ts.date())
    candidate_symbols = pipeline._get_stock_pool(limit=candidate_limit)
    
    logger.info(
        "[PASSIVE入口] start=%s end=%s top_n=%s initial_capital=%s candidates=%s",
        start_date, end_date, top_n, initial_capital, len(candidate_symbols)
    )
    
    # 批量预加载所有候选股票的历史数据
    logger.info("[PASSIVE性能优化] 开始批量预加载候选股票历史数据...")
    price_cache: Dict[str, pd.DataFrame] = {}
    
    try:
        from src.core.unified_data_access_factory import get_unified_data_access
        
        uda = get_unified_data_access()
        
        # 计算需要的数据范围（向前多取一些天数以确保有足够历史数据）
        data_start_date = (start_date - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
        data_end_date = end_date.strftime("%Y-%m-%d")
        
        import time
        batch_start_time = time.time()
        
        # 批量获取数据
        bulk_data = uda.get_bulk_stock_data(
            symbols=candidate_symbols,
            start_date=data_start_date,
            end_date=data_end_date
        )
        
        batch_elapsed = time.time() - batch_start_time
        
        # 预处理数据：转换为标准格式并缓存
        successful_count = 0
        for symbol, df in bulk_data.items():
            if df is not None and not df.empty:
                try:
                    frame = df.copy()
                    frame.index = pd.to_datetime(frame.index)
                    frame = frame.sort_index().dropna(how="all")
                    price_cache[symbol] = frame
                    successful_count += 1
                except Exception as e:
                    logger.debug(f"[PASSIVE性能优化] 预处理数据失败 {symbol}: {e}")
        
        logger.info(
            f"[PASSIVE性能优化] 批量预加载完成: 成功={successful_count}/{len(candidate_symbols)} "
            f"耗时={batch_elapsed:.2f}秒"
        )
        
    except Exception as e:
        logger.warning(
            f"[PASSIVE性能优化] 批量预加载失败，将回退到逐只获取模式: {e}",
            exc_info=True
        )
        price_cache = {}
    
    # ============================================================================
    # 将预加载缓存注入到 pipeline 对象，让 pick_stocks 能够使用
    # ============================================================================
    pipeline._preload_cache = price_cache
    logger.info(f"[PASSIVE性能优化] 缓存已注入到pipeline: {len(price_cache)}只股票")
    
    # 确保_price_on函数作为内部函数定义

    def _load_price_series(symbol: str, days: int) -> pd.DataFrame:
        """加载股票价格序列，优先使用预加载的缓存数据"""
        # 优先使用预加载的缓存
        cached = price_cache.get(symbol)
        if cached is not None and not cached.empty:
            return cached
        
        # 缓存中没有数据，回退到逐只获取（带日志）
        logger.debug(f"[PASSIVE回退] 股票 {symbol} 未在批量预加载中，逐只获取数据")
        df = pipeline._fetch_history(symbol, end_date=pd.Timestamp(end_ts.date()), days=days)
        if df is None or df.empty:
            frame = pd.DataFrame()
        else:
            frame = df.copy()
            frame.index = pd.to_datetime(df.index)
            frame = frame.sort_index().dropna(how="all")
        price_cache[symbol] = frame
        return frame

    def _price_on(
        symbol: str,
        ts: pd.Timestamp,
        days: int = 90,
        preferred_cols: Tuple[str, ...] = ("close", "Close", "last", "Last", "price", "Price"),
    ) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
        days_to_fetch = max(days, pipeline.lookback_days + 5)
        frame = _load_price_series(symbol, days_to_fetch)
        if frame.empty:
            frame = _load_price_series(symbol, 180)
            if frame.empty:
                return None, None, None

        for col in preferred_cols:
            if col not in frame.columns:
                continue
            series = frame[col].dropna()
            if series.empty:
                continue
            hist = series.loc[:ts]
            price = hist.iloc[-1] if not hist.empty else None
            price_time = hist.index[-1] if not hist.empty else None
            if price is None or pd.isna(price):
                future = series.loc[ts:]
                if not future.empty:
                    price = future.iloc[0]
                    price_time = future.index[0]
                    hist = series.loc[:price_time]
            if price is None or pd.isna(price):
                continue
            prev_candidates = hist.iloc[:-1]
            prev = prev_candidates.iloc[-1] if not prev_candidates.empty else None
            price_time_ts = pd.Timestamp(price_time) if price_time is not None else None
            return float(price), price_time_ts, (float(prev) if prev is not None else None)
        return None, None, None

    # 生成调仓日期
    rebal_dates = _generate_rebalance_dates(start_date, end_date, _get_rebalance_interval_days())
    if rebal_dates.empty:
        rebal_dates = pd.DatetimeIndex([start_date])

    def _ensure_valid_picks(
        picks: List[PickResult],
        as_of: pd.Timestamp,
        target_n: int,
        *,
        allow_expand: bool = True
    ) -> List[PickResult]:
        valid: List[PickResult] = []
        for pick in picks:
            price, price_time, _ = _price_on(pick.symbol, as_of)
            if price is None or price <= 0:
                logger.debug("[PASSIVE初选] 无效价格 %s @ %s price=%s", pick.symbol, as_of, price)
                continue
            valid.append(pick)
        if len(valid) >= target_n:
            return valid[:target_n]
        if not allow_expand:
            return valid
        # 动态扩充：尝试从更大的股票池补足
        expand_limit = max(target_n * 10, len(candidate_symbols) or 0)
        expand_symbols = pipeline._get_stock_pool(limit=expand_limit)
        logger.info(
            "[PASSIVE补充] 初选有效=%s 目标=%s 扩充池大小=%s",
            len(valid), target_n, len(expand_symbols)
        )
        seen = {p.symbol for p in valid}
        for sym in expand_symbols:
            if sym in seen:
                continue
            price, price_time, _ = _price_on(sym, as_of)
            if price is None or price <= 0:
                logger.debug("[PASSIVE补充] 无效价格 %s @ %s price=%s", sym, as_of, price)
                continue
            valid.append(PickResult(symbol=sym, score=0.0, reason="补充", signal=None, risk_score=0.5))
            seen.add(sym)
            if len(valid) >= target_n:
                break
        return valid[:target_n]

    picks_initial = pipeline.pick_stocks(as_of_date=start_date, candidates=candidate_symbols, top_n=top_n * 2, skip_signals=True)
    logger.info("[PASSIVE初选] pick_stocks返回=%s (skip_signals=True，跳过信号生成)", len(picks_initial))
    valid_initial = _ensure_valid_picks(picks_initial, start_date, top_n)
    if len(valid_initial) < top_n:
        logger.error(
            "回溯建仓失败：指定日期有效候选不足，有效=%s 期望=%s symbols=%s",
            len(valid_initial), top_n, [p.symbol for p in valid_initial]
        )
        raise RuntimeError("回溯建仓失败：指定日期缺少有效候选股票")

    holdings_current = pipeline._equal_weight_holdings(valid_initial, as_of_date=start_date, capital=initial_capital)
    if not holdings_current:
        logger.error("回溯建仓失败：无法根据候选股票生成持仓，有效候选=%s", [p.symbol for p in valid_initial])
        raise RuntimeError("回溯建仓失败：无法根据候选股票生成持仓")

    rebalance_events: List[Dict[str, Any]] = []
    nav_history: List[Dict[str, Any]] = []
    nav_index: Dict[str, int] = {}

    initial_nav_time = datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
    initial_record = {
        "date": initial_nav_time.strftime("%Y-%m-%d"),
        "nav_date": initial_nav_time.isoformat(),
        "nav_value": 1.0,
        "total_value": initial_capital,
    }
    nav_history.append(initial_record)
    nav_index[initial_record["date"]] = 0
    last_nav_at = initial_nav_time
    capital = initial_capital

    create_holdings_meta, create_total_value = _build_event_meta(
        holdings_current,
        start_date,
        None,
        initial_capital=initial_capital,
        commission_rate=0.0,
        price_provider=lambda sym, ts: _price_on(
            sym,
            ts,
            preferred_cols=("open", "Open", "close", "Close", "last", "Last", "price", "Price"),
        ),
        prev_price_provider=lambda sym, ts: _price_on(
            sym,
            ts,
            preferred_cols=("close", "Close", "open", "Open", "last", "Last", "price", "Price"),
        ),
    )
    rebalance_events.append(
        _build_rebalance_event_record(
            event_time=initial_nav_time,
            event_type="create",
            description=f"回溯建仓 Top{len(holdings_current)} 持仓",
            holdings=create_holdings_meta,
            total_value=create_total_value,  # 使用实际持仓市值
            initial_capital=initial_capital,
        )
    )

    prev_rebalance_ts = start_date

    def _calculate_cash_from_meta(meta: List[Dict[str, Any]], available_capital: float) -> float:
        invested = 0.0
        fallback_weight = 0.0
        for item in meta or []:
            shares = float(item.get("shares") or 0.0)
            price = item.get("price")
            if price in (None, "", 0):
                fallback_weight += float(item.get("weight") or 0.0)
                continue
            invested += shares * float(price)
        if fallback_weight > 0 and available_capital > 0:
            invested += available_capital * max(0.0, min(fallback_weight, 1.0))
        cash = available_capital - invested
        return float(cash) if cash > 1e-6 else 0.0

    def _append_nav_series(series: pd.Series, cash_reserved: float) -> None:
        nonlocal capital, last_nav_at
        if series is None or series.empty:
            if cash_reserved > 0:
                nav_time = last_nav_at
                total = cash_reserved
                record = {
                    "date": nav_time.strftime("%Y-%m-%d"),
                    "nav_date": nav_time.isoformat(),
                    "nav_value": total / initial_capital if initial_capital > 0 else 1.0,
                    "total_value": float(total),
                }
                key = record["date"]
                if key in nav_index:
                    nav_history[nav_index[key]] = record
                else:
                    nav_index[key] = len(nav_history)
                    nav_history.append(record)
            return
        total_series = series.astype(float) + cash_reserved
        capital = float(total_series.iloc[-1])
        for ts, total in total_series.items():
            ts = pd.Timestamp(ts)
            nav_time = datetime.combine(ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
            record = {
                "date": nav_time.strftime("%Y-%m-%d"),
                "nav_date": nav_time.isoformat(),
                "nav_value": float(total) / initial_capital if initial_capital > 0 else 1.0,
                "total_value": float(total),
            }
            key = record["date"]
            if key in nav_index:
                nav_history[nav_index[key]] = record
            else:
                nav_index[key] = len(nav_history)
                nav_history.append(record)
            last_nav_at = nav_time

    cash_balance = _calculate_cash_from_meta(create_holdings_meta, initial_capital)

    last_date = start_date
    for dt in rebal_dates:
        if dt == start_date:
            continue
        seg_nav = pipeline._portfolio_nav(holdings_current, start_date=last_date, end_date=dt)
        if seg_nav is not None and not seg_nav.empty:
            _append_nav_series(seg_nav, cash_balance)
        
        # ============================================================================
        # P1 优化：在调仓前计算当前时刻的实际净值
        # ============================================================================
        # 计算调仓时刻（dt）的实际持仓价值，避免使用上一周期的旧净值
        current_total_value = 0.0
        for h in holdings_current:
            price, _, _ = _price_on(h.symbol, pd.Timestamp(dt))
            if price and price > 0:
                current_total_value += price * h.shares
        
        # 如果成功计算出当前净值，则更新 capital
        if current_total_value > 0:
            capital = current_total_value
            logger.debug(
                f"[PASSIVE调仓] 更新capital为调仓时刻净值: dt={dt} capital={capital:.2f} "
                f"(原值={seg_nav.iloc[-1] if seg_nav is not None and not seg_nav.empty else 'N/A'})"
            )
        
        last_date = dt

        picks_now = pipeline.pick_stocks(as_of_date=dt, candidates=candidate_symbols, top_n=top_n * 2, skip_signals=True)
        valid_now = _ensure_valid_picks(picks_now, pd.Timestamp(dt), top_n)
        if len(valid_now) < top_n:
            logger.warning(
                "[PASSIVE调仓] 有效候选不足 date=%s 有效=%s symbols=%s", dt, len(valid_now), [p.symbol for p in valid_now]
            )
            continue
        new_holdings = pipeline._equal_weight_holdings(valid_now, as_of_date=dt, capital=capital)
        if not new_holdings:
            logger.warning(
                "[PASSIVE调仓] 无法生成持仓 date=%s symbols=%s", dt, [p.symbol for p in valid_now]
            )
            continue
        holdings_current = new_holdings
        event_time = datetime.combine(dt.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        holdings_meta, rebalance_total_value = _build_event_meta(
            holdings_current,
            pd.Timestamp(dt),
            prev_rebalance_ts,
            initial_capital=capital,
            commission_rate=0.0,
            price_provider=lambda sym, ts: _price_on(
                sym,
                ts,
                preferred_cols=("open", "Open", "close", "Close", "last", "Last", "price", "Price"),
            ),
            prev_price_provider=lambda sym, ts: _price_on(
                sym,
                ts,
                preferred_cols=("close", "Close", "open", "Open", "last", "Last", "price", "Price"),
            ),
        )
        rebalance_events.append(
            _build_rebalance_event_record(
                event_time=event_time,
                event_type="rebalance",
                description=f"{event_time.date()} 调仓",
                holdings=holdings_meta,
                total_value=rebalance_total_value,  # 使用实际持仓市值
                initial_capital=initial_capital,
            )
        )
        prev_rebalance_ts = pd.Timestamp(dt)
        cash_balance = _calculate_cash_from_meta(holdings_meta, capital)

    if last_date < end_date:
        seg_nav = pipeline._portfolio_nav(holdings_current, start_date=last_date, end_date=end_date)
        if seg_nav is not None and not seg_nav.empty:
            _append_nav_series(seg_nav, cash_balance)

    if not nav_history:
        raise RuntimeError("回溯建仓未生成净值数据")

    cost_price_map: Dict[str, float] = {}
    opened_at_map: Dict[str, datetime] = {}
    if rebalance_events:
        last_event = rebalance_events[-1]
        event_time = _parse_datetime(last_event.get("timestamp")) or last_nav_at
        for item in last_event.get("holdings") or []:
            symbol = item.get("symbol")
            if not symbol:
                continue
            entry_price = item.get("entry_price")
            if entry_price in (None, ""):
                entry_price = item.get("price")
            if entry_price not in (None, ""):
                try:
                    cost_price_map[symbol] = float(entry_price)  # type: ignore[arg-type]
                except Exception:
                    pass
            opened_at_map[symbol] = event_time

    snapshots = _build_holding_snapshots(
        holdings_current,
        pipeline=pipeline,
        initial_capital=initial_capital,
        as_of=last_nav_at,
        cost_price_map=cost_price_map if cost_price_map else None,
        opened_at_map=opened_at_map if opened_at_map else None,
        portfolio_id=-1,  # 回溯模拟使用临时组合ID
    )

    quotes: Dict[str, PriceQuote] = {}
    final_total_value = nav_history[-1]["total_value"]
    for snapshot in snapshots:
        price, price_time, prev_price = _price_on(snapshot.symbol, pd.Timestamp(end_date))
        latest_price = float(price or 0.0)
        quotes[snapshot.symbol] = PriceQuote(
            latest_price=latest_price,
            previous_price=float(prev_price) if prev_price else None,
            last_trade_at=price_time.to_pydatetime() if price_time is not None else None,
        )

    if len(nav_history) >= 2:
        prev_total_value = nav_history[-2]["total_value"]
        daily_return_pct = (final_total_value - prev_total_value) / prev_total_value if prev_total_value > 0 else 0.0
    else:
        daily_return_pct = 0.0
    total_return_pct = (final_total_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
    valuation = PortfolioValuation(
        nav_total=float(final_total_value),
        nav_value=float(nav_history[-1]["nav_value"]),
        daily_return_pct=float(daily_return_pct),
        total_return_pct=float(total_return_pct),
        last_valued_at=last_nav_at,
    )

    return snapshots, quotes, rebalance_events, nav_history, valuation, last_nav_at


def _persist_portfolio(
    detail: PortfolioDetail,
    valuation: PortfolioValuation,
    quotes: Dict[str, PriceQuote],
    nav_history: List[Dict[str, Any]],
) -> int:
    placeholder = _placeholder()
    created_at = detail.created_at
    updated_at = detail.created_at
    last_valued_at = valuation.last_valued_at or detail.created_at
    portfolio_columns = [
        "name",
        "top_n",
        "initial_capital",
        "holdings_count",
        "benchmark",
        "risk_level",
        "strategy_tags",
        "notes",
        "nav_value",
        "total_value",
        "daily_return_pct",
        "total_return_pct",
        "last_valued_at",
        "created_at",
        "updated_at",
    ]
    portfolio_values = [
        detail.name,
        detail.top_n,
        detail.initial_capital,
        detail.holdings_count,
        detail.benchmark,
        detail.risk_level,
        _json_dumps(detail.strategy_tags),
        detail.notes,
        valuation.nav_value,
        valuation.nav_total,
        valuation.daily_return_pct,
        valuation.total_return_pct,
        last_valued_at,
        created_at,
        updated_at,
    ]

    holdings_rows = []
    for holding in detail.holdings:
        quote = quotes.get(holding.symbol) or PriceQuote(latest_price=0.0, previous_price=None, last_trade_at=None)
        holdings_rows.append(
            [
                holding.symbol,
                holding.code,
                holding.name,
                holding.weight,
                holding.shares,
                holding.cost_price,
                quote.latest_price,
                quote.previous_price,
                holding.opened_at,
                quote.last_trade_at,
                created_at,
                updated_at,
            ]
        )

    nav_rows = [
        [
            item.get("date"),
            item.get("nav_value"),
            item.get("total_value"),
            created_at,
        ]
        for item in nav_history
    ]

    rebalance_rows = []
    for record in detail.rebalance_history:
        payload = {
            "holdings": record.get("holdings"),
        }
        if record.get("nav_value") is not None:
            payload["nav_value"] = record.get("nav_value")
        if record.get("total_value") is not None:
            payload["total_value"] = record.get("total_value")
        rebalance_rows.append(
            [
                _parse_datetime(record.get("timestamp")) or created_at,
                record.get("type"),
                record.get("description"),
                _json_dumps(payload),
                created_at,
            ]
        )

    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()
        try:
            insert_sql = (
                f"INSERT INTO portfolios ({', '.join(portfolio_columns)}) VALUES ({_placeholders(len(portfolio_columns))})"
            )
            cursor.execute(insert_sql, portfolio_values)
            portfolio_id = cursor.lastrowid

            if holdings_rows:
                delete_holdings = f"DELETE FROM portfolio_holdings WHERE portfolio_id = {placeholder}"
                cursor.execute(delete_holdings, (portfolio_id,))
                insert_holdings = (
                    "INSERT INTO portfolio_holdings (portfolio_id, symbol, code, name, weight, shares, cost_price, latest_price, previous_price, opened_at, last_trade_at, created_at, updated_at) "
                    f"VALUES ({placeholder}, {_placeholders(12)})"
                )
                params = [
                    [portfolio_id] + row for row in holdings_rows
                ]
                cursor.executemany(insert_holdings, params)

            if nav_rows:
                delete_nav = f"DELETE FROM portfolio_nav_history WHERE portfolio_id = {placeholder}"
                cursor.execute(delete_nav, (portfolio_id,))
                insert_nav = (
                    "INSERT INTO portfolio_nav_history (portfolio_id, nav_date, nav_value, total_value, created_at) "
                    f"VALUES ({placeholder}, {_placeholders(4)})"
                )
                params = [
                    [portfolio_id] + row for row in nav_rows
                ]
                cursor.executemany(insert_nav, params)

            if rebalance_rows:
                insert_rebalance = (
                    "INSERT INTO portfolio_rebalances (portfolio_id, event_time, event_type, description, details, created_at) "
                    f"VALUES ({placeholder}, {_placeholders(5)})"
                )
                params = [
                    [portfolio_id] + row for row in rebalance_rows
                ]
                cursor.executemany(insert_rebalance, params)

            conn.commit()
            return int(portfolio_id)
        except Exception:
            conn.rollback()
            logger.exception("写入组合数据失败")
            raise
        finally:
            cursor.close()


def _update_portfolio(
    pid: int,
    detail: PortfolioDetail,
    valuation: PortfolioValuation,
    quotes: Dict[str, PriceQuote],
    nav_history: List[Dict[str, Any]],
) -> None:
    placeholder = _placeholder()
    updated_at = _now()
    last_valued_at = valuation.last_valued_at or updated_at
    ph = _placeholder()
    update_sql = (
        f"UPDATE portfolios SET holdings_count = {ph}, nav_value = {ph}, total_value = {ph}, daily_return_pct = {ph}, "
        f"total_return_pct = {ph}, last_valued_at = {ph}, updated_at = {ph} WHERE id = {ph}"
    )
    summary_params = (
        detail.holdings_count,
        valuation.nav_value,
        valuation.nav_total,
        valuation.daily_return_pct,
        valuation.total_return_pct,
        last_valued_at,
        updated_at,
        pid,
    )

    holdings_rows = []
    for holding in detail.holdings:
        quote = quotes.get(holding.symbol) or PriceQuote(latest_price=0.0, previous_price=None, last_trade_at=None)
        holdings_rows.append(
            [
                holding.symbol,
                holding.code,
                holding.name,
                holding.weight,
                holding.shares,
                holding.cost_price,
                quote.latest_price,
                quote.previous_price,
                holding.opened_at,
                quote.last_trade_at,
                updated_at,
            ]
        )

    nav_rows = [
        [
            item.get("date"),
            item.get("nav_value"),
            item.get("total_value"),
            updated_at,
        ]
        for item in nav_history
    ]

    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(update_sql, summary_params)

            delete_holdings = f"DELETE FROM portfolio_holdings WHERE portfolio_id = {placeholder}"
            cursor.execute(delete_holdings, (pid,))
            if holdings_rows:
                insert_holdings = (
                    "INSERT INTO portfolio_holdings (portfolio_id, symbol, code, name, weight, shares, cost_price, latest_price, previous_price, opened_at, last_trade_at, created_at, updated_at) "
                    f"VALUES ({placeholder}, {_placeholders(12)})"
                )
                params = [
                    [pid] + row[:-1] + [row[-1], row[-1]]
                    for row in holdings_rows
                ]
                cursor.executemany(insert_holdings, params)

            delete_nav = f"DELETE FROM portfolio_nav_history WHERE portfolio_id = {placeholder}"
            cursor.execute(delete_nav, (pid,))
            if nav_rows:
                insert_nav = (
                    "INSERT INTO portfolio_nav_history (portfolio_id, nav_date, nav_value, total_value, created_at) "
                    f"VALUES ({placeholder}, {_placeholders(4)})"
                )
                params = [
                    [pid] + row for row in nav_rows
                ]
                cursor.executemany(insert_nav, params)

            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("更新组合数据失败")
            raise
        finally:
            cursor.close()


def _calculate_cost_price(
    weight: float,
    shares: float,
    initial_capital: float,
    commission_rate: float = 0.0,
    override_price: Optional[float] = None,
) -> float:
    """
    统一计算股票持仓的成本价格
    
    参数:
        weight: 持仓权重
        shares: 持仓份额
        initial_capital: 初始资金
        commission_rate: 佣金率，默认为0
        override_price: 覆盖的成本价格，如不为None且大于0则直接使用
    
    返回:
        计算得到的成本价格
    """
    if override_price is not None and override_price > 0:
        return float(override_price)
    
    if shares <= 0 or initial_capital <= 0:
        return 0.0
    
    alloc = initial_capital * weight
    # 确保分配资金大于0，避免除以零或产生极小值
    if alloc <= 0:
        return 0.0
        
    return (alloc * (1 - commission_rate)) / shares


def _build_holding_snapshots(
    holdings: List[Holding],
    *,
    pipeline: PortfolioPipeline,
    initial_capital: float,
    as_of: datetime,
    cost_price_map: Optional[Dict[str, float]] = None,
    opened_at_map: Optional[Dict[str, datetime]] = None,
    portfolio_id: Optional[int] = None,
) -> List[PortfolioHoldingSnapshot]:
    """
    构建持仓快照，优先使用基于交易流水的成本价计算
    
    参数:
        portfolio_id: 投资组合ID，用于获取交易流水计算成本价
    """
    snapshots: List[PortfolioHoldingSnapshot] = []
    commission = getattr(pipeline, "commission_rate", 0.0)
    
    for h in holdings:
        symbol = h.symbol
        code = _strip_suffix(symbol)
        name = _resolve_stock_name(symbol, code)
        
        # 优先使用覆盖的成本价
        override = cost_price_map.get(symbol) if cost_price_map else None
        if override is not None and override > 0:
            cost_price = float(override)
        # 其次使用基于交易流水的成本价
        elif portfolio_id is not None:
            avg_cost_from_trades = _calculate_avg_cost_from_trades(portfolio_id, symbol)
            if avg_cost_from_trades is not None and avg_cost_from_trades > 0:
                cost_price = avg_cost_from_trades
            else:
                # 兜底：使用权重计算成本价
                cost_price = _calculate_cost_price(
                    h.weight,
                    h.shares,
                    initial_capital,
                    commission,
                    None
                )
        else:
            # 没有portfolio_id时使用权重计算
            cost_price = _calculate_cost_price(
                h.weight,
                h.shares,
                initial_capital,
                commission,
                None
            )
        
        opened_at_value = (
            opened_at_map.get(symbol)
            if opened_at_map and opened_at_map.get(symbol)
            else as_of
        )
        snapshots.append(
            PortfolioHoldingSnapshot(
                symbol=symbol,
                code=code,
                name=name,
                weight=h.weight,
                shares=h.shares,
                cost_price=cost_price,
                opened_at=opened_at_value,
            )
        )
    return snapshots


def _load_snapshots_from_rows(holdings_rows: List[Dict[str, Any]]) -> List[PortfolioHoldingSnapshot]:
    snapshots: List[PortfolioHoldingSnapshot] = []
    for row in holdings_rows:
        symbol = row.get("symbol") or ""
        code = row.get("code") or _strip_suffix(symbol)
        snapshots.append(
            PortfolioHoldingSnapshot(
                symbol=symbol,
                code=code,
                name=row.get("name") or _resolve_stock_name(symbol, code),
                weight=float(row.get("weight") or 0.0),
                shares=float(row.get("shares") or 0.0),
                cost_price=float(row.get("cost_price") or 0.0),
                opened_at=_parse_datetime(row.get("opened_at")) or _now(),
            )
        )
    return snapshots


def _calculate_daily_nav_with_rebalance_history(
    pipeline: Any,
    detail: PortfolioDetail,
    rebalance_history: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
) -> List[Dict[str, Any]]:
    """
    基于调仓历史计算每日净值。
    
    核心逻辑：
    1. 将调仓历史按时间排序，每次调仓代表一个持仓期间
    2. 对每个持仓期间，获取该期间所有股票的每日价格
    3. 计算每日组合总价值和净值
    """
    if not rebalance_history:
        return []
    
    # 按时间排序调仓历史
    sorted_rebalances = sorted(rebalance_history, key=lambda x: x.get('timestamp', ''))
    
    nav_history = []
    initial_capital = detail.initial_capital if detail.initial_capital > 0 else 1.0
    
    for i, rebalance in enumerate(sorted_rebalances):
        # 当前持仓
        holdings = rebalance.get('holdings', [])
        if not holdings:
            continue
        
        # 确定这个持仓期间的起止时间
        period_start_str = rebalance.get('timestamp', '')
        if not period_start_str:
            continue
        period_start = _parse_datetime(period_start_str)
        if not period_start:
            continue
        
        # 下一次调仓时间或结束时间
        if i < len(sorted_rebalances) - 1:
            next_rebalance_str = sorted_rebalances[i + 1].get('timestamp', '')
            period_end = _parse_datetime(next_rebalance_str) if next_rebalance_str else end_date
        else:
            period_end = end_date
        
        # 不能超过查询的结束时间
        if period_end > end_date:
            period_end = end_date
        
        # 获取这个期间的所有股票代码
        symbols = [h.get('symbol') for h in holdings if h.get('symbol')]
        if not symbols:
            continue
        
        # 获取价格数据
        try:
            start_str = (pd.Timestamp(period_start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end_str = pd.Timestamp(period_end).strftime("%Y-%m-%d")
            batch_data = pipeline.data_access.get_bulk_stock_data(symbols, start_date=start_str, end_date=end_str)
        except (AttributeError, Exception):
            days = (period_end - period_start).days + 10
            batch_data = {sym: pipeline._fetch_history(sym, end_date=pd.Timestamp(period_end), days=days) for sym in symbols}
        
        # 处理每个股票的价格数据
        price_frames = {}
        for symbol in symbols:
            df = batch_data.get(symbol)
            if df is None or df.empty:
                continue
            frame = df.copy()
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index()
            # 只保留当前期间的数据
            frame = frame[(frame.index >= pd.Timestamp(period_start)) & (frame.index <= pd.Timestamp(period_end))]
            if not frame.empty:
                price_frames[symbol] = frame
        
        if not price_frames:
            continue
        
        # 获取所有交易日期的并集
        index_union = None
        for frame in price_frames.values():
            idx = frame.index
            index_union = idx if index_union is None else index_union.union(idx)
        
        if index_union is None:
            continue
        
        # 为每个交易日计算净值
        for dt in index_union.sort_values():
            total_value = 0.0
            valid = True
            
            for holding in holdings:
                symbol = holding.get('symbol')
                shares = holding.get('shares', 0)
                
                if not symbol or not shares:
                    continue
                
                frame = price_frames.get(symbol)
                if frame is None or dt not in frame.index:
                    valid = False
                    break
                
                # 查找价格列
                price_col = None
                for col in ("close", "Close", "last", "Last", "price", "Price"):
                    if col in frame.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    valid = False
                    break
                
                price = float(frame.loc[dt, price_col])
                total_value += shares * price
            
            if valid:
                nav_history.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "nav_value": round(total_value / initial_capital if initial_capital else 0.0, 6),
                    "total_value": round(total_value, 2),
                })
    
    # 去重和排序
    if nav_history:
        unique_map = {}
        for row in nav_history:
            unique_map[row["date"]] = row
        nav_history = [unique_map[d] for d in sorted(unique_map.keys())]
    
    return nav_history


def _calculate_portfolio_valuation(
    detail: PortfolioDetail,
    *,
    as_of: datetime,
) -> Tuple[PortfolioValuation, Dict[str, PriceQuote], List[Dict[str, Any]]]:
    pipeline = _get_pipeline(detail.initial_capital, detail.top_n)
    price_frames: Dict[str, pd.DataFrame] = {}
    nav_total = 0.0
    prev_total = 0.0
    prev_available = True
    last_trade: Optional[datetime] = None
    quotes: Dict[str, PriceQuote] = {}

    end_ts = pd.Timestamp(as_of)
    symbols = [h.symbol for h in detail.holdings]
    
    # 计算需要获取数据的起始日期：从组合创建日期开始（而不是固定的90天）
    # 这样可以确保获取到组合整个生命周期的每日净值数据
    if detail.created_at:
        start_ts = pd.Timestamp(detail.created_at)
        # 往前多取几天以确保数据完整
        data_start_date = (start_ts - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    else:
        # 如果没有创建日期，默认取180天
        data_start_date = (end_ts - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    
    if symbols:
        try:
            # 获取从组合创建日期到现在的历史数据，以便计算每日净值
            batch_data = pipeline.data_access.get_bulk_stock_data(symbols, start_date=data_start_date, end_date=end_ts.strftime("%Y-%m-%d"))
        except AttributeError:
            # 计算天数
            days_needed = max((end_ts - start_ts).days + 10, 180) if detail.created_at else 180
            batch_data = {sym: pipeline._fetch_history(sym, end_date=end_ts, days=days_needed) for sym in symbols}
    else:
        batch_data = {}

    for holding in detail.holdings:
        df = batch_data.get(holding.symbol)
        if df is None or df.empty:
            # 如果初始获取失败，尝试获取更多历史数据（180天）
            df = pipeline._fetch_history(holding.symbol, end_date=end_ts, days=180)
        if df is None or df.empty:
            quotes[holding.symbol] = PriceQuote(latest_price=0.0, previous_price=None, last_trade_at=None)
            continue
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        frame = frame[frame.index <= end_ts]
        price_frames[holding.symbol] = frame
        price_col = None
        for col in ("close", "Close", "last", "Last", "price", "Price"):
            if col in frame.columns:
                price_col = col
                break
        if price_col is None or frame.empty:
            quotes[holding.symbol] = PriceQuote(latest_price=0.0, previous_price=None, last_trade_at=None)
            continue
        latest_price = float(frame[price_col].iloc[-1])
        previous_price = float(frame[price_col].iloc[-2]) if len(frame) > 1 else None
        last_ts = pd.Timestamp(frame.index[-1]).to_pydatetime()
        nav_total += holding.shares * latest_price
        if previous_price is not None:
            prev_total += holding.shares * previous_price
        else:
            prev_available = False
        if last_ts and (last_trade is None or last_ts > last_trade):
            last_trade = last_ts
        quotes[holding.symbol] = PriceQuote(
            latest_price=latest_price,
            previous_price=previous_price,
            last_trade_at=last_ts,
        )

    initial_capital = detail.initial_capital if detail.initial_capital > 0 else 1.0
    nav_value = nav_total / initial_capital if initial_capital else 0.0
    total_return_pct = (nav_total - initial_capital) / initial_capital if initial_capital else 0.0
    daily_return_pct = (nav_total - prev_total) / prev_total if prev_available and prev_total > 0 else 0.0
    as_of_date = as_of.date()
    stale_trading_day = False
    if symbols:
        if last_trade is None:
            stale_trading_day = True
        else:
            last_trade_date = last_trade.date()
            stale_trading_day = last_trade_date < as_of_date
    if stale_trading_day:
        daily_return_pct = 0.0
        for quote in quotes.values():
            quote.previous_price = None
    valuation = PortfolioValuation(
        nav_total=nav_total,
        nav_value=nav_value,
        daily_return_pct=daily_return_pct,
        total_return_pct=total_return_pct,
        last_valued_at=last_trade or as_of,
    )

    index_union: Optional[pd.DatetimeIndex] = None
    for frame in price_frames.values():
        idx = frame.index
        index_union = idx if index_union is None else index_union.union(idx)

    # 获取组合创建日期，用于过滤净值历史（只保留创建日期及之后的数据）
    created_date = detail.created_at.date() if detail.created_at else None
    
    # 如果有调仓历史，使用基于调仓历史的每日净值计算方法
    # 这样可以正确反映不同时期的持仓变化
    nav_history: List[Dict[str, Any]] = []
    if detail.rebalance_history and len(detail.rebalance_history) > 0:
        # 使用调仓历史计算每日净值
        try:
            nav_history = _calculate_daily_nav_with_rebalance_history(
                pipeline=pipeline,
                detail=detail,
                rebalance_history=detail.rebalance_history,
                start_date=detail.created_at,
                end_date=as_of,
            )
            logger.info(f"基于调仓历史计算每日净值成功: {len(nav_history)} 条记录")
        except Exception as e:
            logger.warning(f"使用调仓历史计算每日净值失败: {e}，回退到当前持仓计算", exc_info=True)
            # 回退到原有逻辑
            nav_history = []
    
    # 如果没有调仓历史或计算失败，使用当前持仓计算（原有逻辑）
    if not nav_history and index_union is not None:
        logger.info("使用当前持仓计算净值历史")
        base_capital = initial_capital if initial_capital > 0 else 1.0
        for dt in index_union.sort_values():
            # 过滤掉组合创建日期之前的数据
            if created_date and dt.date() < created_date:
                continue
                
            total_value = 0.0
            valid = True
            for holding in detail.holdings:
                frame = price_frames.get(holding.symbol)
                if frame is None or dt not in frame.index:
                    valid = False
                    break
                price_col = None
                for col in ("close", "Close", "last", "Last", "price", "Price"):
                    if col in frame.columns:
                        price_col = col
                        break
                if price_col is None:
                    valid = False
                    break
                price_val = float(frame.loc[dt, price_col])
                total_value += holding.shares * price_val
            if not valid:
                continue
            nav_history.append(
                {
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "nav_value": round(total_value / base_capital if base_capital else 0.0, 6),
                    "total_value": round(total_value, 2),
                }
            )

    # ---------------------------------------------------------------------
    # 去重处理：有些数据源索引可能出现重复日期（如 1970-01-01），
    # 这会导致写入 portfolio_nav_history 表时违反唯一索引约束。
    # 使用字典确保日期唯一，然后保持日期顺序输出。
    # ---------------------------------------------------------------------
    if nav_history:
        unique_map: Dict[str, Dict[str, Any]] = {}
        for row in nav_history:
            unique_map[row["date"]] = row  # 后出现的记录会覆盖之前的值，保持更靠后的价格
        # 按日期顺序排序，保留最近 180 条
        nav_history = [unique_map[d] for d in sorted(unique_map.keys())]

    return valuation, quotes, nav_history[-180:]


def _should_refresh(portfolio_row: Dict[str, Any], nav_rows: List[Dict[str, Any]], as_of: datetime) -> bool:
    last_valued = _parse_datetime(portfolio_row.get("last_valued_at"))
    if last_valued is None:
        return True
    delta_seconds = (as_of - last_valued).total_seconds()
    if delta_seconds >= _REFRESH_THRESHOLD_MINUTES * 60:
        return True
    if nav_rows:
        last_nav = nav_rows[-1]
        last_nav_date = last_nav.get("nav_date")
        if last_nav_date:
            nav_dt = _parse_datetime(last_nav_date)
            if nav_dt and nav_dt.date() < as_of.date():
                return True
    return False


def _refresh_portfolio(pid: int, portfolio_row: Dict[str, Any], holdings_rows: List[Dict[str, Any]], as_of: datetime) -> None:
    """刷新组合估值，包括每日净值计算"""
    holdings = _load_snapshots_from_rows(holdings_rows)
    
    # 获取调仓历史用于计算每日净值
    rebalance_query = (
        f"SELECT event_time, event_type, description, details FROM portfolio_rebalances "
        f"WHERE portfolio_id = {_placeholder()} ORDER BY event_time ASC"
    )
    rebalance_rows = _db_manager.execute_query(rebalance_query, (pid,))
    
    # 构建调仓历史
    initial_capital = float(portfolio_row.get("initial_capital") or 0.0)
    created_at = _parse_datetime(portfolio_row.get("created_at")) or _now()
    rebalance_history = []
    for row in rebalance_rows:
        details = _json_loads(row.get("details"), None)
        holdings_payload = details.get("holdings") if isinstance(details, dict) else details
        formatted_holdings = _format_rebalance_holdings(holdings_payload, initial_capital=initial_capital, adjust_mode="qfq") if holdings_payload else []
        event_ts = _parse_datetime(row.get("event_time")) or created_at
        rebalance_history.append(
            {
                "timestamp": event_ts.isoformat(),
                "date": event_ts.date().isoformat() if hasattr(event_ts, "date") else None,
                "type": row.get("event_type"),
                "description": row.get("description"),
                "holdings": formatted_holdings,
            }
        )
    
    detail = PortfolioDetail(
        id=int(portfolio_row["id"]),
        name=portfolio_row.get("name") or "",
        created_at=created_at,
        top_n=int(portfolio_row.get("top_n") or 0),
        initial_capital=initial_capital,
        holdings_count=int(portfolio_row.get("holdings_count") or len(holdings)),
        benchmark=portfolio_row.get("benchmark") or "",
        risk_level=portfolio_row.get("risk_level") or _infer_risk_level(int(portfolio_row.get("top_n") or 0)),
        strategy_tags=_json_loads(portfolio_row.get("strategy_tags"), []) or [],
        holdings=holdings,
        notes=portfolio_row.get("notes"),
        rebalance_history=rebalance_history,
    )
    valuation, quotes, nav_history = _calculate_portfolio_valuation(detail, as_of=as_of)
    detail.holdings_count = len(holdings)
    _update_portfolio(pid, detail, valuation, quotes, nav_history)


# =========================================================================
# Public API
# =========================================================================

def list_portfolios(as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
    rows = _fetch_portfolio_rows()
    summaries: List[Dict[str, Any]] = []
    for row in rows:
        info = _hydrate_summary(row)
        valuation = _valuation_from_row(row)
        summaries.append(info.summary(valuation))
    return summaries


def get_portfolio_detail(pid: int, as_of: Optional[datetime] = None, refresh: bool = True) -> Optional[Dict[str, Any]]:
    as_of_dt = as_of or _now()
    bundle = _fetch_portfolio_bundle(pid)
    if bundle is None:
        return None
    portfolio_row, holdings_rows, nav_rows, rebalance_rows = bundle
    if refresh and _should_refresh(portfolio_row, nav_rows, as_of_dt):
        try:
            _refresh_portfolio(pid, portfolio_row, holdings_rows, as_of_dt)
            bundle = _fetch_portfolio_bundle(pid)
            if bundle is None:
                return None
            portfolio_row, holdings_rows, nav_rows, rebalance_rows = bundle
        except Exception:
            logger.warning("刷新组合估值失败，返回现有数据", exc_info=True)
    detail, valuation, quotes, nav_history = _hydrate_detail(portfolio_row, holdings_rows, nav_rows, rebalance_rows, as_of_dt)
    return detail.to_dict(valuation, quotes, nav_history, as_of_dt)


def create_portfolio_auto(
    name: str,
    top_n: int = 20,
    initial_capital: float = 1_000_000.0,
    *,
    auto_trading: Optional[bool] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
    candidate_limit: Optional[int] = None,
) -> Dict[str, Any]:
    pipeline = _get_pipeline(initial_capital, top_n)
    auto_trading_flag = _parse_optional_bool(auto_trading, _default_auto_trading())
    candidate_limit_resolved = resolve_candidate_limit(candidate_limit if candidate_limit is not None else pipeline.candidate_limit)
    pipeline.reset_weights()
    if weight_overrides:
        pipeline.apply_weight_overrides(weight_overrides)
    try:
        as_of_dt = _now()
        use_selector = os.getenv("PORTFOLIO_AUTO_USE_SELECTOR", "1").lower() not in {"0", "false", "off"}
        symbol_limit_env = int(os.getenv("PORTFOLIO_AUTO_SYMBOL_LIMIT", str(_AUTO_SYMBOL_LIMIT_DEFAULT)) or _AUTO_SYMBOL_LIMIT_DEFAULT)
        symbol_limit = candidate_limit_resolved if candidate_limit_resolved is not None else symbol_limit_env

        holdings_raw: List[Holding] = []
        holdings_meta: List[Dict[str, Any]] = []
        used_selector = False
        used_price_fallback = False

        if use_selector:
            try:
                picks_meta = _fetch_quick_picks(top_n, symbol_limit, force_refresh=False)
                holdings_raw, holdings_meta = _build_holdings_from_picks(
                    picks_meta,
                    initial_capital=initial_capital,
                    pipeline=pipeline,
                )
                used_selector = bool(holdings_raw)
                if not holdings_raw:
                    logger.warning("快速选股未返回有效持仓，回退至 PortfolioPipeline")
            except Exception:
                logger.exception("快速选股建仓失败，回退至 PortfolioPipeline")
                holdings_raw = []
                holdings_meta = []

        if not holdings_raw:
            recent_symbols = _symbols_with_recent_data(
                limit=max(top_n * 3, symbol_limit),
                lookback_days=_FALLBACK_LOOKBACK_DAYS,
                min_rows=max(5, _FALLBACK_MIN_ROWS),
            )
            holdings_raw, holdings_meta = _build_holdings_from_recent_prices(
                recent_symbols,
                initial_capital=initial_capital,
                pipeline=pipeline,
            )
            used_price_fallback = bool(holdings_raw)
            if used_price_fallback:
                logger.info("使用本地等权兜底方案生成组合持仓（symbols=%s）", len(holdings_raw))

        if not holdings_raw:
            candidate_symbols = pipeline._get_stock_pool(limit=candidate_limit_resolved)
            picks: List[PickResult] = pipeline.pick_stocks(as_of_date=as_of_dt, candidates=candidate_symbols, top_n=top_n)
            holdings_raw = pipeline._equal_weight_holdings(
                picks,
                as_of_date=as_of_dt,
                capital=initial_capital,
            )
            for holding in holdings_raw:
                price = _get_latest_price(holding.symbol)
                holdings_meta.append(
                    {
                        "symbol": holding.symbol,
                        "weight": holding.weight,
                        "shares": holding.shares,
                        "price": price if price > 0 else None,
                        "probability": None,
                        "expected_return": None,
                        "score": None,
                        "sentiment": None,
                    }
                )

        if not holdings_raw:
            raise RuntimeError("无法生成组合持仓，请稍后重试")

        meta_map = {meta["symbol"]: meta for meta in holdings_meta}
        cost_price_map_auto: Dict[str, float] = {}
        for symbol, meta in meta_map.items():
            price_val = meta.get("price")
            if price_val in (None, ""):
                continue
            try:
                cost_price_map_auto[symbol] = float(price_val)  # type: ignore[arg-type]
            except Exception:
                continue
        # 为了确保创建组合时净值为1.0，我们需要确保成本价格与初始资本一致
        # 构建一个基于初始资本的成本价格映射
        if holdings_raw:
            # 计算基于初始资本的每股价格
            total_weight = sum(h.weight for h in holdings_raw)
            if total_weight > 0:
                for h in holdings_raw:
                    # 确保成本价格映射存在
                    if not cost_price_map_auto:
                        cost_price_map_auto = {}
                    # 使用基于权重和初始资本计算的价格，而不是市场价格
                    # 这确保了初始净值计算为1.0
                    # 使用归一化权重计算成本价格
                    normalized_weight = h.weight / total_weight
                    cost_price_map_auto[h.symbol] = _calculate_cost_price(
                        normalized_weight,
                        h.shares,
                        initial_capital,
                        0.0,  # 这里不使用佣金，因为目的是确保初始净值为1.0
                        None
                    )
        
        snapshots = _build_holding_snapshots(
            holdings_raw,
            pipeline=pipeline,
            initial_capital=initial_capital,
            as_of=as_of_dt,
            cost_price_map=cost_price_map_auto if cost_price_map_auto else None,
            portfolio_id=-1,  # 临时组合，尚未持久化
        )
        detail = PortfolioDetail(
            id=-1,
            name=name,
            created_at=as_of_dt,
            top_n=top_n,
            initial_capital=initial_capital,
            holdings_count=len(snapshots),
            benchmark="沪深300",
            risk_level=_infer_risk_level(top_n),
            strategy_tags=[
                "智能选股",
                f"Top{top_n}",
                "快速建仓" if used_selector else ("本地等权" if used_price_fallback else "等权分散"),
                "自动调仓" if auto_trading_flag else "自动调仓:关闭",
            ],
            holdings=snapshots,
            notes=(
                "智能推荐自动建仓（快速选股模式）"
                if used_selector
                else (
                    "数据库等权兜底建仓"
                    if used_price_fallback
                    else "智能推荐自动建仓"
                )
            ),
            rebalance_history=[
                _build_rebalance_event_record(
                    event_time=as_of_dt,
                    event_type="create",
                    description=f"自动建仓 Top{top_n} 持仓",
                    holdings=[
                        {
                            "symbol": h.symbol,
                            "weight": h.weight,
                            "shares": h.shares,
                            "price": meta_map.get(h.symbol, {}).get("price"),
                            "entry_price": meta_map.get(h.symbol, {}).get("price"),
                            "return_pct": 0.0,
                            "probability": meta_map.get(h.symbol, {}).get("probability"),
                            "expected_return": meta_map.get(h.symbol, {}).get("expected_return"),
                            "score": meta_map.get(h.symbol, {}).get("score"),
                            "sentiment": meta_map.get(h.symbol, {}).get("sentiment"),
                        }
                        for h in snapshots
                    ],
                    total_value=initial_capital,
                    initial_capital=initial_capital,
                )
            ],
        )
        valuation, quotes, nav_history = _calculate_portfolio_valuation(detail, as_of=as_of_dt)
        portfolio_id = _persist_portfolio(detail, valuation, quotes, nav_history)
        return get_portfolio_detail(portfolio_id, as_of=as_of_dt, refresh=False) or {}
    finally:
        pipeline.reset_weights()


def create_portfolio_backtrack(
    *,
    name: str,
    backtrack_date: date,
    top_n: int = 20,
    initial_capital: float = 1_000_000.0,
    auto_trading: Optional[bool] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
    candidate_limit: Optional[int] = None,
) -> Dict[str, Any]:
    pipeline = _get_pipeline(initial_capital, top_n)
    data_access = getattr(pipeline, "data_access", None)
    if backtrack_date > _now().date():
        raise ValueError("回溯日期不能晚于当前日期")
    effective_date = backtrack_date
    if data_access and not is_trading_day(data_access, effective_date):
        try:
            next_day = next_trading_day(data_access, effective_date)
            logger.info("回溯起始日 %s 非交易日，顺延至 %s", effective_date, next_day)
            effective_date = next_day
        except Exception:
            logger.warning("回溯起始日 %s 顺延失败，继续使用原日期", effective_date, exc_info=True)
    backtrack_date = effective_date
    auto_trading_flag = _parse_optional_bool(auto_trading, _default_auto_trading())
    start_ts = datetime.combine(backtrack_date, datetime.min.time(), tzinfo=_TIMEZONE) + timedelta(hours=15)
    end_ts = _now()
    holdings_snapshots, quotes, rebalance_events, nav_history, valuation, last_nav_at = _simulate_backtrack_portfolio(
        pipeline=pipeline,
        start_ts=start_ts,
        end_ts=end_ts,
        top_n=top_n,
        initial_capital=initial_capital,
        auto_trading=bool(auto_trading_flag),
        weight_overrides=weight_overrides,
        candidate_limit=candidate_limit,
    )
    strategy_tags = [
        "智能选股",
        f"Top{top_n}",
        "回溯建仓",
        "自动调仓" if auto_trading_flag else "自动调仓:关闭",
    ]
    detail = PortfolioDetail(
        id=-1,
        name=name,
        created_at=start_ts,
        top_n=top_n,
        initial_capital=initial_capital,
        holdings_count=len(holdings_snapshots),
        benchmark="沪深300",
        risk_level=_infer_risk_level(top_n),
        strategy_tags=strategy_tags,
        holdings=holdings_snapshots,
        notes=f"回溯建仓（起始 {start_ts.date().isoformat()}）",
        rebalance_history=rebalance_events,
    )
    # 覆盖估值时间，以回溯模拟结果为准
    valuation.last_valued_at = last_nav_at
    portfolio_id = _persist_portfolio(detail, valuation, quotes, nav_history)
    return get_portfolio_detail(portfolio_id, as_of=end_ts, refresh=False) or {}


def delete_portfolio(pid: int) -> bool:
    """删除指定组合，返回是否删除成功。"""
    select_query = f"SELECT id FROM portfolios WHERE id = {_placeholder()}"
    rows = _db_manager.execute_query(select_query, (pid,))
    if not rows:
        return False
    delete_query = f"DELETE FROM portfolios WHERE id = {_placeholder()}"
    _db_manager.execute_update(delete_query, (pid,))
    return True


def create_portfolio_manual_stub(*args, **kwargs):
    """Placeholder for manual portfolio creation."""
    raise NotImplementedError("Manual portfolio creation not implemented yet")


# =========================================================================
# 交易流水相关函数
# =========================================================================

def _persist_trades_bulk(portfolio_id: int, trades: List[Dict[str, Any]]) -> None:
    """
    批量写入交易流水记录
    
    Args:
        portfolio_id: 投资组合ID
        trades: 交易记录列表，每个记录包含symbol, side, qty, price, fee, trade_ts, note, related_event_id
    """
    if not trades:
        return
    
    placeholder = _placeholder()
    sql = (
        "INSERT INTO portfolio_trades (portfolio_id, symbol, side, qty, price, fee, trade_ts, note, related_event_id) "
        f"VALUES ({placeholder}, {_placeholders(8)})"
    )
    
    values = []
    for trade in trades:
        values.append((
            portfolio_id,
            trade.get("symbol") or "",
            trade.get("side") or "BUY",
            int(trade.get("qty") or 0),
            float(trade.get("price") or 0.0),
            float(trade.get("fee") or 0.0),
            trade.get("trade_ts") or _now(),
            trade.get("note") or "",
            trade.get("related_event_id") or None
        ))
    
    try:
        _db_manager.execute_many(sql, values)
        logger.info("成功写入 %d 条交易记录到组合 %d", len(trades), portfolio_id)
    except Exception as exc:
        logger.exception("批量写入交易记录失败: %s", exc)
        raise


def _fetch_portfolio_trades(portfolio_id: int, symbol: Optional[str] = None, 
                           start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    查询投资组合的交易流水记录
    
    Args:
        portfolio_id: 投资组合ID
        symbol: 股票代码筛选（可选）
        start_date: 开始时间筛选（可选）
        end_date: 结束时间筛选（可选）
    
    Returns:
        交易记录列表
    """
    placeholder = _placeholder()
    query = f"SELECT * FROM portfolio_trades WHERE portfolio_id = {placeholder}"
    params = [portfolio_id]
    
    if symbol:
        query += f" AND symbol = {placeholder}"
        params.append(symbol)
    
    if start_date:
        query += f" AND trade_ts >= {placeholder}"
        params.append(start_date)
    
    if end_date:
        query += f" AND trade_ts <= {placeholder}"
        params.append(end_date)
    
    query += " ORDER BY trade_ts DESC, id DESC"
    
    return _db_manager.execute_query(query, params)


def _calculate_avg_cost_from_trades(portfolio_id: int, symbol: str) -> Optional[float]:
    """
    基于交易流水计算指定股票的平均成本价
    
    Args:
        portfolio_id: 投资组合ID
        symbol: 股票代码
    
    Returns:
        平均成本价，如果没有交易记录则返回None
    """
    trades = _fetch_portfolio_trades(portfolio_id, symbol)
    if not trades:
        return None
    
    total_shares = 0
    total_cost = 0.0
    
    for trade in trades:
        qty = int(trade.get("qty") or 0)
        price = float(trade.get("price") or 0.0)
        fee = float(trade.get("fee") or 0.0)
        side = trade.get("side") or "BUY"
        
        if side == "BUY":
            total_shares += qty
            total_cost += (qty * price) + fee
        elif side == "SELL":
            # 卖出时，先计算当前平均成本
            if total_shares > 0:
                avg_cost = total_cost / total_shares
                # 卖出部分股票，减少总成本
                total_cost -= qty * avg_cost
            total_shares -= qty
            
            # 如果卖出后股数为0，重置成本
            if total_shares <= 0:
                total_cost = 0.0
    
    if total_shares > 0:
        return round(total_cost / total_shares, 6)
    else:
        return None


def _update_holding_avg_cost(portfolio_id: int, symbol: str) -> bool:
    """
    更新持仓记录的平均成本价
    
    Args:
        portfolio_id: 投资组合ID
        symbol: 股票代码
    
    Returns:
        是否成功更新
    """
    avg_cost = _calculate_avg_cost_from_trades(portfolio_id, symbol)
    
    placeholder = _placeholder()
    update_sql = f"UPDATE portfolio_holdings SET avg_cost = {placeholder} WHERE portfolio_id = {placeholder} AND symbol = {placeholder}"
    
    try:
        _db_manager.execute_update(update_sql, (avg_cost, portfolio_id, symbol))
        logger.debug("更新组合 %d 股票 %s 的平均成本价为 %s", portfolio_id, symbol, avg_cost)
        return True
    except Exception as exc:
        logger.exception("更新持仓平均成本价失败: %s", exc)
        return False


def _get_portfolio_cash_balance(portfolio_id: int) -> float:
    """
    获取投资组合的现金余额
    
    Args:
        portfolio_id: 投资组合ID
    
    Returns:
        现金余额
    """
    placeholder = _placeholder()
    query = f"SELECT cash_balance FROM portfolios WHERE id = {placeholder}"
    
    try:
        rows = _db_manager.execute_query(query, (portfolio_id,))
        if rows:
            return float(rows[0].get("cash_balance") or 0.0)
        else:
            return 0.0
    except Exception as exc:
        logger.exception("获取组合现金余额失败: %s", exc)
        return 0.0


def _update_portfolio_cash_balance(portfolio_id: int, new_balance: float) -> bool:
    """
    更新投资组合的现金余额
    
    Args:
        portfolio_id: 投资组合ID
        new_balance: 新的现金余额
    
    Returns:
        是否成功更新
    """
    placeholder = _placeholder()
    update_sql = f"UPDATE portfolios SET cash_balance = {placeholder} WHERE id = {placeholder}"
    
    try:
        _db_manager.execute_update(update_sql, (new_balance, portfolio_id))
        logger.debug("更新组合 %d 的现金余额为 %s", portfolio_id, new_balance)
        return True
    except Exception as exc:
        logger.exception("更新组合现金余额失败: %s", exc)
        return False