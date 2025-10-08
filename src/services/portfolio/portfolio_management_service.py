"""Portfolio management service: database-backed CRUD with valuation refresh."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

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
            "created_at": self.created_at.isoformat(),
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
            "last_valued_at": valuation.last_valued_at.isoformat() if valuation.last_valued_at else None,
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
_REBALANCE_INTERVAL_DAYS = int(os.getenv("PORTFOLIO_REBALANCE_INTERVAL_DAYS", "30"))


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


def _format_rebalance_holdings(holdings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """标准化调仓事件中的持仓数据，补充名称、收益率与市值。"""
    formatted: List[Dict[str, Any]] = []
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
        if entry_price <= 0 and price > 0:
            entry_price = price
        value = shares * price if price > 0 else 0.0
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
    return formatted


def _build_rebalance_event_record(
    *,
    event_time: datetime,
    event_type: str,
    description: str,
    holdings: List[Dict[str, Any]],
    total_value: float,
    initial_capital: float,
) -> Dict[str, Any]:
    """生成包含净值信息的调仓事件记录。"""
    nav_value: Optional[float] = None
    if initial_capital and initial_capital > 0:
        nav_value = total_value / initial_capital
    record: Dict[str, Any] = {
        "timestamp": event_time.isoformat(),
        "type": event_type,
        "description": description,
        "holdings": _format_rebalance_holdings(holdings),
        "total_value": round(total_value, 2),
    }
    if nav_value is not None:
        record["nav_value"] = round(nav_value, 6)
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

    nav_history = [
        {
            "date": (row.get("nav_date") or "") if isinstance(row.get("nav_date"), str) else (_parse_datetime(row.get("nav_date")).strftime("%Y-%m-%d") if _parse_datetime(row.get("nav_date")) else ""),
            "nav_value": float(row.get("nav_value") or 0.0),
            "total_value": float(row.get("total_value") or 0.0),
        }
        for row in nav_rows
    ]

    rebalance_history = []
    for row in rebalance_rows:
        details = _json_loads(row.get("details"), None)
        holdings_payload = details.get("holdings") if isinstance(details, dict) else details
        nav_value = details.get("nav_value") if isinstance(details, dict) else None
        total_value = details.get("total_value") if isinstance(details, dict) else None
        formatted_holdings = _format_rebalance_holdings(holdings_payload) if holdings_payload else []
        rebalance_history.append(
            {
                "timestamp": (_parse_datetime(row.get("event_time")) or info.created_at).isoformat(),
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
    pipeline.reset_weights()
    if weight_overrides:
        pipeline.apply_weight_overrides(weight_overrides)
    try:
        if auto_trading:
            return _simulate_backtrack_active(
                pipeline=pipeline,
                start_ts=start_ts,
                end_ts=end_ts,
                top_n=top_n,
                initial_capital=initial_capital,
                candidate_limit=resolved_limit,
            )
        return _simulate_backtrack_passive(
            pipeline=pipeline,
            start_ts=start_ts,
            end_ts=end_ts,
            top_n=top_n,
            initial_capital=initial_capital,
            candidate_limit=resolved_limit,
        )
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
    from src.trading.systems.adaptive_trading_system import AdaptiveTradingSystem

    ats = AdaptiveTradingSystem(initial_capital=initial_capital)
    price_cache: Dict[str, pd.Series] = {}
    last_price_map: Dict[str, float] = {}
    last_price_time: Dict[str, datetime] = {}
    nav_history: List[Dict[str, Any]] = []
    rebalance_events: List[Dict[str, Any]] = []

    start_date = pd.Timestamp(start_ts.date())
    end_date = pd.Timestamp(end_ts.date())
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    # 至少包含起始日
    if start_date not in business_days:
        business_days = business_days.insert(0, start_date)
    rebal_dates = _generate_rebalance_dates(start_date, end_date, _REBALANCE_INTERVAL_DAYS)
    rebal_set = {dt.normalize() for dt in rebal_dates}
    candidate_symbols_all = pipeline._get_stock_pool(limit=candidate_limit)

    def _load_price_series(symbol: str) -> pd.Series:
        series = price_cache.get(symbol)
        if series is not None:
            return series
        total_days = max(5, int((end_date - start_date).days) + pipeline.lookback_days + 5)
        df = pipeline._fetch_history(symbol, end_date=end_date, days=total_days)
        if df is None or df.empty or "close" not in df.columns:
            price_cache[symbol] = pd.Series(dtype=float)
            return price_cache[symbol]
        ser = df["close"].copy()
        ser.index = pd.to_datetime(df.index)
        ser = ser.sort_index()
        ser = ser.dropna()
        price_cache[symbol] = ser
        return ser

    def _price_on(symbol: str, ts: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
        ser = _load_price_series(symbol)
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
        total = ats.current_capital
        for sym, pos in ats.positions.items():
            price = last_price_map.get(sym)
            if price is None:
                price, _, _ = _price_on(sym, pd.Timestamp(end_date))
            if price is None:
                continue
            total += float(pos.get("volume", 0)) * float(price)
        return float(total)

    def _record_event(
        event_time: datetime,
        event_type: str,
        description: str,
        holdings_snapshot: List[Dict[str, Any]],
        total_value: float,
    ) -> None:
        record = _build_rebalance_event_record(
            event_time=event_time,
            event_type=event_type,
            description=description,
            holdings=holdings_snapshot,
            total_value=total_value,
            initial_capital=initial_capital,
        )
        rebalance_events.append(record)

    def _snapshot_holdings(current_prices: Dict[str, float], total_value: float) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for sym, pos in ats.positions.items():
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
        return snapshot

    # 首日初始化建仓
    initial_ts = start_date
    picks_initial = pipeline.pick_stocks(as_of_date=initial_ts, candidates=candidate_symbols_all, top_n=top_n)
    valid_picks: List[PickResult] = []
    for pick in picks_initial:
        price, price_time, _ = _price_on(pick.symbol, initial_ts)
        if price is None or price <= 0:
            continue
        last_price_map[pick.symbol] = price
        if price_time is not None:
            last_price_time[pick.symbol] = price_time.to_pydatetime()
        valid_picks.append(pick)
    if not valid_picks:
        raise RuntimeError("回溯建仓失败：指定日期缺少有效价格数据")

    total_value = initial_capital
    per_value = total_value / len(valid_picks)
    ats.analyze_market_state(_load_price_series(valid_picks[0].symbol).to_frame(name="close"))
    ats.assess_risk_level(_load_price_series(valid_picks[0].symbol).to_frame(name="close"))
    ats.adapt_trading_params(ats.market_state, ats.risk_level)

    build_holdings_meta: List[Dict[str, Any]] = []
    for pick in valid_picks:
        price = last_price_map.get(pick.symbol)
        if price is None or price <= 0:
            continue
        target_volume = int(per_value // price)
        if target_volume <= 0:
            continue
        trade = ats.execute_trade(symbol=pick.symbol, signal="BUY", price=price, volume=target_volume)
        if not trade.get("success"):
            logger.debug("建仓执行失败 %s: %s", pick.symbol, trade.get("error"))
            continue
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
    _record_event(
        datetime.combine(start_ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15),
        "create",
        f"回溯建仓 Top{len(build_holdings_meta)} 持仓",
        build_holdings_meta,
        total_value,
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

    for current in business_days:
        if current < start_date:
            continue
        current_prices: Dict[str, float] = {}
        for sym in list(ats.positions.keys()):
            price, price_time, _ = _price_on(sym, current)
            if price is None:
                continue
            current_prices[sym] = price
            last_price_map[sym] = price
            if price_time is not None:
                last_price_time[sym] = price_time.to_pydatetime()

        if current_prices:
            alerts = ats.evaluate_positions(current_prices)
            for alert in alerts:
                if alert.get("action") not in {"STOP_LOSS", "TAKE_PROFIT"}:
                    continue
                sym = alert["symbol"]
                trig_price = current_prices.get(sym)
                if trig_price is None:
                    continue
                result = ats.close_position(sym, trig_price)
                if result.get("success"):
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
                    )

        if current.normalize() in rebal_set and current > start_date:
            picks_now = pipeline.pick_stocks(as_of_date=current, candidates=candidate_symbols_all, top_n=top_n)
            target_symbols: List[str] = []
            pick_prices: Dict[str, float] = {}
            for pick in picks_now:
                price, price_time, _ = _price_on(pick.symbol, current)
                if price is None or price <= 0:
                    continue
                pick_prices[pick.symbol] = price
                last_price_map[pick.symbol] = price
                if price_time is not None:
                    last_price_time[pick.symbol] = price_time.to_pydatetime()
                target_symbols.append(pick.symbol)

            if target_symbols:
                total_value_before = _portfolio_value()
                per_value = total_value_before / len(target_symbols) if target_symbols else 0.0

                existing_symbols = list(ats.positions.keys())
                for sym in existing_symbols:
                    if sym not in target_symbols:
                        price = current_prices.get(sym) or last_price_map.get(sym)
                        if price is None:
                            continue
                        ats.close_position(sym, price)
                total_value_after_close = _portfolio_value()
                if target_symbols:
                    per_value = total_value_after_close / len(target_symbols) if total_value_after_close > 0 else 0.0
                holdings_snapshot: List[Dict[str, Any]] = []

                for sym in target_symbols:
                    price = pick_prices.get(sym) or last_price_map.get(sym)
                    if price is None or price <= 0:
                        continue
                    target_volume = int(per_value // price)
                    if target_volume <= 0:
                        continue
                    adjust = ats.adjust_position(sym, price=price, target_volume=target_volume)
                    if not adjust.get("success"):
                        logger.debug("调仓失败 %s: %s", sym, adjust.get("error"))
                        continue
                    holdings_snapshot.append(
                        {
                            "symbol": sym,
                            "weight": per_value / total_value if total_value > 0 else 0.0,
                            "shares": target_volume,
                            "price": price,
                            "entry_price": price,
                        }
                    )
                event_total = _portfolio_value()
                if holdings_snapshot and event_total > 0:
                    for meta in holdings_snapshot:
                        meta["weight"] = (meta["shares"] * meta["price"]) / event_total if event_total > 0 else 0.0

                event_time = datetime.combine(current.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
                _record_event(event_time, "rebalance", f"{event_time.date()} 调仓", holdings_snapshot, event_total)

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


def _simulate_backtrack_passive(
    *,
    pipeline: PortfolioPipeline,
    start_ts: datetime,
    end_ts: datetime,
    top_n: int,
    initial_capital: float,
    candidate_limit: Optional[int],
) -> Tuple[List[PortfolioHoldingSnapshot], Dict[str, PriceQuote], List[Dict[str, Any]], List[Dict[str, Any]], PortfolioValuation, datetime]:
    price_cache: Dict[str, pd.Series] = {}

    def _load_price_series(symbol: str, days: int) -> pd.Series:
        cached = price_cache.get(symbol)
        if cached is not None:
            return cached
        df = pipeline._fetch_history(symbol, end_date=pd.Timestamp(end_ts.date()), days=days)
        if df is None or df.empty or "close" not in df.columns:
            price_cache[symbol] = pd.Series(dtype=float)
            return price_cache[symbol]
        ser = df["close"].copy()
        ser.index = pd.to_datetime(df.index)
        ser = ser.sort_index().dropna()
        price_cache[symbol] = ser
        return ser

    def _price_on(symbol: str, ts: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
        days = max(5, pipeline.lookback_days + 5)
        series = _load_price_series(symbol, days)
        if series.empty:
            return None, None, None
        try:
            price = series.asof(ts)
        except Exception:
            price = None
        if price is None or pd.isna(price):
            return None, None, None
        hist = series[series.index <= ts]
        prev = hist.iloc[-2] if len(hist) >= 2 else None
        price_time = hist.index[-1] if len(hist) >= 1 else None
        return float(price), price_time, (float(prev) if prev is not None else None)

    def _build_event_meta(
        holdings: List[Holding],
        event_ts: pd.Timestamp,
        prev_ts: Optional[pd.Timestamp],
    ) -> List[Dict[str, Any]]:
        price_map: Dict[str, float] = {}
        total_value = 0.0
        for h in holdings:
            price, _, _ = _price_on(h.symbol, event_ts)
            price = float(price) if price else 0.0
            price_map[h.symbol] = price
            total_value += price * h.shares
        meta: List[Dict[str, Any]] = []
        for h in holdings:
            price = price_map.get(h.symbol, 0.0)
            prev_price = None
            if prev_ts is not None:
                prev_price, _, _ = _price_on(h.symbol, prev_ts)
            entry_price = price if price > 0 else None
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
        return meta

    start_date = pd.Timestamp(start_ts.date())
    end_date = pd.Timestamp(end_ts.date())
    rebal_dates = _generate_rebalance_dates(start_date, end_date, _REBALANCE_INTERVAL_DAYS)
    if rebal_dates.empty:
        rebal_dates = pd.DatetimeIndex([start_date])

    candidate_symbols = pipeline._get_stock_pool(limit=candidate_limit)
    picks_initial = pipeline.pick_stocks(as_of_date=start_date, candidates=candidate_symbols, top_n=top_n)
    if not picks_initial:
        raise RuntimeError("回溯建仓失败：指定日期缺少有效候选股票")

    holdings_current = pipeline._equal_weight_holdings(picks_initial, as_of_date=start_date, capital=initial_capital)
    if not holdings_current:
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

    create_holdings_meta = _build_event_meta(holdings_current, start_date, None)
    rebalance_events.append(
        _build_rebalance_event_record(
            event_time=initial_nav_time,
            event_type="create",
            description=f"回溯建仓 Top{len(holdings_current)} 持仓",
            holdings=create_holdings_meta,
            total_value=initial_capital,
            initial_capital=initial_capital,
        )
    )

    prev_rebalance_ts = start_date

    def _append_nav_series(series: pd.Series) -> None:
        nonlocal capital, last_nav_at
        if series is None or series.empty:
            return
        capital = float(series.iloc[-1])
        for ts, value in series.items():
            ts = pd.Timestamp(ts)
            nav_time = datetime.combine(ts.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
            record = {
                "date": nav_time.strftime("%Y-%m-%d"),
                "nav_date": nav_time.isoformat(),
                "nav_value": value / initial_capital if initial_capital > 0 else 1.0,
                "total_value": float(value),
            }
            key = record["date"]
            if key in nav_index:
                nav_history[nav_index[key]] = record
            else:
                nav_index[key] = len(nav_history)
                nav_history.append(record)
            last_nav_at = nav_time

    last_date = start_date
    for dt in rebal_dates:
        if dt == start_date:
            continue
        seg_nav = pipeline._portfolio_nav(holdings_current, start_date=last_date, end_date=dt)
        if seg_nav is not None and not seg_nav.empty:
            _append_nav_series(seg_nav)
        last_date = dt

        picks_now = pipeline.pick_stocks(as_of_date=dt, candidates=candidate_symbols, top_n=top_n)
        new_holdings = pipeline._equal_weight_holdings(picks_now, as_of_date=dt, capital=capital)
        if not new_holdings:
            continue
        holdings_current = new_holdings
        event_time = datetime.combine(dt.date(), datetime.min.time(), tzinfo=start_ts.tzinfo) + timedelta(hours=15)
        holdings_meta = _build_event_meta(holdings_current, pd.Timestamp(dt), prev_rebalance_ts)
        rebalance_events.append(
            _build_rebalance_event_record(
                event_time=event_time,
                event_type="rebalance",
                description=f"{event_time.date()} 调仓",
                holdings=holdings_meta,
                total_value=capital,
                initial_capital=initial_capital,
            )
        )
        prev_rebalance_ts = pd.Timestamp(dt)

    if last_date < end_date:
        seg_nav = pipeline._portfolio_nav(holdings_current, start_date=last_date, end_date=end_date)
        if seg_nav is not None and not seg_nav.empty:
            _append_nav_series(seg_nav)

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


def _build_holding_snapshots(
    holdings: List[Holding],
    *,
    pipeline: PortfolioPipeline,
    initial_capital: float,
    as_of: datetime,
    cost_price_map: Optional[Dict[str, float]] = None,
    opened_at_map: Optional[Dict[str, datetime]] = None,
) -> List[PortfolioHoldingSnapshot]:
    snapshots: List[PortfolioHoldingSnapshot] = []
    commission = getattr(pipeline, "commission_rate", 0.0)
    for h in holdings:
        symbol = h.symbol
        code = _strip_suffix(symbol)
        name = _resolve_stock_name(symbol, code)
        override = cost_price_map.get(symbol) if cost_price_map else None
        if override is not None and override > 0:
            cost_price = float(override)
        else:
            alloc = initial_capital * h.weight
            cost_price = (alloc * (1 - commission)) / h.shares if h.shares > 0 else 0.0
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
    if symbols:
        try:
            batch_data = pipeline.data_access.get_stock_data_batch(symbols, start_date=(end_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), end_date=end_ts.strftime("%Y-%m-%d"))  # type: ignore[attr-defined]
        except AttributeError:
            batch_data = {sym: pipeline._fetch_history(sym, end_date=end_ts, days=10) for sym in symbols}
    else:
        batch_data = {}

    for holding in detail.holdings:
        df = batch_data.get(holding.symbol)
        if df is None or df.empty:
            df = pipeline._fetch_history(holding.symbol, end_date=end_ts, days=10)
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

    nav_history: List[Dict[str, Any]] = []
    if index_union is not None:
        base_capital = initial_capital if initial_capital > 0 else 1.0
        for dt in index_union.sort_values():
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
    holdings = _load_snapshots_from_rows(holdings_rows)
    detail = PortfolioDetail(
        id=int(portfolio_row["id"]),
        name=portfolio_row.get("name") or "",
        created_at=_parse_datetime(portfolio_row.get("created_at")) or _now(),
        top_n=int(portfolio_row.get("top_n") or 0),
        initial_capital=float(portfolio_row.get("initial_capital") or 0.0),
        holdings_count=int(portfolio_row.get("holdings_count") or len(holdings)),
        benchmark=portfolio_row.get("benchmark") or "",
        risk_level=portfolio_row.get("risk_level") or _infer_risk_level(int(portfolio_row.get("top_n") or 0)),
        strategy_tags=_json_loads(portfolio_row.get("strategy_tags"), []) or [],
        holdings=holdings,
        notes=portfolio_row.get("notes"),
        rebalance_history=[],
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
        snapshots = _build_holding_snapshots(
            holdings_raw,
            pipeline=pipeline,
            initial_capital=initial_capital,
            as_of=as_of_dt,
            cost_price_map=cost_price_map_auto if cost_price_map_auto else None,
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
    if backtrack_date > _now().date():
        raise ValueError("回溯日期不能晚于当前日期")
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
