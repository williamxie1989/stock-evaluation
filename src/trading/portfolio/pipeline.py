from __future__ import annotations

# === PATCH: build_equal_weight_trades ===
from typing import List, Tuple, Dict, Any
from datetime import date

# 兼容导入 utils
try:
    from src.services.portfolio.portfolio_utils import session_price_qfq
except Exception:
    from services.portfolio.portfolio_utils import session_price_qfq

# 兼容 Holding 类型（你的项目应已有该类型）
try:
    from src.trading.portfolio.types import Holding  # noqa
except Exception:
    from collections import namedtuple
    Holding = namedtuple("Holding", ["symbol", "weight", "shares"])  # 简易兜底

def build_equal_weight_trades(pipeline, picks: List[str], as_of_date: date,
                              initial_capital: float, commission_rate: float,
                              use_open_price: bool = True) -> Tuple[List[Holding], List[Dict[str, Any]]]:
    """
    生成等权建仓的持仓与交易流水（以 as_of_date 的开盘/收盘价执行）；
    外层应保证 as_of_date 为交易日（非交易日请先顺延）。
    """
    if not picks:
        return [], []
    alloc = initial_capital / max(1, len(picks))
    session = "open" if use_open_price else "close"
    holdings, trades = [], []
    for sym in picks:
        px = session_price_qfq(pipeline, sym, as_of_date, session=session)
        qty = int((alloc * (1 - commission_rate)) // px)
        if qty <= 0:
            continue
        fee = round(alloc - qty * px, 6)  # 手续费+零头统一视作费用
        trades.append({"symbol": sym, "side": "BUY", "qty": qty, "price": px, "fee": max(fee, 0.0)})
        holdings.append(Holding(symbol=sym, weight=1.0/len(picks), shares=qty))
    return holdings, trades

