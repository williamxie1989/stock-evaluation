"""Portfolio management service: create/list/get portfolios.
Currently uses in-memory store; can be replaced by DB later.
"""
from __future__ import annotations

import threading
import itertools
from datetime import datetime
from typing import List, Dict, Optional

from dataclasses import dataclass, asdict

from src.trading.portfolio.portfolio_pipeline import PortfolioPipeline, Holding, PickResult

# =========================================================================
# Dataclasses
# =========================================================================

@dataclass
class PortfolioInfo:
    id: int
    name: str
    created_at: datetime
    top_n: int
    initial_capital: float
    holdings_count: int

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["created_at"] = self.created_at.isoformat()
        return out


@dataclass
class PortfolioDetail(PortfolioInfo):
    holdings: List[Holding]

    def to_dict(self) -> Dict:
        base = super().to_dict()
        base["holdings"] = [holding_to_dict(h) for h in self.holdings]
        return base


# =========================================================================
# Utility
# =========================================================================

def holding_to_dict(h: Holding) -> Dict:
    return {
        "symbol": h.symbol,
        "weight": h.weight,
        "shares": h.shares,
    }

# =========================================================================
# In-memory store
# =========================================================================

_lock = threading.Lock()
_next_id = itertools.count(1)
_portfolios: Dict[int, PortfolioDetail] = {}


# =========================================================================
# Public API
# =========================================================================

def list_portfolios() -> List[Dict]:
    """Return list of portfolio summaries."""
    with _lock:
        return [p.to_dict() for p in _portfolios.values()]


def get_portfolio_detail(pid: int) -> Optional[Dict]:
    with _lock:
        p = _portfolios.get(pid)
        return p.to_dict() if p else None


def create_portfolio_auto(name: str, top_n: int = 20, initial_capital: float = 1_000_000.0) -> Dict:
    """Automatically create portfolio using PortfolioPipeline."""
    pipeline = PortfolioPipeline(top_n=top_n, initial_capital=initial_capital)
    picks: List[PickResult] = pipeline.pick_stocks(top_n=top_n)
    as_of_date = datetime.now().date()
    holdings = pipeline._equal_weight_holdings(picks, as_of_date=datetime.now(), capital=initial_capital)  # type: ignore

    info = PortfolioDetail(
        id=-1,  # placeholder, will set later
        name=name,
        created_at=datetime.utcnow(),
        top_n=top_n,
        initial_capital=initial_capital,
        holdings_count=len(holdings),
        holdings=holdings,
    )
    with _lock:
        pid = next(_next_id)
        info.id = pid
        _portfolios[pid] = info
    return info.to_dict()


def create_portfolio_manual_stub(*args, **kwargs):
    """Placeholder for manual portfolio creation."""
    raise NotImplementedError("Manual portfolio creation not implemented yet")