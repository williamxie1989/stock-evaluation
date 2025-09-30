from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Signal:
    """通用交易信号类型（跨 signals/systems 使用）"""
    date: datetime
    type: str  # BUY/SELL
    price: float
    reason: str
    confidence: float
    factor: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "daily"
    volume_confirmation: bool = False
    multi_timeframe_confirmed: bool = False
