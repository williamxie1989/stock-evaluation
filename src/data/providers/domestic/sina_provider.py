"""
Sina 财经数据提供器 - 基础实现
历史日线使用 JSON 接口: http://finance.sina.com.cn/finance/api/json.php?page=1&num=2000&symbol=SH600519&&data=DailyData
该接口无跨域限制, 返回类似 [{"day":"2023-10-09","open":"1480.0", ...}]
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

import pandas as pd

from src.core.interfaces import DataProviderInterface
from src.data.providers.base_provider import DataProviderBase

logger = logging.getLogger(__name__)


class SinaDataProvider(DataProviderBase, DataProviderInterface):
    """通过 新浪财经 接口获取股票数据"""

    BASE_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"

    def __init__(self, timeout: int = 8, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        logger.info("SinaDataProvider 初始化完成")

    @staticmethod
    def _symbol_to_sina_code(symbol: str) -> str:
        symbol = symbol.upper()
        if symbol.endswith(".SH"):
            return f"sh{symbol[:-3]}"
        if symbol.endswith(".SZ"):
            return f"sz{symbol[:-3]}"
        return symbol

    # --------------------------------------------
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        code = self._symbol_to_sina_code(symbol)
        params = {
            "symbol": code,
            "scale": 240,  # 240 分钟近似等价于日K
            "ma": "no",
            "datalen": 1023,
        }
        try:
            # referer 头避免被拒绝
            headers = {"Referer": "https://finance.sina.com.cn"}
            resp = self._request("GET", self.BASE_URL, params=params, headers=headers)
            if resp is None or resp.status_code != 200:
                return None
            try:
                json_data: List[Dict[str, Any]] = json.loads(resp.text)
            except Exception:
                logger.warning("Sina 返回数据无法解析 JSON, raw=%s", resp.text[:120])
                return None
            if not json_data:
                return None
            df = pd.DataFrame(json_data)
            df.rename(columns={
                "day": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])

            # 将数值列转换为数值类型，避免后续比较时出现 '<=' not supported between str and int
            numeric_cols = ["open", "close", "high", "low", "volume"]
            for col in numeric_cols:
                # errors='coerce' 会将无法解析为数字的值设为 NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 丢弃无法解析为数字的行
            df.dropna(subset=numeric_cols, inplace=True)

            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            df = df.loc[mask]
            df["symbol"] = symbol.upper()
            # 重新排序列，保持统一
            return df[["open", "close", "high", "low", "volume", "symbol"]]
        except Exception as e:
            logger.error("Sina get_stock_data failed %s", e)
            return None

    # 占位
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        return {}

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        return None