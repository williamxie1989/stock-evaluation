"""
Eastmoney 数据提供器
使用东财无 token 接口获取股票历史数据和实时数据
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.interfaces import DataProviderInterface
from src.data.providers.base_provider import DataProviderBase

logger = logging.getLogger(__name__)


class EastmoneyDataProvider(DataProviderBase, DataProviderInterface):
    """通过东方财富开放接口获取股票相关数据"""

    def __init__(self, timeout: int = 10, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        logger.info("EastmoneyDataProvider 初始化完成")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _symbol_to_secid(symbol: str) -> str:
        """将 600519.SH 转换为 "1.600519" 等 eastmoney `secid`"""
        symbol = symbol.upper()
        if symbol.endswith(".SH"):
            return f"1.{symbol[:-3]}"
        if symbol.endswith(".SZ"):
            return f"0.{symbol[:-3]}"
        # 默认为上海主板代码
        return f"1.{symbol}"

    @staticmethod
    def _to_yyyymmdd(date_str: str) -> str:
        """YYYY-MM-DD -> YYYYMMDD"""
        return date_str.replace("-", "")

    # ------------------------------------------------------------------
    # interface implementations
    # ------------------------------------------------------------------
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票历史日线数据 (前复权)"""
        try:
            secid = self._symbol_to_secid(symbol)
            beg = self._to_yyyymmdd(start_date)
            end = self._to_yyyymmdd(end_date)
            url = (
                "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
                f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&"
                "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73&"
                "ut=fa5fd1943c7b386f172d6893dbfba10b&lmt=10000&"
                "klt=101&fqt=1&"  # 101 -> 日线; fqt=1 前复权
                f"beg={beg}&end={end}"
            )
            resp = self._request("GET", url, headers={"Referer": "https://quote.eastmoney.com/"})
            if resp is None:
                return None
            json_data = resp.json()
            klines = json_data.get("data", {}).get("klines", [])
            if not klines:
                logger.warning("Eastmoney 无日线数据 %s", symbol)
                return None

            records = [line.split(",") for line in klines]
            columns = [
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "pct_chg",
                "price_change",
                "turnover_rate",
                "pe_ttm",
                "pb",
                "ps",
                "pcf",
                "market_cap",
                "flow_cap",
            ]
            df = pd.DataFrame(records, columns=columns[: len(records[0])])
            # 类型转换
            numeric_cols = df.columns.difference(["date"])
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df["symbol"] = symbol.upper()
            return df
        except Exception as e:
            logger.error("Eastmoney get_stock_data failed for %s: %s", symbol, e)
            return None

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """批量获取实时行情 (简单行情字段)"""
        result: Dict[str, Any] = {}
        try:
            secids = [self._symbol_to_secid(s) for s in symbols]
            url = (
                "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&invt=2&fields="
                "f12,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18,f10,f8,f9,f23&secids=" + ",".join(secids)
            )
            resp = self._request("GET", url, headers={"Referer": "https://quote.eastmoney.com/"})
            if resp is None:
                return result
            data_list = resp.json().get("data", {}).get("diff", [])
            for d in data_list:
                code = d.get("f12")
                result[code] = {
                    "symbol": code,
                    "name": d.get("f14"),
                    "price": d.get("f2"),
                    "pct_change": d.get("f3"),
                    "raw": d,
                }
        except Exception as e:
            logger.error("Eastmoney get_realtime_data failed: %s", e)
        return result

    # 其他接口简单实现/占位符
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """获取沪深 A 股列表"""
        try:
            url = (
                "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&fid=f3&"
                "fs=m:0+t:6,m:0+t:13,m:1+t:2&fields=f12,f14"
            )
            resp = self._request("GET", url, headers={"Referer": "https://quote.eastmoney.com/"})
            if resp is None:
                return None
            diff = resp.json().get("data", {}).get("diff", [])
            if not diff:
                return None
            df = pd.DataFrame([{"symbol": d["f12"], "name": d["f14"]} for d in diff])
            return df
        except Exception as e:
            logger.error("Eastmoney get_stock_list failed: %s", e)
            return None