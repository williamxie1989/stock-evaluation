"""
Eastmoney 数据提供器
使用东财无token接口获取股票历史数据和实时数据
"""
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.core.interfaces import DataProviderInterface

logger = logging.getLogger(__name__)


class EastmoneyDataProvider(DataProviderInterface):
    """通过东财接口获取股票数据"""

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.timeout = timeout
        logger.info("EastmoneyDataProvider 初始化完成")

    # ----------------------- helpers ----------------------- #
    @staticmethod
    def _symbol_to_secid(symbol: str) -> str:
        """将 600519.SH 转换为 eastmoney secid 1.600519"""
        symbol = symbol.upper()
        if symbol.endswith(".SH"):
            return f"1.{symbol[:-3]}"
        if symbol.endswith(".SZ"):
            return f"0.{symbol[:-3]}"
        # 默认尝试上海
        return f"1.{symbol}"

    @staticmethod
    def _to_yyyymmdd(date_str: str) -> str:
        """YYYY-MM-DD -> YYYYMMDD"""
        return date_str.replace("-", "")

    # ----------------------- interface methods ----------------------- #
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            secid = self._symbol_to_secid(symbol)
            beg = self._to_yyyymmdd(start_date)
            end = self._to_yyyymmdd(end_date)

            url = (
                "http://push2his.eastmoney.com/api/qt/stock/kline/get?"
                f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&"
                "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73&"
                "klt=101&fqt=1&"
                f"beg={beg}&end={end}"
            )
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            json_data = resp.json()
            klines = json_data.get("data", {}).get("klines", [])
            if not klines:
                logger.warning(f"Eastmoney no kline data for {symbol}")
                return None

            # split
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
            df['symbol'] = symbol.upper()
            return df
        except Exception as e:
            logger.error(f"Eastmoney get_stock_data failed for {symbol}: {e}")
            return None

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        # 使用东财quote接口批量获取
        result: Dict[str, Any] = {}
        try:
            # 组装 secids
            secids = [self._symbol_to_secid(s) for s in symbols]
            url = (
                "http://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&invt=2&fields=f12,f14,f2,f3,f4,f5,f6,f7,f15,f16,f17,f18,f10,f8,f9,f23&secids="
                + ",".join(secids)
            )
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data_list = resp.json().get("data", {}).get("diff", [])
            for d in data_list:
                code = d.get("f12")
                name = d.get("f14")
                price = d.get("f2")
                change_pct = d.get("f3")
                result[code] = {
                    "symbol": code,
                    "name": name,
                    "price": price,
                    "pct_change": change_pct,
                    "raw": d,
                }
        except Exception as e:
            logger.error(f"Eastmoney get_realtime_data failed: {e}")
        return result

    # 其他接口简单实现/占位符
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """使用另一接口获取沪深A股列表"""
        try:
            url = (
                "http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&fid=f3&fs=m:0+t:6,m:0+t:13,m:1+t:2&"
                "fields=f12,f14"
            )
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            diff = resp.json().get("data", {}).get("diff", [])
            if not diff:
                return None
            records = [{"symbol": d["f12"], "name": d["f14"]} for d in diff]
            df = pd.DataFrame(records)
            return df
        except Exception as e:
            logger.error(f"Eastmoney get_stock_list failed: {e}")
            return None