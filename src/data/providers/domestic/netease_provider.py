"""
Netease 数据提供器 (CSV下载)
"""
import logging
import pandas as pd
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.interfaces import DataProviderInterface
from src.data.providers.base_provider import DataProviderBase

logger = logging.getLogger(__name__)


# 继承 DataProviderBase，获得重试/代理等
class NeteaseDataProvider(DataProviderBase, DataProviderInterface):
    """通过网易财经 CSV 接口获取数据"""

    def __init__(self, timeout: int = 10, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        logger.info("NeteaseDataProvider 初始化完成")

    @staticmethod
    def _symbol_to_netease_code(symbol: str) -> str:
        symbol = symbol.upper()
        if symbol.endswith(".SH"):
            return f"0{symbol[:-3]}"
        if symbol.endswith(".SZ"):
            return f"1{symbol[:-3]}"
        return symbol

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            code = self._symbol_to_netease_code(symbol)
            url = (
                "http://quotes.money.163.com/service/chddata.html?"
                f"code={code}&start={start_date.replace('-', '')}&end={end_date.replace('-', '')}&fields=TCLOSE;TOPEN;HIGH;LOW;VOTURNOVER"
            )
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            # 注意编码 GBK
            resp.encoding = "gbk"
            import io
            df = pd.read_csv(io.StringIO(resp.text))
            # 股票交易日期在第一列
            df.rename(columns={
                "日期": "date",
                "收盘价": "close",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "成交量": "volume",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            # 逆序（网易最新在上）
            df = df.sort_index()
            df["symbol"] = symbol.upper()
            return df[["open", "close", "high", "low", "volume", "symbol"]]
        except Exception as e:
            logger.error(f"Netease get_stock_data failed for {symbol}: {e}")
            return None

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        return {}

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        return None