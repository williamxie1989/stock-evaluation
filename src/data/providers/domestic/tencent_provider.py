"""
Tencent 数据提供器 (无token接口)
"""
import logging
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.interfaces import DataProviderInterface
from src.data.providers.base_provider import DataProviderBase

logger = logging.getLogger(__name__)


# 继承 DataProviderBase 以获得速率限制/重试等通用能力
class TencentDataProvider(DataProviderBase, DataProviderInterface):
    """通过腾讯证券接口获取股票数据"""

    def __init__(self, timeout: int = 8, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        logger.info("TencentDataProvider 初始化完成")

    @staticmethod
    def _symbol_to_tencent_code(symbol: str) -> str:
        symbol = symbol.upper()
        if symbol.endswith(".SH"):
            return f"sh{symbol[:-3]}"
        if symbol.endswith(".SZ"):
            return f"sz{symbol[:-3]}"
        return symbol

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            code = self._symbol_to_tencent_code(symbol)
            # k_charts接口，日k=101
            url = (
                "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?param="
                f"{code},day,{start_date},{end_date},640,qfq"
            )
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            json_data = resp.json()

            # 兼容API返回的多种格式，可能data字段是dict或list
            data_section = json_data.get("data") if isinstance(json_data, dict) else None
            klines = []
            if isinstance(data_section, dict):
                klines = data_section.get(code, {}).get("qfqday", [])
            elif isinstance(data_section, list):
                # 在列表中查找包含目标代码的字典
                for item in data_section:
                    if isinstance(item, dict) and code in item:
                        klines = item[code].get("qfqday", [])
                        break
            else:
                logger.warning(f"Unexpected data format from Tencent for {symbol}: {type(data_section)}")

            if not klines:
                logger.warning(f"Tencent no kline data for {symbol}")
                return None
            records = []
            for x in klines:
                try:
                    date_str = x[0]
                    open_p, close_p, high_p, low_p = [float(v) for v in x[1:5]]
                    volume_raw = x[6] if len(x) > 6 else 0
                    if isinstance(volume_raw, dict):
                        volume_val = volume_raw.get("volume") or volume_raw.get("vol") or 0
                    else:
                        volume_val = volume_raw
                    try:
                        volume = int(float(volume_val))
                    except (ValueError, TypeError):
                        volume = 0
                    records.append([date_str, open_p, close_p, high_p, low_p, volume])
                except Exception as inner_e:
                    logger.debug("Skip malformed kline record %s: %s", x, inner_e)

            if not records:
                logger.warning(f"Tencent parsed no records for {symbol}")
                return None
            df = pd.DataFrame(
                records,
                columns=[
                    "date",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                ],
            )
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df["symbol"] = symbol.upper()
            return df
        except Exception as e:
            logger.error(f"Tencent get_stock_data failed for {symbol}: {e}")
            return None

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            codes = [self._symbol_to_tencent_code(s) for s in symbols]
            url = "https://qt.gtimg.cn/q=" + ",".join(codes)
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.text
            for line in text.strip().split("\n"):
                if not line:
                    continue
                # v_sz000001="51~平安银行~000001~10.78~10.80~10.79~273867~289790";
                parts = line.split("=", 1)
                data_str = parts[1].strip(";").strip("\"")
                fields = data_str.split("~")
                if len(fields) < 6:
                    continue
                code = fields[2]
                name = fields[1]
                price = float(fields[3])
                yesterday_close = float(fields[4])
                open_p = float(fields[5])
                volume = int(fields[6])
                result[code] = {
                    "symbol": code,
                    "name": name,
                    "price": price,
                    "yesterday_close": yesterday_close,
                    "open": open_p,
                    "volume": volume,
                }
        except Exception as e:
            logger.error(f"Tencent get_realtime_data failed: {e}")
        return result

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        return None