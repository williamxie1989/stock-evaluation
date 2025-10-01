"""
Tushare 数据提供器
依赖 tushare 库 (pip install tushare)。需要设置环境变量 TUSHARE_TOKEN 或在初始化时传入 token。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import pandas as pd

try:
    import tushare as ts
except ImportError:  # type: ignore
    ts = None  # noqa: N816

from src.core.interfaces import DataProviderInterface
from src.data.providers.base_provider import DataProviderBase

logger = logging.getLogger(__name__)


class TushareDataProvider(DataProviderBase, DataProviderInterface):
    """通过 Tushare Pro 接口获取股票数据

    新增功能：
    1. 动态检测 token 的接口权限，缺失时降级并给出中文日志提示；
    2. 将检测结果缓存在类变量，避免重复查询造成 QPS 浪费；
    """

    # 缓存权限检测结果，避免重复 query
    _PERM_CACHE: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # 权限辅助
    # ------------------------------------------------------------------
    def _has_api_permission(self, api_name: str) -> bool:
        """检查当前 token 是否拥有指定接口权限

        Args:
            api_name: 例如 "daily"
        Returns:
            True 表示有权限；False 表示无权限或检测失败
        """
        if api_name in self._PERM_CACHE:
            return self._PERM_CACHE[api_name]
        if self.pro is None:
            self._PERM_CACHE[api_name] = False
            return False
        try:
            perm_df = self.pro.query("api_permission")  # 官方接口，返回当前 token 的接口白名单
            allowed = api_name in perm_df["api_name"].values  # type: ignore[index]
            self._PERM_CACHE[api_name] = allowed
            if not allowed:
                logger.warning("Tushare token 缺少接口权限: %s，相关功能将被降级", api_name)
            return allowed
        except Exception as e:
            logger.error("检查 Tushare 接口权限失败 %s: %s", api_name, e)
            # 如果无法检测权限，则默认放行，避免阻塞正常数据调用
            self._PERM_CACHE[api_name] = True
            logger.info("无法确认接口 %s 权限，默认视为有权限，后续如调用失败将按实际情况处理", api_name)
            return True

    # ------------------------------------------------------------------
    """通过 Tushare Pro 接口获取股票数据"""

    def __init__(self, token: Optional[str] = None, timeout: int = 10, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        if ts is None:
            logger.error("tushare 未安装, 请先 pip install tushare")
            self.pro = None
        else:
            token = token or os.getenv("TUSHARE_TOKEN")
            if not token:
                # 尝试动态加载 .env 文件
                try:
                    from dotenv import load_dotenv, find_dotenv
                    # 使用 find_dotenv 可从当前路径向上递归查找 .env，避免工作目录不同导致无法加载
                    env_path = find_dotenv()
                    if env_path:
                        load_dotenv(env_path, override=True)
                    else:
                        load_dotenv(override=True)
                    token = os.getenv("TUSHARE_TOKEN")
                except Exception as e:
                    logger.warning("加载 .env 文件以获取 TUSHARE_TOKEN 失败: %s", e)
                    token = None

            if not token:
                logger.warning("未检测到 TUSHARE_TOKEN, 某些接口可能无法调用")
                self.pro = ts.pro_api()
            else:
                self.pro = ts.pro_api(token)
            logger.info("TushareDataProvider 初始化完成")

    # --------------------------------------------
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取日线前复权数据 (后复权可自行修改)。symbol 需形如 600519.SH

        根据官方文档 https://tushare.pro/document/2?doc_id=15 ，接口 daily
        需要至少 level1 权限。
        """
        if self.pro is None:
            return None
        if not self._has_api_permission("daily"):
            return None
        try:
            ts_code = symbol.upper()
            params = {
                "ts_code": ts_code,
                "start_date": start_date.replace("-", ""),
                "end_date": end_date.replace("-", ""),
                # 可选参数复权类型： None/raw 代表不复权，这里固定取 qfq，可按需修改
                # 同时返回字段可限制，这里使用默认
            }
            df = self.pro.daily(**params)
            if df.empty:
                logger.warning("Tushare 无数据 %s", symbol)
                return None
            # tushare 返回 latest->oldest, 转换并重命名
            df.rename(columns={
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "vol": "volume",
                "trade_date": "date",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df["symbol"] = symbol.upper()
            df = df[["open", "close", "high", "low", "volume", "symbol"]]
            return df
        except Exception as e:
            logger.error("Tushare get_stock_data failed for %s: %s", symbol, e)
            return None

    # 其他接口简易占位
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        return {}

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        if self.pro is None:
            return {}
        try:
            df = self.pro.fina_indicator(ts_code=symbol.upper())
            return df.to_dict(orient="records") if not df.empty else {}
        except Exception as e:
            logger.error("Tushare get_financial_data failed %s", e)
            return {}

    def get_market_overview(self) -> Dict[str, Any]:
        return {}

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        if self.pro is None:
            return None
        try:
            df = self.pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date")
            return df
        except Exception as e:
            logger.error("Tushare get_stock_list failed: %s", e)
            return None

    # 新增：批量获取多支股票日线行情，支持列表或逗号分隔字符串
    def get_stock_data_multi(self, symbols: Union[List[str], str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """批量获取多支股票日线前复权数据。

        Args:
            symbols: 股票代码列表，形如 ["600519.SH", "000001.SZ"] 或 "600519.SH,000001.SZ"
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            合并后的DataFrame，按日期和symbol索引，包含 open/close/high/low/volume 字段。
        """
        if self.pro is None:
            return None
        if not self._has_api_permission("daily"):
            return None
        try:
            # 处理输入格式
            if isinstance(symbols, str):
                symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
            else:
                symbol_list = symbols

            ts_code_param = ",".join([s.upper() for s in symbol_list])
            params = {
                "ts_code": ts_code_param,
                "start_date": start_date.replace("-", ""),
                "end_date": end_date.replace("-", ""),
            }
            df = self.pro.daily(**params)
            if df.empty:
                logger.warning("Tushare 无数据 多股票 %s", ts_code_param)
                return None
            # 统一列名并处理
            df.rename(columns={
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "vol": "volume",
                "trade_date": "date",
                "ts_code": "symbol",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = df["symbol"].str.upper()
            df.set_index(["date", "symbol"], inplace=True)
            df.sort_index(inplace=True)
            return df[["open", "close", "high", "low", "volume"]]
        except Exception as e:
            logger.error("Tushare get_stock_data_multi failed: %s", e)
            return None

    # 新增：按交易日获取全市场或指定股票行情
    def get_stock_data_by_date(self, trade_date: str, symbols: Optional[Union[List[str], str]] = None) -> Optional[pd.DataFrame]:
        """按单个交易日获取行情。

        Args:
            trade_date: 交易日 YYYY-MM-DD
            symbols: 可选，股票代码列表或逗号分隔字符串。若为空则返回全市场。
        """
        if self.pro is None:
            return None
        if not self._has_api_permission("daily"):
            return None
        try:
            params: Dict[str, Any] = {
                "trade_date": trade_date.replace("-", ""),
            }
            if symbols:
                if isinstance(symbols, str):
                    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
                else:
                    symbol_list = symbols
                params["ts_code"] = ",".join([s.upper() for s in symbol_list])
            df = self.pro.daily(**params)
            if df.empty:
                logger.warning("Tushare 无数据 trade_date=%s symbols=%s", trade_date, symbols)
                return None
            df.rename(columns={
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "vol": "volume",
                "trade_date": "date",
                "ts_code": "symbol",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = df["symbol"].str.upper()
            df.set_index(["date", "symbol"], inplace=True)
            df.sort_index(inplace=True)
            return df[["open", "close", "high", "low", "volume"]]
        except Exception as e:
            logger.error("Tushare get_stock_data_by_date failed: %s", e)
            return None

    def get_market_data_by_date_range(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[Union[List[str], str]] = None,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """按日期范围批量获取行情数据。

        通过循环 ``get_stock_data_by_date`` 实现，内部带简单重试与指数退避。

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            symbols: 可选，股票代码列表或逗号分隔字符串。若为空则返回全市场。
            max_retries: 单个交易日调用失败时的最大重试次数。

        Returns:
            合并后的 ``DataFrame``，多重索引 [date, symbol]，如全部失败则返回 ``None``。
        """
        if self.pro is None:
            return None
        if not self._has_api_permission("daily"):
            return None

        try:
            # 生成日期序列，先假设工作日频率，后续可基于 trade_cal 优化
            date_range = pd.date_range(start=start_date, end=end_date, freq="B")
            all_frames: List[pd.DataFrame] = []
            for dt in date_range:
                trade_date = dt.strftime("%Y-%m-%d")
                attempt = 0
                while attempt <= max_retries:
                    df = self.get_stock_data_by_date(trade_date, symbols)
                    if df is not None:
                        all_frames.append(df)
                        break
                    attempt += 1
                    if attempt > max_retries:
                        logger.error("获取 %s 行情失败，已达最大重试次数", trade_date)
                        break
                    sleep_s = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning("%s 第 %s/%s 次重试，%.2fs 后继续", trade_date, attempt, max_retries, sleep_s)
                    time.sleep(sleep_s)
            if not all_frames:
                return None
            combined = pd.concat(all_frames)
            combined.sort_index(inplace=True)
            return combined
        except Exception as e:
            logger.error("get_market_data_by_date_range error: %s", e)
            return None