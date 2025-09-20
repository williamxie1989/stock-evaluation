import os
import re
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import akshare as ak
import pandas as pd
import numpy as np
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

from db import DatabaseManager
from stock_status_filter import StockStatusFilter
from akshare_data_provider import AkshareDataProvider


class EnhancedDataProvider:
    """
    增强版数据提供者，支持多数据源互补获取历史价格数据
    支持的数据源：东财、新浪、腾讯、网易、雪球、同花顺（TDX）、AkShare默认
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 重试参数可由环境变量覆盖：ENH_MAX_RETRIES / ENH_RETRY_DELAY（秒）
        try:
            self.max_retries = max(1, int(os.getenv("ENH_MAX_RETRIES", "3")))
        except Exception:
            self.max_retries = 3
        try:
            self.retry_delay = max(0.1, float(os.getenv("ENH_RETRY_DELAY", "1.5")))
        except Exception:
            self.retry_delay = 1.5
        try:
            self.logger.info(
                f"EnhancedDataProvider retries configured: max_retries={self.max_retries}, retry_delay={self.retry_delay}s"
            )
        except Exception:
            pass

        # 默认仅启用已实现且较稳定的数据源，减少占位接口导致的失败与额外压力
        self.data_sources = [
            {"name": "eastmoney", "func": self._get_data_from_eastmoney},
            {"name": "sina", "func": self._get_data_from_sina},
            {"name": "akshare_default", "func": self._get_data_from_akshare_default},
        ]
        # 可选启用实验性数据源（占位/未完全实现），通过环境变量控制
        try:
            enable_exp = str(os.getenv("ENH_ENABLE_EXPERIMENTAL", "0")).lower() in {"1", "true", "yes"}
        except Exception:
            enable_exp = False
        if enable_exp:
            self.data_sources.extend([
                {"name": "tencent", "func": self._get_data_from_tencent},
                {"name": "xueqiu", "func": self._get_data_from_xueqiu},
                {"name": "netease", "func": self._get_data_from_netease},
                {"name": "tdx_api", "func": self._get_data_from_tdx_api},
            ])
            self.logger.info("ENH_ENABLE_EXPERIMENTAL=1: 已启用实验性数据源 [tencent, xueqiu, netease, tdx_api]")

        # 记录最近一次成功获取数据所使用的数据源
        self.last_used_source: Optional[str] = None
        # 记录最近一轮各数据源尝试的诊断信息
        self.last_attempts: List[Dict[str, Any]] = []
        # 新增：用于状态过滤的DB与过滤器
        try:
            self.db_manager = DatabaseManager()
        except Exception:
            self.db_manager = None
        try:
            self.stock_filter = StockStatusFilter()
        except Exception:
            self.stock_filter = None
        
        # 初始化基础数据提供者，用于委托某些方法
        self.base_provider = AkshareDataProvider()
        try:
            self.logger.info(f"已设置数据源优先级: {[s['name'] for s in self.data_sources]}")
        except Exception:
            pass
        
    def get_all_stock_list(self):
        """
        委托给基础数据提供者获取全市场股票列表
        """
        return self.base_provider.get_all_stock_list()
    
    def get_ah_spot(self):
        """
        委托给基础数据提供者获取A+H股实时行情
        """
        return self.base_provider.get_ah_spot()

    def set_preferred_sources(self, sources: List[str]) -> None:
        """根据传入的名称顺序调整数据源优先级，支持常用别名映射"""
        if not sources:
            return
        # 别名映射：对外常称的“akshare”映射为内部名称“akshare_default”
        alias_map = {"akshare": "akshare_default"}
        normalized = [alias_map.get(s, s) for s in sources]

        name_to_source = {s["name"]: s for s in self.data_sources}
        new_order = []
        for name in normalized:
            if name in name_to_source and name_to_source[name] not in new_order:
                new_order.append(name_to_source[name])
        # 追加未提及的数据源
        for s in self.data_sources:
            if s not in new_order:
                new_order.append(s)
        self.data_sources = new_order
        self.logger.info(f"已重排数据源优先级: {[s['name'] for s in self.data_sources]}")

    def get_stock_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        多数据源获取股票历史数据
        """
        ak_symbol = self._convert_symbol_format(symbol)
        # 过滤无效代码：None.*、空串、000000、非6位数字
        if (
            ak_symbol is None
            or str(ak_symbol).strip() == ""
            or str(ak_symbol).upper().startswith("NONE")
            or str(ak_symbol) == "000000"
            or not re.fullmatch(r"\d{6}", str(ak_symbol))
        ):
            self.logger.warning(f"无效股票代码，跳过: original={symbol}, normalized={ak_symbol}")
            return None

        # 增量同步专用前置过滤：排除688/689/900开头及停牌、退市
        try:
            is_incremental = isinstance(period, str) and period.lower().endswith("d")
        except Exception:
            is_incremental = False
        if is_incremental:
            # 代码前缀过滤
            if str(ak_symbol).startswith(("688", "689", "900")):
                self.last_used_source = None
                self.last_attempts = [{
                    "source": "precheck",
                    "status": "filtered",
                    "reason": "code_prefix_excluded",
                    "symbol": str(ak_symbol)
                }]
                try:
                    self.logger.info(f"增量同步过滤（前缀）: {symbol} -> {ak_symbol}")
                except Exception:
                    pass
                return None
            # 停牌/退市过滤（使用名称与状态判定）
            try:
                stock_name = None
                if self.db_manager is not None:
                    with self.db_manager.get_conn() as conn:
                        dfn = pd.read_sql_query(
                            "SELECT name FROM stocks WHERE symbol = ? LIMIT 1",
                            conn,
                            params=[symbol]
                        )
                        if not dfn.empty:
                            stock_name = dfn.iloc[0]["name"]
            except Exception:
                stock_name = None
            try:
                if self.stock_filter is not None:
                    check = self.stock_filter.should_filter_stock(
                        stock_name or "",
                        symbol,
                        include_st=False,
                        include_suspended=True,
                        db_manager=self.db_manager,
                        exclude_star_market=False,
                        last_n_days=30,
                        include_no_trades_last_n_days=False,
                    )
                    if check.get("should_filter"):
                        reason = check.get("reason") or "filtered"
                        self.last_used_source = None
                        self.last_attempts = [{
                            "source": "precheck",
                            "status": "filtered",
                            "reason": reason,
                            "symbol": symbol
                        }]
                        try:
                            self.logger.info(f"增量同步过滤（状态）: {symbol}, reason={reason}")
                        except Exception:
                            pass
                        return None
            except Exception:
                # 过滤过程异常不影响后续数据拉取
                pass

        # 重置最近一次使用来源与诊断信息
        self.last_used_source = None
        self.last_attempts = []

        for s in self.data_sources:
            name, func = s["name"], s["func"]
            attempt_info: Dict[str, Any] = {"source": name, "start": datetime.now().isoformat()}
            try:
                data = func(ak_symbol, period)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # 标准化列名与类型，确保包含[date, open, high, low, close, volume]
                    std = self._standardize_hist_df(data)
                    if isinstance(std, pd.DataFrame) and not std.empty and all(
                        c in std.columns for c in ["date", "open", "high", "low", "close", "volume"]
                    ):
                        attempt_info.update({"success": True, "rows": len(std), "status": "success"})
                        self.last_attempts.append(attempt_info)
                        self.last_used_source = name
                        self.logger.info(f"{name} 成功获取 {ak_symbol} 历史数据, 行数={len(std)}")
                        return std
                    else:
                        attempt_info.update({
                            "success": False,
                            "error": "standardization failed or missing required columns",
                            "status": "empty"
                        })
                        self.last_attempts.append(attempt_info)
                else:
                    attempt_info.update({"success": False, "error": "empty result", "status": "empty"})
                    self.last_attempts.append(attempt_info)
            except Exception as e:
                attempt_info.update({"success": False, "error": str(e), "status": "exception"})
                self.last_attempts.append(attempt_info)
                self.logger.debug(f"{name} 获取 {ak_symbol} 数据失败: {e}")
            # 为下一数据源尝试加入轻微抖动，减少短时间内过多请求
            time.sleep(random.uniform(0.15, 0.35))

        self.logger.error(f"所有数据源都无法获取 {symbol} 的历史数据")
        return None

    def _retry_with_backoff(self, func, *args, **kwargs):
        """带指数退避和抖动的重试封装"""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                retryable = any(
                    k in msg
                    for k in [
                        "rate limit",
                        "too many requests",
                        "timeout",
                        "timed out",
                        "connection",
                        "reset",
                        "proxy",
                        "temporarily unavailable",
                    ]
                )
                if retryable and attempt < self.max_retries - 1:
                    base = self.retry_delay * (2 ** attempt)
                    delay = base * (0.8 + 0.4 * random.random())
                    self.logger.warning(
                        f"数据源调用失败(尝试{attempt + 1}/{self.max_retries}): {e}, {delay:.2f}s后重试"
                    )
                    time.sleep(delay)
                    continue
                raise e
        if last_err:
            raise last_err
        return None

    def _get_data_from_eastmoney(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从东财获取数据"""
        try:
            data = self._retry_with_backoff(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=self._get_start_date(period),
                adjust="qfq",
            )
            return data
        except Exception as e:
            self.logger.debug(f"东财接口失败: {e}")
            return None

    def _get_data_from_sina(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从新浪获取数据"""
        try:
            data = self._retry_with_backoff(
                ak.stock_zh_a_daily,
                symbol=f"sh{symbol}" if str(symbol).startswith("6") else f"sz{symbol}",
                start_date=self._get_start_date(period),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
            return data
        except Exception as e:
            self.logger.debug(f"新浪接口失败: {e}")
            return None

    def _get_data_from_tencent(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从腾讯获取数据（暂为占位，需替换为合适的历史行情接口）"""
        try:
            market_code = f"sh{symbol}" if str(symbol).startswith("6") else f"sz{symbol}"
            _ = self._retry_with_backoff(ak.stock_individual_info_em, symbol=market_code)
            return None
        except Exception as e:
            self.logger.debug(f"腾讯接口失败: {e}")
            return None

    def _get_data_from_netease(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从网易获取数据（占位）"""
        try:
            return None
        except Exception as e:
            self.logger.debug(f"网易接口失败: {e}")
            return None

    def _get_data_from_xueqiu(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从雪球获取数据（占位，雪球需要认证与Cookie）"""
        try:
            return None
        except Exception as e:
            self.logger.debug(f"雪球接口失败: {e}")
            return None

    def _get_data_from_akshare_default(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """AkShare默认接口作为最后兜底"""
        try:
            data = self._retry_with_backoff(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=self._get_start_date(period),
                adjust="qfq",
            )
            return data
        except Exception as e:
            self.logger.debug(f"AkShare默认接口失败: {e}")
            return None

    def _get_data_from_tdx_api(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """通过通达信(TDX)获取数据（占位，未实现完整历史）"""
        try:
            return None
        except Exception as e:
            self.logger.debug(f"TDX接口失败: {e}")
            return None

    def _convert_symbol_format(self, symbol: Any) -> Optional[str]:
        """将可能的多种股票代码格式规范化为6位数字字符串"""
        if symbol is None:
            return None
        s = str(symbol).strip().upper()
        # 去掉常见后缀与前缀
        s = s.replace(".SH", "").replace(".SZ", "").replace("SH", "").replace("SZ", "")
        # 提取6位数字
        m = re.search(r"(\d{6})", s)
        return m.group(1) if m else None

    def _get_start_date(self, period: str) -> str:
        """根据period计算开始日期，返回YYYYMMDD"""
        now = datetime.now()
        p = (period or "").lower()
        if p in {"1y", "12mo"}:
            dt = now - timedelta(days=365)
        elif p in {"6mo", "180d"}:
            dt = now - timedelta(days=180)
        elif p in {"2y"}:
            dt = now - timedelta(days=365 * 2)
        elif p in {"3y"}:
            dt = now - timedelta(days=365 * 3)
        elif p in {"5y"}:
            dt = now - timedelta(days=365 * 5)
        else:
            dt = now - timedelta(days=365)
        return dt.strftime("%Y%m%d")

    def _standardize_hist_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """将不同数据源的历史数据标准化为列: [date, open, high, low, close, volume(, amount)]
        - 统一日期到 pandas datetime
        - 只保留需要的列，丢弃无法解析的行
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df_local = df.copy()

        # 若日期在索引，尝试提升为列
        try:
            if df_local.index.name in ("date", "Date", "日期", "交易日期"):
                df_local = df_local.reset_index()
        except Exception:
            pass

        col_map_candidates = [
            {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"},
            {"交易日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low", "收盘价": "close", "成交量": "volume", "成交额": "amount"},
            {"交易日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"},
            {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Amount": "amount"},
            {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "amount": "amount"},
        ]

        chosen = None
        for cmap in col_map_candidates:
            inter = set(cmap.keys()) & set(df_local.columns)
            if len(inter) >= 5:  # 至少能覆盖必需列
                chosen = cmap
                break

        if chosen is not None:
            df_local = df_local.rename(columns=chosen)
        else:
            # 回退：统一小写，尽力匹配
            lower_map = {c: str(c).lower() for c in df_local.columns}
            df_local = df_local.rename(columns=lower_map)

        keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df_local.columns]
        df_local = df_local[keep_cols]

        # 必需列校验
        required = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df_local.columns for c in required):
            return pd.DataFrame(columns=required)

        # 类型转换
        df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
        df_local = df_local[pd.notna(df_local["date"])]
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df_local.columns:
                df_local[c] = pd.to_numeric(df_local[c], errors="coerce")

        return df_local.sort_values("date").reset_index(drop=True)