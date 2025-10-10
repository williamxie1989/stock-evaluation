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
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, adjust: str = "") -> Optional[pd.DataFrame]:
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

    def get_stock_data_with_adjust(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        一次性获取不复权、前复权、后复权三种模式的日线数据，并合并到一个DataFrame：
        包含列：open, high, low, close（不复权）以及 open_qfq/high_qfq/low_qfq/close_qfq、open_hfq/high_hfq/low_hfq/close_hfq。
        注：新浪API本身不直接支持复权数据，此方法通过akshare库获取复权数据。
        """
        try:
            # 由于新浪API不直接支持复权数据，我们尝试使用akshare库
            # 先检查akshare是否已安装
            import importlib
            akshare_spec = importlib.util.find_spec("akshare")
            if akshare_spec is None:
                logger.warning("akshare库未安装，无法获取复权数据")
                # 返回不复权数据作为备选
                raw_data = self.get_stock_data(symbol, start_date, end_date)
                if raw_data is not None:
                    raw_data = raw_data.reset_index()
                    # 添加空的复权列
                    for suffix in ['qfq', 'hfq']:
                        for col in ['open', 'high', 'low', 'close']:
                            raw_data[f'{col}_{suffix}'] = None
                    # 添加amount列（如果不存在）
                    if 'amount' not in raw_data.columns:
                        raw_data['amount'] = None
                    # 确保所有必要的列存在
                    keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                                'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                                'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'symbol']
                    for col in keep_cols:
                        if col not in raw_data.columns:
                            raw_data[col] = None
                    return raw_data[keep_cols].sort_values('date').drop_duplicates(subset=['date'])
                return None

            # 如果akshare已安装，则使用它获取复权数据
            import akshare as ak
            logger.info(f"使用akshare获取{symbol}的复权数据")

            # 转换股票代码格式，akshare只需要纯数字
            clean_symbol = symbol.replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')
            
            # 支持 datetime 类型输入
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime('%Y%m%d')
            else:
                # 转换日期格式，akshare需要YYYYMMDD格式
                start_date_str = start_date.replace('-', '') if '-' in start_date else start_date
            
            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime('%Y%m%d')
            else:
                # 转换日期格式，akshare需要YYYYMMDD格式
                end_date_str = end_date.replace('-', '') if '-' in end_date else end_date

            # 获取三种复权数据，添加错误重试机制
            raw, qfq, hfq = None, None, None
            max_retries = 2
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    raw = ak.stock_zh_a_hist(symbol=clean_symbol, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="")
                    qfq = ak.stock_zh_a_hist(symbol=clean_symbol, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
                    hfq = ak.stock_zh_a_hist(symbol=clean_symbol, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="hfq")
                    break  # 如果成功获取，跳出循环
                except Exception as retry_e:
                    retry_count += 1
                    logger.warning(f"第{retry_count}次尝试获取akshare数据失败: {retry_e}")
                    if retry_count >= max_retries:
                        logger.error(f"达到最大重试次数，无法获取akshare数据")
                        # 如果获取失败，返回带空复权列的基础数据
                        raw_data = self.get_stock_data(symbol, start_date, end_date)
                        if raw_data is not None:
                            raw_data = raw_data.reset_index()
                            # 添加空的复权列
                            for suffix in ['qfq', 'hfq']:
                                for col in ['open', 'high', 'low', 'close']:
                                    raw_data[f'{col}_{suffix}'] = None
                            # 添加amount列（如果不存在）
                            if 'amount' not in raw_data.columns:
                                raw_data['amount'] = None
                            # 确保所有必要的列存在
                            keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                                        'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                                        'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'symbol']
                            for col in keep_cols:
                                if col not in raw_data.columns:
                                    raw_data[col] = None
                            return raw_data[keep_cols].sort_values('date').drop_duplicates(subset=['date'])
                        return None

            def _basic(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                if df is None or df.empty:
                    return None
                out = df.copy()
                # 将中文列名转换为英文标准列名
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                # 只映射存在的列
                existing_mapping = {k: v for k, v in column_mapping.items() if k in out.columns}
                if existing_mapping:
                    out = out.rename(columns=existing_mapping)
                
                # 确保日期列格式正确
                if 'date' in out.columns:
                    out['date'] = pd.to_datetime(out['date'])
                
                keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'] if c in out.columns]
                return out[keep]

            raw_b = _basic(raw)
            qfq_b = _basic(qfq)
            hfq_b = _basic(hfq)

            # 合并时加suffixes，避免重复列
            result = None
            if raw_b is not None:
                result = raw_b.sort_values('date').reset_index(drop=True)
            elif qfq_b is not None:
                result = qfq_b.sort_values('date').reset_index(drop=True)
            elif hfq_b is not None:
                result = hfq_b.sort_values('date').reset_index(drop=True)
            else:
                return None

            # 合并前复权
            if qfq_b is not None:
                qfq_part = qfq_b.rename(columns={
                    'open': 'open_qfq', 'high': 'high_qfq', 'low': 'low_qfq', 'close': 'close_qfq'
                })[['date', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq']]
                result = pd.merge(result, qfq_part, on='date', how='outer', suffixes=(None, '_qfq'))

            # 合并后复权
            if hfq_b is not None:
                hfq_part = hfq_b.rename(columns={
                    'open': 'open_hfq', 'high': 'high_hfq', 'low': 'low_hfq', 'close': 'close_hfq'
                })[['date', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq']]
                result = pd.merge(result, hfq_part, on='date', how='outer', suffixes=(None, '_hfq'))

            # 只保留目标列
            keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                        'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                        'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq']
            for col in keep_cols:
                if col not in result.columns:
                    result[col] = None
            result = result[keep_cols]
            result['symbol'] = symbol
            result = result.sort_values('date').drop_duplicates(subset=['date'])
            return result
        except Exception as e:
            logger.error(f"获取复权数据失败: {symbol} {e}")
            # 如果akshare方式失败，返回不复权数据作为备选
            raw_data = self.get_stock_data(symbol, start_date, end_date)
            if raw_data is not None:
                raw_data = raw_data.reset_index()
                # 添加空的复权列
                for suffix in ['qfq', 'hfq']:
                    for col in ['open', 'high', 'low', 'close']:
                        raw_data[f'{col}_{suffix}'] = None
                # 添加amount列（如果不存在）
                if 'amount' not in raw_data.columns:
                    raw_data['amount'] = None
                # 确保所有必要的列存在
                keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                            'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                            'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'symbol']
                for col in keep_cols:
                    if col not in raw_data.columns:
                        raw_data[col] = None
                return raw_data[keep_cols].sort_values('date').drop_duplicates(subset=['date'])
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