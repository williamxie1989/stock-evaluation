"""TDX .day 文件解析器
将通达信日线二进制文件解析为 pandas DataFrame，字段格式满足数据库写入要求。
"""
from __future__ import annotations

import os
import struct
import logging
from datetime import datetime
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

_RECORD_SIZE = 32  # 每条记录字节数
_STRUCT_FMT = "<IIIIIfII"  # 对应字段格式：date, open, high, low, close, amount(float), vol, reserved
_STRUCT = struct.Struct(_STRUCT_FMT)
_EXPECTED_SIZE = _STRUCT.size
assert _EXPECTED_SIZE == _RECORD_SIZE, "TDX日线记录大小不匹配"


def _parse_one_record(buf: bytes):
    date_i, open_i, high_i, low_i, close_i, amount_f, vol_i, _ = _STRUCT.unpack(buf)
    # 日期解析
    date_str = str(date_i)
    if len(date_str) != 8:
        # 无效日期
        return None
    try:
        date_dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None

    # 按通达信规则，价格放大100倍，成交量为股，通常需除以100，金额单位为元/10?
    open_p = open_i / 100.0
    high_p = high_i / 100.0
    low_p = low_i / 100.0
    close_p = close_i / 100.0
    # 成交量(股) -> 手
    volume = vol_i  # 保留原始股数，数据库后续可转换
    amount = amount_f  # 金额保留原始
    return date_dt, open_p, high_p, low_p, close_p, volume, amount


def read_day_file(path: str, symbol: str) -> pd.DataFrame:
    """读取单个 .day 文件，返回带symbol列的DataFrame。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    file_size = os.path.getsize(path)
    if file_size % _RECORD_SIZE != 0:
        logger.warning("文件大小不是32字节的整数倍: %s", path)

    records: List[dict] = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(_RECORD_SIZE)
            if not buf or len(buf) < _RECORD_SIZE:
                break
            parsed = _parse_one_record(buf)
            if parsed is None:
                continue
            date_dt, open_p, high_p, low_p, close_p, volume, amount = parsed
            records.append(
                {
                    "symbol": symbol,
                    "date": date_dt.strftime("%Y-%m-%d"),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume,
                    "amount": amount,
                }
            )
    if not records:
        logger.info("文件无有效记录: %s", path)
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df