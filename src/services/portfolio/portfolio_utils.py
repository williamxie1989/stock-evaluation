# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import date, datetime, timedelta
import pandas as pd

TRADING_CALENDAR_SAFETY_LIMIT = 365

def is_trading_day(data_access, as_of: date, sample_symbol: str | None = None) -> bool:
    sym = sample_symbol or "000001.SZ"
    try:
        df = data_access.get_stock_data(sym, start_date=as_of.strftime("%Y-%m-%d"),
                                        end_date=as_of.strftime("%Y-%m-%d"), adjust_mode="qfq")
    except Exception:
        df = None
    return df is not None and getattr(df, "empty", True) is False

def next_trading_day(data_access, ref: date, sample_symbol: str | None = None) -> date:
    d = ref
    for _ in range(TRADING_CALENDAR_SAFETY_LIMIT):
        d = d + timedelta(days=1)
        if is_trading_day(data_access, d, sample_symbol):
            return d
    raise RuntimeError("无法找到后续交易日，请检查交易所日历或数据源。")

def session_price_qfq(pipeline, symbol: str, trade_day: date, session: str = "open", asof: bool = True) -> float:
    """
    获取指定交易日的开盘/收盘（前复权）价格
    
    Args:
        pipeline: 数据管道对象
        symbol: 股票代码
        trade_day: 交易日
        session: 交易时段，'open'或'close'
        asof: 是否启用as-of填充，当当日无价格时取最近价格
    
    Returns:
        前复权价格
    """
    # 获取最近几天的历史数据，支持as-of填充
    days_to_fetch = 10 if asof else 2  # asof模式下获取更多数据用于回溯
    df = pipeline._fetch_history(symbol, end_date=pd.Timestamp(trade_day), days=days_to_fetch)
    
    if df is None or df.empty:
        if asof:
            # 尝试获取更长时间范围的数据
            df = pipeline._fetch_history(symbol, end_date=pd.Timestamp(trade_day), days=30)
            if df is None or df.empty:
                raise RuntimeError(f"{symbol} 在 {trade_day} 及之前30天内均无行情")
        else:
            raise RuntimeError(f"{symbol} 在 {trade_day} 无行情")
    
    df = df.sort_index()
    
    # 检查当日是否有数据
    if str(trade_day) in df.index.astype(str):
        row = df.loc[str(trade_day)]
        col = "open" if session.lower() == "open" else "close"
        return float(row[col])
    
    # 当日无数据，启用as-of填充
    if asof:
        # 找到最近的有数据的交易日
        available_dates = df.index[df.index <= pd.Timestamp(trade_day)]
        if len(available_dates) > 0:
            latest_date = available_dates[-1]
            row = df.loc[latest_date]
            col = "close"  # as-of填充使用收盘价
            return float(row[col])
    
    # 无as-of填充或找不到可用数据
    raise RuntimeError(f"{symbol} 在 {trade_day} 无K线，且as-of填充失败")


def get_execution_price(pipeline, symbol: str, signal_date: date, exec_mode: str = "passive", 
                       price_type: str = "close", asof: bool = True) -> tuple[float, date]:
    """
    根据成交规则获取执行价格和执行日期
    
    Args:
        pipeline: 数据管道对象
        symbol: 股票代码
        signal_date: 信号生成日期
        exec_mode: 执行模式，'passive'（被动）或'active'（主动）
        price_type: 价格类型，'open'或'close'
        asof: 是否启用as-of填充
    
    Returns:
        (执行价格, 执行日期)
    """
    # 获取数据访问对象用于交易日判断
    data_access = getattr(pipeline, 'data_access', None)
    
    if exec_mode == "passive":
        # 被动回测：T日收盘价成交
        exec_date = signal_date
        exec_session = price_type if price_type in ["open", "close"] else "close"
    else:
        # 主动回测：T+1日开盘价成交
        if data_access:
            try:
                exec_date = next_trading_day(data_access, signal_date)
            except RuntimeError:
                # 如果找不到下一个交易日，使用信号日期
                exec_date = signal_date
        else:
            # 如果没有数据访问对象，简单加1天
            exec_date = signal_date + timedelta(days=1)
        exec_session = "open"
    
    # 获取执行价格
    try:
        price = session_price_qfq(pipeline, symbol, exec_date, exec_session, asof)
        return price, exec_date
    except RuntimeError as e:
        # 如果价格获取失败，尝试使用收盘价
        if exec_session != "close":
            try:
                price = session_price_qfq(pipeline, symbol, exec_date, "close", asof)
                return price, exec_date
            except RuntimeError:
                pass
        raise e
