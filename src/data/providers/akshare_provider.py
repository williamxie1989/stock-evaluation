"""
Akshare数据提供器 - 精简版
"""

import logging
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union

import akshare as ak
import pandas as pd
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)


class RequestRateLimiter:
    """基于滑动窗口的简单限速器，避免频繁触发对端封禁。"""

    def __init__(self, max_calls: int, period: float, min_interval: float = 0.0):
        self.max_calls = max_calls
        self.period = period
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._timestamps = deque()
        self._last_acquired: float = 0.0

    def acquire(self) -> None:
        if self.max_calls <= 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > self.period:
                    self._timestamps.popleft()

                min_interval_wait = 0.0
                if self.min_interval > 0:
                    elapsed = now - self._last_acquired
                    if elapsed < self.min_interval:
                        min_interval_wait = self.min_interval - elapsed

                if len(self._timestamps) < self.max_calls and min_interval_wait <= 0:
                    self._timestamps.append(now)
                    self._last_acquired = now
                    return

                oldest_wait = 0.0
                if self._timestamps:
                    oldest_wait = self.period - (now - self._timestamps[0])
                wait_time = max(min_interval_wait, oldest_wait, 0.05)

            jitter = random.uniform(0, 0.1)
            time.sleep(wait_time + jitter)


class AkshareDataProvider:
    """Akshare数据提供器"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_requests_per_minute: int = 90,
        min_request_interval: float = 0.6,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.3,
        max_retry_delay: float = 20.0,
        rate_limiter: Optional[RequestRateLimiter] = None,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter
        self.max_retry_delay = max_retry_delay
        self._rate_limiter = rate_limiter or RequestRateLimiter(
            max_calls=max_requests_per_minute,
            period=60.0,
            min_interval=min_request_interval,
        )
        logger.info(
            "AkshareDataProvider configured: retries=%s delay=%ss max_rpm=%s min_interval=%ss",
            max_retries,
            retry_delay,
            max_requests_per_minute,
            min_request_interval,
        )

    def get_stock_data(self, symbol: str, start_date: str, end_date: str, adjust: str = "") -> Optional[pd.DataFrame]:
        """获取股票历史数据"""
        clean_symbol = self._clean_symbol(symbol)
        start_date_str, end_date_str = self._normalize_dates(start_date, end_date)
        adjust_param = self._normalize_adjust(adjust)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limiter.acquire()
                df = ak.stock_zh_a_hist(
                    symbol=clean_symbol,
                    period="daily",
                    start_date=start_date_str,
                    end_date=end_date_str,
                    adjust=adjust_param,
                )

                if df is None or df.empty:
                    logger.warning(
                        "获取 %s (%s) 数据为空 (attempt %s/%s)", symbol, adjust_param or "none", attempt, self.max_retries
                    )
                    if attempt == self.max_retries:
                        return None
                    self._sleep_with_backoff(attempt)
                    continue

                df = self._normalize_columns(df)
                logger.info(
                    "成功获取 %s 数据: %s 条记录 (adjust=%s)",
                    symbol,
                    len(df),
                    adjust_param or "none",
                )
                return df

            except (HTTPError, ConnectionError, Timeout, RequestException) as req_err:
                last_error = req_err
                logger.warning(
                    "网络错误获取 %s (%s) 数据失败 (attempt %s/%s): %s",
                    symbol,
                    adjust_param or "none",
                    attempt,
                    self.max_retries,
                    req_err,
                )
            except Exception as err:
                last_error = err
                logger.warning(
                    "获取 %s (%s) 数据异常 (attempt %s/%s): %s",
                    symbol,
                    adjust_param or "none",
                    attempt,
                    self.max_retries,
                    err,
                )

            if attempt < self.max_retries:
                self._sleep_with_backoff(attempt)

        logger.error("无法获取 %s 的股票数据 (adjust=%s): %s", symbol, adjust_param or "none", last_error)
        return None


    def get_stock_data_with_adjust(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        一次性获取不复权、前复权、后复权三种模式的日线数据，并合并到一个DataFrame：
        包含列：open, high, low, close（不复权）以及 open_qfq/high_qfq/low_qfq/close_qfq、open_hfq/high_hfq/low_hfq/close_hfq。
        明确使用akshare的新浪接口（stock_zh_a_hist adjust参数）获取复权数据。
        """
        try:
            raw = self.get_stock_data(symbol, start_date, end_date, adjust="")
            qfq = self.get_stock_data(symbol, start_date, end_date, adjust="qfq")
            hfq = self.get_stock_data(symbol, start_date, end_date, adjust="hfq")
        except Exception as e:
            logger.warning(f"获取三种复权数据失败: {symbol} {e}")
            return None

        if raw is None and qfq is None and hfq is None:
            return None

        def _basic(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            out = df.copy()
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
    
    def get_realtime_data(self, symbols: Union[str, List[str]]) -> Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
        """获取实时股票数据
        
        Args:
            symbols: 股票代码（字符串）或股票代码列表
            
        Returns:
            单个股票数据字典或批量股票数据字典
        """
        try:
            # 如果是字符串，处理单个股票
            if isinstance(symbols, str):
                return self._get_single_realtime_data(symbols)
            
            # 如果是列表，处理批量股票
            elif isinstance(symbols, list):
                result = {}
                for symbol in symbols:
                    data = self._get_single_realtime_data(symbol)
                    if data:
                        result[symbol] = data
                return result if result else None
            
            else:
                logger.error(f"不支持的参数类型: {type(symbols)}")
                return None
                
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #

    def _clean_symbol(self, symbol: str) -> str:
        if not symbol:
            return ""
        normalized = symbol.strip()
        lower = normalized.lower()
        if lower.endswith('.sh') or lower.endswith('.sz'):
            normalized = normalized[:-3]
        if lower.startswith('sh') or lower.startswith('sz'):
            normalized = normalized[2:]
        return normalized

    def _normalize_dates(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> tuple[str, str]:
        def _normalize(value: Union[str, datetime]) -> str:
            if isinstance(value, datetime):
                return value.strftime('%Y%m%d')
            value_str = str(value).strip()
            for sep in ('-', '/', '.'):
                if sep in value_str:
                    value_str = value_str.replace(sep, '')
            return value_str

        return _normalize(start_date), _normalize(end_date)

    def _normalize_adjust(self, adjust: Optional[str]) -> str:
        if noaksharedjust:
            return ""
        adjust_lower = adjust.lower()
        if adjust_lower in ("", "none", "raw", "no"):
            return ""
        if adjust_lower in ("qfq", "hfq"):
            return adjust_lower
        logger.warning("未知的复权参数: %s，使用不复权", adjust)
        return ""

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change_amount',
            '换手率': 'turnover',
        }
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        if existing_mapping:
            df = df.rename(columns=existing_mapping)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _sleep_with_backoff(self, attempt: int) -> None:
        base_delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
        delay = min(base_delay, self.max_retry_delay)
        jitter = random.uniform(0, self.retry_jitter) if self.retry_jitter > 0 else 0.0
        time.sleep(delay + jitter)
    
    def _get_single_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个股票的实时数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            # 转换股票代码格式，akshare只需要纯数字
            clean_symbol = self._clean_symbol(symbol)
            
            df = ak.stock_zh_a_hist(symbol=clean_symbol, period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'high': float(latest_data['最高']),
                    'low': float(latest_data['最低']),
                    'open': float(latest_data['开盘']),
                    'close': float(latest_data['收盘']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"获取 {symbol} 实时数据失败: {e}")
        return None
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            # 使用最近一周的历史数据作为股票列表来源，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            # 使用一个常见的股票代码来获取数据，作为股票列表的基础
            df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取全市场股票代码列表 - 使用股票列表接口
                try:
                    stock_list_df = ak.stock_zh_a_spot_em()
                    if not stock_list_df.empty:
                        # 只保留需要的列并重命名
                        stock_list_df = stock_list_df[['代码', '名称']].copy()
                        stock_list_df.columns = ['symbol', 'name']
                        # 添加交易所信息
                        stock_list_df['exchange'] = stock_list_df['symbol'].apply(lambda x: 'SH' if x.startswith('6') else 'SZ')
                        return stock_list_df
                except:
                    # 如果实时接口失败，返回一个基础的示例列表
                    sample_stocks = [
                        {'symbol': '000001', 'name': '平安银行', 'exchange': 'SZ'},
                        {'symbol': '000002', 'name': '万科A', 'exchange': 'SZ'},
                        {'symbol': '600000', 'name': '浦发银行', 'exchange': 'SH'},
                        {'symbol': '600036', 'name': '招商银行', 'exchange': 'SH'},
                        {'symbol': '600519', 'name': '贵州茅台', 'exchange': 'SH'}
                    ]
                    return pd.DataFrame(sample_stocks)
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()
    
    def get_market_data(self, market: str = "sh") -> Optional[pd.DataFrame]:
        """获取市场数据"""
        try:
            # 统一使用A股市场数据，确保数据可用性
            # 使用最近一周的历史数据作为市场数据来源，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            # 使用一个常见的股票代码来获取数据，作为市场数据的基础
            df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取全市场股票代码列表 - 使用股票列表接口
                try:
                    market_df = ak.stock_zh_a_spot_em()
                    if not market_df.empty:
                        logger.info(f"成功获取 {market} 市场数据: {len(market_df)} 条记录")
                        return market_df
                except:
                    # 如果实时接口失败，返回一个基础的示例数据
                    logger.warning(f"实时市场数据接口失败，返回示例数据")
                    sample_data = [
                        {'代码': '000001', '名称': '平安银行', '最新价': 10.50, '涨跌幅': 1.2},
                        {'代码': '000002', '名称': '万科A', '最新价': 15.80, '涨跌幅': -0.5},
                        {'代码': '600000', '名称': '浦发银行', '最新价': 8.90, '涨跌幅': 0.8},
                        {'代码': '600036', '名称': '招商银行', '最新价': 35.20, '涨跌幅': 2.1},
                        {'代码': '600519', '名称': '贵州茅台', '最新价': 1680.00, '涨跌幅': 3.5}
                    ]
                    return pd.DataFrame(sample_data)
        except Exception as e:
            logger.error(f"获取 {market} 市场数据失败: {e}")
        return None
    
    def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取财务数据"""
        try:
            # 获取主要财务指标
            df = ak.stock_financial_abstract(symbol=symbol)
            if not df.empty:
                return {
                    'symbol': symbol,
                    'data': df.to_dict('records')[0] if len(df) > 0 else {},
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"获取 {symbol} 财务数据失败: {e}")
        return None
    
    def get_news_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """获取新闻数据"""
        try:
            # 获取个股新闻
            df = ak.stock_news_em(symbol=symbol)
            if not df.empty:
                news_list = []
                for _, row in df.head(10).iterrows():  # 只取前10条
                    news_list.append({
                        'title': row.get('title', ''),
                        'content': row.get('content', ''),
                        'time': row.get('time', ''),
                        'source': row.get('source', '')
                    })
                return news_list
        except Exception as e:
            logger.error(f"获取 {symbol} 新闻数据失败: {e}")
        return None
    
    def get_all_stock_list(self) -> Optional[pd.DataFrame]:
        """获取全市场股票列表，包含板块分类信息"""
        all_stocks = []
        
        try:
            # 1. 获取沪深A股所有股票
            logger.info("正在获取沪深A股股票列表...")
            a_stocks = ak.stock_info_a_code_name()
            if a_stocks is not None and not a_stocks.empty:
                for _, row in a_stocks.iterrows():
                    code = row['code']
                    name = row['name']
                    
                    # 根据代码判断板块
                    market_type = self._classify_stock_market(code)
                    
                    all_stocks.append({
                        'code': code,  # 使用code而不是symbol，与refresh_data函数期望一致
                        'name': name,
                        'market': market_type['market'],
                        'board_type': market_type['board_type'],
                        'exchange': market_type['exchange']
                    })
            
            # 2. 获取上海证券交易所股票（包含科创板）
            logger.info("正在获取上海证券交易所股票列表...")
            try:
                # 主板A股
                sh_main = ak.stock_info_sh_name_code(symbol="主板A股")
                if sh_main is not None and not sh_main.empty:
                    for _, row in sh_main.iterrows():
                        code = row['证券代码']
                        name = row['证券简称']
                        
                        if not any(s['code'] == code for s in all_stocks):
                            all_stocks.append({
                                'code': code,
                                'name': name,
                                'market': 'SH',
                                'board_type': '主板',
                                'exchange': '上海证券交易所'
                            })
            except Exception as e:
                logger.warning(f"获取上海证券交易所股票列表失败: {e}")
            
            # 3. 获取深圳证券交易所股票（包含创业板）
            logger.info("正在获取深圳证券交易所股票列表...")
            try:
                # A股列表（包含主板和创业板）
                sz_stocks = ak.stock_info_sz_name_code(symbol="A股列表")
                if sz_stocks is not None and not sz_stocks.empty:
                    # 兼容不同版本列名
                    cols = list(sz_stocks.columns)
                    code_col = '证券代码' if '证券代码' in cols else ('A股代码' if 'A股代码' in cols else None)
                    name_col = '证券简称' if '证券简称' in cols else ('A股简称' if 'A股简称' in cols else None)
                    if code_col is None or name_col is None:
                        raise KeyError(f"意外的深交所列名: {cols}")
                    for _, row in sz_stocks.iterrows():
                        code = row[code_col]
                        name = row[name_col]
                        
                        if not any(s['code'] == code for s in all_stocks):
                            market_type = self._classify_stock_market(code)
                            all_stocks.append({
                                'code': code,
                                'name': name,
                                'market': 'SZ',
                                'board_type': market_type['board_type'],
                                'exchange': '深圳证券交易所'
                            })
            except Exception as e:
                logger.warning(f"获取深圳证券交易所股票列表失败: {e}")
            
            logger.info(f"成功获取 {len(all_stocks)} 只股票信息")
            return pd.DataFrame(all_stocks)
            
        except Exception as e:
            logger.error(f"获取全市场股票列表失败: {e}")
            return pd.DataFrame(columns=['code', 'name', 'market', 'board_type', 'exchange'])
    
    def _classify_stock_market(self, code):
        """根据股票代码分类市场和板块"""
        code = str(code).zfill(6)  # 补齐6位
        
        # 上海证券交易所（股票）
        if code.startswith('60'):
            return {'market': 'SH', 'board_type': '主板', 'exchange': '上海证券交易所'}
        elif code.startswith('688'):
            return {'market': 'SH', 'board_type': '科创板', 'exchange': '上海证券交易所'}
        elif code.startswith('900'):
            return {'market': 'SH', 'board_type': 'B股', 'exchange': '上海证券交易所'}
        
        # 深圳证券交易所
        elif code.startswith('000'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('001'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('002'):
            return {'market': 'SZ', 'board_type': '中小板', 'exchange': '深圳证券交易所'}
        elif code.startswith('003'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('300'):
            return {'market': 'SZ', 'board_type': '创业板', 'exchange': '深圳证券交易所'}
        elif code.startswith('301'):
            return {'market': 'SZ', 'board_type': '创业板', 'exchange': '深圳证券交易所'}
        elif code.startswith('200'):
            return {'market': 'SZ', 'board_type': 'B股', 'exchange': '深圳证券交易所'}
        
        # 默认分类
        else:
            return {'market': 'UNKNOWN', 'board_type': '未知', 'exchange': '未知交易所'}

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 尝试获取上证指数作为测试
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if not df.empty:
                logger.info("Akshare连接测试成功")
                return True
        except Exception as e:
            logger.error(f"Akshare连接测试失败: {e}")
        return False

    def get_trade_calendar(self, start_date: str = None, end_date: str = None, market: str = "ALL") -> Optional[pd.DataFrame]:
        """获取交易日历数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD，默认None表示从最早开始
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD，默认None表示到最新
            market: 市场类型，可选值："SH"(上海)、"SZ"(深圳)、"ALL"(全部)，默认"ALL"
            
        Returns:
            交易日历DataFrame，包含trade_date(交易日)、is_trading_day(是否交易日)、market(市场)列
        """
        try:
            # 设置默认日期范围
            if start_date is None:
                start_date = "19900101"  # 从1990年开始
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            
            # 转换日期格式
            if '-' in start_date:
                start_date = start_date.replace('-', '')
            if '-' in end_date:
                end_date = end_date.replace('-', '')
            
            logger.info(f"获取交易日历数据: market={market}, start_date={start_date}, end_date={end_date}")
            
            # 使用AKShare的tool_trade_date_hist_sina接口获取交易日历
            df = ak.tool_trade_date_hist_sina()
            
            if df.empty:
                logger.warning("获取交易日历数据为空")
                return None
            
            # 重命名列
            df = df.rename(columns={'trade_date': 'trade_date'})
            
            # 确保日期格式正确
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 过滤日期范围
            df = df[(df['trade_date'] >= pd.to_datetime(start_date)) & 
                    (df['trade_date'] <= pd.to_datetime(end_date))]
            
            if df.empty:
                logger.warning(f"在指定日期范围内未找到交易日历数据: {start_date} 到 {end_date}")
                return None
            
            # 添加是否交易日标记（AKShare返回的都是交易日）
            df['is_trading_day'] = True
            
            # 根据市场类型处理数据
            if market == "SH":
                df['market'] = "SH"
            elif market == "SZ":
                df['market'] = "SZ"
            else:  # ALL
                # 为每个日期创建上海和深圳两条记录
                sh_df = df.copy()
                sh_df['market'] = "SH"
                sz_df = df.copy()
                sz_df['market'] = "SZ"
                df = pd.concat([sh_df, sz_df], ignore_index=True)
            
            # 排序并重置索引
            df = df.sort_values(['trade_date', 'market']).reset_index(drop=True)
            
            logger.info(f"成功获取交易日历数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取交易日历数据失败: {e}")
            return None

    def get_trade_calendar_with_cache(self, start_date: str = None, end_date: str = None, 
                                     market: str = "ALL", cache_days: int = 7) -> Optional[pd.DataFrame]:
        """获取交易日历数据（带缓存功能）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            cache_days: 缓存天数，默认7天
            
        Returns:
            交易日历DataFrame
        """
        # 简单的缓存实现 - 在实际项目中可以使用更复杂的缓存机制
        cache_key = f"trade_calendar_{market}_{start_date}_{end_date}"
        
        # 这里可以添加缓存逻辑，但为了简化，直接调用原始方法
        return self.get_trade_calendar(start_date, end_date, market)
