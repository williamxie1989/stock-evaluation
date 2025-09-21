#!/usr/bin/env python3
"""
优化版增强数据提供者
主要优化：
1. 智能重试策略 - 减少重试次数和延迟
2. 熔断器机制 - 快速跳过失效数据源
3. 连接池复用 - 减少连接开销
4. 自适应数据源选择 - 优先使用成功率高的数据源
"""

import os
import re
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict, deque
import threading

import akshare as ak
import pandas as pd
import numpy as np
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from db import DatabaseManager
from stock_status_filter import StockStatusFilter
from akshare_data_provider import AkshareDataProvider


class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """通过熔断器调用函数"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e


class DataSourceStats:
    """数据源统计信息"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.success_history = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.last_success_time = None
        self.total_calls = 0
        self.total_successes = 0
        self.lock = threading.Lock()
    
    def record_call(self, success: bool, response_time: float):
        """记录调用结果"""
        with self.lock:
            self.total_calls += 1
            if success:
                self.total_successes += 1
                self.last_success_time = time.time()
            
            self.success_history.append(success)
            self.response_times.append(response_time)
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    def get_avg_response_time(self) -> float:
        """获取平均响应时间"""
        if not self.response_times:
            return 0.0  # 返回0.0而不是inf，避免JSON序列化问题
        return sum(self.response_times) / len(self.response_times)
    
    def get_priority_score(self) -> float:
        """获取优先级分数（越高越好）"""
        success_rate = self.get_success_rate()
        avg_time = self.get_avg_response_time()
        
        # 成功率权重70%，响应时间权重30%
        time_score = max(0, 1 - (avg_time / 10))  # 10秒以上响应时间得分为0
        return success_rate * 0.7 + time_score * 0.3


class OptimizedEnhancedDataProvider:
    """
    优化版增强数据提供者
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 优化的重试参数
        self.max_retries = max(1, int(os.getenv("OPT_MAX_RETRIES", "2")))  # 减少到2次
        self.base_retry_delay = max(0.1, float(os.getenv("OPT_RETRY_DELAY", "0.5")))  # 减少到0.5秒
        self.max_retry_delay = 2.0  # 最大重试延迟2秒
        
        # 熔断器配置
        self.circuit_breakers = {}
        self.failure_threshold = int(os.getenv("OPT_FAILURE_THRESHOLD", "3"))
        self.recovery_timeout = int(os.getenv("OPT_RECOVERY_TIMEOUT", "30"))
        
        # 数据源统计
        self.source_stats = {}
        
        # 连接池配置
        self.session = self._create_session()
        
        # 数据源配置（按优先级排序）
        self.data_sources = [
            {"name": "eastmoney", "func": self._get_data_from_eastmoney, "priority": 1},
            {"name": "sina", "func": self._get_data_from_sina, "priority": 2},
            {"name": "akshare_default", "func": self._get_data_from_akshare_default, "priority": 3},
        ]
        
        # 初始化熔断器和统计
        for source in self.data_sources:
            name = source["name"]
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout
            )
            self.source_stats[name] = DataSourceStats()
        
        # 其他组件
        self.last_used_source = None
        self.last_attempts = []
        
        try:
            self.db_manager = DatabaseManager()
        except Exception:
            self.db_manager = None
        
        try:
            self.stock_filter = StockStatusFilter()
        except Exception:
            self.stock_filter = None
        
        self.base_provider = AkshareDataProvider()
        
        self.logger.info(f"OptimizedEnhancedDataProvider initialized: max_retries={self.max_retries}, "
                        f"base_delay={self.base_retry_delay}s, failure_threshold={self.failure_threshold}")

    def _create_session(self) -> requests.Session:
        """创建优化的HTTP会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置超时
        session.timeout = (5, 10)  # 连接超时5秒，读取超时10秒
        
        return session

    def get_all_stock_list(self):
        """委托给基础提供者"""
        return self.base_provider.get_all_stock_list()

    def get_ah_spot(self):
        """委托给基础提供者"""
        return self.base_provider.get_ah_spot()

    def set_preferred_sources(self, sources: List[str]) -> None:
        """设置首选数据源"""
        if not sources:
            return
        
        # 重新排序数据源
        preferred_sources = []
        other_sources = []
        
        for source in self.data_sources:
            if source["name"] in sources:
                preferred_sources.append(source)
            else:
                other_sources.append(source)
        
        # 按照preferred_sources的顺序排列
        ordered_preferred = []
        for pref_name in sources:
            for source in preferred_sources:
                if source["name"] == pref_name:
                    ordered_preferred.append(source)
                    break
        
        self.data_sources = ordered_preferred + other_sources
        self.logger.info(f"数据源优先级已更新: {[s['name'] for s in self.data_sources]}")

    def get_sorted_data_sources(self) -> List[Dict[str, Any]]:
        """根据统计信息动态排序数据源"""
        sources_with_scores = []
        
        for source in self.data_sources:
            name = source["name"]
            stats = self.source_stats[name]
            circuit_breaker = self.circuit_breakers[name]
            
            # 跳过熔断状态的数据源
            if circuit_breaker.state == "OPEN":
                continue
            
            score = stats.get_priority_score()
            sources_with_scores.append((source, score))
        
        # 按分数降序排列
        sources_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [source for source, score in sources_with_scores]

    def get_stock_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        优化版多数据源获取股票历史数据
        """
        ak_symbol = self._convert_symbol_format(symbol)
        
        # 基本验证
        if (ak_symbol is None or str(ak_symbol).strip() == "" or 
            str(ak_symbol).upper().startswith("NONE") or str(ak_symbol) == "000000" or 
            not re.fullmatch(r"\d{6}", str(ak_symbol))):
            self.logger.warning(f"无效股票代码，跳过: original={symbol}, normalized={ak_symbol}")
            return None

        # 增量同步前置过滤
        if self._should_skip_incremental_sync(symbol, ak_symbol, period):
            return None

        # 重置状态
        self.last_used_source = None
        self.last_attempts = []

        # 获取动态排序的数据源
        sorted_sources = self.get_sorted_data_sources()
        
        if not sorted_sources:
            self.logger.warning("所有数据源都处于熔断状态")
            return None

        for source in sorted_sources:
            name, func = source["name"], source["func"]
            circuit_breaker = self.circuit_breakers[name]
            stats = self.source_stats[name]
            
            attempt_info = {"source": name, "start": datetime.now().isoformat()}
            start_time = time.time()
            
            try:
                # 通过熔断器调用
                data = circuit_breaker.call(self._call_with_optimized_retry, func, ak_symbol, period)
                
                if isinstance(data, pd.DataFrame) and not data.empty:
                    std = self._standardize_hist_df(data)
                    if (isinstance(std, pd.DataFrame) and not std.empty and 
                        all(c in std.columns for c in ["date", "open", "high", "low", "close", "volume"])):
                        
                        response_time = time.time() - start_time
                        stats.record_call(True, response_time)
                        
                        attempt_info.update({
                            "success": True, 
                            "rows": len(std), 
                            "status": "success",
                            "response_time": response_time
                        })
                        self.last_attempts.append(attempt_info)
                        self.last_used_source = name
                        
                        self.logger.info(f"{name} 成功获取 {ak_symbol} 历史数据, 行数={len(std)}, "
                                       f"耗时={response_time:.2f}s")
                        return std
                    else:
                        response_time = time.time() - start_time
                        stats.record_call(False, response_time)
                        attempt_info.update({
                            "success": False,
                            "error": "standardization failed",
                            "status": "empty",
                            "response_time": response_time
                        })
                else:
                    response_time = time.time() - start_time
                    stats.record_call(False, response_time)
                    attempt_info.update({
                        "success": False, 
                        "error": "empty result", 
                        "status": "empty",
                        "response_time": response_time
                    })
                
            except Exception as e:
                response_time = time.time() - start_time
                stats.record_call(False, response_time)
                attempt_info.update({
                    "success": False, 
                    "error": str(e), 
                    "status": "exception",
                    "response_time": response_time
                })
                self.logger.debug(f"{name} 获取 {ak_symbol} 数据失败: {e}")
            
            self.last_attempts.append(attempt_info)
            
            # 减少数据源间延迟
            if len(sorted_sources) > 1:
                time.sleep(random.uniform(0.05, 0.15))  # 减少到0.05-0.15秒

        self.logger.error(f"所有可用数据源都无法获取 {symbol} 的历史数据")
        return None

    def _should_skip_incremental_sync(self, symbol: str, ak_symbol: str, period: str) -> bool:
        """检查是否应该跳过增量同步"""
        try:
            is_incremental = isinstance(period, str) and period.lower().endswith("d")
        except Exception:
            is_incremental = False
        
        if not is_incremental:
            return False
        
        # 代码前缀过滤
        if str(ak_symbol).startswith(("688", "689", "900")):
            self.last_used_source = None
            self.last_attempts = [{
                "source": "precheck",
                "status": "filtered",
                "reason": "code_prefix_excluded",
                "symbol": str(ak_symbol)
            }]
            self.logger.info(f"增量同步过滤（前缀）: {symbol} -> {ak_symbol}")
            return True
        
        # 停牌/退市过滤
        try:
            if self.stock_filter and self.db_manager:
                stock_name = self._get_stock_name(symbol)
                check = self.stock_filter.should_filter_stock(
                    stock_name or "", symbol, include_st=False, include_suspended=True,
                    db_manager=self.db_manager, exclude_star_market=False,
                    last_n_days=30, include_no_trades_last_n_days=False
                )
                if check.get("should_filter"):
                    reason = check.get("reason", "filtered")
                    self.last_used_source = None
                    self.last_attempts = [{
                        "source": "precheck",
                        "status": "filtered", 
                        "reason": reason,
                        "symbol": symbol
                    }]
                    self.logger.info(f"增量同步过滤（状态）: {symbol}, reason={reason}")
                    return True
        except Exception:
            pass
        
        return False

    def _get_stock_name(self, symbol: str) -> Optional[str]:
        """获取股票名称"""
        try:
            if self.db_manager:
                with self.db_manager.get_conn() as conn:
                    dfn = pd.read_sql_query(
                        "SELECT name FROM stocks WHERE symbol = ? LIMIT 1",
                        conn, params=[symbol]
                    )
                    if not dfn.empty:
                        return dfn.iloc[0]["name"]
        except Exception:
            pass
        return None

    def _call_with_optimized_retry(self, func, *args, **kwargs):
        """优化的重试机制"""
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                error_msg = str(e).lower()
                
                # 判断是否为可重试错误
                retryable_errors = [
                    "rate limit", "too many requests", "timeout", "timed out",
                    "connection", "reset", "proxy", "temporarily unavailable",
                    "remote end closed", "connection aborted"
                ]
                
                is_retryable = any(err in error_msg for err in retryable_errors)
                
                if is_retryable and attempt < self.max_retries - 1:
                    # 使用线性退避而非指数退避，减少延迟
                    delay = min(self.base_retry_delay * (attempt + 1), self.max_retry_delay)
                    jitter = delay * 0.1 * random.random()  # 10%抖动
                    total_delay = delay + jitter
                    
                    self.logger.warning(f"数据源调用失败(尝试{attempt + 1}/{self.max_retries}): "
                                      f"{e}, {total_delay:.2f}s后重试")
                    time.sleep(total_delay)
                    continue
                
                raise e
        
        if last_err:
            raise last_err
        return None

    def _get_data_from_eastmoney(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从东财获取数据"""
        try:
            data = ak.stock_zh_a_hist(
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
            market_symbol = f"sh{symbol}" if str(symbol).startswith("6") else f"sz{symbol}"
            data = ak.stock_zh_a_daily(
                symbol=market_symbol,
                start_date=self._get_start_date(period),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
            return data
        except Exception as e:
            self.logger.debug(f"新浪接口失败: {e}")
            return None

    def _get_data_from_akshare_default(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从AkShare默认接口获取数据"""
        try:
            return self.base_provider.get_stock_historical_data(symbol, period)
        except Exception as e:
            self.logger.debug(f"AkShare默认接口失败: {e}")
            return None

    def _convert_symbol_format(self, symbol: Any) -> Optional[str]:
        """转换股票代码格式"""
        if symbol is None:
            return None
        
        symbol_str = str(symbol).strip().upper()
        
        # 移除后缀
        if '.' in symbol_str:
            symbol_str = symbol_str.split('.')[0]
        
        # 确保是6位数字
        if re.fullmatch(r'\d{6}', symbol_str):
            return symbol_str
        
        return None

    def _get_start_date(self, period: str) -> str:
        """根据周期计算开始日期"""
        try:
            if period.endswith('d'):
                days = int(period[:-1])
                start_date = datetime.now() - timedelta(days=days)
            elif period.endswith('y'):
                years = int(period[:-1])
                start_date = datetime.now() - timedelta(days=years * 365)
            else:
                start_date = datetime.now() - timedelta(days=365)
            
            return start_date.strftime('%Y%m%d')
        except Exception:
            return (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    def _standardize_hist_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化历史数据DataFrame"""
        if df is None or df.empty:
            return df
        
        # 复制数据避免修改原始数据
        result = df.copy()
        
        # 标准化列名映射
        column_mapping = {
            '日期': 'date', 'Date': 'date', 'DATE': 'date',
            '开盘': 'open', 'Open': 'open', 'OPEN': 'open', '开盘价': 'open',
            '最高': 'high', 'High': 'high', 'HIGH': 'high', '最高价': 'high',
            '最低': 'low', 'Low': 'low', 'LOW': 'low', '最低价': 'low',
            '收盘': 'close', 'Close': 'close', 'CLOSE': 'close', '收盘价': 'close',
            '成交量': 'volume', 'Volume': 'volume', 'VOLUME': 'volume', 'vol': 'volume'
        }
        
        # 重命名列
        result = result.rename(columns=column_mapping)
        
        # 确保必需列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in result.columns:
                self.logger.warning(f"缺少必需列: {col}")
                return pd.DataFrame()
        
        # 数据类型转换
        try:
            result['date'] = pd.to_datetime(result['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        except Exception as e:
            self.logger.warning(f"数据类型转换失败: {e}")
            return pd.DataFrame()
        
        # 移除无效数据
        result = result.dropna(subset=['date', 'close'])
        result = result[result['close'] > 0]
        
        # 按日期排序
        result = result.sort_values('date').reset_index(drop=True)
        
        return result

    def get_source_statistics(self) -> Dict[str, Any]:
        """获取数据源统计信息"""
        stats = {}
        for name, source_stats in self.source_stats.items():
            circuit_breaker = self.circuit_breakers[name]
            stats[name] = {
                "success_rate": source_stats.get_success_rate(),
                "avg_response_time": source_stats.get_avg_response_time(),
                "priority_score": source_stats.get_priority_score(),
                "total_calls": source_stats.total_calls,
                "total_successes": source_stats.total_successes,
                "circuit_breaker_state": circuit_breaker.state,
                "failure_count": circuit_breaker.failure_count
            }
        return stats

    def reset_circuit_breakers(self):
        """重置所有熔断器"""
        for circuit_breaker in self.circuit_breakers.values():
            with circuit_breaker.lock:
                circuit_breaker.state = "CLOSED"
                circuit_breaker.failure_count = 0
                circuit_breaker.last_failure_time = None
        self.logger.info("所有熔断器已重置")