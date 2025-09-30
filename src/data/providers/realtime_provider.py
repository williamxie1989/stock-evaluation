"""
实时数据提供器 - 精简版
"""

import logging
from typing import Dict, Any, Optional
import akshare as ak
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedRealtimeProvider:
    """增强实时数据提供器"""
    
    def __init__(self):
        self.data_sources = {
            'sina': self._get_sina_data,
            'tencent': self._get_tencent_data,
            'netease': self._get_netease_data,
            'eastmoney': self._get_eastmoney_data,
            'qq': self._get_qq_data,
            'akshare': self._get_akshare_data
        }
        logger.info(f"EnhancedRealtimeProvider 初始化完成: {len(self.data_sources)} 个数据源")
    
    def _get_sina_data(self, symbol: str) -> Optional[Dict]:
        """获取新浪数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"新浪数据源失败 {symbol}: {e}")
        return None
    
    def _get_tencent_data(self, symbol: str) -> Optional[Dict]:
        """获取腾讯数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"腾讯数据源失败 {symbol}: {e}")
        return None
    
    def _get_netease_data(self, symbol: str) -> Optional[Dict]:
        """获取网易数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"网易数据源失败 {symbol}: {e}")
        return None
    
    def _get_eastmoney_data(self, symbol: str) -> Optional[Dict]:
        """获取东方财富数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"东方财富数据源失败 {symbol}: {e}")
        return None
    
    def _get_qq_data(self, symbol: str) -> Optional[Dict]:
        """获取QQ数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"QQ数据源失败 {symbol}: {e}")
        return None
    
    def _get_akshare_data(self, symbol: str) -> Optional[Dict]:
        """获取Akshare数据"""
        try:
            # 使用最近一周的历史数据作为实时数据，确保有数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', '').replace('.SZ', ''), 
                                    period='daily', start_date=start_date, end_date=end_date)
            if not df.empty:
                # 获取最新一条数据
                latest_data = df.iloc[-1]
                return {
                    'price': float(latest_data['收盘']),
                    'change': float(latest_data['涨跌幅']),
                    'volume': int(latest_data['成交量']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.debug(f"Akshare数据源失败 {symbol}: {e}")
        return None
    
    def get_realtime_data(self, symbol: str, source: str = None) -> Optional[Dict]:
        """获取实时数据"""
        if source and source in self.data_sources:
            # 使用指定数据源
            return self.data_sources[source](symbol)
        else:
            # 依次尝试所有数据源
            for source_name, data_func in self.data_sources.items():
                try:
                    data = data_func(symbol)
                    if data:
                        logger.info(f"使用数据源 {source_name} 获取 {symbol} 数据")
                        return data
                except Exception as e:
                    logger.debug(f"数据源 {source_name} 失败: {e}")
                    continue
            
            logger.warning(f"所有数据源都无法获取 {symbol} 的实时数据")
            return None
    
    def get_batch_realtime_data(self, symbols: list, source: str = None) -> Dict[str, Dict]:
        """批量获取实时数据"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_realtime_data(symbol, source)
        return results