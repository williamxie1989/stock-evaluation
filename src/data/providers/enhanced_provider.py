"""
增强数据提供器 - 精简版
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .akshare_provider import AkshareDataProvider

logger = logging.getLogger(__name__)

class EnhancedDataProvider:
    """增强数据提供器 - 组合多个数据源"""
    
    def __init__(self, primary_provider: str = "akshare", **kwargs):
        self.primary_provider = primary_provider
        self.akshare_provider = AkshareDataProvider(**kwargs)
        logger.info(f"EnhancedDataProvider initialized with primary provider: {primary_provider}")
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票历史数据"""
        if self.primary_provider == "akshare":
            return self.akshare_provider.get_stock_data(symbol, start_date, end_date)
        else:
            logger.warning(f"Unknown primary provider: {self.primary_provider}, falling back to akshare")
            return self.akshare_provider.get_stock_data(symbol, start_date, end_date)
    
    def get_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取实时股票数据"""
        return self.akshare_provider.get_realtime_data(symbol)
    
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """获取股票列表"""
        return self.akshare_provider.get_stock_list()
    
    def get_market_data(self, market: str = "sh") -> Optional[pd.DataFrame]:
        """获取市场数据"""
        return self.akshare_provider.get_market_data(market)
    
    def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取财务数据"""
        return self.akshare_provider.get_financial_data(symbol)
    
    def get_news_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """获取新闻数据"""
        return self.akshare_provider.get_news_data(symbol)
    
    def get_enhanced_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """获取增强股票数据（包含多个维度）"""
        try:
            # 获取历史数据
            historical_data = self.get_stock_data(symbol, start_date, end_date)
            if historical_data is None:
                return None
            
            # 获取实时数据
            realtime_data = self.get_realtime_data(symbol)
            
            # 获取财务数据
            financial_data = self.get_financial_data(symbol)
            
            # 获取新闻数据
            news_data = self.get_news_data(symbol)
            
            return {
                'symbol': symbol,
                'historical_data': historical_data,
                'realtime_data': realtime_data,
                'financial_data': financial_data,
                'news_data': news_data,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"获取增强股票数据失败 {symbol}: {e}")
            return None
    
    def get_batch_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """批量获取股票数据"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date)
                if data is not None:
                    results[symbol] = data
                else:
                    logger.warning(f"无法获取 {symbol} 的数据")
            except Exception as e:
                logger.error(f"获取 {symbol} 数据时出错: {e}")
        
        logger.info(f"批量获取数据完成: {len(results)}/{len(symbols)} 成功")
        return results
    
    def get_batch_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """批量获取实时数据"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_realtime_data(symbol)
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"获取 {symbol} 实时数据时出错: {e}")
        
        logger.info(f"批量获取实时数据完成: {len(results)}/{len(symbols)} 成功")
        return results
    
    def test_connection(self) -> bool:
        """测试连接"""
        return self.akshare_provider.test_connection()