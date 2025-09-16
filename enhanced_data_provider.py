import akshare as ak
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, List, Dict, Any
import random

class EnhancedDataProvider:
    """
    增强版数据提供者，支持多数据源互补获取历史价格数据
    支持的数据源：东财、新浪、腾讯、网易等
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 数据源配置，按优先级排序
        self.data_sources = [
            {'name': 'eastmoney', 'func': self._get_data_from_eastmoney},
            {'name': 'sina', 'func': self._get_data_from_sina},
            {'name': 'tencent', 'func': self._get_data_from_tencent},
            {'name': 'netease', 'func': self._get_data_from_netease},
            {'name': 'akshare_default', 'func': self._get_data_from_akshare_default}
        ]
        
    def get_stock_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        多数据源获取股票历史数据
        
        Args:
            symbol: 股票代码，如 '000001.SZ', '600000.SH'
            period: 时间周期，如 '1y', '2y', '3y'
            
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        ak_symbol = self._convert_symbol_format(symbol)
        
        for source in self.data_sources:
            try:
                self.logger.info(f"尝试从 {source['name']} 获取 {symbol} 的历史数据")
                data = source['func'](ak_symbol, period)
                
                if data is not None and not data.empty and len(data) > 30:
                    self.logger.info(f"成功从 {source['name']} 获取到 {len(data)} 条数据")
                    return self._standardize_data(data)
                else:
                    self.logger.warning(f"{source['name']} 返回数据不足或为空")
                    
            except Exception as e:
                self.logger.warning(f"从 {source['name']} 获取数据失败: {str(e)}")
                continue
                
        self.logger.error(f"所有数据源都无法获取 {symbol} 的历史数据")
        return None
        
    def _convert_symbol_format(self, symbol: str) -> str:
        """转换股票代码格式为akshare格式，统一将 .SS 视为 .SH"""
        if symbol is None:
            return symbol
        s = str(symbol).strip().upper()
        # 将 .SS 归一为 .SH
        if s.endswith('.SS'):
            s = s[:-3] + '.SH'
        # 对于带后缀的 A 股，返回纯数字代码给 akshare
        if s.endswith('.SZ') or s.endswith('.SH'):
            return s.split('.')[0]
        return s
        
    def _get_data_from_eastmoney(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从东财获取数据"""
        try:
            # 使用东财接口
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=self._get_start_date(period), adjust="qfq")
            return data
        except Exception as e:
            self.logger.debug(f"东财接口失败: {e}")
            return None
            
    def _get_data_from_sina(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从新浪获取数据"""
        try:
            # 使用新浪接口
            data = ak.stock_zh_a_daily(symbol=f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}", 
                                     start_date=self._get_start_date(period),
                                     end_date=datetime.now().strftime('%Y%m%d'),
                                     adjust="qfq")
            return data
        except Exception as e:
            self.logger.debug(f"新浪接口失败: {e}")
            return None
            
    def _get_data_from_tencent(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从腾讯获取数据"""
        try:
            # 使用腾讯接口
            market_code = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
            data = ak.stock_individual_info_em(symbol=market_code)
            # 注意：这里需要根据实际的腾讯接口调整
            return None  # 暂时返回None，需要找到合适的腾讯历史数据接口
        except Exception as e:
            self.logger.debug(f"腾讯接口失败: {e}")
            return None
            
    def _get_data_from_netease(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从网易获取数据"""
        try:
            # 使用网易接口 - 需要找到合适的网易历史数据接口
            return None  # 暂时返回None
        except Exception as e:
            self.logger.debug(f"网易接口失败: {e}")
            return None
            
    def _get_data_from_akshare_default(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """使用akshare默认接口作为最后的fallback"""
        try:
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            return data
        except Exception as e:
            self.logger.debug(f"akshare默认接口失败: {e}")
            return None
            
    def _get_start_date(self, period: str) -> str:
        """根据周期计算开始日期"""
        end_date = datetime.now()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "3y":
            start_date = end_date - timedelta(days=1095)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)  # 默认1年
            
        return start_date.strftime('%Y%m%d')
        
    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式"""
        if data is None or data.empty:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            
        # 处理不同数据源的列名
        column_mapping = {
            '日期': 'date',
            '开盘': 'open', 
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # 重命名列
        data = data.rename(columns=column_mapping)
        
        # 确保必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 0  # 如果没有成交量数据，设为0
                else:
                    self.logger.warning(f"缺少必要列: {col}")
                    return pd.DataFrame(columns=required_columns)
                    
        # 选择需要的列并排序
        data = data[required_columns].copy()
        
        # 确保日期格式正确
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            
        # 按日期排序
        data = data.sort_values('date').reset_index(drop=True)
        
        return data
        
    def batch_get_historical_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """批量获取多只股票的历史数据"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"正在获取 {symbol} 的历史数据...")
            data = self.get_stock_historical_data(symbol, period)
            
            if data is not None and not data.empty:
                results[symbol] = data
                self.logger.info(f"成功获取 {symbol} 的 {len(data)} 条历史数据")
            else:
                self.logger.warning(f"无法获取 {symbol} 的历史数据")
                
            # 添加随机延迟避免请求过于频繁
            time.sleep(random.uniform(0.1, 0.5))
            
        return results
        
    def get_market_stocks_with_data(self, market: str, board_type: str, min_data_points: int = 30) -> List[str]:
        """获取指定市场板块中有足够历史数据的股票列表"""
        from db import DatabaseManager
        
        db = DatabaseManager()
        
        # 获取该市场板块的所有股票
        with db.get_conn() as conn:
            cursor = conn.cursor()
            query = """
                SELECT symbol FROM stocks 
                WHERE market = ? AND board_type = ?
                ORDER BY symbol
            """
            cursor.execute(query, (market, board_type))
            stocks = [row[0] for row in cursor.fetchall()]
            
        self.logger.info(f"开始检查 {market} {board_type} 的 {len(stocks)} 只股票的数据完整性")
        
        stocks_with_data = []
        
        for symbol in stocks[:50]:  # 限制检查数量避免过长时间
            try:
                data = self.get_stock_historical_data(symbol, "1y")
                if data is not None and len(data) >= min_data_points:
                    stocks_with_data.append(symbol)
                    self.logger.debug(f"{symbol}: 有 {len(data)} 条数据")
                else:
                    self.logger.debug(f"{symbol}: 数据不足")
                    
                # 添加延迟
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"检查 {symbol} 数据时出错: {e}")
                continue
                
        self.logger.info(f"{market} {board_type} 中有 {len(stocks_with_data)} 只股票有足够的历史数据")
        return stocks_with_data


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    provider = EnhancedDataProvider()
    
    # 测试单只股票数据获取
    test_symbols = ['000001.SZ', '600000.SH', '688001.SH']  # 深市主板、沪市主板、科创板
    
    for symbol in test_symbols:
        print(f"\n测试获取 {symbol} 的历史数据:")
        data = provider.get_stock_historical_data(symbol, "1y")
        if data is not None:
            print(f"成功获取 {len(data)} 条数据")
            print(data.head())
        else:
            print("获取失败")