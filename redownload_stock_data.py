#!/usr/bin/env python3
"""
重新下载股票数据脚本
从原始数据源（akshare）重新获取股票数据并存储到数据库
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from akshare_data_provider import AkshareDataProvider
from db import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataRedownloader:
    def __init__(self):
        self.data_provider = AkshareDataProvider()
        self.db_manager = DatabaseManager()
        
    def get_existing_stocks(self) -> List[str]:
        """从数据库获取已有的股票代码列表"""
        try:
            with self.db_manager.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM stocks")
                stocks = [row[0] for row in cursor.fetchall()]
                logger.info(f"数据库中找到 {len(stocks)} 只股票")
                return stocks
        except Exception as e:
            logger.error(f"获取现有股票列表失败: {e}")
            return []
    
    def download_stock_data(self, stock_symbol: str, period: str = "3y") -> bool:
        """下载单只股票的数据"""
        try:
            logger.info(f"开始下载股票 {stock_symbol} 的数据")
            
            # 使用akshare获取数据
            data_result = self.data_provider.get_stock_data(stock_symbol, period)
            
            if data_result is None or data_result.get('stock_data') is None:
                logger.warning(f"股票 {stock_symbol} 数据获取失败")
                return False
            
            stock_data = data_result['stock_data']
            stock_info = data_result.get('stock_info', {})
            
            if stock_data.empty:
                logger.warning(f"股票 {stock_symbol} 数据为空")
                return False
            
            # 重置索引以获取日期列
            stock_data.reset_index(inplace=True)
            
            # 准备股票基本信息
            stock_row = {
                'symbol': stock_symbol,
                'name': stock_info.get('longName', stock_symbol),
                'market': self._get_market_from_symbol(stock_symbol),
                'industry': stock_info.get('industry', ''),
                'market_cap': stock_info.get('marketCap', 0)
            }
            
            # 准备价格数据
            price_rows = []
            for _, row in stock_data.iterrows():
                price_row = {
                    'symbol': stock_symbol,
                    'date': row.get('Date', row.get('date', '')).strftime('%Y-%m-%d') if hasattr(row.get('Date', row.get('date', '')), 'strftime') else str(row.get('Date', row.get('date', ''))),
                    'open': float(row.get('Open', row.get('open', 0)) or 0),
                    'high': float(row.get('High', row.get('high', 0)) or 0),
                    'low': float(row.get('Low', row.get('low', 0)) or 0),
                    'close': float(row.get('Close', row.get('close', 0)) or 0),
                    'volume': float(row.get('Volume', row.get('volume', 0)) or 0),
                    'amount': float(row.get('Amount', row.get('amount', 0)) or 0),
                    'source': 'akshare'
                }
                price_rows.append(price_row)
            
            # 批量写入数据库
            success = self._save_to_database(stock_row, price_rows)
            
            if success:
                logger.info(f"股票 {stock_symbol} 数据下载成功，共 {len(price_rows)} 条记录")
            else:
                logger.error(f"股票 {stock_symbol} 数据保存失败")
            
            return success
            
        except Exception as e:
            logger.error(f"下载股票 {stock_symbol} 数据时出错: {e}")
            return False
    
    def _get_market_from_symbol(self, symbol: str) -> str:
        """从股票代码获取市场标识"""
        if symbol.endswith('.SS') or symbol.endswith('.SH'):
            return 'SH'
        elif symbol.endswith('.SZ'):
            return 'SZ'
        elif symbol.endswith('.HK'):
            return 'UNKNOWN'  # 港股代码已移除，统一标记为UNKNOWN
        else:
            return 'UNKNOWN'
    
    def _save_to_database(self, stock_row: Dict, price_rows: List[Dict]) -> bool:
        """保存数据到数据库"""
        try:
            with self.db_manager.get_conn() as conn:
                cursor = conn.cursor()
                
                # 插入或更新股票基本信息
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks (symbol, name, market, industry, market_cap)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    stock_row['symbol'],
                    stock_row['name'],
                    stock_row['market'],
                    stock_row['industry'],
                    stock_row['market_cap']
                ))
                
                # 清空该股票的现有价格数据
                cursor.execute("DELETE FROM prices_daily WHERE symbol = ?", (stock_row['symbol'],))
                
                # 批量插入价格数据
                for price_row in price_rows:
                    cursor.execute("""
                        INSERT OR REPLACE INTO prices_daily 
                        (symbol, date, open, high, low, close, volume, amount, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        price_row['symbol'],
                        price_row['date'],
                        price_row['open'],
                        price_row['high'],
                        price_row['low'],
                        price_row['close'],
                        price_row['volume'],
                        price_row['amount'],
                        price_row['source']
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"保存数据到数据库失败: {e}")
            return False
    
    def download_all_stocks(self, stock_list: Optional[List[str]] = None, period: str = "3y"):
        """下载所有股票的数据"""
        if stock_list is None:
            stock_list = self.get_existing_stocks()
        
        if not stock_list:
            logger.warning("没有股票需要下载")
            return
        
        total = len(stock_list)
        success_count = 0
        failed_stocks = []
        
        logger.info(f"开始下载 {total} 只股票的数据")
        
        for i, stock_symbol in enumerate(stock_list, 1):
            logger.info(f"进度: {i}/{total} - {stock_symbol}")
            
            success = self.download_stock_data(stock_symbol, period)
            
            if success:
                success_count += 1
            else:
                failed_stocks.append(stock_symbol)
            
            # 每下载10只股票后暂停1秒，避免请求过于频繁
            if i % 10 == 0:
                time.sleep(1)
        
        logger.info(f"数据下载完成！成功: {success_count}/{total}, 失败: {len(failed_stocks)}")
        
        if failed_stocks:
            logger.warning(f"下载失败的股票: {failed_stocks}")
    
    def verify_downloaded_data(self) -> Dict:
        """验证下载的数据"""
        try:
            with self.db_manager.get_conn() as conn:
                cursor = conn.cursor()
                
                # 统计股票数量
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stocks")
                stock_count = cursor.fetchone()[0]
                
                # 统计价格数据记录数
                cursor.execute("SELECT COUNT(*) FROM prices_daily")
                price_count = cursor.fetchone()[0]
                
                # 统计日期范围
                cursor.execute("""
                    SELECT MIN(date), MAX(date), COUNT(DISTINCT date) 
                    FROM prices_daily
                """)
                date_info = cursor.fetchone()
                
                # 检查数据完整性
                cursor.execute("""
                    SELECT symbol, COUNT(*) as record_count 
                    FROM prices_daily 
                    GROUP BY symbol 
                    ORDER BY record_count DESC
                """)
                symbol_stats = cursor.fetchall()
                
                result = {
                    'stock_count': stock_count,
                    'price_record_count': price_count,
                    'date_range': {
                        'min_date': date_info[0],
                        'max_date': date_info[1],
                        'distinct_dates': date_info[2]
                    },
                    'symbol_stats': symbol_stats[:10]  # 前10只股票的数据量统计
                }
                
                logger.info("数据验证结果:")
                logger.info(f"股票数量: {stock_count}")
                logger.info(f"价格记录数: {price_count}")
                logger.info(f"日期范围: {date_info[0]} 到 {date_info[1]} ({date_info[2]} 个不同日期)")
                
                return result
                
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {}

def main():
    """主函数"""
    redownloader = StockDataRedownloader()
    
    # 下载所有股票数据
    redownloader.download_all_stocks(period="3y")
    
    # 验证下载的数据
    verification_result = redownloader.verify_downloaded_data()
    
    print("\n=== 数据重新下载完成 ===")
    print(f"股票数量: {verification_result.get('stock_count', 0)}")
    print(f"价格记录数: {verification_result.get('price_record_count', 0)}")
    
    date_range = verification_result.get('date_range', {})
    if date_range:
        print(f"日期范围: {date_range.get('min_date', 'N/A')} 到 {date_range.get('max_date', 'N/A')}")
        print(f"不同日期数: {date_range.get('distinct_dates', 0)}")

if __name__ == "__main__":
    main()