#!/usr/bin/env python3
"""
从akshare重新下载股票数据
获取股票列表并下载所有股票的历史数据
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

try:
    import akshare as ak
except ImportError:
    print("请先安装akshare: pip install akshare")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AkshareDataDownloader:
    def __init__(self, db_path: str = "stock_data.sqlite3"):
        self.db_path = db_path
        self.max_retries = 3
        self.retry_delay = 2
        
    def get_conn(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path, timeout=30)
    
    def init_database(self):
        """初始化数据库表结构"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 创建股票基本信息表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        name TEXT,
                        market TEXT,
                        industry TEXT,
                        market_cap REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol)
                    )
                """)
                
                # 创建股票价格表（兼容旧表结构）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date 
                    ON stock_prices(symbol, date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_date 
                    ON stock_prices(date)
                """)
                
                conn.commit()
                logger.info("数据库表结构初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def get_a_share_stock_list(self) -> List[Dict]:
        """获取A股股票列表"""
        try:
            logger.info("正在获取A股股票列表...")
            
            # 获取A股代码和名称
            stock_info = ak.stock_info_a_code_name()
            
            if stock_info is None or stock_info.empty:
                logger.warning("获取A股列表失败，返回空数据")
                return []
            
            stocks = []
            for _, row in stock_info.iterrows():
                code = str(row.get('code', ''))
                name = str(row.get('name', ''))
                
                # 过滤无效数据
                if not code or code == 'nan' or len(code) < 6:
                    continue
                
                # 根据代码规则确定市场
                if code.startswith('6'):
                    market = 'SH'
                    symbol = f"{code}.SH"
                elif code.startswith(('0', '3')):
                    market = 'SZ'
                    symbol = f"{code}.SZ"
                else:
                    continue
                
                stocks.append({
                    'symbol': symbol,
                    'code': code,
                    'name': name,
                    'market': market
                })
            
            logger.info(f"获取到 {len(stocks)} 只A股股票")
            return stocks
            
        except Exception as e:
            logger.error(f"获取A股列表失败: {e}")
            return []
    
    def download_stock_history(self, symbol: str, code: str, period: str = "daily") -> Optional[pd.DataFrame]:
        """下载单只股票的历史数据"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"下载 {symbol} 的历史数据 (尝试 {attempt + 1}/{self.max_retries})")
                
                # 获取历史行情数据
                stock_df = ak.stock_zh_a_hist(symbol=code, period=period, adjust="qfq")
                
                if stock_df is None or stock_df.empty:
                    logger.warning(f"股票 {symbol} 返回空数据")
                    return None
                
                # 检查必要的列
                required_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                missing_columns = [col for col in required_columns if col not in stock_df.columns]
                
                if missing_columns:
                    logger.error(f"股票 {symbol} 数据缺少列: {missing_columns}")
                    return None
                
                # 重命名列
                stock_df = stock_df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })
                
                # 添加股票代码列
                stock_df['symbol'] = symbol
                
                # 转换日期格式
                stock_df['date'] = pd.to_datetime(stock_df['date']).dt.strftime('%Y-%m-%d')
                
                # 确保数值列的数据类型
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce').fillna(0)
                
                # 添加adj_close列（使用close值）
                stock_df['adj_close'] = stock_df['close']
                
                # 按日期排序
                stock_df = stock_df.sort_values('date')
                
                logger.info(f"股票 {symbol} 数据下载成功，共 {len(stock_df)} 条记录")
                return stock_df
                
            except Exception as e:
                logger.error(f"下载 {symbol} 数据失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue
        
        return None
    
    def save_stock_data(self, stock_info: Dict, stock_data: pd.DataFrame) -> bool:
        """保存股票数据到数据库"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                symbol = stock_info['symbol']
                
                # 插入股票基本信息
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks (symbol, name, market)
                    VALUES (?, ?, ?)
                """, (
                    symbol,
                    stock_info.get('name', ''),
                    stock_info.get('market', '')
                ))
                
                # 清空该股票的现有价格数据
                cursor.execute("DELETE FROM stock_prices WHERE symbol = ?", (symbol,))
                
                # 批量插入价格数据
                for _, row in stock_data.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO stock_prices 
                        (symbol, date, open, high, low, close, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        float(row['close'])  # adj_close使用close值
                    ))
                
                conn.commit()
                logger.info(f"股票 {symbol} 数据保存成功，共 {len(stock_data)} 条记录")
                return True
                
        except Exception as e:
            logger.error(f"保存股票 {symbol} 数据失败: {e}")
            return False
    
    def download_all_stocks(self, max_stocks: int = 100, start_index: int = 0):
        """下载所有股票的数据"""
        # 获取股票列表
        stock_list = self.get_a_share_stock_list()
        
        if not stock_list:
            logger.error("无法获取股票列表")
            return
        
        # 限制下载数量
        if max_stocks > 0:
            stock_list = stock_list[start_index:start_index + max_stocks]
        
        total = len(stock_list)
        success_count = 0
        failed_stocks = []
        
        logger.info(f"开始下载 {total} 只股票的历史数据")
        
        for i, stock_info in enumerate(stock_list, 1):
            symbol = stock_info['symbol']
            code = stock_info['code']
            
            logger.info(f"进度: {i}/{total} - {symbol} ({stock_info['name']})")
            
            # 下载历史数据
            stock_data = self.download_stock_history(symbol, code)
            
            if stock_data is not None and not stock_data.empty:
                # 保存到数据库
                success = self.save_stock_data(stock_info, stock_data)
                
                if success:
                    success_count += 1
                else:
                    failed_stocks.append(symbol)
            else:
                failed_stocks.append(symbol)
                logger.warning(f"股票 {symbol} 数据下载失败")
            
            # 每下载5只股票后暂停1秒，避免请求过于频繁
            if i % 5 == 0:
                time.sleep(1)
                logger.info(f"已下载 {i} 只股票，成功 {success_count} 只")
        
        logger.info(f"数据下载完成！成功: {success_count}/{total}, 失败: {len(failed_stocks)}")
        
        if failed_stocks:
            logger.warning(f"下载失败的股票: {failed_stocks[:20]}")  # 只显示前20个
    
    def verify_data(self) -> Dict:
        """验证下载的数据"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 统计股票数量
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stocks")
                stock_count = cursor.fetchone()[0]
                
                # 统计价格数据记录数
                cursor.execute("SELECT COUNT(*) FROM stock_prices")
                price_count = cursor.fetchone()[0]
                
                # 统计日期范围
                cursor.execute("""
                    SELECT MIN(date), MAX(date), COUNT(DISTINCT date) 
                    FROM stock_prices
                """)
                date_info = cursor.fetchone()
                
                # 获取数据量最多的股票
                cursor.execute("""
                    SELECT symbol, COUNT(*) as record_count 
                    FROM stock_prices 
                    GROUP BY symbol 
                    ORDER BY record_count DESC 
                    LIMIT 10
                """)
                top_stocks = cursor.fetchall()
                
                result = {
                    'stock_count': stock_count,
                    'price_record_count': price_count,
                    'date_range': {
                        'min_date': date_info[0],
                        'max_date': date_info[1],
                        'distinct_dates': date_info[2]
                    },
                    'top_stocks': top_stocks
                }
                
                logger.info("=== 数据验证结果 ===")
                logger.info(f"股票数量: {stock_count}")
                logger.info(f"价格记录数: {price_count}")
                logger.info(f"日期范围: {date_info[0]} 到 {date_info[1]} ({date_info[2]} 个不同日期)")
                logger.info("数据量最多的股票:")
                for symbol, count in top_stocks:
                    logger.info(f"  {symbol}: {count} 条记录")
                
                return result
                
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {}

def main():
    """主函数"""
    downloader = AkshareDataDownloader()
    
    # 初始化数据库
    downloader.init_database()
    
    # 下载股票数据（限制为前50只作为示例）
    logger.info("开始下载股票数据...")
    downloader.download_all_stocks(max_stocks=50)
    
    # 验证下载的数据
    verification_result = downloader.verify_data()
    
    print("\n=== 数据下载完成 ===")
    print(f"股票数量: {verification_result.get('stock_count', 0)}")
    print(f"价格记录数: {verification_result.get('price_record_count', 0)}")
    
    date_range = verification_result.get('date_range', {})
    if date_range:
        print(f"日期范围: {date_range.get('min_date', 'N/A')} 到 {date_range.get('max_date', 'N/A')}")

if __name__ == "__main__":
    main()