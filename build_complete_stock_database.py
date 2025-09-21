#!/usr/bin/env python3
"""
构建完整的股票数据库
根据项目中的筛选条件，获取所有符合条件的A股股票数据
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import re

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

class CompleteStockDatabaseBuilder:
    def __init__(self, db_path: str = "stock_data.sqlite3"):
        self.db_path = db_path
        self.max_retries = 3
        self.retry_delay = 2
        
        # 股票筛选配置（基于项目中的设置）
        self.filter_config = {
            # 排除B股
            'exclude_b_share': True,
            # 排除科创板（可选，根据需求调整）
            'exclude_star_market': False,  # 改为False以包含科创板
            # 排除北交所（已移除）
            'exclude_bse_stock': True,
            # 包含ST股票
            'include_st': True,
            # 包含停牌股票
            'include_suspended': True,
            # 排除指数类
            'exclude_indices': True,
            # 最小历史数据条数
            'min_history_records': 5,
            # 排除88开头的板块指数
            'exclude_88_series': True,
            # 排除特定板块类型
            'exclude_board_types': ['指数', '行业指数', '板块', '基金', 'ETF']
        }
        
    def get_conn(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path, timeout=30)
    
    def init_database(self):
        """初始化数据库表结构"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 创建股票基本信息表（与原始db.py保持一致）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,              -- 股票代码（A股代码）
                        name TEXT,                         -- 股票名称
                        market TEXT,                       -- 市场标识，如 SH/SZ/HK
                        board_type TEXT,                   -- 板块类型（主板、创业板、科创板等）
                        exchange TEXT,                     -- 交易所名称
                        ah_pair TEXT,                      -- 若有，对应另一市场代码，例如 H 股代码
                        industry TEXT,                     -- 行业
                        market_cap REAL,                   -- 总市值（元）
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
                
                # 如果stock_prices表已存在但缺少adj_close列，添加它
                cursor.execute("PRAGMA table_info(stock_prices)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'adj_close' not in columns:
                    cursor.execute("ALTER TABLE stock_prices ADD COLUMN adj_close REAL")
                    logger.info("已添加adj_close列到stock_prices表")
                
                # 创建A股价格数据表（新表）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prices_daily (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        amount REAL,
                        source TEXT,
                        UNIQUE(symbol, date)
                    )
                """)
                
                # 如果prices_daily表已存在但缺少某些列，添加它们
                cursor.execute("PRAGMA table_info(prices_daily)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'amount' not in columns:
                    cursor.execute("ALTER TABLE prices_daily ADD COLUMN amount REAL")
                    logger.info("已添加amount列到prices_daily表")
                if 'source' not in columns:
                    cursor.execute("ALTER TABLE prices_daily ADD COLUMN source TEXT")
                    logger.info("已添加source列到prices_daily表")
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stocks_board_type ON stocks(board_type)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date 
                    ON stock_prices(symbol, date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_prices_date 
                    ON stock_prices(date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date 
                    ON prices_daily(symbol, date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prices_daily_date 
                    ON prices_daily(date)
                """)
                
                conn.commit()
                logger.info("数据库表结构初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _classify_stock_market(self, code: str) -> Dict[str, str]:
        """根据股票代码分类市场和板块"""
        code = str(code).zfill(6)  # 补齐6位
        
        # 排除88开头的指数
        if code.startswith('88'):
            return {'market': 'INDEX', 'board_type': '指数', 'exchange': '指数'}
        
        # 上海证券交易所
        if code.startswith('60'):
            return {'market': 'SH', 'board_type': '主板', 'exchange': '上海证券交易所'}
        elif code.startswith('688') or code.startswith('689'):
            return {'market': 'SH', 'board_type': '科创板', 'exchange': '上海证券交易所'}
        elif code.startswith('900'):
            return {'market': 'SH', 'board_type': 'B股', 'exchange': '上海证券交易所'}
        
        # 深圳证券交易所
        elif code.startswith('000') or code.startswith('001') or code.startswith('003'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('002'):
            return {'market': 'SZ', 'board_type': '中小板', 'exchange': '深圳证券交易所'}
        elif code.startswith('300') or code.startswith('301'):
            return {'market': 'SZ', 'board_type': '创业板', 'exchange': '深圳证券交易所'}
        elif code.startswith('200'):
            return {'market': 'SZ', 'board_type': 'B股', 'exchange': '深圳证券交易所'}
        
        # 北交所代码（已移除）
        elif code.startswith(('4', '8')):
            return {'market': 'UNKNOWN', 'board_type': '已移除', 'exchange': '北京证券交易所(已移除)'}
        
        # 默认分类
        else:
            return {'market': 'UNKNOWN', 'board_type': '未知', 'exchange': '未知交易所'}
    
    def _should_filter_stock(self, code: str, name: str) -> Dict[str, any]:
        """判断是否应该过滤该股票"""
        # 获取分类信息
        classification = self._classify_stock_market(code)
        market = classification['market']
        board_type = classification['board_type']
        
        # 排除B股
        if self.filter_config['exclude_b_share'] and board_type == 'B股':
            return {'should_filter': True, 'reason': 'b_share'}
        
        # 排除科创板
        if self.filter_config['exclude_star_market'] and board_type == '科创板':
            return {'should_filter': True, 'reason': 'star_market'}
        
        # 排除北交所
        if self.filter_config['exclude_bse_stock'] and market == 'UNKNOWN' and '已移除' in board_type:
            return {'should_filter': True, 'reason': 'bse_stock_removed'}
        
        # 排除指数类
        if self.filter_config['exclude_indices'] and board_type in ['指数', '行业指数', '板块', '基金', 'ETF']:
            return {'should_filter': True, 'reason': 'index'}
        
        # 排除88开头的指数
        if self.filter_config['exclude_88_series'] and code.startswith('88'):
            return {'should_filter': True, 'reason': '88_series_index'}
        
        # 检查ST股票
        if not self.filter_config['include_st'] and self._is_st_stock(name):
            return {'should_filter': True, 'reason': 'st_stock'}
        
        # 检查停牌股票
        if not self.filter_config['include_suspended'] and self._is_suspended_stock(name):
            return {'should_filter': True, 'reason': 'suspended'}
        
        return {'should_filter': False, 'reason': None}
    
    def _is_st_stock(self, name: str) -> bool:
        """判断是否为ST股票"""
        if not name:
            return False
        name = str(name).upper()
        return 'ST' in name or '*ST' in name or '退' in name
    
    def _is_suspended_stock(self, name: str) -> bool:
        """判断是否为停牌股票"""
        if not name:
            return False
        name = str(name).upper()
        return '停牌' in name or '暂停' in name
    
    def get_all_a_share_stocks(self) -> List[Dict]:
        """获取所有A股股票列表"""
        all_stocks = []
        
        try:
            logger.info("开始获取A股股票列表...")
            
            # 1. 获取上海证券交易所股票
            logger.info("正在获取上海证券交易所股票列表...")
            try:
                # 主板A股
                sh_main = ak.stock_info_sh_name_code(symbol="主板A股")
                if sh_main is not None and not sh_main.empty:
                    for _, row in sh_main.iterrows():
                        code = str(row['证券代码']).zfill(6)
                        name = str(row['证券简称'])
                        
                        # 过滤检查
                        filter_check = self._should_filter_stock(code, name)
                        if filter_check['should_filter']:
                            logger.debug(f"跳过过滤股票: {code} - {name} ({filter_check['reason']})")
                            continue
                        
                        classification = self._classify_stock_market(code)
                        
                        all_stocks.append({
                            'symbol': f"{code}.SH",
                            'name': name,
                            'market': classification['market'],
                            'board_type': classification['board_type'],
                            'exchange': classification['exchange'],
                            'industry': row.get('所属行业', ''),
                            'market_cap': None
                        })
                        
            except Exception as e:
                logger.error(f"获取上海证券交易所主板股票失败: {e}")
            
            try:
                # 科创板
                sh_star = ak.stock_info_sh_name_code(symbol="科创板")
                if sh_star is not None and not sh_star.empty:
                    for _, row in sh_star.iterrows():
                        code = str(row['证券代码']).zfill(6)
                        name = str(row['证券简称'])
                        
                        # 过滤检查
                        filter_check = self._should_filter_stock(code, name)
                        if filter_check['should_filter']:
                            logger.debug(f"跳过过滤股票: {code} - {name} ({filter_check['reason']})")
                            continue
                        
                        classification = self._classify_stock_market(code)
                        
                        all_stocks.append({
                            'symbol': f"{code}.SH",
                            'name': name,
                            'market': classification['market'],
                            'board_type': classification['board_type'],
                            'exchange': classification['exchange'],
                            'industry': row.get('所属行业', ''),
                            'market_cap': None
                        })
                        
            except Exception as e:
                logger.error(f"获取上海证券交易所科创板股票失败: {e}")
            
            # 2. 获取深圳证券交易所股票
            logger.info("正在获取深圳证券交易所股票列表...")
            try:
                sz_stocks = ak.stock_info_sz_name_code()
                if sz_stocks is not None and not sz_stocks.empty:
                    for _, row in sz_stocks.iterrows():
                        code = str(row['A股代码']).zfill(6)
                        name = str(row['A股简称'])
                        
                        # 过滤检查
                        filter_check = self._should_filter_stock(code, name)
                        if filter_check['should_filter']:
                            logger.debug(f"跳过过滤股票: {code} - {name} ({filter_check['reason']})")
                            continue
                        
                        classification = self._classify_stock_market(code)
                        
                        all_stocks.append({
                            'symbol': f"{code}.SZ",
                            'name': name,
                            'market': classification['market'],
                            'board_type': classification['board_type'],
                            'exchange': classification['exchange'],
                            'industry': row.get('行业', ''),
                            'market_cap': None
                        })
                        
            except Exception as e:
                logger.error(f"获取深圳证券交易所股票失败: {e}")
            
            # 3. 获取基础的股票代码和名称（备用方案）
            if not all_stocks:
                logger.info("使用备用方案获取股票列表...")
                try:
                    stock_info = ak.stock_info_a_code_name()
                    if stock_info is not None and not stock_info.empty:
                        for _, row in stock_info.iterrows():
                            code = str(row['code']).zfill(6)
                            name = str(row['name'])
                            
                            # 过滤检查
                            filter_check = self._should_filter_stock(code, name)
                            if filter_check['should_filter']:
                                logger.debug(f"跳过过滤股票: {code} - {name} ({filter_check['reason']})")
                                continue
                            
                            classification = self._classify_stock_market(code)
                            
                            all_stocks.append({
                                'symbol': f"{code}.{classification['market']}",
                                'name': name,
                                'market': classification['market'],
                                'board_type': classification['board_type'],
                                'exchange': classification['exchange'],
                                'industry': '',
                                'market_cap': None
                            })
                            
                except Exception as e:
                    logger.error(f"获取备用股票列表失败: {e}")
            
            logger.info(f"共获取到 {len(all_stocks)} 只A股股票")
            
            # 去重
            seen = set()
            unique_stocks = []
            for stock in all_stocks:
                if stock['symbol'] not in seen:
                    seen.add(stock['symbol'])
                    unique_stocks.append(stock)
            
            logger.info(f"去重后剩余 {len(unique_stocks)} 只股票")
            return unique_stocks
            
        except Exception as e:
            logger.error(f"获取A股股票列表失败: {e}")
            return []
    
    def save_stocks_to_database(self, stocks: List[Dict]) -> bool:
        """保存股票信息到数据库"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 清空现有股票数据
                cursor.execute("DELETE FROM stocks")
                logger.info("已清空现有股票数据")
                
                # 批量插入股票数据
                for stock in stocks:
                    cursor.execute("""
                        INSERT INTO stocks 
                        (symbol, name, market, board_type, exchange, ah_pair, industry, market_cap)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        stock['symbol'],
                        stock['name'],
                        stock['market'],
                        stock['board_type'],
                        stock['exchange'],
                        None,  # ah_pair
                        stock['industry'],
                        stock['market_cap']
                    ))
                
                conn.commit()
                logger.info(f"成功保存 {len(stocks)} 只股票到数据库")
                return True
                
        except Exception as e:
            logger.error(f"保存股票数据到数据库失败: {e}")
            return False
    
    def download_stock_history(self, symbol: str, code: str, period: str = "daily") -> Optional[pd.DataFrame]:
        """下载单只股票的历史数据
        
        Args:
            symbol: 股票代码，如 000001.SZ
            code: 股票代码，如 000001
            period: 周期，默认daily
        """
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
                
                # 按日期排序
                stock_df = stock_df.sort_values('date')
                
                # 检查数据量
                if len(stock_df) < self.filter_config['min_history_records']:
                    logger.warning(f"股票 {symbol} 数据量不足: {len(stock_df)} < {self.filter_config['min_history_records']}")
                    return None
                
                logger.info(f"股票 {symbol} 数据下载成功，共 {len(stock_df)} 条记录")
                return stock_df
                
            except Exception as e:
                logger.error(f"下载 {symbol} 数据失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue
        
        return None
    
    def save_stock_prices(self, stock_data: pd.DataFrame, table_name: str = "stock_prices") -> bool:
        """保存股票价格数据到数据库"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                symbol = stock_data['symbol'].iloc[0]
                
                # 清空该股票的现有价格数据
                cursor.execute(f"DELETE FROM {table_name} WHERE symbol = ?", (symbol,))
                
                # 批量插入价格数据
                for _, row in stock_data.iterrows():
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO {table_name} 
                        (symbol, date, open, high, low, close, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']),
                        float(row['close'])  # adj_close 用 close 填充
                    ))
                
                conn.commit()
                logger.info(f"股票 {symbol} 价格数据保存成功，共 {len(stock_data)} 条记录")
                return True
                
        except Exception as e:
            logger.error(f"保存股票 {symbol} 价格数据失败: {e}")
            return False
    
    def save_stock_prices_daily(self, stock_data: pd.DataFrame) -> bool:
        """保存股票历史价格数据到prices_daily表（无adj_close列）"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                symbol = stock_data['symbol'].iloc[0]
                
                # 清空该股票的现有价格数据
                cursor.execute("DELETE FROM prices_daily WHERE symbol = ?", (symbol,))
                
                # 批量插入价格数据
                for _, row in stock_data.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO prices_daily 
                        (symbol, date, open, high, low, close, volume, amount, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']),
                        0.0,  # amount
                        'akshare'  # source
                    ))
                
                conn.commit()
                logger.info(f"股票 {symbol} 价格数据保存成功，共 {len(stock_data)} 条记录")
                return True
                
        except Exception as e:
            logger.error(f"保存股票 {symbol} 价格数据到prices_daily失败: {e}")
            return False
    
    def download_all_stock_prices(self, stocks: List[Dict], max_stocks: int = None):
        """下载所有股票的价格数据
        
        Args:
            stocks: 股票列表
            max_stocks: 最大下载数量，None表示下载全部
        """
        if max_stocks:
            stocks = stocks[:max_stocks]
        
        total = len(stocks)
        success_count = 0
        failed_stocks = []
        
        logger.info(f"开始下载 {total} 只股票的历史价格数据")
        
        for i, stock in enumerate(stocks, 1):
            symbol = stock['symbol']
            # 从symbol中提取股票代码（去掉.SZ/.SH后缀）
            code = symbol.split('.')[0]
            name = stock['name']
            
            logger.info(f"进度: {i}/{total} - {symbol} ({name})")
            
            # 下载历史数据
            stock_data = self.download_stock_history(symbol, code)
            
            if stock_data is not None and not stock_data.empty:
                # 保存到stock_prices表（有adj_close列）
                success1 = self.save_stock_prices(stock_data, "stock_prices")
                # 保存到prices_daily表（没有adj_close列）
                success2 = self.save_stock_prices_daily(stock_data)
                
                if success1 and success2:
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
        
        logger.info(f"价格数据下载完成！成功: {success_count}/{total}, 失败: {len(failed_stocks)}")
        
        if failed_stocks:
            logger.warning(f"下载失败的股票: {failed_stocks[:20]}")  # 只显示前20个
    
    def verify_database(self) -> Dict:
        """验证数据库完整性"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                result = {}
                
                # 统计股票数量
                cursor.execute("SELECT COUNT(*) FROM stocks")
                result['stock_count'] = cursor.fetchone()[0]
                
                # 按市场统计
                cursor.execute("""
                    SELECT market, board_type, COUNT(*) as count 
                    FROM stocks 
                    GROUP BY market, board_type 
                    ORDER BY count DESC
                """)
                result['market_distribution'] = cursor.fetchall()
                
                # 统计价格数据记录数
                cursor.execute("SELECT COUNT(*) FROM stock_prices")
                result['stock_prices_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM prices_daily")
                result['prices_daily_count'] = cursor.fetchone()[0]
                
                # 统计有价格数据的股票数量
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices")
                result['stocks_with_prices'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices_daily")
                result['stocks_with_daily_prices'] = cursor.fetchone()[0]
                
                # 统计日期范围
                cursor.execute("""
                    SELECT MIN(date), MAX(date), COUNT(DISTINCT date) 
                    FROM stock_prices
                """)
                stock_prices_date_info = cursor.fetchone()
                
                cursor.execute("""
                    SELECT MIN(date), MAX(date), COUNT(DISTINCT date) 
                    FROM prices_daily
                """)
                prices_daily_date_info = cursor.fetchone()
                
                result['date_ranges'] = {
                    'stock_prices': {
                        'min_date': stock_prices_date_info[0],
                        'max_date': stock_prices_date_info[1],
                        'distinct_dates': stock_prices_date_info[2]
                    },
                    'prices_daily': {
                        'min_date': prices_daily_date_info[0],
                        'max_date': prices_daily_date_info[1],
                        'distinct_dates': prices_daily_date_info[2]
                    }
                }
                
                # 获取数据量最多的股票
                cursor.execute("""
                    SELECT symbol, COUNT(*) as record_count 
                    FROM stock_prices 
                    GROUP BY symbol 
                    ORDER BY record_count DESC 
                    LIMIT 10
                """)
                result['top_stocks_by_records'] = cursor.fetchall()
                
                return result
                
        except Exception as e:
            logger.error(f"数据库验证失败: {e}")
            return {}

def main():
    """主函数"""
    builder = CompleteStockDatabaseBuilder()
    
    # 1. 初始化数据库
    logger.info("开始初始化数据库...")
    builder.init_database()
    
    # 2. 获取所有A股股票列表
    logger.info("开始获取A股股票列表...")
    all_stocks = builder.get_all_a_share_stocks()
    
    if not all_stocks:
        logger.error("无法获取股票列表")
        return
    
    # 3. 保存股票基本信息到数据库
    logger.info("保存股票基本信息到数据库...")
    success = builder.save_stocks_to_database(all_stocks)
    
    if not success:
        logger.error("保存股票基本信息失败")
        return
    
    # 4. 下载股票历史价格数据（限制数量以避免过长时间运行）
    logger.info("开始下载股票历史价格数据...")
    # 先下载前100只作为测试，可以根据需要调整
    builder.download_all_stock_prices(all_stocks, max_stocks=100)
    
    # 5. 验证数据库
    logger.info("验证数据库完整性...")
    verification_result = builder.verify_database()
    
    # 6. 输出结果
    print("\n" + "="*60)
    print("数据库构建完成报告")
    print("="*60)
    
    print(f"股票总数: {verification_result.get('stock_count', 0)}")
    print(f"stock_prices记录数: {verification_result.get('stock_prices_count', 0)}")
    print(f"prices_daily记录数: {verification_result.get('prices_daily_count', 0)}")
    print(f"有价格数据的股票数: {verification_result.get('stocks_with_prices', 0)}")
    
    print("\n市场分布:")
    for market, board_type, count in verification_result.get('market_distribution', []):
        print(f"  {market} - {board_type}: {count}只")
    
    date_ranges = verification_result.get('date_ranges', {})
    if date_ranges:
        stock_prices_range = date_ranges.get('stock_prices', {})
        print(f"\nstock_prices日期范围: {stock_prices_range.get('min_date', 'N/A')} 到 {stock_prices_range.get('max_date', 'N/A')}")
        
        prices_daily_range = date_ranges.get('prices_daily', {})
        print(f"prices_daily日期范围: {prices_daily_range.get('min_date', 'N/A')} 到 {prices_daily_range.get('max_date', 'N/A')}")
    
    print("\n数据量最多的前10只股票:")
    for symbol, count in verification_result.get('top_stocks_by_records', []):
        print(f"  {symbol}: {count}条记录")

if __name__ == "__main__":
    main()