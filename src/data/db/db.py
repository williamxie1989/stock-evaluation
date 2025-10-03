import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

# 数据库文件路径
DB_PATH = os.path.join(os.path.dirname(__file__), "../../../data/stock_data.db")

# 使用新的统一标准化器
from .symbol_standardizer import standardize_symbol, get_symbol_standardizer

# 保持向后兼容
try:
    from standardize_stock_codes import normalize_symbol as legacy_normalize
except ImportError:
    # 如果没有旧的标准化模块，使用新的标准化器
    def normalize_symbol(symbol: str) -> str:
        """标准化股票代码（向后兼容）"""
        return standardize_symbol(symbol)


class DatabaseManager:
    """
    SQLite数据库管理器
    管理表结构：stocks、prices_daily、quotes_realtime、factors、model_metrics
    提供增量写入与基础查询能力
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """初始化数据库表结构"""
        with self.get_conn() as conn:
            cur = conn.cursor()
            
            # stocks 表 - 股票基本信息
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,  -- 股票代码（A股代码）
                    name TEXT,             -- 股票名称
                    market TEXT,           -- 市场标识，如 SH/SZ
                    board_type TEXT,       -- 板块类型（主板、创业板、科创板等）
                    exchange TEXT,         -- 交易所名称
                    ah_pair TEXT,          -- 若有，对应另一市场代码，例如 H 股代码
                    industry TEXT,         -- 行业
                    market_cap REAL,       -- 总市值（元）
                    UNIQUE(symbol)
                )
            """)
            
            # prices_daily 表 - 日线行情
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prices_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,    -- YYYY-MM-DD
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    source TEXT,           -- 数据来源（akshare接口名等）
                    UNIQUE(symbol, date)
                )
            """)
            
            # quotes_realtime 表 - 实时行情快照
            cur.execute("""
                CREATE TABLE IF NOT EXISTS quotes_realtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,      -- 时间戳 ISO 格式
                    price REAL,
                    change REAL,
                    change_pct REAL,
                    volume REAL,
                    source TEXT
                )
            """)
            
            # factors 表 - 因子数据
            cur.execute("""
                CREATE TABLE IF NOT EXISTS factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    value REAL,
                    UNIQUE(symbol, date, factor_name)
                )
            """)
            
            # model_metrics 表 - 模型与回测指标
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    train_start TEXT,
                    train_end TEXT,
                    test_start TEXT,
                    test_end TEXT,
                    win_rate REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    annual_return REAL,
                    created_at TEXT
                )
            """)
            
            # predictions 表 - 股票预测结果
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    prob_up_30d REAL,
                    expected_return_30d REAL,
                    confidence REAL,
                    score REAL,
                    sentiment TEXT,
                    prediction INTEGER,
                    UNIQUE(symbol, date)
                )
            """)
            
            # 执行迁移并提交
            self._migrate_database(cur)
            conn.commit()

    def _migrate_database(self, cur):
        """数据库迁移：为现有表添加新字段"""
        try:
            # 检查并添加新字段
            tables = ['stocks', 'prices_daily', 'quotes_realtime', 'factors', 'model_metrics']
            for table in tables:
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    existing_cols = {row[1] for row in cur.fetchall()}
                    
                    # 为stocks表添加updated_at字段
                    if table == 'stocks' and 'updated_at' not in existing_cols:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN updated_at TEXT")
                        logging.info(f"为表 {table} 添加字段 updated_at")
                    
                    # 这里可以添加其他新字段的迁移逻辑
                    # 例如：if 'new_column' not in existing_cols:
                    #         cur.execute(f"ALTER TABLE {table} ADD COLUMN new_column TEXT")
                    
                except Exception as e:
                    logging.warning(f"迁移表 {table} 时出错: {e}")
                    
        except Exception as e:
            logging.warning(f"数据库迁移失败: {e}")

    def insert_stock(self, symbol: str, name: str = None, market: str = None, 
                    board_type: str = None, exchange: str = None, 
                    ah_pair: str = None, industry: str = None, 
                    market_cap: float = None) -> bool:
        """插入或更新股票信息"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO stocks 
                    (symbol, name, market, board_type, exchange, ah_pair, industry, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, name, market, board_type, exchange, ah_pair, industry, market_cap))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"插入股票信息失败 {symbol}: {e}")
            return False

    def insert_price_daily(self, symbol: str, date: str, open_price: float, 
                          high: float, low: float, close: float, 
                          volume: float, amount: float = None, source: str = None) -> bool:
        """插入日线行情数据"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO prices_daily 
                    (symbol, date, open, high, low, close, volume, amount, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, date, open_price, high, low, close, volume, amount, source))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"插入日线数据失败 {symbol} {date}: {e}")
            return False

    def insert_quote_realtime(self, symbol: str, timestamp: str, price: float,
                             change: float, change_pct: float, volume: float, source: str = None) -> bool:
        """插入实时行情数据"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO quotes_realtime 
                    (symbol, ts, price, change, change_pct, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timestamp, price, change, change_pct, volume, source))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"插入实时行情失败 {symbol}: {e}")
            return False

    def insert_factor(self, symbol: str, date: str, factor_name: str, value: float) -> bool:
        """插入因子数据"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO factors 
                    (symbol, date, factor_name, value)
                    VALUES (?, ?, ?, ?)
                """, (symbol, date, factor_name, value))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"插入因子数据失败 {symbol} {date} {factor_name}: {e}")
            return False

    def insert_model_metrics(self, model_name: str, train_start: str, train_end: str,
                           test_start: str, test_end: str, win_rate: float,
                           sharpe: float, max_drawdown: float, annual_return: float) -> bool:
        """插入模型指标数据"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                created_at = datetime.now().isoformat()
                cur.execute("""
                    INSERT INTO model_metrics 
                    (model_name, train_start, train_end, test_start, test_end, 
                     win_rate, sharpe, max_drawdown, annual_return, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (model_name, train_start, train_end, test_start, test_end,
                      win_rate, sharpe, max_drawdown, annual_return, created_at))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"插入模型指标失败 {model_name}: {e}")
            return False

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """获取股票信息"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logging.error(f"获取股票信息失败 {symbol}: {e}")
            return None

    def get_price_daily(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取日线行情数据"""
        try:
            symbol = normalize_symbol(symbol)
            with self.get_conn() as conn:
                query = "SELECT * FROM prices_daily WHERE symbol = ?"
                params = [symbol]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                return df if not df.empty else None
        except Exception as e:
            logging.error(f"获取日线数据失败 {symbol}: {e}")
            return None

    def get_all_stocks(self) -> Optional[pd.DataFrame]:
        """获取所有股票列表"""
        try:
            with self.get_conn() as conn:
                df = pd.read_sql_query("SELECT * FROM stocks ORDER BY symbol", conn)
                return df if not df.empty else None
        except Exception as e:
            logging.error(f"获取股票列表失败: {e}")
            return None

    def delete_old_realtime_quotes(self, days: int = 7) -> bool:
        """删除旧的实时行情数据"""
        try:
            cutoff_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM quotes_realtime WHERE ts < ?", (cutoff_date,))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"删除旧实时行情数据失败: {e}")
            return False

    def execute_sql(self, sql: str, params: tuple = None) -> Optional[List]:
        """执行自定义SQL查询"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                
                if sql.strip().upper().startswith("SELECT"):
                    return cur.fetchall()
                else:
                    conn.commit()
                    return True
        except Exception as e:
            logging.error(f"执行SQL失败: {e}")
            return None

    def list_symbols(self, markets: list[str] = None, market: str = None, 
                    board_type: str = None, limit: int = None) -> list[dict]:
        """获取股票列表"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                
                # 构建查询条件
                where_clauses = [
                    "symbol NOT LIKE '88%'",
                    "(board_type IS NULL OR board_type NOT IN ('指数','行业指数','板块','基金','ETF'))"
                ]
                params = []
                
                # 处理markets参数（列表）
                if markets:
                    placeholders = ','.join(['?' for _ in markets])
                    where_clauses.append(f"market IN ({placeholders})")
                    params.extend(markets)
                # 处理market参数（单个）
                elif market:
                    where_clauses.append("market = ?")
                    params.append(market)
                
                # 处理board_type参数
                if board_type:
                    where_clauses.append("board_type = ?")
                    params.append(board_type)
                
                where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                order_limit_sql = " ORDER BY symbol"
                if limit:
                    order_limit_sql += " LIMIT ?"
                    params.append(limit)
                
                sql = f"SELECT symbol, name, market, ah_pair, board_type FROM stocks{where_sql}{order_limit_sql}"
                cur.execute(sql, params)
                
                result = []
                for row in cur.fetchall():
                    result.append({
                        'symbol': row[0],
                        'name': row[1],
                        'market': row[2],
                        'ah_pair': row[3],
                        'board_type': row[4] if len(row) > 4 else None
                    })
                
                return result
                
        except Exception as e:
            logging.error(f"获取股票列表失败: {e}")
            return []

    def upsert_stocks(self, rows: list) -> int:
        """批量插入或更新股票信息"""
        if not rows:
            return 0
            
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                
                # 过滤和标准化股票数据
                cleaned_rows = []
                for row in rows:
                    symbol = row.get('symbol')
                    if not symbol:
                        continue
                    
                    # 标准化股票代码
                    normalized_symbol = normalize_symbol(symbol)
                    if not normalized_symbol:
                        continue
                    
                    # 过滤掉无效的股票代码
                    num = normalized_symbol.split(".")[0]
                    suf = normalized_symbol.split(".")[-1]
                    
                    # 过滤规则
                    if num.startswith("88"):
                        continue
                    if suf == 'BJ':  # 跳过北交所
                        continue
                    if suf == 'SH' and not (num.startswith("60") or num.startswith("688") or 
                                          num.startswith("689") or num.startswith("900")):
                        continue
                    if suf == 'SZ' and not num.startswith(("000", "001", "002", "003", "300", "301", "200")):
                        continue
                    
                    cleaned_row = {
                        'symbol': normalized_symbol,
                        'name': row.get('name'),
                        'market': row.get('market'),
                        'board_type': row.get('board_type'),
                        'exchange': row.get('exchange'),
                        'ah_pair': row.get('ah_pair'),
                        'industry': row.get('industry'),
                        'market_cap': row.get('market_cap')
                    }
                    cleaned_rows.append(cleaned_row)
                
                if not cleaned_rows:
                    return 0
                
                # SQLite的UPSERT语法
                for row in cleaned_rows:
                    cur.execute("""
                        INSERT OR REPLACE INTO stocks 
                        (symbol, name, market, board_type, exchange, ah_pair, industry, market_cap, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """, (row['symbol'], row['name'], row['market'], row['board_type'], 
                          row['exchange'], row['ah_pair'], row['industry'], row['market_cap']))
                
                conn.commit()
                
                affected_rows = len(cleaned_rows)
                logging.info(f"股票数据写入完成: {affected_rows}条记录")
                return affected_rows
                
        except Exception as e:
            logging.error(f"股票数据写入失败: {e}")
            return 0

    def get_market_summary(self) -> dict:
        """获取市场概览统计"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                
                # 按市场统计
                cur.execute("SELECT market, COUNT(*) as count FROM stocks GROUP BY market ORDER BY count DESC")
                market_stats = []
                for row in cur.fetchall():
                    market_stats.append({
                        'market': row[0],
                        'count': row[1]
                    })
                
                # 按板块统计
                cur.execute("SELECT board_type, COUNT(*) as count FROM stocks GROUP BY board_type ORDER BY count DESC")
                board_stats = []
                for row in cur.fetchall():
                    board_stats.append({
                        'board_type': row[0] if row[0] else '未知',
                        'count': row[1]
                    })
                
                # 按交易所统计
                cur.execute("SELECT exchange, COUNT(*) as count FROM stocks GROUP BY exchange ORDER BY count DESC")
                exchange_stats = []
                for row in cur.fetchall():
                    exchange_stats.append({
                        'exchange': row[0] if row[0] else '未知',
                        'count': row[1]
                    })
                
                # 总数统计
                cur.execute("SELECT COUNT(*) as total FROM stocks")
                total_count = cur.fetchone()[0]
                
                result = {
                    'total_stocks': int(total_count),
                    'by_market': market_stats,
                    'by_board_type': board_stats,
                    'by_exchange': exchange_stats,
                    'last_updated': datetime.now().isoformat()
                }
                
                return result
                
        except Exception as e:
            logging.error(f"获取市场概览失败: {e}")
            return {'total_stocks': 0, 'by_market': [], 'by_board_type': [], 'by_exchange': []}

    def get_available_markets(self) -> dict:
        """获取可用的市场列表"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                
                # 查询各市场的股票数量
                cur.execute("""
                    SELECT market, COUNT(*) as count, exchange 
                    FROM stocks 
                    GROUP BY market, exchange 
                    ORDER BY count DESC
                """)
                
                markets = {}
                for row in cur.fetchall():
                    market_code = row[0]
                    count = row[1]
                    exchange = row[2] if len(row) > 2 else None
                    
                    # 检查数据新鲜度（简化版）
                    cur.execute("""
                        SELECT MAX(pd.date) as latest_date
                        FROM stocks s
                        LEFT JOIN prices_daily pd ON s.symbol = pd.symbol
                        WHERE s.market = ?
                    """, (market_code,))
                    
                    freshness_row = cur.fetchone()
                    latest_date = freshness_row[0] if freshness_row and freshness_row[0] else None
                    
                    is_fresh = False
                    if latest_date:
                        try:
                            latest_date_obj = datetime.strptime(latest_date, '%Y-%m-%d').date()
                            today = datetime.now().date()
                            days_behind = (today - latest_date_obj).days
                            is_fresh = days_behind <= 1
                        except:
                            is_fresh = False
                    
                    markets[market_code] = {
                        'market_code': market_code,
                        'exchange': exchange,
                        'stock_count': int(count),
                        'data_freshness': {
                            'latest_date': latest_date,
                            'days_behind': None,
                            'is_fresh': bool(is_fresh)
                        },
                        'available': bool(count > 0 and is_fresh),
                        'enabled': 1
                    }
                
                return {
                    'success': 1,
                    'markets': markets,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"获取可用市场失败: {e}")
            return {
                'success': 0,
                'error': str(e)
            }

    def get_latest_dates_by_symbol(self) -> dict:
        """获取每个symbol最新的日线日期"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT symbol, MAX(date) as max_date
                    FROM prices_daily
                    GROUP BY symbol
                """)
                
                result = {}
                for row in cur.fetchall():
                    if row[1]:
                        result[row[0]] = row[1]
                
                return result
                
        except Exception as e:
            logging.error(f"获取最新日期失败: {e}")
            return {}


if __name__ == "__main__":
    # 测试数据库管理器
    db = DatabaseManager()
    print("数据库管理器初始化完成")
    
    # 测试插入股票信息
    db.insert_stock("000001", "平安银行", "SZ", "主板", "深交所")
    print("股票信息插入测试完成")
    
    # 测试查询
    stock_info = db.get_stock_info("000001")
    print(f"股票信息: {stock_info}")