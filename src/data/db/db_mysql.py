#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL数据库管理器 - SQLite到MySQL迁移版本
兼容原有的DatabaseManager接口，使用MySQL作为后端数据库
"""

import mysql.connector
from mysql.connector import Error, pooling
from contextlib import contextmanager
from typing import Optional, Iterable, Dict, Any, List
import pandas as pd
from datetime import datetime
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# 尝试加载项目根目录下的 .env 文件（若存在）
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)

# MySQL连接配置
MYSQL_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'stock_user'),
    'password': os.getenv('DB_PASSWORD', 'stock_pass'),
    'database': os.getenv('DB_NAME', 'stock_evaluation'),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
    'autocommit': True,
    'time_zone': '+8:00',
}

# 使用新的统一标准化器
from .symbol_standardizer import standardize_symbol, get_symbol_standardizer

# 保持向后兼容
try:
    from standardize_stock_codes import normalize_symbol as _legacy_normalize_symbol
except Exception:
    # 如果没有旧的标准化模块，使用新的标准化器
    def _legacy_normalize_symbol(symbol: str) -> str:
        """标准化股票代码（向后兼容）"""
        return standardize_symbol(symbol)

class MySQLDatabaseManager:
    """
    MySQL数据库管理器
    - 管理表结构：stocks、prices_daily、quotes_realtime、factors、model_metrics
    - 提供增量写入与基础查询能力
    - 兼容原有的DatabaseManager接口
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化MySQL数据库管理器
        
        Args:
            config: MySQL连接配置，如果为None则使用默认配置
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        self.config = config or MYSQL_CONFIG
        self.connection_pool = None
        self._init_connection_pool()
        self._init_schema()
        
    def _init_connection_pool(self):
        """初始化MySQL连接池"""
        try:
            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="stock_evaluation_pool",
                pool_size=10,
                pool_reset_session=1,
                **self.config
            )
            self.logger.info("MySQL连接池初始化成功")
        except Error as e:
            self.logger.error(f"MySQL连接池初始化失败: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        """
        获取数据库连接的上下文管理器
        使用连接池提高性能
        """
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _init_schema(self):
        """初始化数据库表结构"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            
            # stocks 表 - 股票基础信息
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL UNIQUE,
                    name VARCHAR(100),
                    market VARCHAR(10),
                    board_type VARCHAR(50),
                    exchange VARCHAR(50),
                    ah_pair VARCHAR(20),
                    industry VARCHAR(100),
                    market_cap DECIMAL(20,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_stocks_symbol (symbol),
                    INDEX idx_stocks_market (market),
                    INDEX idx_stocks_board_type (board_type),
                    INDEX idx_stocks_industry (industry)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # prices_daily 表 - 日线行情数据
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices_daily (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4),
                    volume BIGINT,
                    amount DECIMAL(20,2),
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_symbol_date (symbol, date),
                    INDEX idx_prices_daily_symbol (symbol),
                    INDEX idx_prices_daily_date (date),
                    INDEX idx_prices_daily_symbol_date (symbol, date),
                    INDEX idx_prices_daily_volume (volume)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # 为旧库增加复权列（若不存在）
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN open_qfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("open_qfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN high_qfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("high_qfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN low_qfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("low_qfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN close_qfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("close_qfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN open_hfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("open_hfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN high_hfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("high_hfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN low_hfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("low_hfq 列可能已存在或添加失败: %s", e)
            try:
                cursor.execute("ALTER TABLE prices_daily ADD COLUMN close_hfq DECIMAL(12,4) NULL")
            except Error as e:
                if e.errno != 1060:
                    self.logger.debug("close_hfq 列可能已存在或添加失败: %s", e)
            # 若历史表缺少 updated_at 列，则补充
            try:
                cursor.execute("""
                    ALTER TABLE prices_daily ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                """)
            except Error as e:
                # 1060 Duplicate column name means已存在
                if e.errno != 1060:
                    raise

            # 调整价格字段精度，防止溢出
            try:
                cursor.execute("""
                    ALTER TABLE prices_daily 
                        MODIFY COLUMN open DECIMAL(12,4),
                        MODIFY COLUMN high DECIMAL(12,4),
                        MODIFY COLUMN low DECIMAL(12,4),
                        MODIFY COLUMN close DECIMAL(12,4)
                """)
            except Error as e:
                # 1091 means column doesn't exist / not necessary; 1064 syntax etc.,忽略常见错误
                pass
            
            # quotes_realtime 表 - 实时行情快照
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quotes_realtime (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    ts DATETIME NOT NULL,
                    price DECIMAL(10,4),
                    change_val DECIMAL(10,4),
                    change_pct DECIMAL(8,4),
                    volume BIGINT,
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_quotes_realtime_symbol (symbol),
                    INDEX idx_quotes_realtime_ts (ts)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # factors 表 - 因子数据
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS factors (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    factor_name VARCHAR(100) NOT NULL,
                    value DECIMAL(20,6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_symbol_date_factor (symbol, date, factor_name),
                    INDEX idx_factors_symbol (symbol),
                    INDEX idx_factors_date (date),
                    INDEX idx_factors_factor_name (factor_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # model_metrics 表 - 模型评估指标
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    train_start DATE,
                    train_end DATE,
                    test_start DATE,
                    test_end DATE,
                    win_rate DECIMAL(5,4),
                    sharpe DECIMAL(10,6),
                    max_drawdown DECIMAL(8,4),
                    annual_return DECIMAL(8,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_model_metrics_model_name (model_name),
                    INDEX idx_model_metrics_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # predictions 表 - 股票预测结果
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    prob_up_30d DECIMAL(6,4),
                    expected_return_30d DECIMAL(10,6),
                    confidence DECIMAL(6,2),
                    score DECIMAL(8,2),
                    sentiment VARCHAR(20),
                    prediction TINYINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_symbol_date (symbol, date),
                    INDEX idx_predictions_symbol (symbol),
                    INDEX idx_predictions_date (date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            conn.commit()
            self.logger.info("数据库表结构初始化完成")
    
    def upsert_stocks(self, rows: Iterable[Dict[str, Any]]) -> int:
        """
        插入或更新股票基础信息
        使用MySQL的INSERT ... ON DUPLICATE KEY UPDATE语法替代SQLite的ON CONFLICT
        """
        rows = list(rows) if rows else []
        if not rows:
            return 0
            
        cleaned = self._normalize_and_filter_stocks(rows)
        if not cleaned:
            return 0
            
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # MySQL的UPSERT语法
                sql = """
                    INSERT INTO stocks (symbol, name, market, board_type, exchange, ah_pair, industry, market_cap)
                    VALUES (%(symbol)s, %(name)s, %(market)s, %(board_type)s, %(exchange)s, %(ah_pair)s, %(industry)s, %(market_cap)s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        market = VALUES(market),
                        board_type = VALUES(board_type),
                        exchange = VALUES(exchange),
                        ah_pair = VALUES(ah_pair),
                        industry = COALESCE(VALUES(industry), industry),
                        market_cap = COALESCE(VALUES(market_cap), market_cap),
                        updated_at = CURRENT_TIMESTAMP
                """
                
                cursor.executemany(sql, cleaned)
                conn.commit()
                
                affected_rows = cursor.rowcount
                self.logger.info(f"股票数据写入完成: {len(cleaned)}条记录, 影响行数: {affected_rows}")
                return affected_rows
                
        except Error as e:
            self.logger.error(f"股票数据写入失败: {e}")
            conn.rollback()
            raise
    
    def upsert_prices_daily(self, df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "date",
                           rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """
        插入或更新日线行情数据
        使用MySQL的INSERT ... ON DUPLICATE KEY UPDATE语法
        """
        if df is None or df.empty:
            return 0
            
        # 数据预处理和标准化
        data = self._prepare_price_data(df, symbol_col, date_col, rename_map, source)
        if data.empty:
            return 0
            
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # MySQL的UPSERT语法
                sql = """
                    INSERT INTO prices_daily (symbol, date, open, high, low, close, volume, amount, source)
                    VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(amount)s, %(source)s)
                    ON DUPLICATE KEY UPDATE
                        open = COALESCE(VALUES(open), open),
                        high = COALESCE(VALUES(high), high),
                        low = COALESCE(VALUES(low), low),
                        close = COALESCE(VALUES(close), close),
                        volume = COALESCE(VALUES(volume), volume),
                        amount = COALESCE(VALUES(amount), amount),
                        source = COALESCE(VALUES(source), source),
                        updated_at = CURRENT_TIMESTAMP
                """
                
                # 转换为字典列表
                records = data.to_dict('records')
                cursor.executemany(sql, records)
                conn.commit()
                
                affected_rows = cursor.rowcount
                self.logger.info(f"价格数据写入完成: {len(records)}条记录, 影响行数: {affected_rows}")
                return affected_rows
                
        except Error as e:
            self.logger.error(f"价格数据写入失败: {e}")
            conn.rollback()
            raise
    
    def insert_quotes_realtime(self, df: pd.DataFrame, symbol_col: str = "symbol", ts: Optional[str] = None,
                              rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """插入实时行情快照数据"""
        if df is None or df.empty:
            return 0
            
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 数据预处理
                data = df.copy()
                if rename_map:
                    data = data.rename(columns=rename_map)
                if symbol_col != "symbol":
                    data.rename(columns={symbol_col: "symbol"}, inplace=1)
                
                # 设置时间戳
                if ts is None:
                    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                
                # 重命名change列避免与SQL关键字冲突
                if 'change' in data.columns:
                    data.rename(columns={'change': 'change_val'}, inplace=1)
                
                sql = """
                    INSERT INTO quotes_realtime (symbol, ts, price, change_val, change_pct, volume, source)
                    VALUES (%(symbol)s, %(ts)s, %(price)s, %(change_val)s, %(change_pct)s, %(volume)s, %(source)s)
                """
                
                # 添加时间戳和源信息
                records = []
                for _, row in data.iterrows():
                    record = row.to_dict()
                    record['ts'] = ts
                    record['source'] = source
                    records.append(record)
                
                cursor.executemany(sql, records)
                conn.commit()
                
                inserted_rows = cursor.rowcount
                self.logger.info(f"实时行情数据写入完成: {len(records)}条记录")
                return inserted_rows
                
        except Error as e:
            self.logger.error(f"实时行情数据写入失败: {e}")
            conn.rollback()
            raise
    
    def get_latest_dates_by_symbol(self) -> Dict[str, str]:
        """获取每个symbol最新的日线日期"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, MAX(date) as max_date
                    FROM prices_daily
                    GROUP BY symbol
                """)
                
                result = {}
                for row in cursor.fetchall():
                    if row[1]:
                        result[row[0]] = row[1].strftime('%Y-%m-%d')
                
                return result
                
        except Error as e:
            self.logger.error(f"获取最新日期失败: {e}")
            raise
    
    def get_last_n_bars(self, symbols: list[str] = None, n: int = 2) -> pd.DataFrame:
        """获取每个symbol最近n根K线"""
        try:
            with self.get_conn() as conn:
                if symbols:
                    # 规范化股票代码
                    symbols_norm = self._normalize_symbols(symbols)
                    placeholders = ','.join(['%s'] * len(symbols_norm))
                    sql = f"""
                        SELECT symbol, date, open, high, low, close, volume 
                        FROM prices_daily 
                        WHERE symbol IN ({placeholders})
                        ORDER BY symbol, date
                    """
                    df = pd.read_sql_query(sql, conn, params=symbols_norm)
                else:
                    sql = """
                        SELECT symbol, date, open, high, low, close, volume 
                        FROM prices_daily 
                        ORDER BY symbol, date
                    """
                    df = pd.read_sql_query(sql, conn)
                
                if df.empty:
                    return df
                
                df['date'] = pd.to_datetime(df['date'])
                return df.groupby('symbol').tail(n)
                
        except Error as e:
            self.logger.error(f"获取K线数据失败: {e}")
            raise
    
    def list_symbols(self, markets: list[str] = None, limit: int = None) -> list[dict]:
        """获取股票列表"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                
                # 构建查询条件
                where_clauses = [
                    "symbol NOT LIKE '88%'",
                    "(board_type IS NULL OR board_type NOT IN ('指数','行业指数','板块','基金','ETF'))"
                ]
                params = []
                
                if markets:
                    placeholders = ','.join(['%s'] * len(markets))
                    where_clauses.insert(0, f"market IN ({placeholders})")
                    params.extend(markets)
                
                where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                order_limit_sql = " ORDER BY symbol"
                if limit:
                    order_limit_sql += " LIMIT %s"
                    params.append(limit)
                
                sql = f"SELECT symbol, name, market, ah_pair FROM stocks{where_sql}{order_limit_sql}"
                cursor.execute(sql, params)
                
                result = []
                for row in cursor.fetchall():
                    result.append({
                        'symbol': row[0],
                        'name': row[1],
                        'market': row[2],
                        'ah_pair': row[3]
                    })
                
                return result
                
        except Error as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return []
    
    def _normalize_and_filter_stocks(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化和过滤股票数据"""
        cleaned = []
        skipped = 0
        
        for s in rows:
            sym = s.get('symbol')
            ns = None
            
            # 尝试标准化股票代码
            if _normalize_symbol and sym:
                try:
                    _sym_try = str(sym)
                    if _sym_try.upper().endswith('.SS'):
                        _sym_try = _sym_try[:-3] + '.SH'
                    ns = _normalize_symbol(_sym_try)
                except Exception:
                    ns = None
            else:
                ns = None
            
            # 如果无法标准化，进行快速校验
            if ns is None:
                if isinstance(sym, str) and sym.upper().endswith((".SH", ".SZ", ".SS")):
                    num = sym.split(".")[0]
                    suf = sym.split(".")[-1].upper()
                    if suf == 'SS':
                        suf = 'SH'
                    
                    # 过滤规则
                    if num.startswith("88"):
                        skipped += 1
                        continue
                    if suf == 'BJ':
                        skipped += 1
                        continue
                    if suf == 'SH' and not (num.startswith("60") or num.startswith("688") or 
                                          num.startswith("689") or num.startswith("900")):
                        skipped += 1
                        continue
                    if suf == 'SZ' and not num.startswith(("000", "001", "002", "003", "300", "301", "200")):
                        skipped += 1
                        continue
                    
                    ns = f"{num}.{suf}"
                else:
                    skipped += 1
                    continue
            
            # 构建清理后的数据
            s2 = dict(s)
            s2['symbol'] = ns
            s2['industry'] = s.get('industry')
            s2['market_cap'] = s.get('market_cap')
            cleaned.append(s2)
        
        if skipped > 0:
            self.logger.info(f"股票数据过滤完成: 有效{len(cleaned)}条, 跳过{skipped}条")
        
        return cleaned
    
    def _prepare_price_data(self, df: pd.DataFrame, symbol_col: str, date_col: str, 
                           rename_map: Optional[Dict[str, str]], source: str) -> pd.DataFrame:
        """准备价格数据"""
        data = df.copy()
        if rename_map:
            data = data.rename(columns=rename_map)
        
        # 标准化列名
        if symbol_col != "symbol":
            data.rename(columns={symbol_col: "symbol"}, inplace=True)
        if date_col != "date":
            data.rename(columns={date_col: "date"}, inplace=True)
        
        # 检查必要列
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 日期标准化
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        
        # 添加缺失列
        if 'amount' not in data.columns:
            data['amount'] = None
        
        # 标准化股票代码并过滤
        rows = data[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]].to_dict("records")
        cleaned = []
        skipped = 0
        
        for r in rows:
            sym = r.get('symbol')
            ns = None
            
            # 标准化股票代码
            if _normalize_symbol and sym:
                try:
                    _sym_try = str(sym)
                    if _sym_try.upper().endswith('.SS'):
                        _sym_try = _sym_try[:-3] + '.SH'
                    ns = _normalize_symbol(_sym_try)
                except Exception:
                    ns = None
            else:
                ns = None
            
            if ns is None:
                if isinstance(sym, str) and sym.upper().endswith((".SH", ".SZ", ".SS")):
                    num = sym.split(".")[0]
                    suf = sym.split(".")[-1].upper()
                    if suf == 'SS':
                        suf = 'SH'
                    
                    # 过滤规则
                    if num.startswith("88"):
                        skipped += 1
                        continue
                    if suf == 'BJ':
                        skipped += 1
                        continue
                    if suf == 'SH' and not (num.startswith("60") or num.startswith("688") or 
                                          num.startswith("689") or num.startswith("900")):
                        skipped += 1
                        continue
                    if suf == 'SZ' and not num.startswith(("000", "001", "002", "003", "300", "301", "200")):
                        skipped += 1
                        continue
                    
                    ns = f"{num}.{suf}"
                else:
                    skipped += 1
                    continue
            
            r2 = dict(r)
            r2['symbol'] = ns
            r2['source'] = source
            cleaned.append(r2)
        
        if skipped > 0:
            self.logger.info(f"价格数据过滤完成: 有效{len(cleaned)}条, 跳过{skipped}条")
        
        return pd.DataFrame(cleaned)
    
    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        """标准化股票代码列表"""
        normalized = []
        for s in symbols:
            try:
                if isinstance(s, str) and s.upper().endswith('.SS'):
                    normalized.append(s[:-3] + '.SH')
                else:
                    normalized.append(s)
            except Exception:
                normalized.append(s)
        
        # 去重
        return list(dict.fromkeys(normalized))
    
    def execute_query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行自定义查询并返回DataFrame"""
        try:
            with self.get_conn() as conn:
                return pd.read_sql_query(sql, conn, params=params)
        except Error as e:
            self.logger.error(f"查询执行失败: {e}")
            raise
    
    def execute_update(self, sql: str, params: tuple = None) -> int:
        """执行更新/删除操作"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                conn.commit()
                return cursor.rowcount
        except Error as e:
            self.logger.error(f"更新执行失败: {e}")
            conn.rollback()
            raise


# 向后兼容的别名
DatabaseManager = MySQLDatabaseManager