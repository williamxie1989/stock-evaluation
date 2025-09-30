#!/usr/bin/env python3
"""
统一数据库管理器
兼容SQLite和MySQL，提供统一的数据库操作接口
支持数据库类型切换和连接池管理
"""

import sqlite3
import logging
from typing import Optional, List, Dict, Any, Union, ContextManager
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
import os

# 导入字段映射工具
from ..field_mapping import FieldMapper

# 导入配置管理器
try:
    # 确保在导入配置管理器之前设置环境变量
    import os
    if 'DB_TYPE' not in os.environ:
        os.environ['DB_TYPE'] = 'mysql'  # 默认使用MySQL
    
    from database_config import get_database_config, get_db_connection
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # 备用配置 - 使用正确的密码
    MYSQL_CONFIG = {
        'host': 'localhost', 'port': 3306, 'user': 'stock_user',
        'password': 'stock_password', 'database': 'stock_evaluation',
        'charset': 'utf8mb4', 'autocommit': True
    }
    SQLITE_CONFIG = {'database': 'stock_data.sqlite3', 'timeout': 30}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedDatabaseManager:
    """
    统一数据库管理器
    兼容SQLite和MySQL，提供统一的数据库操作接口
    """
    
    def __init__(self, db_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化统一数据库管理器
        
        Args:
            db_type: 数据库类型 ('mysql' 或 'sqlite')，如果为None则使用配置管理器
            config: 数据库配置字典，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        
        # 使用配置管理器或备用配置
        if CONFIG_AVAILABLE:
            self.config_manager = get_database_config()
            if db_type:
                self.config_manager.set_db_type(db_type)
            self.db_type = self.config_manager.db_type
            self.config = self.config_manager.get_config()
        else:
            # 检查环境变量
            import os
            env_db_type = os.getenv('DB_TYPE')
            self.db_type = db_type or env_db_type or 'sqlite'
            self.config = config or (MYSQL_CONFIG if self.db_type == 'mysql' else SQLITE_CONFIG)
            self.config_manager = None
        
        self._connection_pool = None
        self._init_connection_pool()
        
        logger.info(f"统一数据库管理器初始化完成，类型: {self.db_type}")
    
    def _init_connection_pool(self):
        """初始化连接池"""
        if self.db_type == 'mysql':
            if CONFIG_AVAILABLE and self.config_manager:
                self._connection_pool = self.config_manager.connection_pool
            else:
                # 备用MySQL连接池初始化
                try:
                    import mysql.connector
                    from mysql.connector import pooling
                    pool_config = {
                        'pool_name': 'unified_db_pool',
                        'pool_size': 10,
                        'pool_reset_session': True,
                        **self.config
                    }
                    self._connection_pool = pooling.MySQLConnectionPool(**pool_config)
                    logger.info("MySQL连接池初始化成功")
                except Exception as e:
                    logger.error(f"MySQL连接池初始化失败: {e}")
                    self._connection_pool = None
        else:
            # SQLite不需要连接池
            self._connection_pool = None
            logger.info("SQLite连接初始化成功")
    
    @contextmanager
    def get_connection(self) -> ContextManager[Any]:
        """
        获取数据库连接的上下文管理器
        
        Yields:
            数据库连接对象
        """
        conn = None
        try:
            if self.db_type == 'mysql' and self._connection_pool:
                conn = self._connection_pool.get_connection()
            elif self.db_type == 'mysql':
                import mysql.connector
                conn = mysql.connector.connect(**self.config)
            else:  # sqlite
                conn = sqlite3.connect(**self.config)
                conn.row_factory = sqlite3.Row  # 使查询结果可以按列名访问
            
            yield conn
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        执行查询语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # 获取列名
                if self.db_type == 'mysql':
                    columns = [desc[0] for desc in cursor.description]
                else:  # sqlite
                    columns = [desc[0] for desc in cursor.description]
                
                # 转换结果为字典列表
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
            finally:
                cursor.close()
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        执行更新语句（INSERT, UPDATE, DELETE）
        
        Args:
            query: SQL更新语句
            params: 语句参数
            
        Returns:
            影响的行数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # 提交事务
                if self.db_type == 'sqlite':
                    conn.commit()
                
                return cursor.rowcount
            finally:
                cursor.close()
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        批量执行语句
        
        Args:
            query: SQL语句
            params_list: 参数列表
            
        Returns:
            总影响行数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, params_list)
                
                # 提交事务
                if self.db_type == 'sqlite':
                    conn.commit()
                
                return cursor.rowcount
            finally:
                cursor.close()
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            连接是否成功
        """
        try:
            with self.get_connection() as conn:
                # 执行一个简单的查询来测试连接
                cursor = conn.cursor()
                if self.db_type == 'mysql':
                    cursor.execute("SELECT 1")
                else:
                    cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                return result is not None and result[0] == 1
        except Exception as e:
            self.logger.error(f"数据库连接测试失败: {e}")
            return False

    def upsert_prices_daily(self, df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "date",
                             rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """
        插入或更新日线行情数据到 prices_daily 表，以兼容旧版 API。

        该实现复用 ``insert_dataframe`` 方法：
        1. 对输入 ``DataFrame`` 做基础字段标准化与去重处理；
        2. 对 MySQL 数据库， ``insert_dataframe`` 内部使用 ``INSERT IGNORE``，可自动忽略重复主键；
        3. 对 SQLite 数据库，先行去重后插入，避免唯一键冲突。

        Args:
            df: 待写入的数据 ``DataFrame``。
            symbol_col: 股票代码列名，默认为 ``symbol``。
            date_col: 交易日期列名，默认为 ``date``。
            rename_map: 额外的列重命名映射，可选。
            source: 数据来源标识，将写入 ``source`` 列（若存在）。

        Returns:
            实际插入的行数。
        """
        if df is None or df.empty:
            self.logger.warning("upsert_prices_daily: 传入 DataFrame 为空，跳过处理")
            return 0

        data = df.copy()

        # 可选的列重命名映射
        if rename_map:
            data = data.rename(columns=rename_map)

        # 统一列名
        if symbol_col != "symbol" and symbol_col in data.columns:
            data.rename(columns={symbol_col: "symbol"}, inplace=True)
        if date_col != "date" and date_col in data.columns:
            data.rename(columns={date_col: "date"}, inplace=True)

        # 填充数据来源列
        if source:
            if "source" not in data.columns:
                data["source"] = source
            else:
                data["source"].fillna(source, inplace=True)

        # 日期格式统一
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # 去重避免主键冲突
        if {"symbol", "date"}.issubset(data.columns):
            data = data.drop_duplicates(subset=["symbol", "date"], keep="last")

        inserted_rows = self.insert_dataframe(data, "prices_daily", if_exists="append")
        self.logger.info(f"upsert_prices_daily: 实际插入 {inserted_rows} 行")
        return inserted_rows
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        
        Returns:
            数据库信息字典
        """
        try:
            if self.db_type == 'mysql':
                result = self.execute_query("SELECT VERSION()")
                version = result[0]['VERSION()'] if result else 'Unknown'
                return {
                    'type': 'mysql',
                    'version': version,
                    'host': self.config.get('host', 'localhost'),
                    'database': self.config.get('database', 'unknown')
                }
            else:
                result = self.execute_query("SELECT sqlite_version()")
                version = result[0]['sqlite_version()'] if result else 'Unknown'
                return {
                    'type': 'sqlite',
                    'version': version,
                    'database': self.config.get('database', 'unknown')
                }
        except Exception as e:
            self.logger.error(f"获取数据库信息失败: {e}")
            return {'type': self.db_type, 'version': 'Unknown', 'error': str(e)}
    
    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        Args:
            table_name: 表名
            
        Returns:
            表是否存在
        """
        try:
            if self.db_type == 'mysql':
                result = self.execute_query(
                    "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                    (self.config.get('database'), table_name)
                )
            else:
                result = self.execute_query(
                    "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
            return result[0]['count'] > 0 if result else False
        except Exception as e:
            self.logger.error(f"检查表存在性失败: {e}")
            return False
    
    def get_row_count(self, table_name: str) -> int:
        """
        获取表的行数
        
        Args:
            table_name: 表名
            
        Returns:
            行数
        """
        try:
            result = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            return result[0]['count'] if result else 0
        except Exception as e:
            self.logger.error(f"获取表行数失败: {e}")
            return 0
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            
        Returns:
            表结构信息列表
        """
        try:
            if self.db_type == 'mysql':
                return self.execute_query(f"DESCRIBE {table_name}")
            else:
                return self.execute_query(f"PRAGMA table_info({table_name})")
        except Exception as e:
            self.logger.error(f"获取表结构信息失败: {e}")
            return []
    
    def switch_database(self, db_type: str, config: Optional[Dict[str, Any]] = None):
        """
        切换数据库类型
        
        Args:
            db_type: 新的数据库类型
            config: 新的数据库配置
        """
        self.db_type = db_type
        self.config = config or (MYSQL_CONFIG if db_type == 'mysql' else SQLITE_CONFIG)
        self._init_connection_pool()
        self.logger.info(f"数据库切换完成，新类型: {db_type}")
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            
        Returns:
            表结构信息列表
        """
        if self.db_type == 'mysql':
            query = f"DESCRIBE {table_name}"
        else:  # sqlite
            query = f"PRAGMA table_info({table_name})"
        
        return self.execute_query(query)
    
    def get_tables(self) -> List[str]:
        """
        获取所有表名
        
        Returns:
            表名列表
        """
        if self.db_type == 'mysql':
            query = "SHOW TABLES"
            results = self.execute_query(query)
            # MySQL返回的表名在第一个列中
            return [list(row.values())[0] for row in results]
        else:  # sqlite
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            results = self.execute_query(query)
            return [row['name'] for row in results]
    
    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        Args:
            table_name: 表名
            
        Returns:
            表是否存在
        """
        tables = self.get_tables()
        return table_name in tables
    
    def get_row_count(self, table_name: str) -> int:
        """
        获取表的行数
        
        Args:
            table_name: 表名
            
        Returns:
            行数
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        results = self.execute_query(query)
        return results[0]['count'] if results else 0
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str = 'append', batch_size: int = 1000) -> int:
        """
        插入DataFrame数据到数据库
        
        Args:
            df: 要插入的数据框
            table_name: 目标表名
            if_exists: 如果表存在时的处理方式 ('fail', 'replace', 'append')
            batch_size: 批量插入大小
            
        Returns:
            插入的行数
        """
        if df.empty:
            logger.warning("DataFrame为空，无需插入")
            return 0
        
        # 首先进行字段映射标准化
        df_normalized = FieldMapper.normalize_fields(df, table_name)
        
        # 确保包含所有必需字段
        df_normalized = FieldMapper.ensure_required_fields(df_normalized, table_name)
        
        # 验证数据结构
        if not FieldMapper.validate_data_structure(df_normalized, table_name):
            logger.error(f"表 {table_name} 数据结构验证失败，无法插入数据")
            return 0
        
        total_inserted = 0
        
        # 分批处理
        for i in range(0, len(df_normalized), batch_size):
            batch_df = df_normalized.iloc[i:i+batch_size]
            
            # 构建插入语句
            columns = list(batch_df.columns)
            placeholders = ', '.join(['?' if self.db_type == 'sqlite' else '%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            if self.db_type == 'mysql':
                insert_query = f"INSERT IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            else:
                insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # 转换数据为参数列表
            params_list = [tuple(row) for row in batch_df.values]
            
            # 执行批量插入
            inserted = self.execute_many(insert_query, params_list)
            total_inserted += inserted
            
            logger.info(f"批量插入进度: {min(i+batch_size, len(df_normalized))}/{len(df_normalized)}")
        
        logger.info(f"DataFrame插入完成，总计: {total_inserted} 行")
        return total_inserted
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        
        Returns:
            数据库信息字典
        """
        info = {
            'db_type': self.db_type,
            'config': self.config,
            'tables': self.get_tables(),
            'connection_pool_active': self._connection_pool is not None,
        }
        
        if self.db_type == 'mysql':
            try:
                version_result = self.execute_query("SELECT VERSION() as version")
                info['version'] = version_result[0]['version'] if version_result else 'Unknown'
            except:
                info['version'] = 'Unknown'
        else:  # sqlite
            try:
                version_result = self.execute_query("SELECT sqlite_version() as version")
                info['version'] = version_result[0]['version'] if version_result else 'Unknown'
            except:
                info['version'] = 'Unknown'
        
        return info
    
    def switch_database(self, db_type: str, config: Optional[Dict[str, Any]] = None):
        """
        切换数据库类型
        
        Args:
            db_type: 新的数据库类型 ('mysql' 或 'sqlite')
            config: 新的配置（可选）
        """
        logger.info(f"切换数据库类型: {self.db_type} -> {db_type}")
        
        # 关闭现有连接池
        self._connection_pool = None
        
        # 更新配置
        self.db_type = db_type
        if config:
            self.config = config
        elif CONFIG_AVAILABLE and self.config_manager:
            self.config_manager.set_db_type(db_type)
            self.config = self.config_manager.get_config()
        else:
            self.config = MYSQL_CONFIG if db_type == 'mysql' else SQLITE_CONFIG
        
        # 重新初始化连接池
        self._init_connection_pool()
        
        logger.info(f"数据库切换完成，新类型: {self.db_type}")
    
    async def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票历史数据DataFrame
        """
        try:
            # 根据数据库类型选择合适的参数占位符
            if self.db_type == 'mysql':
                query = """
                    SELECT symbol, date, open, high, low, close, volume, amount, source
                    FROM prices_daily 
                    WHERE symbol = %s AND date >= %s AND date <= %s
                    ORDER BY date ASC
                """
            else:  # sqlite
                query = """
                    SELECT symbol, date, open, high, low, close, volume, amount, source
                    FROM prices_daily 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date ASC
                """
            params = (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            results = self.execute_query(query, params)
            
            if not results:
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def list_symbols(self, markets: List[str] = None, market: str = None, 
                    board_type: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取股票列表
        
        Args:
            markets: 市场代码列表 (可选)
            market: 市场代码 (可选，单个)
            board_type: 板块类型 (可选)
            limit: 限制返回数量 (可选)
            
        Returns:
            股票列表
        """
        try:
            # 构建查询条件
            where_clauses = [
                "symbol NOT LIKE '88%'",
                "(board_type IS NULL OR board_type NOT IN ('指数','行业指数','板块','基金','ETF'))"
            ]
            params = []
            
            # 处理markets参数（列表）
            if markets:
                if self.db_type == 'mysql':
                    placeholders = ','.join(['%s'] * len(markets))
                else:  # sqlite
                    placeholders = ','.join(['?'] * len(markets))
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
            
            # 根据数据库类型选择合适的参数占位符
            if self.db_type == 'mysql':
                query = f"SELECT symbol, name, market, ah_pair, board_type FROM stocks{where_sql}{order_limit_sql}"
                # 修复MySQL参数占位符
                if params:
                    query = query.replace('?', '%s')
            else:  # sqlite
                query = f"SELECT symbol, name, market, ah_pair, board_type FROM stocks{where_sql}{order_limit_sql}"
            
            results = self.execute_query(query, tuple(params) if params else None)
            
            # 格式化结果
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'symbol': row.get('symbol'),
                    'name': row.get('name'),
                    'market': row.get('market'),
                    'ah_pair': row.get('ah_pair'),
                    'board_type': row.get('board_type')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return []
    
    async def get_stock_list(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取股票列表 - 兼容UnifiedDataAccessLayer的接口
        
        Args:
            market: 市场代码 (可选)
            
        Returns:
            股票列表
        """
        try:
            # 调用现有的list_symbols方法，但使用不同的参数格式
            if market:
                return self.list_symbols(market=market)
            else:
                return self.list_symbols()
                
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return []
    
    async def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        保存股票数据到数据库
        
        Args:
            symbol: 股票代码
            data: 股票数据DataFrame
            
        Returns:
            是否成功
        """
        try:
            if data.empty:
                return True
            
            # 确保数据格式正确
            if 'symbol' not in data.columns:
                data['symbol'] = symbol
            
            # 重置索引以包含日期列
            if data.index.name == 'date':
                data_reset = data.reset_index()
            else:
                data_reset = data.copy()
            
            # 字段标准化和必需字段检查
            data_reset = FieldMapper.normalize_fields(data_reset, 'prices_daily')
            data_reset = FieldMapper.ensure_required_fields(data_reset, 'prices_daily')

            # 确保date列存在且格式正确
            if 'date' in data_reset.columns:
                data_reset['date'] = pd.to_datetime(data_reset['date']).dt.strftime('%Y-%m-%d')

            # 去重，避免重复主键冲突
            if 'symbol' in data_reset.columns and 'date' in data_reset.columns:
                data_reset = data_reset.drop_duplicates(subset=['symbol', 'date'], keep='last')

            # 仅保留数据库允许的列，避免未知列错误
            allowed_columns = FieldMapper.REQUIRED_FIELDS['prices_daily']
            data_reset = data_reset[[col for col in data_reset.columns if col in allowed_columns]]

            # 插入数据
            inserted_rows = self.insert_dataframe(data_reset, 'prices_daily', if_exists='append')
            logger.info(f"Saved {inserted_rows} rows of data for {symbol}")
            
            return inserted_rows > 0
            
        except Exception as e:
            logger.error(f"Error saving stock data for {symbol}: {e}")
            return False
    
    async def upsert_stocks(self, stocks: List[Dict[str, Any]]) -> bool:
        """
        更新或插入股票列表
        
        Args:
            stocks: 股票列表
            
        Returns:
            是否成功
        """
        try:
            if not stocks:
                return True
            
            # 转换为DataFrame
            df = pd.DataFrame(stocks)
            
            # 先尝试插入，如果违反唯一约束则更新
            for _, stock in df.iterrows():
                symbol = stock.get('symbol')
                if not symbol:
                    continue
                
                # 检查股票是否已存在
                existing = self.execute_query("SELECT id FROM stocks WHERE symbol = ?", (symbol,))
                
                if existing:
                    # 更新现有记录
                    update_query = """
                        UPDATE stocks SET 
                            name = ?, market = ?, board_type = ?, exchange = ?,
                            ah_pair = ?, industry = ?, market_cap = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    """
                    params = (
                        stock.get('name'), stock.get('market'), stock.get('board_type'),
                        stock.get('exchange'), stock.get('ah_pair'), stock.get('industry'),
                        stock.get('market_cap'), symbol
                    )
                    self.execute_update(update_query, params)
                else:
                    # 插入新记录
                    insert_query = """
                        INSERT INTO stocks (symbol, name, market, board_type, exchange, 
                                        ah_pair, industry, market_cap)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    params = (
                        symbol, stock.get('name'), stock.get('market'), stock.get('board_type'),
                        stock.get('exchange'), stock.get('ah_pair'), stock.get('industry'),
                        stock.get('market_cap')
                    )
                    self.execute_update(insert_query, params)
            
            logger.info(f"Upserted {len(stocks)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting stocks: {e}")
            return False
    
    async def save_realtime_data(self, realtime_data: Dict[str, Dict[str, Any]]) -> bool:
        """
        保存实时数据到数据库
        
        Args:
            realtime_data: 实时数据字典
            
        Returns:
            是否成功
        """
        try:
            if not realtime_data:
                return True
            
            # 将实时数据转换为适合插入的格式
            data_list = []
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for symbol, data in realtime_data.items():
                data_list.append({
                    'symbol': symbol,
                    'date': current_time.split(' ')[0],  # 只取日期部分
                    'open': data.get('open'),
                    'high': data.get('high'),
                    'low': data.get('low'),
                    'close': data.get('current'),
                    'volume': data.get('volume'),
                    'amount': data.get('amount'),
                    'source': 'realtime'
                })
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(data_list)
            return await self.save_stock_data(None, df)
            
        except Exception as e:
            logger.error(f"Error saving realtime data: {e}")
            return False
    
    def get_latest_dates_by_symbol(self) -> Dict[str, str]:
        """获取每个symbol最新的日线日期"""
        try:
            query = """
                SELECT symbol, MAX(date) as max_date
                FROM prices_daily
                GROUP BY symbol
            """
            results = self.execute_query(query)
            
            result = {}
            for row in results:
                symbol = row.get('symbol') if isinstance(row, dict) else row[0]
                max_date = row.get('max_date') if isinstance(row, dict) else row[1]
                if symbol and max_date:
                    result[symbol] = str(max_date)
            
            return result
            
        except Exception as e:
            logger.error(f"获取最新日期失败: {e}")
            return {}

    def get_last_n_bars(self, symbols: List[str] = None, n: int = 2) -> pd.DataFrame:
        """
        获取每个symbol最近n根K线

        Args:
            symbols: 股票代码列表，如果为None则获取所有股票
            n: 获取的K线数量

        Returns:
            K线数据DataFrame
        """
        try:
            if symbols:
                # 构建参数化查询
                if self.db_type == 'mysql':
                    placeholders = ','.join(['%s'] * len(symbols))
                else:  # sqlite
                    placeholders = ','.join(['?'] * len(symbols))

                query = f"""
                    SELECT symbol, date, open, high, low, close, volume 
                    FROM prices_daily 
                    WHERE symbol IN ({placeholders})
                    ORDER BY symbol, date
                """
                results = self.execute_query(query, tuple(symbols))
            else:
                query = """
                    SELECT symbol, date, open, high, low, close, volume 
                    FROM prices_daily 
                    ORDER BY symbol, date
                """
                results = self.execute_query(query)

            if not results:
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])

            # 按symbol分组，取每组最后n条记录
            return df.groupby('symbol').tail(n)

        except Exception as e:
            logger.error(f"Error getting last n bars: {e}")
            return pd.DataFrame()

# 全局数据库管理器实例
_db_manager = None

def get_database_manager(db_type: Optional[str] = None, 
                        config: Optional[Dict[str, Any]] = None) -> UnifiedDatabaseManager:
    """
    获取统一数据库管理器实例
    
    Args:
        db_type: 数据库类型
        config: 数据库配置
        
    Returns:
        UnifiedDatabaseManager实例
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = UnifiedDatabaseManager(db_type, config)
    elif db_type and _db_manager.db_type != db_type:
        _db_manager.switch_database(db_type, config)
    
    return _db_manager


def reset_database_manager():
    """重置数据库管理器实例"""
    global _db_manager
    _db_manager = None

    def upsert_prices_daily(self, df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "date",
                             rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """
        插入或更新日线行情数据到 prices_daily 表，以兼容旧版 API。

        该实现复用 ``insert_dataframe`` 方法：
        1. 对输入 ``DataFrame`` 做基础字段标准化与去重处理；
        2. 对 MySQL 数据库， ``insert_dataframe`` 内部使用 ``INSERT IGNORE``，可自动忽略重复主键；
        3. 对 SQLite 数据库，先行去重后插入，避免唯一键冲突。

        Args:
            df: 待写入的数据 ``DataFrame``。
            symbol_col: 股票代码列名，默认为 ``symbol``。
            date_col: 交易日期列名，默认为 ``date``。
            rename_map: 额外的列重命名映射，可选。
            source: 数据来源标识，将写入 ``source`` 列（若存在）。

        Returns:
            实际插入的行数。
        """
        if df is None or df.empty:
            return 0

        data = df.copy()

        # 可选的列重命名映射
        if rename_map:
            data = data.rename(columns=rename_map)

        # 统一列名
        if symbol_col != "symbol" and symbol_col in data.columns:
            data.rename(columns={symbol_col: "symbol"}, inplace=True)
        if date_col != "date" and date_col in data.columns:
            data.rename(columns={date_col: "date"}, inplace=True)

        # 填充数据来源列
        if source:
            if "source" not in data.columns:
                data["source"] = source
            else:
                data["source"].fillna(source, inplace=True)

        # 日期格式统一
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # 去重避免主键冲突
        if {"symbol", "date"}.issubset(data.columns):
            data = data.drop_duplicates(subset=["symbol", "date"], keep="last")

        # 执行插入
        inserted_rows = self.insert_dataframe(data, "prices_daily", if_exists="append")
        logger.info(f"upsert_prices_daily: 实际插入 {inserted_rows} 行")
        return inserted_rows


def reset_database_manager():
    """重置数据库管理器实例"""
    global _db_manager
    _db_manager = None

if __name__ == "__main__":
    # 测试统一数据库管理器
    print("=== 统一数据库管理器测试 ===")
    
    # 测试MySQL
    print("\n--- MySQL测试 ---")
    mysql_manager = get_database_manager('mysql')
    print(f"数据库信息: {mysql_manager.get_database_info()}")
    
    # 测试表操作
    if mysql_manager.table_exists('stocks'):
        print(f"stocks表存在，行数: {mysql_manager.get_row_count('stocks')}")
        table_info = mysql_manager.get_table_info('stocks')
        print(f"表结构: {[col['Field'] if mysql_manager.db_type == 'mysql' else col['name'] for col in table_info][:5]}...")
    
    # 测试SQLite
    print("\n--- SQLite测试 ---")
    sqlite_manager = get_database_manager('sqlite')
    print(f"数据库信息: {sqlite_manager.get_database_info()}")
    
    # 测试切换
    print("\n--- 数据库切换测试 ---")
    manager = get_database_manager()
    print(f"当前数据库类型: {manager.db_type}")
    
    # 切换回MySQL
    manager.switch_database('mysql')
    print(f"切换后数据库类型: {manager.db_type}")
    
    print("\n=== 测试完成 ===")