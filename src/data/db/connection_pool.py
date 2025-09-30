"""
数据库连接池管理器
提供高性能的数据库连接池管理
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union
from pathlib import Path

# 数据库适配器
try:
    import mysql.connector.pooling
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from src.data.db.config_loader import get_config_loader

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """数据库连接池管理器"""
    
    _instance = None
    _lock = threading.Lock()
    _pools = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseConnectionPool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config_loader = get_config_loader()
            self.db_type = None
            self.pool_config = None
            self.pool = None
            self.connection_count = 0
            self.max_connections = 10
            self.min_connections = 1
    
    def initialize(self, environment: str = None):
        """
        初始化连接池
        
        Args:
            environment: 环境名称，如果为None则使用当前环境
        """
        if environment:
            self.config_loader.set_environment(environment)
        
        config = self.config_loader.get_config()
        self.db_type = self.config_loader.get_db_type()
        self.pool_config = config.get(self.db_type, {})
        
        # 获取连接池配置
        pool_settings = config.get('pool', {})
        self.max_connections = pool_settings.get('max_connections', 10)
        self.min_connections = pool_settings.get('min_connections', 1)
        
        if self.db_type == 'mysql':
            self._init_mysql_pool()
        elif self.db_type == 'sqlite':
            self._init_sqlite_pool()
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
        
        logger.info(f"数据库连接池初始化完成，类型: {self.db_type}")
    
    def _init_mysql_pool(self):
        """初始化MySQL连接池"""
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL连接器未安装，请运行: pip install mysql-connector-python")
        
        try:
            # MySQL连接池配置
            pool_config = {
                'pool_name': 'stock_evaluation_pool',
                'pool_size': self.max_connections,
                'pool_reset_session': True,
                'host': self.pool_config.get('host', 'localhost'),
                'port': self.pool_config.get('port', 3306),
                'user': self.pool_config.get('user'),
                'password': self.pool_config.get('password'),
                'database': self.pool_config.get('database', 'stock_evaluation'),
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci',
                'autocommit': True,
                'time_zone': '+00:00',
                'sql_mode': 'STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION'
            }
            
            # 创建连接池
            self.pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
            
            logger.info(f"MySQL连接池创建成功，大小: {self.max_connections}")
            
        except Exception as e:
            logger.error(f"MySQL连接池初始化失败: {e}")
            raise
    
    def _init_sqlite_pool(self):
        """初始化SQLite连接池"""
        if not SQLITE_AVAILABLE:
            raise ImportError("SQLite3不可用")
        
        try:
            db_path = self.pool_config.get('database', 'stock_data.sqlite3')
            
            # 确保目录存在
            db_dir = Path(db_path).parent
            if db_dir and not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
            
            # SQLite使用简单的连接管理
            self.pool = {
                'database': db_path,
                'connections': {},
                'lock': threading.Lock()
            }
            
            logger.info(f"SQLite连接管理初始化完成，数据库: {db_path}")
            
        except Exception as e:
            logger.error(f"SQLite连接管理初始化失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self, timeout: int = 30):
        """
        获取数据库连接
        
        Args:
            timeout: 连接超时时间（秒）
            
        Yields:
            数据库连接对象
        """
        connection = None
        start_time = time.time()
        
        try:
            if self.db_type == 'mysql':
                connection = self._get_mysql_connection(timeout)
            elif self.db_type == 'sqlite':
                connection = self._get_sqlite_connection(timeout)
            else:
                raise ValueError(f"不支持的数据库类型: {self.db_type}")
            
            yield connection
            
        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            if connection:
                self._close_connection(connection)
            raise
            
        finally:
            if connection:
                self._return_connection(connection)
    
    def _get_mysql_connection(self, timeout: int):
        """获取MySQL连接"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                connection = self.pool.get_connection()
                self.connection_count += 1
                
                # 测试连接
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchall()  # 确保读取所有结果
                cursor.close()
                
                return connection
                
            except mysql.connector.PoolError:
                logger.warning("连接池已满，等待重试...")
                time.sleep(0.1)
                continue
                
            except Exception as e:
                logger.error(f"获取MySQL连接失败: {e}")
                raise
        
        raise TimeoutError(f"获取连接超时 ({timeout}秒)")
    
    def _get_sqlite_connection(self, timeout: int):
        """获取SQLite连接"""
        thread_id = threading.current_thread().ident
        
        with self.pool['lock']:
            if thread_id in self.pool['connections']:
                return self.pool['connections'][thread_id]
            
            try:
                connection = sqlite3.connect(
                    self.pool['database'],
                    timeout=timeout,
                    isolation_level=None,  # 自动提交模式
                    check_same_thread=False
                )
                
                # 启用外键支持
                connection.execute("PRAGMA foreign_keys = ON")
                
                # 优化SQLite性能
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.execute("PRAGMA cache_size = -64000")  # 64MB缓存
                connection.execute("PRAGMA temp_store = MEMORY")
                
                self.pool['connections'][thread_id] = connection
                self.connection_count += 1
                
                return connection
                
            except Exception as e:
                logger.error(f"获取SQLite连接失败: {e}")
                raise
    
    def _close_connection(self, connection):
        """关闭连接"""
        try:
            if self.db_type == 'mysql':
                if connection and connection.is_connected():
                    connection.close()
                    self.connection_count -= 1
            elif self.db_type == 'sqlite':
                if connection:
                    connection.close()
                    self.connection_count -= 1
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")
    
    def _return_connection(self, connection):
        """归还连接到池"""
        try:
            if self.db_type == 'mysql':
                # MySQL连接池自动管理
                pass
            elif self.db_type == 'sqlite':
                # SQLite连接保持在线程中
                pass
        except Exception as e:
            logger.error(f"归还连接失败: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        stats = {
            'db_type': self.db_type,
            'active_connections': self.connection_count,
            'max_connections': self.max_connections,
            'min_connections': self.min_connections
        }
        
        if self.db_type == 'mysql' and self.pool:
            try:
                stats.update({
                    'pool_size': self.pool.pool_size,
                    'available_connections': len(self.pool._cnx_queue.queue)
                })
            except:
                pass
        
        return stats
    
    def close_all_connections(self):
        """关闭所有连接"""
        try:
            if self.db_type == 'mysql':
                # MySQL连接池自动管理
                pass
            elif self.db_type == 'sqlite':
                with self.pool['lock']:
                    for thread_id, connection in self.pool['connections'].items():
                        try:
                            connection.close()
                        except:
                            pass
                    self.pool['connections'].clear()
            
            self.connection_count = 0
            logger.info("所有数据库连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")

# 全局连接池实例
_connection_pool = None

def get_connection_pool() -> DatabaseConnectionPool:
    """获取数据库连接池实例"""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = DatabaseConnectionPool()
    
    return _connection_pool

def init_connection_pool(environment: str = None):
    """
    初始化数据库连接池
    
    Args:
        environment: 环境名称
    """
    pool = get_connection_pool()
    pool.initialize(environment)

def close_connection_pool():
    """关闭数据库连接池"""
    global _connection_pool
    
    if _connection_pool:
        _connection_pool.close_all_connections()
        _connection_pool = None