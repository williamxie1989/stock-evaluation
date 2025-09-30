#!/usr/bin/env python3
"""
数据库配置管理器
支持MySQL和SQLite数据库的统一配置管理
"""

import os
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConfigManager:
    """数据库配置管理器"""
    
    def __init__(self):
        self.db_type = self._get_db_type_from_env()
        self.mysql_config = self._get_mysql_config()
        self.sqlite_config = self._get_sqlite_config()
        self.connection_pool = None
        self._init_connection_pool()
    
    def _get_db_type_from_env(self) -> str:
        """从环境变量获取数据库类型"""
        db_type = os.getenv('DB_TYPE', 'mysql').lower()
        if db_type not in ['mysql', 'sqlite']:
            logger.warning(f"不支持的数据库类型: {db_type}，使用默认mysql")
            db_type = 'mysql'
        return db_type
    
    def _get_mysql_config(self) -> Dict[str, Any]:
        """获取MySQL配置"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', 'stock_user'),
            'password': os.getenv('DB_PASSWORD', 'stock_password'),  # 修复密码
            'database': os.getenv('DB_NAME', 'stock_evaluation'),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': True,
            'time_zone': '+8:00',
            'sql_mode': 'STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION',
            'connection_timeout': 30,
            'use_unicode': True,
        }
    
    def _get_sqlite_config(self) -> Dict[str, Any]:
        """获取SQLite配置"""
        return {
            'database': os.getenv('SQLITE_DB_PATH', 'stock_data.sqlite3'),
            'timeout': 30,
            'isolation_level': None,  # 自动提交模式
        }
    
    def _init_connection_pool(self):
        """初始化连接池"""
        if self.db_type == 'mysql':
            try:
                import mysql.connector
                from mysql.connector import pooling
                
                # MySQL连接池配置 - 只使用支持的参数
                pool_config = {
                    'pool_name': 'stock_db_pool',
                    'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
                    'pool_reset_session': True,
                    **self.mysql_config
                }
                
                self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
                logger.info("MySQL连接池初始化成功")
            except Exception as e:
                logger.error(f"MySQL连接池初始化失败: {e}")
                self.connection_pool = None
        else:
            logger.info("SQLite不需要连接池")
            self.connection_pool = None
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前数据库配置"""
        return self.mysql_config if self.db_type == 'mysql' else self.sqlite_config
    
    def set_db_type(self, db_type: str):
        """设置数据库类型"""
        if db_type not in ['mysql', 'sqlite']:
            raise ValueError(f"不支持的数据库类型: {db_type}")
        
        old_type = self.db_type
        self.db_type = db_type
        logger.info(f"数据库类型切换为: {db_type}")
        
        # 如果切换类型，重新初始化连接池
        if old_type != db_type:
            self._init_connection_pool()
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if self.db_type == 'mysql':
                import mysql.connector
                conn = mysql.connector.connect(**self.mysql_config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
            else:  # sqlite
                import sqlite3
                conn = sqlite3.connect(**self.sqlite_config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def get_connection_string(self) -> str:
        """获取数据库连接字符串"""
        if self.db_type == 'mysql':
            return f"mysql+mysqlconnector://{self.mysql_config['user']}:{self.mysql_config['password']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}"
        else:
            return f"sqlite:///{self.sqlite_config['database']}"

# 全局配置管理器实例
_config_manager = None

def get_database_config() -> DatabaseConfigManager:
    """获取数据库配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DatabaseConfigManager()
    return _config_manager

def reset_database_config():
    """重置数据库配置管理器实例"""
    global _config_manager
    _config_manager = None

@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    config_manager = get_database_config()
    
    if config_manager.db_type == 'mysql':
        import mysql.connector
        conn = mysql.connector.connect(**config_manager.mysql_config)
    else:  # sqlite
        import sqlite3
        conn = sqlite3.connect(**config_manager.sqlite_config)
    
    try:
        yield conn
    finally:
        conn.close()

if __name__ == "__main__":
    # 测试配置管理器
    print("=== 数据库配置管理器测试 ===")
    
    config_manager = get_database_config()
    print(f"当前数据库类型: {config_manager.db_type}")
    print(f"配置信息: {config_manager.get_config()}")
    print(f"连接字符串: {config_manager.get_connection_string()}")
    
    # 测试连接
    print(f"连接测试结果: {config_manager.test_connection()}")
    
    # 测试上下文管理器
    print("\n--- 上下文管理器测试 ---")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if config_manager.db_type == 'mysql':
                cursor.execute("SELECT VERSION()")
                result = cursor.fetchone()
                print(f"MySQL版本: {result[0]}")
            else:
                cursor.execute("SELECT sqlite_version()")
                result = cursor.fetchone()
                print(f"SQLite版本: {result[0]}")
            cursor.close()
        print("上下文管理器测试成功")
    except Exception as e:
        print(f"上下文管理器测试失败: {e}")
    
    print("\n=== 测试完成 ===")