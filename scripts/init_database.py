#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建数据库和表结构
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库表结构定义
MYSQL_TABLES = {
    "stocks": """
        CREATE TABLE IF NOT EXISTS stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL UNIQUE,
            name VARCHAR(100) NOT NULL,
            industry VARCHAR(100),
            sector VARCHAR(100),
            market_cap BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_symbol (symbol),
            INDEX idx_industry (industry)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    
    "stock_prices": """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open_price DECIMAL(10, 4),
            close_price DECIMAL(10, 4),
            high_price DECIMAL(10, 4),
            low_price DECIMAL(10, 4),
            volume BIGINT,
            turnover DECIMAL(15, 2),
            amplitude DECIMAL(5, 2),
            change_percent DECIMAL(5, 2),
            change_amount DECIMAL(10, 4),
            turnover_rate DECIMAL(5, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_symbol_date (symbol, date),
            INDEX idx_symbol (symbol),
            INDEX idx_date (date),
            INDEX idx_symbol_date (symbol, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    
    "market_data": """
        CREATE TABLE IF NOT EXISTS market_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            real_time_price DECIMAL(10, 4),
            bid_price DECIMAL(10, 4),
            ask_price DECIMAL(10, 4),
            bid_volume INT,
            ask_volume INT,
            market_status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_symbol_date (symbol, date),
            INDEX idx_symbol (symbol),
            INDEX idx_date (date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    
    "trading_signals": """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            signal_date DATE NOT NULL,
            signal_type VARCHAR(20) NOT NULL,
            signal_strength DECIMAL(5, 2),
            strategy_name VARCHAR(50),
            parameters JSON,
            confidence_score DECIMAL(5, 2),
            is_valid BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_symbol (symbol),
            INDEX idx_signal_date (signal_date),
            INDEX idx_signal_type (signal_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    
    "backtest_results": """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            parameters JSON,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            initial_capital DECIMAL(15, 2),
            final_capital DECIMAL(15, 2),
            total_return DECIMAL(8, 4),
            annualized_return DECIMAL(8, 4),
            max_drawdown DECIMAL(8, 4),
            sharpe_ratio DECIMAL(8, 4),
            win_rate DECIMAL(5, 2),
            total_trades INT,
            profit_trades INT,
            loss_trades INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_strategy (strategy_name),
            INDEX idx_date_range (start_date, end_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    
    "system_logs": """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            level VARCHAR(20) NOT NULL,
            module VARCHAR(100),
            function_name VARCHAR(100),
            message TEXT,
            error_stack TEXT,
            context JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_level (level),
            INDEX idx_created_at (created_at),
            INDEX idx_module (module)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
}

SQLITE_TABLES = {
    "stocks": """
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            industry TEXT,
            sector TEXT,
            market_cap INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "stock_prices": """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open_price REAL,
            close_price REAL,
            high_price REAL,
            low_price REAL,
            volume INTEGER,
            turnover REAL,
            amplitude REAL,
            change_percent REAL,
            change_amount REAL,
            turnover_rate REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        )
    """,
    
    "market_data": """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            real_time_price REAL,
            bid_price REAL,
            ask_price REAL,
            bid_volume INTEGER,
            ask_volume INTEGER,
            market_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        )
    """,
    
    "trading_signals": """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_date DATE NOT NULL,
            signal_type TEXT NOT NULL,
            signal_strength REAL,
            strategy_name TEXT,
            parameters TEXT,
            confidence_score REAL,
            is_valid BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "backtest_results": """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            parameters TEXT,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            initial_capital REAL,
            final_capital REAL,
            total_return REAL,
            annualized_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            total_trades INTEGER,
            profit_trades INTEGER,
            loss_trades INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "system_logs": """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            module TEXT,
            function_name TEXT,
            message TEXT,
            error_stack TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
}

class DatabaseInitializer:
    """数据库初始化器"""
    
    def __init__(self, environment: str = 'development'):
        """
        初始化数据库初始化器
        
        Args:
            environment: 环境名称 (development, testing, production)
        """
        self.environment = environment
        self.config_loader = None
        self.db_type = None
        self.db_config = None
        
        # 初始化配置加载器
        self._init_config()
    
    def _init_config(self):
        """初始化配置"""
        try:
            from src.data.db.config_loader import get_config_loader
            
            self.config_loader = get_config_loader()
            self.config_loader.set_environment(self.environment)
            
            # 获取数据库配置
            config = self.config_loader.get_config()
            self.db_type = os.getenv('DB_TYPE', 'mysql').lower()
            self.db_config = config.get(self.db_type, {})
            
            logger.info(f"数据库类型: {self.db_type}")
            logger.info(f"当前环境: {self.environment}")
            
        except Exception as e:
            logger.error(f"配置初始化失败: {e}")
            raise
    
    def create_database_mysql(self):
        """创建MySQL数据库"""
        try:
            import mysql.connector
            from mysql.connector import errorcode
            
            # 连接到MySQL服务器（不指定数据库）
            connection_config = self.db_config.copy()
            database_name = connection_config.pop('database', 'stock_evaluation')
            
            conn = mysql.connector.connect(**connection_config)
            cursor = conn.cursor()
            
            try:
                # 创建数据库
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                logger.info(f"数据库 '{database_name}' 创建成功")
                
                # 使用数据库
                cursor.execute(f"USE {database_name}")
                
                # 创建表
                self._create_tables_mysql(cursor)
                
                conn.commit()
                logger.info("MySQL数据库初始化完成")
                
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    logger.error("MySQL访问被拒绝，请检查用户名和密码")
                elif err.errno == errorcode.ER_BAD_DB_ERROR:
                    logger.error("数据库不存在")
                else:
                    logger.error(f"MySQL错误: {err}")
                raise
            
            finally:
                cursor.close()
                conn.close()
                
        except ImportError:
            logger.error("MySQL连接器未安装，请运行: pip install mysql-connector-python")
            raise
        except Exception as e:
            logger.error(f"MySQL数据库创建失败: {e}")
            raise
    
    def create_database_sqlite(self):
        """创建SQLite数据库"""
        try:
            import sqlite3
            
            db_path = self.db_config.get('database', 'stock_data.sqlite3')
            
            # 确保目录存在
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # 连接数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            try:
                # 启用外键支持
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # 创建表
                self._create_tables_sqlite(cursor)
                
                conn.commit()
                logger.info(f"SQLite数据库 '{db_path}' 初始化完成")
                
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"SQLite数据库创建失败: {e}")
            raise
    
    def _create_tables_mysql(self, cursor):
        """创建MySQL表"""
        for table_name, sql in MYSQL_TABLES.items():
            try:
                cursor.execute(sql)
                logger.info(f"表 '{table_name}' 创建成功")
            except Exception as e:
                logger.error(f"表 '{table_name}' 创建失败: {e}")
                raise
    
    def _create_tables_sqlite(self, cursor):
        """创建SQLite表"""
        for table_name, sql in SQLITE_TABLES.items():
            try:
                cursor.execute(sql)
                logger.info(f"表 '{table_name}' 创建成功")
            except Exception as e:
                logger.error(f"表 '{table_name}' 创建失败: {e}")
                raise
    
    def initialize(self):
        """初始化数据库"""
        logger.info(f"开始初始化 {self.db_type} 数据库...")
        
        try:
            if self.db_type == 'mysql':
                self.create_database_mysql()
            elif self.db_type == 'sqlite':
                self.create_database_sqlite()
            else:
                raise ValueError(f"不支持的数据库类型: {self.db_type}")
            
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if self.db_type == 'mysql':
                import mysql.connector
                conn = mysql.connector.connect(**self.db_config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
            else:  # sqlite
                import sqlite3
                db_path = self.db_config.get('database', 'stock_data.sqlite3')
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据库初始化工具')
    parser.add_argument('--env', '-e', default='development', 
                       choices=['development', 'testing', 'production'],
                       help='环境名称 (默认: development)')
    parser.add_argument('--test-connection', '-t', action='store_true',
                       help='仅测试数据库连接')
    parser.add_argument('--db-type', '-d', 
                       choices=['mysql', 'sqlite'],
                       help='数据库类型 (覆盖环境变量)')
    
    args = parser.parse_args()
    
    # 如果指定了数据库类型，设置环境变量
    if args.db_type:
        os.environ['DB_TYPE'] = args.db_type
    
    try:
        # 创建初始化器
        initializer = DatabaseInitializer(args.env)
        
        # 测试连接
        if args.test_connection:
            print(f"测试 {args.env} 环境的数据库连接...")
            success = initializer.test_connection()
            print(f"连接测试结果: {'成功' if success else '失败'}")
            return 0 if success else 1
        
        # 初始化数据库
        print(f"初始化 {args.env} 环境的数据库...")
        initializer.initialize()
        print("数据库初始化完成")
        
        return 0
        
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())