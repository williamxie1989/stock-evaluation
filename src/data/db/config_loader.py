#!/usr/bin/env python3
"""
数据库配置加载器
提供统一的数据库配置加载和管理功能
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseConfigLoader:
    """数据库配置加载器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        """
        if config_dir is None:
            # 从当前文件位置计算项目根目录
            self.config_dir = Path(__file__).parent.parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_file = self.config_dir / "database_config.json"
        self.env_file = None
        self._config_data = None
        self._current_env = None
        
        # 设置当前环境
        self._set_current_environment()
    
    def _set_current_environment(self):
        """设置当前环境"""
        # 从环境变量获取，默认为development
        env = os.getenv('APP_ENV', 'development').lower()
        
        # 验证环境名称
        valid_envs = ['development', 'testing', 'production']
        if env not in valid_envs:
            logger.warning(f"无效的环境名称: {env}，使用默认development")
            env = 'development'
        
        self._current_env = env
        self.env_file = self.config_dir / f".env.{env}"
        
        logger.info(f"当前环境: {env}")
    
    def _load_config_file(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            logger.error(f"配置文件不存在: {self.config_file}")
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _load_env_file(self) -> Dict[str, Any]:
        """加载环境变量文件"""
        if not self.env_file or not self.env_file.exists():
            logger.warning(f"环境变量文件不存在: {self.env_file}")
            return {}
        
        env_vars = {}
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
            logger.info(f"成功加载环境变量文件: {self.env_file}")
            return env_vars
        except Exception as e:
            logger.error(f"加载环境变量文件失败: {e}")
            return {}
    
    def _resolve_env_variables(self, value: str, env_vars: Dict[str, Any]) -> str:
        """解析环境变量引用"""
        if isinstance(value, str) and '${' in value:
            # 处理 ${VAR} 和 ${VAR:default} 格式
            import re
            def replace_env_var(match):
                var_expr = match.group(1)
                if ':' in var_expr:
                    var_name, default_value = var_expr.split(':', 1)
                    return env_vars.get(var_name, default_value)
                else:
                    return env_vars.get(var_expr, '')
            
            return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
        return value
    
    def _resolve_config_variables(self, config: Dict[str, Any], env_vars: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析配置中的环境变量引用"""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_config_variables(value, env_vars)
            elif isinstance(value, str):
                resolved[key] = self._resolve_env_variables(value, env_vars)
            else:
                resolved[key] = value
        return resolved
    
    def get_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定环境的配置
        
        Args:
            environment: 环境名称，默认为当前环境
            
        Returns:
            数据库配置字典
        """
        if environment is None:
            environment = self._current_env
        
        # 验证环境名称
        valid_envs = ['development', 'testing', 'production']
        if environment not in valid_envs:
            logger.error(f"无效的环境名称: {environment}")
            raise ValueError(f"无效的环境名称: {environment}")
        
        # 加载配置文件
        config_data = self._load_config_file()
        
        # 获取指定环境的配置
        if environment not in config_data:
            logger.error(f"配置文件中缺少 {environment} 环境的配置")
            raise KeyError(f"配置文件中缺少 {environment} 环境的配置")
        
        env_config = config_data[environment]
        
        # 加载环境变量
        env_vars = self._load_env_file()
        
        # 合并系统环境变量
        env_vars.update({k: v for k, v in os.environ.items() if k.startswith(('DB_', 'SQLITE_', 'REDIS_', 'API_', 'LOG_', 'SYNC_', 'CACHE_', 'SECRET_', 'JWT_', 'AKSHARE_', 'TUSHARE_', 'EASTMONEY_'))})
        
        # 解析配置中的环境变量引用
        resolved_config = self._resolve_config_variables(env_config, env_vars)
        
        logger.info(f"成功获取 {environment} 环境的数据库配置")
        return resolved_config
    
    def get_db_type(self) -> str:
        """获取数据库类型"""
        return os.getenv('DB_TYPE', 'mysql').lower()
    
    def get_pool_config(self) -> Dict[str, Any]:
        """获取连接池配置"""
        config = self.get_config()
        return config.get('pool', {
            'max_connections': 10,
            'min_connections': 1,
            'connection_timeout': 30,
            'idle_timeout': 300,
            'max_idle_time': 600
        })
    
    def get_mysql_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """获取MySQL配置"""
        config = self.get_config(environment)
        if 'mysql' not in config:
            logger.error(f"配置中缺少MySQL配置")
            raise KeyError(f"配置中缺少MySQL配置")
        return config['mysql']
    
    def get_sqlite_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """获取SQLite配置"""
        config = self.get_config(environment)
        if 'sqlite' not in config:
            logger.error(f"配置中缺少SQLite配置")
            raise KeyError(f"配置中缺少SQLite配置")
        return config['sqlite']
    
    def get_current_environment(self) -> str:
        """获取当前环境"""
        return self._current_env
    
    def set_environment(self, environment: str):
        """设置当前环境"""
        valid_envs = ['development', 'testing', 'production']
        if environment not in valid_envs:
            logger.error(f"无效的环境名称: {environment}")
            raise ValueError(f"无效的环境名称: {environment}")
        
        self._current_env = environment
        self.env_file = self.config_dir / f".env.{environment}"
        logger.info(f"设置当前环境为: {environment}")
    
    def test_connection(self, environment: Optional[str] = None, db_type: str = 'mysql') -> bool:
        """
        测试数据库连接
        
        Args:
            environment: 环境名称
            db_type: 数据库类型 (mysql 或 sqlite)
            
        Returns:
            连接是否成功
        """
        try:
            config = self.get_config(environment)
            
            if db_type == 'mysql':
                mysql_config = config['mysql']
                import mysql.connector
                conn = mysql.connector.connect(**mysql_config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
            else:  # sqlite
                sqlite_config = config['sqlite']
                import sqlite3
                conn = sqlite3.connect(**sqlite_config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return result is not None
                
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False

# 全局配置加载器实例
_config_loader = None

def get_config_loader(config_dir: Optional[str] = None) -> DatabaseConfigLoader:
    """获取数据库配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = DatabaseConfigLoader(config_dir)
    return _config_loader

def get_database_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """便捷函数：获取数据库配置"""
    loader = get_config_loader()
    return loader.get_config(environment)

def get_mysql_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """便捷函数：获取MySQL配置"""
    loader = get_config_loader()
    return loader.get_mysql_config(environment)

def get_sqlite_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """便捷函数：获取SQLite配置"""
    loader = get_config_loader()
    return loader.get_sqlite_config(environment)

if __name__ == "__main__":
    # 测试配置加载器
    print("=== 数据库配置加载器测试 ===")
    
    loader = get_config_loader()
    
    # 测试不同环境
    for env in ['development', 'testing', 'production']:
        print(f"\n--- {env.upper()} 环境 ---")
        try:
            config = loader.get_config(env)
            print(f"MySQL配置: {config.get('mysql', {}).get('host', 'N/A')}")
            print(f"SQLite配置: {config.get('sqlite', {}).get('database', 'N/A')}")
            
            # 测试连接
            if env == 'testing':
                # 测试环境使用SQLite
                result = loader.test_connection(env, 'sqlite')
                print(f"SQLite连接测试: {'成功' if result else '失败'}")
            else:
                # 其他环境使用MySQL
                result = loader.test_connection(env, 'mysql')
                print(f"MySQL连接测试: {'成功' if result else '失败'}")
                
        except Exception as e:
            print(f"配置加载失败: {e}")
    
    print("\n=== 测试完成 ===")