"""
数据缓存管理器
实现本地数据缓存和离线模式支持
"""

import os
import pickle
import time
import hashlib
import asyncio
from typing import Any, Optional, Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "./data_cache", default_ttl_hours: int = 24):
        """
        初始化数据缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            default_ttl_hours: 默认缓存过期时间（小时）
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl_hours * 3600  # 转换为秒
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "expired": 0
        }
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"数据缓存初始化完成，目录: {cache_dir}")
    
    def _get_cache_key(self, data_type: str, identifier: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            data_type: 数据类型（如：stock_data, market_data等）
            identifier: 数据标识符（如：股票代码）
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        # 创建参数哈希
        param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_base = f"{data_type}_{identifier}"
        
        if param_str:
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_base += f"_{param_hash}"
        
        return key_base + ".pkl"
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件完整路径"""
        return os.path.join(self.cache_dir, cache_key)
    
    def is_cached(self, data_type: str, identifier: str, **kwargs) -> bool:
        """
        检查数据是否已缓存且未过期
        
        Args:
            data_type: 数据类型
            identifier: 数据标识符
            **kwargs: 其他参数
            
        Returns:
            bool: 是否有效缓存
        """
        cache_key = self._get_cache_key(data_type, identifier, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not os.path.exists(cache_file):
            return False
        
        # 检查是否过期
        file_mtime = os.path.getmtime(cache_file)
        if time.time() - file_mtime > self.default_ttl:
            self.cache_stats["expired"] += 1
            return False
        
        return True
    
    def get_cached_data(self, data_type: str, identifier: str, **kwargs) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            data_type: 数据类型
            identifier: 数据标识符
            **kwargs: 其他参数
            
        Returns:
            缓存数据，如果不存在或过期则返回None
        """
        if not self.is_cached(data_type, identifier, **kwargs):
            self.cache_stats["misses"] += 1
            return None
        
        cache_key = self._get_cache_key(data_type, identifier, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.cache_stats["hits"] += 1
            logger.debug(f"缓存命中: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"读取缓存文件失败 {cache_file}: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    def set_cached_data(self, data: Any, data_type: str, identifier: str, **kwargs) -> bool:
        """
        设置缓存数据
        
        Args:
            data: 要缓存的数据
            data_type: 数据类型
            identifier: 数据标识符
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功缓存
        """
        cache_key = self._get_cache_key(data_type, identifier, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache_stats["writes"] += 1
            logger.debug(f"数据已缓存: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"写入缓存文件失败 {cache_file}: {e}")
            return False
    
    def clear_expired_cache(self) -> int:
        """
        清理过期缓存
        
        Returns:
            int: 清理的文件数量
        """
        cleared_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    file_mtime = os.path.getmtime(file_path)
                    if time.time() - file_mtime > self.default_ttl:
                        os.remove(file_path)
                        cleared_count += 1
                        logger.debug(f"清理过期缓存: {filename}")
                except Exception as e:
                    logger.warning(f"清理缓存文件失败 {filename}: {e}")
        
        logger.info(f"清理完成，共清理 {cleared_count} 个过期缓存文件")
        return cleared_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        cache_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in cache_files
        ) / (1024 * 1024)  # MB
        
        return {
            "cache_dir": self.cache_dir,
            "total_files": len(cache_files),
            "cache_size_mb": round(cache_size, 2),
            "stats": self.cache_stats.copy(),
            "hit_rate": (
                self.cache_stats["hits"] / max(1, self.cache_stats["hits"] + self.cache_stats["misses"]) * 100
            )
        }


class CachedDataProvider:
    """带缓存的数据提供器"""
    
    def __init__(self, data_cache: DataCache, network_monitor=None):
        """
        初始化缓存数据提供器
        
        Args:
            data_cache: 数据缓存实例
            network_monitor: 网络监控器实例
        """
        self.cache = data_cache
        self.network_monitor = network_monitor
        self.offline_mode = False
    
    async def get_data(
        self, 
        data_type: str, 
        identifier: str, 
        fetch_func: callable,
        force_fetch: bool = False,
        **kwargs
    ) -> Any:
        """
        获取数据（优先使用缓存）
        
        Args:
            data_type: 数据类型
            identifier: 数据标识符
            fetch_func: 数据获取函数
            force_fetch: 是否强制重新获取
            **kwargs: 其他参数
            
        Returns:
            数据
        """
        # 检查网络状态
        if self.network_monitor:
            try:
                network_available = await self.network_monitor.check_connectivity()
                self.offline_mode = not network_available
            except:
                self.offline_mode = True
        
        # 离线模式或强制获取时，尝试使用缓存
        if self.offline_mode and not force_fetch:
            cached_data = self.cache.get_cached_data(data_type, identifier, **kwargs)
            if cached_data is not None:
                logger.info(f"离线模式，使用缓存数据: {data_type}/{identifier}")
                return cached_data
            else:
                logger.warning(f"离线模式，无可用缓存数据: {data_type}/{identifier}")
                raise ConnectionError("离线模式且无可用缓存数据")
        
        # 在线模式，检查缓存
        if not force_fetch:
            cached_data = self.cache.get_cached_data(data_type, identifier, **kwargs)
            if cached_data is not None:
                logger.debug(f"使用缓存数据: {data_type}/{identifier}")
                return cached_data
        
        # 需要重新获取数据
        try:
            logger.info(f"获取实时数据: {data_type}/{identifier}")
            
            if hasattr(fetch_func, '__call__'):
                # 同步函数
                if not asyncio.iscoroutinefunction(fetch_func):
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, fetch_func, identifier, **kwargs)
                else:
                    # 异步函数
                    data = await fetch_func(identifier, **kwargs)
            else:
                raise ValueError("fetch_func 必须是可调用对象")
            
            # 缓存获取的数据
            if data is not None:
                self.cache.set_cached_data(data, data_type, identifier, **kwargs)
            
            return data
            
        except Exception as e:
            logger.error(f"获取数据失败: {data_type}/{identifier}, 错误: {e}")
            
            # 尝试使用缓存作为降级方案
            if not force_fetch:
                cached_data = self.cache.get_cached_data(data_type, identifier, **kwargs)
                if cached_data is not None:
                    logger.warning(f"获取实时数据失败，使用缓存数据: {data_type}/{identifier}")
                    return cached_data
            
            raise e


# 测试代码
if __name__ == "__main__":
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    def test_data_cache():
        """测试数据缓存"""
        cache = DataCache(cache_dir="./test_cache", default_ttl_hours=1)
        
        # 测试数据
        test_data = {
            "symbol": "000001.SZ",
            "price": 15.67,
            "volume": 1000000,
            "timestamp": datetime.now()
        }
        
        # 设置缓存
        success = cache.set_cached_data(test_data, "stock", "000001.SZ", timeframe="1d")
        print(f"设置缓存结果: {success}")
        
        # 检查缓存
        is_cached = cache.is_cached("stock", "000001.SZ", timeframe="1d")
        print(f"缓存存在: {is_cached}")
        
        # 获取缓存
        cached_data = cache.get_cached_data("stock", "000001.SZ", timeframe="1d")
        print(f"获取缓存数据: {cached_data}")
        
        # 获取缓存信息
        cache_info = cache.get_cache_info()
        print(f"缓存信息: {cache_info}")
        
        # 清理测试缓存
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
    
    async def test_cached_provider():
        """测试缓存数据提供器"""
        
        def mock_fetch_stock_data(symbol, **kwargs):
            """模拟股票数据获取函数"""
            return {
                "symbol": symbol,
                "price": 15.67 + hash(symbol) % 10 / 10,  # 模拟不同价格
                "volume": 1000000,
                "timestamp": datetime.now()
            }
        
        cache = DataCache(cache_dir="./test_cache", default_ttl_hours=1)
        provider = CachedDataProvider(cache)
        
        # 第一次获取（应该调用fetch函数）
        data1 = await provider.get_data("stock", "000001.SZ", mock_fetch_stock_data)
        print(f"第一次获取数据: {data1}")
        
        # 第二次获取（应该使用缓存）
        data2 = await provider.get_data("stock", "000001.SZ", mock_fetch_stock_data)
        print(f"第二次获取数据: {data2}")
        
        # 强制重新获取
        data3 = await provider.get_data("stock", "000001.SZ", mock_fetch_stock_data, force_fetch=True)
        print(f"强制获取数据: {data3}")
        
        # 清理测试缓存
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
    
    # 运行测试
    print("=== 测试数据缓存 ===")
    test_data_cache()
    
    print("\n=== 测试缓存数据提供器 ===")
    asyncio.run(test_cached_provider())