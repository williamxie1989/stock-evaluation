"""
网络优化集成模块
将重试机制和缓存策略集成到主系统中
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from network_retry_manager import RetryManager, NetworkMonitor
from data_cache_manager import DataCache, CachedDataProvider

logger = logging.getLogger(__name__)


class NetworkOptimizationManager:
    """网络优化管理器"""
    
    def __init__(self, cache_dir: str = "./data_cache", default_ttl_hours: int = 24):
        """
        初始化网络优化管理器
        
        Args:
            cache_dir: 缓存目录
            default_ttl_hours: 默认缓存过期时间
        """
        # 初始化组件
        self.retry_manager = RetryManager(max_retries=3, base_delay=1.0, max_delay=30.0)
        self.network_monitor = NetworkMonitor()
        self.data_cache = DataCache(cache_dir=cache_dir, default_ttl_hours=default_ttl_hours)
        self.cached_provider = CachedDataProvider(self.data_cache, self.network_monitor)
        
        # 系统状态
        self.system_status = "initializing"
        self.last_network_check = None
        self.optimization_enabled = True
        
        logger.info("网络优化管理器初始化完成")
    
    async def initialize(self) -> bool:
        """
        初始化网络优化系统
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 检查网络连接
            network_available = await self.network_monitor.check_connectivity()
            self.last_network_check = asyncio.get_event_loop().time()
            
            if network_available:
                self.system_status = "online"
                logger.info("网络优化系统初始化成功，当前为在线模式")
            else:
                self.system_status = "offline"
                logger.warning("网络优化系统初始化成功，当前为离线模式")
            
            return True
            
        except Exception as e:
            logger.error(f"网络优化系统初始化失败: {e}")
            self.system_status = "error"
            return False
    
    async def get_data_with_optimization(
        self, 
        data_type: str, 
        identifier: str, 
        fetch_func: callable,
        force_fetch: bool = False,
        **kwargs
    ) -> Any:
        """
        使用优化策略获取数据
        
        Args:
            data_type: 数据类型
            identifier: 数据标识符
            fetch_func: 数据获取函数
            force_fetch: 是否强制重新获取
            **kwargs: 其他参数
            
        Returns:
            数据
        """
        if not self.optimization_enabled:
            # 优化功能禁用，直接调用原始函数
            logger.debug("网络优化功能已禁用，直接获取数据")
            return await self._execute_fetch_function(fetch_func, identifier, **kwargs)
        
        try:
            # 使用缓存数据提供器
            data = await self.cached_provider.get_data(
                data_type, identifier, fetch_func, force_fetch, **kwargs
            )
            
            # 更新系统状态
            await self._update_system_status()
            
            return data
            
        except Exception as e:
            logger.error(f"使用优化策略获取数据失败: {e}")
            
            # 尝试降级到基本重试机制
            try:
                logger.info("尝试使用基本重试机制获取数据")
                data = await self.retry_manager.execute_with_retry(
                    lambda: self._execute_fetch_function(fetch_func, identifier, **kwargs),
                    timeout=30
                )
                
                # 如果成功获取数据，缓存起来
                if data is not None:
                    self.data_cache.set_cached_data(data, data_type, identifier, **kwargs)
                
                return data
                
            except Exception as retry_error:
                logger.error(f"所有优化策略均失败: {retry_error}")
                raise retry_error
    
    async def _execute_fetch_function(self, fetch_func: callable, identifier: str, **kwargs) -> Any:
        """执行数据获取函数"""
        if asyncio.iscoroutinefunction(fetch_func):
            return await fetch_func(identifier, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fetch_func, identifier, **kwargs)
    
    async def _update_system_status(self):
        """更新系统状态"""
        try:
            # 每5分钟检查一次网络状态
            current_time = asyncio.get_event_loop().time()
            if (self.last_network_check is None or 
                current_time - self.last_network_check > 300):  # 5分钟
                
                network_available = await self.network_monitor.check_connectivity()
                self.last_network_check = current_time
                
                if network_available:
                    self.system_status = "online"
                else:
                    self.system_status = "offline"
        except Exception as e:
            logger.warning(f"更新系统状态失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        cache_info = self.data_cache.get_cache_info()
        retry_stats = self.retry_manager.get_stats()
        network_status = self.network_monitor.get_status()
        
        return {
            "system_status": self.system_status,
            "optimization_enabled": self.optimization_enabled,
            "last_network_check": self.last_network_check,
            "cache_info": cache_info,
            "retry_stats": retry_stats,
            "network_status": network_status
        }
    
    def enable_optimization(self):
        """启用网络优化功能"""
        self.optimization_enabled = True
        logger.info("网络优化功能已启用")
    
    def disable_optimization(self):
        """禁用网络优化功能"""
        self.optimization_enabled = False
        logger.info("网络优化功能已禁用")
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 清理过期缓存
            cleared_count = self.data_cache.clear_expired_cache()
            logger.info(f"清理完成，共清理 {cleared_count} 个过期缓存文件")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局网络优化管理器实例
_global_optimization_manager: Optional[NetworkOptimizationManager] = None


def get_global_optimization_manager() -> NetworkOptimizationManager:
    """获取全局网络优化管理器实例"""
    global _global_optimization_manager
    if _global_optimization_manager is None:
        _global_optimization_manager = NetworkOptimizationManager()
    return _global_optimization_manager


async def initialize_network_optimization() -> bool:
    """初始化全局网络优化系统"""
    manager = get_global_optimization_manager()
    return await manager.initialize()


async def get_optimized_data(
    data_type: str, 
    identifier: str, 
    fetch_func: callable,
    force_fetch: bool = False,
    **kwargs
) -> Any:
    """
    使用全局优化管理器获取数据
    
    Args:
        data_type: 数据类型
        identifier: 数据标识符
        fetch_func: 数据获取函数
        force_fetch: 是否强制重新获取
        **kwargs: 其他参数
        
    Returns:
        数据
    """
    manager = get_global_optimization_manager()
    return await manager.get_data_with_optimization(
        data_type, identifier, fetch_func, force_fetch, **kwargs
    )


def get_optimization_status() -> Dict[str, Any]:
    """获取全局优化系统状态"""
    manager = get_global_optimization_manager()
    return manager.get_system_status()


# 测试代码
if __name__ == "__main__":
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    async def test_optimization_manager():
        """测试网络优化管理器"""
        
        def mock_fetch_stock_data(symbol, **kwargs):
            """模拟股票数据获取函数"""
            return {
                "symbol": symbol,
                "price": 15.67 + hash(symbol) % 10 / 10,
                "volume": 1000000,
                "timestamp": "2025-01-01 10:00:00"
            }
        
        # 创建优化管理器
        manager = NetworkOptimizationManager(cache_dir="./test_optimization_cache")
        
        # 初始化
        success = await manager.initialize()
        print(f"初始化结果: {success}")
        
        # 获取系统状态
        status = manager.get_system_status()
        print(f"系统状态: {status['system_status']}")
        
        # 测试数据获取
        try:
            data = await manager.get_data_with_optimization(
                "stock", "000001.SZ", mock_fetch_stock_data
            )
            print(f"获取数据成功: {data}")
        except Exception as e:
            print(f"获取数据失败: {e}")
        
        # 再次获取（应该使用缓存）
        try:
            data2 = await manager.get_data_with_optimization(
                "stock", "000001.SZ", mock_fetch_stock_data
            )
            print(f"第二次获取数据: {data2}")
        except Exception as e:
            print(f"第二次获取数据失败: {e}")
        
        # 获取优化统计信息
        final_status = manager.get_system_status()
        print(f"最终系统状态: {final_status}")
        
        # 清理
        await manager.cleanup()
        
        # 清理测试缓存
        import shutil
        if os.path.exists("./test_optimization_cache"):
            shutil.rmtree("./test_optimization_cache")
    
    async def test_global_optimization():
        """测试全局优化功能"""
        
        def mock_fetch_market_data(market, **kwargs):
            """模拟市场数据获取函数"""
            return {
                "market": market,
                "index": 3000 + hash(market) % 100,
                "timestamp": "2025-01-01 10:00:00"
            }
        
        # 初始化全局优化系统
        success = await initialize_network_optimization()
        print(f"全局初始化结果: {success}")
        
        # 使用全局函数获取数据
        try:
            data = await get_optimized_data(
                "market", "SH", mock_fetch_market_data
            )
            print(f"全局获取数据成功: {data}")
        except Exception as e:
            print(f"全局获取数据失败: {e}")
        
        # 获取状态
        status = get_optimization_status()
        print(f"全局优化状态: {status['system_status']}")
    
    # 运行测试
    print("=== 测试网络优化管理器 ===")
    asyncio.run(test_optimization_manager())
    
    print("\n=== 测试全局优化功能 ===")
    asyncio.run(test_global_optimization())