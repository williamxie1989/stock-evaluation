"""
网络重试管理器
实现指数退避重试策略和连接监控
"""

import time
import asyncio
import aiohttp
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RetryManager:
    """重试管理器类"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        """
        初始化重试管理器
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_stats = {"total_attempts": 0, "successful_retries": 0, "failed_retries": 0}
    
    def exponential_backoff(self, attempt: int) -> float:
        """
        计算指数退避延迟时间
        
        Args:
            attempt: 当前重试次数（从0开始）
            
        Returns:
            延迟时间（秒）
        """
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay
    
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        使用重试机制执行函数
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            timeout: 超时时间（秒）
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 所有重试失败后的异常
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            self.retry_stats["total_attempts"] += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    if timeout:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        result = await func(*args, **kwargs)
                else:
                    # 同步函数在异步环境中执行
                    loop = asyncio.get_event_loop()
                    if timeout:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, func, *args), 
                            timeout=timeout
                        )
                    else:
                        result = await loop.run_in_executor(None, func, *args)
                
                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                
                logger.info(f"函数执行成功，尝试次数: {attempt + 1}")
                return result
                
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as e:
                last_exception = e
                logger.warning(f"网络错误 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt == self.max_retries:
                    self.retry_stats["failed_retries"] += 1
                    logger.error(f"所有重试失败: {e}")
                    raise last_exception
                
                # 计算延迟时间并等待
                delay = self.exponential_backoff(attempt)
                logger.info(f"等待 {delay:.2f} 秒后重试...")
                await asyncio.sleep(delay)
                
            except Exception as e:
                # 非网络错误，直接抛出
                logger.error(f"非网络错误，不重试: {e}")
                raise e
    
    def get_stats(self) -> dict:
        """获取重试统计信息"""
        return self.retry_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.retry_stats = {"total_attempts": 0, "successful_retries": 0, "failed_retries": 0}


class NetworkMonitor:
    """网络状态监控器"""
    
    def __init__(self, test_urls: Optional[list] = None):
        """
        初始化网络监控器
        
        Args:
            test_urls: 测试URL列表
        """
        self.test_urls = test_urls or [
            "https://www.baidu.com",
            "https://www.google.com",
            "https://www.qq.com"
        ]
        self.status = "unknown"
        self.last_check = None
        self.retry_manager = RetryManager(max_retries=2, base_delay=1.0)
    
    async def check_connectivity(self) -> bool:
        """
        检查网络连接状态
        
        Returns:
            bool: 网络是否可用
        """
        async def test_url(url):
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        return response.status == 200
            except:
                return False
        
        for url in self.test_urls:
            try:
                # 使用重试机制测试URL
                result = await self.retry_manager.execute_with_retry(test_url, url, timeout=3)
                if result:
                    self.status = "connected"
                    self.last_check = time.time()
                    logger.info(f"网络连接正常，测试URL: {url}")
                    return True
            except Exception as e:
                logger.warning(f"测试URL {url} 失败: {e}")
                continue
        
        self.status = "disconnected"
        self.last_check = time.time()
        logger.error("网络连接失败，所有测试URL均不可用")
        return False
    
    def get_status(self) -> dict:
        """获取网络状态信息"""
        return {
            "status": self.status,
            "last_check": self.last_check,
            "test_urls": self.test_urls
        }


# 全局重试管理器实例
default_retry_manager = RetryManager()


async def retry_network_call(func: Callable, *args, **kwargs) -> Any:
    """
    使用默认重试管理器执行网络调用
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果
    """
    return await default_retry_manager.execute_with_retry(func, *args, **kwargs)


# 测试代码
if __name__ == "__main__":
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    async def test_network_monitor():
        """测试网络监控器"""
        monitor = NetworkMonitor()
        status = await monitor.check_connectivity()
        print(f"网络状态: {status}")
        print(f"状态信息: {monitor.get_status()}")
    
    async def test_retry_manager():
        """测试重试管理器"""
        
        async def mock_failing_call(attempts_to_succeed=3):
            """模拟会失败的调用"""
            nonlocal call_count
            call_count += 1
            
            if call_count < attempts_to_succeed:
                raise ConnectionError(f"模拟连接错误 (第{call_count}次调用)")
            
            return f"成功结果 (第{call_count}次调用)"
        
        call_count = 0
        retry_manager = RetryManager(max_retries=5, base_delay=0.1)
        
        try:
            result = await retry_manager.execute_with_retry(mock_failing_call, 3)
            print(f"重试测试结果: {result}")
            print(f"重试统计: {retry_manager.get_stats()}")
        except Exception as e:
            print(f"重试测试失败: {e}")
    
    # 运行测试
    async def main():
        print("=== 测试网络监控器 ===")
        await test_network_monitor()
        
        print("\n=== 测试重试管理器 ===")
        await test_retry_manager()
    
    asyncio.run(main())