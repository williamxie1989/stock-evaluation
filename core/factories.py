"""
Core factories: 提供统一的获取数据提供者与实时行情提供者的入口，避免上层直接依赖具体实现。
支持通过环境变量进行实现选择与配置：
- DATA_PROVIDER_IMPL: enhanced | optimized （默认 enhanced）
- DATA_PROVIDER_SOURCES: 逗号分隔的优先级列表，例如 "eastmoney,sina,akshare"
- REALTIME_PROVIDER_IMPL: enhanced （预留扩展）
"""
from __future__ import annotations
import os
from typing import Optional

# 懒加载单例，避免重复初始化重资源对象
__data_provider_singleton = None
__realtime_provider_singleton = None


def get_data_provider():
    """获取数据提供者实现（单例），默认使用 EnhancedDataProvider。
    通过环境变量 DATA_PROVIDER_IMPL 切换为 OptimizedEnhancedDataProvider。
    可选通过 DATA_PROVIDER_SOURCES 设置数据源优先级（逗号分隔）。
    """
    global __data_provider_singleton
    if __data_provider_singleton is not None:
        return __data_provider_singleton

    impl = (os.getenv("DATA_PROVIDER_IMPL") or "enhanced").strip().lower()
    provider = None

    try:
        if impl == "optimized":
            from optimized_enhanced_data_provider import OptimizedEnhancedDataProvider
            provider = OptimizedEnhancedDataProvider()
        else:
            from enhanced_data_provider import EnhancedDataProvider
            provider = EnhancedDataProvider()
    except Exception as e:
        # 回退到基础实现
        try:
            from enhanced_data_provider import EnhancedDataProvider
            provider = EnhancedDataProvider()
        except Exception:
            raise e

    # 可选：设置数据源优先级
    sources_env = os.getenv("DATA_PROVIDER_SOURCES")
    if sources_env and hasattr(provider, "set_preferred_sources"):
        sources = [s.strip() for s in sources_env.split(",") if s.strip()]
        try:
            provider.set_preferred_sources(sources)
        except Exception:
            pass

    __data_provider_singleton = provider
    return provider


def get_realtime_provider():
    """获取实时行情提供者实现（单例），当前默认使用 EnhancedRealtimeProvider。"""
    global __realtime_provider_singleton
    if __realtime_provider_singleton is not None:
        return __realtime_provider_singleton

    impl = (os.getenv("REALTIME_PROVIDER_IMPL") or "enhanced").strip().lower()
    try:
        # 预留扩展：根据 impl 切换不同实现
        from enhanced_realtime_provider import EnhancedRealtimeProvider
        provider = EnhancedRealtimeProvider()
    except Exception as e:
        raise e

    __realtime_provider_singleton = provider
    return provider