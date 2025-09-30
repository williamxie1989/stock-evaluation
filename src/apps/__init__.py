"""
股票评估系统 - 应用入口模块

提供Web服务、CLI工具、脚本等应用入口。
"""

# 延迟导入核心应用组件，避免循环导入
_api_app = None

def get_api_app():
    """延迟获取API应用实例"""
    global _api_app
    if _api_app is None:
        from .api.app import app as _api_app
    return _api_app

# 属性访问器，保持向后兼容性
api_app = property(lambda self: get_api_app())

# 导入脚本工具（这些导入不应该导致循环导入）
try:
    from .scripts.selector_service import IntelligentStockSelector
    from .scripts.stock_status_filter import StockStatusFilter
    from .scripts.enhanced_features import FeatureGenerator, EnhancedFeatureGenerator
    from ..data.providers.akshare_provider import AkshareDataProvider
    from .scripts.concurrent_data_sync_service import ConcurrentDataSyncService
    from .scripts.concurrent_enhanced_data_provider import ConcurrentEnhancedDataProvider
except ImportError:
    # 如果脚本工具导入失败，可以在这里处理
    IntelligentStockSelector = None
    StockStatusFilter = None
    FeatureGenerator = None
    EnhancedFeatureGenerator = None
    AkshareDataProvider = None
    ConcurrentDataSyncService = None
    ConcurrentEnhancedDataProvider = None

__all__ = [
    'get_api_app',
    'IntelligentStockSelector',
    'StockStatusFilter',
    'FeatureGenerator',
    'EnhancedFeatureGenerator',
    'AkshareDataProvider',
    'ConcurrentDataSyncService',
    'ConcurrentEnhancedDataProvider'
]