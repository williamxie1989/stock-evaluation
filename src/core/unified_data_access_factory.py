"""
统一数据访问层工厂 - 创建和配置统一数据访问层实例
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..data.db.unified_database_manager import UnifiedDatabaseManager
from ..data.unified_data_access import UnifiedDataAccessLayer, DataAccessConfig
from ..data.providers.data_source_validator import DataSourceValidator

logger = logging.getLogger(__name__)


class UnifiedDataAccessFactory:
    """统一数据访问层工厂"""
    
    _instance: Optional['UnifiedDataAccessFactory'] = None
    _data_access_layer: Optional[UnifiedDataAccessLayer] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(UnifiedDataAccessFactory, cls).__new__(cls)
        return cls._instance
    
    def create_unified_data_access(self, 
                                 db_config: Optional[Dict[str, Any]] = None,
                                 data_access_config: Optional[Dict[str, Any]] = None,
                                 validate_sources: bool = True) -> UnifiedDataAccessLayer:
        """
        创建统一数据访问层实例
        
        Args:
            db_config: 数据库配置
            data_access_config: 数据访问配置
            validate_sources: 是否验证数据源
            
        Returns:
            统一数据访问层实例
        """
        try:
            logger.info("Creating unified data access layer...")
            
            # 创建数据库管理器
            db_manager = self._create_database_manager(db_config)
            
            # 创建数据访问配置
            config = self._create_data_access_config(data_access_config)
            
            # 创建统一数据访问层
            data_access_layer = UnifiedDataAccessLayer(db_manager, config)
            
            # 初始化数据源
            initialization_results = data_access_layer.initialize_data_sources(validate_sources)
            
            # 保存实例
            self._data_access_layer = data_access_layer
            
            logger.info("Unified data access layer created successfully")
            
            return data_access_layer
            
        except Exception as e:
            logger.error(f"Failed to create unified data access layer: {e}")
            raise
    
    def get_unified_data_access(self) -> Optional[UnifiedDataAccessLayer]:
        """
        获取已创建的统一数据访问层实例
        
        Returns:
            统一数据访问层实例，如果未创建则返回None
        """
        return self._data_access_layer
    
    def _create_database_manager(self, db_config: Optional[Dict[str, Any]] = None) -> UnifiedDatabaseManager:
        """创建数据库管理器"""
        if db_config is None:
            # 使用默认配置
            return UnifiedDatabaseManager()
        else:
            # 使用自定义配置
            return UnifiedDatabaseManager(**db_config)
    
    def _create_data_access_config(self, config_dict: Optional[Dict[str, Any]] = None) -> DataAccessConfig:
        """创建数据访问配置"""
        if config_dict is None:
            return DataAccessConfig()
        else:
            return DataAccessConfig(**config_dict)
    
    def validate_data_sources(self, data_access_layer: Optional[UnifiedDataAccessLayer] = None) -> Dict[str, Any]:
        """
        验证数据源
        
        Args:
            data_access_layer: 统一数据访问层实例，如果为None则使用已创建的实例
            
        Returns:
            验证结果
        """
        try:
            if data_access_layer is None:
                data_access_layer = self._data_access_layer
            
            if data_access_layer is None:
                raise ValueError("No unified data access layer instance available")
            
            logger.info("Starting data source validation...")
            
            # 使用数据源验证器进行验证
            validator = data_access_layer.data_validator
            validation_results = validator.validate_all_sources()
            
            # 保存验证报告
            report_filename = validator.save_validation_report(validation_results)
            
            # 打印验证摘要
            validator.print_validation_summary(validation_results)
            
            logger.info(f"Data source validation completed. Report saved to {report_filename}")
            
            return {
                'validation_results': validation_results,
                'report_filename': report_filename,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Failed to validate data sources: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def compare_data_sources_performance(self, data_access_layer: Optional[UnifiedDataAccessLayer] = None,
                                        symbols: Optional[list] = None,
                                        days: int = 30) -> Dict[str, Any]:
        """
        比较数据源性能
        
        Args:
            data_access_layer: 统一数据访问层实例
            symbols: 测试股票代码列表
            days: 测试天数
            
        Returns:
            性能比较结果
        """
        try:
            if data_access_layer is None:
                data_access_layer = self._data_access_layer
            
            if data_access_layer is None:
                raise ValueError("No unified data access layer instance available")
            
            logger.info("Starting data source performance comparison...")
            
            # 使用数据源验证器进行性能比较
            validator = data_access_layer.data_validator
            performance_results = validator.compare_sources_performance(symbols, days)
            
            logger.info("Data source performance comparison completed")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Failed to compare data source performance: {e}")
            return {'error': str(e)}
    
    def get_data_quality_report(self, data_access_layer: Optional[UnifiedDataAccessLayer] = None) -> Dict[str, Any]:
        """
        获取数据质量报告
        
        Args:
            data_access_layer: 统一数据访问层实例
            
        Returns:
            数据质量报告
        """
        try:
            if data_access_layer is None:
                data_access_layer = self._data_access_layer
            
            if data_access_layer is None:
                raise ValueError("No unified data access layer instance available")
            
            return data_access_layer.get_data_quality_report()
            
        except Exception as e:
            logger.error(f"Failed to get data quality report: {e}")
            return {'error': str(e)}
    
    def reset_data_access_layer(self) -> None:
        """重置统一数据访问层实例"""
        if self._data_access_layer:
            self._data_access_layer.clear_all_caches()
            self._data_access_layer = None
            logger.info("Unified data access layer reset")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """获取工厂统计信息"""
        return {
            'instance_created': self._data_access_layer is not None,
            'factory_created_at': datetime.now().isoformat(),
            'data_access_layer_status': 'active' if self._data_access_layer else 'not_created'
        }


# 全局工厂实例
_data_access_factory = None


def get_unified_data_access_factory() -> UnifiedDataAccessFactory:
    """获取统一数据访问层工厂实例"""
    global _data_access_factory
    if _data_access_factory is None:
        _data_access_factory = UnifiedDataAccessFactory()
    return _data_access_factory


def create_unified_data_access(db_config: Optional[Dict[str, Any]] = None,
                              data_access_config: Optional[Dict[str, Any]] = None,
                              validate_sources: bool = True) -> UnifiedDataAccessLayer:
    """
    便捷函数：创建统一数据访问层实例
    
    Args:
        db_config: 数据库配置
        data_access_config: 数据访问配置
        validate_sources: 是否验证数据源
        
    Returns:
        统一数据访问层实例
    """
    factory = get_unified_data_access_factory()
    return factory.create_unified_data_access(db_config, data_access_config, validate_sources)


def get_unified_data_access() -> Optional[UnifiedDataAccessLayer]:
    """
    便捷函数：获取已创建的统一数据访问层实例
    
    Returns:
        统一数据访问层实例
    """
    factory = get_unified_data_access_factory()
    return factory.get_unified_data_access()