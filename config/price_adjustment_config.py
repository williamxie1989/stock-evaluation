"""
价格调整模式配置

该配置文件定义了不同功能模块使用的价格调整模式：
- 模型训练：使用后复权数据（hfq）
- 智能选股：使用前复权数据（qfq）  
- 投资组合：使用前复权数据（qfq）
- 市场分析：使用前复权数据（qfq）
"""

# 功能模块价格调整模式配置
PRICE_ADJUSTMENT_CONFIG = {
    # 机器学习相关
    "ml_training": "hfq",           # 模型训练使用后复权
    "ml_prediction": "hfq",         # 模型预测使用后复权
    "feature_engineering": "hfq",   # 特征工程使用后复权
    
    # 交易和投资组合相关
    "portfolio_management": "qfq",   # 投资组合管理使用前复权
    "stock_selection": "qfq",       # 智能选股使用前复权
    "market_analysis": "qfq",       # 市场分析使用前复权
    "trading_signals": "qfq",       # 交易信号使用前复权
    
    # 数据展示相关
    "data_visualization": "qfq",    # 数据可视化使用前复权
    "reports": "qfq",               # 报告生成使用前复权
    
    # 默认模式
    "default": "origin"             # 默认使用未复权数据
}

# 价格字段映射
PRICE_FIELD_MAPPING = {
    "origin": {
        "open": "open",
        "close": "close", 
        "high": "high",
        "low": "low"
    },
    "qfq": {
        "open": "open_qfq",
        "close": "close_qfq",
        "high": "high_qfq", 
        "low": "low_qfq"
    },
    "hfq": {
        "open": "open_hfq",
        "close": "close_hfq",
        "high": "high_hfq",
        "low": "low_hfq"
    }
}

def get_adjust_mode_for_module(module_name: str) -> str:
    """
    获取指定模块的价格调整模式
    
    Args:
        module_name: 模块名称
        
    Returns:
        价格调整模式字符串
    """
    return PRICE_ADJUSTMENT_CONFIG.get(module_name, PRICE_ADJUSTMENT_CONFIG["default"])

def get_price_field_mapping(adjust_mode: str) -> dict:
    """
    获取指定调整模式的字段映射
    
    Args:
        adjust_mode: 调整模式
        
    Returns:
        字段映射字典
    """
    return PRICE_FIELD_MAPPING.get(adjust_mode, PRICE_FIELD_MAPPING["origin"])
