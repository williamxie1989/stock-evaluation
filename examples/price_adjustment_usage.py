"""
价格调整模式使用示例

展示如何在不同的功能模块中使用不同的价格调整模式
"""

from src.data.unified_data_access import UnifiedDataAccessLayer, DataAccessConfig
from config.price_adjustment_config import get_adjust_mode_for_module

def example_ml_training():
    """模型训练示例 - 使用后复权数据"""
    print("=== 模型训练示例（后复权数据）===")
    
    # 创建数据访问层，默认使用后复权
    config = DataAccessConfig(default_adjust_mode="hfq")
    data_access = UnifiedDataAccessLayer(config=config)
    
    # 获取股票数据用于模型训练
    stock_data = data_access.get_stock_data(
        symbol="000001.SZ",
        start_date="2023-01-01", 
        end_date="2023-12-31",
        fields=["open", "close", "high", "low", "volume"]
    )
    
    if stock_data is not None:
        print(f"获取到 {len(stock_data)} 条记录")
        print("列名:", stock_data.columns.tolist())
        print("前5行数据:")
        print(stock_data.head())
    else:
        print("未获取到数据")

def example_portfolio_management():
    """投资组合管理示例 - 使用前复权数据"""
    print("\n=== 投资组合管理示例（前复权数据）===")
    
    # 创建数据访问层，默认使用前复权
    config = DataAccessConfig(default_adjust_mode="qfq")
    data_access = UnifiedDataAccessLayer(config=config)
    
    # 获取股票数据用于投资组合分析
    stock_data = data_access.get_stock_data(
        symbol="000001.SZ",
        start_date="2023-01-01",
        end_date="2023-12-31", 
        fields=["open", "close", "high", "low", "volume"]
    )
    
    if stock_data is not None:
        print(f"获取到 {len(stock_data)} 条记录")
        print("列名:", stock_data.columns.tolist())
        print("前5行数据:")
        print(stock_data.head())
    else:
        print("未获取到数据")

def example_stock_selection():
    """智能选股示例 - 使用前复权数据"""
    print("\n=== 智能选股示例（前复权数据）===")
    
    # 使用配置获取调整模式
    adjust_mode = get_adjust_mode_for_module("stock_selection")
    print(f"选股模块使用调整模式: {adjust_mode}")
    
    # 创建数据访问层
    config = DataAccessConfig(default_adjust_mode=adjust_mode)
    data_access = UnifiedDataAccessLayer(config=config)
    
    # 获取多只股票数据进行比较
    symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    
    for symbol in symbols:
        stock_data = data_access.get_stock_data(
            symbol=symbol,
            start_date="2023-11-01",
            end_date="2023-12-31",
            fields=["open", "close", "high", "low", "volume"]
        )
        
        if stock_data is not None and not stock_data.empty:
            latest_close = stock_data.iloc[-1]['close']
            print(f"{symbol}: 最新收盘价 = {latest_close:.2f}")
        else:
            print(f"{symbol}: 无数据")

def example_dynamic_adjustment():
    """动态调整模式示例"""
    print("\n=== 动态调整模式示例 ===")
    
    # 创建数据访问层
    data_access = UnifiedDataAccessLayer()
    
    symbol = "000001.SZ"
    start_date = "2023-12-01"
    end_date = "2023-12-31"
    fields = ["open", "close", "high", "low", "volume"]
    
    # 获取不同调整模式的数据
    for mode in ["origin", "qfq", "hfq"]:
        print(f"\n--- {mode.upper()} 模式 ---")
        stock_data = data_access.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            adjust_mode=mode
        )
        
        if stock_data is not None and not stock_data.empty:
            print(f"数据行数: {len(stock_data)}")
            print(f"列名: {stock_data.columns.tolist()}")
            if 'close' in stock_data.columns:
                print(f"最新收盘价: {stock_data.iloc[-1]['close']:.2f}")
        else:
            print("无数据")

if __name__ == "__main__":
    # 运行示例
    example_ml_training()
    example_portfolio_management() 
    example_stock_selection()
    example_dynamic_adjustment()
