#!/usr/bin/env python3
"""
优化数据源配置，优先使用稳定的新浪和东方财富接口
"""

import sys   
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_data_provider import EnhancedDataProvider

def optimize_data_sources():
    """优化数据源优先级配置"""
    
    # 创建数据提供器实例
    provider = EnhancedDataProvider()
    
    # 获取当前数据源配置
    current_sources = [s['name'] for s in provider.data_sources]
    print(f"当前数据源优先级: {current_sources}")
    
    # 优化配置：优先使用新浪和东方财富
    optimized_sources = ["sina", "eastmoney", "akshare_default"]
    provider.set_preferred_sources(optimized_sources)
    
    # 验证优化后的配置
    optimized_sources = [s['name'] for s in provider.data_sources]
    print(f"优化后数据源优先级: {optimized_sources}")
    
    # 测试优化后的数据获取
    print("\n=== 测试优化后的数据获取 ===")
    
    test_symbols = ["600036", "000001", "601318"]  # 招商银行、平安银行、中国平安
    
    for symbol in test_symbols:
        print(f"\n测试股票: {symbol}")
        try:
            data = provider.get_stock_historical_data(symbol, "1y")
            if data is not None and not data.empty:
                print(f"  ✓ 数据获取成功，数据源: {provider.last_used_source}")
                print(f"    数据条数: {len(data)}")
                print(f"    最新日期: {data['date'].iloc[-1] if 'date' in data.columns else 'N/A'}")
            else:
                print(f"  ✗ 数据获取失败")
                if provider.last_attempts:
                    for attempt in provider.last_attempts:
                        print(f"    {attempt['source']}: {attempt['status']}")
        except Exception as e:
            print(f"  ✗ 异常: {e}")
    
    return provider

def create_config_file():
    """创建优化配置文件"""
    
    config_content = """# 数据源优化配置
# 优先使用稳定的新浪和东方财富接口

# 环境变量配置
export ENH_ENABLE_EXPERIMENTAL=0  # 关闭实验性数据源，提高稳定性

# 数据源优先级（从高到低）
PREFERRED_DATA_SOURCES=sina,eastmoney,akshare_default

# 重试配置
MAX_RETRIES=3
RETRY_DELAY=2

# 超时配置（秒）
REQUEST_TIMEOUT=10

# 日志级别
LOG_LEVEL=INFO
"""
    
    with open('data_source_config.env', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ 已创建优化配置文件: data_source_config.env")

def main():
    """主函数"""
    print("开始优化数据源配置...")
    
    # 优化数据源配置
    provider = optimize_data_sources()
    
    # 创建配置文件
    create_config_file()
    
    print("\n=== 优化完成 ===")
    print("✓ 数据源优先级已优化为: sina -> eastmoney -> akshare_default")
    print("✓ 已创建配置文件 data_source_config.env")
    print("\n建议:")
    print("1. 系统将优先使用新浪和东方财富接口，避免akshare连接问题")
    print("2. 如需启用腾讯等实验性接口，可设置 ENH_ENABLE_EXPERIMENTAL=1")
    print("3. 当前配置已最大程度保证数据获取的稳定性")

if __name__ == "__main__":
    main()