#!/usr/bin/env python3
"""
更新股票筛选逻辑，添加指数标的过滤
在数据同步、选股、展示、模型训练等关键逻辑中同步移除指数标的
"""

import sqlite3
import os
import re

def connect_db():
    """连接数据库"""
    db_path = '/Users/xieyongliang/stock-evaluation/stock_data.sqlite3'
    return sqlite3.connect(db_path)

def get_known_indices():
    """获取已知的指数标的列表"""
    return [
        '000112.SZ',  # 380电信
        '000113.SZ',  # 380公用  
        '000974.SZ',  # 800金融
        '000147.SZ',  # 优势消费
        '000914.SZ',  # 300金融
        '000841.SZ',  # 800医药
        '000808.SZ',  # 医药生物
        '000109.SZ',  # 380医药
        '000994.SZ',  # 全指通信
        '000141.SZ',  # 380动态
        '000145.SZ',  # 优势资源
        '000128.SZ',  # 380基本
    ]

def is_likely_index(symbol, name):
    """判断是否为指数标的"""
    # 已知的指数代码
    known_indices = get_known_indices()
    if symbol in known_indices:
        return True
    
    # 指数名称特征
    index_patterns = [
        r'\d+电信',      # 380电信
        r'\d+公用',      # 380公用
        r'\d+金融',      # 300金融、800金融
        r'\d+医药',      # 380医药、800医药
        r'\d+消费',      # 优势消费
        r'\d+通信',      # 全指通信
        r'\d+动态',      # 380动态
        r'\d+资源',      # 优势资源
        r'\d+基本',      # 380基本
        r'\d+等权',      # 材料等权等
        r'\d+指数',      # 各种指数
        r'优势\w+',     # 优势消费、优势资源等
        r'全指\w+',     # 全指通信等
    ]
    
    for pattern in index_patterns:
        if re.search(pattern, name):
            return True
    
    return False

def update_db_queries():
    """更新数据库查询，添加指数过滤"""
    print("=== 更新数据库查询 ===")
    
    # 这里列出需要在各个模块中添加的过滤条件
    index_filters = [
        "AND symbol NOT IN ('000112.SZ', '000113.SZ', '000974.SZ', '000147.SZ', '000914.SZ', '000841.SZ', '000808.SZ', '000109.SZ', '000994.SZ', '000141.SZ', '000145.SZ', '000128.SZ')",
        "AND name NOT LIKE '%380电信%'",
        "AND name NOT LIKE '%380公用%'", 
        "AND name NOT LIKE '%800金融%'",
        "AND name NOT LIKE '%300金融%'",
        "AND name NOT LIKE '%380医药%'",
        "AND name NOT LIKE '%800医药%'",
        "AND name NOT LIKE '%医药生物%'",
        "AND name NOT LIKE '%优势消费%'",
        "AND name NOT LIKE '%优势资源%'",
        "AND name NOT LIKE '%380动态%'",
        "AND name NOT LIKE '%380基本%'",
        "AND name NOT LIKE '%全指通信%'",
        "AND name NOT LIKE '%等权%'",
        "AND name NOT LIKE '%指数%'",
    ]
    
    print("需要在SQL查询中添加的过滤条件:")
    for filter_condition in index_filters:
        print(f"  {filter_condition}")
    
    return index_filters

def check_existing_filters():
    """检查现有代码中的股票筛选逻辑"""
    print("\\n=== 检查现有股票筛选逻辑 ===")
    
    # 需要检查的关键文件
    key_files = [
        'data_sync_service.py',
        'stock_list_manager.py', 
        'stock_status_filter.py',
        'market_selector_service.py',
        'selector_service.py',
        'db.py',
        'app.py'
    ]
    
    for filename in key_files:
        filepath = f'/Users/xieyongliang/stock-evaluation/{filename}'
        if os.path.exists(filepath):
            print(f"\\n检查 {filename}:")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 查找股票筛选相关代码
            if 'NOT LIKE' in content and ('88%' in content or '指数' in content):
                print("  ✓ 已包含指数过滤逻辑")
            elif 'WHERE' in content and 'symbol' in content:
                print("  ⚠ 包含股票筛选但可能缺少指数过滤")
            else:
                print("  - 未找到明显的股票筛选逻辑")

def update_stock_status_filter():
    """更新股票状态过滤器"""
    print("\\n=== 更新股票状态过滤器 ===")
    
    filter_file = '/Users/xieyongliang/stock-evaluation/stock_status_filter.py'
    if not os.path.exists(filter_file):
        print("  股票状态过滤器文件不存在")
        return
    
    with open(filter_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经包含指数过滤
    if 'is_likely_index' in content or '指数' in content:
        print("  ✓ 股票状态过滤器已包含指数判断逻辑")
        return
    
    # 添加指数判断函数
    new_function = '''
def is_likely_index(symbol, name):
    """判断是否为指数标的"""
    # 已知的指数代码
    known_indices = [
        '000112.SZ', '000113.SZ', '000974.SZ', '000147.SZ', 
        '000914.SZ', '000841.SZ', '000808.SZ', '000109.SZ', 
        '000994.SZ', '000141.SZ', '000145.SZ', '000128.SZ'
    ]
    if symbol in known_indices:
        return True
    
    # 指数名称特征
    import re
    index_patterns = [
        r'\\d+电信', r'\\d+公用', r'\\d+金融', r'\\d+医药',
        r'\\d+消费', r'\\d+通信', r'\\d+动态', r'\\d+资源',
        r'\\d+基本', r'\\d+等权', r'\\d+指数', r'优势\\w+',
        r'全指\\w+'
    ]
    
    for pattern in index_patterns:
        if re.search(pattern, name):
            return True
    
    return False
'''
    
    # 在should_filter_stock函数中添加指数检查
    if 'def should_filter_stock' in content:
        # 在函数开始处添加指数检查
        old_pattern = r'(def should_filter_stock.*?):\n(.*?)(return|if|#)'
        new_content = content.replace(
            'def should_filter_stock',
            new_function + '\\n\\ndef should_filter_stock'
        )
        
        # 在函数体中添加指数过滤
        if 'include_st:' in new_content:
            # 在参数检查后面添加指数检查
            insert_point = new_content.find('if not name and not symbol:')
            if insert_point != -1:
                index_check = '''\n    # 检查是否为指数标的
    if is_likely_index(symbol, name):
        return {'should_filter': True, 'reason': '指数标的'}\n\n'''
                new_content = new_content[:insert_point] + index_check + new_content[insert_point:]
        
        with open(filter_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("  ✓ 已更新股票状态过滤器，添加指数判断逻辑")

def main():
    """主函数"""
    print("开始更新股票筛选逻辑...")
    
    # 1. 分析已知指数
    known_indices = get_known_indices()
    print(f"已知指数标的: {len(known_indices)} 个")
    
    # 2. 更新数据库查询建议
    index_filters = update_db_queries()
    
    # 3. 检查现有筛选逻辑
    check_existing_filters()
    
    # 4. 更新股票状态过滤器
    update_stock_status_filter()
    
    print("\\n=== 更新建议总结 ===")
    print("1. 在数据库查询中添加指数过滤条件")
    print("2. 在股票状态过滤器中添加指数判断逻辑") 
    print("3. 定期检查是否有新的指数标的被错误添加")
    print("4. 在数据同步时避免将指数标的添加为股票")
    
    print("\\n✓ 股票筛选逻辑更新完成")

if __name__ == '__main__':
    main()