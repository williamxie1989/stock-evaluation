#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/xieyongliang/stock-evaluation')

from db import DatabaseManager

def detailed_price_analysis():
    """详细分析价格数据和股票代码匹配问题"""
    db = DatabaseManager()
    
    with db.get_conn() as conn:
        cursor = conn.cursor()
        
        print("=== 详细价格数据分析 ===")
        
        # 1. 检查prices_daily表中的股票代码格式
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM prices_daily 
            WHERE symbol LIKE '60%'
            GROUP BY symbol
            ORDER BY symbol
            LIMIT 20
        """)
        
        price_60_stocks = cursor.fetchall()
        print(f"\nprices_daily表中60开头的股票 (前20只):")
        print(f"{'代码':<12} {'记录数':<8} {'开始日期':<12} {'结束日期':<12}")
        print("-" * 50)
        
        for symbol, count, first_date, last_date in price_60_stocks:
            print(f"{symbol:<12} {count:<8} {first_date:<12} {last_date:<12}")
        
        # 2. 检查stocks表中603开头股票的详细信息
        cursor.execute("""
            SELECT symbol, name, market, board_type
            FROM stocks 
            WHERE symbol LIKE '603%'
            ORDER BY symbol
            LIMIT 10
        """)
        
        stocks_603 = cursor.fetchall()
        print(f"\nstocks表中603开头股票 (前10只):")
        print(f"{'代码':<12} {'名称':<12} {'市场':<6} {'板块':<8}")
        print("-" * 45)
        
        for symbol, name, market, board_type in stocks_603:
            print(f"{symbol:<12} {name[:10]:<12} {market:<6} {board_type or 'None':<8}")
        
        # 3. 检查代码转换逻辑
        print(f"\n=== 代码转换测试 ===")
        test_codes = ['603000.SH', '603001.SH', '603002.SH']
        
        for code in test_codes:
            converted = code.replace('.SH', '.SS') if code.endswith('.SH') else code
            
            # 检查原代码是否在prices_daily中
            cursor.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol = ?", (code,))
            original_count = cursor.fetchone()[0]
            
            # 检查转换后代码是否在prices_daily中
            cursor.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol = ?", (converted,))
            converted_count = cursor.fetchone()[0]
            
            print(f"{code} -> {converted}: 原代码({original_count}条) 转换后({converted_count}条)")
        
        # 4. 查找实际存在的沪市股票代码格式
        cursor.execute("""
            SELECT DISTINCT symbol
            FROM prices_daily 
            WHERE (symbol LIKE '60%.SS' OR symbol LIKE '60%.SH')
            ORDER BY symbol
            LIMIT 20
        """)
        
        actual_sh_codes = cursor.fetchall()
        print(f"\nprices_daily中实际的沪市代码格式:")
        for (symbol,) in actual_sh_codes:
            print(f"  {symbol}")
        
        # 5. 检查是否有其他格式的603股票
        cursor.execute("""
            SELECT DISTINCT symbol
            FROM prices_daily 
            WHERE symbol LIKE '%603%'
            ORDER BY symbol
            LIMIT 10
        """)
        
        any_603_codes = cursor.fetchall()
        print(f"\nprices_daily中包含603的任何代码:")
        if any_603_codes:
            for (symbol,) in any_603_codes:
                print(f"  {symbol}")
        else:
            print("  无任何包含603的代码")
        
        # 6. 统计各种后缀的股票数量
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN symbol LIKE '%.SS' THEN '.SS'
                    WHEN symbol LIKE '%.SH' THEN '.SH'
                    WHEN symbol LIKE '%.SZ' THEN '.SZ'
                    WHEN symbol LIKE '%.HK' THEN '.HK'
                    ELSE 'OTHER'
                END as suffix,
                COUNT(DISTINCT symbol) as count
            FROM prices_daily
            GROUP BY suffix
            ORDER BY count DESC
        """)
        
        suffix_stats = cursor.fetchall()
        print(f"\n=== 股票代码后缀统计 ===")
        print(f"{'后缀':<8} {'股票数':<8}")
        print("-" * 20)
        
        for suffix, count in suffix_stats:
            print(f"{suffix:<8} {count:<8}")
        
        # 7. 检查数据源问题
        print(f"\n=== 数据源分析 ===")
        
        # 检查是否所有沪市股票都缺失
        cursor.execute("""
            SELECT COUNT(*) 
            FROM stocks 
            WHERE market = 'SH'
        """)
        
        total_sh_stocks = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT s.symbol)
            FROM stocks s
            INNER JOIN prices_daily p ON (
                CASE 
                    WHEN s.symbol LIKE '%.SH' THEN REPLACE(s.symbol, '.SH', '.SS')
                    ELSE s.symbol
                END = p.symbol
            )
            WHERE s.market = 'SH'
        """)
        
        sh_with_data = cursor.fetchone()[0]
        
        print(f"沪市股票总数: {total_sh_stocks}只")
        print(f"有价格数据的沪市股票: {sh_with_data}只")
        print(f"缺失数据比例: {(total_sh_stocks - sh_with_data) / total_sh_stocks * 100:.1f}%")
        
        # 8. 建议解决方案
        print(f"\n=== 问题诊断和建议 ===")
        
        if sh_with_data == 0:
            print("🔴 严重问题：所有沪市股票都没有价格数据")
            print("可能原因：")
            print("  1. 数据源不包含沪市股票数据")
            print("  2. 股票代码格式不匹配")
            print("  3. 数据导入过程中过滤了沪市股票")
        elif sh_with_data < total_sh_stocks * 0.5:
            print("🟡 部分问题：大部分沪市股票缺少价格数据")
        else:
            print("🟢 数据基本正常")
        
        print("\n建议解决方案：")
        print("  1. 检查数据获取脚本，确保包含沪市股票")
        print("  2. 验证股票代码格式转换逻辑")
        print("  3. 重新获取或导入沪市股票价格数据")
        print("  4. 检查数据源API是否正常返回沪市数据")

if __name__ == "__main__":
    detailed_price_analysis()