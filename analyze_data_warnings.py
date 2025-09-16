#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/xieyongliang/stock-evaluation')

from db import DatabaseManager

def analyze_insufficient_data():
    """分析历史数据不足的股票情况"""
    db = DatabaseManager()
    
    print("=== 股票历史数据充足性分析 ===")
    
    with db.get_conn() as conn:
        cursor = conn.cursor()
        
        # 统计各市场股票的价格数据情况
        cursor.execute("""
            SELECT 
                s.market,
                s.board_type,
                COUNT(s.symbol) as total_stocks,
                COUNT(CASE WHEN price_count >= 30 THEN 1 END) as sufficient_data,
                COUNT(CASE WHEN price_count < 30 AND price_count > 0 THEN 1 END) as insufficient_data,
                COUNT(CASE WHEN price_count = 0 THEN 1 END) as no_data
            FROM stocks s
            LEFT JOIN (
                SELECT 
                    CASE 
                        WHEN symbol LIKE '%.SH' THEN REPLACE(symbol, '.SH', '.SS')
                        ELSE symbol
                    END as converted_symbol,
                    symbol as original_symbol,
                    COUNT(*) as price_count
                FROM prices_daily 
                GROUP BY symbol
            ) p ON (
                CASE 
                    WHEN s.symbol LIKE '%.SH' THEN REPLACE(s.symbol, '.SH', '.SS')
                    ELSE s.symbol
                END = p.converted_symbol
            )
            GROUP BY s.market, s.board_type
            ORDER BY s.market, s.board_type
        """)
        
        results = cursor.fetchall()
        
        print("\n按市场和板块统计:")
        print(f"{'市场':<8} {'板块':<12} {'总数':<6} {'充足':<6} {'不足':<6} {'无数据':<6} {'充足率':<8}")
        print("-" * 60)
        
        total_stocks = 0
        total_sufficient = 0
        
        for row in results:
            market, board_type, total, sufficient, insufficient, no_data = row
            sufficient_rate = (sufficient / total * 100) if total > 0 else 0
            
            total_stocks += total
            total_sufficient += sufficient
            
            print(f"{market:<8} {board_type or 'None':<12} {total:<6} {sufficient:<6} {insufficient:<6} {no_data:<6} {sufficient_rate:<7.1f}%")
        
        overall_rate = (total_sufficient / total_stocks * 100) if total_stocks > 0 else 0
        print("-" * 60)
        print(f"{'总计':<21} {total_stocks:<6} {total_sufficient:<6} {'':<6} {'':<6} {overall_rate:<7.1f}%")
        
        # 查看603开头股票的具体情况（用户提到的警告中的股票）
        print("\n=== 603开头股票分析 ===")
        
        cursor.execute("""
            SELECT 
                s.symbol,
                s.name,
                s.market,
                s.board_type,
                COALESCE(p.price_count, 0) as price_count,
                p.latest_date
            FROM stocks s
            LEFT JOIN (
                SELECT 
                    CASE 
                        WHEN symbol LIKE '%.SH' THEN REPLACE(symbol, '.SH', '.SS')
                        ELSE symbol
                    END as converted_symbol,
                    symbol as original_symbol,
                    COUNT(*) as price_count,
                    MAX(date) as latest_date
                FROM prices_daily 
                GROUP BY symbol
            ) p ON (
                CASE 
                    WHEN s.symbol LIKE '%.SH' THEN REPLACE(s.symbol, '.SH', '.SS')
                    ELSE s.symbol
                END = p.converted_symbol
            )
            WHERE s.symbol LIKE '603%'
            ORDER BY s.symbol
            LIMIT 20
        """)
        
        stocks_603 = cursor.fetchall()
        
        print(f"603开头股票样例 (前20只):")
        print(f"{'股票代码':<12} {'股票名称':<12} {'市场':<6} {'板块':<8} {'数据天数':<8} {'最新日期':<12}")
        print("-" * 70)
        
        insufficient_603 = 0
        for stock in stocks_603:
            symbol, name, market, board_type, price_count, latest_date = stock
            if price_count < 30:
                insufficient_603 += 1
            
            status = "充足" if price_count >= 30 else "不足" if price_count > 0 else "无数据"
            print(f"{symbol:<12} {name[:10]:<12} {market:<6} {board_type or 'None':<8} {price_count:<8} {latest_date or 'N/A':<12} [{status}]")
        
        # 统计603开头股票总数
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE symbol LIKE '603%'")
        total_603 = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM stocks s
            LEFT JOIN (
                SELECT 
                    CASE 
                        WHEN symbol LIKE '%.SH' THEN REPLACE(symbol, '.SH', '.SS')
                        ELSE symbol
                    END as converted_symbol,
                    COUNT(*) as price_count
                FROM prices_daily 
                GROUP BY symbol
            ) p ON (
                CASE 
                    WHEN s.symbol LIKE '%.SH' THEN REPLACE(s.symbol, '.SH', '.SS')
                    ELSE s.symbol
                END = p.converted_symbol
            )
            WHERE s.symbol LIKE '603%' AND COALESCE(p.price_count, 0) < 30
        """)
        
        insufficient_603_total = cursor.fetchone()[0]
        
        print(f"\n603开头股票统计:")
        print(f"  总数: {total_603}只")
        print(f"  数据不足: {insufficient_603_total}只")
        print(f"  不足比例: {insufficient_603_total/total_603*100:.1f}%")
        
        # 分析数据不足的原因
        print("\n=== 数据不足原因分析 ===")
        
        # 检查stocks表的结构
        cursor.execute("PRAGMA table_info(stocks)")
        stocks_columns = [col[1] for col in cursor.fetchall()]
        print(f"stocks表字段: {', '.join(stocks_columns)}")
        
        # 检查prices_daily表中是否有603开头的数据
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM prices_daily 
            WHERE symbol LIKE '603%'
        """)
        
        prices_603_count = cursor.fetchone()[0]
        print(f"prices_daily表中603开头股票数: {prices_603_count}只")
        
        # 检查是否有.SS后缀的603股票数据
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM prices_daily 
            WHERE symbol LIKE '603%.SS'
        """)
        
        prices_603_ss_count = cursor.fetchone()[0]
        print(f"prices_daily表中603%.SS格式股票数: {prices_603_ss_count}只")
        
        # 检查价格数据的时间范围
        cursor.execute("""
            SELECT 
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(DISTINCT date) as trading_days
            FROM prices_daily
        """)
        
        date_range = cursor.fetchone()
        print(f"价格数据时间范围: {date_range[0]} 到 {date_range[1]} ({date_range[2]}个交易日)")
        
        print("\n=== 建议 ===")
        if insufficient_603_total > total_603 * 0.5:
            print("⚠️  603开头股票数据不足比例较高，建议:")
            print("   1. 检查数据源是否包含这些股票的历史数据")
            print("   2. 考虑降低历史数据要求的阈值（当前30天）")
            print("   3. 针对新上市股票使用不同的数据要求")
        else:
            print("✅ 603开头股票数据情况基本正常")

if __name__ == "__main__":
    analyze_insufficient_data()