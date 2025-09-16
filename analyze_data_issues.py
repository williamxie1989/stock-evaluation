#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析历史数据不足问题
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_data_issues():
    db_path = '/Users/xieyongliang/stock-evaluation/stock_data.sqlite3'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("历史数据不足问题深入分析")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. 分析stocks表中的symbol格式问题
    print("\n1. 股票代码格式分析")
    print("-" * 40)
    
    cursor.execute("""
        SELECT symbol, name, market, COUNT(*) as count
        FROM stocks
        WHERE symbol IS NULL OR symbol = 'None.SZ' OR symbol = 'None.SS' OR symbol = 'None.BJ'
        GROUP BY symbol, name, market
        ORDER BY count DESC
        LIMIT 20
    """)
    null_symbols = cursor.fetchall()
    
    if null_symbols:
        print("发现异常股票代码:")
        for symbol, name, market, count in null_symbols:
            print(f"  {symbol} ({market}) - {name}: {count} 条记录")
    else:
        print("未发现NULL或异常的股票代码")
    
    # 2. 分析各市场的股票代码格式
    cursor.execute("""
        SELECT 
            market,
            CASE 
                WHEN symbol LIKE '%.SZ' THEN '深交所格式(.SZ)'
                WHEN symbol LIKE '%.SS' THEN '上交所格式(.SS)'
                WHEN symbol LIKE '%.BJ' THEN '北交所格式(.BJ)'
                WHEN symbol LIKE '%' AND symbol NOT LIKE '%.%' THEN '无后缀格式'
                ELSE '其他格式'
            END as format_type,
            COUNT(*) as count
        FROM stocks
        GROUP BY market, format_type
        ORDER BY market, count DESC
    """)
    format_stats = cursor.fetchall()
    
    print("\n各市场股票代码格式分布:")
    current_market = None
    for market, format_type, count in format_stats:
        if market != current_market:
            print(f"\n{market}市场:")
            current_market = market
        print(f"  {format_type}: {count} 只")
    
    # 3. 分析prices_daily表中的数据分布
    print("\n\n2. 历史价格数据分布分析")
    print("-" * 40)
    
    # 按股票代码格式统计历史数据
    cursor.execute("""
        SELECT 
            CASE 
                WHEN p.symbol LIKE '%.SZ' THEN '深交所格式(.SZ)'
                WHEN p.symbol LIKE '%.SS' THEN '上交所格式(.SS)'
                WHEN p.symbol LIKE '%.BJ' THEN '北交所格式(.BJ)'
                WHEN p.symbol NOT LIKE '%.%' THEN '无后缀格式'
                ELSE '其他格式'
            END as format_type,
            COUNT(DISTINCT p.symbol) as stocks_with_data,
            COUNT(*) as total_records,
            AVG(daily_count) as avg_days_per_stock
        FROM prices_daily p
        JOIN (
            SELECT symbol, COUNT(*) as daily_count
            FROM prices_daily
            GROUP BY symbol
        ) dc ON p.symbol = dc.symbol
        GROUP BY format_type
        ORDER BY stocks_with_data DESC
    """)
    price_format_stats = cursor.fetchall()
    
    print("按代码格式统计历史数据:")
    for format_type, stocks, records, avg_days in price_format_stats:
        print(f"  {format_type}: {stocks} 只股票, {records:,} 条记录, 平均 {avg_days:.0f} 天/股")
    
    # 4. 分析stocks表和prices_daily表的匹配情况
    print("\n\n3. 股票表与价格数据匹配分析")
    print("-" * 40)
    
    # 统计各市场的匹配情况
    cursor.execute("""
        SELECT 
            s.market,
            COUNT(s.symbol) as total_stocks,
            COUNT(p.symbol) as stocks_with_price_data,
            ROUND(COUNT(p.symbol) * 100.0 / COUNT(s.symbol), 2) as match_rate
        FROM stocks s
        LEFT JOIN (
            SELECT DISTINCT symbol FROM prices_daily
        ) p ON s.symbol = p.symbol
        GROUP BY s.market
        ORDER BY match_rate DESC
    """)
    match_stats = cursor.fetchall()
    
    print("各市场股票与价格数据匹配率:")
    for market, total, with_data, rate in match_stats:
        print(f"  {market}: {with_data}/{total} ({rate}%)")
    
    # 5. 查找具体的不匹配案例
    print("\n\n4. 不匹配股票详细分析")
    print("-" * 40)
    
    # 沪市主板不匹配的股票
    cursor.execute("""
        SELECT s.symbol, s.name, s.market
        FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        WHERE s.market = 'SH' AND p.symbol IS NULL
        AND s.symbol LIKE '60%'
        LIMIT 10
    """)
    sh_no_data = cursor.fetchall()
    
    print("沪市主板无价格数据的股票样本:")
    for symbol, name, market in sh_no_data:
        print(f"  {symbol} - {name}")
    
    # 深市主板不匹配的股票
    cursor.execute("""
        SELECT s.symbol, s.name, s.market
        FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        WHERE s.market = 'SZ' AND p.symbol IS NULL
        AND s.symbol LIKE '00%'
        LIMIT 10
    """)
    sz_no_data = cursor.fetchall()
    
    print("\n深市主板无价格数据的股票样本:")
    for symbol, name, market in sz_no_data:
        print(f"  {symbol} - {name}")
    
    # 6. 分析价格数据中存在但stocks表中不存在的股票
    print("\n\n5. 孤立价格数据分析")
    print("-" * 40)
    
    cursor.execute("""
        SELECT p.symbol, COUNT(*) as days
        FROM prices_daily p
        LEFT JOIN stocks s ON p.symbol = s.symbol
        WHERE s.symbol IS NULL
        GROUP BY p.symbol
        ORDER BY days DESC
        LIMIT 20
    """)
    orphan_prices = cursor.fetchall()
    
    print("价格数据中存在但stocks表中缺失的股票:")
    for symbol, days in orphan_prices:
        print(f"  {symbol}: {days} 天数据")
    
    # 7. 分析数据源问题
    print("\n\n6. 数据源分析")
    print("-" * 40)
    
    cursor.execute("""
        SELECT source, COUNT(*) as records, COUNT(DISTINCT symbol) as stocks
        FROM prices_daily
        GROUP BY source
        ORDER BY records DESC
    """)
    source_stats = cursor.fetchall()
    
    print("各数据源统计:")
    for source, records, stocks in source_stats:
        print(f"  {source}: {records:,} 条记录, {stocks} 只股票")
    
    # 8. 分析最近数据更新情况
    print("\n\n7. 最近数据更新分析")
    print("-" * 40)
    
    recent_dates = [(1, '1天前'), (7, '7天前'), (30, '30天前'), (90, '90天前')]
    
    for days, desc in recent_dates:
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol)
            FROM prices_daily
            WHERE date >= ?
        """, (cutoff_date,))
        count = cursor.fetchone()[0]
        print(f"  {desc}有数据更新的股票: {count} 只")
    
    # 9. 推荐解决方案
    print("\n\n8. 问题总结与建议")
    print("-" * 40)
    
    print("\n发现的主要问题:")
    print("1. 数据覆盖率低: 78.4%的股票无历史数据")
    print("2. 股票代码格式不一致: stocks表和prices_daily表使用不同格式")
    print("3. 数据孤立: 部分价格数据对应的股票在stocks表中不存在")
    print("4. 市场匹配率差异大: 不同市场的数据完整性差异显著")
    
    print("\n建议解决方案:")
    print("1. 统一股票代码格式 (建议使用带后缀格式如.SZ/.SS/.BJ)")
    print("2. 补充缺失的股票基础信息")
    print("3. 重新同步历史价格数据")
    print("4. 建立数据质量监控机制")
    print("5. 优化数据导入流程，确保格式一致性")
    
    conn.close()
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    analyze_data_issues()