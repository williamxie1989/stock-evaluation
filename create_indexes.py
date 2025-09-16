#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库索引创建脚本
用于提高股票数据查询效率
"""

import sqlite3
import os
from datetime import datetime

class DatabaseIndexManager:
    def __init__(self, db_path='stock_data.sqlite3'):
        self.db_path = db_path
        
    def create_indexes(self):
        """创建数据库索引"""
        if not os.path.exists(self.db_path):
            print(f"❌ 数据库文件不存在: {self.db_path}")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("=== 创建数据库索引 ===")
            
            # 定义索引
            indexes = [
                # prices_daily表索引
                ("idx_prices_daily_symbol", "CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol ON prices_daily(symbol)"),
                ("idx_prices_daily_date", "CREATE INDEX IF NOT EXISTS idx_prices_daily_date ON prices_daily(date)"),
                ("idx_prices_daily_symbol_date", "CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date ON prices_daily(symbol, date)"),
                ("idx_prices_daily_volume", "CREATE INDEX IF NOT EXISTS idx_prices_daily_volume ON prices_daily(volume)"),
                ("idx_prices_daily_market_cap", "CREATE INDEX IF NOT EXISTS idx_prices_daily_market_cap ON prices_daily(market_cap)"),
                
                # stocks表索引
                ("idx_stocks_symbol", "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)"),
                ("idx_stocks_market", "CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market)"),
                ("idx_stocks_board", "CREATE INDEX IF NOT EXISTS idx_stocks_board ON stocks(board)"),
                ("idx_stocks_market_board", "CREATE INDEX IF NOT EXISTS idx_stocks_market_board ON stocks(market, board)"),
                ("idx_stocks_status", "CREATE INDEX IF NOT EXISTS idx_stocks_status ON stocks(status)"),
                
                # financial_data表索引（如果存在）
                ("idx_financial_data_symbol", "CREATE INDEX IF NOT EXISTS idx_financial_data_symbol ON financial_data(symbol)"),
                ("idx_financial_data_date", "CREATE INDEX IF NOT EXISTS idx_financial_data_date ON financial_data(report_date)"),
                ("idx_financial_data_symbol_date", "CREATE INDEX IF NOT EXISTS idx_financial_data_symbol_date ON financial_data(symbol, report_date)"),
            ]
            
            created_count = 0
            for index_name, sql in indexes:
                try:
                    cursor.execute(sql)
                    print(f"✅ 创建索引: {index_name}")
                    created_count += 1
                except sqlite3.Error as e:
                    if "no such table" in str(e).lower():
                        print(f"⚠️  跳过索引 {index_name}: 表不存在")
                    else:
                        print(f"❌ 创建索引 {index_name} 失败: {e}")
            
            # 提交更改
            conn.commit()
            
            print(f"\n=== 索引创建完成 ===")
            print(f"成功创建索引数量: {created_count}")
            
            # 分析表以更新统计信息
            print("\n=== 更新表统计信息 ===")
            tables = ['stocks', 'prices_daily', 'financial_data']
            for table in tables:
                try:
                    cursor.execute(f"ANALYZE {table}")
                    print(f"✅ 分析表: {table}")
                except sqlite3.Error as e:
                    if "no such table" in str(e).lower():
                        print(f"⚠️  跳过表 {table}: 不存在")
                    else:
                        print(f"❌ 分析表 {table} 失败: {e}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ 创建索引时发生错误: {e}")
            return False
    
    def show_indexes(self):
        """显示现有索引"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n=== 现有索引列表 ===")
            cursor.execute("""
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type = 'index' 
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            indexes = cursor.fetchall()
            if indexes:
                for name, sql in indexes:
                    print(f"📋 {name}")
                    if sql:
                        print(f"   SQL: {sql}")
                    print()
            else:
                print("❌ 未找到用户创建的索引")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ 查询索引时发生错误: {e}")
    
    def check_query_performance(self):
        """检查查询性能"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n=== 查询性能测试 ===")
            
            # 测试查询
            test_queries = [
                ("按股票代码查询", "SELECT COUNT(*) FROM prices_daily WHERE symbol = '000001'"),
                ("按日期范围查询", "SELECT COUNT(*) FROM prices_daily WHERE date >= '2024-01-01'"),
                ("按市场查询股票", "SELECT COUNT(*) FROM stocks WHERE market = 'SZ'"),
                ("复合查询", "SELECT COUNT(*) FROM stocks s JOIN prices_daily p ON s.symbol = p.symbol WHERE s.market = 'SH' AND p.date >= '2024-01-01'"),
            ]
            
            for desc, query in test_queries:
                start_time = datetime.now()
                try:
                    cursor.execute(query)
                    result = cursor.fetchone()[0]
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    print(f"✅ {desc}: {result}条记录, 耗时: {duration:.2f}ms")
                except Exception as e:
                    print(f"❌ {desc}: 查询失败 - {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ 性能测试时发生错误: {e}")

def main():
    """主函数"""
    manager = DatabaseIndexManager()
    
    # 创建索引
    success = manager.create_indexes()
    
    if success:
        # 显示索引
        manager.show_indexes()
        
        # 性能测试
        manager.check_query_performance()
    
    print("\n=== 索引管理完成 ===")

if __name__ == "__main__":
    main()