#!/usr/bin/env python3
"""
清理prices_daily表中的重复数据
"""
import sqlite3
import pandas as pd

def cleanup_duplicate_prices(db_path='stock_data.db'):
    """清理prices_daily表中的重复数据，保留每组(symbol, date)中的最新记录"""
    print("开始清理prices_daily表中的重复数据...")
    
    with sqlite3.connect(db_path) as conn:
        # 首先检查有多少重复数据
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, COUNT(*) as cnt 
            FROM prices_daily 
            GROUP BY symbol, date 
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        print(f"发现 {len(duplicates)} 组重复数据")
        
        if duplicates:
            # 创建临时表存储要保留的记录（使用rowid）
            cursor.execute("""
                CREATE TEMPORARY TABLE prices_daily_temp AS
                SELECT * FROM prices_daily
                WHERE rowid IN (
                    SELECT MAX(rowid) 
                    FROM prices_daily 
                    GROUP BY symbol, date
                )
            """)
            
            # 删除原表中的所有数据
            cursor.execute("DELETE FROM prices_daily")
            
            # 将清理后的数据插回原表
            cursor.execute("""
                INSERT INTO prices_daily 
                SELECT * FROM prices_daily_temp
            """)
            
            # 删除临时表
            cursor.execute("DROP TABLE prices_daily_temp")
            
            conn.commit()
            print("重复数据清理完成")
        else:
            print("没有发现重复数据")
        
        # 创建唯一索引
        try:
            cursor.execute("CREATE UNIQUE INDEX idx_prices_daily_symbol_date ON prices_daily(symbol, date)")
            conn.commit()
            print("唯一索引创建成功")
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                print("唯一索引已存在")
            else:
                print(f"创建索引失败: {e}")

if __name__ == "__main__":
    cleanup_duplicate_prices()