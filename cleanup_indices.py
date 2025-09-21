#!/usr/bin/env python3
"""
清理数据库中的指数标的
"""

import sqlite3

def cleanup_indices():
    """
    清理数据库中的指数数据
    """
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # 需要清理的指数代码
    indices_to_remove = [
        '000001',  # 上交所指数 - 000001
    ]
    
    print(f"准备清理 {len(indices_to_remove)} 个指数标的...")
    
    for symbol in indices_to_remove:
        try:
            # 删除价格数据
            cursor.execute("DELETE FROM stock_prices WHERE symbol = ?", (symbol,))
            deleted_prices = cursor.rowcount
            
            # 删除其他相关表的数据（如果有的话）
            tables_to_check = ['backtest_results', 'parameter_adjustments', 'strategy_adaptations']
            total_deleted = deleted_prices
            
            for table in tables_to_check:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE symbol = ?", (symbol,))
                    total_deleted += cursor.rowcount
                except sqlite3.OperationalError:
                    # 表不存在，跳过
                    pass
            
            print(f"已清理 {symbol}: 删除 {total_deleted} 条记录")
            
        except Exception as e:
            print(f"清理 {symbol} 失败: {e}")
    
    conn.commit()
    conn.close()
    
    print("指数清理完成！")

if __name__ == "__main__":
    cleanup_indices()
