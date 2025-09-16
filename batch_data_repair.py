#!/usr/bin/env python3
"""
批量数据修复脚本
系统性地修复各市场股票的历史价格数据
"""

import logging
import time
from typing import List, Dict, Any
from data_repair_service import DataRepairService
from db import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_market_stats() -> Dict[str, Any]:
    """获取各市场数据统计"""
    db_manager = DatabaseManager()
    with db_manager.get_conn() as conn:
        cursor = conn.cursor()
        
        # 获取各市场股票数量和数据覆盖情况
        query = """
        SELECT 
            s.market,
            s.board_type,
            COUNT(*) as total_stocks,
            COUNT(DISTINCT p.symbol) as stocks_with_data,
            ROUND(COUNT(DISTINCT p.symbol) * 100.0 / COUNT(*), 2) as coverage_rate
        FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        GROUP BY s.market, s.board_type
        ORDER BY s.market, s.board_type
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        stats = []
        for row in results:
            market, board_type, total, with_data, coverage = row
            board_type = board_type or 'N/A'
            stats.append({
                'market': market,
                'board_type': board_type,
                'total_stocks': total,
                'stocks_with_data': with_data,
                'coverage_rate': coverage
            })
        
        return stats

def repair_market_data(market: str, board_type: str = None, batch_size: int = 20, max_stocks: int = 100):
    """修复指定市场的数据"""
    logger.info(f"开始修复 {market} {board_type or ''} 的数据")
    
    repair_service = DataRepairService()
    
    try:
        # 获取需要修复的股票列表
        db_manager = DatabaseManager()
        with db_manager.get_conn() as conn:
            cursor = conn.cursor()
            
            if board_type and board_type != 'N/A':
                query = """
                SELECT s.symbol FROM stocks s
                LEFT JOIN prices_daily p ON s.symbol = p.symbol
                WHERE s.market = ? AND s.board_type = ? AND p.symbol IS NULL
                LIMIT ?
                """
                cursor.execute(query, (market, board_type, max_stocks))
            else:
                query = """
                SELECT s.symbol FROM stocks s
                LEFT JOIN prices_daily p ON s.symbol = p.symbol
                WHERE s.market = ? AND s.board_type IS NULL AND p.symbol IS NULL
                LIMIT ?
                """
                cursor.execute(query, (market, max_stocks))
            
            symbols = [row[0] for row in cursor.fetchall()]
        
        if not symbols:
            logger.info(f"没有找到需要修复的股票")
            return {'success': 0, 'failed': 0}
        
        logger.info(f"找到 {len(symbols)} 只股票需要修复")
        
        # 分批修复
        success_count = 0
        failed_count = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 只股票")
            
            for symbol in batch:
                try:
                    result = repair_service._repair_single_stock(symbol)
                    if result:
                        success_count += 1
                        logger.info(f"成功修复 {symbol} 的数据")
                    else:
                        failed_count += 1
                        logger.warning(f"修复 {symbol} 失败")
                    
                    # 避免请求过于频繁
                    time.sleep(0.2)
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"修复 {symbol} 时发生异常: {str(e)}")
            
            # 批次间休息
            if i + batch_size < len(symbols):
                logger.info(f"批次完成，休息 2 秒...")
                time.sleep(2)
        
        logger.info(f"修复完成: 成功 {success_count} 只，失败 {failed_count} 只")
        return {'success': success_count, 'failed': failed_count}
        
    except Exception as e:
        logger.error(f"修复过程中发生错误: {str(e)}")
        return {'success': 0, 'failed': 0, 'error': str(e)}

def main():
    """主函数"""
    logger.info("开始批量数据修复")
    
    # 获取当前数据统计
    logger.info("获取当前数据统计...")
    stats = get_market_stats()
    
    print("\n=== 当前数据覆盖情况 ===")
    for stat in stats:
        print(f"{stat['market']} {stat['board_type']}: {stat['coverage_rate']}% ({stat['stocks_with_data']}/{stat['total_stocks']})")
    
    # 优先修复覆盖率低的市场
    priority_markets = [
        ('SH', '科创板'),
        ('SZ', '创业板'),
        ('SZ', '中小板'),
        ('SH', '主板'),
        ('SZ', '主板'),
        ('BJ', '北交所'),
    ]
    
    total_results = {'success': 0, 'failed': 0}
    
    for market, board_type in priority_markets:
        # 找到对应的统计信息
        market_stat = next((s for s in stats if s['market'] == market and s['board_type'] == board_type), None)
        
        if market_stat and market_stat['coverage_rate'] < 50:  # 只修复覆盖率低于50%的市场
            logger.info(f"\n开始修复 {market} {board_type}")
            result = repair_market_data(market, board_type, batch_size=15, max_stocks=50)
            
            total_results['success'] += result.get('success', 0)
            total_results['failed'] += result.get('failed', 0)
            
            # 市场间休息
            time.sleep(3)
    
    # 特别处理603开头股票
    logger.info("\n开始修复603开头股票")
    db_manager = DatabaseManager()
    with db_manager.get_conn() as conn:
        cursor = conn.cursor()
        
        query = """
        SELECT s.symbol FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        WHERE s.symbol LIKE '603%' AND p.symbol IS NULL
        LIMIT 30
        """
        cursor.execute(query)
        symbols_603 = [row[0] for row in cursor.fetchall()]
    
    if symbols_603:
        repair_service = DataRepairService()
        success_603 = 0
        failed_603 = 0
        
        for symbol in symbols_603:
            try:
                result = repair_service._repair_single_stock(symbol)
                if result:
                    success_603 += 1
                    logger.info(f"成功修复 {symbol} 的数据")
                else:
                    failed_603 += 1
                    logger.warning(f"修复 {symbol} 失败")
                
                time.sleep(0.3)
                
            except Exception as e:
                failed_603 += 1
                logger.error(f"修复 {symbol} 时发生异常: {str(e)}")
        
        total_results['success'] += success_603
        total_results['failed'] += failed_603
        
        logger.info(f"603开头股票修复完成: 成功 {success_603} 只，失败 {failed_603} 只")
    
    # 最终统计
    logger.info(f"\n=== 批量修复完成 ===")
    logger.info(f"总计: 成功 {total_results['success']} 只，失败 {total_results['failed']} 只")
    
    # 获取修复后的统计
    logger.info("\n获取修复后数据统计...")
    final_stats = get_market_stats()
    
    print("\n=== 修复后数据覆盖情况 ===")
    for stat in final_stats:
        print(f"{stat['market']} {stat['board_type']}: {stat['coverage_rate']}% ({stat['stocks_with_data']}/{stat['total_stocks']})")

if __name__ == "__main__":
    main()