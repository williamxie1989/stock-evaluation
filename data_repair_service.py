import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
from db import DatabaseManager
from enhanced_data_provider import EnhancedDataProvider

class DataRepairService:
    """
    数据修复服务
    - 检测缺失的历史价格数据
    - 使用多数据源补充缺失数据
    - 批量修复指定市场的数据
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_provider = EnhancedDataProvider()
        self.logger = logging.getLogger(__name__)
        
    def analyze_data_gaps(self) -> Dict[str, Any]:
        """分析各市场板块的数据缺失情况"""
        with self.db_manager.get_conn() as conn:
            cursor = conn.cursor()
            
            # 查询各市场板块的股票数量
            stock_query = """
                SELECT market, board_type, COUNT(*) as total_stocks
                FROM stocks 
                WHERE market IN ('SH', 'SZ', 'BJ')
                GROUP BY market, board_type
                ORDER BY market, board_type
            """
            cursor.execute(stock_query)
            stock_counts = cursor.fetchall()
            
            # 查询有价格数据的股票数量
            price_query = """
                SELECT s.market, s.board_type, COUNT(DISTINCT s.symbol) as stocks_with_data
                FROM stocks s
                JOIN prices_daily p ON s.symbol = p.symbol
                WHERE s.market IN ('SH', 'SZ', 'BJ')
                GROUP BY s.market, s.board_type
                ORDER BY s.market, s.board_type
            """
            cursor.execute(price_query)
            price_counts = cursor.fetchall()
            
        # 整理数据
        analysis = {}
        
        # 创建价格数据字典便于查找
        price_dict = {}
        for market, board_type, count in price_counts:
            key = f"{market}_{board_type}"
            price_dict[key] = count
            
        # 分析每个市场板块
        for market, board_type, total in stock_counts:
            key = f"{market}_{board_type}"
            with_data = price_dict.get(key, 0)
            missing = total - with_data
            
            analysis[f"{market}_{board_type}"] = {
                'market': market,
                'board_type': board_type,
                'total_stocks': total,
                'stocks_with_data': with_data,
                'missing_data_stocks': missing,
                'data_coverage': (with_data / total * 100) if total > 0 else 0
            }
            
        return analysis
        
    def get_stocks_without_data(self, market: str, board_type: str, limit: int = None) -> List[str]:
        """获取指定市场板块中没有价格数据的股票列表"""
        with self.db_manager.get_conn() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT s.symbol
                FROM stocks s
                LEFT JOIN prices_daily p ON s.symbol = p.symbol
                WHERE s.market = ? AND s.board_type = ? AND p.symbol IS NULL
                ORDER BY s.symbol
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query, (market, board_type))
            return [row[0] for row in cursor.fetchall()]
            
    def repair_market_data(self, market: str, board_type: str, batch_size: int = 10, max_stocks: int = None) -> Dict[str, Any]:
        """修复指定市场板块的数据"""
        self.logger.info(f"开始修复 {market} {board_type} 的历史价格数据")
        
        # 获取需要修复的股票列表
        stocks_to_repair = self.get_stocks_without_data(market, board_type, max_stocks)
        
        if not stocks_to_repair:
            self.logger.info(f"{market} {board_type} 没有需要修复的股票")
            return {
                'success': True,
                'message': '没有需要修复的股票',
                'repaired_count': 0,
                'failed_count': 0
            }
            
        self.logger.info(f"找到 {len(stocks_to_repair)} 只股票需要修复数据")
        
        repaired_count = 0
        failed_count = 0
        failed_stocks = []
        
        # 分批处理
        for i in range(0, len(stocks_to_repair), batch_size):
            batch = stocks_to_repair[i:i + batch_size]
            self.logger.info(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 只股票")
            
            for symbol in batch:
                try:
                    success = self._repair_single_stock(symbol)
                    if success:
                        repaired_count += 1
                        self.logger.info(f"成功修复 {symbol} 的数据")
                    else:
                        failed_count += 1
                        failed_stocks.append(symbol)
                        self.logger.warning(f"修复 {symbol} 失败")
                        
                except Exception as e:
                    failed_count += 1
                    failed_stocks.append(symbol)
                    self.logger.error(f"修复 {symbol} 时出错: {e}")
                    
                # 添加延迟避免请求过于频繁
                time.sleep(0.2)
                
            # 批次间稍长延迟
            if i + batch_size < len(stocks_to_repair):
                time.sleep(1)
                
        result = {
            'success': True,
            'message': f'修复完成: 成功 {repaired_count} 只，失败 {failed_count} 只',
            'repaired_count': repaired_count,
            'failed_count': failed_count,
            'failed_stocks': failed_stocks
        }
        
        self.logger.info(f"修复完成: {result['message']}")
        return result
        
    def _repair_single_stock(self, symbol: str) -> bool:
        """修复单只股票的历史数据"""
        try:
            # 获取历史数据
            data = self.data_provider.get_stock_historical_data(symbol, "2y")
            
            if data is None or data.empty:
                self.logger.warning(f"无法获取 {symbol} 的历史数据")
                return False
                
            if len(data) < 30:
                self.logger.warning(f"{symbol} 的历史数据不足30条")
                return False
                
            # 保存到数据库
            self._save_price_data(symbol, data)
            return True
            
        except Exception as e:
            self.logger.error(f"修复 {symbol} 时出错: {e}")
            return False
            
    def _save_price_data(self, symbol: str, data: pd.DataFrame):
        """保存价格数据到数据库"""
        with self.db_manager.get_conn() as conn:
            cursor = conn.cursor()
            
            # 准备插入数据
            insert_query = """
                INSERT OR REPLACE INTO prices_daily 
                (symbol, date, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            records = []
            for _, row in data.iterrows():
                records.append((
                    symbol,
                    row['date'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']) if pd.notna(row['volume']) else 0,
                    'enhanced_provider'  # 数据来源标识
                ))
                
            cursor.executemany(insert_query, records)
            conn.commit()
            
            self.logger.debug(f"为 {symbol} 保存了 {len(records)} 条价格数据")
            
    def repair_all_markets(self, max_stocks_per_market: int = 50) -> Dict[str, Any]:
        """修复所有市场的数据"""
        self.logger.info("开始修复所有市场的历史价格数据")
        
        # 分析数据缺失情况
        analysis = self.analyze_data_gaps()
        
        results = {}
        total_repaired = 0
        total_failed = 0
        
        # 优先修复数据覆盖率低的市场
        markets_to_repair = [
            ('SH', '主板'),
            ('SH', '科创板'),
            ('BJ', '北交所'),
            ('SZ', '创业板'),
            ('SZ', '中小板')
        ]
        
        for market, board_type in markets_to_repair:
            key = f"{market}_{board_type}"
            if key in analysis and analysis[key]['missing_data_stocks'] > 0:
                self.logger.info(f"\n开始修复 {market} {board_type}")
                
                result = self.repair_market_data(
                    market, 
                    board_type, 
                    batch_size=5, 
                    max_stocks=max_stocks_per_market
                )
                
                results[key] = result
                total_repaired += result['repaired_count']
                total_failed += result['failed_count']
                
                # 市场间延迟
                time.sleep(2)
                
        return {
            'success': True,
            'message': f'全部修复完成: 成功 {total_repaired} 只，失败 {total_failed} 只',
            'total_repaired': total_repaired,
            'total_failed': total_failed,
            'market_results': results
        }
        
    def verify_repair_results(self) -> Dict[str, Any]:
        """验证修复结果"""
        self.logger.info("验证数据修复结果")
        
        # 重新分析数据缺失情况
        analysis_after = self.analyze_data_gaps()
        
        summary = {}
        for key, info in analysis_after.items():
            summary[key] = {
                'market': info['market'],
                'board_type': info['board_type'],
                'total_stocks': info['total_stocks'],
                'stocks_with_data': info['stocks_with_data'],
                'data_coverage': round(info['data_coverage'], 2)
            }
            
        return {
            'success': True,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    repair_service = DataRepairService()
    
    # 分析数据缺失情况
    print("\n=== 数据缺失分析 ===")
    analysis = repair_service.analyze_data_gaps()
    for key, info in analysis.items():
        print(f"{info['market']} {info['board_type']}: "
              f"总数 {info['total_stocks']}, "
              f"有数据 {info['stocks_with_data']}, "
              f"覆盖率 {info['data_coverage']:.1f}%")
    
    # 测试修复科创板数据（少量股票）
    print("\n=== 测试修复科创板数据 ===")
    result = repair_service.repair_market_data('SH', '科创板', batch_size=3, max_stocks=5)
    print(f"修复结果: {result['message']}")
    
    # 验证结果
    print("\n=== 验证修复结果 ===")
    verification = repair_service.verify_repair_results()
    for key, info in verification['summary'].items():
        if 'SH' in key and '科创板' in key:
            print(f"{info['market']} {info['board_type']}: "
                  f"总数 {info['total_stocks']}, "
                  f"有数据 {info['stocks_with_data']}, "
                  f"覆盖率 {info['data_coverage']}%")