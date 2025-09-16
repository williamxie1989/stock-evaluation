import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from akshare_data_provider import AkshareDataProvider
from db import DatabaseManager
from stock_list_manager import StockListManager
import time

class DataSyncService:
    """
    独立的行情数据同步服务
    - 支持全量和增量数据同步
    - 与选股功能完全解耦
    - 支持多市场数据同步控制
    """
    
    def __init__(self):
        self.data_provider = AkshareDataProvider()
        self.db_manager = DatabaseManager()
        self.stock_manager = StockListManager()
        self.logger = logging.getLogger(__name__)
        
    def sync_market_data(self, 
                        sync_type: str = "incremental",
                        markets: List[str] = None,
                        max_symbols: int = 0,
                        batch_size: int = 10,
                        delay_seconds: float = 1.0) -> Dict[str, Any]:
        """
        同步市场数据
        
        Args:
            sync_type: 同步类型 ("full" | "incremental")
            markets: 要同步的市场列表 ["SH", "SZ", "BJ"] (None表示所有A股市场)
            max_symbols: 最大处理股票数量 (0表示不限制)
            batch_size: 批量处理大小
            delay_seconds: 批次间延时
            
        Returns:
            同步结果字典
        """
        try:
            self.logger.info(f"开始{sync_type}数据同步...")
            
            # 1. 更新股票列表
            stock_update_result = self._update_stock_list(markets)
            if not stock_update_result['success']:
                return stock_update_result
                
            # 2. 获取需要同步的股票列表
            target_stocks = self._get_sync_target_stocks(markets, max_symbols)
            if not target_stocks:
                return {
                    'success': False,
                    'error': '没有找到需要同步的股票',
                    'data': {}
                }
                
            # 3. 执行数据同步
            sync_result = self._execute_data_sync(
                target_stocks, 
                sync_type, 
                batch_size, 
                delay_seconds
            )
            
            # 4. 记录同步状态
            self._record_sync_status(sync_type, sync_result)
            
            return {
                'success': True,
                'sync_type': sync_type,
                'markets': markets or ['SH', 'SZ', 'BJ'],
                'data': sync_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据同步失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def _update_stock_list(self, markets: List[str] = None) -> Dict[str, Any]:
        """更新股票列表"""
        try:
            self.logger.info("正在更新股票列表...")
            
            # 获取全市场股票列表
            all_stocks = self.data_provider.get_all_stock_list()
            if all_stocks is None or all_stocks.empty:
                return {
                    'success': False,
                    'error': '获取股票列表失败'
                }
            
            # 过滤指定市场
            if markets:
                all_stocks = all_stocks[all_stocks['market'].isin(markets)]
            else:
                # 默认只处理A股市场，排除H股
                all_stocks = all_stocks[all_stocks['market'].isin(['SH', 'SZ', 'BJ'])]
            
            # 转换为数据库格式并插入
            stock_rows = []
            for _, row in all_stocks.iterrows():
                # akshare_data_provider返回的是'symbol'字段，不是'code'
                code = row['symbol']
                symbol = f"{code}.{row['market']}"
                stock_rows.append({
                    'symbol': symbol,
                    'name': row['name'],
                    'market': row['market'],
                    'board_type': row.get('board_type', ''),
                    'exchange': row.get('exchange', ''),
                    'ah_pair': row.get('ah_pair', None)
                })
            
            self.db_manager.upsert_stocks(stock_rows)
            
            self.logger.info(f"成功更新 {len(stock_rows)} 只股票信息")
            return {
                'success': True,
                'count': len(stock_rows)
            }
            
        except Exception as e:
            self.logger.error(f"更新股票列表失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_sync_target_stocks(self, markets: List[str] = None, max_symbols: int = 0) -> List[Dict[str, Any]]:
        """获取需要同步的股票列表"""
        try:
            with self.db_manager.get_conn() as conn:
                query = "SELECT symbol, name, market FROM stocks WHERE 1=1"
                params = []
                
                if markets:
                    placeholders = ','.join(['?' for _ in markets])
                    query += f" AND market IN ({placeholders})"
                    params.extend(markets)
                else:
                    # 默认只处理A股
                    query += " AND market IN ('SH', 'SZ', 'BJ')"
                
                query += " ORDER BY symbol"
                
                if max_symbols > 0:
                    query += f" LIMIT {max_symbols}"
                
                df = pd.read_sql_query(query, conn, params=params)
                return df.to_dict('records')
                
        except Exception as e:
            self.logger.error(f"获取同步目标股票失败: {e}")
            return []
    
    def _execute_data_sync(self, 
                          target_stocks: List[Dict[str, Any]], 
                          sync_type: str,
                          batch_size: int,
                          delay_seconds: float) -> Dict[str, Any]:
        """执行数据同步"""
        total_stocks = len(target_stocks)
        processed_stocks = 0
        inserted_daily_rows = 0
        quotes_rows = 0
        failed_stocks = []
        
        self.logger.info(f"开始处理 {total_stocks} 只股票的{sync_type}数据同步")
        
        # 分批处理
        for i in range(0, total_stocks, batch_size):
            batch = target_stocks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            self.logger.info(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch)} 只股票")
            
            for stock in batch:
                try:
                    symbol = stock['symbol']
                    market = stock['market']
                    
                    # 获取历史数据
                    if sync_type == "full":
                        # 全量同步：获取所有历史数据
                        daily_data = self.data_provider.get_stock_data(symbol, period="max")
                    else:
                        # 增量同步：只获取最近的数据
                        daily_data = self.data_provider.get_stock_data(symbol, period="1m")
                    
                    # 处理get_stock_data返回的数据格式
                    if daily_data is not None:
                        # 如果返回的是字典格式，提取stock_data字段
                        if isinstance(daily_data, dict):
                            if 'stock_data' in daily_data and daily_data['stock_data'] is not None:
                                daily_data = daily_data['stock_data']
                            else:
                                daily_data = None
                    
                    if daily_data is not None and not daily_data.empty:
                        # 添加symbol列
                        daily_data = daily_data.copy()
                        daily_data['symbol'] = symbol
                        
                        # 重置索引，将Date从索引转为列
                        daily_data = daily_data.reset_index()
                        
                        # 列名转换：将akshare的大写列名转换为数据库期望的小写列名
                        column_mapping = {
                            'Date': 'date',
                            'Open': 'open', 
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        }
                        daily_data = daily_data.rename(columns=column_mapping)
                        
                        # 插入日线数据
                        rows_inserted = self.db_manager.upsert_prices_daily(daily_data, symbol_col="symbol", date_col="date")
                        inserted_daily_rows += rows_inserted
                    
                    # 暂时跳过实时行情获取（AkshareDataProvider中未实现get_realtime_quote方法）
                    # TODO: 实现实时行情获取功能
                    # realtime_data = self.data_provider.get_realtime_quote(symbol)
                    # if realtime_data and not realtime_data.empty:
                    #     rows_inserted = self.db_manager.insert_quotes_realtime(realtime_data, symbol_col="symbol")
                    #     quotes_rows += rows_inserted
                    
                    processed_stocks += 1
                    self.logger.info(f"已处理 {processed_stocks}/{total_stocks} 只股票: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"处理股票 {stock['symbol']} 失败: {e}")
                    failed_stocks.append({
                        'symbol': stock['symbol'],
                        'error': str(e)
                    })
            
            # 批次间延时
            if i + batch_size < total_stocks and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        return {
            'total_stocks': total_stocks,
            'processed_stocks': processed_stocks,
            'inserted_daily_rows': inserted_daily_rows,
            'quotes_rows': quotes_rows,
            'failed_stocks': failed_stocks,
            'success_rate': processed_stocks / total_stocks if total_stocks > 0 else 0
        }
    
    def _record_sync_status(self, sync_type: str, sync_result: Dict[str, Any]):
        """记录同步状态到数据库"""
        try:
            with self.db_manager.get_conn() as conn:
                cur = conn.cursor()
                
                # 创建同步状态表（如果不存在）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sync_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sync_type TEXT NOT NULL,
                        sync_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_stocks INTEGER,
                        processed_stocks INTEGER,
                        inserted_daily_rows INTEGER,
                        quotes_rows INTEGER,
                        success_rate REAL,
                        status TEXT
                    )
                """)
                
                # 插入同步记录
                cur.execute("""
                    INSERT INTO sync_status 
                    (sync_type, total_stocks, processed_stocks, inserted_daily_rows, 
                     quotes_rows, success_rate, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sync_type,
                    sync_result['total_stocks'],
                    sync_result['processed_stocks'],
                    sync_result['inserted_daily_rows'],
                    sync_result['quotes_rows'],
                    sync_result['success_rate'],
                    'completed'
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"记录同步状态失败: {e}")
    
    def get_last_sync_info(self) -> Dict[str, Any]:
        """获取最后一次同步信息"""
        try:
            with self.db_manager.get_conn() as conn:
                query = """
                    SELECT sync_type, sync_time, total_stocks, processed_stocks, 
                           success_rate, status
                    FROM sync_status 
                    ORDER BY sync_time DESC 
                    LIMIT 1
                """
                
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    return df.iloc[0].to_dict()
                else:
                    return {'message': '暂无同步记录'}
                    
        except Exception as e:
            self.logger.error(f"获取同步信息失败: {e}")
            return {'error': str(e)}
    
    def check_data_freshness(self, markets: List[str] = None) -> Dict[str, Any]:
        """检查数据新鲜度"""
        try:
            with self.db_manager.get_conn() as conn:
                query = """
                    SELECT s.market, MAX(pd.date) as latest_date, COUNT(DISTINCT s.symbol) as stock_count
                    FROM stocks s
                    LEFT JOIN prices_daily pd ON s.symbol = pd.symbol
                    WHERE 1=1
                """
                params = []
                
                if markets:
                    placeholders = ','.join(['?' for _ in markets])
                    query += f" AND s.market IN ({placeholders})"
                    params.extend(markets)
                else:
                    query += " AND s.market IN ('SH', 'SZ', 'BJ')"
                
                query += " GROUP BY s.market ORDER BY s.market"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                # 计算数据新鲜度
                today = datetime.now().date()
                freshness_info = []
                
                for _, row in df.iterrows():
                    latest_date = pd.to_datetime(row['latest_date']).date() if row['latest_date'] else None
                    days_behind = (today - latest_date).days if latest_date else None
                    
                    freshness_info.append({
                        'market': row['market'],
                        'stock_count': row['stock_count'],
                        'latest_date': str(latest_date) if latest_date else None,
                        'days_behind': days_behind,
                        'is_fresh': days_behind is not None and days_behind <= 1
                    })
                
                return {
                    'success': True,
                    'check_time': datetime.now().isoformat(),
                    'markets': freshness_info
                }
                
        except Exception as e:
            self.logger.error(f"检查数据新鲜度失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    sync_service = DataSyncService()
    
    # 检查数据新鲜度
    freshness = sync_service.check_data_freshness()
    print(f"数据新鲜度: {freshness}")
    
    # 执行增量同步
    result = sync_service.sync_market_data(
        sync_type="incremental",
        markets=["SH", "SZ"],
        max_symbols=10,
        batch_size=5
    )
    print(f"同步结果: {result}")