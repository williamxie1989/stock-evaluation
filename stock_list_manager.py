import logging
import pandas as pd
from typing import List, Dict, Any
from akshare_data_provider import AkshareDataProvider
from db import DatabaseManager
from stock_status_filter import StockStatusFilter
from datetime import datetime

class StockListManager:
    """股票列表管理器：负责获取、更新和管理全市场股票列表"""
    
    def __init__(self):
        self.data_provider = AkshareDataProvider()
        self.db_manager = DatabaseManager()
        self.stock_filter = StockStatusFilter()
        self.logger = logging.getLogger(__name__)
    
    def update_all_stocks(self) -> Dict[str, Any]:
        """更新全市场股票列表"""
        try:
            self.logger.info("开始更新全市场股票列表...")
            
            # 获取全市场股票列表
            stocks_df = self.data_provider.get_all_stock_list()
            
            if stocks_df.empty:
                self.logger.warning("未获取到股票数据")
                return {'success': False, 'message': '未获取到股票数据', 'count': 0}
            
            # 转换为数据库格式
            stock_records = []
            for _, row in stocks_df.iterrows():
                stock_records.append({
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'market': row['market'],
                    'board_type': row['board_type'],
                    'exchange': row['exchange'],
                    'ah_pair': row.get('ah_pair', None)
                })
            
            # 应用股票过滤器，过滤不需要的股票
            filter_result = self.stock_filter.filter_stock_list(
                stock_records,
                include_st=True,
                include_suspended=True,
                db_manager=self.db_manager,
                exclude_b_share=True,
                exclude_star_market=True,
                exclude_bse_stock=True
            )
            
            filtered_stocks = filter_result['filtered_stocks']
            
            self.logger.info(f"股票过滤结果: 原始{filter_result['statistics']['total']}只, "
                           f"保留{filter_result['statistics']['filtered']}只, "
                           f"移除{filter_result['statistics']['removed']}只")
            if filter_result['statistics']['removal_reasons']:
                self.logger.info(f"移除原因: {filter_result['statistics']['removal_reasons']}")
            
            # 批量插入数据库（只插入过滤后的股票）
            self.db_manager.upsert_stocks(filtered_stocks)
            
            # 统计信息
            stats = self._get_market_statistics(stocks_df)
            
            self.logger.info(f"成功更新 {len(filtered_stocks)} 只股票信息")
            return {
                'success': True,
                'message': f'成功更新 {len(filtered_stocks)} 只股票（原始{len(stock_records)}只，过滤后{len(filtered_stocks)}只）',
                'count': len(filtered_stocks),
                'original_count': len(stock_records),
                'filter_statistics': filter_result['statistics'],
                'statistics': stats,
                'update_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"更新股票列表失败: {e}")
            return {'success': False, 'message': f'更新失败: {str(e)}', 'count': 0}
    
    def get_stocks_by_market(self, market: str = None, board_type: str = None, 
                           apply_filter: bool = True) -> List[Dict[str, Any]]:
        """根据市场和板块获取股票列表"""
        try:
            with self.db_manager.get_conn() as conn:
                query = "SELECT symbol, name, market, board_type, exchange FROM stocks WHERE 1=1"
                params = []
                
                if market:
                    query += " AND market = ?"
                    params.append(market)
                
                if board_type:
                    query += " AND board_type = ?"
                    params.append(board_type)
                
                query += " ORDER BY symbol"
                
                df = pd.read_sql_query(query, conn, params=params)
                stocks = df.to_dict('records')
                
                # 应用股票过滤器（可选）
                if apply_filter and stocks:
                    filter_result = self.stock_filter.filter_stock_list(
                        stocks,
                        include_st=True,
                        include_suspended=True,
                        db_manager=self.db_manager,
                        exclude_b_share=True,
                        exclude_star_market=True,
                        exclude_bse_stock=True
                    )
                    return filter_result['filtered_stocks']
                
                return stocks
                
        except Exception as e:
            self.logger.error(f"查询股票列表失败: {e}")
            return []
    
    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """获取所有股票列表"""
        return self.get_stocks_by_market()
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场概览统计"""
        try:
            with self.db_manager.get_conn() as conn:
                # 按市场统计
                market_stats = pd.read_sql_query(
                    "SELECT market, COUNT(*) as count FROM stocks GROUP BY market ORDER BY count DESC",
                    conn
                )
                
                # 按板块统计
                board_stats = pd.read_sql_query(
                    "SELECT board_type, COUNT(*) as count FROM stocks GROUP BY board_type ORDER BY count DESC",
                    conn
                )
                
                # 按交易所统计
                exchange_stats = pd.read_sql_query(
                    "SELECT exchange, COUNT(*) as count FROM stocks GROUP BY exchange ORDER BY count DESC",
                    conn
                )
                
                # 总数统计
                total_count = pd.read_sql_query("SELECT COUNT(*) as total FROM stocks", conn).iloc[0]['total']
                
                # 转换numpy类型为Python原生类型以支持JSON序列化
                def convert_numpy_types(data):
                    if isinstance(data, list):
                        return [convert_numpy_types(item) for item in data]
                    elif isinstance(data, dict):
                        return {key: convert_numpy_types(value) for key, value in data.items()}
                    elif hasattr(data, 'item'):  # numpy类型
                        return data.item()
                    else:
                        return data
                
                result = {
                    'total_stocks': int(total_count),
                    'by_market': market_stats.to_dict('records'),
                    'by_board_type': board_stats.to_dict('records'),
                    'by_exchange': exchange_stats.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
                
                return convert_numpy_types(result)
                
        except Exception as e:
            self.logger.error(f"获取市场概览失败: {e}")
            return {'total_stocks': 0, 'by_market': [], 'by_board_type': [], 'by_exchange': []}
    
    def _get_market_statistics(self, stocks_df: pd.DataFrame) -> Dict[str, Any]:
        """计算市场统计信息"""
        try:
            stats = {
                'total': len(stocks_df),
                'by_market': stocks_df['market'].value_counts().to_dict(),
                'by_board_type': stocks_df['board_type'].value_counts().to_dict(),
                'by_exchange': stocks_df['exchange'].value_counts().to_dict()
            }
            return stats
        except Exception as e:
            self.logger.error(f"计算统计信息失败: {e}")
            return {}
    
    def get_candidate_stocks(self, market_filter: List[str] = None, 
                           board_filter: List[str] = None, 
                           limit: int = None,
                           apply_filter: bool = True) -> List[str]:
        """获取候选股票代码列表（用于替换原有的种子股票机制）"""
        try:
            with self.db_manager.get_conn() as conn:
                query = "SELECT symbol, name, market, board_type, exchange FROM stocks WHERE 1=1"
                params = []
    
                # 默认过滤：排除行业板块/指数（如88开头）及常见指数/基金类别
                query += " AND symbol NOT LIKE '88%'"
                query += " AND (board_type IS NULL OR board_type NOT IN ('指数','行业指数','板块','基金','ETF'))"
                
                if market_filter:
                    placeholders = ','.join(['?' for _ in market_filter])
                    query += f" AND market IN ({placeholders})"
                    params.extend(market_filter)
                
                if board_filter:
                    placeholders = ','.join(['?' for _ in board_filter])
                    query += f" AND board_type IN ({placeholders})"
                    params.extend(board_filter)
                
                query += " ORDER BY symbol"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=params)
                stocks = df.to_dict('records')
                
                # 应用股票过滤器（可选）
                if apply_filter and stocks:
                    filter_result = self.stock_filter.filter_stock_list(
                        stocks,
                        include_st=True,
                        include_suspended=True,
                        db_manager=self.db_manager,
                        exclude_b_share=True,
                        exclude_star_market=True,
                        exclude_bse_stock=True
                    )
                    return [stock['symbol'] for stock in filter_result['filtered_stocks']]
                
                return [stock['symbol'] for stock in stocks]
                
        except Exception as e:
            self.logger.error(f"获取候选股票失败: {e}")
            return []

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    manager = StockListManager()
    
    # 更新股票列表
    result = manager.update_all_stocks()
    print(f"更新结果: {result}")
    
    # 获取市场概览
    summary = manager.get_market_summary()
    print(f"市场概览: {summary}")
    
    # 获取创业板股票
    gem_stocks = manager.get_stocks_by_market(board_type='创业板')
    print(f"创业板股票数量: {len(gem_stocks)}")