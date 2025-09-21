import re
import logging
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
import pandas as pd


# 已知指数代码列表
KNOWN_INDICES = [
    '000112.SZ',  # 380电信
    '000974.SZ',  # 800金融
    '000914.SZ',  # 300金融
    '000913.SZ',  # 300银行
    '000915.SZ',  # 300地产
    '000916.SZ',  # 300消费
    '000917.SZ',  # 300成长
    '000918.SZ',  # 300价值
    '000919.SZ',  # 300R成长
    '000920.SZ',  # 300R价值
    '000921.SZ',  # 300等权
    '000922.SZ',  # 300分层
]

# 指数名称模式
INDEX_PATTERNS = [
    r'\d+电信',      # 匹配"380电信"这样的格式
    r'\d+金融',      # 匹配"800金融"这样的格式
    r'\d+银行',      # 匹配"300银行"这样的格式
    r'\d+地产',      # 匹配"300地产"这样的格式
    r'\d+消费',      # 匹配"300消费"这样的格式
    r'\d+成长',      # 匹配"300成长"这样的格式
    r'\d+价值',      # 匹配"300价值"这样的格式
    r'\d+R成长',     # 匹配"300R成长"这样的格式
    r'\d+R价值',     # 匹配"300R价值"这样的格式
    r'\d+等权',      # 匹配"300等权"这样的格式
    r'\d+分层',      # 匹配"300分层"这样的格式
    r'^HSI$',        # 恒生指数
    r'^恒生',         # 恒生相关
    r'^上证',         # 上证相关
    r'^中证',         # 中证相关
    r'^沪深',         # 沪深相关
    r'^创业板',       # 创业板相关
    r'^科创板',       # 科创板相关
    r'^科技',         # 科技相关
]


def is_likely_index(symbol: str = None, name: str = None) -> bool:
    """
    判断是否为指数标的
    
    Args:
        symbol: 股票代码
        name: 股票名称
        
    Returns:
        bool: 是否为指数标的
    """
    if symbol and symbol in KNOWN_INDICES:
        return True
    
    if name:
        for pattern in INDEX_PATTERNS:
            if re.search(pattern, name):
                return True
    
    return False


class StockStatusFilter:
    """股票状态过滤器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_st_stock(self, name: str) -> bool:
        """判断是否为ST股票"""
        if not name:
            return False
        name = name.upper()
        return 'ST' in name or '*ST' in name or 'S*ST' in name or 'SST' in name
    
    def is_suspended_stock(self, name: str) -> bool:
        """判断是否为停牌股票"""
        if not name:
            return False
        return '停牌' in name
    
    def is_delisted_stock(self, name: str, symbol: str = None) -> bool:
        """判断是否为退市股票"""
        
        # 检查是否为指数标的
        if is_likely_index(symbol, name):
            return True
        
        if not name and not symbol:
            return False
        
        # 检查退市标识
        name = name.upper() if name else ""
        return '退市' in name or '终止上市' in name or '摘牌' in name
    
    def is_invalid_stock(self, name: str, symbol: str = None) -> bool:
        """判断是否为无效股票"""
        if not name and not symbol:
            return True
        
        # 检查特殊标识
        name = name.upper() if name else ""
        invalid_keywords = ['无效', '错误', '测试', 'DEMO', 'NULL', 'NONE']
        return any(keyword in name for keyword in invalid_keywords)
    
    def _is_b_share_code(self, symbol: str) -> bool:
        """判断是否为B股代码"""
        if not symbol:
            return False
        return symbol.startswith('900') or symbol.startswith('200')
    
    def _is_star_market_code(self, symbol: str) -> bool:
        """判断是否为科创板代码"""
        if not symbol:
            return False
        return symbol.startswith('688') or symbol.startswith('689')
    
    def _is_bse_stock_code(self, symbol: str) -> bool:
        """判断是否为北交所股票代码（已移除，不再支持BJ股票）"""
        if not symbol:
            return False
        # BJ股票已移除，直接返回True表示需要过滤
        return symbol.endswith('.BJ')
    
    def _has_no_trades_in_last_days(self, symbol: str, db_manager, days: int = 10) -> bool:
        """检查过去N天是否无成交"""
        try:
            with db_manager.get_conn() as conn:
                query = """
                    SELECT SUM(COALESCE(volume, 0)) as total_volume
                    FROM prices_daily
                    WHERE symbol = ? AND date >= date('now', '-{} days')
                """.format(days)
                result = conn.execute(query, (symbol,)).fetchone()
                return result is None or result[0] is None or result[0] <= 0
        except Exception as e:
            self.logger.warning(f"检查股票{symbol}近{days}天成交量失败: {e}")
            return False
    
    def should_filter_stock(self, name: str, symbol: str = None, 
                           include_st: bool = True, 
                           include_suspended: bool = True,
                           db_manager: Any = None,
                           include_no_trades_last_n_days: bool = True,
                           last_n_days: int = 10,
                           exclude_b_share: bool = True,
                           exclude_star_market: bool = True,
                           exclude_bse_stock: bool = True) -> Dict[str, Any]:
        """
        综合判断是否应该过滤该股票
        
        Args:
            name: 股票名称
            symbol: 股票代码
            include_st: 是否过滤ST股票
            include_suspended: 是否过滤停牌股票
            db_manager: 可选，若提供则启用基于成交量的停牌/退市检测
            include_no_trades_last_n_days: 是否过滤过去N天无成交的股票
            last_n_days: 过去天数阈值，默认10天
            exclude_b_share: 是否排除B股（900/200开头）
            exclude_star_market: 是否排除科创板（688/689开头）股票
            exclude_bse_stock: 是否排除北交所股票
            
        Returns:
            Dict包含是否过滤和具体原因
        """
        if not name and not symbol:
            return {'should_filter': False, 'reason': None}
        
        # 优先：排除B股代码（900/200开头）
        if exclude_b_share and symbol and self._is_b_share_code(symbol):
            return {'should_filter': True, 'reason': 'b_share'}
        
        # 可选：排除科创板股票（688/689开头）
        if exclude_star_market and symbol and self._is_star_market_code(symbol):
            return {'should_filter': True, 'reason': 'star_market'}
        
        # 强制排除北交所股票（BJ股票已移除，不再支持）
        if symbol and self._is_bse_stock_code(symbol):
            return {'should_filter': True, 'reason': 'bse_stock_removed'}
        
        # 检查退市股票（始终过滤）
        if self.is_delisted_stock(name, symbol):
            return {'should_filter': True, 'reason': 'delisted'}
        
        # 检查指数标的
        if is_likely_index(symbol, name):
            return {'should_filter': True, 'reason': 'index'}
        
        # 检查ST股票
        if include_st and self.is_st_stock(name):
            return {'should_filter': True, 'reason': 'st_stock'}
        
        # 检查停牌股票（基于名称）
        if include_suspended and self.is_suspended_stock(name):
            return {'should_filter': True, 'reason': 'suspended'}
        
        # 检查其他无效股票
        if self.is_invalid_stock(name, symbol):
            return {'should_filter': True, 'reason': 'invalid'}
        
        # 检查过去N天是否无成交（基于数据库 volume）
        if include_no_trades_last_n_days and symbol and db_manager is not None:
            if self._has_no_trades_in_last_days(symbol, db_manager, days=last_n_days):
                return {'should_filter': True, 'reason': f'no_trades_last_{last_n_days}d'}
        
        return {'should_filter': False, 'reason': None}
    
    def filter_stock_list(self, stocks: List[Dict[str, Any]], 
                         include_st: bool = True, 
                         include_suspended: bool = True,
                         db_manager: Any = None,
                         include_no_trades_last_n_days: bool = True,
                         last_n_days: int = 10,
                         exclude_b_share: bool = True,
                         exclude_star_market: bool = True,
                         exclude_bse_stock: bool = True) -> Dict[str, Any]:
        """
        过滤股票列表
        
        Args:
            stocks: 股票列表，每个元素包含name和symbol字段
            include_st: 是否过滤ST股票
            include_suspended: 是否过滤停牌股票
            db_manager: 可选，若提供则启用基于成交量的停牌/退市检测
            include_no_trades_last_n_days: 是否过滤过去N天无成交的股票
            last_n_days: 过去天数阈值，默认10天
            exclude_b_share: 是否排除B股（900/200开头）
            exclude_star_market: 是否排除科创板（688/689开头）股票
            exclude_bse_stock: 是否排除北交所股票
            
        Returns:
            过滤结果和统计信息
        """
        if not stocks:
            return {
                'filtered_stocks': [],
                'removed_stocks': [],
                'statistics': {
                    'total': 0,
                    'filtered': 0,
                    'removed': 0,
                    'removal_reasons': {}
                }
            }
        
        # 预计算：过去N天内无成交的股票集合（一次性SQL）
        no_trades_set: Set[str] = set()
        if include_no_trades_last_n_days and db_manager is not None:
            try:
                symbols = [str(s.get('symbol', '')) for s in stocks if s.get('symbol')]
                symbols = list({s for s in symbols if s})  # 去重
                if symbols:
                    cutoff = (datetime.today() - timedelta(days=last_n_days)).date().isoformat()
                    placeholders = ",".join(["?"] * len(symbols))
                    query = f"""
                        SELECT symbol, SUM(COALESCE(volume,0)) AS vol_sum
                        FROM prices_daily
                        WHERE date >= ? AND symbol IN ({placeholders})
                        GROUP BY symbol
                    """
                    params = [cutoff] + symbols
                    with db_manager.get_conn() as conn:
                        df = pd.read_sql_query(query, conn, params=params)
                    have_rows = set(df['symbol'].astype(str)) if not df.empty else set()
                    zero_or_null = set(df[df['vol_sum'].fillna(0) <= 0]['symbol'].astype(str)) if not df.empty else set()
                    # 无任何行表示该期间无记录 => 视为无成交
                    no_trades_set = zero_or_null | (set(symbols) - have_rows)
            except Exception as e:
                self.logger.warning(f"批量检查近{last_n_days}天成交量失败: {e}")
                no_trades_set = set()
        
        filtered_stocks = []
        removed_stocks = []
        removal_reasons = {}
        
        for stock in stocks:
            name = stock.get('name', '')
            symbol = stock.get('symbol', '')
            
            # 基础状态过滤（不在此处触发逐只DB查询）
            filter_result = self.should_filter_stock(
                name, symbol, include_st, include_suspended,
                db_manager=None,  # 避免逐只查询
                include_no_trades_last_n_days=False,  # 在下方使用预计算集合判断
                exclude_b_share=exclude_b_share,
                exclude_star_market=exclude_star_market,
                exclude_bse_stock=exclude_bse_stock
            )
            
            # 检查指数标的（独立的指数过滤）
            if is_likely_index(symbol, name):
                removed_stocks.append({**stock, 'removal_reason': 'index'})
                removal_reasons['index'] = removal_reasons.get('index', 0) + 1
                continue
            
            if filter_result['should_filter']:
                removed_stocks.append({**stock, 'removal_reason': filter_result['reason']})
                removal_reasons[filter_result['reason']] = removal_reasons.get(filter_result['reason'], 0) + 1
                continue
            
            # 附加：过去N天无成交过滤（基于预计算集合）
            if include_no_trades_last_n_days and symbol and symbol in no_trades_set:
                reason = f'no_trades_last_{last_n_days}d'
                removed_stocks.append({**stock, 'removal_reason': reason})
                removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
                continue
            
            filtered_stocks.append(stock)
        
        return {
            'filtered_stocks': filtered_stocks,
            'removed_stocks': removed_stocks,
            'statistics': {
                'total': len(stocks),
                'filtered': len(filtered_stocks),
                'removed': len(removed_stocks),
                'removal_reasons': removal_reasons
            }
        }
    
    def get_invalid_stocks_from_db(self, db_manager, 
                                  include_st: bool = True, 
                                  include_suspended: bool = True) -> List[Dict[str, Any]]:
        """从数据库中识别并返回无效股票列表（用于清理）"""
        results = []
        try:
            with db_manager.get_conn() as conn:
                df = pd.read_sql_query("SELECT symbol, name FROM stocks", conn)
            stocks = df.to_dict('records') if df is not None and not df.empty else []
            filtered = self.filter_stock_list(stocks, include_st, include_suspended, db_manager=db_manager)
            return filtered.get('removed_stocks', [])
        except Exception as e:
            self.logger.warning(f"从数据库识别无效股票失败: {e}")
            return results
    
    def remove_invalid_stocks_from_db(self, db_manager, 
                                     include_st: bool = True, 
                                     include_suspended: bool = True,
                                     dry_run: bool = True) -> Dict[str, Any]:
        """从数据库中移除无效股票（谨慎使用）"""
        try:
            invalid = self.get_invalid_stocks_from_db(db_manager, include_st, include_suspended)
            removed = 0
            with db_manager.get_conn() as conn:
                for item in invalid:
                    symbol = item.get('symbol')
                    if not symbol:
                        continue
                    if not dry_run:
                        conn.execute("DELETE FROM stocks WHERE symbol = ?", (symbol,))
                        removed += 1
            return {
                'success': True,
                'total_invalid': len(invalid),
                'removed': removed,
                'dry_run': dry_run
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    filter_obj = StockStatusFilter()
    
    # 简单自测
    test_stocks = [
        {'symbol': '600000.SH', 'name': '浦发银行'},
        {'symbol': '000001.SZ', 'name': 'ST平安'},
        {'symbol': '300001.SZ', 'name': '*ST特锐'},
        {'symbol': '600001.SH', 'name': '退市大唐'},
        {'symbol': '000002.SZ', 'name': '万科A'},
        {'symbol': '300002.SZ', 'name': '神州泰岳'},
        {'symbol': '688001.SH', 'name': '某科创板'},
        {'symbol': '689001.SH', 'name': '某科创板2'},
        {'symbol': '900001.SH', 'name': '某B股'},
        {'symbol': '830001.BJ', 'name': '某北交所股票(已移除)'},
        {'symbol': '430001.BJ', 'name': '某北交所股票2(已移除)'}, # BJ股票已移除，测试用例保留
        {'symbol': '000112.SZ', 'name': '380电信'},
        {'symbol': '000974.SZ', 'name': '800金融'},
        {'symbol': '000914.SZ', 'name': '300金融'},
    ]
    
    result = filter_obj.filter_stock_list(test_stocks, exclude_star_market=True, exclude_bse_stock=True)
    
    print("过滤结果:")
    print(f"总数: {result['statistics']['total']}")
    print(f"保留: {result['statistics']['filtered']}")
    print(f"移除: {result['statistics']['removed']}")
    print(f"移除原因: {result['statistics']['removal_reasons']}")
    
    print("\n保留的股票:")
    for stock in result['filtered_stocks']:
        print(f"  {stock['symbol']} - {stock['name']}")
    
    print("\n移除的股票:")
    for stock in result['removed_stocks']:
        print(f"  {stock['symbol']} - {stock['name']} ({stock['removal_reason']})")