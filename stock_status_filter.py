#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import logging
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
import pandas as pd

class StockStatusFilter:
    """股票状态过滤器：识别和过滤退市、ST、停牌等不可交易股票"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 退市股票名称关键词
        self.delisted_keywords = [
            '退市', '退', 'ST退', '*ST退', 
            '终止上市', '摘牌', '已退市'
        ]
        
        # ST股票名称模式
        self.st_patterns = [
            r'^ST\s*',      # ST开头
            r'^\*ST\s*',    # *ST开头
            r'^S\*ST\s*',   # S*ST开头
            r'^SST\s*',      # SST开头
            r'^S\s*',        # S开头（部分停牌股票）
        ]
        
        # 停牌相关关键词
        self.suspended_keywords = [
            '停牌', '暂停上市', '暂停交易', 
            '长期停牌', '临时停牌'
        ]
        
        # 其他不可交易状态
        self.other_invalid_keywords = [
            '清算', '破产', '重整', '注销',
            '吊销', '终止', '解散'
        ]
        
        # 新增：已知退市代码列表（用于修正历史并购/更名导致的名称未包含“退市”的情况）
        # 注意：同时包含带/不带交易所后缀的写法，避免数据源不一致导致漏判
        self.known_delisted_symbols: set[str] = {
            '000022', '000022.SZ',  # 深赤湾A（已退市/更名并不再交易）
        }
    
    def is_delisted_stock(self, name: str, symbol: str = None) -> bool:
        """判断是否为退市股票"""
        if not name and not symbol:
            return False
        
        # 1) 先按已知退市代码列表拦截（最精确）
        if symbol:
            s = str(symbol).strip()
            base = s.split('.')[0]
            if s in self.known_delisted_symbols or base in self.known_delisted_symbols:
                self.logger.debug(f"命中已知退市代码: {s}")
                return True
        
        if not name:
            return False
            
        name = str(name).strip()
        
        # 2) 名称关键词判定
        for keyword in self.delisted_keywords:
            if keyword in name:
                return True
        
        # 3) 特殊代码段判定
        if symbol:
            symbol = str(symbol).strip()
            # 一些退市股票可能有特殊的代码格式
            if symbol.startswith('400') or symbol.startswith('420'):
                return True
        
        return False
    
    def is_st_stock(self, name: str) -> bool:
        """判断是否为ST股票"""
        if not name:
            return False
            
        name = str(name).strip().upper()
        
        # 检查ST模式
        for pattern in self.st_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        
        return False
    
    def is_suspended_stock(self, name: str) -> bool:
        """判断是否为停牌股票（基于名称）"""
        if not name:
            return False
            
        name = str(name).strip()
        
        # 检查停牌关键词
        for keyword in self.suspended_keywords:
            if keyword in name:
                return True
        
        return False
    
    def is_invalid_stock(self, name: str, symbol: str = None) -> bool:
        """判断是否为其他不可交易股票"""
        if not name:
            return False
            
        name = str(name).strip()
        
        # 检查其他无效状态关键词
        for keyword in self.other_invalid_keywords:
            if keyword in name:
                return True
        
        return False

    def _is_b_share_code(self, symbol: str) -> bool:
        """判断是否为B股代码（900/200开头）"""
        if not symbol:
            return False
        code = str(symbol).split('.')[0].strip()
        return code.startswith('900') or code.startswith('200')

    def _is_star_market_code(self, symbol: str) -> bool:
        """判断是否为科创板A股代码（688/689开头）"""
        if not symbol:
            return False
        code = str(symbol).split('.')[0].strip()
        return code.startswith('688') or code.startswith('689')

    def _is_bse_stock_code(self, symbol: str) -> bool:
        """判断是否为北交所股票代码（8开头且以.BJ结尾，或43/83开头）"""
        if not symbol:
            return False
        symbol_str = str(symbol).strip()
        # 北交所股票通常格式为 830001.BJ 或 430001.BJ
        if '.BJ' in symbol_str.upper():
            return True
        # 也可能是8开头的6位数字代码
        code = symbol_str.split('.')[0].strip()
        return (code.startswith('8') and len(code) == 6) or code.startswith('43') or code.startswith('83')

    def _has_no_trades_in_last_days(self, symbol: str, db_manager: Any, days: int = 10) -> bool:
        """判断过去days天内是否完全无成交（volume为0或无记录）"""
        try:
            if db_manager is None or not symbol:
                return False  # 无法判断则不据此过滤
            cutoff = (datetime.today() - timedelta(days=days)).date().isoformat()
            with db_manager.get_conn() as conn:
                df = pd.read_sql_query(
                    "SELECT date, volume FROM prices_daily WHERE symbol = ? AND date >= ? ORDER BY date DESC",
                    conn,
                    params=[symbol, cutoff]
                )
            if df.empty:
                return True  # 过去days天完全无记录 => 视为无成交
            # 将volume缺失视为0
            vol_sum = pd.to_numeric(df['volume'], errors='coerce').fillna(0).sum()
            return vol_sum <= 0
        except Exception as e:
            self.logger.warning(f"检查近{days}天成交量失败 {symbol}: {e}")
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
        
        # 可选：排除北交所股票
        if exclude_bse_stock and symbol and self._is_bse_stock_code(symbol):
            return {'should_filter': True, 'reason': 'bse_stock'}
        
        # 检查退市股票（始终过滤）
        if self.is_delisted_stock(name, symbol):
            return {'should_filter': True, 'reason': 'delisted'}
        
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
        {'symbol': '830001.BJ', 'name': '某北交所股票'},
        {'symbol': '430001.BJ', 'name': '某北交所股票2'},
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