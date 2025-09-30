"""
股票列表管理器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class StockListManager:
    """股票列表管理器 - 管理股票列表和分类"""
    
    def __init__(self, data_dir: str = "data/stocks"):
        self.data_dir = data_dir
        self.stock_lists = {}
        self.stock_categories = {}
        self.market_data_cache = {}
        
        # 创建数据目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        logger.info(f"StockListManager initialized with data_dir: {data_dir}")
    
    def load_stock_list(self, market: str = 'all') -> List[Dict[str, Any]]:
        """加载股票列表"""
        try:
            # 模拟股票数据
            stock_list = []
            
            if market == 'all' or market == 'sh':
                # 上海股票
                stock_list.extend([
                    {'symbol': '600000', 'name': '浦发银行', 'market': 'sh', 'sector': '银行', 'list_date': '1999-11-10'},
                    {'symbol': '600036', 'name': '招商银行', 'market': 'sh', 'sector': '银行', 'list_date': '2002-04-09'},
                    {'symbol': '600519', 'name': '贵州茅台', 'market': 'sh', 'sector': '白酒', 'list_date': '2001-08-27'},
                    {'symbol': '600887', 'name': '伊利股份', 'market': 'sh', 'sector': '食品饮料', 'list_date': '1996-03-12'},
                    {'symbol': '600309', 'name': '万华化学', 'market': 'sh', 'sector': '化工', 'list_date': '2001-01-05'}
                ])
            
            if market == 'all' or market == 'sz':
                # 深圳股票
                stock_list.extend([
                    {'symbol': '000001', 'name': '平安银行', 'market': 'sz', 'sector': '银行', 'list_date': '1991-04-03'},
                    {'symbol': '000002', 'name': '万科A', 'market': 'sz', 'sector': '房地产', 'list_date': '1991-01-29'},
                    {'symbol': '000858', 'name': '五粮液', 'market': 'sz', 'sector': '白酒', 'list_date': '1998-04-27'},
                    {'symbol': '002415', 'name': '海康威视', 'market': 'sz', 'sector': '电子', 'list_date': '2010-05-28'},
                    {'symbol': '300750', 'name': '宁德时代', 'market': 'sz', 'sector': '新能源', 'list_date': '2018-06-11'}
                ])
            
            # 保存到缓存
            self.stock_lists[market] = stock_list
            
            logger.info(f"加载股票列表成功: {market}, 共 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"加载股票列表失败: {e}")
            return []
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取股票信息"""
        try:
            # 从缓存中查找
            for market_list in self.stock_lists.values():
                for stock in market_list:
                    if stock['symbol'] == symbol:
                        return stock
            
            # 如果没找到，返回模拟数据
            stock_info = {
                'symbol': symbol,
                'name': f'股票{symbol}',
                'market': 'sh' if symbol.startswith('6') else 'sz',
                'sector': '综合',
                'list_date': '2000-01-01'
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"获取股票信息失败: {e}")
            return None
    
    def categorize_stocks(self, stock_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """对股票进行分类"""
        try:
            categories = {}
            
            for stock in stock_list:
                sector = stock.get('sector', '其他')
                if sector not in categories:
                    categories[sector] = []
                categories[sector].append(stock)
            
            self.stock_categories = categories
            
            logger.info(f"股票分类完成: 共 {len(categories)} 个分类")
            return categories
            
        except Exception as e:
            logger.error(f"股票分类失败: {e}")
            return {}
    
    def filter_stocks(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据条件过滤股票"""
        try:
            all_stocks = self.load_stock_list('all')
            filtered_stocks = []
            
            for stock in all_stocks:
                # 检查市场
                if 'market' in criteria and stock['market'] not in criteria['market']:
                    continue
                
                # 检查板块
                if 'sector' in criteria and stock['sector'] not in criteria['sector']:
                    continue
                
                # 检查上市时间
                if 'min_list_days' in criteria:
                    list_date = datetime.strptime(stock['list_date'], '%Y-%m-%d')
                    days_listed = (datetime.now() - list_date).days
                    if days_listed < criteria['min_list_days']:
                        continue
                
                # 检查名称关键词
                if 'name_keyword' in criteria and criteria['name_keyword'] not in stock['name']:
                    continue
                
                filtered_stocks.append(stock)
            
            logger.info(f"股票过滤完成: 从 {len(all_stocks)} 只过滤到 {len(filtered_stocks)} 只")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"股票过滤失败: {e}")
            return []
    
    def get_sector_distribution(self) -> Dict[str, int]:
        """获取板块分布"""
        try:
            if not self.stock_categories:
                self.categorize_stocks(self.load_stock_list('all'))
            
            distribution = {}
            for sector, stocks in self.stock_categories.items():
                distribution[sector] = len(stocks)
            
            return distribution
            
        except Exception as e:
            logger.error(f"获取板块分布失败: {e}")
            return {}
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """获取市场统计信息"""
        try:
            all_stocks = self.load_stock_list('all')
            
            # 按市场统计
            sh_stocks = [s for s in all_stocks if s['market'] == 'sh']
            sz_stocks = [s for s in all_stocks if s['market'] == 'sz']
            
            # 按板块统计
            sector_dist = self.get_sector_distribution()
            
            # 按上市时间统计
            list_dates = [datetime.strptime(s['list_date'], '%Y-%m-%d') for s in all_stocks]
            newest_stock = max(list_dates)
            oldest_stock = min(list_dates)
            
            stats = {
                'total_stocks': len(all_stocks),
                'sh_stocks': len(sh_stocks),
                'sz_stocks': len(sz_stocks),
                'sector_count': len(sector_dist),
                'top_sectors': sorted(sector_dist.items(), key=lambda x: x[1], reverse=True)[:5],
                'newest_listing': newest_stock.strftime('%Y-%m-%d'),
                'oldest_listing': oldest_stock.strftime('%Y-%m-%d')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取市场统计信息失败: {e}")
            return {}
    
    def save_stock_list(self, stock_list: List[Dict[str, Any]], filename: str):
        """保存股票列表到文件"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stock_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"股票列表保存成功: {file_path}")
            
        except Exception as e:
            logger.error(f"保存股票列表失败: {e}")
    
    def load_stock_list_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """从文件加载股票列表"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"股票列表文件不存在: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                stock_list = json.load(f)
            
            logger.info(f"从文件加载股票列表成功: {file_path}, 共 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"从文件加载股票列表失败: {e}")
            return []
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索股票"""
        try:
            all_stocks = self.load_stock_list('all')
            results = []
            
            query_lower = query.lower()
            
            for stock in all_stocks:
                # 搜索股票代码
                if query_lower in stock['symbol'].lower():
                    results.append(stock)
                    continue
                
                # 搜索股票名称
                if query_lower in stock['name'].lower():
                    results.append(stock)
                    continue
                
                # 搜索板块
                if query_lower in stock['sector'].lower():
                    results.append(stock)
                    continue
            
            # 限制结果数量
            if len(results) > limit:
                results = results[:limit]
            
            logger.info(f"股票搜索完成: 查询 '{query}' 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"股票搜索失败: {e}")
            return []
    
    def get_watchlist(self) -> List[Dict[str, Any]]:
        """获取自选股列表"""
        try:
            # 从文件加载自选股
            watchlist_file = os.path.join(self.data_dir, 'watchlist.json')
            
            if not os.path.exists(watchlist_file):
                return []
            
            with open(watchlist_file, 'r', encoding='utf-8') as f:
                watchlist = json.load(f)
            
            # 获取详细信息
            detailed_watchlist = []
            for symbol in watchlist:
                stock_info = self.get_stock_info(symbol)
                if stock_info:
                    detailed_watchlist.append(stock_info)
            
            return detailed_watchlist
            
        except Exception as e:
            logger.error(f"获取自选股列表失败: {e}")
            return []
    
    def add_to_watchlist(self, symbol: str) -> bool:
        """添加到自选股"""
        try:
            # 获取当前自选股
            current_watchlist = self.get_watchlist()
            current_symbols = [s['symbol'] for s in current_watchlist]
            
            # 检查是否已存在
            if symbol in current_symbols:
                logger.info(f"股票 {symbol} 已在自选股中")
                return True
            
            # 检查股票是否存在
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                logger.warning(f"股票 {symbol} 不存在")
                return False
            
            # 添加到自选股
            current_symbols.append(symbol)
            
            # 保存到文件
            watchlist_file = os.path.join(self.data_dir, 'watchlist.json')
            with open(watchlist_file, 'w', encoding='utf-8') as f:
                json.dump(current_symbols, f, ensure_ascii=False, indent=2)
            
            logger.info(f"股票 {symbol} 已添加到自选股")
            return True
            
        except Exception as e:
            logger.error(f"添加到自选股失败: {e}")
            return False
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """从自选股中移除"""
        try:
            # 获取当前自选股
            current_watchlist = self.get_watchlist()
            current_symbols = [s['symbol'] for s in current_watchlist]
            
            # 检查是否存在
            if symbol not in current_symbols:
                logger.info(f"股票 {symbol} 不在自选股中")
                return True
            
            # 从自选股中移除
            current_symbols.remove(symbol)
            
            # 保存到文件
            watchlist_file = os.path.join(self.data_dir, 'watchlist.json')
            with open(watchlist_file, 'w', encoding='utf-8') as f:
                json.dump(current_symbols, f, ensure_ascii=False, indent=2)
            
            logger.info(f"股票 {symbol} 已从自选股中移除")
            return True
            
        except Exception as e:
            logger.error(f"从自选股中移除失败: {e}")
            return False
    
    def get_stock_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取股票推荐"""
        try:
            all_stocks = self.load_stock_list('all')
            
            # 简单的推荐逻辑：选择知名股票
            recommended_stocks = []
            
            # 蓝筹股推荐
            blue_chips = ['600036', '600519', '000002', '000858', '002415']
            for symbol in blue_chips:
                stock_info = self.get_stock_info(symbol)
                if stock_info:
                    stock_info['recommendation_reason'] = '优质蓝筹股'
                    recommended_stocks.append(stock_info)
            
            # 成长股推荐
            growth_stocks = ['300750', '002230', '300124']
            for symbol in growth_stocks:
                stock_info = self.get_stock_info(symbol)
                if stock_info:
                    stock_info['recommendation_reason'] = '高成长股'
                    recommended_stocks.append(stock_info)
            
            # 限制推荐数量
            if len(recommended_stocks) > limit:
                recommended_stocks = recommended_stocks[:limit]
            
            logger.info(f"获取股票推荐完成: 共 {len(recommended_stocks)} 只股票")
            return recommended_stocks
            
        except Exception as e:
            logger.error(f"获取股票推荐失败: {e}")
            return []
    
    def reset(self):
        """重置股票列表管理器"""
        self.stock_lists.clear()
        self.stock_categories.clear()
        self.market_data_cache.clear()
        logger.info("股票列表管理器已重置")