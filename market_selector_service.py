import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from db import DatabaseManager
from selector_service import IntelligentStockSelector
from stock_status_filter import StockStatusFilter
from enum import Enum

class MarketType(Enum):
    """市场类型枚举"""
    SH_MAIN = "沪市主板"  # 上海主板
    SZ_MAIN = "深市主板"  # 深圳主板  
    CHINEXT = "创业板"    # 创业板
    # BEIJING = "北交所"   # 北京证券交易所 - BJ股票已移除
    HK = "H股"          # 港股（暂不同步）

class MarketSelectorService:
    """
    多市场选股服务
    - 支持多市场范围选择
    - 与数据同步功能完全解耦
    - 基于现有数据执行选股逻辑
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.stock_selector = IntelligentStockSelector()
        self.stock_filter = StockStatusFilter()
        self.logger = logging.getLogger(__name__)
        
        # 市场映射配置 - 修复后使用exchange字段匹配
        self.market_mapping = {
            MarketType.SH_MAIN: {'exchange': '上海证券交易所', 'market': 'SH', 'board_types': ['主板']},
            MarketType.SZ_MAIN: {'exchange': '深圳证券交易所', 'market': 'SZ', 'board_types': ['主板']},
            MarketType.CHINEXT: {'exchange': '深圳证券交易所', 'market': 'SZ', 'board_types': ['创业板']},
            # MarketType.BEIJING: {'exchange': '北京证券交易所', 'market': 'BJ', 'board_types': ['北交所']},  # BJ股票已移除
            MarketType.HK: {'exchange': '香港证券交易所', 'market': 'HK', 'board_types': ['主板', '创业板']}
        }
    
    def get_available_markets(self) -> Dict[str, Any]:
        """获取可用的市场列表及其股票数量"""
        try:
            market_info = {}
            
            for market_type, config in self.market_mapping.items():
                # 查询该市场的股票数量
                stock_count = self._get_market_stock_count(
                    config['exchange'],
                    config['market'], 
                    config['board_types']
                )
                
                # 检查数据新鲜度
                data_freshness = self._check_market_data_freshness(config['market'])
                
                market_info[market_type.value] = {
                    'market_code': config['market'],
                    'board_types': config['board_types'],
                    'stock_count': int(stock_count),  # 确保是Python int类型
                    'data_freshness': data_freshness,
                    'available': bool(stock_count > 0 and data_freshness['is_fresh']),
                    'enabled': bool(market_type != MarketType.HK)  # H股暂时不启用
                }
            
            return {
                'success': True,
                'markets': market_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取可用市场失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_stocks_by_markets(self, selected_markets: List[str]) -> List[str]:
        """
        根据选定市场获取股票列表
        
        Args:
            selected_markets: 选定的市场列表，如 ["A股主板", "创业板", "科创板"]
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 市场名称映射
            market_mapping = {
                "A股主板": ["沪市主板", "深市主板"],
                "创业板": ["创业板"],
                "科创板": ["科创板"],
                "北交所": ["北交所"],  # BJ股票已移除
                "沪市主板": ["沪市主板"],
                "深市主板": ["深市主板"]
            }
            
            # 展开市场名称
            expanded_markets = []
            for market in selected_markets:
                if market in market_mapping:
                    expanded_markets.extend(market_mapping[market])
                else:
                    expanded_markets.append(market)
            
            # 去重
            expanded_markets = list(set(expanded_markets))
            
            # 调用现有的私有方法获取候选股票
            stock_symbols = self._get_candidate_stocks_by_markets(expanded_markets)
            
            self.logger.info(f"从市场 {selected_markets} 获取到 {len(stock_symbols)} 只股票")
            return stock_symbols
            
        except Exception as e:
            self.logger.error(f"获取市场股票列表失败: {e}")
            return []

    def select_stocks_by_markets(self, 
                               selected_markets: List[str],
                               selection_criteria: Dict[str, Any] = None,
                               top_n: int = 20) -> Dict[str, Any]:
        """
        基于选定市场进行智能选股
        
        Args:
            selected_markets: 选定的市场列表 (MarketType的value值)
            selection_criteria: 选股条件
            top_n: 返回股票数量
            
        Returns:
            选股结果
        """
        try:
            self.logger.info(f"开始基于市场 {selected_markets} 进行选股")
            
            # 验证选定的市场
            valid_markets = [m.value for m in MarketType if m != MarketType.HK]  # 排除H股和已移除的BJ股
            invalid_markets = [m for m in selected_markets if m not in valid_markets]
            
            if invalid_markets:
                return {
                    'success': False,
                    'error': f'无效的市场选择: {invalid_markets}',
                    'valid_markets': valid_markets
                }
            
            # 2. 获取候选股票池
            candidate_stocks = self._get_candidate_stocks_by_markets(selected_markets)
            
            if not candidate_stocks:
                return {
                    'success': False,
                    'error': '在选定市场中未找到候选股票',
                    'selected_markets': selected_markets
                }
            
            # 3. 执行选股逻辑
            selection_result = self._execute_stock_selection(
                candidate_stocks, 
                selection_criteria, 
                top_n
            )
            
            # 4. 整理返回结果
            return {
                'success': True,
                'selected_markets': selected_markets,
                'candidate_count': len(candidate_stocks),
                'selection_criteria': selection_criteria or {},
                'results': selection_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"多市场选股失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_market_stock_count(self, exchange: str, market: str, board_types: List[str]) -> int:
        """获取指定市场的股票数量"""
        try:
            with self.db_manager.get_conn() as conn:
                query = "SELECT COUNT(*) as count FROM stocks WHERE exchange = ? AND market = ?"
                params = [exchange, market]
                
                if board_types:
                    placeholders = ','.join(['?' for _ in board_types])
                    query += f" AND board_type IN ({placeholders})"
                    params.extend(board_types)
                
                df = pd.read_sql_query(query, conn, params=params)
                return df.iloc[0]['count'] if not df.empty else 0
                
        except Exception as e:
            self.logger.error(f"获取市场股票数量失败: {e}")
            return 0
    
    def _check_market_data_freshness(self, market: str) -> Dict[str, Any]:
        """检查市场数据新鲜度"""
        try:
            with self.db_manager.get_conn() as conn:
                query = """
                    SELECT MAX(pd.date) as latest_date
                    FROM stocks s
                    LEFT JOIN prices_daily pd ON s.symbol = pd.symbol
                    WHERE s.market = ?
                """
                
                df = pd.read_sql_query(query, conn, params=[market])
                
                if not df.empty and df.iloc[0]['latest_date']:
                    latest_date = pd.to_datetime(df.iloc[0]['latest_date']).date()
                    today = datetime.now().date()
                    days_behind = (today - latest_date).days
                    
                    return {
                        'latest_date': str(latest_date),
                        'days_behind': int(days_behind),
                        'is_fresh': bool(days_behind <= 1)
                    }
                else:
                    return {
                        'latest_date': None,
                        'days_behind': None,
                        'is_fresh': False
                    }
                    
        except Exception as e:
            self.logger.error(f"检查市场数据新鲜度失败: {e}")
            return {
                'latest_date': None,
                'days_behind': None,
                'is_fresh': False
            }
    
    def _get_candidate_stocks_by_markets(self, selected_markets: List[str]) -> List[str]:
        """根据选定市场获取候选股票列表，只返回有历史数据的股票"""
        try:
            candidate_stocks = []
            
            for market_name in selected_markets:
                # 找到对应的市场类型
                market_type = None
                for mt in MarketType:
                    if mt.value == market_name:
                        market_type = mt
                        break
                
                if not market_type or market_type not in self.market_mapping:
                    continue
                
                config = self.market_mapping[market_type]
                
                # 查询该市场有历史数据的股票
                with self.db_manager.get_conn() as conn:
                    query = """
                    SELECT DISTINCT s.symbol, s.name, s.market, s.board_type, s.exchange
                    FROM stocks s 
                    INNER JOIN prices_daily p ON (
                        CASE 
                            WHEN s.symbol LIKE '%.SH' THEN REPLACE(s.symbol, '.SH', '.SS')
                            ELSE s.symbol
                        END = p.symbol
                    )
                    WHERE s.exchange = ? AND s.market = ?
                    AND s.symbol NOT LIKE '88%'
                    AND (s.board_type IS NULL OR s.board_type NOT IN ('指数','行业指数','板块','基金','ETF'))
                    """
                    params = [config['exchange'], config['market']]
                    
                    if config['board_types']:
                        placeholders = ','.join(['?' for _ in config['board_types']])
                        query += f" AND s.board_type IN ({placeholders})"
                        params.extend(config['board_types'])
                    
                    # 确保有足够的历史数据（至少5条记录，适应数据覆盖率较低的情况）
                    query += """
                    GROUP BY s.symbol, s.name, s.market, s.board_type, s.exchange
                    HAVING COUNT(p.date) >= 5
                    ORDER BY s.symbol
                    """
                    
                    df = pd.read_sql_query(query, conn, params=params)
                    market_stocks = df.to_dict('records')
                    
                    # 应用股票过滤器
                    if market_stocks:
                        filter_result = self.stock_filter.filter_stock_list(
                            market_stocks,
                            include_st=True,
                            include_suspended=True,
                            db_manager=self.db_manager,
                            exclude_b_share=True,
                            exclude_star_market=True,
                            exclude_bse_stock=True
                        )
                        filtered_stocks = [stock['symbol'] for stock in filter_result['filtered_stocks']]
                        candidate_stocks.extend(filtered_stocks)
                        
                        self.logger.info(f"市场 {market_name} 原始股票 {len(market_stocks)} 只，过滤后 {len(filtered_stocks)} 只")
                    
            # 去重并排序
            candidate_stocks = sorted(list(set(candidate_stocks)))
            
            self.logger.info(f"从选定市场获取到 {len(candidate_stocks)} 只候选股票")
            return candidate_stocks
            
        except Exception as e:
            self.logger.error(f"获取候选股票失败: {e}")
            return []
    
    def _execute_stock_selection(self, 
                               candidate_stocks: List[str],
                               selection_criteria: Dict[str, Any] = None,
                               top_n: int = 20) -> Dict[str, Any]:
        """执行股票选择逻辑"""
        try:
            # 首先尝试使用智能选股服务进行预测
            self.logger.info(f"开始对 {len(candidate_stocks)} 只候选股票进行智能预测")
            
            # 加载模型（如果尚未加载）
            if not self.stock_selector.model:
                self.logger.info("正在加载预测模型...")
                model_loaded = self.stock_selector.load_model()
                if not model_loaded:
                    self.logger.warning("模型加载失败，将使用备用评分方法")
                    return self._fallback_stock_selection(candidate_stocks, selection_criteria, top_n)
            
            # 使用智能选股服务进行预测
            predictions = self.stock_selector.predict_stocks(candidate_stocks, top_n=top_n)
            
            if predictions:
                self.logger.info(f"智能预测成功，获得 {len(predictions)} 只股票的预测结果")
                
                # 为预测结果添加额外的分析数据
                enhanced_results = []
                for pred in predictions:
                    try:
                        # 获取股票基本信息
                        stock_info = self._get_stock_analysis_data(pred['symbol'])
                        if stock_info:
                            # 整合预测结果和基本信息
                            enhanced_result = {
                                'symbol': pred['symbol'],
                                'name': pred['name'],
                                'market': stock_info.get('market', ''),
                                'board_type': stock_info.get('board_type', ''),
                                'score': pred['score'],
                                'probability': pred['probability'],
                                'sentiment': pred['sentiment'],
                                'confidence': pred['confidence'],
                                'last_close': pred['last_close'],
                                'analysis': stock_info
                            }
                            enhanced_results.append(enhanced_result)
                    except Exception as e:
                        self.logger.warning(f"增强股票 {pred['symbol']} 信息失败: {e}")
                        # 即使增强失败，也保留原始预测结果
                        enhanced_results.append(pred)
                
                return {
                    'total_analyzed': len(candidate_stocks),
                    'top_stocks': enhanced_results,
                    'selection_criteria': selection_criteria or {},
                    'analysis_time': datetime.now().isoformat(),
                    'method': 'intelligent_prediction'
                }
            else:
                self.logger.warning("智能预测未返回结果，使用备用评分方法")
                return self._fallback_stock_selection(candidate_stocks, selection_criteria, top_n)
            
        except Exception as e:
            self.logger.error(f"执行股票选择失败: {e}")
            # 发生异常时使用备用方法
            return self._fallback_stock_selection(candidate_stocks, selection_criteria, top_n)
    
    def _fallback_stock_selection(self, 
                                candidate_stocks: List[str],
                                selection_criteria: Dict[str, Any] = None,
                                top_n: int = 20) -> Dict[str, Any]:
        """备用股票选择方法（基于传统评分）"""
        try:
            self.logger.info("使用备用评分方法进行选股")
            
            # 默认选股条件
            default_criteria = {
                'min_market_cap': 10,  # 最小市值（亿元）- 由于数据库中暂无市值数据，此条件暂不生效
                'max_pe_ratio': 100,    # 最大市盈率 - 由于数据库中暂无市盈率数据，此条件暂不生效
                'min_volume_ratio': 0.5,  # 最小成交量比率 - 降低要求以适应当前数据
                'technical_score_threshold': 0.3  # 技术分析得分阈值 - 降低要求
            }
            
            if selection_criteria:
                default_criteria.update(selection_criteria)
            
            # 分批处理候选股票，避免一次性处理过多
            batch_size = 100
            all_results = []
            
            for i in range(0, len(candidate_stocks), batch_size):
                batch_stocks = candidate_stocks[i:i + batch_size]
                
                # 对每只股票进行评分
                batch_results = []
                for symbol in batch_stocks:
                    try:
                        # 获取股票基本信息和技术指标
                        stock_info = self._get_stock_analysis_data(symbol)
                        if stock_info:
                            score = self._calculate_stock_score(stock_info, default_criteria)
                            if score > 0:  # 只保留有效评分的股票
                                # 基于评分估算概率和信心度
                                normalized_score = min(score / 100.0, 1.0)  # 将评分标准化到0-1
                                probability = max(0.5, min(0.95, 0.5 + normalized_score * 0.4))  # 50%-95%范围
                                confidence = max(60.0, min(90.0, 60.0 + normalized_score * 30.0))  # 60%-90%范围
                                sentiment = "看多" if probability > 0.6 else "中性"
                                
                                batch_results.append({
                                    'symbol': symbol,
                                    'name': stock_info.get('name', ''),
                                    'market': stock_info.get('market', ''),
                                    'board_type': stock_info.get('board_type', ''),
                                    'score': score,
                                    'probability': probability,
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'last_close': stock_info.get('latest_price', 0),
                                    'analysis': stock_info
                                })
                    except Exception as e:
                        self.logger.warning(f"分析股票 {symbol} 失败: {e}")
                        continue
                
                all_results.extend(batch_results)
            
            # 按评分排序并取前N只
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_stocks = all_results[:top_n]
            
            return {
                'total_analyzed': len(all_results),
                'top_stocks': top_stocks,
                'selection_criteria': default_criteria,
                'analysis_time': datetime.now().isoformat(),
                'method': 'fallback_scoring'
            }
            
        except Exception as e:
            self.logger.error(f"备用股票选择失败: {e}")
            return {
                'total_analyzed': 0,
                'top_stocks': [],
                'error': str(e),
                'method': 'fallback_scoring'
            }
    
    def _get_stock_analysis_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取股票分析数据"""
        try:
            with self.db_manager.get_conn() as conn:
                # 获取股票基本信息
                stock_query = "SELECT * FROM stocks WHERE symbol = ?"
                stock_df = pd.read_sql_query(stock_query, conn, params=[symbol])
                
                if stock_df.empty:
                    return None
                
                stock_info = stock_df.iloc[0].to_dict()
                
                # 获取最近的价格数据
                price_query = """
                    SELECT * FROM prices_daily 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT 30
                """
                price_df = pd.read_sql_query(price_query, conn, params=[symbol])
                
                if not price_df.empty:
                    stock_info['recent_prices'] = price_df.to_dict('records')
                    stock_info['latest_price'] = price_df.iloc[0]['close']
                    stock_info['price_change_30d'] = (
                        (price_df.iloc[0]['close'] - price_df.iloc[-1]['close']) / 
                        price_df.iloc[-1]['close'] * 100
                    ) if len(price_df) > 1 else 0
                
                # 获取实时行情
                quote_query = """
                    SELECT * FROM quotes_realtime 
                    WHERE symbol = ? 
                    ORDER BY ts DESC 
                    LIMIT 1
                """
                quote_df = pd.read_sql_query(quote_query, conn, params=[symbol])
                
                if not quote_df.empty:
                    stock_info['realtime_quote'] = quote_df.iloc[0].to_dict()
                
                return stock_info
                
        except Exception as e:
            self.logger.error(f"获取股票分析数据失败 {symbol}: {e}")
            return None
    
    def _calculate_stock_score(self, stock_info: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """计算股票评分"""
        try:
            # 首先进行基本面筛选，不符合条件的直接返回0分
            
            # 市值筛选（如果有市值数据）
            market_cap = stock_info.get('market_cap')
            if market_cap is not None:
                min_market_cap = criteria.get('min_market_cap', 0)
                if market_cap < min_market_cap:
                    return 0.0
            
            # 市盈率筛选（如果有市盈率数据）
            pe_ratio = stock_info.get('pe_ratio')
            if pe_ratio is not None and pe_ratio > 0:
                max_pe_ratio = criteria.get('max_pe_ratio', 100)
                if pe_ratio > max_pe_ratio:
                    return 0.0
            
            # 成交量比率筛选
            recent_prices = stock_info.get('recent_prices', [])
            if recent_prices and len(recent_prices) >= 5:
                recent_volume = sum([p['volume'] for p in recent_prices[:5]]) / 5
                avg_volume = sum([p['volume'] for p in recent_prices]) / len(recent_prices)
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                min_volume_ratio = criteria.get('min_volume_ratio', 1.0)
                if volume_ratio < min_volume_ratio:
                    return 0.0
            
            # 通过基本面筛选后，开始计算评分
            score = 0.0
            
            # 基础分数
            base_score = 50.0
            
            # 价格趋势评分 (30%权重)
            price_change_30d = stock_info.get('price_change_30d', 0)
            if price_change_30d > 10:
                score += 30
            elif price_change_30d > 0:
                score += 20
            elif price_change_30d > -10:
                score += 10
            
            # 成交量评分 (20%权重)
            if recent_prices and len(recent_prices) >= 5:
                recent_volume = sum([p['volume'] for p in recent_prices[:5]]) / 5
                avg_volume = sum([p['volume'] for p in recent_prices]) / len(recent_prices)
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio >= 2.0:
                    score += 20
                elif volume_ratio >= 1.5:
                    score += 15
                elif volume_ratio >= 1.2:
                    score += 10
                elif volume_ratio >= 0.8:
                    score += 8
                elif volume_ratio >= 0.5:
                    score += 5
            
            # 市场和板块加分 (10%权重)
            market = stock_info.get('market', '')
            board_type = stock_info.get('board_type', '')
            
            if board_type == '创业板':
                score += 8   # 创业板加分
            elif board_type == '主板':
                score += 5   # 主板稳定性加分
            
            # 数据完整性评分 (10%权重)
            if stock_info.get('recent_prices'):
                score += 5
            if stock_info.get('realtime_quote'):
                score += 5
            
            return base_score + score
            
        except Exception as e:
            self.logger.error(f"计算股票评分失败: {e}")
            return 0.0
    
    def get_market_statistics(self, selected_markets: List[str] = None) -> Dict[str, Any]:
        """获取市场统计信息"""
        try:
            if not selected_markets:
                selected_markets = [m.value for m in MarketType if m != MarketType.HK]
            
            statistics = {}
            
            for market_name in selected_markets:
                market_type = None
                for mt in MarketType:
                    if mt.value == market_name:
                        market_type = mt
                        break
                
                if not market_type or market_type not in self.market_mapping:
                    continue
                
                config = self.market_mapping[market_type]
                
                # 获取市场统计
                with self.db_manager.get_conn() as conn:
                    query = """
                        SELECT 
                            COUNT(*) as total_stocks,
                            COUNT(CASE WHEN pd.date IS NOT NULL THEN 1 END) as stocks_with_data,
                            MAX(pd.date) as latest_data_date
                        FROM stocks s
                        LEFT JOIN prices_daily pd ON s.symbol = pd.symbol
                        WHERE s.market = ?
                    """
                    params = [config['market']]
                    
                    if config['board_types']:
                        placeholders = ','.join(['?' for _ in config['board_types']])
                        query += f" AND s.board_type IN ({placeholders})"
                        params.extend(config['board_types'])
                    
                    df = pd.read_sql_query(query, conn, params=params)
                    
                    if not df.empty:
                        row = df.iloc[0]
                        statistics[market_name] = {
                            'total_stocks': row['total_stocks'],
                            'stocks_with_data': row['stocks_with_data'],
                            'data_coverage': row['stocks_with_data'] / row['total_stocks'] if row['total_stocks'] > 0 else 0,
                            'latest_data_date': row['latest_data_date']
                        }
            
            return {
                'success': True,
                'statistics': statistics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取市场统计失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    selector = MarketSelectorService()
    
    # 获取可用市场
    markets = selector.get_available_markets()
    print(f"可用市场: {markets}")
    
    # 多市场选股测试
    result = selector.select_stocks_by_markets(
        selected_markets=["沪市主板", "深市主板", "创业板"],
        top_n=10
    )
    print(f"选股结果: {result}")