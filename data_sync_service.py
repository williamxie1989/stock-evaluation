import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from akshare_data_provider import AkshareDataProvider
from enhanced_data_provider import EnhancedDataProvider
from db import DatabaseManager
from stock_list_manager import StockListManager
from stock_status_filter import StockStatusFilter
import time
import re
import random

class DataSyncService:
    """
    独立的行情数据同步服务
    - 支持全量和增量数据同步
    - 与选股功能完全解耦
    - 支持多市场数据同步控制
    """
    
    def __init__(self):
        # 股票列表仍使用 AkshareDataProvider
        self.list_provider = AkshareDataProvider()
        # 行情价格数据使用 EnhancedDataProvider（多数据源）
        self.price_provider = EnhancedDataProvider()
        self.db_manager = DatabaseManager()
        self.stock_manager = StockListManager()
        self.stock_filter = StockStatusFilter()
        self.logger = logging.getLogger(__name__)
        
    def sync_market_data(self, 
                        sync_type: str = "incremental",
                        markets: List[str] = None,
                        max_symbols: int = 0,
                        batch_size: int = 10,
                        delay_seconds: float = 1.0,
                        preferred_sources: Optional[List[str]] = None,
                        # 新增：目标筛选策略
                        top_n_by: Optional[str] = None,  # 可选：'market_cap' 或 'amount'
                        top_n: int = 0,
                        amount_window_days: int = 5) -> Dict[str, Any]:
        """
        同步市场数据
        
        Args:
            sync_type: 同步类型 ("full" | "incremental" | 其他值按增量处理)
            markets: 要同步的市场列表 ["SH", "SZ", "BJ"] (None表示所有A股市场)
            max_symbols: 最大处理股票数量 (0表示不限制)
            batch_size: 批量处理大小
            delay_seconds: 批次间延时
            preferred_sources: 数据源优先级，如 ["eastmoney","sina","akshare"]
            top_n_by: 按何种指标筛选Top N（'market_cap'市值 或 'amount'近N日成交额）
            top_n: Top N的N值（>0时生效）
            amount_window_days: 计算成交额Top N时的窗口天数
            
        Returns:
            同步结果字典
        """
        try:
            self.logger.info(f"开始{sync_type}数据同步...")
            # 设置数据源优先级（如果提供）
            if preferred_sources:
                try:
                    self.price_provider.set_preferred_sources(preferred_sources)
                    self.logger.info(f"使用数据源优先级: {preferred_sources}")
                except Exception as e:
                    self.logger.warning(f"设置数据源优先级失败: {e}")
            
            # 1. 更新股票列表
            stock_update_result = self._update_stock_list(markets)
            if not stock_update_result['success']:
                return stock_update_result
                
            # 2. 获取需要同步的股票列表（支持TopN筛选）
            target_stocks = self._get_sync_target_stocks(
                markets=markets, 
                max_symbols=max_symbols,
                top_n_by=top_n_by,
                top_n=top_n,
                amount_window_days=amount_window_days
            )
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
                **sync_result
            }
        except Exception as e:
            self.logger.error(f"同步市场数据失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _update_stock_list(self, markets: List[str] = None) -> Dict[str, Any]:
        """更新股票列表（使用StockListManager）"""
        try:
            # 默认更新所有市场（SH、SZ、BJ），根据参数过滤
            result = self.stock_manager.update_all_stocks()
            if not result or not result.get('success'):
                return result or {"success": False, "error": "未知错误"}
            # 兼容返回结构，提供 updated_count 字段
            return {
                'success': True,
                'updated_count': int(result.get('count', 0)),
                'details': result
            }
        except Exception as e:
            self.logger.error(f"更新股票列表失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_sync_target_stocks(self, markets: List[str] = None, max_symbols: int = 0,
                                 top_n_by: Optional[str] = None, top_n: int = 0,
                                 amount_window_days: int = 5) -> List[Dict[str, Any]]:
        """获取需要同步的股票列表（可选按市值/成交额TopN筛选）"""
        try:
            with self.db_manager.get_conn() as conn:
                # 标准A股市场过滤
                market_filter_sql = ""
                params: List[Any] = []
                if markets:
                    placeholders = ','.join(['?' for _ in markets])
                    market_filter_sql = f" AND s.market IN ({placeholders})"
                    params.extend(markets)
                else:
                    market_filter_sql = " AND s.market IN ('SH', 'SZ', 'BJ')"
                
                if top_n_by and top_n and top_n > 0:
                    by = top_n_by.lower().strip()
                    if by == 'market_cap':
                        # 按市值TopN（需stocks.market_cap有值）
                        query = (
                            "SELECT s.symbol, s.name, s.market \n"
                            "FROM stocks s \n"
                            "WHERE 1=1" + market_filter_sql + " AND s.market_cap IS NOT NULL \n"
                            "ORDER BY s.market_cap DESC \n"
                            f"LIMIT {int(top_n)}"
                        )
                        df = pd.read_sql_query(query, conn, params=params)
                        stocks = df.to_dict('records')
                        
                        # 应用股票过滤器
                        filter_result = self.stock_filter.filter_stock_list(
                            stocks,
                            include_st=True,
                            include_suspended=True,
                            db_manager=self.db_manager,
                            exclude_b_share=True,
                            exclude_star_market=True,
                            exclude_bse_stock=True
                        )
                        
                        self.logger.info(f"市值TopN股票过滤结果: 总数{filter_result['statistics']['total']}, "
                                       f"保留{filter_result['statistics']['filtered']}, "
                                       f"移除{filter_result['statistics']['removed']}")
                        
                        return filter_result['filtered_stocks']
                    elif by == 'amount':
                        # 按近N日成交额TopN
                        query = (
                            "WITH recent AS (\n"
                            "    SELECT symbol, SUM(amount) AS total_amount \n"
                            "    FROM prices_daily \n"
                            "    WHERE date >= date('now', ?) \n"
                            "    GROUP BY symbol\n"
                            ")\n"
                            "SELECT s.symbol, s.name, s.market \n"
                            "FROM stocks s \n"
                            "LEFT JOIN recent r ON s.symbol = r.symbol \n"
                            "WHERE 1=1" + market_filter_sql + " \n"
                            "ORDER BY COALESCE(r.total_amount, 0) DESC \n"
                            f"LIMIT {int(top_n)}"
                        )
                        offset = f"-{max(amount_window_days - 1, 0)} day"
                        df = pd.read_sql_query(query, conn, params=[offset] + params)
                        stocks = df.to_dict('records')
                        
                        # 应用股票过滤器
                        filter_result = self.stock_filter.filter_stock_list(
                            stocks,
                            include_st=True,
                            include_suspended=True,
                            db_manager=self.db_manager,
                            exclude_b_share=True,
                            exclude_star_market=True,
                            exclude_bse_stock=True
                        )
                        
                        self.logger.info(f"成交额TopN股票过滤结果: 总数{filter_result['statistics']['total']}, "
                                       f"保留{filter_result['statistics']['filtered']}, "
                                       f"移除{filter_result['statistics']['removed']}")
                        
                        return filter_result['filtered_stocks']
                
                # 默认：不使用TopN，按symbol排序
                query = "SELECT symbol, name, market FROM stocks s WHERE 1=1" + market_filter_sql + " ORDER BY symbol"
                if max_symbols > 0:
                    query += f" LIMIT {int(max_symbols)}"
                df = pd.read_sql_query(query, conn, params=params)
                stocks = df.to_dict('records')
                
                # 应用股票过滤器
                filter_result = self.stock_filter.filter_stock_list(
                    stocks,
                    include_st=True,
                    include_suspended=True,
                    db_manager=self.db_manager,
                    exclude_b_share=True,
                    exclude_star_market=True,
                    exclude_bse_stock=True
                )
                
                self.logger.info(f"股票过滤结果: 总数{filter_result['statistics']['total']}, "
                               f"保留{filter_result['statistics']['filtered']}, "
                               f"移除{filter_result['statistics']['removed']}")
                if filter_result['statistics']['removal_reasons']:
                    self.logger.info(f"移除原因: {filter_result['statistics']['removal_reasons']}")
                
                return filter_result['filtered_stocks']
                
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
        failed_stocks: List[Dict[str, Any]] = []
        source_usage_counts: Dict[str, int] = {}
        attempt_summary: Dict[str, Dict[str, int]] = {}
        
        self.logger.info(f"开始处理 {total_stocks} 只股票的{sync_type}数据同步")
        
        # 增量边界：从数据库获取每个symbol最新日期
        latest_map: Dict[str, str] = {} if sync_type == "full" else self.db_manager.get_latest_dates_by_symbol()
        
        # 分批处理
        for i in range(0, total_stocks, batch_size):
            batch = target_stocks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            self.logger.info(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch)} 只股票")
            
            for stock in batch:
                try:
                    symbol = stock['symbol']  # 形如 600000.SH
                    
                    # 计算增量起始日期
                    start_date: Optional[str] = None
                    if sync_type != "full":
                        last_date = latest_map.get(symbol)
                        if last_date:
                            try:
                                start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                            except Exception:
                                start_date = None
                    
                    # 选择历史抓取周期：全量尽量长，增量更短以降低压力
                    period = "5y" if sync_type == "full" else "180d"
                    df_hist = self.price_provider.get_stock_historical_data(symbol=symbol, period=period)
                    
                    # 诊断尝试信息（无论成功失败均记录）
                    attempts = getattr(self.price_provider, 'last_attempts', []) or []
                    last_source = getattr(self.price_provider, 'last_used_source', None)
                    # 汇总尝试统计
                    for att in attempts:
                        src = att.get('source') or 'unknown'
                        status = att.get('status') or 'unknown'
                        if src not in attempt_summary:
                            attempt_summary[src] = {'success': 0, 'empty': 0, 'exception': 0, 'unknown': 0}
                        if status not in attempt_summary[src]:
                            attempt_summary[src][status] = 0
                        attempt_summary[src][status] += 1
                    
                    if df_hist is not None and not df_hist.empty:
                        # 增量过滤
                        if start_date is not None and 'date' in df_hist.columns:
                            # 统一类型：将date列转为datetime，再与起始日期(Timestamp)比较，避免 'datetime.date' 与 'str' 比较错误
                            try:
                                df_hist['date'] = pd.to_datetime(df_hist['date'])
                            except Exception:
                                pass
                            df_hist = df_hist[df_hist['date'] >= pd.to_datetime(start_date)]
                        
                        if not df_hist.empty:
                            df_hist = df_hist.copy()
                            df_hist['symbol'] = symbol
                            # 插入日线数据（已标准化列: date/open/high/low/close/volume/可选amount）
                            # 记录来源：EnhancedDataProvider.last_used_source
                            source = last_source or getattr(self.price_provider, 'last_used_source', None)
                            rows_inserted = self.db_manager.upsert_prices_daily(
                                df_hist,
                                symbol_col="symbol",
                                date_col="date",
                                source=source or None
                            )
                            inserted_daily_rows += rows_inserted
                            if source:
                                source_usage_counts[source] = source_usage_counts.get(source, 0) + 1
                        else:
                            # 有数据但被增量过滤后为空，视为无新增
                            self.logger.info(f"{symbol} 无新增数据（已最新）")
                    else:
                        # 获取失败或空数据
                        failed_stocks.append({
                            'symbol': symbol,
                            'error': 'no_data',
                            'attempts': attempts
                        })
                    
                    processed_stocks += 1
                    # 每只股票之间加入轻微抖动，降低瞬时请求峰值
                    time.sleep(random.uniform(0.08, 0.2))
                    self.logger.info(f"已处理 {processed_stocks}/{total_stocks} 只股票: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"处理股票 {stock['symbol']} 失败: {e}")
                    attempts = getattr(self.price_provider, 'last_attempts', []) or []
                    failed_stocks.append({
                        'symbol': stock['symbol'],
                        'error': str(e),
                        'attempts': attempts
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
            'source_usage_counts': source_usage_counts,
            'attempt_summary': attempt_summary,
            'success_rate': processed_stocks / total_stocks if total_stocks > 0 else 0
        }

    def _record_sync_status(self, sync_type: str, sync_result: Dict[str, Any]):
        """记录同步状态（可扩展写入数据库或日志）"""
        try:
            success_rate = sync_result.get('success_rate', 0)
            failed_count = len(sync_result.get('failed_stocks', []))
            self.logger.info(
                f"{sync_type} 同步完成: success_rate={success_rate:.2%}, failed={failed_count}, inserted_daily_rows={sync_result.get('inserted_daily_rows', 0)}"
            )
        except Exception:
            pass

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
                        'latest_date': str(latest_date) if latest_date else None,
                        'days_behind': days_behind,
                        'is_fresh': days_behind is not None and days_behind <= 1,
                        'stock_count': int(row['stock_count'] or 0)
                    })
                
                return {
                    'success': True,
                    'report_time': datetime.now().isoformat(),
                    'markets': freshness_info
                }
        
        except Exception as e:
            self.logger.error(f"检查数据新鲜度失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def coverage_and_freshness_report(self, markets: List[str] = None, window_days: int = 5) -> Dict[str, Any]:
        """覆盖率与新鲜度简报：统计近N日内有数据的股票比例及各市场新鲜度"""
        try:
            with self.db_manager.get_conn() as conn:
                # 市场过滤
                market_filter = []
                params: List[Any] = []
                if markets:
                    placeholders = ','.join(['?' for _ in markets])
                    market_filter.append(f"s.market IN ({placeholders})")
                    params.extend(markets)
                else:
                    market_filter.append("s.market IN ('SH','SZ','BJ')")
                where_sql = " AND ".join(["1=1"] + market_filter)
                
                # 计算最近的交易日（以库内最大日期为准）
                end_date_df = pd.read_sql_query("SELECT MAX(date) AS end_date FROM prices_daily", conn)
                end_date = end_date_df['end_date'].iloc[0] if not end_date_df.empty else None
                # 窗口偏移
                offset = f"-{max(window_days - 1, 0)} day"
                
                # 每个市场的覆盖率
                query = (
                    "SELECT s.market, COUNT(DISTINCT s.symbol) AS total_symbols, \n"
                    "       COUNT(DISTINCT CASE WHEN ed.end_date IS NOT NULL AND pd.date BETWEEN date(ed.end_date, ?) AND ed.end_date THEN s.symbol END) AS covered_symbols \n"
                    "FROM stocks s \n"
                    "CROSS JOIN (SELECT MAX(date) AS end_date FROM prices_daily) ed \n"
                    "LEFT JOIN prices_daily pd ON pd.symbol = s.symbol \n"
                    f"WHERE {where_sql} \n"
                    "GROUP BY s.market ORDER BY s.market"
                )
                df = pd.read_sql_query(query, conn, params=[offset] + params)
                
                markets_report = []
                total_total = 0
                total_covered = 0
                today = datetime.now().date()
                
                # 获取每个市场的最新日期
                latest_df = pd.read_sql_query(
                    "SELECT s.market, MAX(pd.date) AS latest_date \n"
                    "FROM stocks s LEFT JOIN prices_daily pd ON s.symbol=pd.symbol \n"
                    f"WHERE {where_sql} \n"
                    "GROUP BY s.market", conn, params=params)
                latest_map = {r['market']: r['latest_date'] for _, r in latest_df.iterrows()}
                
                for _, row in df.iterrows():
                    market = row['market']
                    total_symbols = int(row['total_symbols'] or 0)
                    covered_symbols = int(row['covered_symbols'] or 0)
                    total_total += total_symbols
                    total_covered += covered_symbols
                    latest_date = pd.to_datetime(latest_map.get(market)).date() if latest_map.get(market) else None
                    days_behind = (today - latest_date).days if latest_date else None
                    markets_report.append({
                        'market': market,
                        'total_symbols': total_symbols,
                        'covered_symbols': covered_symbols,
                        'coverage_rate': round(covered_symbols / total_symbols, 4) if total_symbols else None,
                        'latest_date': str(latest_date) if latest_date else None,
                        'days_behind': days_behind,
                        'is_fresh': days_behind is not None and days_behind <= 1
                    })
                
                overall = {
                    'total_symbols': total_total,
                    'covered_symbols': total_covered,
                    'coverage_rate': round(total_covered / total_total, 4) if total_total else None
                }
                
                return {
                    'success': True,
                    'report_time': datetime.now().isoformat(),
                    'window_days': window_days,
                    'end_date': str(end_date) if end_date else None,
                    'overall': overall,
                    'markets': markets_report
                }
        
        except Exception as e:
            self.logger.error(f"生成覆盖率与新鲜度报表失败: {e}")
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