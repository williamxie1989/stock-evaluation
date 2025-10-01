from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
# StockAnalyzer 已在第四阶段被归档，使用其他服务替代
# from ...core.analyzer import StockAnalyzer
import json
import os
from ...data.db.unified_database_manager import UnifiedDatabaseManager  # 使用统一数据库管理器
from ...core.unified_data_access_factory import get_unified_data_access  # 统一数据访问层
from core.factories import get_data_provider, get_realtime_provider
import pandas as pd
import re
from ...trading.signals.signal_generator import SignalGenerator
from ...apps.scripts.selector_service import IntelligentStockSelector
from ...services.stock.stock_list_manager import StockListManager
from ...services.market.market_selector_service import MarketSelectorService
from ...data.sync.data_sync_service import DataSyncService
from ...apps.scripts.concurrent_data_sync_service import ConcurrentDataSyncService

# EnhancedDataProvider 已在第四阶段被归档，使用统一数据访问层替代
# from ...data.providers.optimized_enhanced_data_provider import OptimizedEnhancedDataProvider
# EnhancedDataProvider 已在第四阶段被归档，使用AkshareDataProvider替代
# from ...apps.scripts.enhanced_data_provider import EnhancedDataProvider
# from ...data.providers.enhanced_realtime_provider import EnhancedRealtimeProvider

from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
import time
import asyncio
from contextlib import asynccontextmanager

# 配置日志
logger = logging.getLogger(__name__)
logging.getLogger('src.data.field_mapping').setLevel(logging.WARNING)
logging.getLogger('src.ml.features.enhanced_features').setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期处理"""
    # 启动时逻辑
    await startup_event()
    
    # yield 控制权给应用
    yield
    
    # 可选择在此加入关闭时的清理逻辑
    pass

app = FastAPI(
    title="股票分析系统", 
    description="基于大模型的股票分析系统",
    lifespan=lifespan
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=1,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

class StockRequest(BaseModel):
    symbol: str

# analyzer = StockAnalyzer()  # 已在第四阶段被归档，使用其他服务替代
# 初始化统一数据访问层（替代原有的分散数据提供者）
try:
    # 首先尝试获取已存在的实例
    _unified_data_access = get_unified_data_access()
    if _unified_data_access is None:
        # 如果实例不存在，则创建新的实例
        logger.info("创建新的统一数据访问层实例...")
        from src.core.unified_data_access_factory import create_unified_data_access
        _unified_data_access = create_unified_data_access()
    _db = _unified_data_access.db_manager  # 从统一访问层获取数据库管理器
    logger.info("统一数据访问层初始化成功")
except Exception as e:
    logger.error(f"统一数据访问层初始化失败: {e}")
    # 使用备用方案，直接创建数据库管理器
    logger.info("使用备用方案创建数据库管理器...")
    from ...data.db.unified_database_manager import UnifiedDatabaseManager
    _db = UnifiedDatabaseManager(db_type='mysql')
    _unified_data_access = None
# 初始化服务（使用统一数据访问层）
data_sync_service = DataSyncService(data_provider=_unified_data_access)
# 注入统一数据访问层实例，避免内部重复创建提供器
concurrent_data_sync_service = ConcurrentDataSyncService(data_provider=_unified_data_access, max_workers=8, db_batch_size=50)
# 已移除 concurrent_data_provider 实例化，统一通过 _unified_data_access
market_selector_service = MarketSelectorService()

# 选股结果内存缓存（按参数维度缓存）
PICKS_CACHE: dict = {}
PICKS_CACHE_TTL_SECONDS: int = int(os.environ.get("PICKS_CACHE_TTL", "900"))  # 默认15分钟

async def startup_event():
    """
    应用启动时智能初始化数据
    """
    global selector_service, signal_generator, stock_list_manager
    
    # 复用全局实例并在启动时预加载模型
    selector_service = IntelligentStockSelector(_db)
    try:
        loaded = selector_service.load_models(period='30d') or selector_service.load_model()
        if loaded:
            logger.info("[startup] 模型预加载完成")
        else:
            logger.warning("[startup] 未找到可用模型，将在首次请求时回退到原有逻辑")
    except Exception as e:
        logger.error(f"[startup] 模型预加载失败: {e}")
    
    signal_generator = SignalGenerator()
    stock_list_manager = StockListManager()
    
    # 后台预热 stock-picks 缓存（不阻塞启动）
    if os.environ.get("SKIP_PREWARM", "0") not in ("1", "1", "1"):
        async def _prewarm_stock_picks():
            try:
                # 预热时使用与智能选股API相同的默认参数，确保缓存命中
                limit_symbols = 500  # 与API默认参数一致
                top_n = 60  # 与API默认参数一致
                logger.info(f"[startup] 开始后台预热 /api/stock-picks 缓存 (limit_symbols={limit_symbols}, top_n={top_n})")
                # 调用异步的get_stock_picks方法，使用与API相同的参数
                result = await get_stock_picks(top_n=top_n, limit_symbols=limit_symbols)
                logger.info(f"[startup] 预热完成，结果: {len(result.get('data', {}).get('picks', [])) if result.get('success') else '失败'}")
            except Exception as e:
                logger.warning(f"[startup] 预热失败: {e}")
        asyncio.create_task(_prewarm_stock_picks())
    
    print("应用启动，检查数据状态...")
    # 支持通过环境变量跳过启动时的数据同步，以便快速启动服务进行接口联调
    if os.environ.get("SKIP_STARTUP_SYNC", "0") in ("1", "1", "1"):
        print("已设置SKIP_STARTUP_SYNC，跳过启动时数据初始化。")
        return

    try:
        # 检查数据库中是否已有股票数据
        existing_stocks = _db.list_symbols(limit=1)
        
        if not existing_stocks:
            # 首次启动，完整初始化 - 扩大股票覆盖范围，暂时只处理A股
            print("检测到首次启动，开始完整数据初始化...")
            result = await refresh_data(max_symbols=1000, full=1, batch_size=25, delay_seconds=1.5)
            if result.get("success"):
                data = result.get("data", {})
                print(f"完整数据初始化完成: 处理了{data.get('symbols', 0)}只股票，插入{data.get('inserted_daily_rows', 0)}条日线数据，{data.get('quotes_rows', 0)}条实时行情")
            else:
                print(f"完整数据初始化失败: {result.get('error', '未知错误')}")
        else:
            # 已有数据，进行增量更新，暂时只处理A股
            print(f"检测到已有股票数据，进行增量更新...")
            result = await refresh_data(max_symbols=0, full=0, batch_size=20, delay_seconds=0.8)  # max_symbols=0表示处理所有股票
            if result.get("success"):
                data = result.get("data", {})
                print(f"增量数据更新完成: 处理了{data.get('symbols', 0)}只股票，插入{data.get('inserted_daily_rows', 0)}条日线数据，{data.get('quotes_rows', 0)}条实时行情")
            else:
                print(f"增量数据更新失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"数据初始化失败: {e}")
        # 不阻止应用启动，即使数据初始化失败


# 修正类型注解为 Optional[str]
def _to_a_symbol_with_suffix(code: Optional[str]) -> Optional[str]:
    if code is None or str(code).lower() == 'nan':
        return None
    code = str(code).zfill(6)
    # 统一外部构造为 .SH，内部保持对 .SS 的兼容
    return f"{code}.SH" if code.startswith('6') else f"{code}.SZ"





@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/analysis")
async def read_analysis(symbol: Optional[str] = None):
    """分析页面路由"""
    return FileResponse("static/analysis.html")

@app.post("/api/analyze")
async def analyze_stock(request: StockRequest):
    """分析股票"""
    try:
        # StockAnalyzer 已在第四阶段被归档，使用其他服务替代
        # result = analyzer.analyze_stock(request.symbol)
        
        # 临时返回错误信息，提示服务重构中
        return {
            "success": 1,
            "data": {
                "success": 0,
                "error": "股票分析服务正在重构中，请使用其他API端点"
            }
        }
        
    except Exception as e:
        return {
            "success": 1,
            "data": {
                "success": 0,
                "error": f"在API层发生意外错误: {e}"
            }
        }

@app.get("/api/popular-stocks")
async def get_popular_stocks():
    """获取热门股票列表"""
    popular_stocks = [
        {"symbol": "600036.SH", "name": "招商银行"},
        {"symbol": "601318.SH", "name": "中国平安"},
        {"symbol": "600519.SH", "name": "贵州茅台"},
        {"symbol": "000858.SZ", "name": "五粮液"},
        {"symbol": "000333.SZ", "name": "美的集团"},
        {"symbol": "000001.SH", "name": "上证指数"},
        {"symbol": "399001.SZ", "name": "深证成指"},
        {"symbol": "000300.SH", "name": "沪深300"}
    ]
    return {"stocks": popular_stocks}


@app.post("/api/refresh-data")
async def refresh_data(max_symbols: int = 50, full: bool = 0, batch_size: int = 10, delay_seconds: float = 1.0):
    """
    刷新全市场股票元数据与增量日线/实时行情：
    - 写入 stocks（包含全市场股票信息）
    - 增量写入 prices_daily（根据已有最新日期向后补齐）
    - 追加 quotes_realtime 快照
    
    参数说明：
    - max_symbols: 限制本次处理的股票数，避免耗时过长（0表示处理所有股票）
    - full: 是否完整初始化（1）还是增量更新（0）
    - batch_size: 批量处理大小，防止API限流
    - delay_seconds: 批次间延时，控制API调用频率
    """
    try:
        # 1) 获取全市场股票列表并写入 stocks 表
        print("正在获取全市场股票列表...")
        all_stocks = _unified_data_access.get_all_stock_list()
        stock_rows = []
        
        if all_stocks is None or all_stocks.empty:
            print("获取全市场股票列表失败，使用种子股票列表...")
            # 回退方案：使用种子股票
            seed_symbols = [
                {"symbol": "600519.SH", "name": "贵州茅台"},
                {"symbol": "000858.SZ", "name": "五粮液"},
                {"symbol": "300750.SZ", "name": "宁德时代"},
                {"symbol": "002594.SZ", "name": "比亚迪"},
                {"symbol": "601318.SH", "name": "中国平安"}
            ]
            for s in seed_symbols:
                sym = s["symbol"]
                market = "SH" if (sym.endswith(".SH") or sym.endswith(".SS")) else "SZ"
                stock_rows.append({
                    "symbol": sym,
                    "name": s.get("name"),
                    "market": market,
                    "ah_pair": None
                })
        else:
            # 使用全市场股票数据
            print(f"成功获取全市场股票列表，共 {len(all_stocks)} 只股票")
            for _, row in all_stocks.iterrows():
                code = row.get('code')
                name = row.get('name')
                market = row.get('market', 'UNKNOWN')
                board_type = row.get('board_type', '')

                # 基础有效性校验
                if code is None:
                    continue
                code_str = str(code).strip()
                if not code_str.isdigit():
                    continue

                # 根据市场类型生成symbol（并进行长度规范与补零）
                symbol = None
                if market == 'SH':
                    if len(code_str) != 6:
                        continue
                    symbol = f"{code_str}.SH"
                elif market == 'SZ':
                    if len(code_str) != 6:
                        continue
                    symbol = f"{code_str}.SZ"
                elif market == 'BJ':
                    continue  # BJ股票已移除，跳过北交所
                else:
                    continue  # 跳过未知市场

                stock_rows.append({
                    "symbol": symbol,
                    "name": name,
                    "market": market,
                    "board_type": board_type,
                    "ah_pair": None  # 暂时不处理A+H配对关系
                })
        if stock_rows:
            _db.upsert_stocks(stock_rows)
        
        # 2) 选择需要更新的 symbols（去重）
        symbols = list(dict.fromkeys([s["symbol"] for s in stock_rows]))
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        elif max_symbols == 0:
            # max_symbols=0 表示处理所有股票，不做限制
            pass

        # 3) 获取已有最新日期，计算增量开始时间
        latest_map = {} if full else _db.get_latest_dates_by_symbol()

        # 4) 增量拉取并写入日线（批量处理）
        import asyncio
        total_rows = 0
        processed_count = 0
        
        # 分批处理股票，防止API限流
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}，包含 {len(batch_symbols)} 只股票")
            
            for symbol in batch_symbols:
                market = 'A'
                code_digits = re.sub(r"\D", "", symbol.split('.')[0]).zfill(6)
                start_date = None
                if not full and latest_map.get(symbol):
                    try:
                        latest_date = pd.to_datetime(latest_map[symbol], errors="coerce")
                        if not pd.isnull(latest_date):
                            start_date = (latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        else:
                            start_date = None
                    except Exception as e:
                        logger.warning(f"解析最新日期失败: {e}")
                        start_date = None
                df_hist = await _unified_data_access.get_historical_data(symbol=code_digits, start_date=start_date, end_date=datetime.now().strftime('%Y-%m-%d'))  # type: pd.DataFrame
                if df_hist is not None and not df_hist.empty:
                    df_hist = df_hist.copy()
                    df_hist['symbol'] = symbol
                    total_rows += _db.upsert_prices_daily(df_hist, source='akshare_ah_daily')
                processed_count += 1
                print(f"已处理 {processed_count}/{len(symbols)} 只股票: {symbol}")
            
            # 批次间延时，防止API限流
            if i + batch_size < len(symbols):  # 不是最后一批
                print(f"批次完成，等待 {delay_seconds} 秒...")
                await asyncio.sleep(delay_seconds)

        # 5) 实时行情快照 - 已移除启动时批量获取，避免性能问题和API限制
        quotes_rows = []
        
        # 注释掉启动时的实时行情批量获取，改为按需获取
        # # 使用增强版实时行情提供器获取A股实时行情
        # # 获取所有A股代码列表
        # all_stocks = _unified_data_access.get_all_stock_list()
        # 
        # if all_stocks is not None and not all_stocks.empty:
        #     # 使用批量获取功能，提高效率
        #     symbols = all_stocks['symbol'].tolist()
        #     
        #     # 批量获取实时行情
        #     batch_quotes = _realtime_provider.get_batch_realtime_quotes(symbols)
        #     
        #     for symbol, quote in batch_quotes.items():
        #         try:
        #             if quote and 'price' in quote:
        #                 quotes_rows.append({
        #                     'symbol': symbol,
        #                     'price': float(quote['price']),
        #                     'change_pct': quote.get('change_pct'),
        #                     'volume': quote.get('volume')
        #                 })
        #         except Exception as e:
        #             logger.warning(f"处理股票 {symbol} 实时行情失败: {e}")
        # 
        # if quotes_rows:
        #     _db.insert_quotes_realtime(pd.DataFrame(quotes_rows), source='enhanced_realtime_provider')
        # else:
        #     logger.warning("实时行情快照获取失败，未获取到有效数据")

        return {"success": 1, "data": {"symbols": len(symbols), "inserted_daily_rows": total_rows, "quotes_rows": 0}}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stock-picks")
async def get_stock_picks(top_n: int = 60, limit_symbols: Optional[int] = 500, force_refresh: bool = 0, debug: int = 0):
    """
    获取智能选股推荐
    可选 limit_symbols 用于限制参与预测的股票数量，以便快速联调与验证。
    新增：内存缓存与耗时统计（debug=1 返回详细耗时）。
    """
    try:
        # 复用全局 selector 实例
        selector = selector_service

        # 缓存键与TTL检查
        cache_key = f"{limit_symbols or 0}:{top_n}"
        now = datetime.now()
        if not force_refresh:
            cached = PICKS_CACHE.get(cache_key)
            if cached and cached.get('expires_at') and cached['expires_at'] > now:
                resp = cached['response']
                # 标记缓存命中
                if isinstance(resp, dict):
                    resp = {**resp, 'cached': 1}
                return resp

        t0 = time.monotonic()

        # 快速路径：限制参与预测的股票数量
        if limit_symbols and limit_symbols > 0:
            # 优先加载新模型，失败回退旧模型；若均失败则回退到原逻辑
            try:
                _ = selector.load_models(period='30d') or selector.load_model()
            except Exception as e:
                logger.warning(f"加载模型失败，将回退原逻辑: {e}")
                return selector.get_stock_picks(top_n)
            t_fetch0 = time.monotonic()
            symbols_data = _db.list_symbols(markets=['SH','SZ'])
            t_fetch1 = time.monotonic()
            
            # 在快速路径中也应用股票过滤
            from ...services.stock.stock_status_filter import StockStatusFilter
            stock_filter = StockStatusFilter()
            
            valid_symbols = []
            for symbol_info in symbols_data:
                code = symbol_info.get('symbol')
                # 过滤非A股
                if not code or not (code.endswith('.SH') or code.endswith('.SZ')):
                    continue
                # 股票状态过滤
                if not stock_filter.should_include(code):
                    continue
                valid_symbols.append(code)

            # 截断至 limit_symbols
            candidate_symbols = valid_symbols[:limit_symbols]

            # 预测并排序
            predicts = selector.predict_top_n(candidate_symbols, top_n)
            t1 = time.monotonic()
            
            # 组装响应
            response = {
                "success": 1,
                "data": {
                    "picks": predicts,
                    "timing": {
                        "fetch_symbols_ms": int((t_fetch1 - t_fetch0) * 1000),
                        "total_ms": int((t1 - t0) * 1000),
                    }
                }
            }

            # 写入缓存
            try:
                ttl = timedelta(seconds=PICKS_CACHE_TTL_SECONDS)
                PICKS_CACHE[cache_key] = {
                    'response': response,
                    'expires_at': datetime.now() + ttl
                }
            except Exception:
                pass

            return response

        # 默认路径：不限制股票数量
        result = selector.get_stock_picks(top_n)
        t1 = time.monotonic()

        response = {
            "success": 1,
            "data": {
                "picks": result,
                "timing": {"total_ms": int((t1 - t0) * 1000)}
            }
        }

        try:
            ttl = timedelta(seconds=PICKS_CACHE_TTL_SECONDS)
            PICKS_CACHE[cache_key] = {
                'response': response,
                'expires_at': datetime.now() + ttl
            }
        except Exception:
            pass

        return response
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/stocks/update")
async def update_stock_list():
    """使用统一数据访问层更新股票列表"""
    try:
        # 使用统一数据访问层获取股票列表
        all_stocks = await _unified_data_access.get_stock_list()
        if all_stocks is None or len(all_stocks) == 0:
            return {"success": 0, "error": "无法获取全市场股票列表"}
        
        stock_rows = []
        for stock in all_stocks:
            code = stock.get('code')
            name = stock.get('name')
            market = stock.get('market', 'UNKNOWN')
            board_type = stock.get('board_type', '')
            if not code:
                continue
            code_str = str(code).strip()
            if not code_str.isdigit() or len(code_str) != 6:
                continue
            if market not in ('SH', 'SZ'):
                continue
            symbol = f"{code_str}.{market}"
            stock_rows.append({
                "symbol": symbol,
                "name": name,
                "market": market,
                "board_type": board_type,
                "ah_pair": None
            })
        
        # 使用统一数据访问层更新股票数据
        await _unified_data_access.upsert_stocks(stock_rows)
        return {"success": 1, "updated": len(stock_rows)}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stocks/summary")
async def get_market_summary():
    try:
        markets = _db.get_market_summary()
        return {"success": 1, "markets": markets}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stocks/list")
async def get_stocks(market: str = None, board_type: str = None, limit: int = None):
    try:
        stocks = _db.list_symbols(market=market, board_type=board_type, limit=limit)
        return {"success": 1, "stocks": stocks}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stocks/candidates")
async def get_candidate_stocks(market: str = None, board_type: str = None, limit: int = 100):
    try:
        candidates = market_selector_service.get_candidate_stocks(market, board_type, limit)
        return {"success": 1, "candidates": candidates}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/markets")
async def get_available_markets():
    try:
        return {"success": 1, "markets": _db.get_available_markets()}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/select_stocks_by_markets")
async def select_stocks_by_markets(request: dict):
    try:
        selected_markets = request.get('markets')
        top_n = int(request.get('top_n', 50))
        result = market_selector_service.select_stocks_by_markets(selected_markets, top_n)
        return {"success": 1, "data": result}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/market_statistics")
async def get_market_statistics(selected_markets: Optional[List[str]] = None):
    try:
        stats = market_selector_service.get_market_statistics(selected_markets)
        return {"success": 1, "data": stats}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/sync_data_optimized")
async def sync_market_data_optimized(request: dict = None):
    """使用统一数据访问层执行增量市场数据同步

    前端"优化同步"按钮调用此端点。如果统一数据访问层尚未初始化，
    本函数会尝试即时创建，确保不会因 NoneType 造成调用失败。
    """
    try:
        global _unified_data_access
        if request is None:
            request = {}
        symbols = request.get('symbols')
        trade_date = request.get('trade_date')  # 新增: 指定交易日同步
        batch_size = int(request.get('batch_size', 20))
        delay_seconds = float(request.get('delay_seconds', 1.0))
        step_days = int(request.get('step_days', 15))  # 新增: 同步分段天数

        # 若统一数据访问层未就绪，尝试重新初始化一次
        if _unified_data_access is None:
            try:
                from src.core.unified_data_access_factory import create_unified_data_access
                _unified_data_access = create_unified_data_access()
                logger.info("已重新创建统一数据访问层实例，准备执行同步")
            except Exception as init_err:
                logger.error(f"无法初始化统一数据访问层: {init_err}")
                return {"success": 0, "error": "统一数据访问层初始化失败"}

        # 根据是否指定 trade_date 选择同步策略
        if trade_date:
            # 按日期批量同步（内部支持分段）
            # 若未指定 symbols，则同步全市场
            if not symbols:
                stats = await _unified_data_access.sync_market_data_all_by_date(
                    trade_date=trade_date,
                    step_days=step_days,
                    batch_size=batch_size
                )
            else:
                # 否则逐股票同步
                synced_total, errors_total = 0, 0
                for sym in symbols:
                    res = await _unified_data_access.sync_market_data_by_date(
                        symbol=sym,
                        start_date=pd.to_datetime(trade_date),
                        end_date=pd.to_datetime(trade_date),
                        step_days=step_days
                    )
                    synced_total += res.get("synced", 0)
                    errors_total += res.get("errors", 0)
                stats = {"synced": synced_total, "errors": errors_total}
        else:
            # 未指定 trade_date，先尝试按“昨日”批量同步，失败则回退增量同步
            try:
                yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
                stats = await _unified_data_access.sync_market_data_all_by_date(
                    trade_date=yesterday,
                    step_days=step_days,
                    batch_size=batch_size
                )
            except Exception as date_sync_err:
                logger.warning(f"按日期批量同步失败，回退逐股票增量同步: {date_sync_err}")
                stats = await _unified_data_access.sync_market_data(
                    symbols=symbols,  # None 表示全市场
                    batch_size=batch_size,
                    delay=delay_seconds
                )
        return {"success": 1, "data": stats}
    except Exception as e:
        logger.error(f"优化同步失败: {e}")
        return {"success": 0, "error": str(e)}

@app.post("/api/sync_data_concurrent")
async def sync_market_data_concurrent(request: dict = None):
    try:
        if request is None:
            request = {}
        markets = request.get('markets')
        days = int(request.get('days', 365))
        symbols = request.get('symbols')
        batch_size = int(request.get('batch_size', 20))
        delay_seconds = float(request.get('delay_seconds', 1.0))

        # 使用统一数据访问层进行并发同步（内部已支持批次节流）
        stats = await _unified_data_access.sync_market_data(
            symbols=symbols,  # 若提供精确股票，直接传入；否则 None 表示全市场
            batch_size=batch_size,
            delay=delay_seconds
        )
        return {"success": 1, "data": stats}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/sync_data")
async def sync_market_data(request: dict = None):
    try:
        if request is None:
            request = {}
        markets = request.get('markets')
        days = int(request.get('days', 365))
        symbols = request.get('symbols')
        batch_size = int(request.get('batch_size', 20))
        delay_seconds = float(request.get('delay_seconds', 1.0))

        stats = data_sync_service.sync_market_data(
            sync_type="incremental",
            markets=markets,
            max_symbols=0,
            batch_size=batch_size,
            delay_seconds=delay_seconds
        )
        return {"success": 1, "data": stats}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/sync/market-data")
async def sync_market_data_endpoint(request: dict = None):
    try:
        if request is None:
            request = {}
        sync_type = request.get('sync_type', 'incremental')
        markets = request.get('markets')
        max_symbols = int(request.get('max_symbols', 0))

        stats = data_sync_service.sync_market_data(
            sync_type=sync_type,
            markets=markets,
            max_symbols=max_symbols
        )
        return {"success": 1, "data": stats}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/data/sync")
async def sync_data(request: dict):
    try:
        action = request.get('action')
        if action == 'stocks':
            return await update_stock_list()
        elif action == 'daily_prices':
            return await refresh_data(max_symbols=int(request.get('max_symbols', 50)))
        else:
            return {"success": 0, "error": f"未知同步操作: {action}"}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/coverage_report")
async def coverage_report(markets: str = None, window_days: int = 5):
    try:
        selected = markets.split(',') if markets else None
        report = data_sync_service.get_coverage_report(selected_markets=selected, window_days=window_days)
        return {"success": 1, "data": report}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/sync_status")
async def get_sync_status():
    try:
        status = data_sync_service.get_sync_status()
        return {"success": 1, "data": status}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/data_freshness")
async def check_data_freshness(markets: str = None):
    try:
        selected = markets.split(',') if markets else None
        freshness = data_sync_service.get_data_freshness(selected_markets=selected)
        return {"success": 1, "data": freshness}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/data_sources/validate")
async def validate_data_sources():
    """验证所有可用的数据源，返回数据源质量报告"""
    try:
        validation_report = await _unified_data_access.validate_data_sources()
        return {"success": 1, "data": validation_report}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/data_sources/report")
async def get_data_sources_report():
    """获取当前数据源使用情况和质量报告"""
    try:
        report = _unified_data_access.get_data_quality_report()
        return {"success": 1, "data": report}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.post("/api/data_sources/switch")
async def switch_data_source(request: dict):
    """切换主要数据源"""
    try:
        source_name = request.get('source')
        if not source_name:
            return {"success": 0, "error": "未指定数据源名称"}
        
        success = await _unified_data_access.set_primary_data_source(source_name)
        if success:
            return {"success": 1, "message": f"已切换到数据源: {source_name}"}
        else:
            return {"success": 0, "error": f"无法切换到数据源: {source_name}"}
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stock/{symbol}/realtime")
async def get_stock_realtime_data(symbol: str):
    """获取股票实时数据"""
    try:
        # 清理股票代码格式
        clean_symbol = symbol.replace('.SZ', '').replace('.SH', '')
        
        # 使用统一数据访问层获取实时数据（需要传递列表）
        realtime_data = await _unified_data_access.get_realtime_data([clean_symbol])
        
        if realtime_data and clean_symbol in realtime_data:
            # 获取股票数据
            data = realtime_data[clean_symbol]
            return {
                "success": 1,
                "data": {
                    "symbol": symbol,
                    "price": float(data.get('close', data.get('price', 0))),
                    "change": float(data.get('change', 0)),
                    "change_percent": float(data.get('change_percent', 0)),
                    "volume": int(data.get('volume', 0)),
                    "timestamp": data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                }
            }
        else:
            return {"success": 0, "error": "未找到该股票的实时数据"}
    except Exception as e:
        return {"success": 0, "error": f"获取实时数据失败: {str(e)}"}

@app.post("/api/stocks/realtime/batch")
async def get_stocks_realtime_data(request: dict):
    """批量获取股票实时数据"""
    try:
        symbols = request.get('symbols', [])
        if not symbols:
            return {"success": 0, "error": "未提供股票代码列表"}
        
        # 清理股票代码格式
        clean_symbols = [s.replace('.SZ', '').replace('.SH', '') for s in symbols]
        
        # 使用统一数据访问层批量获取实时数据
        realtime_data = await _unified_data_access.get_realtime_data(clean_symbols)
        
        if realtime_data:
            # 转换数据格式为API响应格式
            result = []
            for i, symbol in enumerate(clean_symbols):
                if symbol in realtime_data:
                    data = realtime_data[symbol]
                    original_symbol = symbols[i]  # 保持原始格式
                    result.append({
                        "symbol": original_symbol,
                        "price": float(data.get('close', data.get('price', 0))),
                        "change": float(data.get('change', 0)),
                        "change_percent": float(data.get('change_percent', 0)),
                        "volume": int(data.get('volume', 0)),
                        "timestamp": data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    })
            
            return {"success": 1, "data": result}
        else:
            return {"success": 0, "error": "未找到实时数据"}
    except Exception as e:
        return {"success": 0, "error": f"批量获取实时数据失败: {str(e)}"}

@app.get("/api/stock/{symbol}/history")
async def get_stock_history_data(symbol: str, start_date: str = None, end_date: str = None, days: int = 30):
    """获取股票历史数据"""
    try:
        # 如果未指定日期，使用默认值
        from datetime import datetime, timedelta
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 使用统一数据访问层获取历史数据
        historical_data = await _unified_data_access.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if historical_data is not None and not historical_data.empty:
            # 转换数据格式为API响应格式
            history_list = []
            for index, row in historical_data.iterrows():
                # 获取日期 - 从历史数据索引中获取
                if isinstance(index, pd.Timestamp):
                    date_str = index.strftime('%Y-%m-%d')
                else:
                    # 如果索引不是Timestamp，尝试从date列获取
                    date_value = row.get('date')
                    if pd.isna(date_value):
                        date_str = index.strftime('%Y-%m-%d') if hasattr(index, 'strftime') else str(index)
                    elif hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_value)
                
                history_list.append({
                    "date": date_str,
                    "open": float(row.get('open', 0)),
                    "close": float(row.get('close', 0)),
                    "high": float(row.get('high', 0)),
                    "low": float(row.get('low', 0)),
                    "volume": int(row.get('volume', 0)),
                    "turnover": float(row.get('turnover', row.get('amount', 0)))
                })
            
            return {
                "success": 1,
                "data": {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": len(history_list),
                    "history": history_list
                }
            }
        else:
            return {"success": 0, "error": "未找到该股票的历史数据"}
    except Exception as e:
        return {"success": 0, "error": f"获取历史数据失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8003"))
    uvicorn.run(
        "src.apps.api.app:app",
        host="0.0.0.0", 
        port=port,
        reload=0,
        workers=1,
        log_level="info"
    )
    # logging.getLogger('src.data.field_mapping').setLevel(logging.WARNING)
    # logging.getLogger('src.ml.features.enhanced_features').setLevel(logging.WARNING)