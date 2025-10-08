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
from src.cache.prefetch_scheduler import CachePrefetchScheduler
from src.services.portfolio.portfolio_service import generate_portfolio_holdings

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
from zoneinfo import ZoneInfo

# 配置日志
logger = logging.getLogger(__name__)
logging.getLogger('src.data.field_mapping').setLevel(logging.WARNING)
logging.getLogger('src.ml.features.enhanced_features').setLevel(logging.WARNING)

_APP_TZ_NAME = os.getenv("APP_TIMEZONE", "Asia/Shanghai")
try:
    _APP_TZ = ZoneInfo(_APP_TZ_NAME)
except Exception:
    logger.warning("APP_TIMEZONE '%s' 无效，使用 UTC", _APP_TZ_NAME)
    _APP_TZ = ZoneInfo("UTC")
def _local_now() -> datetime:
    return datetime.now(_APP_TZ).replace(tzinfo=None)

cache_prefetch_scheduler: CachePrefetchScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期处理"""
    await startup_event()
    try:
        yield
    finally:
        global cache_prefetch_scheduler
        if cache_prefetch_scheduler:
            try:
                cache_prefetch_scheduler.shutdown(wait=False)
            except Exception as exc:  # pragma: no cover
                logger.warning("CachePrefetchScheduler shutdown failed: %s", exc)
            finally:
                cache_prefetch_scheduler = None

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
    global selector_service, signal_generator, stock_list_manager, cache_prefetch_scheduler
    
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

    # 启动缓存预取调度器
    try:
        preload_env = os.getenv("PRELOAD_SYMBOLS", "")
        prefetch_symbols = [s.strip() for s in preload_env.split(',') if s.strip()]
        if not prefetch_symbols:
            symbol_limit = int(os.getenv("PREFETCH_SYMBOL_LIMIT", "200"))
            symbol_rows = _db.list_symbols(markets=['SH', 'SZ'], limit=symbol_limit)
            prefetch_symbols = [row.get('symbol') for row in symbol_rows if row.get('symbol')]

        lookback_days = int(os.getenv("PREFETCH_LOOKBACK_DAYS", "365"))
        interval_minutes = int(os.getenv("PREFETCH_INTERVAL_MINUTES", "60"))

        if _unified_data_access is not None and prefetch_symbols:
            cache_prefetch_scheduler = CachePrefetchScheduler(
                _unified_data_access,
                symbols=prefetch_symbols,
                lookback_days=lookback_days,
                interval_minutes=interval_minutes,
            )
            cache_prefetch_scheduler.start()
            logger.info(
                "[startup] CachePrefetchScheduler started (symbols=%s, lookback=%s, interval=%smin)",
                len(prefetch_symbols),
                lookback_days,
                interval_minutes,
            )
        else:
            logger.info(
                "[startup] 跳过缓存预取：access=%s, symbols=%s",
                bool(_unified_data_access),
                len(prefetch_symbols),
            )
    except Exception as exc:
        logger.warning("[startup] 启动缓存预取任务失败: %s", exc)
    
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

@app.get("/index.html")
async def read_index_html():
    """兼容直接访问 index.html"""
    return FileResponse("static/index.html")

@app.get("/analysis")
async def read_analysis(symbol: Optional[str] = None):
    """分析页面路由"""
    return FileResponse("static/analysis.html")

@app.get("/portfolios")
async def read_portfolios():
    """组合列表页面路由"""
    return FileResponse("static/portfolios.html")

@app.get("/portfolios.html")
async def read_portfolios_html():
    """兼容直接访问 portfolios.html"""
    return FileResponse("static/portfolios.html")

@app.get("/portfolio_detail.html")
async def read_portfolio_detail():
    """组合详情页面路由"""
    return FileResponse("static/portfolio_detail.html")

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

        # 3）委托给 sync_market_data_optimized 处理增量/全量同步，避免重复代码
        request_params = {
            "symbols": symbols,
            "batch_size": batch_size,
            "delay_seconds": delay_seconds,
            "step_days": 15,
        }
        # full 情况下不限制 trade_date，由 sync_market_data_optimized 内部策略决定
        result = await sync_market_data_optimized(request_params)
        return result
    except Exception as e:
        return {"success": 0, "error": str(e)}

@app.get("/api/stock-picks")
async def get_stock_picks(top_n: int = 60, limit_symbols: Optional[int] = 3000, force_refresh: bool = 0, debug: int = 0):
    """
    获取智能选股推荐
    可选 limit_symbols 用于限制参与预测的股票数量，以便快速联调与验证。
    新增：内存缓存与耗时统计（debug=1 返回详细耗时）。
    """
    try:
        # 复用全局 selector 实例
        selector = selector_service

        def with_cache_metrics(payload: dict) -> dict:
            uda = getattr(selector, "data_access", None) or _unified_data_access
            if uda is None or not isinstance(payload, dict):
                return payload
            try:
                report = uda.get_data_quality_report()
                perf = report.get("performance", {}) if isinstance(report, dict) else {}
                if perf:
                    cache_metrics = {
                        "requests": perf.get("requests", 0),
                        "avg_latency_ms": perf.get("avg_latency_ms"),
                        "l0_hit_rate": perf.get("l0_hit_rate"),
                        "l1_hit_rate": perf.get("l1_hit_rate"),
                        "l2_hit_rate": perf.get("l2_hit_rate"),
                        "db_share": perf.get("db_share"),
                        "provider_share": perf.get("provider_share"),
                    }
                    payload.setdefault("data", {})["cache_metrics"] = cache_metrics
            except Exception as exc:  # pragma: no cover
                logger.debug("附加缓存指标失败: %s", exc)
            return payload

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
            response = with_cache_metrics(response)

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
        response = with_cache_metrics(response)

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

# ---------- Portfolios CRUD ----------
from fastapi.responses import JSONResponse
from src.services.portfolio.portfolio_management_service import (
    list_portfolios,
    create_portfolio_auto,
    get_portfolio_detail,
    delete_portfolio,
)


def _parse_as_of(as_of: str | None) -> datetime:
    if not as_of:
        return _local_now()
    try:
        return datetime.fromisoformat(as_of)
    except ValueError:
        try:
            return datetime.strptime(as_of, "%Y-%m-%d")
        except ValueError:
            return _local_now()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return default


@app.get("/api/portfolios")
def api_list_portfolios(as_of: str | None = None, refresh: str | None = None):
    as_of_dt = _parse_as_of(as_of)
    refresh_flag = _parse_bool(refresh, default=False)
    portfolios = list_portfolios(as_of=as_of_dt)
    if refresh_flag:
        for item in portfolios:
            try:
                get_portfolio_detail(item["id"], as_of=as_of_dt, refresh=True)
            except Exception:
                logger.warning("刷新组合 %s 失败，返回旧数据", item["id"], exc_info=True)
        portfolios = list_portfolios(as_of=as_of_dt)
    return {
        "success": True,
        "generated_at": as_of_dt.isoformat(),
        "count": len(portfolios),
        "portfolios": portfolios,
    }

@app.post("/api/portfolios")
def api_create_portfolio(request: dict):
    name = request.get("name") or f"组合{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    mode = request.get("mode", "auto")
    top_n = int(request.get("top_n", 20))
    capital = float(request.get("initial_capital", 1_000_000))
    if top_n < 1:
        return JSONResponse(status_code=400, content={"success": False, "error": "top_n must be >= 1"})
    if capital <= 0:
        return JSONResponse(status_code=400, content={"success": False, "error": "initial_capital must be > 0"})
    if mode != "auto":
        return JSONResponse(status_code=400, content={"success": False, "error": "Unsupported mode"})
    try:
        data = create_portfolio_auto(name=name, top_n=top_n, initial_capital=capital)
        return {"success": True, "portfolio": data}
    except Exception as exc:  # pragma: no cover - 防御性捕获
        logger.exception("自动创建组合失败")
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

@app.get("/api/portfolios/{pid}")
def api_get_portfolio(pid: int, as_of: str | None = None, refresh: str | None = None):
    as_of_dt = _parse_as_of(as_of)
    refresh_flag = _parse_bool(refresh, default=True)
    data = get_portfolio_detail(pid, as_of=as_of_dt, refresh=refresh_flag)
    if not data:
        return JSONResponse(status_code=404, content={"success": False, "error": "Portfolio not found"})
    return {
        "success": True,
        "generated_at": as_of_dt.isoformat(),
        "portfolio": data,
    }

@app.delete("/api/portfolios/{pid}")
def api_delete_portfolio(pid: int):
    try:
        removed = delete_portfolio(pid)
    except Exception as exc:
        logger.exception("删除组合失败")
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    if not removed:
        return JSONResponse(status_code=404, content={"success": False, "error": "Portfolio not found"})
    return {"success": True}

@app.get("/api/portfolios/{pid}/holdings")
def api_get_portfolio_holdings(pid: int, as_of: str | None = None, refresh: str | None = None):
    as_of_dt = _parse_as_of(as_of)
    refresh_flag = _parse_bool(refresh, default=False)
    data = get_portfolio_detail(pid, as_of=as_of_dt, refresh=refresh_flag)
    if not data:
        return JSONResponse(status_code=404, content={"success": False, "error": "Portfolio not found"})
    holdings = data.get("holdings", [])
    return {
        "success": True,
        "generated_at": as_of_dt.isoformat(),
        "holdings": holdings,
        "metrics": data.get("metrics", {}),
        "nav_history": data.get("nav_history", []),
    }

# ---------- Portfolio Holdings (simulated) ----------

@app.get("/api/portfolio/holdings")
def get_portfolio_holdings(as_of_date: str | None = None, top_n: int = 20):
    try:
        resp = generate_portfolio_holdings(as_of_date=as_of_date, top_n=top_n)
        if not resp.get("success", True):
            return JSONResponse(status_code=500, content=resp)
        return resp
    except Exception as e:
        logger.exception("获取组合持仓失败")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


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
        import time  # 新增用于耗时统计
        start_ts = time.time()
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
                # 日期批量同步后，检查仍落后于昨日的股票并补漏
                try:
                    latest_map_raw = _db.get_latest_dates_by_symbol()
                    latest_map = {sym: pd.to_datetime(dt_str, errors="coerce").normalize()
                                  for sym, dt_str in latest_map_raw.items() if pd.notnull(pd.to_datetime(dt_str, errors="coerce"))}
                    target_syms = symbols or list(latest_map.keys())
                    symbols_pending = [s for s in target_syms if (latest_map.get(s) or pd.Timestamp("1970-01-01")) < pd.to_datetime(yesterday)]
                    if symbols_pending:
                        inc_stats = await _unified_data_access.sync_market_data(
                            symbols=symbols_pending,
                            batch_size=batch_size,
                            delay=delay_seconds
                        )
                        # 合并统计
                        stats["synced"] = stats.get("synced", 0) + inc_stats.get("synced", 0)
                        stats["errors"] = stats.get("errors", 0) + inc_stats.get("errors", 0)
                except Exception as gap_err:
                    logger.warning(f"批量同步后补漏失败: {gap_err}")
            except Exception as date_sync_err:
                logger.warning(f"按日期批量同步失败，回退逐股票增量同步: {date_sync_err}")
                stats = await _unified_data_access.sync_market_data(
                    symbols=symbols,  # None 表示全市场
                    batch_size=batch_size,
                    delay=delay_seconds
                )
        # 组装统计信息供前端展示
        total_ops = stats.get("synced", 0) + stats.get("errors", 0)
        statistics = {
            "success_rate": round(stats.get("synced", 0) / total_ops, 4) if total_ops else 0,
            "elapsed_time": round(time.time() - start_ts, 2),
            "avg_time_per_stock": round((time.time() - start_ts) / stats.get("synced", 1), 2) if stats.get("synced", 0) else None,
        }
        # 采集源相关统计(如果后台返回则透传，否则默认0)
        source_statistics = {
            "total_retries": stats.get("total_retries", 0),
            "circuit_breaker_triggers": stats.get("circuit_breaker_triggers", 0),
        }

        return {"success": 1, "data": stats, "statistics": statistics, "source_statistics": source_statistics}
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
