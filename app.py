from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import StockAnalyzer
import json
import os
from db import DatabaseManager
from akshare_data_provider import AkshareDataProvider
from enhanced_data_provider import EnhancedDataProvider
import pandas as pd
import re
from signal_generator import SignalGenerator
from selector_service import IntelligentStockSelector
from stock_list_manager import StockListManager
from market_selector_service import MarketSelectorService
from data_sync_service import DataSyncService
from concurrent_data_sync_service import ConcurrentDataSyncService
from concurrent_enhanced_data_provider import ConcurrentEnhancedDataProvider
from optimized_enhanced_data_provider import OptimizedEnhancedDataProvider
from datetime import datetime, timedelta
import logging
import time
import asyncio
from contextlib import asynccontextmanager

# 配置日志
logger = logging.getLogger(__name__)

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

class StockRequest(BaseModel):
    symbol: str

analyzer = StockAnalyzer()
# 初始化数据库与数据提供器
_db = DatabaseManager()
_provider = EnhancedDataProvider()  # 使用增强数据提供者，支持过滤逻辑
_optimized_provider = OptimizedEnhancedDataProvider()  # 优化版数据提供者，减少重试延迟
# 初始化服务
data_sync_service = DataSyncService()
concurrent_data_sync_service = ConcurrentDataSyncService(max_workers=8, db_batch_size=50)
concurrent_data_provider = ConcurrentEnhancedDataProvider(max_workers=6)
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
    if os.environ.get("SKIP_PREWARM", "0") not in ("1", "true", "True"):
        async def _prewarm_stock_picks():
            try:
                logger.info("[startup] 开始后台预热 /api/stock-picks 缓存 (limit_symbols=500, top_n=10)")
                await get_stock_picks(top_n=10, limit_symbols=500, force_refresh=True, debug=1)
                logger.info("[startup] 预热完成")
            except Exception as e:
                logger.warning(f"[startup] 预热失败: {e}")
        asyncio.create_task(_prewarm_stock_picks())
    
    print("应用启动，检查数据状态...")
    # 支持通过环境变量跳过启动时的数据同步，以便快速启动服务进行接口联调
    if os.environ.get("SKIP_STARTUP_SYNC", "0") in ("1", "true", "True"):
        print("已设置SKIP_STARTUP_SYNC，跳过启动时数据初始化。")
        return

    try:
        # 检查数据库中是否已有股票数据
        existing_stocks = _db.list_symbols(limit=1)
        
        if not existing_stocks:
            # 首次启动，完整初始化 - 扩大股票覆盖范围，暂时只处理A股
            print("检测到首次启动，开始完整数据初始化...")
            result = await refresh_data(max_symbols=1000, full=True, batch_size=25, delay_seconds=1.5, include_hk=False)
            if result.get("success"):
                data = result.get("data", {})
                print(f"完整数据初始化完成: 处理了{data.get('symbols', 0)}只股票，插入{data.get('inserted_daily_rows', 0)}条日线数据，{data.get('quotes_rows', 0)}条实时行情")
            else:
                print(f"完整数据初始化失败: {result.get('error', '未知错误')}")
        else:
            # 已有数据，进行增量更新，暂时只处理A股
            print(f"检测到已有股票数据，进行增量更新...")
            result = await refresh_data(max_symbols=0, full=False, batch_size=20, delay_seconds=0.8, include_hk=False)  # max_symbols=0表示处理所有股票
            if result.get("success"):
                data = result.get("data", {})
                print(f"增量数据更新完成: 处理了{data.get('symbols', 0)}只股票，插入{data.get('inserted_daily_rows', 0)}条日线数据，{data.get('quotes_rows', 0)}条实时行情")
            else:
                print(f"增量数据更新失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"数据初始化失败: {e}")
        # 不阻止应用启动，即使数据初始化失败


def _to_a_symbol_with_suffix(code: str | None) -> str | None:
    if code is None or str(code).lower() == 'nan':
        return None
    code = str(code).zfill(6)
    # 统一外部构造为 .SH，内部保持对 .SS 的兼容
    return f"{code}.SH" if code.startswith('6') else f"{code}.SZ"


def _to_h_symbol_with_suffix(code: str | None) -> str | None:
    if code is None or str(code).lower() == 'nan':
        return None
    code = str(code).zfill(5)
    return f"{code}.HK"


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/analysis")
async def read_analysis(symbol: str = None):
    """分析页面路由"""
    return FileResponse("static/analysis.html")

@app.post("/api/analyze")
async def analyze_stock(request: StockRequest):
    """分析股票"""
    try:
        # analyzer.analyze_stock 会在内部处理异常并返回一个包含 success 标志的字典
        result = analyzer.analyze_stock(request.symbol)
        
        # 无论分析成功与否，都返回一个成功的HTTP响应，
        # 具体的业务成功或失败由 'data' 中的内容决定
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        # 这个捕获块现在只处理 analyze_stock 调用之外的意外错误
        # 比如请求体验证失败等
        return {
            "success": True,
            "data": {
                "success": False,
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
async def refresh_data(max_symbols: int = 50, full: bool = False, batch_size: int = 10, delay_seconds: float = 1.0, include_hk: bool = False):
    """
    刷新全市场股票元数据与增量日线/实时行情：
    - 写入 stocks（包含全市场股票信息）
    - 增量写入 prices_daily（根据已有最新日期向后补齐）
    - 追加 quotes_realtime 快照
    
    参数说明：
    - max_symbols: 限制本次处理的股票数，避免耗时过长（0表示处理所有股票）
    - full: 是否完整初始化（True）还是增量更新（False）
    - batch_size: 批量处理大小，防止API限流
    - delay_seconds: 批次间延时，控制API调用频率
    - include_hk: 是否包含港股（False表示只处理A股）
    """
    try:
        # 1) 获取全市场股票列表并写入 stocks 表
        print("正在获取全市场股票列表...")
        all_stocks = _provider.get_all_stock_list()
        stock_rows = []
        
        if all_stocks is None or all_stocks.empty:
            print("获取全市场股票列表失败，使用A+H股配对作为回退方案...")
            # 回退方案：使用 A+H 配对列表
            pairs = _provider.get_ah_stock_list()
            if pairs is None or pairs.empty:
                print("A+H股配对也失败，使用种子股票列表...")
                # 最终回退：使用种子股票
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
                # 使用A+H配对数据
                for _, r in pairs.iterrows():
                    name = r.get('name')
                    code_a = r.get('code_a') if 'code_a' in pairs.columns else None
                    code_h = r.get('code_h') if 'code_h' in pairs.columns else None
                    sym_a = _to_a_symbol_with_suffix(code_a)
                    sym_h = _to_h_symbol_with_suffix(code_h)
                    if sym_a:
                        market = 'SH' if (str(sym_a).endswith('.SH') or str(sym_a).endswith('.SS')) else 'SZ'
                        stock_rows.append({"symbol": sym_a, "name": name, "market": market, "ah_pair": sym_h})
                    if sym_h:
                        stock_rows.append({"symbol": sym_h, "name": name, "market": "HK", "ah_pair": sym_a})
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
                    if len(code_str) != 6:
                        continue
                    symbol = f"{code_str}.BJ"
                elif market == 'HK':
                    if not include_hk:
                        continue  # 跳过港股
                    # 港股常见为 4-5 位，统一补零到 5 位
                    if len(code_str) not in (4, 5):
                        continue
                    symbol = f"{code_str.zfill(5)}.HK"
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
                if symbol.endswith('.HK'):
                    market = 'H'
                    # 港股代码格式：保持5位数字，前面补0
                    code_digits = re.sub(r"\D", "", symbol.split('.')[0]).zfill(5)
                else:
                    market = 'A'
                    code_digits = re.sub(r"\D", "", symbol.split('.')[0]).zfill(6)
                start_date = None
                if not full and latest_map.get(symbol):
                    try:
                        start_date = (pd.to_datetime(latest_map[symbol]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    except Exception:
                        start_date = None
                # 对于港股，需要传递字符串格式的代码（如"02318"）
                symbol_for_akshare = code_digits if market == 'A' else str(code_digits).zfill(5)
                df_hist = _provider.get_ah_daily(symbol=symbol_for_akshare, market=market, start_date=start_date)
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

        # 5) 实时行情快照
        quotes_rows = []
        spot = _provider.get_ah_spot()
        if spot is not None and not spot.empty:
            # A 股侧
            if set(['code_a', 'price_a']).issubset(spot.columns):
                for _, r in spot.iterrows():
                    c, p = r.get('code_a'), r.get('price_a')
                    if pd.notna(c) and pd.notna(p):
                        sym = _to_a_symbol_with_suffix(str(c))
                        if sym:
                            quotes_rows.append({
                                'symbol': sym,
                                'price': float(p),
                                'change_pct': r.get('pct_chg_a'),
                                'volume': r.get('volume_a') if 'volume_a' in spot.columns else None
                            })
            # H 股侧
            if set(['code_h', 'price_h']).issubset(spot.columns):
                for _, r in spot.iterrows():
                    c, p = r.get('code_h'), r.get('price_h')
                    if pd.notna(c) and pd.notna(p):
                        sym = _to_h_symbol_with_suffix(str(c))
                        if sym:
                            quotes_rows.append({
                                'symbol': sym,
                                'price': float(p),
                                'change_pct': r.get('pct_chg_h'),
                                'volume': r.get('volume_h') if 'volume_h' in spot.columns else None
                            })
        if quotes_rows:
            _db.insert_quotes_realtime(pd.DataFrame(quotes_rows), source='akshare_ah_spot')

        return {"success": True, "data": {"symbols": len(symbols), "inserted_daily_rows": total_rows, "quotes_rows": len(quotes_rows)}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/stock-picks")
async def get_stock_picks(top_n: int = 10, limit_symbols: int | None = 500, force_refresh: bool = False, debug: int = 0):
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
                    resp = {**resp, 'cached': True}
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
            from stock_status_filter import StockStatusFilter
            stock_filter = StockStatusFilter()
            
            valid_symbols = []
            for symbol_info in symbols_data:
                filter_check = stock_filter.should_filter_stock(
                    symbol_info.get('name', ''), 
                    symbol_info.get('symbol', ''),
                    include_st=True,
                    include_suspended=True,
                    db_manager=_db,
                    exclude_star_market=True,
                    last_n_days=30
                )
                
                if not filter_check['should_filter']:
                    valid_symbols.append(symbol_info['symbol'])
            t_filter1 = time.monotonic()
            
            # 近30日换手率优先排序后再截取
            t_rank0 = time.monotonic()
            try:
                lookback_days = 30
                date_threshold = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

                def _chunk(lst, n):
                    for i in range(0, len(lst), n):
                        yield lst[i:i+n]

                # 聚合近30日成交额（缺失时用量价乘积近似）
                frames = []
                mkt_caps_frames = []
                with _db.get_conn() as conn:
                    # prices_daily
                    for chunk in _chunk(valid_symbols, 800):
                        placeholders = ','.join(['?'] * len(chunk))
                        sql = f"SELECT symbol, date, volume, amount, close FROM prices_daily WHERE symbol IN ({placeholders}) AND date >= ?"
                        params = list(chunk) + [date_threshold]
                        df_chunk = pd.read_sql_query(sql, conn, params=params)
                        if df_chunk is not None and not df_chunk.empty:
                            frames.append(df_chunk)
                    # stocks 市值
                    for chunk in _chunk(valid_symbols, 800):
                        placeholders = ','.join(['?'] * len(chunk))
                        sql = f"SELECT symbol, market_cap FROM stocks WHERE symbol IN ({placeholders})"
                        params = list(chunk)
                        df_caps = pd.read_sql_query(sql, conn, params=params)
                        if df_caps is not None and not df_caps.empty:
                            mkt_caps_frames.append(df_caps)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    if 'amount' not in df.columns or df['amount'].isna().all():
                        df['amount_fill'] = df['volume'].fillna(0) * df['close'].fillna(0)
                    else:
                        df['amount_fill'] = df['amount'].fillna(df['volume'].fillna(0) * df['close'].fillna(0))
                    agg = df.groupby('symbol', as_index=False)['amount_fill'].sum().rename(columns={'amount_fill': 'sum_amount_30'})
                else:
                    agg = pd.DataFrame({'symbol': valid_symbols, 'sum_amount_30': [0.0] * len(valid_symbols)})

                # 合并市值并计算换手率（以成交额/总市值作为近似）
                if mkt_caps_frames:
                    caps = pd.concat(mkt_caps_frames, ignore_index=True).drop_duplicates(subset=['symbol'])
                    df_rank = pd.merge(agg, caps, on='symbol', how='left')
                else:
                    df_rank = agg.copy()
                    df_rank['market_cap'] = None

                # 计算 rate（可能缺失）
                def _calc_rate(row):
                    try:
                        mc = row.get('market_cap')
                        if mc is not None and pd.notna(mc) and float(mc) > 0:
                            return float(row['sum_amount_30']) / float(mc)
                        return None
                    except Exception:
                        return None

                df_rank['rate'] = df_rank.apply(_calc_rate, axis=1)
                df_rank['has_rate'] = df_rank['rate'].apply(lambda x: 1 if pd.notna(x) else 0)
                df_rank.sort_values(by=['has_rate', 'rate', 'sum_amount_30'], ascending=[False, False, False], inplace=True)
                sorted_symbols = df_rank['symbol'].tolist()
                # 追加未参与排序的剩余标的，保持兼容性
                remaining = [s for s in valid_symbols if s not in set(sorted_symbols)]
                sorted_symbols.extend(remaining)
                symbols = sorted_symbols[:int(limit_symbols)]
            except Exception as e:
                logger.warning(f"按近30日换手率排序失败，回退为原始顺序: {e}")
                symbols = valid_symbols[:int(limit_symbols)]
            t_rank1 = time.monotonic()
            
            picks = selector.predict_stocks(symbols, top_n)
            t_pred1 = time.monotonic()
            
            response = {
                'success': True,
                'data': {
                    'picks': picks or [],
                    'model_type': 'ml_cls+reg' if getattr(selector, 'reg_model_data', None) and getattr(selector, 'cls_model_data', None) else ('machine_learning' if getattr(selector, 'model', None) else 'technical_indicators'),
                    'generated_at': datetime.now().isoformat(),
                    'timings': {
                        'total_sec': round(t_pred1 - t0, 3),
                        'fetch_symbols_sec': round(t_fetch1 - t_fetch0, 3),
                        'filter_sec': round(t_filter1 - t_fetch1, 3),
                        'rank_sec': round(t_rank1 - t_rank0, 3),
                        'predict_sec': round(t_pred1 - t_rank1, 3)
                    } if debug else None
                },
                'cached': False
            }
            
            # 写入缓存
            PICKS_CACHE[cache_key] = {
                'response': response,
                'expires_at': now + timedelta(seconds=PICKS_CACHE_TTL_SECONDS)
            }
            
            return response
        
        # 原有完整路径
        t_full0 = time.monotonic()
        raw_result = selector.get_stock_picks(top_n)
        t_full1 = time.monotonic()
        
        # 统一响应结构：始终返回 {'success': bool, 'data': {'picks': [...], 'model_type': str?, 'generated_at': str, 'timings': {...}?}, 'cached': bool}
        def _normalize_response(res):
            success = True
            data = {}
            # dict 类型
            if isinstance(res, dict):
                success = res.get('success', True)
                # 已是标准结构
                if isinstance(res.get('data'), dict) and 'picks' in res['data']:
                    data = dict(res['data'])
                # 顶层带 picks
                elif 'picks' in res:
                    data = {
                        'picks': res.get('picks') or []
                    }
                    if 'model_type' in res:
                        data['model_type'] = res.get('model_type')
                else:
                    data = {'picks': []}
            # list/tuple 直接视为 picks 列表
            elif isinstance(res, (list, tuple)):
                data = {'picks': list(res)}
            else:
                data = {'picks': []}

            # 补充字段
            if 'generated_at' not in data:
                data['generated_at'] = datetime.now().isoformat()
            # 添加耗时
            data['timings'] = {
                'total_sec': round(t_full1 - t0, 3),
                'full_path_sec': round(t_full1 - t_full0, 3)
            } if debug else None
            return {
                'success': success,
                'data': data,
                'cached': False
            }

        response = _normalize_response(raw_result)

        # 写入缓存并返回
        PICKS_CACHE[cache_key] = {
            'response': response,
            'expires_at': now + timedelta(seconds=PICKS_CACHE_TTL_SECONDS)
        }
        
        return response
        
    except Exception as e:
        return {
            'success': False,
            'message': str(e),
            'data': {'picks': []}
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 股票列表管理API
@app.post("/api/stocks/update")
async def update_stock_list():
    """更新全市场股票列表"""
    try:
        result = stock_list_manager.update_all_stocks()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新股票列表失败: {str(e)}")

@app.get("/api/stocks/summary")
async def get_market_summary():
    """获取市场概览统计"""
    try:
        summary = stock_list_manager.get_market_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场概览失败: {str(e)}")

@app.get("/api/stocks/list")
async def get_stocks(market: str = None, board_type: str = None, limit: int = None):
    """获取股票列表"""
    try:
        if limit:
            stocks = stock_list_manager.get_candidate_stocks(
                market_filter=[market] if market else None,
                board_filter=[board_type] if board_type else None,
                limit=limit
            )
            return {"stocks": stocks, "count": len(stocks)}
        else:
            stocks = stock_list_manager.get_stocks_by_market(market, board_type)
            return {"stocks": stocks, "count": len(stocks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")

@app.get("/api/stocks/candidates")
async def get_candidate_stocks(market: str = None, board_type: str = None, limit: int = 100):
    """获取候选股票代码列表（用于选股分析）"""
    try:
        candidates = stock_list_manager.get_candidate_stocks(
            market_filter=[market] if market else None,
            board_filter=[board_type] if board_type else None,
            limit=limit
        )
        return {"candidates": candidates, "count": len(candidates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取候选股票失败: {str(e)}")

@app.get("/api/markets")
async def get_available_markets():
    """获取可用的市场列表及其状态"""
    try:
        result = market_selector_service.get_available_markets()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/select_stocks_by_markets")
async def select_stocks_by_markets(request: dict):
    """基于多市场选择进行智能选股"""
    try:
        # 支持两种参数名：'markets' 和 'selected_markets'
        selected_markets = request.get('markets', request.get('selected_markets', []))
        selection_criteria = request.get('selection_criteria', {})
        top_n = request.get('top_n', 20)
        
        if not selected_markets:
            return {
                "success": False,
                "error": "请至少选择一个市场"
            }
        
        print(f"开始多市场选股: {selected_markets}, top_n={top_n}")
        
        result = market_selector_service.select_stocks_by_markets(
            selected_markets=selected_markets,
            selection_criteria=selection_criteria,
            top_n=top_n
        )
        
        # 提取选中的股票到顶层字段
        if result.get('success') and result.get('results'):
            stocks = result['results'].get('top_stocks', [])
            result['stocks'] = stocks
            result['stock_count'] = len(stocks)
        else:
            result['stocks'] = []
            result['stock_count'] = 0
        
        return result
        
    except Exception as e:
        print(f"多市场选股失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/market_statistics")
async def get_market_statistics(selected_markets: str = None):
    """获取市场统计信息"""
    try:
        markets_list = None
        if selected_markets:
            markets_list = [m.strip() for m in selected_markets.split(',')]
        
        result = market_selector_service.get_market_statistics(markets_list)
        return result
        
    except Exception as e:
        logger.error(f"获取市场统计失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/sync_data_optimized")
async def sync_market_data_optimized(request: dict = None):
    """
    优化版数据同步API - 使用改进的重试机制和熔断器
    """
    try:
        # 解析请求参数
        if request is None:
            request = {}
        
        sync_type = request.get("sync_type", "incremental")
        markets = request.get("markets", ["A股主板", "创业板", "科创板"])
        batch_size = min(int(request.get("batch_size", 50)), 100)
        delay_seconds = max(0.1, min(float(request.get("delay_seconds", 0.2)), 2.0))
        
        # 推导同步类型
        if sync_type == "incremental":
            period = "30d"
        elif sync_type == "full":
            period = "1y"
        else:
            period = sync_type
        
        logger.info(f"开始优化版数据同步: type={sync_type}, markets={markets}, "
                   f"batch_size={batch_size}, delay={delay_seconds}s")
        
        start_time = time.time()
        
        # 获取股票列表
        stock_symbols = market_selector_service.get_stocks_by_markets(markets)
        if not stock_symbols:
            return {"success": False, "message": "未找到符合条件的股票"}
        
        total_stocks = len(stock_symbols)
        logger.info(f"获取到 {total_stocks} 只股票，开始同步...")
        
        # 使用优化版数据提供者进行同步
        success_count = 0
        error_count = 0
        skipped_count = 0
        errors = []
        
        for i, symbol in enumerate(stock_symbols):
            try:
                # 获取历史数据
                data = _optimized_provider.get_stock_historical_data(symbol, period)
                
                if data is not None and not data.empty:
                    # 保存到数据库
                    with _db.get_conn() as conn:
                        data.to_sql('stock_data', conn, if_exists='append', index=False)
                    success_count += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"优化版同步进度: {i + 1}/{total_stocks} "
                                   f"(成功: {success_count}, 错误: {error_count})")
                else:
                    skipped_count += 1
                    
            except Exception as e:
                error_count += 1
                error_msg = f"{symbol}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"优化版同步失败: {error_msg}")
            
            # 批次间延迟
            if (i + 1) % batch_size == 0 and i < total_stocks - 1:
                time.sleep(delay_seconds)
        
        elapsed_time = time.time() - start_time
        
        # 获取数据源统计信息
        source_stats = _optimized_provider.get_source_statistics()
        
        result = {
            "success": True,
            "message": f"优化版数据同步完成",
            "statistics": {
                "total_stocks": total_stocks,
                "success_count": success_count,
                "error_count": error_count,
                "skipped_count": skipped_count,
                "success_rate": f"{success_count/total_stocks*100:.1f}%",
                "elapsed_time": f"{elapsed_time:.2f}s",
                "avg_time_per_stock": f"{elapsed_time/total_stocks:.2f}s"
            },
            "source_statistics": source_stats,
            "errors": errors[:10] if errors else []  # 只返回前10个错误
        }
        
        logger.info(f"优化版数据同步完成: 成功 {success_count}/{total_stocks} "
                   f"({success_count/total_stocks*100:.1f}%), 耗时 {elapsed_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"优化版数据同步异常: {e}")
        return {
            "success": False,
            "message": f"优化版数据同步失败: {str(e)}",
            "statistics": {},
            "source_statistics": {},
            "errors": [str(e)]
        }


@app.post("/api/sync_data_concurrent")
async def sync_market_data_concurrent(request: dict = None):
    """并发数据同步API - 高性能版本"""
    try:
        if request is None:
            request = {}
            
        sync_type = request.get('sync_type', 'incremental')  # auto, full, incremental
        markets = request.get('markets', ['SH', 'SZ'])  # 默认同步沪深市场
        force_full = request.get('force_full', False)
        preferred_sources = request.get('preferred_sources')  # e.g. ["eastmoney","sina","tencent","xueqiu","akshare"]
        max_workers = request.get('max_workers', 8)  # 并发线程数
        db_batch_size = request.get('db_batch_size', 50)  # 数据库批量操作大小
        req_max_symbols = request.get('max_symbols', 0)
        # TopN筛选参数
        top_n_by = request.get('top_n_by')  # 'market_cap' 或 'amount'
        top_n = request.get('top_n', 0)
        amount_window_days = request.get('amount_window_days', 5)
        
        # auto 模式：根据force_full推导；明确full则全量
        effective_sync_type = 'full' if force_full or sync_type == 'full' else 'incremental'
        
        logger.info(f"开始并发数据同步: sync_type={effective_sync_type}, markets={markets}, max_workers={max_workers}, max_symbols={req_max_symbols}")
        
        result = concurrent_data_sync_service.sync_market_data(
            sync_type=effective_sync_type,
            markets=markets,
            max_symbols=req_max_symbols,
            max_workers=max_workers,
            db_batch_size=db_batch_size,
            preferred_sources=preferred_sources,
            top_n_by=top_n_by,
            top_n=top_n,
            amount_window_days=amount_window_days
        )
        
        return result
        
    except Exception as e:
        logger.error(f"并发数据同步失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/sync_data")
async def sync_market_data(request: dict = None):
    """手动触发行情数据同步"""
    try:
        if request is None:
            request = {}
            
        sync_type = request.get('sync_type', 'auto')  # auto, full, incremental
        markets = request.get('markets', ['SH', 'SZ'])  # 默认同步沪深市场
        force_full = request.get('force_full', False)
        preferred_sources = request.get('preferred_sources')  # e.g. ["eastmoney","sina","tencent","xueqiu","akshare"]
        batch_size = request.get('batch_size', 10)
        delay_seconds = request.get('delay_seconds', 1.0)
        # 默认增量也跑全市场（不限制数量），若需要限制可在请求中传入具体数值
        req_max_symbols = request.get('max_symbols', 0)
        # 新增：TopN筛选参数
        top_n_by = request.get('top_n_by')  # 'market_cap' 或 'amount'
        top_n = request.get('top_n', 0)
        amount_window_days = request.get('amount_window_days', 5)
        
        # auto 模式：根据force_full推导；明确full则全量
        effective_sync_type = 'full' if force_full or sync_type == 'full' else 'incremental'
        
        logger.info(f"开始数据同步: sync_type={effective_sync_type}, markets={markets}, force_full={force_full}, preferred_sources={preferred_sources}, max_symbols={req_max_symbols}, top_n_by={top_n_by}, top_n={top_n}, amount_window_days={amount_window_days}")
        
        result = data_sync_service.sync_market_data(
            sync_type=effective_sync_type,
            markets=markets,
            max_symbols=req_max_symbols,  # 0 表示不限制（全市场）
            batch_size=batch_size,
            delay_seconds=delay_seconds,
            preferred_sources=preferred_sources,
            top_n_by=top_n_by,
            top_n=top_n,
            amount_window_days=amount_window_days
        )
        
        return result
        
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/data/sync")
async def sync_data(request: dict):
    """同步市场数据"""
    try:
        sync_type = request.get('sync_type', 'incremental')
        markets = request.get('markets', ['SH', 'SZ'])
        max_symbols = request.get('max_symbols', 0)  # 默认不限制
        batch_size = request.get('batch_size', 10)
        delay_seconds = request.get('delay_seconds', 1.0)
        preferred_sources = request.get('preferred_sources')
        # 新增：TopN筛选参数
        top_n_by = request.get('top_n_by')  # 'market_cap' 或 'amount'
        top_n = request.get('top_n', 0)
        amount_window_days = request.get('amount_window_days', 5)
        
        # 如果显式要求full或max_symbols为0，执行全量；否则增量
        effective_sync_type = 'full' if sync_type == 'full' or max_symbols == 0 and sync_type != 'incremental' else 'incremental'
        
        result = data_sync_service.sync_market_data(
            sync_type=effective_sync_type,
            markets=markets,
            max_symbols=max_symbols,
            batch_size=batch_size,
            delay_seconds=delay_seconds,
            preferred_sources=preferred_sources,
            top_n_by=top_n_by,
            top_n=top_n,
            amount_window_days=amount_window_days
        )
        
        return result
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# 新增：覆盖率与新鲜度报表 API
@app.get("/api/coverage_report")
async def coverage_report(markets: str = None, window_days: int = 5):
    """覆盖率与新鲜度简报"""
    try:
        market_list = markets.split(',') if markets else None
        result = data_sync_service.coverage_and_freshness_report(market_list, window_days)
        return result
    except Exception as e:
        logger.error(f"生成覆盖率与新鲜度报表失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/sync_status")
async def get_sync_status():
    """获取数据同步状态"""
    try:
        result = data_sync_service.get_last_sync_info()
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"获取同步状态失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/data_freshness")
async def check_data_freshness(markets: str = None):
    """检查数据新鲜度"""
    try:
        market_list = markets.split(',') if markets else None
        result = data_sync_service.check_data_freshness(market_list)
        return result
        
    except Exception as e:
        logger.error(f"检查数据新鲜度失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)