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
import pandas as pd
import re
from signal_generator import SignalGenerator
from selector_service import IntelligentStockSelector
from stock_list_manager import StockListManager
from market_selector_service import MarketSelectorService
from data_sync_service import DataSyncService
from datetime import datetime
import logging

# 配置日志
logger = logging.getLogger(__name__)

app = FastAPI(title="股票分析系统", description="基于大模型的股票分析系统")

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
_provider = AkshareDataProvider()
# 初始化服务
data_sync_service = DataSyncService()
market_selector_service = MarketSelectorService()

@app.on_event("startup")
async def startup_event():
    """
    应用启动时智能初始化数据
    """
    global selector_service, signal_generator, stock_list_manager
    
    selector_service = IntelligentStockSelector()
    signal_generator = SignalGenerator()
    stock_list_manager = StockListManager()
    
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
                
                # 根据市场类型生成symbol
                if market == 'SH':
                    symbol = f"{code}.SH"
                elif market == 'SZ':
                    symbol = f"{code}.SZ"
                elif market == 'BJ':
                    symbol = f"{code}.BJ"
                elif market == 'HK':
                    if not include_hk:
                        continue  # 跳过港股
                    symbol = f"{code}.HK"
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
async def get_stock_picks(top_n: int = 10, limit_symbols: int | None = None):
    """
    获取智能选股推荐
    可选 limit_symbols 用于限制参与预测的股票数量，以便快速联调与验证。
    """
    try:
        from selector_service import IntelligentStockSelector
        
        # 使用全局的数据库管理器实例
        selector = IntelligentStockSelector(_db)
        
        # 快速路径：限制参与预测的股票数量
        if limit_symbols and limit_symbols > 0:
            # 优先加载新模型，失败回退旧模型；若均失败则回退到原逻辑
            if not (selector.load_models(period='30d') or selector.load_model()):
                # 无模型可用，回退到原有逻辑（可能采用技术指标）
                return selector.get_stock_picks(top_n)
            
            symbols_data = _db.list_symbols()
            symbols = [s['symbol'] for s in symbols_data][:int(limit_symbols)]
            picks = selector.predict_stocks(symbols, top_n)
            
            return {
                'success': True,
                'data': {
                    'picks': picks,
                    'model_type': 'ml_cls+reg' if getattr(selector, 'reg_model_data', None) and getattr(selector, 'cls_model_data', None) else ('machine_learning' if getattr(selector, 'model', None) else 'technical_indicators'),
                    'generated_at': datetime.now().isoformat()
                }
            }
        
        # 原有完整路径
        result = selector.get_stock_picks(top_n)
        return result
        
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

@app.post("/api/sync_data")
async def sync_market_data(request: dict = None):
    """手动触发行情数据同步"""
    try:
        if request is None:
            request = {}
            
        sync_type = request.get('sync_type', 'auto')  # auto, full, incremental
        markets = request.get('markets', ['SH', 'SZ'])  # 默认同步沪深市场
        force_full = request.get('force_full', False)
        
        logger.info(f"开始数据同步: sync_type={sync_type}, markets={markets}, force_full={force_full}")
        
        result = data_sync_service.sync_market_data(
            sync_type=sync_type,
            markets=markets,
            max_symbols=0 if force_full else 100  # 全量同步时不限制，增量同步限制数量
        )
        
        return result
        
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
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

@app.post("/api/data/sync")
async def sync_data(request: dict):
    """同步市场数据"""
    try:
        sync_type = request.get('sync_type', 'incremental')
        markets = request.get('markets', ['SH', 'SZ'])
        max_symbols = request.get('max_symbols', 100)
        batch_size = request.get('batch_size', 10)
        
        result = data_sync_service.sync_market_data(
            sync_type=sync_type,
            markets=markets,
            max_symbols=max_symbols,
            batch_size=batch_size
        )
        
        return result
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)