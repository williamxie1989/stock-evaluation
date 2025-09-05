from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import StockAnalyzer
import json
import os

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
        {"symbol": "600036.SS", "name": "招商银行"},
        {"symbol": "601318.SS", "name": "中国平安"},
        {"symbol": "600519.SS", "name": "贵州茅台"},
        {"symbol": "000858.SZ", "name": "五粮液"},
        {"symbol": "000333.SZ", "name": "美的集团"},
        {"symbol": "000001.SS", "name": "上证指数"},
        {"symbol": "399001.SZ", "name": "深证成指"},
        {"symbol": "000300.SS", "name": "沪深300"}
    ]
    return {"stocks": popular_stocks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)