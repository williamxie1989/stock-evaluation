"""端到端回测脚本：加载模型→选股→建仓→调仓→收益评估

用法示例：
python scripts/run_portfolio_pipeline.py \
    --model_dir models/good \
    --classifier xgboost_cls \
    --regressor xgboost_reg \
    --w_model 0.4 --w_signal 0.4 --w_risk 0.2 \
    --top_n 20 --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List

# 使脚本可独立运行：将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.trading.portfolio.portfolio_pipeline import PortfolioPipeline
from src.trading.systems.adaptive_trading_system import AdaptiveTradingSystem
from src.core.unified_data_access_factory import create_unified_data_access

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio pipeline with adaptive trading system")
    parser.add_argument('--model_dir', type=str, default='models', help='模型目录')
    parser.add_argument('--classifier', type=str, default='classifier', help='分类模型文件名(无.pkl)')
    parser.add_argument('--regressor', type=str, default='regressor', help='回归模型文件名(无.pkl)')
    parser.add_argument('--w_model', type=float, default=0.5, help='模型分权重')
    parser.add_argument('--w_signal', type=float, default=0.3, help='信号分权重')
    parser.add_argument('--w_risk', type=float, default=0.2, help='风险分权重')
    parser.add_argument('--top_n', type=int, default=20, help='Top N 选股数量')
    parser.add_argument('--start', type=str, default='2024-01-01', help='回测开始日期')
    parser.add_argument('--end', type=str, default='2024-06-30', help='回测结束日期')
    return parser


def get_candidate_stocks(limit: int = 500) -> List[str]:
    """获取候选股票集合，避免一次性处理过多"""
    data_access = create_unified_data_access()
    stocks_df = data_access.get_all_stock_list()
    cands: List[str] = []
    if stocks_df is not None and not stocks_df.empty:
        for _, r in stocks_df.iterrows():
            sym = r.get('symbol')
            if isinstance(sym, str) and (sym.endswith('.SH') or sym.endswith('.SZ')):
                cands.append(sym)
            if len(cands) >= limit:
                break
    return cands


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    args = build_arg_parser().parse_args()

    # ---------- 初始化组合管线 ----------
    pipeline = PortfolioPipeline(
        initial_capital=100_000,
        commission_rate=0.0003,
        lookback_days=120,
        top_n=args.top_n,
        rebalance_freq='30D',
        model_dir=args.model_dir,
        classifier_name=args.classifier,
        regressor_name=args.regressor,
        w_model=args.w_model,
        w_signal=args.w_signal,
        w_risk=args.w_risk,
    )

    candidates = get_candidate_stocks(limit=300)

    # ---------- 执行管线 ----------
    result = pipeline.run(start_date=args.start, end_date=args.end, candidates=candidates)
    metrics = result.get('metrics', {})
    nav = result.get('nav')
    picks_history = result.get('picks_history', [])

    logger.info('Backtest Metrics: %s', metrics)
    if picks_history:
        logger.info('First period picks: %s', picks_history[0])

    # ---------- AdaptiveTradingSystem 集成示例 ----------
    ats = AdaptiveTradingSystem(initial_capital=100_000)

    # 以回测开始日期的 picks 建仓
    start_dt = pd.Timestamp(args.start)
    initial_picks = pipeline.pick_stocks(as_of_date=start_dt, candidates=candidates)
    for p in initial_picks:
        # 获取最新价格（start_dt 收盘价）
        df_price = pipeline._fetch_history(p.symbol, end_date=start_dt, days=5)
        if df_price.empty:
            continue
        price = float(df_price['close'].iloc[-1])
        ats.analyze_market_state(df_price)
        ats.assess_risk_level(df_price)
        ats.adapt_trading_params(ats.market_state, ats.risk_level)
        ats.execute_trade(symbol=p.symbol, signal='BUY', price=price)

    # 根据结束日期的价格检查止盈止损
    end_dt = pd.Timestamp(args.end)
    price_dict: Dict[str, float] = {}
    for pos in ats.positions.values():
        sym = pos['symbol'] if isinstance(pos, dict) else pos.get('symbol')
        df_price = pipeline._fetch_history(sym, end_date=end_dt, days=5)
        if not df_price.empty:
            price_dict[sym] = float(df_price['close'].iloc[-1])
    alerts = ats.evaluate_positions(price_dict)
    logger.info('Position alerts: %s', alerts)

    # ---------- 导出结果 ----------
    out_dir = Path('outputs/portfolio')
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(nav, pd.Series) and not nav.empty:
        nav.to_csv(out_dir / 'nav.csv')
    pd.DataFrame(picks_history).to_csv(out_dir / 'picks_history.csv', index=False)
    pd.DataFrame(alerts).to_csv(out_dir / 'position_alerts.csv', index=False)


if __name__ == '__main__':
    main()
