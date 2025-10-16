#!/usr/bin/env python3
"""将季度基本面数据日频化并写回数据库。"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.unified_data_access import UnifiedDataAccessLayer, DataAccessConfig
from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.ml.features.fundamental_features import FundamentalFeatureGenerator

logger = logging.getLogger(__name__)


def load_symbols(args) -> list[str]:
    if args.symbols:
        return args.symbols
    if args.symbol_file:
        path = Path(args.symbol_file)
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    raise ValueError("必须使用 --symbols 或 --symbol-file 提供股票列表")


def main():
    parser = argparse.ArgumentParser(description="基本面季度数据日频化")
    parser.add_argument("--symbols", nargs="*", help="股票代码列表")
    parser.add_argument("--symbol-file", help="存放股票代码的文件")
    parser.add_argument("--start-date", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    symbols = load_symbols(args)
    if not symbols:
        logger.error("未获取到有效股票列表")
        return

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)

    config = DataAccessConfig()
    data_access = UnifiedDataAccessLayer(config=config)
    db_manager = UnifiedDatabaseManager()
    generator = FundamentalFeatureGenerator(db_manager=db_manager, use_db_cache=True)

    success = 0
    total_rows = 0

    for symbol in symbols:
        try:
            price_df = data_access.get_stock_data(
                symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                adjust_mode='none'
            )
            if price_df is None or len(price_df) == 0:
                logger.warning("%s: 无价格数据，跳过", symbol)
                continue

            if 'date' in price_df.columns:
                trade_dates = pd.to_datetime(price_df['date'])
            else:
                trade_dates = pd.to_datetime(price_df.index)

            daily_features = generator.build_daily_dataframe(symbol, trade_dates)
            if daily_features.empty:
                logger.warning("%s: 日频基本面为空", symbol)
                continue

            target_symbol = daily_features['symbol'].iloc[0] if 'symbol' in daily_features.columns else symbol
            symbol_variants = {symbol, target_symbol}
            generator.data_manager.delete_daily_expanded(list(symbol_variants))
            rows = generator.data_manager.save_daily_expanded(target_symbol, daily_features)
            success += 1
            total_rows += rows
            logger.info("%s: 写入 %d 条日频化记录", symbol, rows)
        except Exception as exc:
            logger.error("%s: 日频化失败 - %s", symbol, exc)

    logger.info("完成：成功处理 %d 支股票，写入 %d 条记录", success, total_rows)


if __name__ == "__main__":
    main()
