"""根据 hsjday 目录下的通达信日线 .day 文件，将数据导入 MySQL 数据库。

用法：

python -m src.apps.scripts.fill_tdx_data \
    --root /path/to/hsjday \
    --db mysql \
    --batch 500

root 默认读取项目根目录下 hsjday。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# 动态导入项目包
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.providers.tdx_reader import read_day_file  # noqa: E402
from src.data.db.db_mysql import MySQLDatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def scan_day_files(root: Path) -> List[Path]:
    """递归扫描 root 下所有 .day 文件"""
    files = list(root.rglob("*.day"))
    logger.info("发现 %d 个 .day 文件", len(files))
    return files

def infer_symbol_from_path(path: Path) -> str:
    """根据文件名推断 symbol，如 sh000001.day -> 000001.SH"""
    name = path.stem  # 如 sh000001 或 sz162601
    # 通达信代码通常为前缀两位 + 6 位数字，共 8 位；有些指数代码可能为 7 位（前缀+5位数字）。
    if len(name) < 8:
        return ""
    market_prefix = name[:2].lower()
    code = name[2:]

    if market_prefix == "sh":
        return f"{code}.SH"
    elif market_prefix == "sz":
        return f"{code}.SZ"
    else:
        return ""

def main():
    parser = argparse.ArgumentParser(description="导入 TDX .day 数据到 MySQL")
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "hsjday"), help="hsjday 根目录")
    parser.add_argument("--batch", type=int, default=500, help="每批次写入记录数量")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个文件，调试用")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        logger.error("hsjday 目录不存在: %s", root_path)
        sys.exit(1)

    files = scan_day_files(root_path)
    if args.limit:
        files = files[: args.limit]

    db = MySQLDatabaseManager()

    batch_records: List[dict] = []
    total_written = 0

    for idx, file_path in enumerate(files, 1):
        symbol = infer_symbol_from_path(file_path)
        if not symbol:
            logger.warning("无法解析 symbol，跳过: %s", file_path)
            continue

        df = read_day_file(str(file_path), symbol)
        if df.empty:
            continue
        # 追加到 batch
        batch_records.extend(df.to_dict("records"))

        # 若超过 batch size，则写入数据库
        if len(batch_records) >= args.batch:
            df_batch = pd.DataFrame(batch_records)
            written = db.upsert_prices_daily(df_batch, source="tdx")
            total_written += written
            batch_records = []
            logger.info("已处理 %d/%d 个文件，总写入 %d 行", idx, len(files), total_written)

    # 处理剩余
    if batch_records:
        df_batch = pd.DataFrame(batch_records)
        written = db.upsert_prices_daily(df_batch, source="tdx")
        total_written += written

    logger.info("导入完成，共写入 %d 行", total_written)

if __name__ == "__main__":
    main()