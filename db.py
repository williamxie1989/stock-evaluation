import sqlite3
from contextlib import contextmanager
from typing import Optional, Iterable, Dict, Any, List
import pandas as pd
from datetime import datetime

DB_PATH = "stock_data.sqlite3"

# 归一化函数（模块级延迟导入，避免循环依赖）
try:
    from standardize_stock_codes import normalize_symbol as _normalize_symbol
except Exception:
    _normalize_symbol = None

class DatabaseManager:
    """
    SQLite 数据库管理器
    - 管理表结构：stocks、prices_daily、quotes_realtime、factors、model_metrics
    - 提供增量写入与基础查询能力
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        with self.get_conn() as conn:
            cur = conn.cursor()
            
            # stocks 元数据
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,              -- 股票代码（A股代码）
                    name TEXT,                         -- 股票名称
                    market TEXT,                       -- 市场标识，如 SH/SZ/HK
                    board_type TEXT,                   -- 板块类型（主板、创业板、科创板等）
                    exchange TEXT,                     -- 交易所名称
                    ah_pair TEXT,                      -- 若有，对应另一市场代码，例如 H 股代码
                    industry TEXT,                     -- 行业
                    market_cap REAL,                   -- 总市值（元）
                    UNIQUE(symbol)
                );
                """
            )

            # 日线行情
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prices_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,                -- YYYY-MM-DD
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    source TEXT,                      -- 数据来源（akshare接口名等）
                    UNIQUE(symbol, date)
                );
                """
            )

            # 实时行情快照（可多次写入，保留最新一条）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS quotes_realtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,                  -- 时间戳 ISO 格式
                    price REAL,
                    change REAL,
                    change_pct REAL,
                    volume REAL,
                    source TEXT
                );
                """
            )

            # 因子表（可扩展）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    value REAL,
                    UNIQUE(symbol, date, factor_name)
                );
                """
            )

            # 模型与回测指标
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    train_start TEXT,
                    train_end TEXT,
                    test_start TEXT,
                    test_end TEXT,
                    win_rate REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    annual_return REAL,
                    created_at TEXT
                );
                """
            )

            # 执行迁移并提交，确保表结构已创建时再尝试添加新字段
            self._migrate_database(cur)
            conn.commit()

    def _migrate_database(self, cur):
        """数据库迁移：为现有表添加新字段"""
        try:
            # 检查stocks表是否存在board_type字段
            cur.execute("PRAGMA table_info(stocks)")
            columns = [column[1] for column in cur.fetchall()]
            
            if 'board_type' not in columns:
                cur.execute("ALTER TABLE stocks ADD COLUMN board_type TEXT")
                print("已添加board_type字段到stocks表")
            
            if 'exchange' not in columns:
                cur.execute("ALTER TABLE stocks ADD COLUMN exchange TEXT")
                print("已添加exchange字段到stocks表")

            # 新增：stocks表添加行业与市值
            if 'industry' not in columns:
                cur.execute("ALTER TABLE stocks ADD COLUMN industry TEXT")
                print("已添加industry字段到stocks表")
            if 'market_cap' not in columns:
                cur.execute("ALTER TABLE stocks ADD COLUMN market_cap REAL")
                print("已添加market_cap字段到stocks表")

            # 新增：prices_daily表添加成交额amount
            cur.execute("PRAGMA table_info(prices_daily)")
            pd_columns = [column[1] for column in cur.fetchall()]
            if 'amount' not in pd_columns:
                cur.execute("ALTER TABLE prices_daily ADD COLUMN amount REAL")
                print("已添加amount字段到prices_daily表")
                
        except Exception as e:
            print(f"数据库迁移失败: {e}")

    def upsert_stocks(self, rows: Iterable[Dict[str, Any]]) -> int:
        """插入或替换股票元数据（保持原签名），增加symbol归一化与非法拦截"""
        rows = list(rows) if rows else []
        if not rows:
            return 0
        cleaned: List[Dict[str, Any]] = []
        skipped = 0
        for s in rows:
            sym = s.get('symbol')
            ns = None
            if _normalize_symbol and sym:
                try:
                    # normalize_symbol 目前不识别 .SS，这里先替换后再尝试
                    _sym_try = str(sym)
                    if _sym_try.upper().endswith('.SS'):
                        _sym_try = _sym_try[:-3] + '.SH'
                    ns = _normalize_symbol(_sym_try)
                except Exception:
                    ns = None
            # 如果无法归一化，尝试保持原值但阻止88开头进入，并兼容 .SS
            if ns is None:
                # 容错：若是标准格式则做快速校验
                if isinstance(sym, str) and sym.upper().endswith((".SH", ".SZ", ".BJ", ".SS")):
                    num = sym.split(".")[0]
                    suf = sym.split(".")[-1].upper()
                    if suf == 'SS':
                        suf = 'SH'
                    if num.startswith("88"):
                        skipped += 1
                        continue
                    if suf == 'BJ' and not num.startswith(("43", "83", "87")):
                        skipped += 1
                        continue
                    if suf == 'SH' and not (num.startswith("60") or num.startswith("688") or num.startswith("689") or num.startswith("900")):
                        skipped += 1
                        continue
                    if suf == 'SZ' and not num.startswith(("000", "001", "002", "003", "300", "301", "200")):
                        skipped += 1
                        continue
                    ns = f"{num}.{suf}"
                else:
                    skipped += 1
                    continue
            # 替换为规范symbol
            s2 = dict(s)
            s2['symbol'] = ns
            # 兼容新增字段：行业与市值
            s2['industry'] = s.get('industry') if 'industry' in s else None
            s2['market_cap'] = s.get('market_cap') if 'market_cap' in s else None
            cleaned.append(s2)
        if not cleaned:
            print(f"[upsert_stocks] 所有输入记录均被过滤，跳过写入。跳过 {skipped} 条。")
            return 0
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO stocks(symbol, name, market, board_type, exchange, ah_pair, industry, market_cap)
                VALUES(:symbol, :name, :market, :board_type, :exchange, :ah_pair, :industry, :market_cap)
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name,
                    market=excluded.market,
                    board_type=excluded.board_type,
                    exchange=excluded.exchange,
                    ah_pair=excluded.ah_pair,
                    industry=COALESCE(excluded.industry, stocks.industry),
                    market_cap=COALESCE(excluded.market_cap, stocks.market_cap)
                """,
                cleaned
            )
            conn.commit()
        if skipped:
            print(f"[upsert_stocks] 已写入 {len(cleaned)} 条，过滤无效 {skipped} 条。")
        # sqlite3 对 executemany 的 rowcount 可能不可用，这里回退为 len(cleaned)
        return cur.rowcount if (getattr(cur, 'rowcount', None) is not None and cur.rowcount >= 0) else len(cleaned)

    def upsert_prices_daily(self, df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "date",
                             rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """
        将 DataFrame 写入 prices_daily，按 (symbol, date) 去重。
        期望列名：symbol, date, open, high, low, close, volume
        可通过 rename_map 进行列名映射。
        返回写入行数。
        """
        if df is None or df.empty:
            return 0
        data = df.copy()
        if rename_map:
            data = data.rename(columns=rename_map)
        # 标准化列名
        if symbol_col != "symbol":
            data.rename(columns={symbol_col: "symbol"}, inplace=True)
        if date_col != "date":
            data.rename(columns={date_col: "date"}, inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                raise ValueError(f"缺少必要列: {col}")
        # 日期标准化
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        # 若无amount列，补None，便于统一写入
        if 'amount' not in data.columns:
            data['amount'] = None

        # 归一化与过滤
        rows = data[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]].to_dict("records")
        cleaned: List[Dict[str, Any]] = []
        skipped = 0
        for r in rows:
            sym = r.get('symbol')
            ns = None
            if _normalize_symbol and sym:
                try:
                    _sym_try = str(sym)
                    if _sym_try.upper().endswith('.SS'):
                        _sym_try = _sym_try[:-3] + '.SH'
                    ns = _normalize_symbol(_sym_try)
                except Exception:
                    ns = None
            if ns is None:
                if isinstance(sym, str) and sym.upper().endswith((".SH", ".SZ", ".BJ", ".SS")):
                    num = sym.split(".")[0]
                    suf = sym.split(".")[-1].upper()
                    if suf == 'SS':
                        suf = 'SH'
                    if num.startswith("88"):
                        skipped += 1
                        continue
                    if suf == 'BJ' and not num.startswith(("43", "83", "87")):
                        skipped += 1
                        continue
                    if suf == 'SH' and not (num.startswith("60") or num.startswith("688") or num.startswith("689") or num.startswith("900")):
                        skipped += 1
                        continue
                    if suf == 'SZ' and not num.startswith(("000", "001", "002", "003", "300", "301", "200")):
                        skipped += 1
                        continue
                    ns = f"{num}.{suf}"
                else:
                    skipped += 1
                    continue
            r2 = dict(r)
            r2['symbol'] = ns
            cleaned.append(r2)
        if not cleaned:
            print(f"[upsert_prices_daily] 所有输入记录均被过滤，跳过写入。跳过 {skipped} 条。")
            return 0

        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO prices_daily(symbol, date, open, high, low, close, volume, amount, source)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    amount = COALESCE(excluded.amount, prices_daily.amount),
                    source = COALESCE(excluded.source, prices_daily.source)
                """,
                [
                    (
                        r["symbol"], r["date"],
                        float(r["open"]) if pd.notna(r["open"]) else None,
                        float(r["high"]) if pd.notna(r["high"]) else None,
                        float(r["low"]) if pd.notna(r["low"]) else None,
                        float(r["close"]) if pd.notna(r["close"]) else None,
                        float(r["volume"]) if pd.notna(r["volume"]) else None,
                        float(r["amount"]) if pd.notna(r["amount"]) else None,
                        source,
                    ) for r in cleaned
                ]
            )
            conn.commit()
            if skipped:
                print(f"[upsert_prices_daily] 已写入 {len(cleaned)} 条，过滤无效 {skipped} 条。")
            # sqlite3 的 executemany 在某些场景下 rowcount 可能为 -1 或不可用，统一回退为 len(cleaned)
            return cur.rowcount if (getattr(cur, 'rowcount', None) is not None and cur.rowcount >= 0) else len(cleaned)

    def insert_quotes_realtime(self, df: pd.DataFrame, symbol_col: str = "symbol", ts: Optional[str] = None,
                               rename_map: Optional[Dict[str, str]] = None, source: str = "") -> int:
        """
        追加实时行情快照。
        期望列名：symbol, price, change, change_pct, volume
        """
        if df is None or df.empty:
            return 0
        data = df.copy()
        if rename_map:
            data = data.rename(columns=rename_map)
        if symbol_col != "symbol":
            data.rename(columns={symbol_col: "symbol"}, inplace=True)
        if ts is None:
            ts = datetime.utcnow().isoformat()
        if "change_pct" in data.columns:
            # 有些来源以百分数形式（如 1.23 表示 1.23%），统一为小数比例或保留原样由前端解释
            pass
        rows = data.to_dict("records")
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO quotes_realtime(symbol, ts, price, change, change_pct, volume, source)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [(r.get("symbol"), ts,
                  r.get("price"), r.get("change"), r.get("change_pct"), r.get("volume"), source) for r in rows]
            )
            conn.commit()
            return cur.rowcount or 0

    def get_latest_dates_by_symbol(self) -> Dict[str, str]:
        """获取每个 symbol 最新的日线日期，用于增量更新"""
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, MAX(date) as max_date
                FROM prices_daily
                GROUP BY symbol
                """
            )
            return {row[0]: row[1] for row in cur.fetchall() if row[1]}

    def get_last_n_bars(self, symbols: list[str] | None = None, n: int = 2) -> pd.DataFrame:
        """获取每个 symbol 的最近 n 根K线。
        - symbols: 可选，若提供则仅查询这些标的；否则查询全表
        - n: 返回每个标的最近的 n 条记录
        """
        with self.get_conn() as conn:
            if symbols:
                # 规范化查询用的符号后缀：将 .SS 统一映射为 .SH，以匹配prices_daily中的存储
                symbols_norm: list[str] = []
                for s in symbols:
                    try:
                        if isinstance(s, str) and s.upper().endswith('.SS'):
                            symbols_norm.append(s[:-3] + '.SH')
                        else:
                            symbols_norm.append(s)
                    except Exception:
                        symbols_norm.append(s)
                # 去重以避免IN子句重复
                symbols_norm = list(dict.fromkeys(symbols_norm))
                placeholders = ','.join(['?'] * len(symbols_norm))
                sql = f"SELECT symbol, date, open, high, low, close, volume FROM prices_daily WHERE symbol IN ({placeholders})"
                df = pd.read_sql_query(sql, conn, params=symbols_norm)
            else:
                df = pd.read_sql_query("SELECT symbol, date, open, high, low, close, volume FROM prices_daily", conn)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]) 
        df.sort_values(["symbol", "date"], inplace=True)
        return df.groupby("symbol").tail(n)

    def list_symbols(self, markets: list[str] | None = None, limit: int = None) -> list[dict]:
        """
        从 stocks 表读取候选标的列表。
        - markets: 可选，['SH','SZ','HK'] 之一；None 表示全部
        - limit: 返回的最大记录数
        返回: [{symbol, name, market, ah_pair}, ...]
        """
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                # 构建统一的过滤条件：排除行业板块/指数（如88开头）及常见指数/基金类别
                where_clauses = ["symbol NOT LIKE '88%'",
                                 "(board_type IS NULL OR board_type NOT IN ('指数','行业指数','板块','基金','ETF'))"]
                params: list[Any] = []

                if markets:
                    placeholders = ','.join(['?'] * len(markets))
                    where_clauses.insert(0, f"market IN ({placeholders})")
                    params.extend(list(markets))

                where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                order_limit_sql = " ORDER BY symbol"
                if limit:
                    order_limit_sql += " LIMIT ?"
                    params.append(limit)

                sql = f"SELECT symbol, name, market, ah_pair FROM stocks{where_sql}{order_limit_sql}"
                cur.execute(sql, params)

                rows = cur.fetchall()
                result = []
                for r in rows:
                    result.append({
                        'symbol': r[0],
                        'name': r[1],
                        'market': r[2],
                        'ah_pair': r[3]
                    })
                return result
        except Exception as e:
            print(f"list_symbols 失败: {e}")
            return []