#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票代码格式统一脚本
解决stocks表和prices_daily表中股票代码格式不一致的问题
"""

import sqlite3
import re
from datetime import datetime

def connect_db():
    """连接数据库"""
    return sqlite3.connect('stock_data.sqlite3')

def analyze_code_formats():
    """分析当前的代码格式问题"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("=== 股票代码格式分析 ===")
    
    # 分析stocks表中的代码格式
    print("\n1. stocks表代码格式分析:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SH' THEN 'SH后缀格式'
                WHEN symbol LIKE '%.SZ' THEN 'SZ后缀格式'
                WHEN symbol LIKE '%.BJ' THEN 'BJ后缀格式（已移除）'
                WHEN LENGTH(symbol) = 6 AND symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]' THEN '纯数字格式'
                WHEN symbol IS NULL OR symbol = '' THEN '空值'
                ELSE '其他格式'
            END as format_type,
            COUNT(*) as count,
            GROUP_CONCAT(symbol, ', ') as samples
        FROM stocks 
        GROUP BY format_type
        ORDER BY count DESC
    """)
    
    stocks_formats = cursor.fetchall()
    for format_type, count, samples in stocks_formats:
        sample_list = samples.split(', ') if samples else []
        display_samples = ', '.join(sample_list[:5])
        if len(sample_list) > 5:
            display_samples += f" (共{len(sample_list)}个)"
        print(f"  {format_type}: {count}个 - 样本: {display_samples}")
    
    # 分析prices_daily表中的代码格式
    print("\n2. prices_daily表代码格式分析:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SH' THEN 'SH后缀格式'
                WHEN symbol LIKE '%.SZ' THEN 'SZ后缀格式'
                WHEN symbol LIKE '%.BJ' THEN 'BJ后缀格式（已移除）'
                WHEN LENGTH(symbol) = 6 AND symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]' THEN '纯数字格式'
                WHEN symbol IS NULL OR symbol = '' THEN '空值'
                ELSE '其他格式'
            END as format_type,
            COUNT(DISTINCT symbol) as unique_count,
            COUNT(*) as total_records
        FROM prices_daily 
        GROUP BY format_type
        ORDER BY unique_count DESC
    """)
    
    prices_formats = cursor.fetchall()
    for format_type, unique_count, total_records in prices_formats:
        print(f"  {format_type}: {unique_count}个股票, {total_records}条记录")
    
    conn.close()
    return stocks_formats, prices_formats

def normalize_symbol(raw: str) -> str:
    """归一化股票代码为 6位数字 + .SH/.SZ/.BJ。
    规则：
    - 支持后缀：.SH/.SZ/.BJ、.XSHG/.XSHE（映射到.SH/.SZ）
    - 支持前缀：SH/SZ/BJ、SSE:/SZSE:/XSHG:/XSHE:，以及 "SH.", "SZ.", "BJ." 形式
    - 支持尾缀：600000SH / 000001SZ / 430001BJ
    - 纯6位数字：根据号段推断
    - 去除空格、统一大写
    号段：
    - 沪市：60xxxx、688xxx、689xxx、900xxx（B股） -> .SH
    - 深市：000xxx、001xxx、002xxx、003xxx、300xxx、301xxx、200xxx（B股） -> .SZ
    - 北交所：43xxxx、83xxxx、87xxxx -> .BJ（已移除，不再支持BJ股票，返回None）
    - 明确排除：以88开头的板块/指数（如880/881同花顺行业与概念）不作为个股，返回None
    返回标准化代码或None（无法识别）
    """
    if raw is None:
        return None
    s = str(raw).strip().upper().replace(" ", "")
    if s in ("", "NONE", "NULL", "NAN"):
        return None

    # 标准化已知后缀
    s = s.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")

    # 去除交易所/前缀
    for pref in ("SSE:", "SZSE:", "XSHG:", "XSHE:"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    for pref in ("SH:", "SZ:", "BJ:", "SH.", "SZ.", "BJ."):
        if s.startswith(pref):
            s = s[len(pref):]
            break

    def _validate_pair(num: str, suf: str) -> str | None:
        # 排除同花顺板块/指数 88xxxx
        if num.startswith("88"):
            return None
        if suf == 'BJ':
            return f"{num}.BJ" if num.startswith(("43", "83", "87")) else None
        if suf == 'SH':
            return f"{num}.SH" if (num.startswith("60") or num.startswith("688") or num.startswith("689") or num.startswith("900")) else None
        if suf == 'SZ':
            return f"{num}.SZ" if num.startswith(("000", "001", "002", "003", "300", "301", "200")) else None
        return None

    # 形如 SH000001 / SZ000001 / BJ430001（BJ已移除，返回None）
    m = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", s)
    if m:
        if m.group(1) == 'BJ':
            return None  # BJ股票已移除
        return _validate_pair(m.group(2), m.group(1))

    # 形如 000001SH / 000001SZ / 430001BJ（BJ已移除，返回None）
    m = re.fullmatch(r"(\d{6})(SH|SZ|BJ)", s)
    if m:
        if m.group(2) == 'BJ':
            return None  # BJ股票已移除
        return _validate_pair(m.group(1), m.group(2))

    # 形如 000001.SH / 000001.SZ / 430001.BJ（BJ已移除，返回None）
    m = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", s)
    if m:
        if m.group(2) == 'BJ':
            return None  # BJ股票已移除
        return _validate_pair(m.group(1), m.group(2))

    # 纯6位数字：根据号段推断（BJ已移除）
    if re.fullmatch(r"\d{6}", s):
        d = s
        # 排除 88xxxx 板块/指数
        if d.startswith("88"):
            return None
        # 北交所代码（43, 83, 87开头）已移除，返回None
        if d.startswith(("43", "83", "87")):
            return None
        if d.startswith("60") or d.startswith("688") or d.startswith("689") or d.startswith("900"):
            return f"{d}.SH"
        if d.startswith(("000", "001", "002", "003", "300", "301", "200")):
            return f"{d}.SZ"
        return None

    # 形如 000001.XSHG / 000001.XSHE
    if re.fullmatch(r"\d{6}\.(XSHG|XSHE)", s):
        return s.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")

    return None

def standardize_stock_codes_v2(dry_run: bool = False):
    """统一股票代码格式（增强版）。
    - dry_run=True：仅统计/预览变更与冲突，不写库
    - dry_run=False：执行备份、更新与去重
    不修改原standardize_stock_codes实现，方便回滚。
    """
    conn = connect_db()
    try:
        conn.execute("PRAGMA busy_timeout = 8000")
    except Exception:
        pass
    cursor = conn.cursor()

    print("\n=== 开始统一股票代码格式（v2）===")
    if dry_run:
        print("[干跑预览] 本次运行不会对数据库进行任何写入操作")

    # 备份
    if not dry_run:
        print("\n正在创建备份...")
        try:
            cursor.execute("DROP TABLE IF EXISTS stocks_backup")
            cursor.execute("CREATE TABLE stocks_backup AS SELECT * FROM stocks")
            cursor.execute("DROP TABLE IF EXISTS prices_daily_backup")
            cursor.execute("CREATE TABLE prices_daily_backup AS SELECT * FROM prices_daily")
            print("备份完成")
        except Exception as e:
            print(f"备份失败: {e}")
            print("继续执行代码标准化...")

    # 1) 标准化 stocks
    print("\n1. 标准化stocks表代码格式...")
    cursor.execute("SELECT rowid, symbol FROM stocks")
    rows = cursor.fetchall()
    upd, dels = [], []
    for rid, sym in rows:
        ns = normalize_symbol(sym)
        if ns is None:
            dels.append(rid)
        elif ns != sym:
            upd.append((ns, rid))

    if dry_run:
        print(f"  将删除异常记录: {len(dels)} 条")
        print(f"  将更新为规范格式: {len(upd)} 条（示例最多10条）")
        for i, (ns, rid) in enumerate(upd[:10]):
            print(f"    示例[{i+1}]: rowid={rid} -> {ns}")
    else:
        if dels:
            batch = 500
            for i in range(0, len(dels), batch):
                part = dels[i:i+batch]
                cursor.execute(
                    f"DELETE FROM stocks WHERE rowid IN ({','.join('?'*len(part))})",
                    part,
                )
                if not dry_run:
                    conn.commit()
            print(f"  删除异常记录: {len(dels)} 条")
        if upd:
            batch = 500
            for i in range(0, len(upd), batch):
                part = upd[i:i+batch]
                cursor.executemany("UPDATE stocks SET symbol=? WHERE rowid=?", part)
                if not dry_run:
                    conn.commit()
            print(f"  更新为规范格式: {len(upd)} 条")

    # 2) 标准化 prices_daily
    print("\n2. 标准化prices_daily表代码格式...")
    cursor.execute("SELECT DISTINCT symbol FROM prices_daily")
    pd_syms = [r[0] for r in cursor.fetchall()]
    mapping, invalid = {}, set()
    for old in pd_syms:
        ns = normalize_symbol(old)
        if ns is None:
            invalid.add(old)
        elif ns != old:
            mapping[old] = ns

    if dry_run:
        print(f"  将删除异常symbol: {len(invalid)} 个")
        print(f"  将更新为规范格式: {len(mapping)} 个")
        # 冲突组（多个旧码 -> 同一新码）
        inv = {}
        for o, n in mapping.items():
            inv.setdefault(n, []).append(o)
        collisions = [(n, olds) for n, olds in inv.items() if len(olds) > 1]
        collisions.sort(key=lambda x: len(x[1]), reverse=True)
        if collisions:
            print(f"  归一化后潜在去重冲突: {len(collisions)} 组（示例最多5组）")
            for i, (n, olds) in enumerate(collisions[:5]):
                more = '...' if len(olds) > 6 else ''
                print(f"    {i+1}) {n} <- {olds[:6]}{more}")
    else:
        if invalid:
            batch = 500
            inv_list = list(invalid)
            for i in range(0, len(inv_list), batch):
                part = inv_list[i:i+batch]
                cursor.execute(
                    f"DELETE FROM prices_daily WHERE symbol IN ({','.join('?'*len(part))})",
                    part,
                )
                conn.commit()
            print(f"  删除异常symbol记录: {len(invalid)} 个")
        if mapping:
            cursor.execute("DROP INDEX IF EXISTS idx_prices_daily_symbol_date")
            updated_cnt = 0
            mapping_items = list(mapping.items())
            for j in range(0, len(mapping_items), 100):
                batch = mapping_items[j:j+100]
                for old, new in batch:
                    cursor.execute("UPDATE prices_daily SET symbol=? WHERE symbol=?", (new, old))
                    updated_cnt += cursor.rowcount
                conn.commit()
            print(f"  更新prices_daily代码: {updated_cnt} 条")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date ON prices_daily(symbol, date)")
            conn.commit()

    print("\n=== 标准化完成 ===")
    conn.close()

def verify_standardization():
    """验证代码格式统一效果"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("\n=== 验证代码格式统一效果 ===")
    
    # 检查统一后的格式分布（BJ已移除）
    print("\n1. stocks表格式分布:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SH' THEN 'SH后缀格式'
                WHEN symbol LIKE '%.SZ' THEN 'SZ后缀格式'
                WHEN symbol LIKE '%.BJ' THEN 'BJ后缀格式（已移除）'
                WHEN symbol IS NULL THEN '空值'
                ELSE '其他格式'
            END as format_type,
            COUNT(*) as count
        FROM stocks 
        GROUP BY format_type
        ORDER BY count DESC
    """)
    
    for format_type, count in cursor.fetchall():
        print(f"  {format_type}: {count}个")
    
    print("\n2. prices_daily表格式分布:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SH' THEN 'SH后缀格式'
                WHEN symbol LIKE '%.SZ' THEN 'SZ后缀格式'
                WHEN symbol LIKE '%.BJ' THEN 'BJ后缀格式（已移除）'
                WHEN symbol IS NULL THEN '空值'
                ELSE '其他格式'
            END as format_type,
            COUNT(DISTINCT symbol) as unique_count,
            COUNT(*) as total_records
        FROM prices_daily 
        GROUP BY format_type
        ORDER BY unique_count DESC
    """)
    
    for format_type, unique_count, total_records in cursor.fetchall():
        print(f"  {format_type}: {unique_count}个股票, {total_records}条记录")
    
    # 检查匹配率
    print("\n3. 数据匹配率分析:")
    
    # 总股票数
    cursor.execute("SELECT COUNT(*) FROM stocks WHERE symbol IS NOT NULL")
    total_stocks = cursor.fetchone()[0]
    
    # 有价格数据的股票数
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices_daily")
    stocks_with_prices = cursor.fetchone()[0]
    
    # 匹配的股票数
    cursor.execute("""
        SELECT COUNT(DISTINCT s.symbol) 
        FROM stocks s 
        INNER JOIN prices_daily p ON s.symbol = p.symbol
        WHERE s.symbol IS NOT NULL
    """)
    matched_stocks = cursor.fetchone()[0]
    
    print(f"  总股票数: {total_stocks}")
    print(f"  有价格数据的股票数: {stocks_with_prices}")
    print(f"  匹配的股票数: {matched_stocks}")
    
    if total_stocks > 0:
        match_rate = (matched_stocks / total_stocks) * 100
        print(f"  匹配率: {match_rate:.2f}%")
    
    # 检查仍然不匹配的股票样本
    print("\n4. 不匹配股票样本:")
    cursor.execute("""
        SELECT s.symbol, s.name, s.market
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        WHERE s.symbol IS NOT NULL AND p.symbol IS NULL
        LIMIT 10
    """)
    
    unmatched = cursor.fetchall()
    if unmatched:
        print("  stocks表中无价格数据的股票:")
        for symbol, name, market in unmatched:
            print(f"    {symbol} - {name} ({market})")
    else:
        print("  所有stocks表中的股票都有价格数据！")
    
    cursor.execute("""
        SELECT DISTINCT p.symbol
        FROM prices_daily p 
        LEFT JOIN stocks s ON p.symbol = s.symbol
        WHERE s.symbol IS NULL
        LIMIT 10
    """)
    
    orphaned = cursor.fetchall()
    if orphaned:
        print("  prices_daily表中的孤立股票代码:")
        for (symbol,) in orphaned:
            print(f"    {symbol}")
    else:
        print("  prices_daily表中没有孤立的股票代码！")
    
    conn.close()

def main():
    """主函数"""
    print("股票代码格式统一工具")
    print("=" * 50)
    
    # 1. 分析当前格式问题
    analyze_code_formats()
    
    # 2. 执行代码格式统一
    standardize_stock_codes_v2(dry_run=True)
    
    # 3. 验证统一效果
    verify_standardization()
    
    print("\n=== 代码格式统一完成 ===")
    print("建议：")
    print("1. 检查匹配率是否有显著提升")
    print("2. 如果仍有不匹配的股票，可能需要手动处理特殊情况")
    print("3. 考虑在应用程序中添加代码格式验证逻辑")

if __name__ == "__main__":
    main()