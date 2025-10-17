# -*- coding: utf-8 -*-
"""标签构建相关工具。"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_labels_corrected(
    features_df: pd.DataFrame,
    price_data: Optional[pd.DataFrame] = None,
    prediction_period: int = 30,
    threshold: float = 0.05,
    price_data_raw: Optional[pd.DataFrame] = None,
    classification_strategy: str = 'absolute',
    quantile: float = 0.7,
    min_samples_per_date: int = 30,
    negative_quantile: Optional[float] = 0.3,
    enable_neutral_band: bool = False,
    neutral_quantile: Optional[float] = 0.5,
    market_returns: Optional[pd.DataFrame] = None,
    use_market_baseline: bool = False,
    market_column: str = 'MKT',
    use_industry_neutral: bool = False,
    industry_column: str = 'industry',
    stock_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """使用前复权价格构建标签，并提供原始价格对照。

    Parameters
    ----------
    features_df : DataFrame
        特征数据（需包含 symbol, date 列）。
    price_data : DataFrame
        前复权价格数据，至少包含 symbol, date, close。
        若为空将回退至 ``stock_data``（兼容旧参数）。
    prediction_period : int
        预测周期（天）。
    threshold : float
        绝对涨幅阈值，作为量化策略的兜底或 absolute 策略阈值。
    price_data_raw : DataFrame, optional
        不复权价格数据，用于额外诊断。
    classification_strategy : {'absolute', 'quantile'}
        分类标签策略，absolute 表示使用固定阈值，quantile 表示按日期截面分位数。
    quantile : float
        当策略为 quantile 时的上分位数（如 0.7 表示取当日排行前 30% 为正类）。
    min_samples_per_date : int
        使用 quantile 策略时当日最小样本数，小于该阈值回退至 absolute 策略。
    negative_quantile : float, optional
        quantile 策略下的下分位数，用于明确负类或中性区。
    enable_neutral_band : bool
        是否启用中性区间，启用时会移除位于上/下分位数之间的样本。
    neutral_quantile : float, optional
        中性区间的上界分位（例如 0.5 表示保留低于中位数的负类）。
    market_returns : DataFrame, optional
        市场基准收益数据，需包含 date 与市场收益列（默认 MKT）。
    use_market_baseline : bool
        是否使用市场基准构建超额收益。
    market_column : str
        market_returns 中的市场收益列名。
    use_industry_neutral : bool
        是否按行业截面去均值，依赖 features_df 中的行业列。
    industry_column : str
        行业列列名（默认 industry）。
    stock_data : DataFrame, optional
        兼容旧接口，等价于 ``price_data``。
    """
    if price_data is None and stock_data is not None:
        price_data = stock_data

    if price_data is None:
        raise ValueError("price_data 缺失，请提供前复权价格数据或使用 stock_data 参数")

    features_df = features_df.copy()
    price_data = price_data.copy()
    price_data_raw = price_data_raw.copy() if price_data_raw is not None else None
    market_returns = market_returns.copy() if market_returns is not None else None

    required_cols = {'symbol', 'date', 'close'}
    missing_cols = required_cols - set(price_data.columns)
    if missing_cols:
        raise ValueError(f"price_data 缺少必要列: {missing_cols}")

    features_df['date'] = pd.to_datetime(features_df['date'])
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data['close'] = pd.to_numeric(price_data['close'], errors='coerce')

    if price_data_raw is not None:
        price_data_raw['date'] = pd.to_datetime(price_data_raw['date'])
        price_data_raw['close'] = pd.to_numeric(price_data_raw['close'], errors='coerce')

    strategy = classification_strategy.lower().strip()
    if strategy not in {'absolute', 'quantile'}:
        raise ValueError(f"classification_strategy 仅支持 'absolute' 或 'quantile'，当前为 {classification_strategy}")

    logger.info(
        "开始计算标签 (prediction_period=%d, strategy=%s, threshold=%.3f, quantile=%.2f)",
        prediction_period,
        strategy,
        threshold,
        quantile
    )

    symbol_results: List[pd.DataFrame] = []
    stats: Dict[str, float] = {
        'total_rows': 0,
        'valid_rows': 0,
        'missing_future': 0,
        'extreme_returns': 0,
        'extreme_raw': 0,
        'quantile_fallback_rows': 0,
        'neutral_dropped': 0,
        'market_baseline_rows': 0,
        'industry_residual_rows': 0
    }

    if not 0 < quantile < 1:
        raise ValueError(f"quantile 必须在 (0, 1) 范围内，当前为 {quantile}")

    if negative_quantile is not None and not 0 < negative_quantile < 1:
        logger.warning("negative_quantile 应位于 (0, 1)，当前值无效，将忽略负分位数设置")
        negative_quantile = None

    if neutral_quantile is not None and not 0 < neutral_quantile < 1:
        logger.warning("neutral_quantile 应位于 (0, 1)，当前值无效，将忽略中性区间")
        neutral_quantile = None
 
    def _prepare_market_baseline(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """标准化市场收益数据并计算未来预测期收益"""
        if df is None or len(df) == 0:
            return None

        market_df = df.copy()

        # 将列名统一为字符串，方便匹配
        market_df.columns = [str(col) for col in market_df.columns]

        if market_column not in market_df.columns:
            lower_map = {col.lower(): col for col in market_df.columns}
            if market_column.lower() in lower_map:
                market_df.rename(columns={lower_map[market_column.lower()]: market_column}, inplace=True)
            elif len(market_df.columns) == 1:
                market_df.rename(columns={market_df.columns[0]: market_column}, inplace=True)
            else:
                logger.warning("未在 market_returns 中找到列 %s，忽略市场基准", market_column)
                return None

        if 'date' not in market_df.columns:
            if market_df.index.name == 'date' or isinstance(market_df.index, pd.DatetimeIndex):
                market_df = market_df.reset_index()
            elif 'index' in market_df.columns:
                market_df.rename(columns={'index': 'date'}, inplace=True)
            else:
                logger.warning("market_returns 缺少 date 信息，忽略市场基准")
                return None

        market_df = market_df[['date', market_column]].copy()
        market_df['date'] = pd.to_datetime(market_df['date'])
        market_df.sort_values('date', inplace=True)
        market_df[market_column] = pd.to_numeric(market_df[market_column], errors='coerce')

        # 计算未来预测期的累计收益
        market_df['__cumprod__'] = (1.0 + market_df[market_column].fillna(0.0)).cumprod()
        market_df['__future_cumprod__'] = market_df['__cumprod__'].shift(-prediction_period)

        with np.errstate(divide='ignore', invalid='ignore'):
            market_df['market_future_return'] = market_df['__future_cumprod__'] / market_df['__cumprod__'] - 1.0

        market_df.drop(columns=['__cumprod__', '__future_cumprod__'], inplace=True)

        return market_df

    symbols = features_df['symbol'].unique()

    for idx, symbol in enumerate(symbols, 1):
        if idx % 50 == 0:
            logger.info("  进度: %d/%d", idx, len(symbols))

        symbol_features = features_df[features_df['symbol'] == symbol].copy()
        symbol_features.sort_values('date', inplace=True)

        price_symbol = price_data[price_data['symbol'] == symbol].copy()
        price_symbol.sort_values('date', inplace=True)
        price_symbol = price_symbol.dropna(subset=['close'])
        price_symbol = price_symbol.drop_duplicates(subset=['date'], keep='last')

        if price_symbol.empty:
            logger.warning("  %s: 价格数据缺失，跳过标签构建", symbol)
            continue

        price_symbol['future_close'] = price_symbol['close'].shift(-prediction_period)
        price_symbol['future_return'] = (
            price_symbol['future_close'] - price_symbol['close']
        ) / price_symbol['close']
        price_symbol['future_return'].replace([np.inf, -np.inf], np.nan, inplace=True)

        merge_cols = ['date', 'future_return']

        if price_data_raw is not None:
            raw_symbol = price_data_raw[price_data_raw['symbol'] == symbol].copy()
            raw_symbol.sort_values('date', inplace=True)
            raw_symbol = raw_symbol.dropna(subset=['close'])
            raw_symbol = raw_symbol.drop_duplicates(subset=['date'], keep='last')

            if not raw_symbol.empty:
                raw_symbol['future_close_raw'] = raw_symbol['close'].shift(-prediction_period)
                raw_symbol['future_return_raw'] = (
                    raw_symbol['future_close_raw'] - raw_symbol['close']
                ) / raw_symbol['close']
                raw_symbol['future_return_raw'].replace([np.inf, -np.inf], np.nan, inplace=True)
                price_symbol = price_symbol.merge(
                    raw_symbol[['date', 'future_return_raw']],
                    on='date',
                    how='left'
                )
                merge_cols.append('future_return_raw')

        symbol_labeled = symbol_features.merge(
            price_symbol[['date'] + merge_cols[1:]],
            on='date',
            how='left'
        )

        stats['total_rows'] += len(symbol_labeled)
        missing_mask = symbol_labeled['future_return'].isna()
        stats['missing_future'] += int(missing_mask.sum())

        if 'future_return_raw' in symbol_labeled.columns:
            stats['extreme_raw'] += int((symbol_labeled['future_return_raw'].abs() > 0.3).sum())

        stats['extreme_returns'] += int((symbol_labeled['future_return'].abs() > 0.3).sum())

        symbol_results.append(symbol_labeled)

    if not symbol_results:
        raise ValueError("没有成功构建标签的股票，所有 symbol 被跳过")

    result = pd.concat(symbol_results, ignore_index=True)

    target_series = result['future_return'].copy()

    if use_market_baseline and market_returns is not None:
        market_baseline = _prepare_market_baseline(market_returns)
        if market_baseline is None:
            logger.warning("市场基准不可用，回退为绝对收益标签")
        else:
            result = result.merge(market_baseline, on='date', how='left')
            stats['market_baseline_rows'] = int(result['market_future_return'].notna().sum())
            if stats['market_baseline_rows'] == 0:
                logger.warning("市场基准数据与股票日期区间无交集，忽略超额收益计算")
                result.drop(columns=['market_future_return'], inplace=True, errors='ignore')
            else:
                result['future_excess_return'] = result['future_return'] - result['market_future_return']
                target_series = result['future_excess_return']

    if use_industry_neutral:
        if industry_column not in result.columns:
            logger.warning("未找到行业列 %s，无法执行行业中性处理", industry_column)
        else:
            base_col = 'future_excess_return' if 'future_excess_return' in result.columns else 'future_return'
            base_series = result[base_col]
            industry_series = result[industry_column].fillna('Unknown').astype(str)
            result['__industry__'] = industry_series
            grouped = result.groupby(['date', '__industry__'])[base_col]
            group_counts = grouped.transform('count')

            # 需要至少两个同日同行业样本才有意义进行残差计算
            min_required = max(min(min_samples_per_date, 3), 2)
            sufficient_mask = group_counts >= min_required

            if sufficient_mask.any():
                industry_mean = grouped.transform('mean')
                residual_series = base_series - industry_mean
                result['future_residual_return'] = residual_series
                if (~sufficient_mask).any():
                    result.loc[~sufficient_mask, 'future_residual_return'] = np.nan
                residual_rows = int(sufficient_mask.sum())
                coverage = residual_rows / len(result) if len(result) else 0.0
                logger.info("  行业中性覆盖率: %.2f%% (%d/%d)", coverage * 100, residual_rows, len(result))
                if coverage < 0.35:
                    logger.warning("  行业中性覆盖率过低(<35%%)，回退使用原始收益标签")
                    stats['industry_residual_rows'] = 0
                    result['future_residual_return'] = np.nan
                    target_series = base_series
                else:
                    stats['industry_residual_rows'] = residual_rows
                    target_series = result['future_residual_return'].where(sufficient_mask, base_series)
            else:
                logger.debug("行业中性跳过: 日期/行业样本不足，使用原始收益标签")
                result['future_residual_return'] = np.nan

    result['label_reg'] = target_series

    if strategy == 'absolute':
        result['label_cls'] = (result['label_reg'] > threshold).astype(int)
    else:
        grouped = result.groupby('date')['label_reg']

        def _quantile_or_nan(series: pd.Series, q: float) -> float:
            valid = series.dropna()
            if len(valid) < max(min_samples_per_date, 1):
                return np.nan
            return float(np.nanquantile(valid, q))

        per_date_high = grouped.transform(lambda s: _quantile_or_nan(s, quantile))

        fallback_mask = per_date_high.isna()
        stats['quantile_fallback_rows'] = int(fallback_mask.sum())

        applied_high = per_date_high.fillna(threshold)
        labels = (result['label_reg'] >= applied_high).astype(int)

        per_date_low = None
        if negative_quantile is not None:
            per_date_low = grouped.transform(lambda s: _quantile_or_nan(s, negative_quantile))

        neutral_upper = None
        if enable_neutral_band and neutral_quantile is not None:
            neutral_upper = grouped.transform(lambda s: _quantile_or_nan(s, neutral_quantile))

        if per_date_low is not None:
            low_mask = (per_date_low.notna()) & (result['label_reg'] <= per_date_low)
            labels = np.where(low_mask, 0, labels)

        result['label_cls'] = labels.astype(int)

        if enable_neutral_band and per_date_low is not None:
            neutral_mask = (result['label_cls'] == 0)
            neutral_mask &= per_date_low.notna()
            if neutral_upper is not None:
                neutral_mask &= neutral_upper.notna()
                neutral_mask &= result['label_reg'] > per_date_low
                neutral_mask &= result['label_reg'] < neutral_upper
            else:
                neutral_mask &= result['label_reg'] > per_date_low
                neutral_mask &= result['label_reg'] < applied_high

            if neutral_mask.any():
                stats['neutral_dropped'] = int(neutral_mask.sum())
                result = result.loc[~neutral_mask].copy()
                per_date_high = per_date_high[~neutral_mask]
                if per_date_low is not None:
                    per_date_low = per_date_low[~neutral_mask]
                if neutral_upper is not None:
                    neutral_upper = neutral_upper[~neutral_mask]
                result.reset_index(drop=True, inplace=True)

        valid_thresholds = per_date_high[~per_date_high.isna()]
        if not valid_thresholds.empty:
            logger.info(
                "  quantile 阈值统计: 均值 %.4f, 中位数 %.4f, 最小 %.4f, 最大 %.4f",
                float(valid_thresholds.mean()),
                float(valid_thresholds.median()),
                float(valid_thresholds.min()),
                float(valid_thresholds.max())
            )
        if stats['quantile_fallback_rows'] > 0:
            logger.info(
                "  %d 条记录因当日样本不足回退为 absolute 阈值",
                stats['quantile_fallback_rows']
            )

    result['label_cls'] = result['label_cls'].astype(int)

    valid_mask = result['label_reg'].notna()
    stats['valid_rows'] = int(valid_mask.sum())

    total_rows = stats['total_rows'] or 1
    logger.info("标签计算完成:")
    logger.info(
        "  有效样本: %d/%d (%.1f%%)",
        stats['valid_rows'],
        stats['total_rows'],
        stats['valid_rows'] / total_rows * 100
    )
    pos_rate = (
        result.loc[valid_mask, 'label_cls'].mean() * 100
        if stats['valid_rows'] else 0.0
    )
    logger.info("  正样本率: %.2f%%", pos_rate)
    logger.info(
        "  平均收益: %.4f",
        result.loc[valid_mask, 'label_reg'].mean() if stats['valid_rows'] else 0.0
    )
    logger.info(
        "  收益标准差: %.4f",
        result.loc[valid_mask, 'label_reg'].std() if stats['valid_rows'] else 0.0
    )
    logger.info(
        "  缺失未来价格: %d (%.1f%%)",
        stats['missing_future'],
        stats['missing_future'] / total_rows * 100
    )
    logger.info("  前复权极端收益(|r|>0.3): %d", stats['extreme_returns'])
    if stats['market_baseline_rows'] > 0:
        logger.info("  市场基准匹配: %d 行", stats['market_baseline_rows'])
    if stats['neutral_dropped'] > 0:
        logger.info("  中性区移除: %d 行", stats['neutral_dropped'])
    if price_data_raw is not None:
        logger.info("  原始价格极端收益(|r|>0.3): %d", stats['extreme_raw'])

    result = result.loc[valid_mask].copy()

    # � 方案C2修改: 保留future_residual_return供后续使用，但不作为训练特征
    # 先备份future_residual_return（如果存在）
    residual_return_backup = None
    if 'future_residual_return' in result.columns:
        residual_return_backup = result['future_residual_return'].copy()

    # �🔒 移除所有可能导致数据泄漏的未来收益列
    # 这些列仅用于标签计算，不应作为特征使用
    leakage_cols = [
        'future_return',           # 未来绝对收益 - 直接泄漏！
        'future_excess_return',    # 未来超额收益 - 直接泄漏！
        'future_residual_return',  # 未来残差收益 - 直接泄漏！
        'future_return_raw',       # 原始价格未来收益
        'market_future_return',    # 市场未来收益
        '__industry__'             # 临时行业列
    ]
    
    cols_to_drop = [col for col in leakage_cols if col in result.columns]
    if cols_to_drop:
        logger.info(f"🔒 移除泄漏特征列: {cols_to_drop}")
        result.drop(columns=cols_to_drop, inplace=True)

    # 🔴 方案C2修改: 恢复future_residual_return（但标记为非特征列）
    # 这个列将在train_c2_solution.py中用于替换label_reg
    if residual_return_backup is not None:
        result['future_residual_return'] = residual_return_backup
        logger.info("✅ 保留 future_residual_return 列供回归标签使用（非训练特征）")

    return result
