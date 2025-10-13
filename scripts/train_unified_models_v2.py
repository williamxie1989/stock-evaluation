# -*- coding: utf-8 -*-
"""
统一模型训练入口 V2
整合新特征体系和增强训练器
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import *
from src.ml.features.unified_feature_builder import UnifiedFeatureBuilder
from src.ml.training.enhanced_trainer_v2 import EnhancedTrainerV2
from src.data.unified_data_access import UnifiedDataAccessLayer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_training_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    prediction_period: int = PREDICTION_PERIOD_DAYS
) -> pd.DataFrame:
    """
    准备训练数据
    
    采用混合价格策略：
    - 特征构建：使用不复权价格（保持技术指标准确性）
    - 标签计算：使用前复权价格（反映真实投资收益）
    
    Parameters
    ----------
    symbols : List[str]
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    prediction_period : int
        预测周期（天）
    
    Returns
    -------
    df : DataFrame
        特征+标签数据
    """
    logger.info("="*80)
    logger.info("准备训练数据（V2增强版）")
    logger.info("="*80)
    logger.info(f"股票数量: {len(symbols)}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info(f"预测周期: {prediction_period}天")
    logger.info(f"价格策略: 特征用不复权 + 标签用前复权")
    
        # 初始化数据访问层（使用数据库）
    from src.data.unified_data_access import UnifiedDataAccessLayer, DataAccessConfig
    config = DataAccessConfig()
    config.use_cache = True  # ✅ 缓存系统正常工作（L1 Redis + L2 Parquet都已验证）
    config.auto_sync = False  # ✅ 训练模式禁用外部同步,仅使用数据库数据
    data_access = UnifiedDataAccessLayer(config=config)
    logger.info("✅ 缓存已启用, 外部同步已禁用(仅使用数据库数据)")
    
    # 初始化数据库管理器
    from src.data.db.unified_database_manager import UnifiedDatabaseManager
    db_manager = UnifiedDatabaseManager()
    
    # 初始化特征构建器
    builder = UnifiedFeatureBuilder(
        data_access=data_access,
        db_manager=db_manager
    )
    
    all_data = []
    failed_symbols = []
    quality_stats = {
        'total_processed': 0,
        'data_insufficient': 0,
        'feature_build_failed': 0,
        'qfq_data_failed': 0,
        'qfq_negative_filtered': 0,
        'qfq_extreme_filtered': 0,
        'no_valid_labels': 0,
        'success': 0
    }
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{len(symbols)}] 处理 {symbol}")
            quality_stats['total_processed'] += 1
            
            # ========== 步骤1: 获取不复权数据用于特征构建 ==========
            stock_data = data_access.get_stock_data(
                symbol,
                start_date,
                end_date,
                adjust_mode='none'  # 不复权
            )
            
            # 🔧 Debug: 检查获取的数据结构
            if stock_data is not None:
                logger.info(f"  ✓ 获取数据: {len(stock_data)} rows, index.name={stock_data.index.name}, is_DatetimeIndex={isinstance(stock_data.index, pd.DatetimeIndex)}")
            
            if stock_data is None or len(stock_data) < LOOKBACK_DAYS:
                logger.warning(f"  跳过: 不复权数据不足 ({len(stock_data) if stock_data is not None else 0} < {LOOKBACK_DAYS})")
                failed_symbols.append((symbol, 'data_insufficient'))
                quality_stats['data_insufficient'] += 1
                continue
            
            # ========== 步骤2: 构建特征（基于不复权价格） ==========
            features_df = builder.build_features_from_dataframe(stock_data, symbol)
            
            if features_df is None or len(features_df) == 0:
                logger.warning(f"  跳过: 特征构建失败")
                failed_symbols.append((symbol, 'feature_build_failed'))
                quality_stats['feature_build_failed'] += 1
                continue
            
            # ========== 步骤3: 获取前复权数据用于标签计算 ==========
            stock_data_qfq = data_access.get_stock_data(
                symbol,
                start_date,
                end_date,
                adjust_mode='qfq'  # 前复权
            )
            
            # 🔧 Debug: 检查qfq数据结构
            if stock_data_qfq is not None:
                logger.info(f"  ✓ 获取qfq数据: {len(stock_data_qfq)} rows, index.name={stock_data_qfq.index.name}, is_DatetimeIndex={isinstance(stock_data_qfq.index, pd.DatetimeIndex)}")
            
            if stock_data_qfq is None or len(stock_data_qfq) == 0:
                logger.warning(f"  跳过: 前复权数据获取失败")
                failed_symbols.append((symbol, 'qfq_data_failed'))
                quality_stats['qfq_data_failed'] += 1
                continue
            
            # 🔧 关键修复：将date索引转为列（add_labels_with_qfq需要date列）
            if stock_data_qfq.index.name == 'date' or isinstance(stock_data_qfq.index, pd.DatetimeIndex):
                stock_data_qfq = stock_data_qfq.reset_index()
                if 'index' in stock_data_qfq.columns and 'date' not in stock_data_qfq.columns:
                    stock_data_qfq.rename(columns={'index': 'date'}, inplace=True)
                logger.debug(f"  ✓ qfq数据date索引已转为列")
            
            # 🔧 修复：将date索引转为列（add_labels_with_qfq需要date列）
            if stock_data_qfq.index.name == 'date' or isinstance(stock_data_qfq.index, pd.DatetimeIndex):
                stock_data_qfq = stock_data_qfq.reset_index()
                if 'index' in stock_data_qfq.columns and 'date' not in stock_data_qfq.columns:
                    stock_data_qfq.rename(columns={'index': 'date'}, inplace=True)
            
            # ========== 步骤4: 数据质量过滤 ==========
            # 检查前复权价格是否有异常值
            initial_qfq_count = len(stock_data_qfq)
            
            # 过滤负数价格（数据错误）
            qfq_negative_mask = (stock_data_qfq['close'] < 0)
            if qfq_negative_mask.sum() > 0:
                logger.warning(f"  发现 {qfq_negative_mask.sum()} 条负数前复权价格")
                quality_stats['qfq_negative_filtered'] += qfq_negative_mask.sum()
                stock_data_qfq = stock_data_qfq[~qfq_negative_mask]
            
            # 过滤极端值（价格 > 10000 或 < 0.01，可能是数据错误）
            qfq_extreme_mask = (stock_data_qfq['close'] > 10000) | (stock_data_qfq['close'] < 0.01)
            if qfq_extreme_mask.sum() > 0:
                logger.warning(f"  发现 {qfq_extreme_mask.sum()} 条极端前复权价格")
                quality_stats['qfq_extreme_filtered'] += qfq_extreme_mask.sum()
                stock_data_qfq = stock_data_qfq[~qfq_extreme_mask]
            
            # 如果过滤后数据太少，跳过
            filtered_count = initial_qfq_count - len(stock_data_qfq)
            if filtered_count > initial_qfq_count * 0.3:  # 超过30%被过滤
                logger.warning(f"  跳过: 前复权数据质量差 (过滤{filtered_count}/{initial_qfq_count})")
                failed_symbols.append((symbol, 'qfq_quality_poor'))
                continue
            
            if len(stock_data_qfq) < LOOKBACK_DAYS:
                logger.warning(f"  跳过: 前复权数据不足 (过滤后{len(stock_data_qfq)} < {LOOKBACK_DAYS})")
                failed_symbols.append((symbol, 'qfq_insufficient'))
                continue
            
            # ========== 步骤5: 添加标签（基于前复权价格） ==========
            features_df = add_labels_with_qfq(features_df, stock_data_qfq, prediction_period)
            
            # 删除缺失标签的行
            features_df = features_df.dropna(subset=['label_cls', 'label_reg'])
            
            if len(features_df) == 0:
                logger.warning(f"  跳过: 无有效标签")
                failed_symbols.append((symbol, 'no_valid_labels'))
                quality_stats['no_valid_labels'] += 1
                continue
            
            # ========== 步骤6: 数据质量最终检查 ==========
            # 过滤异常收益率（绝对值 > 100%，可能是数据错误）
            extreme_return_mask = features_df['label_reg'].abs() > 1.0
            if extreme_return_mask.sum() > 0:
                logger.warning(f"  过滤 {extreme_return_mask.sum()} 条极端收益率记录")
                features_df = features_df[~extreme_return_mask]
            
            if len(features_df) == 0:
                logger.warning(f"  跳过: 过滤后无数据")
                failed_symbols.append((symbol, 'all_filtered'))
                continue
            
            # 添加symbol列
            features_df['symbol'] = symbol
            
            all_data.append(features_df)
            quality_stats['success'] += 1
            logger.info(f"  ✓ 成功: {len(features_df)} 条记录 (正样本率: {features_df['label_cls'].mean():.1%})")
            
        except Exception as e:
            logger.error(f"  处理失败: {e}", exc_info=True)
            failed_symbols.append((symbol, 'exception'))
    
    # 合并数据
    if len(all_data) == 0:
        raise ValueError("没有可用的训练数据，所有股票处理失败")
    
    df = pd.concat(all_data, ignore_index=True)
    
    # ========== 输出数据质量报告 ==========
    logger.info("\n" + "="*80)
    logger.info("数据准备完成 - 质量报告")
    logger.info("="*80)
    logger.info(f"✅ 成功股票: {quality_stats['success']}/{quality_stats['total_processed']} ({quality_stats['success']/quality_stats['total_processed']*100:.1f}%)")
    logger.info(f"📊 总记录数: {len(df):,}")
    logger.info(f"📅 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"📈 正样本率: {df['label_cls'].mean():.2%}")
    logger.info(f"📉 平均收益: {df['label_reg'].mean():.4f}")
    logger.info(f"📊 收益标准差: {df['label_reg'].std():.4f}")
    
    logger.info("\n失败统计:")
    logger.info(f"  数据不足: {quality_stats['data_insufficient']}")
    logger.info(f"  特征构建失败: {quality_stats['feature_build_failed']}")
    logger.info(f"  前复权数据失败: {quality_stats['qfq_data_failed']}")
    logger.info(f"  无有效标签: {quality_stats['no_valid_labels']}")
    
    logger.info("\n数据清洗统计:")
    logger.info(f"  前复权负数过滤: {quality_stats['qfq_negative_filtered']} 条")
    logger.info(f"  前复权极端值过滤: {quality_stats['qfq_extreme_filtered']} 条")
    
    if failed_symbols:
        logger.info(f"\n失败股票详情 (共{len(failed_symbols)}只):")
        failure_reasons = {}
        for symbol, reason in failed_symbols:
            failure_reasons[reason] = failure_reasons.get(reason, []) + [symbol]
        for reason, symbols in failure_reasons.items():
            logger.info(f"  {reason}: {len(symbols)} 只 - {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    
    return df


def add_labels(df: pd.DataFrame, prediction_period: int) -> pd.DataFrame:
    """
    添加分类和回归标签（使用不复权价格，已弃用）
    
    注意：此函数已被 add_labels_with_qfq 替代
    保留仅用于向后兼容
    
    Parameters
    ----------
    df : DataFrame
        特征数据
    prediction_period : int
        预测周期
    
    Returns
    -------
    df : DataFrame
        添加标签后的数据
    """
    logger.warning("使用了已弃用的 add_labels 函数，请改用 add_labels_with_qfq")
    
    # 计算未来收益
    df['future_return'] = df['close'].shift(-prediction_period) / df['close'] - 1
    
    # 分类标签: 收益 > CLS_THRESHOLD
    df['label_cls'] = (df['future_return'] > CLS_THRESHOLD).astype(int)
    
    # 回归标签: 收益率
    df['label_reg'] = df['future_return']
    
    return df


def add_labels_with_qfq(
    features_df: pd.DataFrame,
    stock_data_qfq: pd.DataFrame,
    prediction_period: int
) -> pd.DataFrame:
    """
    添加分类和回归标签（使用前复权价格）
    
    将前复权价格数据与特征数据按日期对齐，计算真实投资收益率标签
    
    Parameters
    ----------
    features_df : DataFrame
        特征数据（基于不复权价格构建）
    stock_data_qfq : DataFrame
        前复权价格数据
    prediction_period : int
        预测周期（天）
    
    Returns
    -------
    df : DataFrame
        添加标签后的数据
    """
    # 确保两个DataFrame都有date列且为datetime类型
    if 'date' not in features_df.columns:
        raise ValueError("features_df 缺少 'date' 列")
    if 'date' not in stock_data_qfq.columns:
        raise ValueError("stock_data_qfq 缺少 'date' 列")
    
    features_df = features_df.copy()
    stock_data_qfq = stock_data_qfq.copy()
    
    # 确保date列为datetime类型
    features_df['date'] = pd.to_datetime(features_df['date'])
    stock_data_qfq['date'] = pd.to_datetime(stock_data_qfq['date'])
    
    # 按日期排序并重置索引
    features_df = features_df.sort_values('date').reset_index(drop=True)
    stock_data_qfq = stock_data_qfq.sort_values('date').reset_index(drop=True)
    
    # 方法1: 简单shift方法（如果数据完全对齐）
    # 先尝试通过日期merge对齐
    # 注意: stock_data_qfq可能包含close_qfq列(从UnifiedDataAccessLayer)或只有close列(测试mock数据)
    close_col = 'close_qfq' if 'close_qfq' in stock_data_qfq.columns else 'close'
    
    qfq_for_merge = stock_data_qfq[['date', close_col]].copy()
    if close_col == 'close':
        qfq_for_merge.rename(columns={'close': 'close_qfq'}, inplace=True)
    
    # 删除features_df中可能存在的复权列，避免merge冲突
    cols_to_drop = [c for c in features_df.columns if '_qfq' in c or '_hfq' in c]
    if cols_to_drop:
        features_df = features_df.drop(columns=cols_to_drop)
        logger.debug(f"  从features_df删除复权列: {cols_to_drop}")
    
    merged = features_df.merge(qfq_for_merge, on='date', how='left')
    
    # 计算未来收益率
    merged['future_close_qfq'] = merged['close_qfq'].shift(-prediction_period)
    merged['future_return'] = (merged['future_close_qfq'] - merged['close_qfq']) / merged['close_qfq']
    
    # 分类标签: 收益 > CLS_THRESHOLD
    merged['label_cls'] = (merged['future_return'] > CLS_THRESHOLD).astype(float)
    
    # 回归标签: 收益率
    merged['label_reg'] = merged['future_return']
    
    # 删除临时列
    result = merged.drop(columns=['close_qfq', 'future_close_qfq', 'future_return'])
    
    return result


def train_models(
    df: pd.DataFrame,
    model_save_dir: str = 'models/v2',
    enable_both_tasks: bool = True
):
    """
    训练模型
    
    Parameters
    ----------
    df : DataFrame
        训练数据
    model_save_dir : str
        模型保存目录
    enable_both_tasks : bool
        是否训练分类和回归两个任务
    """
    logger.info("="*80)
    logger.info("开始训练模型")
    logger.info("="*80)
    
    # 识别实际存在的特征列（排除标签和元数据）
    excluded_cols = {'date', 'symbol', 'label_cls', 'label_reg', 
                     'open', 'high', 'low', 'close', 'volume', 'amount', 'source',
                     'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                     'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'}
    
    # 自动检测数值特征和类别特征
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in excluded_cols:
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    logger.info(f"数值特征: {len(numerical_features)} - {numerical_features[:10]}...")
    logger.info(f"类别特征: {len(categorical_features)} - {categorical_features}")
    
    # 准备特征和标签
    feature_cols = numerical_features + categorical_features
    X = df[feature_cols].copy()
    
    # 初始化训练器
    trainer = EnhancedTrainerV2(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    
    # 训练分类模型
    if enable_both_tasks:
        logger.info("\n" + "="*80)
        logger.info("训练分类任务")
        logger.info("="*80)
        
        y_cls = df['label_cls'].copy()
        
        # 训练多个模型
        cls_models = {}
        
        # LightGBM
        logger.info("\n训练 LightGBM 分类器...")
        cls_lgb = trainer.train_classification_model(
            X, y_cls,
            model_type='lightgbm',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        cls_models['lightgbm'] = cls_lgb
        
        # XGBoost
        logger.info("\n训练 XGBoost 分类器...")
        cls_xgb = trainer.train_classification_model(
            X, y_cls,
            model_type='xgboost',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        cls_models['xgboost'] = cls_xgb
        
        # 选择最优模型
        best_cls_name = max(cls_models, key=lambda k: cls_models[k]['metrics']['val_auc'])
        best_cls = cls_models[best_cls_name]
        
        logger.info(f"\n最优分类模型: {best_cls_name} (AUC={best_cls['metrics']['val_auc']:.4f})")
        
        # 保存所有分类模型
        for name, model in cls_models.items():
            is_best = (name == best_cls_name)
            filepath = os.path.join(model_save_dir, f'cls_{PREDICTION_PERIOD_DAYS}d_{name}.pkl')
            trainer.save_model(model, filepath, is_best=is_best)
        
        # 额外保存最优模型
        best_filepath = os.path.join(model_save_dir, f'cls_{PREDICTION_PERIOD_DAYS}d_best.pkl')
        trainer.save_model(best_cls, best_filepath, is_best=True)
    
    # 训练回归模型
    if enable_both_tasks:
        logger.info("\n" + "="*80)
        logger.info("训练回归任务")
        logger.info("="*80)
        
        y_reg = df['label_reg'].copy()
        
        # 训练多个模型
        reg_models = {}
        
        # LightGBM
        logger.info("\n训练 LightGBM 回归器...")
        reg_lgb = trainer.train_regression_model(
            X, y_reg,
            model_type='lightgbm',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        reg_models['lightgbm'] = reg_lgb
        
        # XGBoost
        logger.info("\n训练 XGBoost 回归器...")
        reg_xgb = trainer.train_regression_model(
            X, y_reg,
            model_type='xgboost',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        reg_models['xgboost'] = reg_xgb
        
        # 选择最优模型
        best_reg_name = max(reg_models, key=lambda k: reg_models[k]['metrics']['val_r2'])
        best_reg = reg_models[best_reg_name]
        
        logger.info(f"\n最优回归模型: {best_reg_name} (R²={best_reg['metrics']['val_r2']:.4f})")
        
        # 保存所有回归模型
        for name, model in reg_models.items():
            is_best = (name == best_reg_name)
            filepath = os.path.join(model_save_dir, f'reg_{PREDICTION_PERIOD_DAYS}d_{name}.pkl')
            trainer.save_model(model, filepath, is_best=is_best)
        
        # 额外保存最优模型
        best_filepath = os.path.join(model_save_dir, f'reg_{PREDICTION_PERIOD_DAYS}d_best.pkl')
        trainer.save_model(best_reg, best_filepath, is_best=True)
    
    logger.info("\n" + "="*80)
    logger.info("✅ 所有模型训练完成")
    logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练统一预测模型 V2')
    parser.add_argument('--symbols', type=str, nargs='+', help='股票代码列表')
    parser.add_argument('--symbol-file', type=str, help='股票代码文件（每行一个）')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--prediction-period', type=int, default=PREDICTION_PERIOD_DAYS,
                        help=f'预测周期（天），默认{PREDICTION_PERIOD_DAYS}')
    parser.add_argument('--model-dir', type=str, default='models/v2',
                        help='模型保存目录')
    parser.add_argument('--classification-only', action='store_true',
                        help='只训练分类模型')
    parser.add_argument('--regression-only', action='store_true',
                        help='只训练回归模型')
    
    args = parser.parse_args()
    
    # 获取股票列表
    if args.symbols:
        symbols = args.symbols
    elif args.symbol_file:
        with open(args.symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        # 默认使用沪深300成分股
        logger.info("未指定股票列表，使用沪深300成分股")
        try:
            import akshare as ak
            df_hs300 = ak.index_stock_cons(symbol="000300")
            symbols = df_hs300['品种代码'].tolist()
        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")
            symbols = ['000001', '600000', '000002']  # Fallback
    
    # 设置日期范围
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start_date or (
        datetime.now() - timedelta(days=LOOKBACK_DAYS + args.prediction_period + 365)
    ).strftime('%Y-%m-%d')
    
    logger.info("训练配置:")
    logger.info(f"  股票数量: {len(symbols)}")
    logger.info(f"  日期范围: {start_date} ~ {end_date}")
    logger.info(f"  预测周期: {args.prediction_period}天")
    logger.info(f"  模型目录: {args.model_dir}")
    
    # 准备数据
    df = prepare_training_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        prediction_period=args.prediction_period
    )
    
    # 训练模型
    enable_both = not (args.classification_only or args.regression_only)
    
    train_models(
        df=df,
        model_save_dir=args.model_dir,
        enable_both_tasks=enable_both
    )
    
    logger.info("\n" + "="*80)
    logger.info("🎉 训练流程全部完成！")
    logger.info("="*80)


if __name__ == '__main__':
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    main()
