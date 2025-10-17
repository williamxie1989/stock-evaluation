# -*- coding: utf-8 -*-
"""
预测系统配置文件
包含训练和预测的统一配置参数
"""

# ============ 预测目标配置 ============
PREDICTION_PERIOD_DAYS = 30  # 未来预测天数
CLS_THRESHOLD = 0.04  # 截面样本不足时的兜底绝对阈值

# 标签构建策略
LABEL_STRATEGY = 'quantile'  # 默认使用横截面quantile策略
LABEL_POSITIVE_QUANTILE = 0.7  # 前30% 作为正类
LABEL_NEGATIVE_QUANTILE = 0.3  # 后30% 作为负类
LABEL_MIN_SAMPLES_PER_DATE = 30  # quantile策略每日最少样本，低于该值回退absolute
ENABLE_LABEL_NEUTRAL_BAND = False  # 是否引入中性区间
LABEL_NEUTRAL_QUANTILE = 0.5  # 中性区间上界（仅当启用neutral band时有效）
LABEL_USE_MARKET_BASELINE = False  # 是否使用市场基准构建超额收益标签
LABEL_USE_INDUSTRY_NEUTRAL = True  # 是否对行业截面做去均值

# 批量训练配置
ENABLE_BATCH_TRAINING = True  # 🆕 是否启用批量训练模式（一次性训练多只股票）
BATCH_TRAINING_SIZE = 50  # 🆕 每批训练的股票数量（可根据内存调整：10-300）
# 批量训练优势：
#   - 使用真正的横截面quantile策略（每天多只股票排序）
#   - 正样本率稳定在设定分位数（如25%）
#   - 避免时间序列quantile的市场周期偏差
# 批量训练成本：
#   - 内存占用: BATCH_SIZE * 数据量 (建议50只股票约需2-4GB)
#   - 训练时间: 比单股票模式慢，但可通过减少BATCH_SIZE平衡

# ============ 数据配置 ============
LOOKBACK_DAYS = 720  # 特征计算回溯天数（用于默认start_date计算）
MIN_TRAINING_DAYS = 250  # 🔧 训练数据足够性检查阈值（低于此值跳过该股票）
MIN_HISTORY_DAYS = 45  # 最少历史数据要求
MIN_VALID_TRADING_DAYS = 30  # 最少有效交易天数（用于confidence计算）

# ============ 特征工程配置 ============
# 价量特征
ENABLE_PRICE_VOLUME_FEATURES = True

# 市场因子
ENABLE_MARKET_FACTOR = True  # 是否启用市场因子
MIN_STOCKS_FOR_MARKET = 20  # 🔧 修复: 从 50 降低到 20，提高市场因子构建成功率

# 行业特征
ENABLE_INDUSTRY_FEATURES = True
INDUSTRY_MIN_FREQUENCY = 0.005  # 行业最小频率，低于此值归为'Other'

# 板块特征
ENABLE_BOARD_ONEHOT = True  # 是否启用板块One-Hot

# 基本面特征配置
FUNDAMENTAL_PUBLISH_DELAY_DAYS = 60  # 🔧 财报公告延迟天数（数据源缺publish_date时使用）
# 说明：当数据源缺少publish_date时，假设财报在report_date后N天公告
# 设置为60天是保守估计，可根据实际情况调整：
#   - 年报/半年报：通常30-90天
#   - 季报：通常20-45天
#   - 建议：60天（平衡准确性和数据可用性）

# 截面标准化特征
ENABLE_CROSS_SECTIONAL_ENRICHMENT = True
CROSS_SECTIONAL_FEATURES = [
    'ret', 'ret_5', 'ret_20', 'ret_60', 'risk_adj_mom_60',
    'short_rev', 'vol_20', 'vol_60', 'downside_vol_20',
    'turnover', 'ADV_20', 'log_ADV_20', 'illiq_20',
    'beta_60', 'alpha_60',
    'idio_vol_60', 'market_R2_60',
    'rel_strength_20', 'rel_strength_60', 'CMF', 'MFI'
]

# ============ 预处理配置 ============
WINSOR_CLIP_QUANTILE = 0.01  # Winsorizer剪尾分位数
HANDLE_UNKNOWN_CATEGORY = 'ignore'  # OneHotEncoder处理未知类别的方式

# ============ 训练配置 ============
# 交叉验证
CV_N_SPLITS = 5  # 交叉验证折数
CV_EMBARGO = 60  # 禁用期（避免信息泄漏）- 已足够长，无需修改
CV_ALLOW_FUTURE = False  # ✅ Stage5修复: 时间序列场景禁止使用未来数据
ENABLE_TIME_SERIES_SPLIT = True  # 是否启用时间序列切分
# 当启用时间序列切分时，使用最后一折作为验证集

# 🔧 滚动窗口配置（关键改进）
USE_ROLLING_WINDOW = True  # 是否使用滚动窗口切分（而非全部历史）
ROLLING_TRAIN_YEARS = 2  # 训练窗口年数（使用最近3年数据）
ROLLING_VAL_YEARS = 0.5  # 验证窗口年数（最近1年）
ROLLING_EMBARGO_DAYS = 40  # 🔧 Stage5修复: 从5天改为40天 (预测期30天+10天缓冲)

# 样本加权配置
ENABLE_SAMPLE_WEIGHTING = False  # 是否对样本加时间衰减权重（近期权重更高）
SAMPLE_WEIGHT_HALFLIFE_YEARS = 1.0  # 权重半衰期（年）

# 特征选择
ENABLE_FEATURE_SELECTION = True  # ✅ 已启用，OneHot后特征选择正常工作
FEATURE_SELECTION_METHOD = 'model_based'  # 'model_based' or 'shap'
MIN_FEATURE_IMPORTANCE = 0.005  # 最小特征重要性阈值,0.001到0.005

# 概率校准
ENABLE_CALIBRATION = True  # 是否启用概率校准
CALIBRATION_METHOD = 'isotonic'  # 'isotonic' or 'sigmoid' (Platt)
CALIBRATION_CV = 3  # 校准交叉验证折数

# 模型选择
AUTO_MODEL_SELECTION = True  # 自动选择最优模型
CLASSIFICATION_MODELS = [
    'logistic',
    'lightgbm',
    'xgboost',
    'random_forest'
]
REGRESSION_MODELS = [
    'ridge',
    'lightgbm',
    'xgboost',
    'stacking'  # Stacking集成
]

# Optuna超参数优化
ENABLE_OPTUNA = True
OPTUNA_N_TRIALS = 30  # ✅ Stage5修复: 与训练脚本一致 (--optuna-trials 30)
OPTUNA_TIMEOUT = 3600  # 优化超时时间(秒)
OPTUNA_PATIENCE = 10  # PatientPruner容忍度

# ============ 模型超参数配置 ============
# LightGBM 超参数 (✅ Stage5修复: 与Optuna约束一致)
LIGHTGBM_PARAMS = {
    'n_estimators': 300,      # ✅ 已调优: 增加树数量
    'max_depth': 5,           # ✅ Stage5修复: 从5改为4，与Optuna一致
    'learning_rate': 0.05,    # ✅ 已调优: 降低学习率配合更多树
    'num_leaves': 31,         # ✅ Stage5修复: 2^4-1=15，匹配max_depth=4
    'min_child_samples': 20,  # ✅ 已调优: 增强正则化（Optuna范围20-100）
    'subsample': 0.7,         # ✅ Stage5修复: 与Optuna 0.6-0.8一致
    'colsample_bytree': 0.7,  # ✅ Stage5修复: 与Optuna 0.6-0.8一致
    'reg_alpha': 1.0,         # ✅ Stage5修复: 从0.1改为5.0（Optuna范围3-10）
    'reg_lambda': 1.0,        # ✅ Stage5修复: 从1.0改为7.0（Optuna范围5-10）
}

# XGBoost 超参数 (✅ Stage5修复: 与Optuna约束一致)
XGBOOST_PARAMS = {
    'n_estimators': 300,      # ✅ 已调优: 增加树数量
    'max_depth': 5,           # ✅ Stage5修复: 从5改为4，与Optuna一致
    'learning_rate': 0.05,    # ✅ 已调优: 降低学习率
    'subsample': 0.7,         # ✅ Stage5修复: 与Optuna 0.6-0.8一致
    'colsample_bytree': 0.7,  # ✅ Stage5修复: 与Optuna 0.6-0.8一致
    'reg_alpha': 1.0,         # ✅ Stage5修复: 从0.1改为5.0（Optuna范围3-10）
    'reg_lambda': 1.0,        # ✅ Stage5修复: 从1.0改为7.0（Optuna范围5-10）
    'min_child_weight': 20,   # ✅ Stage5修复: 从5改为10（Optuna范围5-15）
    'gamma': 2.0,             # ✅ Stage5修复: 从0.1改为2.0（Optuna范围1-5）
}

# Logistic Regression 超参数 (✅ 已应用保守调优)
LOGISTIC_PARAMS = {
    'C': 0.1,                 # ✅ 已调优: 增强L2正则化，减少过拟合
    'max_iter': 2000,         # ✅ 已调优: 增加迭代次数
    'solver': 'saga',         # ✅ 已调优: 支持L1正则化
    'penalty': 'elasticnet',  # ✅ 已调优: L1+L2混合正则化
    'l1_ratio': 0.5,          # ✅ 已调优: elasticnet混合比例
}

# Random Forest 超参数
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
}

# Ridge Regression 超参数
RIDGE_PARAMS = {
    'alpha': 1.0,
    'max_iter': 1000,
}

# ============ 评估指标配置 ============
# 分类评估
CLASSIFICATION_METRICS = [
    'auc',  # AUC-ROC
    'ks',  # KS统计量
    'f1',  # F1分数
    'brier',  # Brier分数
    'calibration',  # 校准曲线
    'top_k_hit_rate'  # Top-K命中率
]
TOP_K_VALUES = [20, 50, 100]  # Top-K评估的K值

# 回归评估
REGRESSION_METRICS = [
    'r2',  # R²决定系数
    'mae',  # 平均绝对误差
    'ic',  # 信息系数(秩相关)
    'quantile_monotonicity'  # 分层单调性
]

# ============ 回归标签优化（阶段1新增）============
NORMALIZE_REGRESSION_LABELS = True  # 标准化回归标签，减少极端值影响
REGRESSION_LABEL_WINSORIZE = True   # 对标签进行缩尾处理
REGRESSION_LABEL_CLIP_PERCENTILE = 0.01  # 上下1%截断


# 通过门槛
MIN_CLASSIFICATION_AUC = 0.50  # ✅ Stage5修复: 至少略好于随机（从0.48提升）
MIN_REGRESSION_R2 = -0.10  # 🔧 允许负R²: 小样本回归可能过拟合，但仍可分析特征重要性
MIN_TOP_K_IMPROVEMENT = 3.0  # Top-K命中率最小提升(百分点)

# ============ 预测服务配置 ============
# 概率/收益限幅
ENABLE_PROBABILITY_CLIPPING = True
PROB_CLIP_MIN = 0.02
PROB_CLIP_MAX = 0.98

ENABLE_RETURN_CLIPPING = True
RETURN_SOFT_CLIP = 0.18  # tanh软限幅
RETURN_HARD_CLIP_MIN = -0.25  # 硬限幅下界
RETURN_HARD_CLIP_MAX = 0.25  # 硬限幅上界

# Confidence计算
CONFIDENCE_MIN = 30  # 最低置信度
CONFIDENCE_MAX = 95  # 最高置信度
CONFIDENCE_FEATURES_WEIGHT = 0.3  # 特征有效性权重
CONFIDENCE_PROB_WEIGHT = 0.4  # 概率差异权重
CONFIDENCE_HISTORY_WEIGHT = 0.3  # 历史数据权重

# 兜底规则（仅在异常情况触发）
ENABLE_RULE_BASED_CALIBRATION = False  # 默认关闭
RULE_TRIGGER_STD_THRESHOLD = 0.02  # 概率标准差低于此值触发兜底

# 随机扰动（已废弃，保留开关用于回归测试）
ENABLE_RANDOM_PERTURBATION = False  # 强制关闭随机扰动

# ============ 模型持久化配置 ============
MODEL_SAVE_FORMAT = 'v2'  # 新格式，包含pipeline、calibrator等
MODEL_DIR = 'models'
BEST_MODEL_SUFFIX = '_best'  # 最优模型后缀

# 模型元数据字段
MODEL_METADATA_FIELDS = [
    'task',  # 'classification' or 'regression'
    'pipeline',  # sklearn Pipeline对象
    'selected_features',  # 特征列表
    'metrics',  # 评估指标字典
    'threshold',  # 分类阈值
    'calibrator',  # 校准器对象（可为None）
    'is_best',  # 是否为最优模型
    'training_date',  # 训练日期
    'config'  # 训练配置快照
]

# ============ 日志与监控配置 ============
LOG_LEVEL = 'INFO'
ENABLE_PERFORMANCE_MONITORING = True  # 启用性能监控
ENABLE_DRIFT_DETECTION = True  # 启用分布漂移检测

# ============ 实验性功能（灰度）============
ENABLE_CLUSTER_FEATURE = False  # 聚类特征（未实现）
ENABLE_ALTERNATIVE_WINSORIZER = False  # 自定义Winsorizer（可选）

# ============ 智能选股性能配置 ============
import os

# 股票池大小限制
MAX_STOCK_POOL_SIZE = int(os.getenv('MAX_STOCK_POOL_SIZE', '500'))  # 默认500只，避免处理全市场

# 特征缓存配置
ENABLE_FEATURE_CACHE = os.getenv('ENABLE_FEATURE_CACHE', 'true').lower() == 'true'
FEATURE_CACHE_TTL = int(os.getenv('FEATURE_CACHE_TTL', '3600'))  # 特征缓存有效期（秒），默认1小时
FEATURE_CACHE_DIR = os.getenv('FEATURE_CACHE_DIR', 'cache/features')  # 特征缓存目录

# 批处理配置
FEATURE_BUILD_BATCH_SIZE = int(os.getenv('FEATURE_BUILD_BATCH_SIZE', '100'))  # 每批处理股票数

# Symbol标准化缓存
ENABLE_SYMBOL_CACHE = os.getenv('ENABLE_SYMBOL_CACHE', 'true').lower() == 'true'
SYMBOL_CACHE_TTL = int(os.getenv('SYMBOL_CACHE_TTL', '86400'))  # Symbol缓存有效期（秒），默认24小时
