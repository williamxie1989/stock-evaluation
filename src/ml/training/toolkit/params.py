# -*- coding: utf-8 -*-
"""模型参数工具。"""


def get_conservative_lgbm_params() -> dict:
    """返回保守的 LightGBM 参数配置（Optuna禁用时的fallback）。
    
    注意: 当enable_optuna=True时，这些参数会被Optuna优化结果覆盖。
    这些参数应与prediction_config.py中的LIGHTGBM_PARAMS保持一致。
    """
    return {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'early_stopping_rounds': 30,
        'verbose': -1
    }


def get_conservative_xgb_params() -> dict:
    """返回保守的 XGBoost 参数配置（Optuna禁用时的fallback）。
    
    注意: 当enable_optuna=True时，这些参数会被Optuna优化结果覆盖。
    这些参数应与prediction_config.py中的XGBOOST_PARAMS保持一致。
    """
    return {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'early_stopping_rounds': 30,
        'verbosity': 0
    }


def get_optimized_lgbm_regression_params() -> dict:
    """返回优化的 LightGBM 回归参数（R²增强版）。
    
    相比保守分类参数的改进：
    - 增加模型复杂度（更深的树，更多叶子）
    - 降低学习率配合更多轮次
    - 增加特征采样率（更充分利用特征）
    - 降低正则化（回归任务对过拟合不如分类敏感）
    """
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,  # 增加复杂度 (vs 分类的 max_depth=3)
        'learning_rate': 0.02,  # 适中学习率 (vs 分类的 0.01)
        'feature_fraction': 0.9,  # 高特征采样 (vs 分类的 0.7)
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,  # 更宽松 (vs 分类的 100)
        'lambda_l1': 0.1,  # 轻正则化 (vs 分类的 2.0)
        'lambda_l2': 0.1,  # 轻正则化 (vs 分类的 10.0)
        'max_depth': 8,  # 明确深度限制
        'min_gain_to_split': 0.01,
        'verbose': -1,
    }


def get_optimized_xgb_regression_params() -> dict:
    """返回优化的 XGBoost 回归参数（R²增强版）。"""
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': 8,  # 增加深度 (vs 分类的 3)
        'learning_rate': 0.02,  # 适中学习率
        'subsample': 0.8,
        'colsample_bytree': 0.9,  # 高特征采样
        'min_child_weight': 50,  # 更宽松 (vs 分类的 20)
        'gamma': 0.01,
        'reg_alpha': 0.1,  # 轻正则化
        'reg_lambda': 0.1,  # 轻正则化
        'max_leaves': 63,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
    }
