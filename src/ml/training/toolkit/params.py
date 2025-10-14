# -*- coding: utf-8 -*-
"""模型参数工具。"""


def get_conservative_lgbm_params() -> dict:
    """返回更保守的 LightGBM 参数配置。"""
    return {
        'n_estimators': 200,  # 🔧 从 300 降至 200
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 2.0,  # 🔧 从 1.0 提升到 2.0
        'reg_lambda': 10.0,  # 🔧 从 5.0 提升到 10.0
        'min_child_samples': 100,  # 🔧 从 50 提升到 100
        'early_stopping_rounds': 20  # 🔧 从 30 降至 20，更早停止
    }


def get_conservative_xgb_params() -> dict:
    """返回更保守的 XGBoost 参数配置。"""
    return {
        'n_estimators': 200,  # 🔧 从 300 降至 200
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 2.0,  # 🔧 从 1.0 提升到 2.0
        'reg_lambda': 10.0,  # 🔧 从 5.0 提升到 10.0
        'min_child_weight': 20,  # 🔧 从 10 提升到 20
        'early_stopping_rounds': 20  # 🔧 从 30 降至 20
    }
