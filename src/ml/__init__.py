"""
股票评估系统 - 机器学习模块

提供特征工程、模型训练、模型评估等机器学习功能。
"""

from .features.enhanced_features import EnhancedFeatureGenerator
# train_unified_models 已在第四阶段被归档，使用 apps.scripts 中的替代模块
# from .training.train_unified_models import EnhancedMLTrainer
from .evaluation.validate_unified_models import UnifiedModelValidator

__all__ = [
    'EnhancedFeatureGenerator',
    # 'EnhancedMLTrainer',  # 已归档
    'UnifiedModelValidator'
]