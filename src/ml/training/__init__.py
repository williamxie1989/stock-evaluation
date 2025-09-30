"""
机器学习训练模块

这个模块负责股票预测模型的训练、评估和管理。

主要功能:
- 模型训练: 支持多种分类和回归模型
- 特征处理: 解析和处理JSON格式的特征数据
- 模型评估: 提供完整的性能评估指标
- 结果管理: 保存训练历史和模型文件

使用示例:
    # 命令行训练
    python -m src.ml.training.train_models --period 10d --max-samples 1000
    
    # 检查训练状态
    python -m src.ml.training.check_status --history --models
    
    # 编程接口
    from src.ml.training.training_manager import TrainingManager
    
    manager = TrainingManager()
    results = manager.train_models()
"""

from .training_manager import TrainingManager
from .unified_trainer import UnifiedTrainer

__all__ = ['TrainingManager', 'UnifiedTrainer']

__version__ = '1.0.0'