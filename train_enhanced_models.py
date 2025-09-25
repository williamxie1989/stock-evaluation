#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一阶段优化 - 增强模型训练脚本
使用增强型ML训练器和特征生成器训练多种先进模型
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ml_trainer import EnhancedMLTrainer
from db import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_enhanced_models():
    """训练增强模型"""
    try:
        logger.info("开始第一阶段优化 - 增强模型训练")
        
        # 初始化数据库管理器
        db_manager = DatabaseManager()
        
        # 初始化增强训练器
        trainer = EnhancedMLTrainer(
            db_manager=db_manager,
            model_dir="models/enhanced",
            use_enhanced_features=True,
            use_bayesian_optimization=True
        )
        
        # 定义训练参数
        training_periods = ['10d', '30d']
        date_ranges = [
            ('2024-01-01', '2024-06-01'),  # 训练期
            ('2024-06-01', '2024-09-01')   # 验证期
        ]
        
        best_results = {}
        
        for period in training_periods:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练周期: {period}")
            logger.info(f"{'='*60}")
            
            # 加载样本数据
            logger.info("加载样本数据...")
            samples = trainer.load_samples_from_db(
                period=period,
                start_date=date_ranges[0][0],
                end_date=date_ranges[0][1]
            )
            
            if len(samples) < 200:
                logger.warning(f"样本数据不足: {len(samples)} < 200")
                continue
            
            logger.info(f"成功加载 {len(samples)} 个样本")
            
            # 准备特征和目标
            logger.info("准备特征和目标变量...")
            X, y, feature_names = trainer.prepare_features_and_target(samples, 'label')
            
            if X.empty or len(y) == 0:
                logger.error("特征或目标数据为空")
                continue
            
            logger.info(f"特征数量: {len(feature_names)}, 样本数量: {len(X)}")
            
            # 对比多个模型
            logger.info("开始模型对比训练...")
            
            # 定义要训练的模型类型
            model_types = ['logistic', 'randomforest']
            
            # 检查XGBoost是否可用
            try:
                import xgboost
                model_types.append('xgboost')
                logger.info("XGBoost可用")
            except ImportError:
                logger.warning("XGBoost未安装，跳过")
            
            # 检查LightGBM是否可用
            try:
                import lightgbm
                model_types.append('lightgbm')
                logger.info("LightGBM可用")
            except ImportError:
                logger.warning("LightGBM未安装，跳过")
            
            # 训练并对比模型
            comparison_result = trainer.compare_models(
                X, y, 
                model_types=model_types,
                test_size=0.2
            )
            
            # 训练集成模型
            logger.info("训练集成模型...")
            try:
                ensemble_result = trainer.train_ensemble_model(
                    X, y,
                    model_types=model_types,
                    use_optimization=True
                )
                
                # 保存集成模型
                ensemble_path = trainer.save_model(
                    ensemble_result, 
                    f"ensemble_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                )
                
                logger.info(f"集成模型已保存: {ensemble_path}")
                
            except Exception as e:
                logger.error(f"集成模型训练失败: {e}")
                ensemble_result = None
            
            # 保存最佳模型
            best_model_type = comparison_result['best_model_type']
            best_result = comparison_result['best_result']
            
            best_model_path = trainer.save_model(
                best_result,
                f"best_{best_model_type}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            
            logger.info(f"最佳模型已保存: {best_model_path}")
            
            # 记录结果
            best_results[period] = {
                'best_model_type': best_model_type,
                'best_model_path': best_model_path,
                'ensemble_model_path': ensemble_path if ensemble_result else None,
                'metrics': best_result['metrics'],
                'feature_count': len(feature_names),
                'sample_count': len(samples)
            }
            
            # 打印详细结果
            logger.info(f"\n{'='*40}")
            logger.info(f"{period} 周期训练结果")
            logger.info(f"{'='*40}")
            logger.info(f"最佳模型: {best_model_type}")
            logger.info(f"测试集AUC: {best_result['metrics']['test_auc']:.4f}")
            logger.info(f"测试集准确率: {best_result['metrics']['test_accuracy']:.4f}")
            logger.info(f"测试集F1: {best_result['metrics']['test_f1']:.4f}")
            
            if ensemble_result:
                logger.info(f"集成模型AUC: {ensemble_result['metrics']['test_auc']:.4f}")
                logger.info(f"集成模型准确率: {ensemble_result['metrics']['test_accuracy']:.4f}")
            
            logger.info(f"特征数量: {len(feature_names)}")
            
            # 打印模型对比结果
            logger.info(f"\n模型对比结果:")
            logger.info(f"{comparison_result['comparison_df'].to_string(index=False, float_format='%.4f')}")
        
        # 总结所有结果
        logger.info(f"\n{'='*60}")
        logger.info("第一阶段优化训练完成")
        logger.info(f"{'='*60}")
        
        for period, result in best_results.items():
            logger.info(f"\n{period} 周期:")
            logger.info(f"  最佳模型: {result['best_model_type']}")
            logger.info(f"  AUC: {result['metrics']['test_auc']:.4f}")
            logger.info(f"  准确率: {result['metrics']['test_accuracy']:.4f}")
            logger.info(f"  F1: {result['metrics']['test_f1']:.4f}")
            logger.info(f"  特征数: {result['feature_count']}")
            logger.info(f"  样本数: {result['sample_count']}")
        
        # 保存训练总结
        summary_path = f"models/enhanced/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(best_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n训练总结已保存: {summary_path}")
        
        return best_results
        
    except Exception as e:
        logger.error(f"增强模型训练失败: {e}", exc_info=True)
        raise

def validate_enhanced_models():
    """验证增强模型效果"""
    try:
        logger.info("开始验证增强模型效果")
        
        trainer = EnhancedMLTrainer(
            model_dir="models/enhanced",
            use_enhanced_features=True
        )
        
        # 加载最新的最佳模型
        model_files = list(Path("models/enhanced").glob("best_*.pkl"))
        if not model_files:
            logger.error("未找到训练好的模型")
            return
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"加载模型: {latest_model}")
        
        model_result = trainer.load_model(str(latest_model))
        
        # 验证模型性能
        logger.info("模型性能验证:")
        logger.info(f"模型类型: {model_result['model_type']}")
        logger.info(f"测试集AUC: {model_result['metrics']['test_auc']:.4f}")
        logger.info(f"测试集准确率: {model_result['metrics']['test_accuracy']:.4f}")
        logger.info(f"测试集F1: {model_result['metrics']['test_f1']:.4f}")
        logger.info(f"测试集精确率: {model_result['metrics']['test_precision']:.4f}")
        logger.info(f"测试集召回率: {model_result['metrics']['test_recall']:.4f}")
        
        # 特征重要性分析
        if 'feature_importance' in model_result and model_result['feature_importance']:
            logger.info(f"\nTop 10 重要特征:")
            sorted_features = sorted(
                model_result['feature_importance'].items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]
            
            for feature, importance in sorted_features:
                logger.info(f"  {feature}: {importance:.4f}")
        
        # 混淆矩阵分析
        if 'confusion_matrix' in model_result:
            cm = np.array(model_result['confusion_matrix'])
            logger.info(f"\n混淆矩阵:")
            logger.info(f"真正例: {cm[1,1]}, 假正例: {cm[0,1]}")
            logger.info(f"假负例: {cm[1,0]}, 真负例: {cm[0,0]}")
        
        logger.info("增强模型验证完成")
        
    except Exception as e:
        logger.error(f"模型验证失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # 确保模型目录存在
        Path("models/enhanced").mkdir(parents=True, exist_ok=True)
        
        # 训练增强模型
        results = train_enhanced_models()
        
        # 验证模型效果
        validate_enhanced_models()
        
        logger.info("第一阶段优化实施完成!")
        
    except Exception as e:
        logger.error(f"第一阶段优化失败: {e}", exc_info=True)
        sys.exit(1)