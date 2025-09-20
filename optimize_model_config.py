#!/usr/bin/env python3
"""
模型优化配置脚本
提供具体的模型参数优化方案
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self):
        self.optimized_configs = {}
    
    def get_logistic_regression_configs(self):
        """获取逻辑回归的优化配置"""
        
        configs = {
            'balanced': {
                'name': '平衡类别权重配置',
                'params': {
                    'C': 0.1,  # 增加正则化，防止过拟合
                    'penalty': 'l1',  # L1正则化，特征选择
                    'solver': 'liblinear',
                    'class_weight': 'balanced',  # 自动平衡类别权重
                    'max_iter': 1000,
                    'random_state': 42
                },
                'description': '使用balanced权重处理类别不平衡，L1正则化进行特征选择'
            },
            
            'elastic_net': {
                'name': 'ElasticNet正则化配置',
                'params': {
                    'C': 0.5,
                    'penalty': 'elasticnet',
                    'solver': 'saga',
                    'l1_ratio': 0.5,  # L1和L2的平衡
                    'class_weight': 'balanced',
                    'max_iter': 2000,
                    'random_state': 42
                },
                'description': '结合L1和L2正则化，更好的特征选择和泛化能力'
            },
            
            'low_regularization': {
                'name': '低正则化配置',
                'params': {
                    'C': 10.0,  # 降低正则化强度
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'class_weight': {0: 0.3, 1: 0.7},  # 手动调整权重
                    'max_iter': 1000,
                    'random_state': 42
                },
                'description': '降低正则化强度，允许模型更复杂，手动调整类别权重'
            }
        }
        
        return configs
    
    def get_random_forest_configs(self):
        """获取随机森林的优化配置"""
        
        configs = {
            'balanced_rf': {
                'name': '平衡随机森林配置',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'description': '使用平衡权重的随机森林，适中的树深度防止过拟合'
            },
            
            'deep_rf': {
                'name': '深度随机森林配置',
                'params': {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'class_weight': 'balanced_subsample',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'description': '更深的树和更多估计器，使用子样本平衡'
            }
        }
        
        return configs
    
    def get_hyperparameter_search_configs(self):
        """获取超参数搜索配置"""
        
        # 逻辑回归超参数网格
        logistic_param_grid = {
            'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', {0: 0.3, 1: 0.7}, {0: 0.4, 1: 0.6}]
        }
        
        # 随机森林超参数网格
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 10, 12, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        return {
            'logistic_regression': logistic_param_grid,
            'random_forest': rf_param_grid
        }
    
    def create_optimized_pipeline_config(self):
        """创建优化的Pipeline配置"""
        
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import Pipeline
        
        configs = {
            'robust_pipeline': {
                'name': '鲁棒性Pipeline',
                'steps': [
                    ('scaler', RobustScaler()),  # 对异常值更鲁棒
                    ('selector', SelectKBest(f_classif, k=50)),  # 特征选择
                    ('classifier', LogisticRegression(
                        C=0.5, 
                        penalty='l1', 
                        solver='liblinear',
                        class_weight='balanced',
                        random_state=42
                    ))
                ],
                'description': '使用RobustScaler和特征选择的鲁棒Pipeline'
            },
            
            'ensemble_pipeline': {
                'name': '集成Pipeline',
                'steps': [
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1
                    ))
                ],
                'description': '基于随机森林的集成Pipeline'
            }
        }
        
        return configs
    
    def get_probability_calibration_config(self):
        """获取概率校准配置"""
        
        from sklearn.calibration import CalibratedClassifierCV
        
        configs = {
            'platt_scaling': {
                'name': 'Platt Scaling校准',
                'method': 'sigmoid',
                'cv': 3,
                'description': '使用Platt scaling进行概率校准，适合小样本'
            },
            
            'isotonic_regression': {
                'name': 'Isotonic Regression校准',
                'method': 'isotonic',
                'cv': 5,
                'description': '使用Isotonic regression校准，更灵活的校准方法'
            }
        }
        
        return configs
    
    def print_optimization_recommendations(self):
        """打印优化建议"""
        
        print("=" * 60)
        print("模型优化配置建议")
        print("=" * 60)
        
        print("\n1. 逻辑回归优化配置:")
        print("-" * 30)
        lr_configs = self.get_logistic_regression_configs()
        for key, config in lr_configs.items():
            print(f"\n{config['name']}:")
            print(f"  描述: {config['description']}")
            print(f"  参数: {config['params']}")
        
        print("\n\n2. 随机森林优化配置:")
        print("-" * 30)
        rf_configs = self.get_random_forest_configs()
        for key, config in rf_configs.items():
            print(f"\n{config['name']}:")
            print(f"  描述: {config['description']}")
            print(f"  参数: {config['params']}")
        
        print("\n\n3. Pipeline优化配置:")
        print("-" * 30)
        pipeline_configs = self.create_optimized_pipeline_config()
        for key, config in pipeline_configs.items():
            print(f"\n{config['name']}:")
            print(f"  描述: {config['description']}")
            print(f"  步骤: {[step[0] for step in config['steps']]}")
        
        print("\n\n4. 概率校准配置:")
        print("-" * 30)
        calib_configs = self.get_probability_calibration_config()
        for key, config in calib_configs.items():
            print(f"\n{config['name']}:")
            print(f"  描述: {config['description']}")
            print(f"  方法: {config['method']}")
            print(f"  交叉验证: {config['cv']}")
        
        print("\n\n5. 实施建议:")
        print("-" * 30)
        print("• 首先尝试'balanced'配置的逻辑回归")
        print("• 如果效果不佳，尝试随机森林'balanced_rf'配置")
        print("• 使用'robust_pipeline'处理特征")
        print("• 最后应用概率校准改善预测分布")
        print("• 建议使用时间序列交叉验证评估模型")
        
        print("\n\n6. 监控指标:")
        print("-" * 30)
        print("• 预测概率的标准差 (目标: >0.1)")
        print("• 预测概率的分布均匀性")
        print("• 分类准确率和召回率平衡")
        print("• 概率校准质量 (Brier Score)")

def create_improved_trainer_config():
    """创建改进的训练器配置文件"""
    
    config = {
        'model_configs': {
            'logistic_regression_balanced': {
                'C': 0.1,
                'penalty': 'l1',
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            },
            
            'random_forest_balanced': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        
        'preprocessing': {
            'scaler': 'RobustScaler',
            'feature_selection': {
                'method': 'SelectKBest',
                'k': 50
            }
        },
        
        'training': {
            'cv_folds': 5,
            'scoring': 'f1',
            'probability_calibration': True,
            'calibration_method': 'sigmoid'
        }
    }
    
    return config

if __name__ == "__main__":
    optimizer = ModelOptimizer()
    optimizer.print_optimization_recommendations()
    
    print("\n" + "=" * 60)
    print("配置文件已生成，可用于改进模型训练")
    print("=" * 60)