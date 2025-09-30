"""
模型训练管理器 - 集成训练流程和结果管理
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import os
import json

from src.ml.training.unified_trainer import UnifiedTrainer
from src.data.db.unified_database_manager import UnifiedDatabaseManager

logger = logging.getLogger(__name__)

class TrainingManager:
    """模型训练管理器 - 负责协调整个训练流程"""
    
    def __init__(self, model_dir: str = "models", 
                 training_config: Optional[Dict[str, Any]] = None):
        self.model_dir = model_dir
        self.training_config = training_config or self._get_default_config()
        self.trainer = UnifiedTrainer(model_dir=model_dir)
        self.db_manager = UnifiedDatabaseManager()
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认训练配置"""
        return {
            'period': '10d',
            'max_samples': 1000,
            'test_size': 0.2,
            'use_grid_search': True,
            'cv_folds': 5,
            'random_state': 42,
            'target_types': ['classification', 'regression']
        }
    
    def load_training_data(self, period: Optional[str] = None, 
                          limit: Optional[int] = None) -> pd.DataFrame:
        """
        从数据库加载训练数据
        
        Args:
            period: 数据周期，如 '10d', '30d'
            limit: 最大样本数量
            
        Returns:
            样本数据DataFrame
        """
        try:
            period = period or self.training_config['period']
            limit = limit or self.training_config['max_samples']
            
            query = """
                SELECT * FROM samples 
                WHERE period = ? 
                ORDER BY date DESC 
                LIMIT ?
            """
            
            results = self.db_manager.execute_query(query, [period, limit])
            
            if not results:
                logger.warning(f"没有找到周期为 {period} 的样本数据")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            logger.info(f"成功加载 {len(df)} 个样本数据 (周期: {period})")
            
            return df
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            return pd.DataFrame()
    
    def train_models(self, samples_df: Optional[pd.DataFrame] = None,
                    save_metadata: bool = True) -> Dict[str, Any]:
        """
        训练模型主流程
        
        Args:
            samples_df: 样本数据，如果为None则自动加载
            save_metadata: 是否保存训练元数据
            
        Returns:
            训练结果字典
        """
        try:
            # 加载数据
            if samples_df is None or samples_df.empty:
                samples_df = self.load_training_data()
            
            if samples_df.empty:
                return {'error': '没有可用的训练数据'}
            
            logger.info(f"开始训练流程，样本数量: {len(samples_df)}")
            
            # 解析特征
            features_df = self.trainer.parse_features_from_samples(samples_df)
            if features_df.empty:
                return {'error': '特征解析失败'}
            
            # 准备训练数据
            X, y_dict = self.trainer.prepare_training_data(
                features_df, target_type='both'
            )
            
            if X.empty or not y_dict:
                return {'error': '训练数据准备失败'}
            
            # 训练结果
            training_results = {
                'timestamp': datetime.now().isoformat(),
                'config': self.training_config,
                'data_info': {
                    'total_samples': len(features_df),
                    'features_count': X.shape[1],
                    'feature_names': X.columns.tolist()
                }
            }
            
            # 训练分类模型
            if 'classification' in self.training_config['target_types'] and 'classification' in y_dict:
                logger.info("开始训练分类模型...")
                cls_results = self.trainer.train_classification_models(
                    X, y_dict['classification'], 
                    use_grid_search=self.training_config['use_grid_search']
                )
                
                if 'best_model' in cls_results:
                    training_results['classification'] = {
                        'best_model': cls_results['best_model'],
                        'best_performance': cls_results['best_result']['performance'],
                        'all_models': {k: v['performance'] for k, v in cls_results['all_results'].items()}
                    }
                    logger.info(f"分类模型训练完成，最佳模型: {cls_results['best_model']}")
                else:
                    training_results['classification'] = {'error': '分类模型训练失败'}
            
            # 训练回归模型
            if 'regression' in self.training_config['target_types'] and 'regression' in y_dict:
                logger.info("开始训练回归模型...")
                reg_results = self.trainer.train_regression_models(
                    X, y_dict['regression'],
                    use_grid_search=self.training_config['use_grid_search']
                )
                
                if 'best_model' in reg_results:
                    training_results['regression'] = {
                        'best_model': reg_results['best_model'],
                        'best_performance': reg_results['best_result']['performance'],
                        'all_models': {k: v['performance'] for k, v in reg_results['all_results'].items()}
                    }
                    logger.info(f"回归模型训练完成，最佳模型: {reg_results['best_model']}")
                else:
                    training_results['regression'] = {'error': '回归模型训练失败'}
            
            # 保存训练元数据
            if save_metadata:
                self._save_training_metadata(training_results)
            
            logger.info("训练流程完成")
            return training_results
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _save_training_metadata(self, results: Dict[str, Any]):
        """保存训练元数据"""
        try:
            metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
            
            # 如果文件已存在，先加载现有数据
            existing_metadata = []
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # 添加新结果
            existing_metadata.append(results)
            
            # 保存更新后的元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练元数据已保存到: {metadata_path}")
            
        except Exception as e:
            logger.error(f"保存训练元数据失败: {e}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """获取训练历史"""
        try:
            metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
            
            if not os.path.exists(metadata_path):
                return []
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            return history
            
        except Exception as e:
            logger.error(f"获取训练历史失败: {e}")
            return []
    
    def get_best_models(self, model_type: str = 'both') -> Dict[str, Any]:
        """
        获取最佳模型信息
        
        Args:
            model_type: 'classification', 'regression', 或 'both'
            
        Returns:
            最佳模型信息字典
        """
        try:
            history = self.get_training_history()
            
            if not history:
                return {}
            
            # 获取最新的训练结果
            latest_results = history[-1]
            
            best_models = {}
            
            if model_type in ['classification', 'both'] and 'classification' in latest_results:
                cls_info = latest_results['classification']
                if 'best_model' in cls_info:
                    best_models['classification'] = {
                        'model_name': cls_info['best_model'],
                        'performance': cls_info['best_performance'],
                        'model_path': os.path.join(self.model_dir, f"{cls_info['best_model']}_classifier.pkl")
                    }
            
            if model_type in ['regression', 'both'] and 'regression' in latest_results:
                reg_info = latest_results['regression']
                if 'best_model' in reg_info:
                    best_models['regression'] = {
                        'model_name': reg_info['best_model'],
                        'performance': reg_info['best_performance'],
                        'model_path': os.path.join(self.model_dir, f"{reg_info['best_model']}_regressor.pkl")
                    }
            
            return best_models
            
        except Exception as e:
            logger.error(f"获取最佳模型失败: {e}")
            return {}
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新训练配置"""
        self.training_config.update(new_config)
        logger.info(f"训练配置已更新: {new_config}")
    
    def validate_models(self) -> Dict[str, Any]:
        """验证模型文件是否存在且可用"""
        try:
            best_models = self.get_best_models()
            validation_results = {}
            
            for model_type, model_info in best_models.items():
                model_path = model_info.get('model_path')
                
                if not model_path or not os.path.exists(model_path):
                    validation_results[model_type] = {
                        'status': 'missing',
                        'message': f'模型文件不存在: {model_path}'
                    }
                    continue
                
                # 尝试加载模型
                try:
                    import joblib
                    model = joblib.load(model_path)
                    validation_results[model_type] = {
                        'status': 'valid',
                        'model_name': model_info['model_name'],
                        'model_path': model_path
                    }
                except Exception as e:
                    validation_results[model_type] = {
                        'status': 'corrupted',
                        'message': f'模型文件损坏: {e}'
                    }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return {'error': str(e)}

# 主函数用于测试
def main():
    """测试训练管理器"""
    import sys
    sys.path.append('/Users/xieyongliang/stock-evaluation')
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 初始化训练管理器
        training_manager = TrainingManager(
            model_dir="models",
            training_config={
                'period': '10d',
                'max_samples': 200,
                'use_grid_search': True,
                'target_types': ['classification', 'regression']
            }
        )
        
        # 执行训练
        results = training_manager.train_models()
        
        if 'error' not in results:
            logger.info("训练管理器测试成功")
            
            # 获取最佳模型
            best_models = training_manager.get_best_models()
            logger.info(f"最佳模型: {best_models}")
            
            # 验证模型
            validation = training_manager.validate_models()
            logger.info(f"模型验证结果: {validation}")
        else:
            logger.error(f"训练管理器测试失败: {results['error']}")
    
    except Exception as e:
        logger.error(f"训练管理器测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()