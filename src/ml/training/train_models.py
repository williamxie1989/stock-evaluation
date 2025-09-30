#!/usr/bin/env python3
"""
模型训练脚本 - 用于训练股票预测模型
"""

import argparse
import logging
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.ml.training.training_manager import TrainingManager

def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 文件处理器
    file_handler = logging.FileHandler('training.log', encoding='utf-8', mode='w')
    file_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练股票预测模型')
    parser.add_argument('--period', type=str, default='10d',
                       help='数据周期 (默认: 10d)')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='最大样本数量 (默认: 1000)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='禁用网格搜索')
    parser.add_argument('--target-type', type=str, choices=['classification', 'regression', 'both'],
                       default='both', help='目标类型 (默认: both)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='模型保存目录 (默认: models)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始模型训练...")
        logger.info(f"参数: period={args.period}, max_samples={args.max_samples}, "
                   f"grid_search={not args.no_grid_search}, target_type={args.target_type}")
        
        # 初始化训练管理器
        training_manager = TrainingManager(
            model_dir=args.model_dir,
            training_config={
                'period': args.period,
                'max_samples': args.max_samples,
                'use_grid_search': not args.no_grid_search,
                'target_types': [args.target_type] if args.target_type != 'both' else ['classification', 'regression']
            }
        )
        
        # 执行训练
        results = training_manager.train_models()
        
        if 'error' in results:
            logger.error(f"训练失败: {results['error']}")
            return 1
        
        # 输出训练结果
        logger.info("训练完成！")
        
        if 'classification' in results and 'best_model' in results['classification']:
            cls_info = results['classification']
            logger.info(f"分类模型:")
            logger.info(f"  最佳模型: {cls_info['best_model']}")
            logger.info(f"  AUC: {cls_info['best_performance'].get('auc', 'N/A'):.4f}")
            logger.info(f"  准确率: {cls_info['best_performance'].get('accuracy', 'N/A'):.4f}")
        
        if 'regression' in results and 'best_model' in results['regression']:
            reg_info = results['regression']
            logger.info(f"回归模型:")
            logger.info(f"  最佳模型: {reg_info['best_model']}")
            logger.info(f"  R²: {reg_info['best_performance'].get('r2', 'N/A'):.4f}")
            logger.info(f"  RMSE: {reg_info['best_performance'].get('rmse', 'N/A'):.4f}")
        
        # 验证模型
        validation = training_manager.validate_models()
        logger.info("模型验证结果:")
        for model_type, status in validation.items():
            if status['status'] == 'valid':
                logger.info(f"  {model_type}: ✓ 正常")
            else:
                logger.info(f"  {model_type}: ✗ {status.get('message', '未知错误')}")
        
        logger.info(f"模型文件保存在: {args.model_dir}/")
        logger.info("训练日志保存在: training.log")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return 130
    except Exception as e:
        logger.error(f"训练过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())