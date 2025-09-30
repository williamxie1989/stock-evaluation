"""
参数优化器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        logger.info(f"ParameterOptimizer initialized with max_workers={max_workers}")
    
    def optimize_parameters(self, 
                          data: pd.DataFrame,
                          strategy_func: Callable,
                          parameter_space: Dict[str, List],
                          optimization_metric: str = 'sharpe_ratio',
                          train_ratio: float = 0.7) -> Dict[str, Any]:
        """优化策略参数"""
        logger.info(f"开始参数优化，参数空间: {parameter_space}")
        
        # 分割训练和测试数据
        train_data, test_data = self._split_data(data, train_ratio)
        logger.info(f"数据分割完成 - 训练集: {len(train_data)}, 测试集: {len(test_data)}")
        
        # 生成所有参数组合
        parameter_combinations = self._generate_parameter_combinations(parameter_space)
        logger.info(f"生成 {len(parameter_combinations)} 个参数组合")
        
        # 并行优化
        best_params, best_score, results = self._parallel_optimization(
            train_data, strategy_func, parameter_combinations, optimization_metric
        )
        
        # 在测试集上验证
        test_result = strategy_func(test_data, **best_params)
        test_score = self._calculate_metric(test_result, optimization_metric)
        
        logger.info(f"参数优化完成 - 最优参数: {best_params}, 训练集得分: {best_score:.4f}, 测试集得分: {test_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'test_score': test_score,
            'all_results': results,
            'train_data_size': len(train_data),
            'test_data_size': len(test_data),
            'optimization_metric': optimization_metric
        }
    
    def _split_data(self, data: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割数据为训练和测试集"""
        split_point = int(len(data) * train_ratio)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        return train_data, test_data
    
    def _generate_parameter_combinations(self, parameter_space: Dict[str, List]) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = list(itertools.product(*values))
        parameter_combinations = []
        
        for combination in combinations:
            params = dict(zip(keys, combination))
            parameter_combinations.append(params)
        
        return parameter_combinations
    
    def _parallel_optimization(self, 
                             data: pd.DataFrame,
                             strategy_func: Callable,
                             parameter_combinations: List[Dict[str, Any]],
                             optimization_metric: str) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """并行优化"""
        results = []
        best_params = None
        best_score = float('-inf')
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._evaluate_parameters, data, strategy_func, params, optimization_metric): params
                for params in parameter_combinations
            }
            
            # 收集结果
            for future in as_completed(futures):
                params = futures[future]
                try:
                    score, result = future.result()
                    results.append({
                        'parameters': params,
                        'score': score,
                        'result': result
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    logger.error(f"参数评估失败 {params}: {e}")
        
        return best_params, best_score, results
    
    def _evaluate_parameters(self, 
                           data: pd.DataFrame,
                           strategy_func: Callable,
                           parameters: Dict[str, Any],
                           optimization_metric: str) -> Tuple[float, Any]:
        """评估一组参数"""
        try:
            # 运行策略
            result = strategy_func(data, **parameters)
            
            # 计算指标
            score = self._calculate_metric(result, optimization_metric)
            
            return score, result
        except Exception as e:
            logger.error(f"参数评估失败: {parameters}, 错误: {e}")
            return float('-inf'), None
    
    def _calculate_metric(self, result: Any, metric: str) -> float:
        """计算优化指标"""
        if hasattr(result, metric):
            return getattr(result, metric)
        elif isinstance(result, dict) and metric in result:
            return result[metric]
        else:
            logger.warning(f"指标 {metric} 不存在，返回0")
            return 0
    
    def grid_search(self, 
                   data: pd.DataFrame,
                   strategy_func: Callable,
                   parameter_space: Dict[str, List],
                   optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """网格搜索（串行版本）"""
        logger.info(f"开始网格搜索，参数空间: {parameter_space}")
        
        # 生成所有参数组合
        parameter_combinations = self._generate_parameter_combinations(parameter_space)
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        # 遍历所有参数组合
        for i, params in enumerate(parameter_combinations):
            if i % 10 == 0:
                logger.info(f"进度: {i+1}/{len(parameter_combinations)}")
            
            try:
                score, result = self._evaluate_parameters(data, strategy_func, params, optimization_metric)
                results.append({
                    'parameters': params,
                    'score': score,
                    'result': result
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"参数评估失败 {params}: {e}")
        
        logger.info(f"网格搜索完成 - 最优参数: {best_params}, 最优得分: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results,
            'optimization_metric': optimization_metric,
            'total_combinations': len(parameter_combinations)
        }
    
    def random_search(self, 
                     data: pd.DataFrame,
                     strategy_func: Callable,
                     parameter_space: Dict[str, Tuple],
                     n_iterations: int = 100,
                     optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """随机搜索"""
        logger.info(f"开始随机搜索，迭代次数: {n_iterations}")
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        for i in range(n_iterations):
            if i % 10 == 0:
                logger.info(f"随机搜索进度: {i+1}/{n_iterations}")
            
            # 随机生成参数
            params = self._generate_random_parameters(parameter_space)
            
            try:
                score, result = self._evaluate_parameters(data, strategy_func, params, optimization_metric)
                results.append({
                    'parameters': params,
                    'score': score,
                    'result': result
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"参数评估失败 {params}: {e}")
        
        logger.info(f"随机搜索完成 - 最优参数: {best_params}, 最优得分: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results,
            'optimization_metric': optimization_metric,
            'n_iterations': n_iterations
        }
    
    def _generate_random_parameters(self, parameter_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """生成随机参数"""
        import random
        
        params = {}
        for key, value_range in parameter_space.items():
            if isinstance(value_range, tuple) and len(value_range) == 2:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    params[key] = random.randint(value_range[0], value_range[1])
                else:
                    params[key] = random.uniform(value_range[0], value_range[1])
            else:
                # 如果是离散值列表，随机选择一个
                params[key] = random.choice(value_range)
        
        return params