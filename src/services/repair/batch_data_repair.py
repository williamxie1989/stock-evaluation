"""
批量数据修复服务 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class BatchDataRepair:
    """批量数据修复服务"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.repair_service = None  # 将在需要时初始化
        logger.info(f"BatchDataRepair initialized with max_workers: {max_workers}")
    
    def _get_repair_service(self):
        """延迟初始化修复服务"""
        if self.repair_service is None:
            from src.services.repair.data_repair_service import DataRepairService
            self.repair_service = DataRepairService()
        return self.repair_service
    
    def repair_stock_data_batch(self, stock_data_dict: Dict[str, pd.DataFrame], 
                               repair_methods: Dict[str, str] = None) -> Dict[str, Any]:
        """批量修复股票数据"""
        try:
            if not stock_data_dict:
                return {'error': '股票数据字典为空'}
            
            logger.info(f"开始批量修复股票数据，共 {len(stock_data_dict)} 支股票")
            
            repair_results = {}
            repair_summary = {
                'total_stocks': len(stock_data_dict),
                'successful_repairs': 0,
                'failed_repairs': 0,
                'start_time': datetime.now(),
                'end_time': None,
                'processing_time': None
            }
            
            # 默认修复方法
            if repair_methods is None:
                repair_methods = {
                    'missing_values': 'interpolation',
                    'outliers': 'winsorize',
                    'price_consistency': 'auto',
                    'time_series': 'forward_fill'
                }
            
            # 使用线程池进行并行修复
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交修复任务
                future_to_stock = {
                    executor.submit(self._repair_single_stock, symbol, data, repair_methods): symbol
                    for symbol, data in stock_data_dict.items()
                }
                
                # 收集结果
                for future in as_completed(future_to_stock):
                    symbol = future_to_stock[future]
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        repair_results[symbol] = result
                        
                        if 'error' in result:
                            repair_summary['failed_repairs'] += 1
                            logger.warning(f"股票 {symbol} 修复失败: {result['error']}")
                        else:
                            repair_summary['successful_repairs'] += 1
                            logger.info(f"股票 {symbol} 修复成功")
                            
                    except Exception as e:
                        repair_results[symbol] = {'error': str(e)}
                        repair_summary['failed_repairs'] += 1
                        logger.error(f"股票 {symbol} 修复过程异常: {e}")
            
            # 计算处理时间
            repair_summary['end_time'] = datetime.now()
            repair_summary['processing_time'] = (
                repair_summary['end_time'] - repair_summary['start_time']
            ).total_seconds()
            
            # 生成最终报告
            final_report = {
                'repair_results': repair_results,
                'repair_summary': repair_summary,
                'repair_methods': repair_methods,
                'success_rate': repair_summary['successful_repairs'] / repair_summary['total_stocks'],
                'average_processing_time_per_stock': (
                    repair_summary['processing_time'] / repair_summary['total_stocks']
                    if repair_summary['total_stocks'] > 0 else 0
                )
            }
            
            logger.info(f"批量修复完成: 成功率={final_report['success_rate']:.2%}, "
                       f"总耗时={repair_summary['processing_time']:.2f}秒")
            
            return final_report
            
        except Exception as e:
            logger.error(f"批量修复股票数据失败: {e}")
            return {'error': str(e)}
    
    def _repair_single_stock(self, symbol: str, data: pd.DataFrame, 
                             repair_methods: Dict[str, str]) -> Dict[str, Any]:
        """修复单个股票数据"""
        try:
            if data.empty:
                return {'error': '数据为空'}
            
            repair_service = self._get_repair_service()
            
            # 执行综合修复
            repair_result = repair_service.comprehensive_repair(data, symbol)
            
            if 'error' in repair_result:
                return repair_result
            
            # 提取关键信息
            repaired_data = repair_result.get('repaired_data', data)
            repair_report = repair_result.get('repair_report', {})
            
            return {
                'symbol': symbol,
                'original_records': len(data),
                'repaired_records': len(repaired_data),
                'quality_score_improvement': repair_report.get('improvement', 0),
                'repair_steps_applied': repair_report.get('repair_steps', []),
                'repaired_data': repaired_data,
                'repair_report': repair_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"修复股票 {symbol} 失败: {e}")
            return {'symbol': symbol, 'error': str(e), 'success': False}
    
    def repair_market_data(self, market_data: pd.DataFrame, 
                          market_symbol: str = 'Market') -> Dict[str, Any]:
        """修复市场数据"""
        try:
            if market_data.empty:
                return {'error': '市场数据为空'}
            
            logger.info(f"开始修复市场数据: {market_symbol}")
            
            repair_service = self._get_repair_service()
            
            # 执行综合修复
            repair_result = repair_service.comprehensive_repair(market_data, market_symbol)
            
            if 'error' in repair_result:
                return repair_result
            
            repaired_data = repair_result.get('repaired_data', market_data)
            repair_report = repair_result.get('repair_report', {})
            
            return {
                'market_symbol': market_symbol,
                'original_records': len(market_data),
                'repaired_records': len(repaired_data),
                'quality_score_improvement': repair_report.get('improvement', 0),
                'repaired_data': repaired_data,
                'repair_report': repair_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"修复市场数据 {market_symbol} 失败: {e}")
            return {'error': str(e)}
    
    def repair_financial_data(self, financial_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """修复财务数据"""
        try:
            if not financial_data:
                return {'error': '财务数据为空'}
            
            logger.info(f"开始修复财务数据，共 {len(financial_data)} 个数据集")
            
            repair_results = {}
            repair_summary = {
                'total_datasets': len(financial_data),
                'successful_repairs': 0,
                'failed_repairs': 0
            }
            
            # 修复每个财务数据集
            for dataset_name, data in financial_data.items():
                try:
                    if data.empty:
                        repair_results[dataset_name] = {'error': '数据为空'}
                        repair_summary['failed_repairs'] += 1
                        continue
                    
                    repair_service = self._get_repair_service()
                    repair_result = repair_service.comprehensive_repair(data, dataset_name)
                    
                    if 'error' in repair_result:
                        repair_results[dataset_name] = repair_result
                        repair_summary['failed_repairs'] += 1
                    else:
                        repair_results[dataset_name] = {
                            'original_records': len(data),
                            'repaired_records': len(repair_result.get('repaired_data', data)),
                            'quality_score_improvement': repair_result.get('repair_report', {}).get('improvement', 0),
                            'repaired_data': repair_result.get('repaired_data', data),
                            'success': True
                        }
                        repair_summary['successful_repairs'] += 1
                        
                except Exception as e:
                    repair_results[dataset_name] = {'error': str(e)}
                    repair_summary['failed_repairs'] += 1
                    logger.error(f"修复财务数据集 {dataset_name} 失败: {e}")
            
            return {
                'repair_results': repair_results,
                'repair_summary': repair_summary,
                'success_rate': repair_summary['successful_repairs'] / repair_summary['total_datasets'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"修复财务数据失败: {e}")
            return {'error': str(e)}
    
    def get_batch_repair_statistics(self) -> Dict[str, Any]:
        """获取批量修复统计信息"""
        try:
            repair_service = self._get_repair_service()
            return repair_service.get_repair_statistics()
            
        except Exception as e:
            logger.error(f"获取批量修复统计信息失败: {e}")
            return {'total_repairs': 0, 'repair_types': {}}
    
    def export_repair_report(self, repair_results: Dict[str, Any], 
                           filename: str = None) -> str:
        """导出修复报告"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_repair_report_{timestamp}.json"
            
            # 转换datetime对象为字符串
            report_copy = repair_results.copy()
            
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            report_copy = convert_datetime(report_copy)
            
            # 保存报告
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_copy, f, ensure_ascii=False, indent=2)
            
            logger.info(f"修复报告已导出: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"导出修复报告失败: {e}")
            return ""
    
    def reset(self):
        """重置批量修复服务"""
        if self.repair_service:
            self.repair_service.reset()
        self.repair_service = None
        logger.info("批量修复服务已重置")