"""
数据修复服务 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class DataRepairService:
    """数据修复服务 - 修复和清理股票数据"""
    
    def __init__(self, repair_log_dir: str = "logs/repair"):
        self.repair_log_dir = repair_log_dir
        self.repair_history = []
        self.repair_rules = {}
        
        # 创建修复日志目录
        if not os.path.exists(repair_log_dir):
            os.makedirs(repair_log_dir)
        
        logger.info(f"DataRepairService initialized with repair_log_dir: {repair_log_dir}")
    
    def detect_data_issues(self, data: pd.DataFrame, symbol: str = 'Unknown') -> Dict[str, Any]:
        """检测数据问题"""
        try:
            if data.empty:
                return {'error': '数据为空'}
            
            issues = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'missing_values': {},
                'outliers': {},
                'inconsistencies': {},
                'data_quality_score': 1.0
            }
            
            # 1. 检测缺失值
            missing_counts = data.isnull().sum()
            missing_pct = (missing_counts / len(data)) * 100
            
            for column in data.columns:
                if missing_pct[column] > 0:
                    issues['missing_values'][column] = {
                        'count': int(missing_counts[column]),
                        'percentage': float(missing_pct[column])
                    }
            
            # 2. 检测异常值
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column in ['open', 'high', 'low', 'close']:
                    # 价格异常值检测
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                    if len(outliers) > 0:
                        issues['outliers'][column] = {
                            'count': len(outliers),
                            'percentage': (len(outliers) / len(data)) * 100,
                            'min_value': float(outliers[column].min()) if len(outliers) > 0 else None,
                            'max_value': float(outliers[column].max()) if len(outliers) > 0 else None
                        }
                
                elif column == 'volume':
                    # 成交量异常值检测
                    volume_mean = data[column].mean()
                    volume_std = data[column].std()
                    
                    if volume_std > 0:
                        z_scores = abs((data[column] - volume_mean) / volume_std)
                        outliers = data[z_scores > 3]  # Z-score > 3 为异常值
                        
                        if len(outliers) > 0:
                            issues['outliers'][column] = {
                                'count': len(outliers),
                                'percentage': (len(outliers) / len(data)) * 100,
                                'threshold': float(volume_mean + 3 * volume_std)
                            }
            
            # 3. 检测数据不一致性
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # 价格逻辑检查
                price_inconsistencies = data[
                    (data['high'] < data['low']) |
                    (data['open'] > data['high']) |
                    (data['open'] < data['low']) |
                    (data['close'] > data['high']) |
                    (data['close'] < data['low'])
                ]
                
                if len(price_inconsistencies) > 0:
                    issues['inconsistencies']['price_logic'] = {
                        'count': len(price_inconsistencies),
                        'percentage': (len(price_inconsistencies) / len(data)) * 100
                    }
            
            # 4. 检测时间序列问题
            if not data.index.is_monotonic_increasing:
                issues['inconsistencies']['time_series'] = {
                    'issue': '时间序列不连续或乱序',
                    'count': len(data) - 1
                }
            
            # 5. 计算数据质量分数
            total_issues = 0
            total_records = len(data)
            
            for category in ['missing_values', 'outliers', 'inconsistencies']:
                for issue in issues[category].values():
                    if isinstance(issue, dict) and 'count' in issue:
                        total_issues += issue['count']
            
            # 数据质量分数 (0-1, 1表示完美)
            issues['data_quality_score'] = max(0.0, 1.0 - (total_issues / total_records))
            
            logger.info(f"数据问题检测完成: {symbol}, 质量分数: {issues['data_quality_score']:.2f}")
            return issues
            
        except Exception as e:
            logger.error(f"数据问题检测失败: {e}")
            return {'error': str(e)}
    
    def repair_missing_values(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """修复缺失值"""
        try:
            if data.empty:
                return data
            
            repaired_data = data.copy()
            original_missing = data.isnull().sum().sum()
            
            if method == 'forward_fill':
                # 前向填充
                repaired_data = repaired_data.fillna(method='ffill')
                # 如果前面没有值，用后向填充
                repaired_data = repaired_data.fillna(method='bfill')
            
            elif method == 'interpolation':
                # 插值法
                numeric_columns = repaired_data.select_dtypes(include=[np.number]).columns
                repaired_data[numeric_columns] = repaired_data[numeric_columns].interpolate(method='linear')
                # 如果还有缺失值，用前后填充
                repaired_data[numeric_columns] = repaired_data[numeric_columns].fillna(method='ffill').fillna(method='bfill')
            
            elif method == 'median_fill':
                # 中位数填充
                for column in repaired_data.columns:
                    if repaired_data[column].dtype in [np.float64, np.int64]:
                        median_value = repaired_data[column].median()
                        repaired_data[column] = repaired_data[column].fillna(median_value)
            
            elif method == 'drop':
                # 删除包含缺失值的行
                repaired_data = repaired_data.dropna()
            
            # 记录修复结果
            repair_result = {
                'method': method,
                'original_missing': int(original_missing),
                'repaired_missing': int(repaired_data.isnull().sum().sum()),
                'records_before': len(data),
                'records_after': len(repaired_data),
                'timestamp': datetime.now()
            }
            
            self.repair_history.append({
                'type': 'missing_values',
                'method': method,
                'result': repair_result
            })
            
            logger.info(f"缺失值修复完成: 方法={method}, 修复前缺失={original_missing}, 修复后缺失={repair_result['repaired_missing']}")
            return repaired_data
            
        except Exception as e:
            logger.error(f"缺失值修复失败: {e}")
            return data
    
    def repair_outliers(self, data: pd.DataFrame, method: str = 'winsorize', threshold: float = 0.05) -> pd.DataFrame:
        """修复异常值"""
        try:
            if data.empty:
                return data
            
            repaired_data = data.copy()
            
            numeric_columns = repaired_data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column in ['open', 'high', 'low', 'close', 'volume']:
                    if method == 'winsorize':
                        # Winsorization: 将异常值替换为分位数
                        lower_quantile = repaired_data[column].quantile(threshold)
                        upper_quantile = repaired_data[column].quantile(1 - threshold)
                        
                        repaired_data[column] = np.where(
                            repaired_data[column] < lower_quantile,
                            lower_quantile,
                            np.where(
                                repaired_data[column] > upper_quantile,
                                upper_quantile,
                                repaired_data[column]
                            )
                        )
                    
                    elif method == 'clip':
                        # 简单截断
                        Q1 = repaired_data[column].quantile(0.25)
                        Q3 = repaired_data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        repaired_data[column] = np.clip(repaired_data[column], lower_bound, upper_bound)
                    
                    elif method == 'median_replace':
                        # 用中位数替换异常值
                        Q1 = repaired_data[column].quantile(0.25)
                        Q3 = repaired_data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        median_value = repaired_data[column].median()
                        repaired_data[column] = np.where(
                            (repaired_data[column] < lower_bound) | (repaired_data[column] > upper_bound),
                            median_value,
                            repaired_data[column]
                        )
            
            # 记录修复结果
            repair_result = {
                'method': method,
                'threshold': threshold,
                'columns_repaired': list(numeric_columns),
                'timestamp': datetime.now()
            }
            
            self.repair_history.append({
                'type': 'outliers',
                'method': method,
                'result': repair_result
            })
            
            logger.info(f"异常值修复完成: 方法={method}, 阈值={threshold}")
            return repaired_data
            
        except Exception as e:
            logger.error(f"异常值修复失败: {e}")
            return data
    
    def repair_price_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """修复价格不一致性"""
        try:
            if data.empty:
                return data
            
            repaired_data = data.copy()
            
            # 检查必需的列
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in repaired_data.columns for col in required_columns):
                logger.warning("缺少必需的价格列，跳过价格一致性修复")
                return repaired_data
            
            # 修复价格逻辑不一致
            # 确保 high >= max(open, close) 且 low <= min(open, close)
            repaired_data['high'] = np.maximum(repaired_data['high'], np.maximum(repaired_data['open'], repaired_data['close']))
            repaired_data['low'] = np.minimum(repaired_data['low'], np.minimum(repaired_data['open'], repaired_data['close']))
            
            # 确保 high >= low
            repaired_data['high'] = np.maximum(repaired_data['high'], repaired_data['low'])
            repaired_data['low'] = np.minimum(repaired_data['low'], repaired_data['high'])
            
            # 记录修复结果
            repair_result = {
                'type': 'price_consistency',
                'columns_processed': required_columns,
                'timestamp': datetime.now()
            }
            
            self.repair_history.append({
                'type': 'inconsistencies',
                'method': 'price_logic',
                'result': repair_result
            })
            
            logger.info("价格一致性修复完成")
            return repaired_data
            
        except Exception as e:
            logger.error(f"价格一致性修复失败: {e}")
            return data
    
    def repair_time_series_gaps(self, data: pd.DataFrame, fill_method: str = 'forward_fill') -> pd.DataFrame:
        """修复时间序列缺口"""
        try:
            if data.empty:
                return data
            
            repaired_data = data.copy()
            
            # 确保索引是datetime类型
            if not isinstance(repaired_data.index, pd.DatetimeIndex):
                logger.warning("数据索引不是DatetimeIndex，跳过时间序列修复")
                return repaired_data
            
            # 重新索引以填充缺失的日期
            if len(repaired_data) > 1:
                # 计算频率（假设是交易日）
                date_diff = repaired_data.index.to_series().diff().dropna()
                mode_diff = date_diff.mode()[0] if len(date_diff.mode()) > 0 else pd.Timedelta(days=1)
                
                # 创建完整的日期范围
                full_range = pd.date_range(
                    start=repaired_data.index.min(),
                    end=repaired_data.index.max(),
                    freq=mode_diff
                )
                
                # 重新索引
                repaired_data = repaired_data.reindex(full_range)
                
                # 填充缺失值
                if fill_method == 'forward_fill':
                    repaired_data = repaired_data.fillna(method='ffill')
                elif fill_method == 'interpolation':
                    numeric_columns = repaired_data.select_dtypes(include=[np.number]).columns
                    repaired_data[numeric_columns] = repaired_data[numeric_columns].interpolate(method='linear')
                
                # 如果还有缺失值，用前后填充
                repaired_data = repaired_data.fillna(method='ffill').fillna(method='bfill')
            
            # 记录修复结果
            repair_result = {
                'method': fill_method,
                'original_records': len(data),
                'repaired_records': len(repaired_data),
                'gaps_filled': len(repaired_data) - len(data),
                'timestamp': datetime.now()
            }
            
            self.repair_history.append({
                'type': 'time_series',
                'method': fill_method,
                'result': repair_result
            })
            
            logger.info(f"时间序列缺口修复完成: 方法={fill_method}, 填充记录={repair_result['gaps_filled']}")
            return repaired_data
            
        except Exception as e:
            logger.error(f"时间序列缺口修复失败: {e}")
            return data
    
    def comprehensive_repair(self, data: pd.DataFrame, symbol: str = 'Unknown') -> Dict[str, Any]:
        """综合修复"""
        try:
            if data.empty:
                return {'error': '数据为空'}
            
            logger.info(f"开始综合数据修复: {symbol}")
            
            # 1. 检测问题
            issues = self.detect_data_issues(data, symbol)
            
            if 'error' in issues:
                return issues
            
            # 2. 修复缺失值
            repaired_data = self.repair_missing_values(data, method='interpolation')
            
            # 3. 修复异常值
            repaired_data = self.repair_outliers(repaired_data, method='winsorize')
            
            # 4. 修复价格不一致性
            repaired_data = self.repair_price_inconsistencies(repaired_data)
            
            # 5. 修复时间序列缺口
            repaired_data = self.repair_time_series_gaps(repaired_data)
            
            # 6. 再次检测问题
            final_issues = self.detect_data_issues(repaired_data, symbol)
            
            # 7. 生成修复报告
            repair_report = {
                'symbol': symbol,
                'original_quality_score': issues.get('data_quality_score', 0),
                'final_quality_score': final_issues.get('data_quality_score', 0),
                'improvement': final_issues.get('data_quality_score', 0) - issues.get('data_quality_score', 0),
                'repair_steps': [
                    'missing_values_repair',
                    'outliers_repair',
                    'price_consistency_repair',
                    'time_series_repair'
                ],
                'original_records': len(data),
                'final_records': len(repaired_data),
                'repair_history': self.repair_history[-4:],  # 最近4次修复记录
                'timestamp': datetime.now()
            }
            
            # 保存修复报告
            self._save_repair_report(repair_report, symbol)
            
            logger.info(f"综合数据修复完成: {symbol}, 质量分数提升: {repair_report['improvement']:.2f}")
            
            return {
                'repaired_data': repaired_data,
                'repair_report': repair_report,
                'quality_improved': repair_report['improvement'] > 0
            }
            
        except Exception as e:
            logger.error(f"综合数据修复失败: {e}")
            return {'error': str(e)}
    
    def _save_repair_report(self, report: Dict[str, Any], symbol: str):
        """保存修复报告"""
        try:
            import json
            from datetime import datetime
            
            # 转换datetime对象为字符串
            report_copy = report.copy()
            for key, value in report_copy.items():
                if isinstance(value, datetime):
                    report_copy[key] = value.isoformat()
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, datetime):
                                    value[i][k] = v.isoformat()
            
            # 保存到文件
            report_file = os.path.join(self.repair_log_dir, f"{symbol}_repair_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_copy, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"修复报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存修复报告失败: {e}")
    
    def get_repair_history(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取修复历史"""
        try:
            if symbol:
                # 过滤特定股票的修复历史
                return [h for h in self.repair_history if h.get('symbol') == symbol]
            else:
                return self.repair_history
                
        except Exception as e:
            logger.error(f"获取修复历史失败: {e}")
            return []
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """获取修复统计信息"""
        try:
            if not self.repair_history:
                return {'total_repairs': 0, 'repair_types': {}}
            
            stats = {
                'total_repairs': len(self.repair_history),
                'repair_types': {},
                'recent_repairs': self.repair_history[-10:],  # 最近10次
                'quality_improvements': []
            }
            
            # 统计修复类型
            for repair in self.repair_history:
                repair_type = repair.get('type', 'unknown')
                if repair_type not in stats['repair_types']:
                    stats['repair_types'][repair_type] = 0
                stats['repair_types'][repair_type] += 1
            
            # 计算质量改进
            quality_reports = [r for r in self.repair_history if 'quality_score' in str(r)]
            if quality_reports:
                improvements = [r.get('result', {}).get('improvement', 0) for r in quality_reports[-20:]]
                stats['quality_improvements'] = {
                    'avg_improvement': np.mean(improvements) if improvements else 0,
                    'max_improvement': max(improvements) if improvements else 0,
                    'min_improvement': min(improvements) if improvements else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取修复统计信息失败: {e}")
            return {'total_repairs': 0, 'repair_types': {}}
    
    def reset(self):
        """重置数据修复服务"""
        self.repair_history.clear()
        self.repair_rules.clear()
        logger.info("数据修复服务已重置")