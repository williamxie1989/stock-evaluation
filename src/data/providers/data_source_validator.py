"""
数据源验证工具 - 验证和评估不同数据源的可靠性
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .unified_data_provider import UnifiedDataProvider
from .akshare_provider import AkshareDataProvider
from .enhanced_realtime_provider import EnhancedRealtimeProvider
from .optimized_enhanced_data_provider import OptimizedEnhancedDataProvider
from .domestic.eastmoney_provider import EastmoneyDataProvider
from .domestic.tencent_provider import TencentDataProvider
from .domestic.netease_provider import NeteaseDataProvider

logger = logging.getLogger(__name__)


class DataSourceValidator:
    """数据源验证器"""
    
    def __init__(self, unified_provider: Optional[UnifiedDataProvider] = None):
        """初始化数据源验证器"""
        self.unified_provider = unified_provider or UnifiedDataProvider()
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
        # 测试股票池 - 包含不同市场和板块的代表性股票
        self.test_symbols = [
            # 主板蓝筹股
            '600519.SH',  # 贵州茅台
            '000858.SZ',  # 五粮液
            '000001.SZ',  # 平安银行
            
            # 科技股
            '300750.SZ',  # 宁德时代
            '002415.SZ',  # 海康威视
            '300059.SZ',  # 东方财富
            
            # 小盘股
            '300001.SZ',  # 特锐德
            '600570.SH',  # 恒生电子
            
            # 不同行业
            '600036.SH',  # 招商银行
            '000002.SZ',  # 万科A
            '600887.SH',  # 伊利股份
        ]
        
        logger.info(f"DataSourceValidator initialized with {len(self.test_symbols)} test symbols")
    
    def validate_all_sources(self) -> Dict[str, Any]:
        """验证所有可用的数据源"""
        logger.info("Starting validation of all data sources...")
        
        # 创建各种数据提供者
        providers = {
            'akshare': AkshareDataProvider(),
            'enhanced_realtime': EnhancedRealtimeProvider(),
            'optimized_enhanced': OptimizedEnhancedDataProvider(),
            'eastmoney': EastmoneyDataProvider(),
            'tencent': TencentDataProvider(),
            'netease': NeteaseDataProvider()
        }
        
        results = {}
        
        for provider_name, provider in providers.items():
            try:
                logger.info(f"Validating {provider_name}...")
                
                # 添加到统一提供者进行验证
                self.unified_provider.add_primary_provider(provider)
                
                # 执行验证
                validation_result = self.unified_provider.validate_data_source(provider, self.test_symbols)
                results[provider_name] = validation_result
                
                # 移除提供者以避免干扰后续验证
                if provider in self.unified_provider.primary_providers:
                    self.unified_provider.primary_providers.remove(provider)
                
                logger.info(f"Validation completed for {provider_name}: score={validation_result['overall_score']:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to validate {provider_name}: {e}")
                results[provider_name] = {
                    'provider': provider_name,
                    'overall_score': 0.0,
                    'error': str(e),
                    'recommendation': 'not_recommended'
                }
        
        # 生成综合报告
        comprehensive_report = self._generate_comprehensive_report(results)
        
        return {
            'individual_results': results,
            'comprehensive_report': comprehensive_report,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_comprehensive_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成综合报告"""
        scores = [result['overall_score'] for result in results.values()]
        
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_sources_tested': len(results),
            'average_score': np.mean(scores) if scores else 0.0,
            'best_performer': max(results.items(), key=lambda x: x[1]['overall_score'])[0] if results else None,
            'worst_performer': min(results.items(), key=lambda x: x[1]['overall_score'])[0] if results else None,
            'score_distribution': {
                'excellent (>=0.8)': sum(1 for score in scores if score >= 0.8),
                'good (0.6-0.8)': sum(1 for score in scores if 0.6 <= score < 0.8),
                'acceptable (0.4-0.6)': sum(1 for score in scores if 0.4 <= score < 0.6),
                'poor (<0.4)': sum(1 for score in scores if score < 0.4)
            },
            'detailed_scores': {name: result['overall_score'] for name, result in results.items()}
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成推荐建议"""
        recommendations = []
        
        # 按评分排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        for provider_name, result in sorted_results:
            score = result['overall_score']
            recommendation = result['recommendation']
            
            rec = {
                'provider': provider_name,
                'score': score,
                'recommendation': recommendation,
                'priority': self._get_priority(recommendation),
                'use_case': self._get_use_case(recommendation),
                'notes': self._get_notes(provider_name, result)
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _get_priority(self, recommendation: str) -> int:
        """获取优先级"""
        priority_map = {
            'highly_recommended': 1,
            'recommended': 2,
            'acceptable': 3,
            'not_recommended': 4
        }
        return priority_map.get(recommendation, 5)
    
    def _get_use_case(self, recommendation: str) -> str:
        """获取使用场景"""
        use_case_map = {
            'highly_recommended': '主数据源，优先使用',
            'recommended': '主数据源，可靠使用',
            'acceptable': '备用数据源，降级使用',
            'not_recommended': '不建议使用'
        }
        return use_case_map.get(recommendation, '未知')
    
    def _get_notes(self, provider_name: str, result: Dict[str, Any]) -> str:
        """获取备注信息"""
        notes = []
        
        # 检查历史数据质量
        historical_tests = result.get('historical_tests', {})
        if historical_tests:
            avg_historical_score = np.mean([
                test.get('quality_score', 0) 
                for test in historical_tests.values() 
                if test.get('success', False)
            ])
            if avg_historical_score < 0.6:
                notes.append("历史数据质量较低")
        
        # 检查实时数据质量
        realtime_tests = result.get('realtime_tests', {})
        if realtime_tests:
            avg_realtime_score = np.mean([
                test.get('quality_score', 0) 
                for test in realtime_tests.values() 
                if test.get('success', False)
            ])
            if avg_realtime_score < 0.6:
                notes.append("实时数据质量较低")
        
        # 基于提供者类型的特定备注
        if provider_name == 'akshare':
            notes.append("数据源丰富，但稳定性依赖akshare库")
        elif provider_name == 'enhanced_realtime':
            notes.append("多数据源支持，适合实时数据")
        elif provider_name == 'optimized_enhanced':
            notes.append("优化版本，具备熔断和重试机制")
        elif provider_name == 'eastmoney':
            notes.append("东财数据接口，覆盖面广，速度较快")
        elif provider_name == 'tencent':
            notes.append("腾讯行情接口，适合分钟级实时数据")
        elif provider_name == 'netease':
            notes.append("网易历史数据接口，补全数据范围")
        
        return "; ".join(notes) if notes else "无特殊备注"
    
    def compare_sources_performance(self, symbols: List[str] = None, 
                                  days: int = 30) -> Dict[str, Any]:
        """比较不同数据源的性能"""
        if symbols is None:
            symbols = self.test_symbols[:3]  # 使用部分测试股票
        
        logger.info(f"Comparing data source performance for {len(symbols)} symbols over {days} days...")
        
        providers = {
            'akshare': AkshareDataProvider(),
            'enhanced_realtime': EnhancedRealtimeProvider(),
            'optimized_enhanced': OptimizedEnhancedDataProvider(),
            'eastmoney': EastmoneyDataProvider(),
            'tencent': TencentDataProvider(),
            'netease': NeteaseDataProvider()
        }
        
        performance_results = {}
        
        for provider_name, provider in providers.items():
            provider_performance = {
                'historical_data': {},
                'realtime_data': {},
                'overall_metrics': {}
            }
            
            try:
                # 测试历史数据获取性能
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                historical_times = []
                historical_successes = 0
                
                for symbol in symbols:
                    try:
                        start_time = time.time()
                        data = provider.get_stock_data(symbol, start_date, end_date)
                        end_time = time.time()
                        
                        if data is not None and not data.empty:
                            historical_times.append(end_time - start_time)
                            historical_successes += 1
                            
                    except Exception as e:
                        logger.debug(f"Historical data failed for {symbol} with {provider_name}: {e}")
                
                if historical_times:
                    provider_performance['historical_data'] = {
                        'avg_time': np.mean(historical_times),
                        'max_time': np.max(historical_times),
                        'min_time': np.min(historical_times),
                        'success_rate': historical_successes / len(symbols),
                        'total_requests': len(symbols),
                        'successful_requests': historical_successes
                    }
                
                # 测试实时数据获取性能
                realtime_times = []
                realtime_successes = 0
                
                if hasattr(provider, 'get_realtime_data'):
                    try:
                        start_time = time.time()
                        data = provider.get_realtime_data(symbols)
                        end_time = time.time()
                        
                        if data:
                            realtime_times.append(end_time - start_time)
                            realtime_successes = len(data)
                            
                    except Exception as e:
                        logger.debug(f"Realtime data failed with {provider_name}: {e}")
                
                if realtime_times:
                    provider_performance['realtime_data'] = {
                        'avg_time': np.mean(realtime_times),
                        'max_time': np.max(realtime_times),
                        'min_time': np.min(realtime_times),
                        'success_rate': realtime_successes / len(symbols),
                        'total_symbols': len(symbols),
                        'successful_symbols': realtime_successes
                    }
                
                # 计算总体指标
                all_times = historical_times + realtime_times
                if all_times:
                    provider_performance['overall_metrics'] = {
                        'avg_response_time': np.mean(all_times),
                        'total_success_rate': (historical_successes + realtime_successes) / (len(symbols) * 2),
                        'reliability_score': self._calculate_reliability_score(provider_performance)
                    }
                
                performance_results[provider_name] = provider_performance
                
            except Exception as e:
                logger.error(f"Failed to test performance for {provider_name}: {e}")
                performance_results[provider_name] = {
                    'error': str(e),
                    'overall_metrics': {'reliability_score': 0.0}
                }
        
        return {
            'performance_results': performance_results,
            'best_performer': self._get_best_performer(performance_results),
            'comparison_summary': self._generate_performance_summary(performance_results)
        }
    
    def _calculate_reliability_score(self, performance: Dict[str, Any]) -> float:
        """计算可靠性评分"""
        score = 0.0
        factors = 0
        
        # 历史数据成功率
        hist_data = performance.get('historical_data', {})
        if hist_data.get('success_rate') is not None:
            score += hist_data['success_rate'] * 0.4
            factors += 0.4
        
        # 实时数据成功率
        realtime_data = performance.get('realtime_data', {})
        if realtime_data.get('success_rate') is not None:
            score += realtime_data['success_rate'] * 0.4
            factors += 0.4
        
        # 响应时间评分（基于平均响应时间）
        all_times = []
        if hist_data.get('avg_time'):
            all_times.append(hist_data['avg_time'])
        if realtime_data.get('avg_time'):
            all_times.append(realtime_data['avg_time'])
        
        if all_times:
            avg_time = np.mean(all_times)
            # 响应时间评分：小于1秒得1分，1-3秒得0.8分，3-5秒得0.6分，超过5秒得0.3分
            if avg_time < 1:
                time_score = 1.0
            elif avg_time < 3:
                time_score = 0.8
            elif avg_time < 5:
                time_score = 0.6
            else:
                time_score = 0.3
            
            score += time_score * 0.2
            factors += 0.2
        
        return score / factors if factors > 0 else 0.0
    
    def _get_best_performer(self, performance_results: Dict[str, Any]) -> Optional[str]:
        """获取最佳表现者"""
        best_provider = None
        best_score = 0.0
        
        for provider_name, performance in performance_results.items():
            if 'error' in performance:
                continue
                
            score = performance.get('overall_metrics', {}).get('reliability_score', 0.0)
            if score > best_score:
                best_score = score
                best_provider = provider_name
        
        return best_provider
    
    def _generate_performance_summary(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能比较摘要"""
        valid_results = {k: v for k, v in performance_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid performance data available'}
        
        scores = [v.get('overall_metrics', {}).get('reliability_score', 0.0) for v in valid_results.values()]
        
        return {
            'providers_tested': len(valid_results),
            'average_reliability_score': np.mean(scores),
            'best_reliability_score': np.max(scores),
            'worst_reliability_score': np.min(scores),
            'ranking': sorted(valid_results.items(), 
                            key=lambda x: x[1].get('overall_metrics', {}).get('reliability_score', 0.0), 
                            reverse=True)
        }
    
    def save_validation_report(self, validation_results: Dict[str, Any], 
                             filename: Optional[str] = None) -> str:
        """保存验证报告"""
        if filename is None:
            filename = f"data_source_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Validation report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            raise
    
    def print_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """打印验证摘要"""
        print("\n" + "="*80)
        print("数据源验证报告摘要")
        print("="*80)
        
        # 综合报告
        comprehensive = validation_results.get('comprehensive_report', {})
        print(f"\n验证日期: {comprehensive.get('validation_date', 'N/A')}")
        print(f"测试数据源数量: {comprehensive.get('total_sources_tested', 0)}")
        print(f"平均评分: {comprehensive.get('average_score', 0.0):.2f}")
        
        if comprehensive.get('best_performer'):
            print(f"最佳表现: {comprehensive['best_performer']}")
        if comprehensive.get('worst_performer'):
            print(f"最差表现: {comprehensive['worst_performer']}")
        
        # 评分分布
        distribution = comprehensive.get('score_distribution', {})
        print(f"\n评分分布:")
        for category, count in distribution.items():
            print(f"  {category}: {count}")
        
        # 推荐建议
        recommendations = validation_results.get('recommendations', [])
        if recommendations:
            print(f"\n推荐建议:")
            print("-"*60)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['provider']} (评分: {rec['score']:.2f})")
                print(f"   推荐等级: {rec['recommendation']}")
                print(f"   使用场景: {rec['use_case']}")
                if rec['notes']:
                    print(f"   备注: {rec['notes']}")
                print()
        
        print("="*80)