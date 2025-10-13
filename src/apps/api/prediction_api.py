#!/usr/bin/env python3
"""
股票预测API服务
提供RESTful API接口用于股票预测
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.ml.prediction.model_predictor import StockPredictor
from src.ml.prediction.enhanced_predictor_v2 import EnhancedPredictorV2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 初始化预测器
predictor = None
predictor_v2 = None

def init_predictor():
    """初始化预测器（同时初始化V1和V2）"""
    global predictor, predictor_v2
    success_v1 = False
    success_v2 = False
    
    # 尝试初始化V2预测器
    try:
        models_v2_dir = project_root / "models" / "v2"
        if models_v2_dir.exists():
            predictor_v2 = EnhancedPredictorV2(models_dir=str(models_v2_dir))
            logger.info("V2预测器初始化成功")
            success_v2 = True
        else:
            logger.warning(f"V2模型目录不存在: {models_v2_dir}")
    except Exception as e:
        logger.warning(f"V2预测器初始化失败: {e}")
    
    # 尝试初始化V1预测器（fallback）
    try:
        models_dir = project_root / "models" / "good"
        if models_dir.exists():
            predictor = StockPredictor(models_dir=str(models_dir))
            logger.info("V1预测器初始化成功")
            success_v1 = True
        else:
            logger.warning(f"V1模型目录不存在: {models_dir}")
    except Exception as e:
        logger.warning(f"V1预测器初始化失败: {e}")
    
    if success_v2 or success_v1:
        logger.info(f"预测器初始化完成 (V2: {success_v2}, V1: {success_v1})")
        return True
    else:
        logger.error("所有预测器初始化失败")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template_string(open('/Users/xieyongliang/stock-evaluation/static/prediction.html').read())

@app.route('/api/predict', methods=['POST'])
def predict():
    """股票预测API（优先使用V2预测器）"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({'error': '缺少股票代码参数'}), 400
        
        symbol = data['symbol'].strip()
        days_back = data.get('days_back', 90)
        use_v2 = data.get('use_v2', True)  # 默认使用V2
        as_of_date = data.get('as_of_date')  # V2支持指定日期
        
        if not symbol:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        logger.info(f"收到预测请求: {symbol}, days_back: {days_back}, use_v2: {use_v2}")
        
        # 优先使用V2预测器
        if use_v2 and predictor_v2 is not None:
            logger.info("使用V2预测器")
            result = predictor_v2.predict(symbol, as_of_date)
            result['predictor_version'] = 'v2'
        elif predictor is not None:
            logger.info("使用V1预测器")
            result = predictor.predict(symbol, days_back)
            result['predictor_version'] = 'v1'
        else:
            return jsonify({'error': '没有可用的预测器'}), 500
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        logger.info(f"预测成功: {symbol} (版本: {result.get('predictor_version')})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测API错误: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'predictor_v1_ready': predictor is not None,
        'predictor_v2_ready': predictor_v2 is not None
    })

@app.route('/api/models')
def models_info():
    """模型信息（支持V1和V2）"""
    try:
        info = {
            'v1_predictor': None,
            'v2_predictor': None
        }
        
        # V1预测器信息
        if predictor is not None:
            info['v1_predictor'] = {
                'regression_model': 'xgboost_regression.pkl',
                'classification_model': 'xgboost_classification.pkl',
                'models_directory': str(predictor.models_dir),
                'feature_generator': 'EnhancedFeatureGenerator'
            }
        
        # V2预测器信息
        if predictor_v2 is not None:
            info['v2_predictor'] = predictor_v2.get_model_info()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测API（仅V2支持）"""
    try:
        data = request.get_json()
        
        if not data or 'symbols' not in data:
            return jsonify({'error': '缺少股票代码列表参数'}), 400
        
        symbols = data['symbols']
        as_of_date = data.get('as_of_date')
        
        if not isinstance(symbols, list) or len(symbols) == 0:
            return jsonify({'error': '股票代码列表格式错误或为空'}), 400
        
        logger.info(f"收到批量预测请求: {len(symbols)} 只股票")
        
        if predictor_v2 is None:
            return jsonify({'error': '批量预测需要V2预测器'}), 500
        
        # 批量预测
        results = predictor_v2.predict_batch(symbols, as_of_date)
        
        return jsonify({
            'total': len(symbols),
            'success': sum(1 for r in results.values() if 'error' not in r),
            'failed': sum(1 for r in results.values() if 'error' in r),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"批量预测API错误: {e}")
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 初始化预测器
    if not init_predictor():
        logger.error("无法初始化预测器，程序退出")
        sys.exit(1)
    
    logger.info("启动股票预测API服务...")
    app.run(host='0.0.0.0', port=8080, debug=False)