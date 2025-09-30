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

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.ml.prediction.model_predictor import StockPredictor

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

def init_predictor():
    """初始化预测器"""
    global predictor
    try:
        # 初始化预测器
        models_dir = project_root / "models" / "good"
        predictor = StockPredictor(models_dir=str(models_dir))
        logger.info("预测器初始化成功")
        return True
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template_string(open('/Users/xieyongliang/stock-evaluation/static/prediction.html').read())

@app.route('/api/predict', methods=['POST'])
def predict():
    """股票预测API"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({'error': '缺少股票代码参数'}), 400
        
        symbol = data['symbol'].strip()
        days_back = data.get('days_back', 90)
        
        if not symbol:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        logger.info(f"收到预测请求: {symbol}, days_back: {days_back}")
        
        # 进行预测
        result = predictor.predict(symbol, days_back)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        logger.info(f"预测成功: {symbol}")
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
        'predictor_ready': predictor is not None
    })

@app.route('/api/models')
def models_info():
    """模型信息"""
    if predictor is None:
        return jsonify({'error': '预测器未初始化'}), 500
    
    try:
        info = {
            'regression_model': 'xgboost_regression.pkl',
            'classification_model': 'xgboost_classification.pkl',
            'models_directory': str(predictor.models_dir),
            'feature_generator': 'EnhancedFeatureGenerator',
            'last_prediction': None
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 初始化预测器
    if not init_predictor():
        logger.error("无法初始化预测器，程序退出")
        sys.exit(1)
    
    logger.info("启动股票预测API服务...")
    app.run(host='0.0.0.0', port=8080, debug=False)