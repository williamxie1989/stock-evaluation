#!/bin/bash

# 股票预测API启动脚本

echo "启动股票预测API服务..."
echo "服务将运行在 http://127.0.0.1:8080"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
if ! python3 -c "import flask" &> /dev/null; then
    echo "错误: 未安装Flask，请运行: pip install flask flask-cors"
    exit 1
fi

# 启动服务
cd /Users/xieyongliang/stock-evaluation
python3 src/apps/api/prediction_api.py