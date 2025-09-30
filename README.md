# 智能股票分析系统

基于大模型的股票技术分析与投资建议系统，能够自动获取股票数据并生成专业的买卖点建议。支持A股及主要指数分析。

## 功能特点

- 📊 自动获取股票及大盘近一年的股价、成交额数据
- 🤖 集成OpenAI大模型进行智能分析
- 📈 多种技术指标计算（移动平均线、MACD、RSI、布林带等）
- ⚡ 实时交易信号生成
- 🌐 现代化的Web界面
- 📱 响应式设计，支持移动端
- 🔍 支持AKShare数据源
- 📉 内置回测引擎
- ⚖️ 风险管理模块

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

1. 复制环境变量文件：
```bash
cp .env.example .env
```

2. 在 `.env` 文件中配置您的API密钥：
```env
# OpenAI配置
OPENAI_API_KEY=sk-your-openai-api-key-here

# 本地大模型配置
LOCAL_MODEL_URL=http://your-local-model-url
LOCAL_MODEL_API_KEY=your-local-api-key
LOCAL_MODEL_NAME=your-model-name
```

注意：可以只配置OpenAI或本地大模型其中一种，优先使用本地大模型配置。

## 运行程序

### 开发模式
```bash
python app.py
```

### 生产模式
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000 即可使用系统。

## 使用方法

1. 在输入框中输入股票代码（如：600036.SS 招商银行）
2. 点击"分析股票"按钮
3. 系统将显示：
   - 股票基本信息
   - 技术指标分析
   - AI投资建议
   - 近期交易信号

## 支持的股票代码格式

- A股: 600036.SS (上海), 000001.SZ (深圳)
- 指数: 000001.SS (上证指数), 399001.SZ (深证成指)
- A股: 000001 (平安银行)
- 美股: AAPL (苹果)

## 技术架构

- 后端: FastAPI + Python
- 前端: HTML + Tailwind CSS + Chart.js
- 数据: yfinance (雅虎财经数据) + AKShare
- AI: OpenAI GPT-3.5/4
- 技术指标: TA-Lib
- 回测引擎: 自定义实现
- 风险管理: 自定义模块

## 注意事项

1. 请确保已安装TA-Lib库（可能需要单独安装）
2. 股票数据来源于雅虎财经，可能存在延迟
3. AI分析结果仅供参考，投资需谨慎
4. 请合理使用OpenAI API，避免超额费用

## 开发计划

- [x] 添加更多技术指标
- [x] 支持自定义分析参数
- [x] 添加历史回测功能
- [x] 集成更多数据源
- [ ] 添加多因子分析
- [ ] 支持策略优化

## 许可证

MIT License