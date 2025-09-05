import json
import requests

# 发送API请求获取实际数据
url = "http://localhost:8001/api/analyze"
headers = {"Content-Type": "application/json"}
data = {"symbol": "600036.SS"}

response = requests.post(url, headers=headers, json=data)
api_response = response.json()

# 保存API响应到文件以便分析
with open("api_response.json", "w") as f:
    json.dump(api_response, f, indent=2, ensure_ascii=False)

print("API响应已保存到api_response.json文件中")
print(f"响应状态: {api_response.get('success', False)}")
print(f"信号数量: {len(api_response.get('signals', []))}")
print(f"最新价格: {api_response.get('latest_price', 0)}")

# 检查信号数据
signals = api_response.get('signals', [])
if signals:
    print("\n信号数据预览:")
    for i, signal in enumerate(signals[:3]):  # 只显示前3个信号
        print(f"  信号 {i+1}: {signal}")