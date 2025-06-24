import requests
import json

# API 配置
API_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "your-api-key"

# 构造请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 构造请求体
data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "Deepseek中有几个e?"}
    ],
    "temperature": 0,
    "stream": True
}

# 发起 POST 请求，开启流式接收
try:
    with requests.post(API_ENDPOINT, headers=headers, json=data, stream=True) as response:
        # 检查响应状态
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)
            exit(1)

        # 实时读取并处理响应流
        for line in response.iter_lines():
            if line:
                # 解析每一行数据
                line_str = line.decode('utf-8')
                if line_str.startswith("data: [DONE]"):
                    break
                if line_str.startswith("data: "):
                    try:
                        data_json = json.loads(line_str[6:])
                        content = data_json.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if content:
                            print(content, end='', flush=True)
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e}")
except requests.exceptions.RequestException as e:
    print(f"请求异常: {e}")
