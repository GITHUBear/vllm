import requests
import json

API_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
question = "Let $ABCD$ be a tetrahedron such that $AB=CD= \\sqrt{41}$, $AC=BD= \\sqrt{80}$, and $BC=AD= \\sqrt{89}$. There exists a point $I$ inside the tetrahedron such that the distances from $I$ to each of the faces of the tetrahedron are all equal. This distance can be written in the form $\\frac{m \\sqrt n}{p}$, where $m$, $n$, and $p$ are positive integers, $m$ and $p$ are relatively prime, and $n$ is not divisible by the square of any prime. Find $m+n+p$."

data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": prompt_template.format(question=question)}
    ],
    "temperature": 0,
    "stream": True,
    "ignore_eos": True,
    "max_tokens": 65536,
}

try:
    with requests.post(API_ENDPOINT, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)
            exit(1)

        for line in response.iter_lines():
            if line:
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
