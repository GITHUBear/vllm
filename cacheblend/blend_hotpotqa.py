import json
import requests
# from tqdm import tqdm

def format_prompt(case_prompt: str):
    prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    last_pos = case_prompt.rfind(prompt)
    split_idxs = []
    for i in range(1, 15):
        title = f"Passage {i}:\n"
        title_pos = case_prompt.find(title)
        # print(title_pos)
        if title_pos == -1:
            break
        split_idxs.append(title_pos)
    split_idxs.append(last_pos)

    start_str = case_prompt[:split_idxs[0]]
    end_str = case_prompt[split_idxs[-1]:]
    docs = []
    for idx in range(0, len(split_idxs) - 1):
        title = f"Passage {idx + 1}:\n"
        docs.append(case_prompt[split_idxs[idx] + len(title):split_idxs[idx+1]])
    return start_str, end_str, docs

API_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def call_llm_gen(prompt):
    data = {
        "model": "/data/shanhaikang.shk/modelscope/qwq32b",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "stream": True,
    }
    try:
        with requests.post(API_ENDPOINT, json=data, headers=headers, stream=True) as response:
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

with open('./hotpotqa_sample.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
    test_case = datas["0"]
    case_prompt = test_case["origin_prompt"]
    ss, es, docs = format_prompt(case_prompt)
    input_prompt = ss
    for idx, doc in enumerate(docs):
        input_prompt += f"Passage {idx + 1}\n<|DOC_SEP|>"
        input_prompt += doc
        input_prompt += "<|DOC_SEP|>"
    input_prompt += es

call_llm_gen(input_prompt)
        # exit(0)