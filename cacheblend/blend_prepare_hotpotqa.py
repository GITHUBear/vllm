import json
import requests
from tqdm import tqdm

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
        "max_tokens": 1,
    }
    response = requests.post(API_ENDPOINT, json=data, headers=headers)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)

with open('/data/shanhaikang.shk/vllm/cacheblend/hotpotqa_sample.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
    num_docs = 0
    for test_case in tqdm(datas.values(), desc="Processing test cases", total=len(datas)):
        case_prompt = test_case["origin_prompt"]
        ss, es, docs = format_prompt(case_prompt)
        num_docs += len(docs)
        for doc in tqdm(docs, desc=f"Generating passages", leave=False, total=len(docs)):
            input_prompt = ss + "Passage 1:\n<|DOC_SEP|>" + doc + "<|DOC_SEP|>" + es
            call_llm_gen(input_prompt)
    print(f"================= Finish {num_docs} docs ============")

    # test_case = datas["1"]
    # case_prompt = test_case["origin_prompt"]
    # ss, es, docs = format_prompt(case_prompt)
    # for doc in tqdm(docs, desc=f"Generating passages", leave=False, total=len(docs)):
    #     prompt = ss + "Passage 1:\n<|DOC_SEP|>" + doc + "<|DOC_SEP|>" + es
    #     call_llm_gen(prompt)
    
    # [(36, 1561)]
    # [(36, 326)]
    # [(36, 8874)]
    # [(36, 420)]
    # [(36, 2088)]
    # [(36, 3526)]
    # [(36, 616)]