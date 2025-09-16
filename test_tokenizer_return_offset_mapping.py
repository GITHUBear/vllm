from transformers import AutoTokenizer
sep = "<|DOC_SEP|>"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", return_offsets_mapping=True)
tokenizer.add_tokens([sep])
sep_token_id = tokenizer.convert_tokens_to_ids(sep)
print(sep_token_id)

text = "<|DOC_SEP|>文档1:aaaa<|DOC_SEP|>文档2: bbbb<|DOC_SEP|>"
inputs = tokenizer(text, return_offsets_mapping=True)

# 获取 token 列表
# tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

# 获取偏移映射
offset_mapping = inputs["offset_mapping"]

# 打印映射关系
for token_id, (start, end) in zip(inputs["input_ids"], offset_mapping):
    print(f"Token: {token_id} | Chars: [{start}:{end}] | Text: '{text[start:end]}'")