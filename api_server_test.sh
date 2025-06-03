API_ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
API_KEY="your-api-key"

curl -s $API_ENDPOINT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-1M",
    "messages": [
      {"role": "system", "content": "你是一个乐于助人的AI助手。"},
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0
  }' | jq