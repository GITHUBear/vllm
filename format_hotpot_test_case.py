import sys

prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words."

for line in sys.stdin:
    line = line.strip()
    last_pos = line.rfind(prompt)
    split_idxs = []
    for i in range(1, 15):
        title = f"Passage {i}:\\n"
        title_pos = line.find(title)
        # print(title_pos)
        if title_pos == -1:
            break
        split_idxs.append(title_pos)
    
    split_idxs.append(last_pos)
    # print(split_idxs)

    new_prompt = line[:split_idxs[0]]
    new_prompt += "<|DOC_SEP|>"
    end_str = line[split_idxs[-1]:]
    for idx in range(0, len(split_idxs) - 1):
        title = f"Passage {idx + 1}:\\n"
        new_prompt += title
        new_prompt += line[split_idxs[idx] + len(title):split_idxs[idx+1]]
        new_prompt += "<|DOC_SEP|>"
    new_prompt += end_str
    
    print(new_prompt)
