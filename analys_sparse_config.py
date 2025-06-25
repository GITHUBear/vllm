import json

with open("./sparse_attention_config.json") as f:
    configs = json.load(f)

    # print(len(configs))
    vs_set = set()
    verticals = []
    slashes = []
    for cfg in configs:
        for head_cfg in cfg.values():
            vertical, slash = head_cfg[1], head_cfg[2]
            vs_set.add((vertical, slash))
            verticals.append(vertical)
            slashes.append(slash)
    print(vs_set)
    print(f"vertical: [{min(verticals)}, {max(verticals)}]")
    print(f"slash: [{min(slashes)}, {max(slashes)}]")
