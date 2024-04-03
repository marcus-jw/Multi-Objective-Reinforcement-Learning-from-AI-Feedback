import json
with open(f'data/datasets/hh-rlhf-train-extracted.jsonl',"r",encoding="utf-8") as file:
    data = [json.loads(line) for line in file]
count = 0
for line in data:
    if line["chosen"] == line["rejected"]:
        count += 1
print("train",count)

with open(f'data/datasets/hh-rlhf-test-extracted.jsonl',"r",encoding="utf-8") as file:
    data = [json.loads(line) for line in file]
count = 0
for line in data:
    if line["chosen"] == line["rejected"]:
        count += 1
print("test",count)