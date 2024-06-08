import os
import json
from openai import OpenAI
client = OpenAI()
path = "data/api_requests/"
files = os.listdir(path)
principle = "bias"
batches = [path + file for file in files if file.startswith(principle)]

ids = []
for index,batch in enumerate(batches):
    print(f"Creating batch {index}")
    batch_input_file = client.files.create(
        file=open(batch, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    ids.append(batch_input_file_id)
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{principle}: batch {index} "
        }
    )
with open(f"data/batch_ids_{principle}.txt", "w") as file:
    file.write(json.dumps(ids))