import json
from tqdm import tqdm

def create_preference_data(in_filepath, out_filepath):
    with open(in_filepath, 'r',encoding='utf-8') as in_file, open(out_filepath, 'w',encoding='utf-8') as out_file:
        dataset = in_file.readlines()

        for line in tqdm(dataset):
            line = json.loads(line)
            d={"prompt": line["prompt"],
                    "responseA": line["chosen"],
                    "responseB": line["rejected"],
                    "logits_A": -0.5108256,
                    "logits_B": -0.9162907}
            out_file.writelines(json.dumps(d) + '\n')

create_preference_data("data/datasets/hh-rlhf-train-extracted.jsonl", "data/datasets/_hh_train_feedback.jsonl")
create_preference_data("data/datasets/hh-rlhf-test-extracted.jsonl", "data/datasets/_hh_test_feedback.jsonl")