data_path="data/datasets/gpt2_XL_hh_train"

python API_feedback/create_feedback_api_requests.py \
    --multi-objective=True \
    --principle_folder=principles \
    --principle_name=toxicity \
    --feedback_model=gpt-3.5-turbo-0125 \
    --dataset_path="${data_path}_responses.jsonl" \
    --save_path=data/datasets/api_requests.jsonl 
python API_feedback/api_parallelization.py \
    --requests_filepath=data/datasets/api_requests.jsonl \
    --save_filepath=data/datasets/api_responses \
    --request_url=https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute=4000 \
    --max_tokens_per_minute=120000 \
    --max_attempts=5 
python API_feedback/process_feedback_api_response.py \
    --response_path=data/datasets/api_responses \
    --save_path="${data_path}_feedback.jsonl" \
    --dataset_path="${data_path}_responses.jsonl" \