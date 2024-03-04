data_path="data/gpt2_XL__hh_train"

python create_feedback_api_requests.py \
    --multi-objective=True \
    --principle_folder=principles \
    --principle_name=toxicity \
    --feedback_model=gpt-3.5-turbo-0125 \
    --dataset_path="${data_path}_responses.jsonl" \
    --save_path=data/api_requests.jsonl 
python api_parallelization.py \
    --requests_filepath=data/api_requests.jsonl \
    --save_filepath=data/api_responses \
    --request_url=https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute=4000 \
    --max_tokens_per_minute=120000 \
    --max_attempts=5 
python process_feedback_api_response.py \
    --response_path=data/api_responses \
    --save_path="${data_path}_feedback.jsonl" \
    --dataset_path="${data_path}_responses.jsonl" \