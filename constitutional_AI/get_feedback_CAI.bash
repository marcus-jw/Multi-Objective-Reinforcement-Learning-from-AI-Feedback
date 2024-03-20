data_path="data/datasets/gemma-2b_hh_test"
python API_feedback/create_feedback_api_requests.py \
    --multi-objective=False \
    --principle_path=constitutional_AI/anthropic_constitution.jsonl \
    --feedback_model=gpt-3.5-turbo-0125 \
    --CoT=False \
    --dataset_path="${data_path}_responses.jsonl" \
    --save_path=data/api_requests.jsonl 
        #--few_shot_path=constitutional_AI/anthropic_few_shot_examples.jsonl \
if [ $? -ne 0 ]; then
    echo "create_feedback_api_requests.py failed. Exiting..."
    exit 1
fi

python API_feedback/api_parallelization.py \
    --requests_filepath=data/api_requests.jsonl \
    --save_filepath=data/api_responses.jsonl \
    --request_url=https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute=4900 \
    --max_tokens_per_minute=150000 \
    --max_attempts=5 
if [ $? -ne 0 ]; then
    echo "api_parallelization.py failed. Exiting..."
    exit 1
fi

python API_feedback/process_feedback_api_response.py \
    --response_path=data/api_responses.jsonl \
    --save_path="${data_path}_feedback.jsonl" \
    --dataset_path="${data_path}_responses.jsonl" 
if [ $? -ne 0 ]; then
    echo "process_feedback_api_response.py failed. Exiting..."
    exit 1
fi
# If all scripts executed successfully, delete the files
# rm data/api_responses.jsonl data/api_requests.jsonl
# echo "Temporary files deleted successfully."