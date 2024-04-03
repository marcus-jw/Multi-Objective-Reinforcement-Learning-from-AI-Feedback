data_path="data/datasets/hh-rlhf-train-extracted"
principle="helpfulness"
echo $principle
python API_feedback/create_feedback_api_requests.py \
    --multi-objective=True \
    --principle_folder=principles \
    --principle_name="${principle}" \
    --feedback_model=gpt-3.5-turbo-0125 \
    --dataset_path="${data_path}.jsonl" \
    --save_path=data/datasets/api_requests.jsonl 
    #--dataset_path="${data_path}_responses.jsonl"
if [ $? -ne 0 ]; then
    echo "create_feedback_api_requests.py failed. Exiting..."
    exit 1
fi
python API_feedback/api_parallelization.py \
    --requests_filepath=data/datasets/api_requests.jsonl \
    --save_filepath=data/datasets/api_responses.jsonl \
    --request_url=https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute=9000 \
    --max_tokens_per_minute=900000 \
    --max_attempts=5 
if [ $? -ne 0 ]; then
    echo "api_parallelization.py failed. Exiting..."
    exit 1
fi
python API_feedback/process_feedback_api_response.py \
    --response_path=data/datasets/api_responses.jsonl \
    --save_path="data/datasets/hh_train_${principle}_feedback.jsonl" \
    --dataset_path="${data_path}.jsonl" 
if [ $? -ne 0 ]; then
    echo "process_feedback_api_response.py failed. Exiting..."
    exit 1
fi
# If all scripts executed successfully, delete the files
# rm data/datasets/api_responses.jsonl data/datasets/api_requests.jsonl
# echo "Temporary files deleted successfully."