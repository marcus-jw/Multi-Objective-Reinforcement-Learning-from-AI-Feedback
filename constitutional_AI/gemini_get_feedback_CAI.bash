data_path="data/datasets/gpt2-medium_hh_train"
python API_feedback/gemini_create_feedback_api_requests.py \
    --multi-objective=False \
    --principle_path=constitutional_AI/anthropic_constitution.jsonl \
    --feedback_model=gemini-1.0-pro \
    --CoT=False \
    --dataset_path="${data_path}_responses.jsonl" \
    --save_path=data/datasets/gemini_api_requests.jsonl 
        #--few_shot_path=constitutional_AI/anthropic_few_shot_examples.jsonl \
if [ $? -ne 0 ]; then
    echo "create_feedback_api_requests.py failed. Exiting..."
    exit 1
fi
python API_feedback/gemini_api_parallelization.py \
    --requests_filepath=data/datasets/gemini_api_requests.jsonl \
    --save_filepath=data/datasets/gemini_api_responses.jsonl \
    --api_endpoint=https://us-central1-aiplatform.googleapis.com \
    --project_id=morlaif \
    --max_requests_per_minute=500 \
    --max_attempts=5 
if [ $? -ne 0 ]; then
    echo "api_parallelization.py failed. Exiting..."
    exit 1
fi

python API_feedback/gemini_process_feedback_api_response.py \
    --response_path=data/datasets/gemini_api_responses.jsonl \
    --save_path="${data_path}_feedback_gemini.jsonl" \
    --dataset_path="${data_path}_responses.jsonl" 
if [ $? -ne 0 ]; then
    echo "process_feedback_api_response.py failed. Exiting..."
    exit 1
fi
# If all scripts executed successfully, delete the files
# rm data/api_responses.jsonl data/api_requests.jsonl
# echo "Temporary files deleted successfully."