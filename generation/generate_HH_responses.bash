model_name="gpt2-medium"
model_folder=""
accelerate launch --config_file accelerate.yaml generate_HH_responses.py \
--model_name="$model_folder$model_name" \
--num_proc=4 \
--batch_size=1 \
--dataset_path="data/hh-rlhf-test-extracted.jsonl" \
--output_path="data/${model_name}_hh_test_responses.jsonl" \
--start_at=0