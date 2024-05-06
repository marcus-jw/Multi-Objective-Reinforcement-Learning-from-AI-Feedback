base_model_name="gpt2-medium"
base_model_folder=""
PM_model_name="gpt2-medium"
accelerate launch --config_file accelerate.yaml PPO_training/PPO_training.py \
    --train.epochs 1256 \
    --train.batch_size 128 \
    --train.minibatch_size 2 \
    --train.seq_length 512 \
    --train.checkpoint_interval 500 \
    --train.total_steps 1256 \
    --train.eval_interval 500 \
    --model.model_path "${base_model_folder}${base_model_name}" \
    --tokenizer.tokenizer_path "${base_model_folder}${base_model_name}" \
    --PM_path "data/PM_LoRAs/${PM_model_name}" \
    --training_set_path "data/datasets/hh-rlhf-train-extracted.jsonl" \
    --test_set_path "data/datasets/hh-rlhf-test-extracted.jsonl" \
    --train.checkpoint_dir "data/trained_models/${base_model_name}_5" \
    --reward_batch_size 4 \
    --MORL True \
    --PMs "ethicality,factuality,helpfulness,sycophancy,toxicity" \
    --scalarizer "linear" \
    --weight_file "data/weights/linear_5.json" 