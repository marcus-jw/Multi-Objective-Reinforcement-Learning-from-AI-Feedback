base_model_name="gpt2-medium"
base_model_folder=""
PM_model_name="gpt2-medium"
principle="CAI"
#accelerate launch --config_file accelerate.yaml PPO_training/PPO_training.py \
python PPO_training/PPO_training.py \
    --train.epochs 10000 \
    --train.batch_size 4 \
    --train.minibatch_size 1 \
    --train.seq_length 512 \
    --train.checkpoint_interval 20000 \
    --train.total_steps 10000 \
    --train.eval_interval 10000 \
    --model.model_path "${base_model_folder}${base_model_name}" \
    --tokenizer.tokenizer_path "${base_model_folder}${base_model_name}" \
    --PM_path "data/PMs/${PM_model_name}_${principle}/final" \
    --training_set_path "data/datasets/hh-rlhf-train-extracted.jsonl" \
    --test_set_path "data/datasets/hh-rlhf-test-extracted.jsonl" \
    --train.checkpoint_dir "data/trained_models/${base_model_name}_${principle}" \
    --reward_batch_size 2 



