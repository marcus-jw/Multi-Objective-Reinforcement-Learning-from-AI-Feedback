base_model_name="gpt2-medium"
base_model_folder=""
PM_model_name="gpt2-medium"
principle="CAI"
#accelerate launch --config_file accelerate.yaml PPO_training/PPO_training.py \
python PPO_training/PPO_training.py \
    --model_name="${base_model_folder}${base_model_name}" \
    --PM_path="data/PMs/${PM_model_name}_${principle}/final" \
    --save_path="data/trained_models/${base_model_name}_${principle}_LoRA" \
    --dataset_path="data/datasets/hh-rlhf-train-extracted.jsonl" \
    --num_proc=4 \
    --mini_batch_size=4 \
    --batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --max_length=512 \
    --remove_unused_columns=False \
    --output_min_length=8 \
    --output_max_length=128 \
    --LoRA=True \
    --LoRA_r=16 \
    --LoRA_alpha=32 \
    --LoRA_dropout=0.1 \
    --save_interval=0.2

    