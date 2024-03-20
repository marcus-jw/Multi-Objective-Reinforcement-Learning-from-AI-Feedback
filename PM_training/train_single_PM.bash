principle=sycophancy
echo "Training $principle PM"
accelerate launch --config_file accelerate.yaml train_PM.py \
    --principle="$principle" \
    --model_name="gpt2-medium" \
    --output_dir="gpt2-med-CAI" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=2e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
    --num_proc=4 
done
