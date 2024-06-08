from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

dataset = load_dataset("mohitpg/openassistant-guanaco-english", split="train")
def replace_text(example):
    for key, value in example.items():
        if isinstance(value, str):
            value = value.replace("[INST]", "Human:")
            example[key] = value.replace("[/INST]", "\nAssistant:")
    return example
dataset = dataset.map(replace_text)

training_args = TrainingArguments(
    output_dir="data/SFT_models",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64
)
model_name="gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
instruction_template = "Human:"
response_template = "Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=1024,
    args=training_args,
)

trainer.train()
trainer.save_model("data/SFT_models/" + model_name)