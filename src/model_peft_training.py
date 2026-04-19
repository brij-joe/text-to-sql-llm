import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Configuration
load_dotenv()
hf_token = os.environ.get("HF_TOKEN") # Note: Ensure your .env has HF_TOKEN=...
print(f"HF_TOKEN: {hf_token}")
model_id = "google/gemma-3-1b-it"
dataset_path = "sql_tuning_data.jsonl"

# 2. Load Dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 3. BitsAndBytes for 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, # Recommended for QLoRA
)

# 4. Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Recommended for CausalLM training

# 5. LoRA Configuration
# Targeting all linear layers is standard for optimal performance on Gemma
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 6. Training Arguments (using SFTConfig for TRL compatibility)
sft_config = SFTConfig(
    output_dir="./sql_model_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    # SFTTrainer specific fields MUST be in SFTConfig in newer TRL versions
    dataset_text_field="text", # CHANGE THIS to match your JSONL column name
    max_length=1024,
    packing=False,
)


# 7. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
    processing_class=tokenizer,
)

# 8. Start Training
trainer.train()

# 9. Save the adapter
trainer.save_model("./final_sql_adapter")
