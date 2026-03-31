import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, TaskType

from torch.cuda.amp import autocast, GradScaler

# =========================
# 1. CONFIG
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16          # increased
LR = 5e-5
EPOCHS = 3
MAX_LENGTH = 128
ACCUM_STEPS = 2          # gradient accumulation

# =========================
# 2. LOAD MODEL & TOKENIZER
# =========================
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# =========================
# 3. PROMPT TUNING CONFIG
# =========================
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="You are a senior SQL expert.",
    tokenizer_name_or_path=MODEL_NAME
)

model = get_peft_model(base_model, peft_config).to(DEVICE)

# Step 2: torch compile
model = torch.compile(model)

model.print_trainable_parameters()

# =========================
# 4. LOAD DATASET
# =========================
dataset = load_dataset("spider")["train"]

def preprocess(example):
    question = example["question"]
    query = example["query"]

    # Step 6 (light prompt for speed)
    text = f"Question: {question}\nSQL: {query}"

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess)

# keep only required columns
dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
)

dataset.set_format(type="torch")

# =========================
# 5. DATALOADER (Optimized)
# =========================
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,      # faster loading
    pin_memory=True     # GPU optimization
)

# =========================
# 6. TRAINING LOOP
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = GradScaler()   # Step 1 (AMP)

model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for step, batch in enumerate(train_loader):

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Step 1: Mixed precision
        with autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss / ACCUM_STEPS

        # backward
        scaler.scale(loss).backward()

        # Step 5: Gradient accumulation
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

# =========================
# 7. INFERENCE
# =========================
model.eval()

def generate_sql(query: str):
    inputs = tokenizer(query, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
query = "Show top 5 customers by order"
print(generate_sql(query))