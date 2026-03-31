import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, TaskType

from torch.cuda.amp import autocast, GradScaler

# =========================
# CONFIG
# =========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
LR = 5e-5
EPOCHS = 3
MAX_LENGTH = 128
ACCUM_STEPS = 2

if DEVICE == "cuda":
    NUM_WORKERS = 4
else:
    NUM_WORKERS = 0

PIN_MEMORY = DEVICE == "cuda"


# =========================
# LOAD MODEL
# =========================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="You are a senior SQL expert.",
        tokenizer_name_or_path=MODEL_NAME
    )

    model = get_peft_model(base_model, peft_config).to(DEVICE)

    # compile
    if DEVICE == "cuda":
        model = torch.compile(model)

    model.print_trainable_parameters()

    return model, tokenizer


# =========================
# DATASET
# =========================
def preprocess(example, tokenizer):
    question = example["question"]
    query = example["query"]

    # lightweight prompt (fast)
    text = f"Question: {question}\nSQL: {query}"

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_data(tokenizer):
    dataset = load_dataset("spider")["train"]

    dataset = dataset.map(lambda x: preprocess(x, tokenizer))

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )

    dataset.set_format(type="torch")

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader


# =========================
# TRAINING
# =========================
def train(model, train_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    use_amp = DEVICE == "cuda"
    scaler = GradScaler(enabled = use_amp)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for step, batch in enumerate(train_loader):

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with autocast(enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")


# =========================
# INFERENCE
# =========================
def generate_sql(model, tokenizer, query: str):
    model.eval()

    inputs = tokenizer(query, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# MAIN
# =========================
def main():
    load_dotenv()

    print(f"Using device: {DEVICE}")

    # load model
    model, tokenizer = load_model()

    # load data
    train_loader = load_data(tokenizer)

    # train
    train(model, train_loader)

    # inference
    query = "Show top 5 customers by order"
    result = generate_sql(model, tokenizer, query)

    print("\nGenerated SQL:\n", result)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()