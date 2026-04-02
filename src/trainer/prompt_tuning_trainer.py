import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, TaskType

from config.training_config import TrainingConfig


class PromptTuningTrainer:
    _instance: Optional["PromptTuningTrainer"] = None

    def __new__(cls, config: TrainingConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: TrainingConfig):
        if hasattr(self, "_initialized"):
            return

        self.config = config
        self.device = config.device

        self.model, self.tokenizer = self._load_model()
        self.train_loader = self._load_data()

        self._initialized = True

    # =========================
    # MODEL
    # =========================
    def _load_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name
        ).to(self.device)

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.config.num_virtual_tokens,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=self.config.init_text,
            tokenizer_name_or_path=self.config.model_name,
        )

        model = get_peft_model(base_model, peft_config).to(self.device)

        print("Skipping torch.compile on Windows for stability")

        model.print_trainable_parameters()
        return model, tokenizer

    # =========================
    # DATA (FIXED)
    # =========================
    def _preprocess(self, example: dict) -> dict:
        prompt = f"Question: {example['question']}\nSQL:"
        target = f" {example['query']}"

        full_text = prompt + target

        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )

        labels = tokenized["input_ids"].copy()

        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        tokenized["labels"] = labels
        return tokenized

    def _load_data(self) -> DataLoader:
        dataset = load_dataset("spider")["train"].select(range(2000))

        dataset = dataset.map(self._preprocess)

        # print test data
        for i in range(5):
            print(f"\n===== Sample {i + 1} =====")
            print(dataset[i])

        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
        )

        dataset.set_format(type="torch")

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    # =========================
    # TRAIN
    # =========================
    def train(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        use_amp = self.device == "cuda"
        scaler = GradScaler(device="cuda", enabled=use_amp)

        self.model.train()

        for epoch in range(self.config.epochs):
            total_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(device_type="cuda", enabled=use_amp):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.config.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

    def get_model_tokenizer(self):
        return self.model, self.tokenizer