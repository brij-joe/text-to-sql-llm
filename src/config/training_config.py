from dataclasses import dataclass

import torch


# =========================
# CONFIG
# =========================
@dataclass
class TrainingConfig:
    # model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name: str = "distilbert/distilgpt2"
    # batch_size: int = 16 // minimized for laptop with GPU
    batch_size: int = 2
    learning_rate: float = 5e-5
    # epochs: int = 3
    epochs: int = 1
    # max_length: int = 128
    max_length: int = 64
    accumulation_steps: int = 2
    num_virtual_tokens: int = 20
    init_text: str = "You are a senior SQL expert."

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_workers(self) -> int:
        # return 4 if self.device == "cuda" else 0
        return 0 if self.device == "cuda" else 0

    @property
    def pin_memory(self) -> bool:
        return self.device == "cuda"
