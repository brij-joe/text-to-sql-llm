from dataclasses import dataclass
import torch

# note: config has been modified to run on a laptop having 8GB GPU and 16GB RAM, 4 core cpu.
@dataclass
class TrainingConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_name: str = "distilbert/distilgpt2"
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 2
    max_length: int = 128
    accumulation_steps: int = 2
    num_virtual_tokens: int = 50
    init_text: str = "Convert the given natural language question into a valid SQL query. Output only SQL."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_workers(self):
        # return 4 if self.device == "cuda" else 0
        return 0 if self.device == "cuda" else 0

    @property
    def pin_memory(self):
        return self.device == "cuda"