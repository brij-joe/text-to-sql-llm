from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    # Small + fast model (keep TinyLlama for compatibility)
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ⚡ SPEED-OPTIMIZED TRAINING
    batch_size: int = 1
    accumulation_steps: int = 1          #  no accumulation (faster)

    learning_rate: float = 1e-3          #  good for prompt tuning
    epochs: int = 1                      #  single pass

    max_length: int = 64                 #  shorter sequences → faster

    num_virtual_tokens: int = 3          #  minimal prompt tuning, this should be over 50
    init_text: str = "Convert the given natural language question into a valid SQL query. Output only SQL."


    # ⚡ VERY SMALL DATA
    num_samples: int = 50                #  runs in minutes

    # DEVICE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SAVE PATH
    @property
    def adapter_path(self):
        root_dir = Path(__file__).resolve().parents[2]
        return root_dir / "trained" / "text_to_sql_model"

    @property
    def num_workers(self):
        # keep 0 → avoids multiprocessing overhead for small data
        # return 4 if self.device == "cuda" else 0
        return 0 if self.device == "cuda" else 0

    @property
    def pin_memory(self):
        return self.device == "cuda"