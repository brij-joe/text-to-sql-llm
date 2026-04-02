import logging

from dotenv import load_dotenv
from pathlib import Path

from config.training_config import TrainingConfig
from trainer.prompt_tuning_trainer import PromptTuningTrainer
from inference.inference_engine import InferenceEngine

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

def main():
    # 1. load the HF_TOKEN
    load_dotenv()

    # 2. Load the training configurations
    config = TrainingConfig()
    print(f"Using device: {config.device}")

    # 3.Control flags (no logic change, just flexibility)
    RUN_TRAINING = True
    RUN_INFERENCE = True

    try:
        # =========================
        # 4. TRAINING
        # =========================
        if RUN_TRAINING:
            print("\nStarting Training...\n")
            trainer = PromptTuningTrainer(config)
            trainer.train()
            trainer.save_model()

            print(f"\nModel saved at: {config.adapter_path}")

        # =========================
        # 5. CHECK MODEL EXISTS
        # =========================
        adapter_path = Path(config.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}\n"
                f"Run training first or check path."
            )

        # =========================
        # 6. INFERENCE
        # =========================
        if RUN_INFERENCE:
            print("\nRunning Inference...\n")

            inference = InferenceEngine(config)

            # Example schema (replace with real schema in production)
            schema = """
            department(id, name, head_id, head_age)
            """

            query = "How many heads of the departments are older than 56?"

            result = inference.generate_sql(query, schema)

            print("\nGenerated SQL:\n", result)

    except Exception as e:
        print("\nError occurred:")
        print(e)


if __name__ == "__main__":
    main()