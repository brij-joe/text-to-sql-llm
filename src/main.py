from dotenv import load_dotenv

from config.training_config import TrainingConfig
from trainer.prompt_tuning_trainer import PromptTuningTrainer
from inference.inference_engine import InferenceEngine


def main():
    load_dotenv()

    config = TrainingConfig()
    print(f"Using device: {config.device}")

    trainer = PromptTuningTrainer(config)

    # Train
    trainer.train()

    # Inference
    model, tokenizer = trainer.get_model_tokenizer()
    inference = InferenceEngine(model, tokenizer, config.device)

    query = "How many heads of the departments are older than 56 ?"
    result = inference.generate_sql(query)

    print("\nGenerated SQL:\n", result)


if __name__ == "__main__":
    main()