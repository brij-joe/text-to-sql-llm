import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.training_config import TrainingConfig


class InferenceEngine:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.__load_model_for_inference()

    def __load_model_for_inference(self):
        """
        Loads base model + trained PEFT adapter
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
        )

        self.model = PeftModel.from_pretrained(base_model, self.config.adapter_path)

        # explicit device placement
        self.model.to(self.config.device)

        # enable cache for inference
        self.model.config.use_cache = True

        self.model.eval()

    @staticmethod
    def clean_sql_output(text: str) -> str:
        text = text.strip()
        for prefix in ["SQL:", "### SQL", "Answer:", "Output:"]:
            if prefix in text:
                text = text.split(prefix, 1)[-1].strip()

        if ";" in text:
            text = text.split(";")[0] + ";"

        return text.strip()

    def generate_sql(self, query: str, schema: str, max_new_tokens: int = 120) -> str:
        prompt = (
            f"### Task\n"
            f"Convert natural language to SQL.\n\n"
            f"### Schema\n{schema}\n\n"
            f"### Question\n{query}\n\n"
            f"### Instructions\n"
            f"- Output ONLY SQL\n"
            f"- No explanation\n"
            f"- No comments\n\n"
            f"### SQL\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return self.clean_sql_output(decoded)