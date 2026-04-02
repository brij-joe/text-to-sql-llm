import torch


class InferenceEngine:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_sql(self, query: str, max_new_tokens: int = 100) -> str:
        self.model.eval()

        prompt = f"Question: {query}\nSQL:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # if "SQL:" in result:
            # result = result.split("SQL:")[-1].strip()

        return result