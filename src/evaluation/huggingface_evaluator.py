from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
from src.evaluation.prompts import evaluation_prompt


class HuggingfaceEvaluator:
    def __init__(self, model_name: str = "casperhansen/llama-3-8b-instruct-awq", device_map: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def evaluate(self, user_message: str):
        messages = [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": user_message},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.pipe(formatted_prompt, max_new_tokens=1000)[0]['generated_text']