
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
from transformers import pipeline
import json
import yaml

from src.configuration import load_config
from src.lightrag.rag import LightRag
from src.llama_index.llama_index_rag import LlamaIndexRag


parser = argparse.ArgumentParser(description='Evaluate a Q/A system using llm as judge.')
parser.add_argument('config', type=str, help='The path to the configuration YAML file.')
args = parser.parse_args()

load_dotenv()


evaluation_prompt = """Your task is to evaluate a Q/A system. 
The user will give you a question, an expected answer and the system's response.
You will evaluate the system's response and provide a score and a text explanation for the score you gave.
We are asking ourselves if the response is correct, accurate and factual, based on the reference answer.

Guidelines:
1. Write a detailed feedback that assess the quality of the response strictly based on the given scores description, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the scores description.
3. Follow the JSON format provided below for your output.

Scores description:
    Score 1: The response is completely incorrect, inaccurate, and/or not factual.
    Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
    Score 3: The response is somewhat correct, accurate, and/or factual.
    Score 4: The response is mostly correct, accurate, and factual.
    Score 5: The response is completely correct, accurate, and factual.

Output Format (JSON only):
{{
    "evaluation": {{
        "feedback": "(your rationale for the rating, as a text)",
        "score": (your rating, as a number between 1 and 5)
    }}
}}

Do not include any additional text—only the JSON object. Any extra content will result in a grade of 0."""


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


class OpenAILikeEvaluator:
    def __init__(self, model_name: str = "Llama-3-70B-Instruct", device_map: str = "cuda"):
        self.llm = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
            request_timeout=60,
            model_name=model_name,
            temperature=0.0,
        )
        self.critique_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    evaluation_prompt,
                ),
                ("human", "{user_message}"),
            ]
        ) | self.llm | JsonOutputParser()

    def evaluate(self, user_message: str):
        return self.critique_chain.invoke({
                "user_message": user_message
            }
        )


if __name__ == "__main__":
    
    config = load_config(args.config)

    logging.basicConfig(
        filename=f"{config.framework}_evaluation.log",
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    ds = load_dataset(config.eval_dataset_path)['train']

    # Load RAG engine
    if config.framework == "llama_index":
        rag_engine = LlamaIndexRag(config)
    elif config.framework == "lightrag":
        rag_engine = LightRag(config.index_dirpath)
    else: 
        raise NotImplementedError(f"Unsupported framework: {config.framework}")

    # Load evaluator
    if config.llm.startswith("huggingface"):
        llm_name = config.llm.split("_")[-1]
        logging.info(f"Loading OpenAI-like evaluator with model {llm_name}...")
        evaluator = HuggingfaceEvaluator(model_name=llm_name)
    elif config.llm == "openai-like":
        logging.info(f"Loading OpenAI-like evaluator with model {config.llm}...")
        evaluator = OpenAILikeEvaluator(model_name=config.llm)
    else:
        raise NotImplementedError(f"Unsupported LLM: {config.llm}")

    # Evaluation loop
    results = []
    for sample_idx, sample in tqdm(enumerate(ds), total=len(ds), desc="RAG evaluation running..."):
        logging.info(f"Processing sample {sample_idx}...")

        try:
            logging.info("- Running RAG engine...")
            rag_answer = rag_engine.run(sample['question'])
            logging.info(f"- RAG response: {rag_answer}")
            sample['rag_answer'] = rag_answer['text']
            sample['rag_context'] = rag_answer['context']
        except Exception as e:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            sample['rag_answer'] = None
            sample['rag_context'] = None
            sample['rag_score'] = None
            sample['rag_feedback'] = None
            results.append(sample)
            continue

        try:
            logging.info("- Running critique chain...")
            critique = evaluator.evaluate(f"Question:{sample['question']}\nExpected Answer:{sample['answer']}\nSystem's response:```{rag_answer}```")
            logging.info(f"- Critique: {critique}")
            sample['rag_score'] = critique['evaluation']['score']
            sample['rag_feedback'] = critique['evaluation']['feedback']
        except Exception as e:
            logging.error(f"Error during evaluation of sample {sample_idx}: {e}")
            sample['rag_score'] = None
            sample['rag_feedback'] = None
        
        results.append({
            'sample_idx': sample_idx,
            'question': sample['question'],
            'answer': sample['answer'],
            'rag_answer': sample['rag_answer'],
            'rag_score': sample['rag_score'],
            'rag_feedback': sample['rag_feedback'],
            'rag_context': sample['rag_context']
        })

    print('Saving results...')
    output_path = os.path.join(config.output_dir, 'eval.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)