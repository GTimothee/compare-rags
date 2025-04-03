
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import os
import argparse
import json
from pathlib import Path

from src.configuration import load_config
from src.evaluation.openai_like_evaluator import OpenAILikeEvaluator
from src.evaluation.huggingface_evaluator import HuggingfaceEvaluator


parser = argparse.ArgumentParser(description='Evaluate a Q/A system using llm as judge.')
parser.add_argument('config', type=str, help='The path to the configuration YAML file.')
args = parser.parse_args()

load_dotenv()


if __name__ == "__main__":
    
    config = load_config(args.config)

    logging.basicConfig(
        filename=f"{config.framework}_evaluation.log",
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    with open(str(Path(config.output_dir, 'answers.json')), 'r') as f:
        samples = json.load(f)

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
    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples), desc="RAG evaluation running..."):
        logging.info(f"Processing sample {sample_idx}...")

        try:
            logging.info("- Running critique chain...")
            critique = evaluator.evaluate(f"Question:{sample['question']}\nExpected Answer:{sample['answer']}\nSystem's response:```{sample['rag_answer']}```")
            logging.info(f"- Critique: {critique}")
            sample['rag_score'] = critique['evaluation']['score']
            sample['rag_feedback'] = critique['evaluation']['feedback']
        except Exception as e:
            logging.error(f"Error during evaluation of sample {sample_idx}: {e}")
            sample['rag_score'] = None
            sample['rag_feedback'] = None
        
        results.append(sample)

    print('Saving results...')
    output_path = os.path.join(config.output_dir, f'eval_{config.llm}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)